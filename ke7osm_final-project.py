from fastai.text import *

import pandas

import fastText as ft

import pickle
path = Config().data_path()
with open ('../input/final-data/finallist.pickle', 'rb') as my_file:

    en_pt = pickle.load(my_file)
labels = ['en', 'pt']



df =  pd.DataFrame.from_records(en_pt, columns=labels)

df.head
df['en'] = df['en'].apply(lambda x:x.lower())

df['pt'] = df['pt'].apply(lambda x:x.lower())
def seq2seq_collate(samples:BatchSamples, pad_idx:int=1, pad_first:bool=True, backwards:bool=False) -> Tuple[LongTensor, LongTensor]:

    "Function that collect samples and adds padding. Flips token order if needed"

    samples = to_data(samples)

    max_len_x,max_len_y = max([len(s[0]) for s in samples]),max([len(s[1]) for s in samples])

    res_x = torch.zeros(len(samples), max_len_x).long() + pad_idx

    res_y = torch.zeros(len(samples), max_len_y).long() + pad_idx

    if backwards: pad_first = not pad_first

    for i,s in enumerate(samples):

        if pad_first: 

            res_x[i,-len(s[0]):],res_y[i,-len(s[1]):] = LongTensor(s[0]),LongTensor(s[1])

        else:         

            res_x[i,:len(s[0]):],res_y[i,:len(s[1]):] = LongTensor(s[0]),LongTensor(s[1])

    if backwards: res_x,res_y = res_x.flip(1),res_y.flip(1)

    return res_x,res_y
class Seq2SeqDataBunch(TextDataBunch):

    "Create a `TextDataBunch` suitable for training an RNN classifier."

    @classmethod

    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', bs:int=32, val_bs:int=None, pad_idx=1,

               pad_first=False, device:torch.device=None, no_check:bool=False, backwards:bool=False, **dl_kwargs) -> DataBunch:

        "Function that transform the `datasets` in a `DataBunch` for classification. Passes `**dl_kwargs` on to `DataLoader()`"

        datasets = cls._init_ds(train_ds, valid_ds, test_ds)

        val_bs = ifnone(val_bs, bs)

        collate_fn = partial(seq2seq_collate, pad_idx=pad_idx, pad_first=pad_first, backwards=backwards)

        train_sampler = SortishSampler(datasets[0].x, key=lambda t: len(datasets[0][t][0].data), bs=bs//2)

        train_dl = DataLoader(datasets[0], batch_size=bs, sampler=train_sampler, drop_last=True, **dl_kwargs)

        dataloaders = [train_dl]

        for ds in datasets[1:]:

            lengths = [len(t) for t in ds.x.items]

            sampler = SortSampler(ds.x, key=lengths.__getitem__)

            dataloaders.append(DataLoader(ds, batch_size=val_bs, sampler=sampler, **dl_kwargs))

        return cls(*dataloaders, path=path, device=device, collate_fn=collate_fn, no_check=no_check)
class Seq2SeqTextList(TextList):

    _bunch = Seq2SeqDataBunch

    _label_cls = TextList
src = Seq2SeqTextList.from_df(df, path = path, cols='pt').split_by_rand_pct().label_from_df(cols='en', label_cls=TextList)
np.percentile([len(o) for o in src.train.x.items] + [len(o) for o in src.valid.x.items], 90)
np.percentile([len(o) for o in src.train.y.items] + [len(o) for o in src.valid.y.items], 90)
src = src.filter_by_func(lambda x,y: len(x) > 50 or len(y) > 50)
data = src.databunch()
data.show_batch()
pt_vecs = ft.load_model('../input/fasttext_pt/cc.pt.300.bin')
def create_emb(vecs, itos, em_sz=300, mult=1.):

    emb = nn.Embedding(len(itos), em_sz, padding_idx=1)

    wgts = emb.weight.data

    vec_dic = {w:vecs.get_word_vector(w) for w in vecs.get_words()}

    miss = []

    for i,w in enumerate(itos):

        try: wgts[i] = tensor(vec_dic[w])

        except: miss.append(w)

    return emb

emb_enc = create_emb(pt_vecs, data.x.vocab.itos)
torch.save(emb_enc, 'pt_emb.pth')
del pt_vecs
en_vecs = ft.load_model('../input/fasttext-english/cc.en.300.bin')
emb_dec = create_emb(en_vecs, data.y.vocab.itos)
torch.save(emb_enc, 'en_emb.pth')
del en_vecs
from fastai.text.models.qrnn import QRNN, QRNNLayer
class Seq2SeqQRNN(nn.Module):

    def __init__(self, emb_enc, emb_dec, n_hid, max_len, n_layers=2, p_inp:float=0.15, p_enc:float=0.25, 

                 p_dec:float=0.1, p_out:float=0.35, p_hid:float=0.05, bos_idx:int=0, pad_idx:int=1):

        super().__init__()

        self.n_layers,self.n_hid,self.max_len,self.bos_idx,self.pad_idx = n_layers,n_hid,max_len,bos_idx,pad_idx

        self.emb_enc = emb_enc

        self.emb_enc_drop = nn.Dropout(p_inp)

        self.encoder = QRNN(emb_enc.weight.size(1), n_hid, n_layers=n_layers, dropout=p_enc)

        self.out_enc = nn.Linear(n_hid, emb_enc.weight.size(1), bias=False)

        self.hid_dp  = nn.Dropout(p_hid)

        self.emb_dec = emb_dec

        self.decoder = QRNN(emb_dec.weight.size(1), emb_dec.weight.size(1), n_layers=n_layers, dropout=p_dec)

        self.out_drop = nn.Dropout(p_out)

        self.out = nn.Linear(emb_dec.weight.size(1), emb_dec.weight.size(0))

        self.out.weight.data = self.emb_dec.weight.data

        

    def forward(self, inp):

        bs,sl = inp.size()

        self.encoder.reset()

        self.decoder.reset()

        hid = self.initHidden(bs)

        emb = self.emb_enc_drop(self.emb_enc(inp))

        enc_out, hid = self.encoder(emb, hid)

        hid = self.out_enc(self.hid_dp(hid))



        dec_inp = inp.new_zeros(bs).long() + self.bos_idx

        outs = []

        for i in range(self.max_len):

            emb = self.emb_dec(dec_inp).unsqueeze(1)

            out, hid = self.decoder(emb, hid)

            out = self.out(self.out_drop(out[:,0]))

            outs.append(out)

            dec_inp = out.max(1)[1]

            if (dec_inp==self.pad_idx).all(): break

        return torch.stack(outs, dim=1)

    

    def initHidden(self, bs): return one_param(self).new_zeros(self.n_layers, bs, self.n_hid)


def seq2seq_loss(out, targ, pad_idx=1):

    bs,targ_len = targ.size()

    _,out_len,vs = out.size()

    if targ_len>out_len: out  = F.pad(out,  (0,0,0,targ_len-out_len,0,0), value=pad_idx)

    if out_len>targ_len: targ = F.pad(targ, (0,out_len-targ_len,0,0), value=pad_idx)

    return CrossEntropyFlat()(out, targ)
def seq2seq_acc(out, targ, pad_idx=1):

    bs,targ_len = targ.size()

    _,out_len,vs = out.size()

    if targ_len>out_len: out  = F.pad(out,  (0,0,0,targ_len-out_len,0,0), value=pad_idx)

    if out_len>targ_len: targ = F.pad(targ, (0,out_len-targ_len,0,0), value=pad_idx)

    out = out.argmax(2)

    return (out==targ).float().mean()
class NGram():

    def __init__(self, ngram, max_n=5000): self.ngram,self.max_n = ngram,max_n

    def __eq__(self, other):

        if len(self.ngram) != len(other.ngram): return False

        return np.all(np.array(self.ngram) == np.array(other.ngram))

    def __hash__(self): return int(sum([o * self.max_n**i for i,o in enumerate(self.ngram)]))
def get_grams(x, n, max_n=5000):

    return x if n==1 else [NGram(x[i:i+n], max_n=max_n) for i in range(len(x)-n+1)]
def get_correct_ngrams(pred, targ, n, max_n=5000):

    pred_grams,targ_grams = get_grams(pred, n, max_n=max_n),get_grams(targ, n, max_n=max_n)

    pred_cnt,targ_cnt = Counter(pred_grams),Counter(targ_grams)

    return sum([min(c, targ_cnt[g]) for g,c in pred_cnt.items()]),len(pred_grams)
class CorpusBLEU(Callback):

    def __init__(self, vocab_sz):

        self.vocab_sz = vocab_sz

        self.name = 'bleu'

    

    def on_epoch_begin(self, **kwargs):

        self.pred_len,self.targ_len,self.corrects,self.counts = 0,0,[0]*4,[0]*4

    

    def on_batch_end(self, last_output, last_target, **kwargs):

        last_output = last_output.argmax(dim=-1)

        for pred,targ in zip(last_output.cpu().numpy(),last_target.cpu().numpy()):

            self.pred_len += len(pred)

            self.targ_len += len(targ)

            for i in range(4):

                c,t = get_correct_ngrams(pred, targ, i+1, max_n=self.vocab_sz)

                self.corrects[i] += c

                self.counts[i]   += t

    

    def on_epoch_end(self, last_metrics, **kwargs):

        precs = [c/t for c,t in zip(self.corrects,self.counts)]

        len_penalty = exp(1 - self.targ_len/self.pred_len) if self.pred_len < self.targ_len else 1

        bleu = len_penalty * ((precs[0]*precs[1]*precs[2]*precs[3]) ** 0.25)

        return add_metrics(last_metrics, bleu)


emb_enc = torch.load('pt_emb.pth')

emb_dec = torch.load('en_emb.pth')
model = Seq2SeqQRNN(emb_enc, emb_dec, 256, 30, n_layers=2)

learn = Learner(data, model, loss_func=seq2seq_loss, metrics=[seq2seq_acc, CorpusBLEU(len(data.y.vocab.itos))])
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(8, 1e-2)
def get_predictions(learn, ds_type=DatasetType.Valid):

    learn.model.eval()

    inputs, targets, outputs = [],[],[]

    with torch.no_grad():

        for xb,yb in progress_bar(learn.dl(ds_type)):

            out = learn.model(xb)

            for x,y,z in zip(xb,yb,out):

                inputs.append(learn.data.train_ds.x.reconstruct(x))

                targets.append(learn.data.train_ds.y.reconstruct(y))

                outputs.append(learn.data.train_ds.y.reconstruct(z.argmax(1)))

    return inputs, targets, outputs
inputs, targets, outputs = get_predictions(learn)
for i in range(705,710):

    print(inputs[i], targets[i], outputs[i], sep = '\n')

    print('\n')
class TeacherForcing(LearnerCallback):

    

    def __init__(self, learn, end_epoch):

        super().__init__(learn)

        self.end_epoch = end_epoch

    

    def on_batch_begin(self, last_input, last_target, train, **kwargs):

        if train: return {'last_input': [last_input, last_target]}

    

    def on_epoch_begin(self, epoch, **kwargs):

        self.learn.model.pr_force = 1 - 0.5 * epoch/self.end_epoch
class Seq2SeqQRNN(nn.Module):

    def __init__(self, emb_enc, emb_dec, n_hid, max_len, n_layers=2, p_inp:float=0.15, p_enc:float=0.25, 

                 p_dec:float=0.1, p_out:float=0.35, p_hid:float=0.05, bos_idx:int=0, pad_idx:int=1):

        super().__init__()

        self.n_layers,self.n_hid,self.max_len,self.bos_idx,self.pad_idx = n_layers,n_hid,max_len,bos_idx,pad_idx

        self.emb_enc = emb_enc

        self.emb_enc_drop = nn.Dropout(p_inp)

        self.encoder = QRNN(emb_enc.weight.size(1), n_hid, n_layers=n_layers, dropout=p_enc)

        self.out_enc = nn.Linear(n_hid, emb_enc.weight.size(1), bias=False)

        self.hid_dp  = nn.Dropout(p_hid)

        self.emb_dec = emb_dec

        self.decoder = QRNN(emb_dec.weight.size(1), emb_dec.weight.size(1), n_layers=n_layers, dropout=p_dec)

        self.out_drop = nn.Dropout(p_out)

        self.out = nn.Linear(emb_dec.weight.size(1), emb_dec.weight.size(0))

        self.out.weight.data = self.emb_dec.weight.data

        self.pr_force = 0.

        

    def forward(self, inp, targ=None):

        bs,sl = inp.size()

        hid = self.initHidden(bs)

        emb = self.emb_enc_drop(self.emb_enc(inp))

        enc_out, hid = self.encoder(emb, hid)

        hid = self.out_enc(self.hid_dp(hid))



        dec_inp = inp.new_zeros(bs).long() + self.bos_idx

        res = []

        for i in range(self.max_len):

            emb = self.emb_dec(dec_inp).unsqueeze(1)

            outp, hid = self.decoder(emb, hid)

            outp = self.out(self.out_drop(outp[:,0]))

            res.append(outp)

            dec_inp = outp.data.max(1)[1]

            if (dec_inp==self.pad_idx).all(): break

            if (targ is not None) and (random.random()<self.pr_force):

                if i>=targ.shape[1]: break

                dec_inp = targ[:,i]

        return torch.stack(res, dim=1)

    

    def initHidden(self, bs): return one_param(self).new_zeros(self.n_layers, bs, self.n_hid)
emb_enc = torch.load('pt_emb.pth')

emb_dec = torch.load('en_emb.pth')
model = Seq2SeqQRNN(emb_enc, emb_dec, 256, 30, n_layers=2)

learn = Learner(data, model, loss_func=seq2seq_loss, metrics=[seq2seq_acc, CorpusBLEU(len(data.y.vocab.itos))],

                callback_fns=partial(TeacherForcing, end_epoch=8))
learn.fit_one_cycle(8, 1e-2)
inputs, targets, outputs = get_predictions(learn)
for i in range(705,710):

    print(inputs[i], targets[i], outputs[i], sep = '\n')

    print('\n')
class Seq2SeqQRNN(nn.Module):

    def __init__(self, emb_enc, emb_dec, n_hid, max_len, n_layers=2, p_inp:float=0.15, p_enc:float=0.25, 

                 p_dec:float=0.1, p_out:float=0.35, p_hid:float=0.05, bos_idx:int=0, pad_idx:int=1):

        super().__init__()

        self.n_layers,self.n_hid,self.max_len,self.bos_idx,self.pad_idx = n_layers,n_hid,max_len,bos_idx,pad_idx

        self.emb_enc = emb_enc

        self.emb_enc_drop = nn.Dropout(p_inp)

        self.encoder = QRNN(emb_enc.weight.size(1), n_hid, n_layers=n_layers, dropout=p_enc, bidirectional=True)

        self.out_enc = nn.Linear(2*n_hid, emb_enc.weight.size(1), bias=False)

        self.hid_dp  = nn.Dropout(p_hid)

        self.emb_dec = emb_dec

        self.decoder = QRNN(emb_dec.weight.size(1), emb_dec.weight.size(1), n_layers=n_layers, dropout=p_dec)

        self.out_drop = nn.Dropout(p_out)

        self.out = nn.Linear(emb_dec.weight.size(1), emb_dec.weight.size(0))

        self.out.weight.data = self.emb_dec.weight.data

        self.pr_force = 0.

        

    def forward(self, inp, targ=None):

        bs,sl = inp.size()

        hid = self.initHidden(bs)

        emb = self.emb_enc_drop(self.emb_enc(inp))

        enc_out, hid = self.encoder(emb, hid)

        

        hid = hid.view(2,self.n_layers, bs, self.n_hid).permute(1,2,0,3).contiguous()

        hid = self.out_enc(self.hid_dp(hid).view(self.n_layers, bs, 2*self.n_hid))



        dec_inp = inp.new_zeros(bs).long() + self.bos_idx

        res = []

        for i in range(self.max_len):

            emb = self.emb_dec(dec_inp).unsqueeze(1)

            outp, hid = self.decoder(emb, hid)

            outp = self.out(self.out_drop(outp[:,0]))

            res.append(outp)

            dec_inp = outp.data.max(1)[1]

            if (dec_inp==self.pad_idx).all(): break

            if (targ is not None) and (random.random()<self.pr_force):

                if i>=targ.shape[1]: break

                dec_inp = targ[:,i]

        return torch.stack(res, dim=1)

    

    def initHidden(self, bs): return one_param(self).new_zeros(2*self.n_layers, bs, self.n_hid)
emb_enc = torch.load('pt_emb.pth')

emb_dec = torch.load('en_emb.pth')
model = Seq2SeqQRNN(emb_enc, emb_dec, 256, 30, n_layers=2)

learn = Learner(data, model, loss_func=seq2seq_loss, metrics=[seq2seq_acc, CorpusBLEU(len(data.y.vocab.itos))],

                callback_fns=partial(TeacherForcing, end_epoch=8))
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(8, 1e-2)
inputs, targets, outputs = get_predictions(learn)
for i in range(705,710):

    print(inputs[i], targets[i], outputs[i], sep = '\n')

    print('\n')
class Seq2SeqQRNN(nn.Module):

    def __init__(self, emb_enc, emb_dec, n_hid, max_len, n_layers=2, p_inp:float=0.15, p_enc:float=0.25, 

                 p_dec:float=0.1, p_out:float=0.35, p_hid:float=0.05, bos_idx:int=0, pad_idx:int=1):

        super().__init__()

        self.n_layers,self.n_hid,self.max_len,self.bos_idx,self.pad_idx = n_layers,n_hid,max_len,bos_idx,pad_idx

        self.emb_enc = emb_enc

        self.emb_enc_drop = nn.Dropout(p_inp)

        self.encoder = QRNN(emb_enc.weight.size(1), n_hid, n_layers=n_layers, dropout=p_enc)

        self.out_enc = nn.Linear(n_hid, emb_enc.weight.size(1), bias=False)

        self.hid_dp  = nn.Dropout(p_hid)

        self.emb_dec = emb_dec

        emb_sz = emb_dec.weight.size(1)

        self.decoder = QRNN(emb_sz + n_hid, emb_dec.weight.size(1), n_layers=n_layers, dropout=p_dec)

        self.out_drop = nn.Dropout(p_out)

        self.out = nn.Linear(emb_sz, emb_dec.weight.size(0))

        self.out.weight.data = self.emb_dec.weight.data #Try tying

        self.enc_att = nn.Linear(n_hid, emb_sz, bias=False)

        self.hid_att = nn.Linear(emb_sz, emb_sz)

        self.V =  init_param(emb_sz)

        self.pr_force = 0.

        

    def forward(self, inp, targ=None):

        bs,sl = inp.size()

        hid = self.initHidden(bs)

        emb = self.emb_enc_drop(self.emb_enc(inp))

        enc_out, hid = self.encoder(emb, hid)

        hid = self.out_enc(self.hid_dp(hid))



        dec_inp = inp.new_zeros(bs).long() + self.bos_idx

        res = []

        enc_att = self.enc_att(enc_out)

        for i in range(self.max_len):

            hid_att = self.hid_att(hid[-1])

            u = torch.tanh(enc_att + hid_att[:,None])

            attn_wgts = F.softmax(u @ self.V, 1)

            ctx = (attn_wgts[...,None] * enc_out).sum(1)

            emb = self.emb_dec(dec_inp)

            outp, hid = self.decoder(torch.cat([emb, ctx], 1)[:,None], hid)

            outp = self.out(self.out_drop(outp[:,0]))

            res.append(outp)

            dec_inp = outp.data.max(1)[1]

            if (dec_inp==self.pad_idx).all(): break

            if (targ is not None) and (random.random()<self.pr_force):

                if i>=targ.shape[1]: break

                dec_inp = targ[:,i]

        return torch.stack(res, dim=1)

    

    def initHidden(self, bs): return one_param(self).new_zeros(self.n_layers, bs, self.n_hid)
emb_enc = torch.load('pt_emb.pth')

emb_dec = torch.load('en_emb.pth')
model = Seq2SeqQRNN(emb_enc, emb_dec, 256, 30, n_layers=2)

learn = Learner(data, model, loss_func=seq2seq_loss, metrics=[seq2seq_acc, CorpusBLEU(len(data.y.vocab.itos))],

                callback_fns=partial(TeacherForcing, end_epoch=8))
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(8, 1e-2)
inputs, targets, outputs = get_predictions(learn)
for i in range(705,710):

    print(inputs[i], targets[i], outputs[i], sep = '\n')

    print('\n')