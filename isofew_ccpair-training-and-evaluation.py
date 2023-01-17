from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
import sys
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(2049)
class Pack():
    special = [
        '〈PAD〉',
        '〈UNK〉',
        '〈NUM〉',
        '〈STR〉',
        '〈EOS〉',
        '〈START〉',
    ]
    
    def __init__(self, filename):
        self.word_index, self.vector, self.starts, self.ends = np.load(filename)
        self.vector += len(Pack.special)
        self.word_index = dict(
            **{w: i for i, w in enumerate(Pack.special)},
            **{w: i + len(Pack.special) for w, i in self.word_index.items()}
        )
        self.index_word = list(self.word_index.keys())
    
    def __len__(self):
        return len(self.starts)
    
    def __getitem__(self, i):
        return self.vector[self.starts[i] : self.ends[i]]

    def pprint(self, item):
        print(' '.join(map(list(self.word_index).__getitem__, item)))
class Dataset():
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.eos = self.dst.word_index['〈EOS〉']
        self.start = self.dst.word_index['〈START〉']

    def __len__(self):
        return len(self.src.starts)

    def __getitem__(self, i):
        return self.src[i], [self.start, *self.dst[i]], [*self.dst[i], self.eos]
    
    def pprint(self, pair):
        print('>>> src')
        self.src.pprint(pair[0])
        print()
        print('>>> dst_in')
        self.dst.pprint(pair[1])
        print()
        print('>>> dst_out')
        self.dst.pprint(pair[2])
ds, df, train, test, valid, valid_same_repo, valid_same_file = np.load('../input/bundle.npy')
ds.pprint(ds[0])
# for the limited computing resources on kaggle kernel
sample = np.random.rand(len(df)) < 0.05
train, test, valid, valid_same_repo, valid_same_file = map(lambda x: x & sample, (train, test, valid, valid_same_repo, valid_same_file))
class BaselineLSTM(nn.Module):
    def __init__(self, n_src, n_dst, n_hid, dropout_emb, dropout_hid):
        super(BaselineLSTM, self).__init__()
        self.n_src, self.n_dst, self.n_hid = n_src, n_dst, n_hid
        self.src_embedding = nn.Embedding(n_src, n_hid)
        self.dst_embedding = nn.Embedding(n_dst, n_hid)
        #self.dst_bias = nn.Parameter(torch.randn(n_dst) / np.sqrt(n_dst))
        self.dst_pred = nn.Linear(n_hid, n_dst)
        self.lstm = nn.LSTM(n_hid, n_hid, batch_first=True)
        self.dropout_emb = nn.Dropout2d(dropout_emb)
        self.dropout_hid = nn.Dropout(dropout_hid)
        self.start = nn.Parameter(torch.LongTensor([Pack.special.index('〈START〉')]), requires_grad=False)
    
    def forward(self, x, y_in):
        x = torch.cat([self.src_embedding(x), self.dst_embedding(y_in)], 1)
        # transpose channel dimension for dropout2d
        x = self.dropout_emb(x.transpose(2, 1)).transpose(2, 1)
        h = self.lstm(x)[0][:, -y_in.size(1):]
        h = self.dropout_hid(h)
        #return torch.matmul(h, self.dst_embedding.weight.t()) + self.dst_bias
        return self.dst_pred(h)
    
    def translate(self, x, maxlen):
        x = torch.cat([self.src_embedding(x), self.dst_embedding(self.start.repeat(len(x), 1))], 1)
        y = torch.empty(len(x), maxlen, dtype=torch.long, device=x.device)
        h, c = self.lstm(x)
        y[:, 0] = self.dst_pred(h[:, -1]).argmax(-1)
        for i in range(1, maxlen):
            h, c = self.lstm(h[:, -1:], c)
            y[:, i] = self.dst_pred(h[:, -1]).argmax(-1)
        return y
def pad(xs, maxlen, pre):
    X = torch.zeros(len(xs), min( max(map(len, xs)), maxlen ), dtype=torch.int64)
    for i, x in enumerate(xs):
        x = torch.LongTensor(x)
        l = min(len(x), maxlen)
        if pre:
            X[i, -l:] = x[-l:]
        else:
            X[i, :l] = x[:l]
    return X
unzip = lambda ts: zip(*ts)
def get_batch(ds, ix, maxlen_src, maxlen_dst):
    return map(pad, unzip(ds[i] for i in ix), (maxlen_src, maxlen_dst, maxlen_dst), (True, False, False))
class Running():
    def __init__(self):
        self.x = 0
        self.n = 1e-9
    
    def add(self, x, n):
        self.x += float(x) * float(n)
        self.n += float(n)
    
    def get(self):
        return self.x / self.n
def ngram(xs, n):
    return zip(*(xs[i : len(xs) - n + i + 1] for i in range(n)))

def count(xs):
    c = {}
    for x in xs:
        if x in c:
            c[x] += 1
        else:
            c[x] = 1
    return c
def bleu(hyp, ref):
    n = len(hyp)
    L = len(ref)
    p = np.exp(min(0, (n - L) / L))
    for i in range(min(n, 4)):
        ch = count(ngram(hyp, i + 1))
        cr = count(ngram(ref, i + 1))
        p *= (sum(min(ch[j], cr[j]) for j in set(ch) & set(cr)) / (n - i)) ** (1 / 4)
    return p
def to_list(vec):
    ret = []
    for i in vec:
        if i == Pack.special.index('〈EOS〉'):
            break
        elif i != Pack.special.index('〈PAD〉'):
            ret.append(int(i))
    return ret
def score(y_pred, y_true, select):
    indices = np.where(select)[0]
    s = 0
    for y, i in zip(y_pred, indices):
        s += bleu(to_list(y), y_true[i])
    return s / len(indices)
def predict(model, ds, select, maxlen_src, maxlen_dst):
    model.cuda()
    model.eval()
    indices = np.where(select)[0]
    batches = range(0, len(indices), bs)
    y = np.empty((len(indices), maxlen_dst), dtype=np.long)
    with torch.no_grad():
        for n, i in enumerate(batches):
            batch = range(i, min(len(indices), i + bs))
            x, *_ = get_batch(ds, indices[batch], maxlen_src, maxlen_dst)
            y[batch] = model.translate(x.cuda(), maxlen_dst).cpu().numpy()
            sys.stdout.write('\rBatch\t{:5d} / {:5d}'.format(n+1, len(batches)))
    return y
def fit(model, ds, select, training=True, log_intvl=10, maxlen_src=400, maxlen_dst=40):
    model.cuda()
    model.train()
    indices = np.where(select)[0]
    batches = range(0, len(indices), bs)
    np.random.shuffle(indices)
    running_loss = Running()
    for n, i in enumerate(batches):
        batch = range(i, min(len(indices), i + bs))
        x, y_in, y_out = get_batch(ds, indices[batch], maxlen_src, maxlen_dst)
        y_out_ = model(x.cuda(), y_in.cuda())
        loss = F.cross_entropy(y_out_.transpose(2, 1), y_out.cuda())
        optim.zero_grad()
        loss.backward()
        optim.step()
        running_loss.add(float(loss), len(batch))
        sys.stdout.write('\rBatch\t{:5d} / {:5d}\t\tloss\t{:5f}'.format(n+1, len(batches), running_loss.get()))
    return running_loss.get()
maxlen_src = 300
maxlen_dst = 30
n_hid = 300
dropout_emb = 0.5
dropout_hid = 0.5
bs = 100
lrs = np.logspace(-.5, -1, 30)
log = []

model = BaselineLSTM(
    ds.src.vector.max() + 1, ds.dst.vector.max() + 1,
    n_hid = n_hid, dropout_emb = dropout_emb, dropout_hid = dropout_hid
)
optim = torch.optim.SGD(model.parameters(), lr=lrs[0], momentum=0.9)

for epoch, lr in enumerate(lrs):
    print('Epoch\t{:5d} / {:5d}\t\tlr\t{:5f}'.format(epoch+1, len(lrs), lr))
    optim.lr = lr
    bleus = []
    loss = fit(model, ds, train, maxlen_src, maxlen_dst)
    sys.stdout.write('\t\t%s\n' % 'train')
    for name, val in (('valid_same_file', valid_same_file), ('valid_same_repo', valid_same_repo), ('valid', valid)):
        bleus.append(score(predict(model, ds, val, maxlen_src, maxlen_dst), ds.dst, val))
        sys.stdout.write('\t\tscore\t{:5f}'.format(bleus[-1]))
        sys.stdout.write('\t\t%s\n' % name)
    log.append([loss, *bleus])
    print()
df = pd.DataFrame(log, columns=['loss', 'bleu_same_file', 'bleu_same_repo', 'bleu_diff_repo'])
df.to_csv('log.csv', index=False)
df
torch.save(model, 'model.pkl')
