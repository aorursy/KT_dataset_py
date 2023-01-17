%load_ext autoreload

%autoreload 2



%matplotlib inline
path = datasets.untar_data(datasets.URLs.IMDB)
path.ls() #note it contains a training folder, and unsupervised folder and a testeing folder 
#o the first thing we do is to create a datablok API list subclass called TextTest 

def read_file(fn): 

    with open(fn, 'r', encoding = 'utf8') as f: return f.read()

    

class TextList(ItemList):

    @classmethod

    def from_files(cls, path, extensions='.txt', recurse=True, include=None, **kwargs): #the get_files with a .txt file 

        return cls(get_files(path, extensions, recurse=recurse, include=include), path, **kwargs)

    

    def get(self, i): #all we have to do is overwrite get function to open a text file we do that in the read_file function above 

        if isinstance(i, Path): return read_file(i)

        return i
il = TextList.from_files(path, include=['train', 'test', 'unsup']) #and now we can create an intemlist 
len(il.items) #see the lenght of it


txt = il[0]#index into it 

txt
sd = SplitData.split_by_func(il, partial(random_splitter, p_valid=0.1)) #split the data in validation and trainin


sd


#export

import spacy,html
#export

#special tokens

UNK, PAD, BOS, EOS, TK_REP, TK_WREP, TK_UP, TK_MAJ = "xxunk xxpad xxbos xxeos xxrep xxwrep xxup xxmaj".split()



def sub_br(t): #fx if we find 'br /' we replace with a new line ('\n')

    "Replaces the <br /> by \n"

    re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)

    return re_br.sub("\n", t)



def spec_add_spaces(t): 

    "Add spaces around / and #"

    return re.sub(r'([/#])', r' \1 ', t)



def rm_useless_spaces(t):

    "Remove multiple spaces"

    return re.sub(' {2,}', ' ', t)



def replace_rep(t):

    "Replace repetitions at the character level: cccc -> TK_REP 4 c"

    def _replace_rep(m:Collection[str]) -> str:

        c,cc = m.groups()

        return f' {TK_REP} {len(cc)+1} {c} '

    re_rep = re.compile(r'(\S)(\1{3,})')

    return re_rep.sub(_replace_rep, t)

    

def replace_wrep(t):

    "Replace word repetitions: word word word -> TK_WREP 3 word"

    def _replace_wrep(m:Collection[str]) -> str:

        c,cc = m.groups()

        return f' {TK_WREP} {len(cc.split())+1} {c} '

    re_wrep = re.compile(r'(\b\w+\W+)(\1{3,})')

    return re_wrep.sub(_replace_wrep, t)



def fixup_text(x):

    "Various messy things we've seen in documents"

    re1 = re.compile(r'  +')

    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(

        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(

        '<br />', "\n").replace('\\"', '"').replace('<unk>',UNK).replace(' @.@ ','.').replace(

        ' @-@ ','-').replace('\\', ' \\ ')

    return re1.sub(' ', html.unescape(x))

    

default_pre_rules = [fixup_text, replace_rep, replace_wrep, spec_add_spaces, rm_useless_spaces, sub_br] #predefined rules these are bit of code that are run before the tokenizasion 

default_spec_tok = [UNK, PAD, BOS, EOS, TK_REP, TK_WREP, TK_UP, TK_MAJ] #default for our special tokens defined in the top of this cell 
replace_rep('cccc') #fx a repetitions of 4x'c' will be this:
replace_wrep('word word word word word ')#fx a repetitions of 5x'word' will be this:
def replace_all_caps(x):

    "Replace tokens in ALL CAPS by their lower version and add `TK_UP` before."

    res = []

    for t in x:

        if t.isupper() and len(t) > 1: res.append(TK_UP); res.append(t.lower())

        else: res.append(t)

    return res



def deal_caps(x):

    "Replace all Capitalized tokens in by their lower version and add `TK_MAJ` before."

    res = []

    for t in x:

        if t == '': continue

        if t[0].isupper() and len(t) > 1 and t[1:].islower(): res.append(TK_MAJ)

        res.append(t.lower())

    return res



def add_eos_bos(x): return [BOS] + x + [EOS] #reset its state with EOS, so it knows we are starting at a new word or something



default_post_rules = [deal_caps, replace_all_caps, add_eos_bos]
replace_all_caps(['I', 'AM', 'SHOUTING']) #put xxup before a caps word and make it lower chase so they can compare to all the other lowerchase words 
deal_caps(['My', 'name', 'is', 'Jeremy']) #same thing for mix chase just here we are using xxmaj and then the word in lowerchase 
from spacy.symbols import ORTH

from concurrent.futures import ProcessPoolExecutor



def parallel(func, arr, max_workers=4):

    if max_workers<2: results = list(progress_bar(map(func, enumerate(arr)), total=len(arr)))

    else:

        with ProcessPoolExecutor(max_workers=max_workers) as ex: #ProcessPoolExecutor run things in parallel so it runs faster 

            return list(progress_bar(ex.map(func, enumerate(arr)), total=len(arr)))

    if any([o is not None for o in results]): return results
class TokenizeProcessor(Processor):

    def __init__(self, lang="en", chunksize=2000, pre_rules=None, post_rules=None, max_workers=4): 

        self.chunksize,self.max_workers = chunksize,max_workers

        self.tokenizer = spacy.blank(lang).tokenizer

        for w in default_spec_tok:

            self.tokenizer.add_special_case(w, [{ORTH: w}])

        self.pre_rules  = default_pre_rules  if pre_rules  is None else pre_rules

        self.post_rules = default_post_rules if post_rules is None else post_rules



    def proc_chunk(self, args): #processing one chunk 

        i,chunk = args

        chunk = [compose(t, self.pre_rules) for t in chunk]

        docs = [[d.text for d in doc] for doc in self.tokenizer.pipe(chunk)]

        docs = [compose(t, self.post_rules) for t in docs]

        return docs



    def __call__(self, items): 

        toks = []

        if isinstance(items[0], Path): items = [read_file(i) for i in items]

        chunks = [items[i: i+self.chunksize] for i in (range(0, len(items), self.chunksize))]

        toks = parallel(self.proc_chunk, chunks, max_workers=self.max_workers) #so run all the chunks (pro_chunk) in parallel for all the chunks 

        return sum(toks, [])

    

    def proc1(self, item): return self.proc_chunk([item])[0]

    

    def deprocess(self, toks): return [self.deproc1(tok) for tok in toks]

    def deproc1(self, tok):    return " ".join(tok)
tp = TokenizeProcessor() #so we tokenize the texst
txt[:250] #eksampel of 250 words
' • '.join(tp(il[:100])[0])[:400] #and then tokenize them 
import collections



class NumericalizeProcessor(Processor):

    def __init__(self, vocab=None, max_vocab=60000, min_freq=2): 

        self.vocab,self.max_vocab,self.min_freq = vocab,max_vocab,min_freq

    

    def __call__(self, items):

        #The vocab is defined on the first use.

        if self.vocab is None: #chech if we have a vocab yet and the vocab is a list of all the unique words 

            #if we dont have it we will create it 

            freq = Counter(p for o in items for p in o)

            self.vocab = [o for o,c in freq.most_common(self.max_vocab) if c >= self.min_freq]

            for o in reversed(default_spec_tok):

                if o in self.vocab: self.vocab.remove(o)

                self.vocab.insert(0, o)

        if getattr(self, 'otoi', None) is None:

            self.otoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.vocab)}) 

        return [self.proc1(o) for o in items]

    def proc1(self, item):  return [self.otoi[o] for o in item] #call object to int on each one from the dictornary 

    

    def deprocess(self, idxs):

        assert self.vocab is not None

        return [self.deproc1(idx) for idx in idxs]

    def deproc1(self, idx): return [self.vocab[i] for i in idx] #just grapping ech one from vocab 
proc_tok,proc_num = TokenizeProcessor(max_workers=8),NumericalizeProcessor()
%time ll = label_by_func(sd, lambda x: 0, proc_x = [proc_tok,proc_num])
ll.train.x_obj(0)
pickle.dump(ll, open(path/'ld.pkl', 'wb')) #dump the labeled list 
ll = pickle.load(open(path/'ld.pkl', 'rb')) #so we can load it again later 
# Just using those for illustration purposes, they're not used otherwise.

from IPython.display import display,HTML

import pandas as pd
stream = """

In this notebook, we will go back over the example of classifying movie reviews we studied in part 1 and dig deeper under the surface. 

First we will look at the processing steps necessary to convert text into numbers and how to customize it. By doing this, we'll have another example of the Processor used in the data block API.

Then we will study how we build a language model and train it.\n

"""

tokens = np.array(tp([stream])[0])
bs,seq_len = 6,15#create a batch size of 6 so in pracise we get 6 lines of tekst 

d_tokens = np.array([tokens[i*seq_len:(i+1)*seq_len] for i in range(bs)])

df = pd.DataFrame(d_tokens)

display(HTML(df.to_html(index=False,header=None)))
bs,bptt = 6,5 

for k in range(3):

    d_tokens = np.array([tokens[i*seq_len + k*bptt:i*seq_len + (k+1)*bptt] for i in range(bs)])

    df = pd.DataFrame(d_tokens)

    display(HTML(df.to_html(index=False,header=None)))
#note the indepened veiable(X) is a word from above, and the depened(y) veriable is the word +1 so the next word in the sentence it is trying to predict 

#so lets create the depened veriable(y) here. Note the PreLoader is the same as the dataset in this chase

class LM_PreLoader():

    def __init__(self, data, bs=64, bptt=70, shuffle=False):

        self.data,self.bs,self.bptt,self.shuffle = data,bs,bptt,shuffle

        total_len = sum([len(t) for t in data.x])

        self.n_batch = total_len // bs

        self.batchify()

    

    def __len__(self): return ((self.n_batch-1) // self.bptt) * self.bs #a dataset is something with a lenght

    

    def __getitem__(self, idx): #and a getiem 

        source = self.batched_data[idx % self.bs]

        seq_idx = (idx // self.bs) * self.bptt

        return source[seq_idx:seq_idx+self.bptt],source[seq_idx+1:seq_idx+self.bptt+1] #so when you index into it it will grap the a indepened veriable(X)

    #and a depened veriable(y) and the indepened veriable is just the seq_idx:seq_idx+self.bptt --> a word in the text.

    #and the depened verible is just that word +1, so the word after it in a sentence so there is just a offset of 1 

    

    def batchify(self):

        texts = self.data.x

        if self.shuffle: texts = texts[torch.randperm(len(texts))]

        stream = torch.cat([tensor(t) for t in texts])

        self.batched_data = stream[:self.n_batch * self.bs].view(self.bs, self.n_batch)
dl = DataLoader(LM_PreLoader(ll.valid, shuffle=True), batch_size=64)
iter_dl = iter(dl)

x1,y1 = next(iter_dl)

x2,y2 = next(iter_dl)
x1.size(),y1.size()


vocab = proc_num.vocab
" ".join(vocab[o] for o in x1[0]) #grap a minibatch one at the time  for the indepened veriable


" ".join(vocab[o] for o in y1[0])#grap a minibatch one at the time  for the depened veriable and now we can see it has a offset of one 


" ".join(vocab[o] for o in x2[0])
#now we can refactor it into a function 

def get_lm_dls(train_ds, valid_ds, bs, bptt, **kwargs):

    return (DataLoader(LM_PreLoader(train_ds, bs, bptt, shuffle=True), batch_size=bs, **kwargs),

            DataLoader(LM_PreLoader(valid_ds, bs, bptt, shuffle=False), batch_size=2*bs, **kwargs))



def lm_databunchify(sd, bs, bptt, **kwargs):

    return DataBunch(*get_lm_dls(sd.train, sd.valid, bs, bptt, **kwargs))


bs,bptt = 64,70

data = lm_databunchify(ll, bs, bptt)
proc_cat = CategoryProcessor()


il = TextList.from_files(path, include=['train', 'test']) #create a item list

sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='test')) #split the data 

ll = label_by_func(sd, parent_labeler, proc_x = [proc_tok, proc_num], proc_y=proc_cat) #and we label it note we have added to preprocessors --> proc_x = [proc_tok, proc_num]


pickle.dump(ll, open(path/'ll_clas.pkl', 'wb'))
ll = pickle.load(open(path/'ll_clas.pkl', 'rb'))
[(ll.train.x_obj(i), ll.train.y_obj(i)) for i in [1,12552]]


#export

from torch.utils.data import Sampler

#for validation set

class SortSampler(Sampler):

    def __init__(self, data_source, key): self.data_source,self.key = data_source,key

    def __len__(self): return len(self.data_source)

    def __iter__(self):

        return iter(sorted(list(range(len(self.data_source))), key=self.key, reverse=True)) #goes through our data --> (data_source) looks how many dokuments there are in it --> (len(self.data_source)

    #create the list from 0 to the number of dokuments --> list(range(len(self.data_source))) and sort them in reversed order (sorted & reverse=True) by some key (self.key and it is a lambda function defined longer data that just tkes the lenght of the dokument ) and returns that iterator(iter)
#we get different bacth sizes so we have to handle that. so the trick is to sort the data first byt lenght so the first minibatch will contain the your realy long

#dokuments and your last minibatch will contain realy short dokuments 



#for training set 

#so it sorts like for the validationset where all minibacth has something at simular lenght but with some randomness#note they do not have identical lenght but just simular  

class SortishSampler(Sampler):

    def __init__(self, data_source, key, bs):

        self.data_source,self.key,self.bs = data_source,key,bs



    def __len__(self) -> int: return len(self.data_source)



    def __iter__(self):

        idxs = torch.randperm(len(self.data_source)) #add random permitation in the megabatches 

        megabatches = [idxs[i:i+self.bs*50] for i in range(0, len(idxs), self.bs*50)] #mega batch is 50 times bigger then a minibatch 

        sorted_idx = torch.cat([tensor(sorted(s, key=self.key, reverse=True)) for s in megabatches]) #sort those megebatches 

        batches = [sorted_idx[i:i+self.bs] for i in range(0, len(sorted_idx), self.bs)]

        max_idx = torch.argmax(tensor([self.key(ck[0]) for ck in batches]))  # find the chunk with the largest key,

        batches[0],batches[max_idx] = batches[max_idx],batches[0]            # then make sure it goes first.

        batch_idxs = torch.randperm(len(batches)-2)

        sorted_idx = torch.cat([batches[i+1] for i in batch_idxs]) if len(batches) > 1 else LongTensor([])

        sorted_idx = torch.cat([batches[0], sorted_idx, batches[-1]])

        return iter(sorted_idx)
#since we have different sizes bathces we need to write a new collate function 

def pad_collate(samples, pad_idx=1, pad_first=False):

    max_len = max([len(s[0]) for s in samples])

    res = torch.zeros(len(samples), max_len).long() + pad_idx #create something that can handle the longest minibatch in the dokument

    for i,s in enumerate(samples): #go through every dokument

        if pad_first: res[i, -len(s[0]):] = LongTensor(s[0]) #and dump it into the big tensor eihter at the end..

        else:         res[i, :len(s[0]) ] = LongTensor(s[0])#.. or at the start 

    return res, tensor([s[1] for s in samples])


bs = 64

train_sampler = SortishSampler(ll.train.x, key=lambda t: len(ll.train[int(t)][0]), bs=bs)

train_dl = DataLoader(ll.train, batch_size=bs, sampler=train_sampler, collate_fn=pad_collate) #pass the sampler and collate to our dataloader 


iter_dl = iter(train_dl)

x,y = next(iter_dl)
lengths = []

for i in range(x.size(0)): lengths.append(x.size(1) - (x[i]==1).sum().item())

lengths[:5], lengths[-1]
x,y = next(iter_dl) #grap a minibatch where we can see padding at the end 

lengths = []

for i in range(x.size(0)): lengths.append(x.size(1) - (x[i]==1).sum().item())

lengths[:5], lengths[-1]
x
#noe lets refactor it into a convenience funtion that did the above 

def get_clas_dls(train_ds, valid_ds, bs, **kwargs):

    train_sampler = SortishSampler(train_ds.x, key=lambda t: len(train_ds.x[t]), bs=bs)

    valid_sampler = SortSampler(valid_ds.x, key=lambda t: len(valid_ds.x[t]))

    return (DataLoader(train_ds, batch_size=bs, sampler=train_sampler, collate_fn=pad_collate, **kwargs),

            DataLoader(valid_ds, batch_size=bs*2, sampler=valid_sampler, collate_fn=pad_collate, **kwargs))



def clas_databunchify(sd, bs, **kwargs):

    return DataBunch(*get_clas_dls(sd.train, sd.valid, bs, **kwargs))
bs,bptt = 64,70

data = clas_databunchify(ll, bs)
%load_ext autoreload

%autoreload 2



%matplotlib inline
path = datasets.untar_data(datasets.URLs.IMDB)
il = TextList.from_files(path, include=['train', 'test', 'unsup'])

sd = SplitData.split_by_func(il, partial(random_splitter, p_valid=0.1))


proc_tok,proc_num = TokenizeProcessor(max_workers=8),NumericalizeProcessor()


ll = label_by_func(sd, lambda x: 0, proc_x = [proc_tok,proc_num])


pickle.dump(ll, open(path/'ll_lm.pkl', 'wb'))

pickle.dump(proc_num.vocab, open(path/'vocab_lm.pkl', 'wb'))
ll = pickle.load(open(path/'ll_lm.pkl', 'rb'))

vocab = pickle.load(open(path/'vocab_lm.pkl', 'rb'))
bs,bptt = 64,70

data = lm_databunchify(ll, bs, bptt)
#model is able to forget and only remember importen things 

class LSTMCell(nn.Module):

    def __init__(self, ni, nh):

        super().__init__()

        self.ih = nn.Linear(ni,4*nh) #from input to hidden layer(ih) and nh is the numbers of hiddan layers normaly 

        self.hh = nn.Linear(nh,4*nh)#from hidden to hidden layer(hh)



    def forward(self, input, state):

        h,c = state

        #One big multiplication for all the gates is better than 4 smaller ones

        gates = (self.ih(input) + self.hh(h)).chunk(4, 1) #split it up in four chunks or in other words splits it up 4 groups of same size 

        ingate,forgetgate,outgate = map(torch.sigmoid, gates[:3]) #3 of them goes through a sigmoid 

        cellgate = gates[3].tanh() #one of them goes throug a tanh



        c = (forgetgate*c) + (ingate*cellgate) #multiply and add from picture above

        h = outgate * c.tanh() #multiply from picture above 

        return h, (h,c)
class LSTMLayer(nn.Module):

    def __init__(self, cell, *cell_args):

        super().__init__()

        self.cell = cell(*cell_args)



    def forward(self, input, state):

        inputs = input.unbind(1)

        outputs = []

        for i in range(len(inputs)): #for loop from before 

            out, state = self.cell(inputs[i], state) #call on whatever cell we call for and in this chase it is a LSTMCell #take the state and update the state 

            outputs += [out]

        return torch.stack(outputs, dim=1), state
lstm = LSTMLayer(LSTMCell, 300, 300) 


x = torch.randn(64, 70, 300)

h = (torch.zeros(64, 300),torch.zeros(64, 300))
%timeit -n 10 y,h1 = lstm(x,h)
lstm = lstm.cuda()

x = x.cuda()

h = (h[0].cuda(), h[1].cuda())
def time_fn(f):

    f()

    torch.cuda.synchronize()


f = partial(lstm,x,h)

time_fn(f)
%timeit -n 10 time_fn(f)
lstm = nn.LSTM(300, 300, 1, batch_first=True) #better faster and great just use this LSTM instead of the self build 
x = torch.randn(64, 70, 300)

h = (torch.zeros(1, 64, 300),torch.zeros(1, 64, 300))
%timeit -n 10 y,h1 = lstm(x,h)
lstm = lstm.cuda()

x = x.cuda()

h = (h[0].cuda(), h[1].cuda())
f = partial(lstm,x,h)

time_fn(f)
%timeit -n 10 time_fn(f)
import torch.jit as jit

from torch import Tensor
#takes the python code and compile it into C++ code so we can creae an on GPU loop (loop from code above)#note jit is hirt and doesnt realy work right now

class LSTMCell(jit.ScriptModule):

    def __init__(self, ni, nh):

        super().__init__()

        self.ni = ni

        self.nh = nh

        self.w_ih = nn.Parameter(torch.randn(4 * nh, ni))

        self.w_hh = nn.Parameter(torch.randn(4 * nh, nh))

        self.bias_ih = nn.Parameter(torch.randn(4 * nh))

        self.bias_hh = nn.Parameter(torch.randn(4 * nh))



    @jit.script_method

    def forward(self, input:Tensor, state:Tuple[Tensor, Tensor])->Tuple[Tensor, Tuple[Tensor, Tensor]]:

        hx, cx = state

        gates = (input @ self.w_ih.t() + self.bias_ih +

                 hx @ self.w_hh.t() + self.bias_hh)

        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)



        ingate = torch.sigmoid(ingate)

        forgetgate = torch.sigmoid(forgetgate)

        cellgate = torch.tanh(cellgate)

        outgate = torch.sigmoid(outgate)



        cy = (forgetgate * cx) + (ingate * cellgate)

        hy = outgate * torch.tanh(cy)



        return hy, (hy, cy)
class LSTMLayer(jit.ScriptModule):

    def __init__(self, cell, *cell_args):

        super().__init__()

        self.cell = cell(*cell_args)



    @jit.script_method

    def forward(self, input:Tensor, state:Tuple[Tensor, Tensor])->Tuple[Tensor, Tuple[Tensor, Tensor]]:

        inputs = input.unbind(1)

        outputs = []

        for i in range(len(inputs)):

            out, state = self.cell(inputs[i], state)

            outputs += [out]

        return torch.stack(outputs, dim=1), state
lstm = LSTMLayer(LSTMCell, 300, 300)


x = torch.randn(64, 70, 300)

h = (torch.zeros(64, 300),torch.zeros(64, 300))
%timeit -n 10 y,h1 = lstm(x,h)


lstm = lstm.cuda()

x = x.cuda()

h = (h[0].cuda(), h[1].cuda())
f = partial(lstm,x,h)

time_fn(f)
%timeit -n 10 time_fn(f)


#export

def dropout_mask(x, sz, p):

    return x.new(*sz).bernoulli_(1-p).div_(1-p) #bernoulli_ means creae 1s and 0s (1-p) and divide by 1-p 
x = torch.randn(10,10)

mask = dropout_mask(x, (10,10), 0.5); mask
(x*mask).std(),x.std()


#export

class RNNDropout(nn.Module):

    def __init__(self, p=0.5):

        super().__init__()

        self.p=p



    def forward(self, x):

        if not self.training or self.p == 0.: return x

        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)

        return x * m
dp = RNNDropout(0.3)

tst_input = torch.randn(3,3,7)

tst_input, dp(tst_input)


#export

import warnings



WEIGHT_HH = 'weight_hh_l0'



class WeightDropout(nn.Module): #same as dropconnect does the same it does not only dropouts on the weights but also the activations 

    def __init__(self, module, weight_p=[0.], layer_names=[WEIGHT_HH]):

        super().__init__()

        self.module,self.weight_p,self.layer_names = module,weight_p,layer_names

        for layer in self.layer_names:

            #Makes a copy of the weights of the selected layers.

            w = getattr(self.module, layer)

            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))

            self.module._parameters[layer] = F.dropout(w, p=self.weight_p, training=False)



    def _setweights(self):

        for layer in self.layer_names:

            raw_w = getattr(self, f'{layer}_raw')

            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=self.training)



    def forward(self, *args):

        self._setweights()

        with warnings.catch_warnings():

            #To avoid the warning that comes because the weights aren't flattened.

            warnings.simplefilter("ignore")

            return self.module.forward(*args)
module = nn.LSTM(5, 2)

dp_module = WeightDropout(module, 0.4)

getattr(dp_module.module, WEIGHT_HH)
tst_input = torch.randn(4,20,5)

h = (torch.zeros(1,20,2), torch.zeros(1,20,2))

x,h = dp_module(tst_input,h)

getattr(dp_module.module, WEIGHT_HH)
#drop out an whole row so it is dropping out whole words in this chase 

class EmbeddingDropout(nn.Module):

    "Applies dropout in the embedding layer by zeroing out some elements of the embedding vector."

    def __init__(self, emb, embed_p):

        super().__init__()

        self.emb,self.embed_p = emb,embed_p

        self.pad_idx = self.emb.padding_idx

        if self.pad_idx is None: self.pad_idx = -1



    def forward(self, words, scale=None):

        if self.training and self.embed_p != 0:

            size = (self.emb.weight.size(0),1)

            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)

            masked_embed = self.emb.weight * mask

        else: masked_embed = self.emb.weight

        if scale: masked_embed.mul_(scale)

        return F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,

                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)
enc = nn.Embedding(100, 7, padding_idx=1)

enc_dp = EmbeddingDropout(enc, 0.5)

tst_input = torch.randint(0,100,(8,))

enc_dp(tst_input)


#export

def to_detach(h):

    "Detaches `h` from its history."

    return h.detach() if type(h) == torch.Tensor else tuple(to_detach(v) for v in h)
#export

class AWD_LSTM(nn.Module):

    "AWD-LSTM inspired by https://arxiv.org/abs/1708.02182."

    initrange=0.1



    def __init__(self, vocab_sz, emb_sz, n_hid, n_layers, pad_token,

                 hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5):

        super().__init__()

        self.bs,self.emb_sz,self.n_hid,self.n_layers = 1,emb_sz,n_hid,n_layers

        self.emb = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)

        self.emb_dp = EmbeddingDropout(self.emb, embed_p)

        self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz), 1, #create a LSTM model for some layers 

                             batch_first=True) for l in range(n_layers)]

        self.rnns = nn.ModuleList([WeightDropout(rnn, weight_p) for rnn in self.rnns]) 

        self.emb.weight.data.uniform_(-self.initrange, self.initrange)

        self.input_dp = RNNDropout(input_p)

        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])



    def forward(self, input):

        bs,sl = input.size()

        if bs!=self.bs:

            self.bs=bs

            self.reset()

        raw_output = self.input_dp(self.emb_dp(input))

        new_hidden,raw_outputs,outputs = [],[],[]

        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)): #call all the different kinds of dropout in this loop 

            raw_output, new_h = rnn(raw_output, self.hidden[l])

            new_hidden.append(new_h)

            raw_outputs.append(raw_output)

            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)

            outputs.append(raw_output) 

        self.hidden = to_detach(new_hidden)

        return raw_outputs, outputs



    def _one_hidden(self, l):

        "Return one hidden state."

        nh = self.n_hid if l != self.n_layers - 1 else self.emb_sz

        return next(self.parameters()).new(1, self.bs, nh).zero_()



    def reset(self):

        "Reset the hidden states."

        self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]
#untop of the LSTM model we can put a linear model untop 

class LinearDecoder(nn.Module):

    def __init__(self, n_out, n_hid, output_p, tie_encoder=None, bias=True):

        super().__init__()

        self.output_dp = RNNDropout(output_p) #with dropout

        self.decoder = nn.Linear(n_hid, n_out, bias=bias) #and layers 

        if bias: self.decoder.bias.data.zero_()

        if tie_encoder: self.decoder.weight = tie_encoder.weight

        else: init.kaiming_uniform_(self.decoder.weight)



    def forward(self, input):

        raw_outputs, outputs = input

        output = self.output_dp(outputs[-1]).contiguous()

        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))

        return decoded, raw_outputs, outputs
class SequentialRNN(nn.Sequential):

    "A sequential module that passes the reset call to its children."

    def reset(self):

        for c in self.children():

            if hasattr(c, 'reset'): c.reset()
#export

def get_language_model(vocab_sz, emb_sz, n_hid, n_layers, pad_token, output_p=0.4, hidden_p=0.2, input_p=0.6, 

                       embed_p=0.1, weight_p=0.5, tie_weights=True, bias=True):

    rnn_enc = AWD_LSTM(vocab_sz, emb_sz, n_hid=n_hid, n_layers=n_layers, pad_token=pad_token,

                       hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p)

    enc = rnn_enc.emb if tie_weights else None

    return SequentialRNN(rnn_enc, LinearDecoder(vocab_sz, emb_sz, output_p, tie_encoder=enc, bias=bias)) #pass the whole seuential model to the LinearDecode which is the one that are gonna predict the next word in a sentence 
tok_pad = vocab.index(PAD)
tst_model = get_language_model(len(vocab), 300, 300, 2, tok_pad)

tst_model = tst_model.cuda()
x,y = next(iter(data.train_dl))
z = tst_model(x.cuda())
len(z)


decoded, raw_outputs, outputs = z
decoded.size()
len(raw_outputs),len(outputs)
[o.size() for o in raw_outputs], [o.size() for o in outputs]


#export

class GradientClipping(Callback):

    def __init__(self, clip=None): self.clip = clip

    def after_backward(self):

        if self.clip:  nn.utils.clip_grad_norm_(self.run.model.parameters(), self.clip) #check the gradient after a after backword pss has benn called and if the total norm of the gradient is bigger then

            #by some number (clip) we will devide them all by some number so it is clipping those gradients 

            #so its let you trtain with a higher learning rate and avoid the gradient of overfitting 


#export

class RNNTrainer(Callback):

    def __init__(self, α, β): self.α,self.β = α,β

    

    def after_pred(self):

        #Save the extra outputs for later and only returns the true output.

        self.raw_out,self.out = self.pred[1],self.pred[2]

        self.run.pred = self.pred[0]

    

    def after_loss(self):

        #AR and TAR

        if self.α != 0.:  self.run.loss += self.α * self.out[-1].float().pow(2).mean() # Activation Regularization #the L2'staf' is not on the weigths but on the activations this time this is gonna make sure our activations is never to high 

        if self.β != 0.:

            h = self.raw_out[-1]

            if h.size(1)>1: self.run.loss += self.β * (h[:,1:] - h[:,:-1]).float().pow(2).mean() #Temporal Activation Regularization # which check how much does each activation change by from seqence step to seqence step and take the squere og that 

                #does this becaue its not a good idea for having things taht massive change from time step to time step 

    def begin_epoch(self):

        #Shuffle the texts at the beginning of the epoch

        if hasattr(self.dl.dataset, "batchify"): self.dl.dataset.batchify()


#export

def cross_entropy_flat(input, target):

    bs,sl = target.size()

    return F.cross_entropy(input.view(bs * sl, -1), target.view(bs * sl))



def accuracy_flat(input, target):

    bs,sl = target.size()

    return accuracy(input.view(bs * sl, -1), target.view(bs * sl))
emb_sz, nh, nl = 300, 300, 2

model = get_language_model(len(vocab), emb_sz, nh, nl, tok_pad, input_p=0.6, output_p=0.4, weight_p=0.5, 

                           embed_p=0.1, hidden_p=0.2)
cbs = [partial(AvgStatsCallback,accuracy_flat),

       CudaCallback, Recorder,

       partial(GradientClipping, clip=0.1),

       partial(RNNTrainer, α=2., β=1.),

       ProgressCallback]
learn = Learner(model, data, cross_entropy_flat, lr=5e-3, cb_funcs=cbs, opt_func=adam_opt())
learn.fit(1) 
%load_ext autoreload

%autoreload 2



%matplotlib inline


#path = datasets.Config().data_path()

#version = '103' #2


#! wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-{version}-v1.zip -P {path}

#! unzip -q -n {path}/wikitext-{version}-v1.zip  -d {path}

#! mv {path}/wikitext-{version}/wiki.train.tokens {path}/wikitext-{version}/train.txt

#! mv {path}/wikitext-{version}/wiki.valid.tokens {path}/wikitext-{version}/valid.txt

#! mv {path}/wikitext-{version}/wiki.test.tokens {path}/wikitext-{version}/test.txt
path = datasets.Config().data_path()/'wikitext-103'


def istitle(line):

    return len(re.findall(r'^ = [^=]* = $', line)) != 0
def read_wiki(filename):

    articles = []

    with open(filename, encoding='utf8') as f:

        lines = f.readlines()

    current_article = ''

    for i,line in enumerate(lines):

        current_article += line

        if i < len(lines)-2 and lines[i+1] == ' \n' and istitle(lines[i+2]):

            current_article = current_article.replace('<unk>', UNK)

            articles.append(current_article)

            current_article = ''

    current_article = current_article.replace('<unk>', UNK)

    articles.append(current_article)

    return articles


train = TextList(read_wiki(path/'train.txt'), path=path) #+read_file(path/'test.txt')

valid = TextList(read_wiki(path/'valid.txt'), path=path)
len(train), len(valid)


sd = SplitData(train, valid)


proc_tok,proc_num = TokenizeProcessor(),NumericalizeProcessor()
ll = label_by_func(sd, lambda x: 0, proc_x = [proc_tok,proc_num])
pickle.dump(ll, open(path/'ld.pkl', 'wb'))


ll = pickle.load( open(path/'ld.pkl', 'rb'))
bs,bptt = 128,70

data = lm_databunchify(ll, bs, bptt)
vocab = ll.train.proc_x[-1].vocab

len(vocab)
dps = np.array([0.1, 0.15, 0.25, 0.02, 0.2]) * 0.2

tok_pad = vocab.index(PAD)


emb_sz, nh, nl = 300, 300, 2

model = get_language_model(len(vocab), emb_sz, nh, nl, tok_pad, *dps)
cbs = [partial(AvgStatsCallback,accuracy_flat),

       CudaCallback, Recorder,

       partial(GradientClipping, clip=0.1),

       partial(RNNTrainer, α=2., β=1.),

       ProgressCallback]
learn = Learner(model, data, cross_entropy_flat, lr=5e-3, cb_funcs=cbs, opt_func=adam_opt())


lr = 5e-3

sched_lr  = combine_scheds([0.3,0.7], cos_1cycle_anneal(lr/10., lr, lr/1e5))

sched_mom = combine_scheds([0.3,0.7], cos_1cycle_anneal(0.8, 0.7, 0.8))

cbsched = [ParamScheduler('lr', sched_lr), ParamScheduler('mom', sched_mom)]
learn.fit(10, cbs=cbsched)
torch.save(learn.model.state_dict(), path/'pretrained.pth')

pickle.dump(vocab, open(path/'vocab.pkl', 'wb'))
%load_ext autoreload

%autoreload 2



%matplotlib inline
path = datasets.untar_data(datasets.URLs.IMDB)


ll = pickle.load(open(path/'ll_lm.pkl', 'rb'))
bs,bptt = 128,70

data = lm_databunchify(ll, bs, bptt)
vocab = ll.train.proc_x[1].vocab
# ! wget http://files.fast.ai/models/wt103_tiny.tgz -P {path}

# ! tar xf {path}/wt103_tiny.tgz -C {path}


dps = tensor([0.1, 0.15, 0.25, 0.02, 0.2]) * 0.5

tok_pad = vocab.index(PAD)
emb_sz, nh, nl = 300, 300, 2

model = get_language_model(len(vocab), emb_sz, nh, nl, tok_pad, *dps)


old_wgts  = torch.load(path/'pretrained'/'pretrained.pth')

old_vocab = pickle.load(open(path/'pretrained'/'vocab.pkl', 'rb'))


idx_house_new, idx_house_old = vocab.index('house'),old_vocab.index('house')
house_wgt  = old_wgts['0.emb.weight'][idx_house_old]

house_bias = old_wgts['1.decoder.bias'][idx_house_old]
#so now we are gonna have different vocab since we are working on a new text that are not from the wiki dataset 

def match_embeds(old_wgts, old_vocab, new_vocab):

    wgts = old_wgts['0.emb.weight']

    bias = old_wgts['1.decoder.bias']

    wgts_m,bias_m = wgts.mean(dim=0),bias.mean()

    new_wgts = wgts.new_zeros(len(new_vocab), wgts.size(1))

    new_bias = bias.new_zeros(len(new_vocab))

    otoi = {v:k for k,v in enumerate(old_vocab)}

    for i,w in enumerate(new_vocab): 

        if w in otoi:# so go through all the embedding vocabs and if it is there we are just gonna copy the embedding over and igf it is not here we are gonna take..

            idx = otoi[w]

            new_wgts[i],new_bias[i] = wgts[idx],bias[idx]

        else: new_wgts[i],new_bias[i] = wgts_m,bias_m #... the weigth and biases and create one 

    old_wgts['0.emb.weight']    = new_wgts

    old_wgts['0.emb_dp.emb.weight'] = new_wgts

    old_wgts['1.decoder.weight']    = new_wgts

    old_wgts['1.decoder.bias']      = new_bias

    return old_wgts
wgts = match_embeds(old_wgts, old_vocab, vocab)
test_near(wgts['0.emb.weight'][idx_house_new],house_wgt)

test_near(wgts['1.decoder.bias'][idx_house_new],house_bias)
model.load_state_dict(wgts)
model
#normal slitter for the data 

def lm_splitter(m):

    groups = []

    for i in range(len(m[0].rnns)): groups.append(nn.Sequential(m[0].rnns[i], m[0].hidden_dps[i]))

    groups += [nn.Sequential(m[0].emb, m[0].emb_dp, m[0].input_dp, m[1])]

    return [list(o.parameters()) for o in groups]
for rnn in model[0].rnns:

    for p in rnn.parameters(): p.requires_grad_(False)
#setup the calllback 

cbs = [partial(AvgStatsCallback,accuracy_flat),

       CudaCallback, Recorder,

       partial(GradientClipping, clip=0.1),

       partial(RNNTrainer, α=2., β=1.),

       ProgressCallback]
learn = Learner(model, data, cross_entropy_flat, opt_func=adam_opt(), #setup our leaner 

                cb_funcs=cbs, splitter=lm_splitter)
lr = 2e-2

cbsched = sched_1cycle([lr], pct_start=0.5, mom_start=0.8, mom_mid=0.7, mom_end=0.8)
learn.fit(1, cbs=cbsched)


for rnn in model[0].rnns:

    for p in rnn.parameters(): p.requires_grad_(True)
lr = 2e-3

cbsched = sched_1cycle([lr/2., lr/2., lr], pct_start=0.5, mom_start=0.8, mom_mid=0.7, mom_end=0.8)
learn.fit(10, cbs=cbsched)
torch.save(learn.model[0].state_dict(), path/'finetuned_enc.pth')


pickle.dump(vocab, open(path/'vocab_lm.pkl', 'wb'))


torch.save(learn.model.state_dict(), path/'finetuned.pth')
vocab = pickle.load(open(path/'vocab_lm.pkl', 'rb'))

proc_tok,proc_num,proc_cat = TokenizeProcessor(),NumericalizeProcessor(vocab=vocab),CategoryProcessor()
il = TextList.from_files(path, include=['train', 'test']) #load up our classifer data bunch 

sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='test'))

ll = label_by_func(sd, parent_labeler, proc_x = [proc_tok, proc_num], proc_y=proc_cat)
pickle.dump(ll, open(path/'ll_clas.pkl', 'wb'))


ll = pickle.load(open(path/'ll_clas.pkl', 'rb'))

vocab = pickle.load(open(path/'vocab_lm.pkl', 'rb'))
bs,bptt = 64,70

data = clas_databunchify(ll, bs)
#export

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


x,y = next(iter(data.train_dl))


x.size()
lengths = x.size(1) - (x == 1).sum(1)

lengths[:5]
tst_emb = nn.Embedding(len(vocab), 300)
tst_emb(x).shape
128*70
packed = pack_padded_sequence(tst_emb(x), lengths, batch_first=True)
packed
packed.data.shape
len(packed.batch_sizes)
8960//70
tst = nn.LSTM(300, 300, 2)
y,h = tst(packed)


unpack = pad_packed_sequence(y, batch_first=True)


unpack[0].shape
unpack[1]
#export

class AWD_LSTM1(nn.Module):

    "AWD-LSTM inspired by https://arxiv.org/abs/1708.02182."

    initrange=0.1



    def __init__(self, vocab_sz, emb_sz, n_hid, n_layers, pad_token,

                 hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5):

        super().__init__()

        self.bs,self.emb_sz,self.n_hid,self.n_layers,self.pad_token = 1,emb_sz,n_hid,n_layers,pad_token

        self.emb = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)

        self.emb_dp = EmbeddingDropout(self.emb, embed_p)

        self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz), 1,

                             batch_first=True) for l in range(n_layers)]

        self.rnns = nn.ModuleList([WeightDropout(rnn, weight_p) for rnn in self.rnns])

        self.emb.weight.data.uniform_(-self.initrange, self.initrange)

        self.input_dp = RNNDropout(input_p)

        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])



    def forward(self, input):

        bs,sl = input.size()

        mask = (input == self.pad_token)

        lengths = sl - mask.long().sum(1)

        n_empty = (lengths == 0).sum()

        if n_empty > 0:

            input = input[:-n_empty]

            lengths = lengths[:-n_empty]

            self.hidden = [(h[0][:,:input.size(0)], h[1][:,:input.size(0)]) for h in self.hidden]

        raw_output = self.input_dp(self.emb_dp(input))

        new_hidden,raw_outputs,outputs = [],[],[]

        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):

            raw_output = pack_padded_sequence(raw_output, lengths, batch_first=True)#take data of different lenght and call on pack_padded_sequence...

            raw_output, new_h = rnn(raw_output, self.hidden[l]) #... and pass that to a rnn ...

            raw_output = pad_packed_sequence(raw_output, batch_first=True)[0] #... and call on pad_packed_sequence and it basicaly takes thing of different lenght and opponialy handles them in a rnn 

            raw_outputs.append(raw_output)

            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)

            outputs.append(raw_output)

            new_hidden.append(new_h)

        self.hidden = to_detach(new_hidden)

        return raw_outputs, outputs, mask



    def _one_hidden(self, l):

        "Return one hidden state."

        nh = self.n_hid if l != self.n_layers - 1 else self.emb_sz

        return next(self.parameters()).new(1, self.bs, nh).zero_()



    def reset(self):

        "Reset the hidden states."

        self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]
#finds which bit of state we wonna use for cassifcation 

class Pooling(nn.Module):

    def forward(self, input):

        raw_outputs,outputs,mask = input

        output = outputs[-1]

        lengths = output.size(1) - mask.long().sum(dim=1)

        avg_pool = output.masked_fill(mask[:,:,None], 0).sum(dim=1) #do an averrge pool 

        avg_pool.div_(lengths.type(avg_pool.dtype)[:,None])

        max_pool = output.masked_fill(mask[:,:,None], -float('inf')).max(dim=1)[0] #and a max pool 

        x = torch.cat([output[torch.arange(0, output.size(0)),lengths-1], max_pool, avg_pool], 1) #and use the final state output[torch.arange(0, output.size(0)),lengths-1]

        # then we Concat pooling tehm all togehter 

        return output,x
emb_sz, nh, nl = 300, 300, 2

tok_pad = vocab.index(PAD)
enc = AWD_LSTM1(len(vocab), emb_sz, n_hid=nh, n_layers=nl, pad_token=tok_pad)

pool = Pooling()

enc.bs = bs

enc.reset()
x,y = next(iter(data.train_dl))

output,c = pool(enc(x))
x


test_near((output.sum(dim=2) == 0).float(), (x==tok_pad).float())
for i in range(bs):

    length = x.size(1) - (x[i]==1).long().sum()

    out_unpad = output[i,:length]

    test_near(out_unpad[-1], c[i,:300])

    test_near(out_unpad.max(0)[0], c[i,300:600])

    test_near(out_unpad.mean(0), c[i,600:])
def bn_drop_lin(n_in, n_out, bn=True, p=0., actn=None):

    layers = [nn.BatchNorm1d(n_in)] if bn else []

    if p != 0: layers.append(nn.Dropout(p))

    layers.append(nn.Linear(n_in, n_out))

    if actn is not None: layers.append(actn)

    return layers
class PoolingLinearClassifier(nn.Module):

    "Create a linear classifier with pooling."



    def __init__(self, layers, drops):

        super().__init__()

        mod_layers = []

        activs = [nn.ReLU(inplace=True)] * (len(layers) - 2) + [None]

        for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], drops, activs):

            mod_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn) #just a list batch norm dropout linear layers 

        self.layers = nn.Sequential(*mod_layers)



    def forward(self, input):

        raw_outputs,outputs,mask = input

        output = outputs[-1]

        lengths = output.size(1) - mask.long().sum(dim=1)

        avg_pool = output.masked_fill(mask[:,:,None], 0).sum(dim=1)

        avg_pool.div_(lengths.type(avg_pool.dtype)[:,None])

        max_pool = output.masked_fill(mask[:,:,None], -float('inf')).max(dim=1)[0]

        x = torch.cat([output[torch.arange(0, output.size(0)),lengths-1], max_pool, avg_pool], 1) #Concat pooling.

        x = self.layers(x)

        return x
def pad_tensor(t, bs, val=0.):

    if t.size(0) < bs:

        return torch.cat([t, val + t.new_zeros(bs-t.size(0), *t.shape[1:])])

    return t
class SentenceEncoder(nn.Module):

    def __init__(self, module, bptt, pad_idx=1):

        super().__init__()

        self.bptt,self.module,self.pad_idx = bptt,module,pad_idx



    def concat(self, arrs, bs):

        return [torch.cat([pad_tensor(l[si],bs) for l in arrs], dim=1) for si in range(len(arrs[0]))]

    

    def forward(self, input):

        bs,sl = input.size()

        self.module.bs = bs

        self.module.reset()

        raw_outputs,outputs,masks = [],[],[]

        for i in range(0, sl, self.bptt): #go through our sentence one bptt at the time 

            r,o,m = self.module(input[:,i: min(i+self.bptt, sl)]) #keep calling that thing 

            masks.append(pad_tensor(m, bs, 1)) #and appending the result 

            raw_outputs.append(r)

            outputs.append(o)

        return self.concat(raw_outputs, bs),self.concat(outputs, bs),torch.cat(masks,dim=1)
def get_text_classifier(vocab_sz, emb_sz, n_hid, n_layers, n_out, pad_token, bptt, output_p=0.4, hidden_p=0.2, 

                        input_p=0.6, embed_p=0.1, weight_p=0.5, layers=None, drops=None):

    "To create a full AWD-LSTM"

    rnn_enc = AWD_LSTM1(vocab_sz, emb_sz, n_hid=n_hid, n_layers=n_layers, pad_token=pad_token,

                        hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p)

    enc = SentenceEncoder(rnn_enc, bptt)

    if layers is None: layers = [50]

    if drops is None:  drops = [0.1] * len(layers)

    layers = [3 * emb_sz] + layers + [n_out] 

    drops = [output_p] + drops

    return SequentialRNN(enc, PoolingLinearClassifier(layers, drops))


emb_sz, nh, nl = 300, 300, 2

dps = tensor([0.4, 0.3, 0.4, 0.05, 0.5]) * 0.25

model = get_text_classifier(len(vocab), emb_sz, nh, nl, 2, 1, bptt, *dps)


def class_splitter(m):

    enc = m[0].module

    groups = [nn.Sequential(enc.emb, enc.emb_dp, enc.input_dp)]

    for i in range(len(enc.rnns)): groups.append(nn.Sequential(enc.rnns[i], enc.hidden_dps[i]))

    groups.append(m[1])

    return [list(o.parameters()) for o in groups]


for p in model[0].parameters(): p.requires_grad_(False)


cbs = [partial(AvgStatsCallback,accuracy),

       CudaCallback, Recorder,

       partial(GradientClipping, clip=0.1),

       ProgressCallback]
model[0].module.load_state_dict(torch.load(path/'finetuned_enc.pth')) #load our finturned encoder 
learn = Learner(model, data, F.cross_entropy, opt_func=adam_opt(), cb_funcs=cbs, splitter=class_splitter)


lr = 1e-2

cbsched = sched_1cycle([lr], mom_start=0.8, mom_mid=0.7, mom_end=0.8)
learn.fit(1, cbs=cbsched)


for p in model[0].module.rnns[-1].parameters(): p.requires_grad_(True)


lr = 5e-3

cbsched = sched_1cycle([lr/2., lr/2., lr/2., lr], mom_start=0.8, mom_mid=0.7, mom_end=0.8)
learn.fit(1, cbs=cbsched)
for p in model[0].parameters(): p.requires_grad_(True)
lr = 1e-3

cbsched = sched_1cycle([lr/8., lr/4., lr/2., lr], mom_start=0.8, mom_mid=0.7, mom_end=0.8)
learn.fit(2, cbs=cbsched)


x,y = next(iter(data.valid_dl))
pred_batch = learn.model.eval()(x.cuda())
pred_ind = []

for inp in x:

    length = x.size(1) - (inp == 1).long().sum()

    inp = inp[:length]

    pred_ind.append(learn.model.eval()(inp[None].cuda()))
assert near(pred_batch, torch.cat(pred_ind))