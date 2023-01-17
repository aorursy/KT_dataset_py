import sys
!curl -s https://course.fast.ai/setup/colab | bash
!git clone https://github.com/yabhi0807/libml1.git /kaggle/tmp/fastai # This is my repo with all the fastai(updated) libraries 
sys.path.append('/kaggle/tmp/fastai')
!mkdir /kaggle/tmp/data/
!ln -s /kaggle/tmp/* /kaggle/working/
!mkdir /kaggle/tmp/data/arxiv/
!cp -r ../input/arxiv-dataset/* /kaggle/tmp/data/arxiv/
!ls /kaggle/tmp/data/arxiv/
%reload_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.model import fit
from fastai.dataset import *

import torchtext
from torchtext import vocab, data
from torchtext.datasets import language_modeling

from fastai.rnn_reg import *
from fastai.rnn_train import *
from fastai.nlp import *
from fastai.lm_rnn import *

import dill as pickle
import random
bs,bptt = 64,70
!pip install feedparser
import os, requests, time
# feedparser isn't a fastai dependency so you may need to install it.
import feedparser
import pandas as pd


class GetArXiv(object):
    def __init__(self, pickle_path, categories=list()):
        """
        :param pickle_path (str): path to pickle data file to save/load
        :param pickle_name (str): file name to save pickle to path
        :param categories (list): arXiv categories to query
        """
        if os.path.isdir(pickle_path):
            pickle_path = f"{pickle_path}{'' if pickle_path[-1] == '/' else '/'}all_arxiv.pkl"
        if len(categories) < 1:
            categories = ['cs*', 'cond-mat.dis-nn', 'q-bio.NC', 'stat.CO', 'stat.ML']
        # categories += ['cs.CV', 'cs.AI', 'cs.LG', 'cs.CL']

        self.categories = categories
        self.pickle_path = pickle_path
        self.base_url = 'http://export.arxiv.org/api/query'

    @staticmethod
    def build_qs(categories):
        """Build query string from categories"""
        return '+OR+'.join(['cat:'+c for c in categories])

    @staticmethod
    def get_entry_dict(entry):
        """Return a dictionary with the items we want from a feedparser entry"""
        try:
            return dict(title=entry['title'], authors=[a['name'] for a in entry['authors']],
                        published=pd.Timestamp(entry['published']), summary=entry['summary'],
                        link=entry['link'], category=entry['category'])
        except KeyError:
            print('Missing keys in row: {}'.format(entry))
            return None

    @staticmethod
    def strip_version(link):
        """Strip version number from arXiv paper link"""
        return link[:-2]

    def fetch_updated_data(self, max_retry=5, pg_offset=0, pg_size=1000, wait_time=15):
        """
        Get new papers from arXiv server
        :param max_retry: max number of time to retry request
        :param pg_offset: number of pages to offset
        :param pg_size: num abstracts to fetch per request
        :param wait_time: num seconds to wait between requests
        """
        i, retry = pg_offset, 0
        df = pd.DataFrame()
        past_links = []
        if os.path.isfile(self.pickle_path):
            df = pd.read_pickle(self.pickle_path)
            df.reset_index()
        if len(df) > 0: past_links = df.link.apply(self.strip_version)

        while True:
            params = dict(search_query=self.build_qs(self.categories),
                          sortBy='submittedDate', start=pg_size*i, max_results=pg_size)
            response = requests.get(self.base_url, params='&'.join([f'{k}={v}' for k, v in params.items()]))
            entries = feedparser.parse(response.text).entries
            if len(entries) < 1:
                if retry < max_retry:
                    retry += 1
                    time.sleep(wait_time)
                    continue
                break

            results_df = pd.DataFrame([self.get_entry_dict(e) for e in entries])
            max_date = results_df.published.max().date()
            new_links = ~results_df.link.apply(self.strip_version).isin(past_links)
            print(f'{i}. Fetched {len(results_df)} abstracts published {max_date} and earlier')
            if not new_links.any():
                break

            df = pd.concat((df, results_df.loc[new_links]), ignore_index=True)
            i += 1
            retry = 0
            time.sleep(wait_time)

        print(f'Downloaded {len(df)-len(past_links)} new abstracts')
        df.sort_values('published', ascending=False).groupby('link').first().reset_index()
        df.to_pickle(self.pickle_path)
        return df

    @classmethod
    def load(cls, pickle_path):
        """Load data from pickle and remove duplicates"""
        return pd.read_pickle(cls(pickle_path).pickle_path)

    @classmethod
    def update(cls, pickle_path, categories=list(), **kwargs):
        """
        Update arXiv data pickle with the latest abstracts
        """
        cls(pickle_path, categories).fetch_updated_data(**kwargs)
        return True
PATH='data/arxiv/'

ALL_ARXIV = f'{PATH}all_arxiv.pkl'

# all_arxiv.pkl: if arxiv hasn't been downloaded yet, it'll take some time to get it - go get some coffee
if not os.path.exists(ALL_ARXIV): GetArXiv.update(ALL_ARXIV)

# arxiv.csv: see dl1/nlp-arxiv.ipynb to get this one
df_mb = pd.read_csv(f'{PATH}arxiv.csv')
df_all = pd.read_pickle(ALL_ARXIV)
def get_txt(df):
    return '<CAT> ' + df.category.str.replace(r'[\.\-]','') + ' <SUMM> ' + df.summary + ' <TITLE> ' + df.title
df_mb['txt'] = get_txt(df_mb)
df_all['txt'] = get_txt(df_all)
n=len(df_all); n
os.makedirs(f'{PATH}trn/yes', exist_ok=True)
os.makedirs(f'{PATH}val/yes', exist_ok=True)
os.makedirs(f'{PATH}trn/no', exist_ok=True)
os.makedirs(f'{PATH}val/no', exist_ok=True)
os.makedirs(f'{PATH}all/trn', exist_ok=True)
os.makedirs(f'{PATH}all/val', exist_ok=True)
os.makedirs(f'{PATH}models', exist_ok=True)
for (i,(_,r)) in enumerate(df_all.iterrows()):
    dset = 'trn' if random.random()>0.1 else 'val'
    open(f'{PATH}all/{dset}/{i}.txt', 'w').write(r['txt'])
for (i,(_,r)) in enumerate(df_mb.iterrows()):
    lbl = 'yes' if r.tweeted else 'no'
    dset = 'trn' if random.random()>0.1 else 'val'
    open(f'{PATH}{dset}/{lbl}/{i}.txt', 'w').write(r['txt'])
!python -m spacy download en_core_web_md
!python -m spacy link en_core_web_md en_core_web_md
from spacy.symbols import ORTH

# install the 'en' model if the next line of code fails by running:
#python -m spacy download en              # default English model (~50MB)
#python -m spacy download en_core_web_md  # larger English model (~1GB)
# my_tok = en_core_web_md.load()
my_tok = spacy.load('en_core_web_md')
# or
# import en_core_web_md
# my_tok = en_core_web_md.load()

my_tok.tokenizer.add_special_case('<SUMM>', [{ORTH: '<SUMM>'}])
my_tok.tokenizer.add_special_case('<CAT>', [{ORTH: '<CAT>'}])
my_tok.tokenizer.add_special_case('<TITLE>', [{ORTH: '<TITLE>'}])
my_tok.tokenizer.add_special_case('<BR />', [{ORTH: '<BR />'}])
my_tok.tokenizer.add_special_case('<BR>', [{ORTH: '<BR>'}])

def my_spacy_tok(x): return [tok.text for tok in my_tok.tokenizer(x)]
TEXT = data.Field(lower=True, tokenize=my_spacy_tok)
FILES = dict(train='trn', validation='val', test='val')
md = LanguageModelData.from_text_files(f'{PATH}all/', TEXT, **FILES, bs=bs, bptt=bptt, min_freq=10)
pickle.dump(TEXT, open(f'{PATH}models/TEXT.pkl','wb'))
len(md.trn_dl), md.nt, len(md.trn_ds), len(md.trn_ds[0].text)
TEXT.vocab.itos[:12]
' '.join(md.trn_ds[0].text[:150])
em_sz = 200
nh = 500
nl = 3
opt_fn = partial(optim.Adam, betas=(0.7, 0.99))
learner = md.get_model(opt_fn, em_sz, nh, nl,
    dropout=0.05, dropouth=0.1, dropouti=0.05, dropoute=0.02, wdrop=0.2)
# dropout=0.4, dropouth=0.3, dropouti=0.65, dropoute=0.1, wdrop=0.5
#                dropouti=0.05, dropout=0.05, wdrop=0.1, dropoute=0.02, dropouth=0.05)
learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
learner.clip=0.3
learner.fit(3e-3, 1, wds=1e-6)
learner.fit(3e-3, 3, wds=1e-6, cycle_len=1, cycle_mult=2)
learner.save_encoder('adam2_enc')
learner.fit(3e-3, 10, wds=1e-6, cycle_len=5, cycle_save_name='adam3_10')
learner.save_encoder('adam3_10_enc')
learner.fit(3e-3, 8, wds=1e-6, cycle_len=10, cycle_save_name='adam3_5')
learner.fit(3e-3, 1, wds=1e-6, cycle_len=20, cycle_save_name='adam3_20')
learner.save_encoder('adam3_20_enc')
learner.save('adam3_20')
def proc_str(s): return TEXT.preprocess(TEXT.tokenize(s))
def num_str(s): return TEXT.numericalize([proc_str(s)])
m=learner.model
s="""<CAT> cscv <SUMM> algorithms that"""
def sample_model(m, s, l=50):
    t = num_str(s)
    m[0].bs=1
    m.eval()
    m.reset()
    res,*_ = m(t)
    print('...', end='')

    for i in range(l):
        n=res[-1].topk(2)[1]
        n = n[1] if n.data[0]==0 else n[0]
        word = TEXT.vocab.itos[n.data[0]]
        print(word, end=' ')
        if word=='<eos>': break
        res,*_ = m(n[0].unsqueeze(0))

    m[0].bs=bs
sample_model(m,"<CAT> csni <SUMM> algorithms that")
sample_model(m,"<CAT> cscv <SUMM> algorithms that")
sample_model(m,"<CAT> cscv <SUMM> algorithms. <TITLE> on ")
sample_model(m,"<CAT> csni <SUMM> algorithms. <TITLE> on ")
sample_model(m,"<CAT> cscv <SUMM> algorithms. <TITLE> towards ")
sample_model(m,"<CAT> csni <SUMM> algorithms. <TITLE> towards ")
TEXT = pickle.load(open(f'{PATH}models/TEXT.pkl','rb'))
class ArxivDataset(torchtext.data.Dataset):
    def __init__(self, path, text_field, label_field, **kwargs):
        fields = [('text', text_field), ('label', label_field)]
        examples = []
        for label in ['yes', 'no']:
            fnames = glob(os.path.join(path, label, '*.txt'));
            assert fnames, f"can't find 'yes.txt' or 'no.txt' under {path}/{label}"
            for fname in fnames:
                with open(fname, 'r') as f: text = f.readline()
                examples.append(data.Example.fromlist([text, label], fields))
        super().__init__(examples, fields, **kwargs)

    @staticmethod
    def sort_key(ex): return len(ex.text)
    
    @classmethod
    def splits(cls, text_field, label_field, root='.data',
               train='train', test='test', **kwargs):
        return super().splits(
            root, text_field=text_field, label_field=label_field,
            train=train, validation=None, test=test, **kwargs)
ARX_LABEL = data.Field(sequential=False)
splits = ArxivDataset.splits(TEXT, ARX_LABEL, PATH, train='trn', test='val')
md2 = TextData.from_splits(PATH, splits, bs)
#            dropout=0.3, dropouti=0.4, wdrop=0.3, dropoute=0.05, dropouth=0.2)
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def prec_at_6(preds,targs):
    precision, recall, _ = precision_recall_curve(targs==2, preds[:,2])
    print(recall[precision>=0.6][0])
    return recall[precision>=0.6][0]
# dropout=0.4, dropouth=0.3, dropouti=0.65, dropoute=0.1, wdrop=0.5
m3 = md2.get_model(opt_fn, 1500, bptt, emb_sz=em_sz, n_hid=nh, n_layers=nl, 
           dropout=0.1, dropouti=0.65, wdrop=0.5, dropoute=0.1, dropouth=0.3)
m3.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
m3.clip=25.
# this notebook has a mess of some things going under 'all/' others not, so a little hack here
!ln -sf ../all/models/adam3_20_enc.h5 {PATH}models/adam3_20_enc.h5
m3.load_encoder(f'adam3_20_enc')
lrs=np.array([1e-4,1e-3,1e-3,1e-2,3e-2])
m3.freeze_to(-1)
m3.fit(lrs/2, 1, metrics=[accuracy])
m3.unfreeze()
m3.fit(lrs, 1, metrics=[accuracy], cycle_len=1)
m3.fit(lrs, 2, metrics=[accuracy], cycle_len=4, cycle_save_name='imdb2')
prec_at_6(*m3.predict_with_targs())
m3.fit(lrs, 4, metrics=[accuracy], cycle_len=2, cycle_save_name='imdb2')
prec_at_6(*m3.predict_with_targs())
