import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os, re, gc, random



import warnings

warnings.filterwarnings('ignore') # to suppress some matplotlib deprecation warnings
from nltk.tokenize import TweetTokenizer

from gensim.models import KeyedVectors



from sklearn.metrics import f1_score

from sklearn.model_selection import StratifiedKFold, train_test_split
import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F



import torch

import torchtext

from torchtext import data, vocab

from torchtext.data import Dataset
from tqdm import tqdm_notebook

torchtext.vocab.tqdm = tqdm_notebook # Replace tqdm to tqdm_notebook in module torchtext
path = "../input"

emb_path = "../input/embeddings"

n_folds = 5

bs = 512

device = 'cuda'
seed = 7777

random.seed(seed)

os.environ['PYTHONHASHSEED'] = str(seed)

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
mispell_dict = {   

    "can't" : "can not", "tryin'":"trying",

    "'m": " am", "'ll": " 'll", "'d" : " 'd'", "..": "  ",".": " . ", ",":" , ",

    "'ve" : " have", "n't": " not","'s": " 's", "'re": " are", "$": " $","’": " ' ",

    "y'all": "you all", 'metoo': 'me too',

    'colour': 'color', 'centre': 'center', 'favourite': 'favorite',

    'travelling': 'traveling', 'counselling': 'counseling',

    'centerd': 'centered',

    'theatre': 'theater','cancelled': 'canceled','labour': 'labor',

    'organisation': 'organization','wwii': 'world war 2', 'citicise': 'criticize',

    'youtu ': 'youtube ','Qoura': 'Quora','sallary': 'salary','Whta': 'What',

    'narcisist': 'narcissist','howdo': 'how do','whatare': 'what are',

    'howcan': 'how can','howmuch': 'how much','howmany': 'how many', 'whydo': 'why do',

    'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation',

    'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'doI': 'do I',

    'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 

    '2k17': '2017', '2k18': '2018','qouta': 'quota', 'exboyfriend': 'ex boyfriend',

    'airhostess': 'air hostess', "whst": 'what', 'watsapp':'whatsapp',

    'demonitisation': 'demonetization', 'demonitization': 'demonetization',

    'demonetisation': 'demonetization','bigdata': 'big data',

    'Quorans': 'Questions','quorans': 'questions','quoran':'question','Quoran':'Question',

    # Slang and  abbreviation

    'Skripal':'russian spy','Doklam':'Tibet', 

    'BNBR':'Be Nice Be Respectful', 'Brexit': 'British exit',

    'Bhakts':'fascists','bhakts':'fascists','Bhakt':'fascist','bhakt':'fascist',

    'SJWs':'Social Justice Warrior','SJW':'Social Justice Warrior',

    'Modiji':'Prime Minister of India', 'Ra apist': 'Rapist', ' apist ':' ape ',

    'wumao':'commenters','cucks': 'cuck', 'Strzok':'stupid phrase','strzok':'stupid phrase',

    

    ' s.p ': ' ', ' S.P ': ' ', 'U.s.p': '', 'U.S.A.': 'USA', 'u.s.a.': 'USA', 'U.S.A': 'USA',

    'u.s.a': 'USA', 'U.S.': 'USA', 'u.s.': 'USA', ' U.S ': ' USA ', ' u.s ': ' USA ', 'U.s.': 'USA',

    ' U.s ': 'USA', ' u.S ': ' USA ', ' fu.k': ' fuck', 'U.K.': 'UK', ' u.k ': ' UK ',

    ' don t ': ' do not ', 'bacteries': 'batteries', ' yr old ': ' years old ', 'Ph.D': 'PhD',

    'cau.sing': 'causing', 'Kim Jong-Un': 'The president of North Korea', 'savegely': 'savagely',

    '2fifth': 'twenty fifth', '2third': 'twenty third',

    '2nineth': 'twenty nineth', '2fourth': 'twenty fourth', '#metoo': 'MeToo',

    'Trumpcare': 'Trump health care system', '4fifth': 'forty fifth', 'Remainers': 'remainder',

    'Terroristan': 'terrorist', 'antibrahmin': 'anti brahmin','culturr': 'culture',

    'fuckboys': 'fuckboy', 'Fuckboys': 'fuckboy', 'Fuckboy': 'fuckboy', 'fuckgirls': 'fuck girls',

    'fuckgirl': 'fuck girl', 'Trumpsters': 'Trump supporters', '4sixth': 'forty sixth',

    'weatern': 'western', '4fourth': 'forty fourth', 'emiratis': 'emirates', 'trumpers': 'Trumpster',

    'indans': 'indians', 'mastuburate': 'masturbate', ' f**k': ' fuck', ' F**k': ' fuck', ' F**K': ' fuck',

    ' u r ': ' you are ', ' u ': ' you ', '操你妈 ': 'fuck your mother', ' e.g.': ' for example',

    'i.e.': 'in other words', '...': '.', 'et.al': 'elsewhere', 'anti-Semitic': 'anti-semitic',

    ' f***': ' fuck', ' f**': ' fuc', ' F***': ' fuck', ' F**': ' fuck',

    ' a****': ' assho', 'a**': 'ass', ' h***': ' hole', 'A****': 'assho', ' A**': ' ass', ' H***': ' hole',

    ' s***': ' shit', ' s**': 'shi', ' S***': ' shit', ' S**': ' shi', ' Sh**': 'shit',

    ' p****': ' pussy', ' p*ssy': ' pussy', ' P****': ' pussy',

    ' p***': ' porn', ' p*rn': ' porn', ' P***': ' porn',' Fck': ' Fuck',' fck': ' fuck',  

    ' st*up*id': ' stupid', ' d***': 'dick', ' di**': ' dick', ' h*ck': ' hack',

    ' b*tch': ' bitch', 'bi*ch': ' bitch', ' bit*h': ' bitch', ' bitc*': ' bitch', ' b****': ' bitch',

    ' b***': ' bitc', ' b**': ' bit', ' b*ll': ' bull',' FATF': 'Western summit conference',

    'Terroristan': 'terrorist Pakistan', 'terroristan': 'terrorist Pakistan',

    ' incel': ' involuntary celibates', ' incels': ' involuntary celibates', 'emiratis': 'Emiratis',

    'weatern': 'western', 'westernise': 'westernize', 'Pizzagate': 'debunked conspiracy theory', 'naïve': 'naive',

    ' HYPSM': ' Harvard, Yale, Princeton, Stanford, MIT', ' HYPS': ' Harvard, Yale, Princeton, Stanford',

    'kompromat': 'compromising material', ' Tharki': ' pervert', ' tharki': 'pervert',

    'Naxali ': 'Naxalite ', 'Naxalities': 'Naxalites','Mewani': 'Indian politician Jignesh Mevani', ' Wjy': ' Why',

    'Fadnavis': 'Indian politician Devendra Fadnavis', 'Awadesh': 'Indian engineer Awdhesh Singh',

    'Awdhesh': 'Indian engineer Awdhesh Singh', 'Khalistanis': 'Sikh separatist movement',

    'madheshi': 'Madheshi','Stupead': 'stupid',  'narcissit': 'narcissist',

}



def clean_latex(text):

    """

    convert r"[math]\vec{x} + \vec{y}" to English

    """

    # edge case

    text = re.sub(r'\[math\]', ' LaTex math ', text)

    text = re.sub(r'\[\/math\]', ' LaTex math ', text)

#     text = re.sub(r'\\', ' LaTex ', text)



    pattern_to_sub = {

        r'\\mathrm': ' LaTex math mode ',

        r'\\mathbb': ' LaTex math mode ',

        r'\\boxed': ' LaTex equation ',

        r'\\begin': ' LaTex equation ',

        r'\\end': ' LaTex equation ',

        r'\\left': ' LaTex equation ',

        r'\\right': ' LaTex equation ',

        r'\\(over|under)brace': ' LaTex equation ',

        r'\\text': ' LaTex equation ',

        r'\\vec': ' vector ',

        r'\\var': ' variable ',

        r'\\theta': ' theta ',

        r'\\mu': ' average ',

        r'\\min': ' minimum ',

        r'\\max': ' maximum ',

        r'\\sum': ' + ',

        r'\\times': ' * ',

        r'\\cdot': ' * ',

        r'\\hat': ' ^ ',

        r'\\frac': ' / ',

        r'\\div': ' / ',

        r'\\sin': ' Sine ',

        r'\\cos': ' Cosine ',

        r'\\tan': ' Tangent ',

        r'\\infty': ' infinity ',

        r'\\int': ' integer ',

        r'\\in': ' in ',

    }

    # post process for look up

    pattern_dict = {k.strip('\\'): v for k, v in pattern_to_sub.items()}

    # init re

    patterns = pattern_to_sub.keys()

    pattern_re = re.compile('(%s)' % '|'.join(patterns))



    def _replace(match):

        """

        reference: https://www.kaggle.com/hengzheng/attention-capsule-why-not-both-lb-0-694 # noqa

        """

        try:

            word = pattern_dict.get(match.group(0).strip('\\'))

        except KeyError:

            word = match.group(0)

            print('!!Error: Could Not Find Key: {}'.format(word))

        return word

    return pattern_re.sub(_replace, text)



def correct_spelling(s, dic):

    for key, corr in dic.items():

        s = s.replace(key, dic[key])

    return s



def tweet_clean(text):

    text = re.sub(r'[^A-Za-z0-9!.,?$\'\"]+', ' ', text) # remove non alphanumeric character

#     text = re.sub(r'https?:/\/\S+', ' ', text) # remove links

    return text #.lower()



def tokenizer(s): 

    s = clean_latex(s)

    s = correct_spelling(s, mispell_dict)

    s = tweet_clean(s)

    return tknzr.tokenize(s)
def find_threshold(y_t, y_p, floor=-1., ceil=1., steps=41):

    thresholds = np.linspace(floor, ceil, steps)

    best_val = 0.0

    for threshold in thresholds:

        val_predict = (y_p > threshold)

        score = f1_score(y_t, val_predict)

        if score > best_val:

            best_threshold = threshold

            best_val = score

    return best_threshold
def splits_cv(data, cv, y=None):

    

    for indices in cv.split(range(len(data)), y):

        (train_data, val_data) = tuple([data.examples[i] for i in index] for index in indices)

        yield tuple(Dataset(d, data.fields) for d in (train_data, val_data) if d)
skf = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = seed)



scores = pd.read_csv('../input/train.csv')

target = scores.target.values

scores = scores.set_index('qid')

scores.drop(columns=['question_text'], inplace=True)



subm = pd.read_csv('../input/test.csv')

subm = subm.set_index('qid')

subm.drop(columns='question_text', inplace=True)
# define the columns that we want to process and how to process

txt_field = data.Field(sequential=True, tokenize=tokenizer, include_lengths=False,  use_vocab=True)

label_field = data.Field(sequential=False, use_vocab=False, is_target=True)

qid_field = data.RawField()



train_fields = [

    ('qid', qid_field), # we dont need this, so no processing

    ('question_text', txt_field), # process it as text

    ('target', label_field) # process it as label

]

test_fields = [

    ('qid', qid_field), 

    ('question_text', txt_field), 

]
# Loading csv file

train_ds = data.TabularDataset(path=os.path.join(path, 'train.csv'), 

                           format='csv',

                           fields=train_fields, 

                           skip_header=True)



test_ds = data.TabularDataset(path=os.path.join(path, 'test.csv'), 

                           format='csv',

                           fields=test_fields, 

                           skip_header=True)
test_ds.fields['qid'].is_target = False

train_ds.fields['qid'].is_target = False
test_loader = data.BucketIterator(test_ds, batch_size=bs, device='cuda',

                                sort_key=lambda x: len(x.question_text),

                                sort_within_batch=True, 

                                shuffle=False, repeat=False)
class RecNN(nn.Module):

    def __init__(self, embs_vocab, hidden_size, layers=1, dropout=0., bidirectional=False):

        super().__init__()



        self.hidden_size = hidden_size

        self.bidirectional = bidirectional

        self.num_layers = layers

        self.emb = nn.Embedding(embs_vocab.size(0), embs_vocab.size(1))

        self.emb.weight.data.copy_(embs_vocab) # load pretrained vectors

        self.emb.weight.requires_grad = False # make embedding non trainable

        

        self.line = nn.Conv1d(300, 300, 1, bias=True) # Linear transformation of embedding vectors

        

        self.lstm = nn.LSTM(embs_vocab.size(1), self.hidden_size,

                            num_layers=layers, bidirectional=bidirectional, dropout=dropout)

        

        self.gru = nn.GRU(embs_vocab.size(1), self.hidden_size,

                            num_layers=layers, bidirectional=bidirectional, dropout=dropout)

        

        self.out = nn.Linear(self.hidden_size*(bidirectional + 1), 32)

        self.last = nn.Linear(32, 1)

                

    def forward(self, x):

        

        embs = self.emb(x)

        lstm, (h, c) = self.lstm(embs)

        

        x = F.relu(self.line(embs.permute(1,2,0)), inplace=True).permute(2,0,1)

        gru, h = self.gru(x, h)

        lstm = lstm + gru

        

        lstm, _ = lstm.max(dim=0, keepdim=False) 

        out = self.out(lstm)

        out = self.last(F.relu(out)).squeeze()

        return out
def OOF_preds(test_df, target, embs_vocab, epochs = 4, alias='prediction', cv=skf,

              loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=(torch.Tensor([2.7])).to(device)),

              bs = 512, embedding_dim = 300, bidirectional=True, n_hidden = 64):



    print('Embedding vocab size: ', embs_vocab.size()[0])

    

    test_df[alias] = 0.

    

    for train, _ in splits_cv(train_ds, cv, target):

        

        train = data.BucketIterator(train, batch_size=bs, device=device,

                        sort_key=lambda x: len(x.question_text),

                        sort_within_batch=True,

                        shuffle=True, repeat=False)



        model = RecNN(embs_vocab, n_hidden, dropout=0., bidirectional=bidirectional).to(device)

        

        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 1e-3,

                         betas=(0.75, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

        

        print('\n')

        for epoch in range(epochs):      

            y_true_train = np.empty(0)

            y_pred_train = np.empty(0)

            total_loss_train = 0          

            model.train()

            for (_, x), y in train:

                y = y.type(dtype=torch.cuda.FloatTensor)

                opt.zero_grad()

                pred = model(x)

                loss = loss_fn(pred, y)

                loss.backward()

                opt.step()



                y_true_train = np.concatenate([y_true_train, y.cpu().data.numpy()], axis = 0)

                y_pred_train = np.concatenate([y_pred_train, pred.cpu().squeeze().data.numpy()], axis = 0)

                total_loss_train += loss.item()



            tacc = f1_score(y_true_train, y_pred_train>0)

            tloss = total_loss_train/len(train)

            print(f'Epoch {epoch+1}: Train loss: {tloss:.4f}, F1: {tacc:.4f}')

        

        # Get prediction for test set

        preds = torch.empty(0)

        qids = []

        for (y, x), _ in test_loader:

            pred = model(x)

            qids.append(y)

            preds = torch.cat([preds, pred.detach().cpu()])

            

        # Save prediction of test set

        preds = torch.sigmoid(preds).numpy()

        qids = [item for sublist in qids for item in sublist]

        test_df.at[qids, alias]  =  test_df.loc[qids][alias].values + preds/n_folds

        

        gc.enable();

        del train

        gc.collect();

        

    gc.enable();

    del embs_vocab, model

    gc.collect();     

    return test_df
def preload_gnews():

    # Google..... bin file.......

    vector_google = KeyedVectors.load_word2vec_format(os.path.join(emb_path, embs_file['gnews']), binary=True)



    stoi = {s:idx for idx, s in enumerate(vector_google.index2word)}

    itos = {idx:s for idx, s in enumerate(vector_google.index2word)}



    cache='cache/'

    path_cache = os.path.join(cache, 'GoogleNews-vectors-negative300.bin')

    file_suffix = '.pt'

    path_pt = path_cache + file_suffix



    torch.save((itos, stoi, torch.from_numpy(vector_google.vectors), vector_google.vectors.shape[1]), path_pt)



    

embs_file = {}

embs_file['wiki'] = 'wiki-news-300d-1M/wiki-news-300d-1M.vec'

embs_file['gnews'] = 'GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'

embs_file['glove'] = 'glove.840B.300d/glove.840B.300d.txt'

embs_file['gram'] = 'paragram_300_sl999/paragram_300_sl999.txt'



embs_vocab = {}



!mkdir cache

preload_gnews()

# specify the path to the localy saved vectors

vec = vocab.Vectors(os.path.join(emb_path, embs_file['gnews']), cache='cache/')

# build the vocabulary using train and validation dataset and assign the vectors

txt_field.build_vocab(train_ds, test_ds, max_size=350000, vectors=vec)

embs_vocab['gnews'] = train_ds.fields['question_text'].vocab.vectors

!rm -r cache



# specify the path to the localy saved vectors

vec = vocab.Vectors(os.path.join(emb_path, embs_file['wiki']), cache='cache/')

# build the vocabulary using train and validation dataset and assign the vectors

txt_field.build_vocab(train_ds, test_ds, max_size=350000, vectors=vec)

embs_vocab['wiki'] = train_ds.fields['question_text'].vocab.vectors



# specify the path to the localy saved vectors

vec = vocab.Vectors(os.path.join(emb_path, embs_file['glove']), cache='cache/')

# build the vocabulary using train and validation dataset and assign the vectors

txt_field.build_vocab(train_ds, test_ds, max_size=350000, vectors=vec)

embs_vocab['glove'] = train_ds.fields['question_text'].vocab.vectors



print('Embedding loaded, vocab size: ', embs_vocab['glove'].size()[0])

!rm -r cache

gc.enable()

del vec

gc.collect(); 
def fill_unknown(vector):

    # fill from Glove

    data = torch.zeros_like(vector)

    data.copy_(vector)

    idx = torch.nonzero(data.sum(dim=1) == 0)

    data[idx] = embs_vocab['glove'][idx]

    # fill from Wiki

    idx = torch.nonzero(data.sum(dim=1) == 0)

    data[idx] = embs_vocab['wiki'][idx]

    # fill from GoogleNews

    idx = torch.nonzero(data.sum(dim=1) == 0)

    data[idx] = embs_vocab['gnews'][idx]

    return data
%%time

subm = OOF_preds(subm, target, epochs = 5, alias='wiki',

#               loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean'), 

              embs_vocab=fill_unknown(embs_vocab['wiki']),

              cv = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = seed),

              embedding_dim = 300, bidirectional=True, n_hidden = 64)
%%time

subm = OOF_preds(subm, target, epochs = 5, alias='glove',

              embs_vocab=fill_unknown(embs_vocab['glove']),

              cv = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = seed+15),

              bs = 512, embedding_dim = 300, bidirectional=True, n_hidden = 64)
%%time

subm = OOF_preds(subm, target, epochs = 5, alias='gnews',

              embs_vocab=fill_unknown(embs_vocab['gnews']),

              cv = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = seed+25),

              bs = 512, embedding_dim = 300, bidirectional=True, n_hidden = 64)
subm.corr()
submission = np.mean(subm.values, axis = 1)
subm['prediction'] = submission > 0.55

subm.prediction = subm.prediction.astype('int')

subm.to_csv('submission.csv', columns=['prediction'])