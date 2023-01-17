import os, gc

from fastai.text import *

from tqdm import tqdm_notebook as tqdm

print(os.listdir("../input"))
# make training deterministic/reproducible

def seed_everything(seed=42):

    random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.backends.cudnn.deterministic = True

seed_everything()



def f1_score(y_pred, targets):

    epsilon = 1e-07



    y_pred = y_pred.argmax(dim=1)

#     targets = targets.argmax(dim=1)



    tp = (y_pred*targets).float().sum(dim=0)

    tn = ((1-targets)*(1-y_pred)).float().sum(dim=0)

    fp = ((1-targets)*y_pred).float().sum(dim=0)

    fn = (targets*(1-y_pred)).sum(dim=0)



    p = tp / (tp + fp + epsilon)

    r = tp / (tp + fn + epsilon)



    f1 = 2*p*r / (p+r+epsilon)

    f1 = torch.where(f1!=f1, torch.zeros_like(f1), f1)

    return f1.mean()
EMBED_SIZE = 100

MAX_FEATURES = 150000

MAX_LENGTH = 100

EMBEDDING_FILE = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
# train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

# train_df.head()

train_df = pd.read_csv('../input/train.csv')



insincere_df = train_df[train_df.target==1]

sincere_df = train_df[train_df.target==0]



sincere_df = sincere_df.iloc[np.random.permutation(len(sincere_df))]

sincere_df = sincere_df[:int(len(insincere_df)*2)]



del train_df



train_df = pd.concat([insincere_df, sincere_df])

train_df = train_df.iloc[np.random.permutation(len(train_df))]



del insincere_df

del sincere_df

gc.collect()
%%time

mispell_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", 

                "could've": "could have", "couldn't": "could not", "didn't": "did not", 

                "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", 

                "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", 

                "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  

                "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",

                "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", 

                "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", 

                "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", 

                "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", 

                "mayn't": "may not", "might've": "might have","mightn't": "might not",

                "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", 

                "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",

                "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", 

                "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", 

                "she'd": "she would", "she'd've": "she would have", "she'll": "she will", 

                "she'll've": "she will have", "she's": "she is", "should've": "should have", 

                "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have",

                "so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", 

                "that's": "that is", "there'd": "there would", "there'd've": "there would have", 

                "there's": "there is", "here's": "here is","they'd": "they would", 

                "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", 

                "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", 

                "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 

                "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", 

                "what'll've": "what will have", "what're": "what are",  "what's": "what is", 

                "what've": "what have", "when's": "when is", "when've": "when have", 

                "where'd": "where did", "where's": "where is", "where've": "where have", 

                "who'll": "who will", "who'll've": "who will have", "who's": "who is", 

                "who've": "who have", "why's": "why is", "why've": "why have", 

                "will've": "will have", "won't": "will not", "won't've": "will not have", 

                "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", 

                "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have",

                "y'all're": "you all are","y'all've": "you all have","you'd": "you would", 

                "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", 

                "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 

                'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 

                'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 

                'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 

                'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 

                'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 

                'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 

                'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 

                'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 

                'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', 

                '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 

                'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 

                'demonitisation': 'demonetization', 'demonitization': 'demonetization', 

                'demonetisation': 'demonetization', "n’t": "not", "n't": "not", "’ve": "have",

                "’re": "are", "’ll": "will", "howmuch": "how much", "i`m": "I am", "can`t": "can not",

                "dosen't": "does not", "what's​": "what is", "did't": "did not", "doesn`t": "dose not",

                "ya'll": "you alll", "it`s": "it is ", "does'nt": "does not", "what`s": "what is",

                "dosn't": "does not", "is'nt": "is not", "don'y": "do not you", "wan't": "will not",

                "that`s": "that is", "didn`t": "dod not", "hold'em": "holdaem", "din't": "did not",

                "isn't": "is not"}



def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re



mispellings, mispellings_re = _get_mispell(mispell_dict)

rx_xxxheight = re.compile('\d+[\'"]\d+(.\d+)?')

rx_xxxtime_or_score = re.compile('\d+:\d+')



def replace(match):

        return mispellings[match.group(0)]



def replace_typical_misspell(text):

    text = rx_xxxheight.sub('xxxheight', text)

    text = rx_xxxtime_or_score.sub('xxxtime_or_score', text)

    return mispellings_re.sub(replace, text)



# Clean speelings

train_df["question_text"] = train_df["question_text"].apply(lambda x: replace_typical_misspell(x))

test_df["question_text"] = test_df["question_text"].apply(lambda x: replace_typical_misspell(x))
%%time

# Load weights

def get_coefs(w, *x):

    return w,np.asarray(x, dtype='float32')





keywords = ['xxunk','xxpad','xxbos','xxfld','xxmaj','xxup','xxrep','xxwrep','xxxheight', 'xxxtime_or_score']



emb_mean = -0.0033469964

emb_std = 0.109855406

nb_r_words = len(keywords)



with open(EMBEDDING_FILE) as f:

    total, size = f.readline().split(' ')

    total = int(total)

    

    embeddings = torch.zeros((total+nb_r_words, EMBED_SIZE), dtype=torch.float32)

    embeddings.normal_(emb_mean, emb_std)

#     embeddings = np.random.normal(emb_mean, emb_std, (total+nb_r_words, EMBED_SIZE)).astype(np.float32)

    

    for i, line in enumerate(tqdm(f, total=total)):

        word, weight = get_coefs(*line.split(' '))

        embeddings[i+nb_r_words] = torch.from_numpy(weight[:EMBED_SIZE])

        keywords.append(word)
# emb_mean,emb_std = embeddings.mean(), embeddings.std()

# emb_mean,emb_std # (-0.00334699644103647, 0.10985540269880754)
vocab = Vocab(itos=keywords)
train_df = train_df.iloc[np.random.permutation(len(train_df))]

cut = int(0.1 * len(train_df)) + 1

train_df, valid_df = train_df[cut:], train_df[:cut]
%%time

data = TextDataBunch.from_df(path='.',

                             train_df=train_df, 

                             valid_df=valid_df,

                             test_df=test_df,

                             text_cols='question_text', 

                             label_cols='target',

                             max_vocab=MAX_FEATURES,

                            vocab=vocab)

print(len(data.vocab.itos))

data.save()

del train_df

del valid_df 

del test_df 

del data

gc.collect()
%%time

data = TextClasDataBunch.load(path='.', bs=64)

data.show_batch()
# # Load weights

# def get_coefs(w, *x):

#     return w,np.asarray(x, dtype='float16')



# embeddings_index = {}

# with open(EMBEDDING_FILE) as f:

#     total, size = f.readline().split(' ')

#     total = int(total)

    

#     for line in tqdm(f, total=total):

#         word, weight = get_coefs(*line.split(' '))

#         embeddings_index[word] = weight[:EMBED_SIZE]
# %%time

# # mean, std

# all_embs = np.stack(list(embeddings_index.values()))

# emb_mean,emb_std = all_embs.mean(), all_embs.std(dtype=np.float32)

# del all_embs

# gc.collect()
# # random weights

# vocab_size = len(data.vocab.itos)

# embedding_matrix = np.random.normal(emb_mean, emb_std, (vocab_size, EMBED_SIZE)).astype(np.float32)

# embedding_matrix.shape
# words_without_vec = 0

# # map pre-trained weights with our data

# for i, word in enumerate(tqdm(data.vocab.itos)):

#     if i >= vocab_size: continue

#     embedding_vector = embeddings_index.get(word)

#     if embedding_vector is not None: 

#         embedding_matrix[i] = embedding_vector

#     else:

#         words_without_vec = words_without_vec+1

#         print(i, word)



# embedding_matrix = torch.from_numpy(embedding_matrix) 

# del embeddings_index

# gc.collect()
# class QuoraInsincere(nn.Module):

#     def __init__(self, embedding_wights):

#         super(QuoraInsincere, self).__init__()

#         self.embeddings = nn.Embedding.from_pretrained(embedding_wights, freeze=False)

#         self.linear1 = nn.Linear(EMBED_SIZE*MAX_LENGTH, 1)

        

#     def forward(self, inputs):

#         x = self.embeddings(inputs)

#         x = x.view(x.size(0), -1)

#         x = self.linear1(x)

#         x = torch.sigmoid(x)        

#         return x

# model = QuoraInsincere(embeddings)
learner = text_classifier_learner(data, drop_mult=0.5, emb_sz=EMBED_SIZE, nl=1, nh=10, max_len=MAX_LENGTH)
learner.model
# load our new weights to extsing model

encoderModel =  next(learner.model.children())

# encoder = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)

encoderModel.encoder.load_state_dict({'weight': embeddings})
learner.unfreeze()
learner.lr_find()
learner.recorder.plot(skip_end=5)
learner.metrics.append(f1_score)
learner.fit_one_cycle(1, 7e-2, moms=(0.8,0.7))
learner.save('first')
learner.load('first');
learner.freeze_to(-2)

learner.fit_one_cycle(2, slice(1e-3,7e-2), moms=(0.8,0.7))
learner.save('second')
learner.load('second');
learner.unfreeze()

learner.fit_one_cycle(4, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
# learner.fit_one_cycle(3, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))
%time learner.predict("How much does a tutor earn in Bangalore?")
preds = learner.get_preds(ds_type=DatasetType.Test)
preds = preds[0].argmax(dim=1)

preds.sum()
test_df = pd.read_csv('../input/test.csv')
test_df.drop(['question_text'], axis=1, inplace=True)

test_df['prediction'] = preds.numpy()
test_df.to_csv("submission.csv", index=False)