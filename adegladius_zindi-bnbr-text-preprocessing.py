# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import operator 
import re
train = pd.read_csv("/kaggle/input/train-data/Train_BNBR.csv") #.drop('label', axis=1)
test = pd.read_csv("/kaggle/input/health-data/Test_health.csv")
length = len(train)
df = pd.concat([train ,test])

print("Number of texts: ", df.shape[0])
train_data = df[:length]
test_data = df[length:]
def load_embed(file):
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    
    if file == '/kaggle/input/glove6b/glove.6B.300d.txt':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))
        
    return embeddings_index
glove = '/kaggle/input/glove6b/glove.6B.300d.txt'
paragram =  '/kaggle/input/paragram-emb/paragram_300_sl999/paragram_300_sl999.txt'
wiki_news = '/kaggle/input/fastext-emb/crawl-300d-2M.vec'
#'/kaggle/input/wikinews/wiki-news-300d-1M1.vec'
print("Extracting GloVe embedding")
embed_glove = load_embed(glove)
print("Extracting Paragram embedding")
embed_paragram = load_embed(paragram)
print("Extracting FastText embedding")
embed_fasttext = load_embed(wiki_news)
def build_vocab(texts):
    sentences = texts.apply(lambda x: x.split()).values
    vocab = {}
    for sentence in sentences:
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab
def check_coverage(vocab, embeddings_index):
    known_words = {}
    unknown_words = {}
    nb_known_words = 0
    nb_unknown_words = 0
    for word in vocab.keys():
        try:
            known_words[word] = embeddings_index[word]
            nb_known_words += vocab[word]
        except:
            unknown_words[word] = vocab[word]
            nb_unknown_words += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))
    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words
vocab = build_vocab(df['text'])
print("Glove : ")
oov_glove = check_coverage(vocab, embed_glove)
print("Paragram : ")
oov_paragram = check_coverage(vocab, embed_paragram)
print("FastText : ")
oov_fasttext = check_coverage(vocab, embed_fasttext)
df['lowered_text'] = df['text'].apply(lambda x: x.lower())
vocab_low = build_vocab(df['lowered_text'])
print("Glove : ")
oov_glove = check_coverage(vocab_low, embed_glove)
print("Paragram : ")
oov_paragram = check_coverage(vocab_low, embed_paragram)
print("FastText : ")
oov_fasttext = check_coverage(vocab_low, embed_fasttext)
def add_lower(embedding, vocab):
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:  
            embedding[word.lower()] = embedding[word]
            count += 1
    print(f"Added {count} words to embedding")
print("Glove : ")
add_lower(embed_glove, vocab)
print("Paragram : ")
add_lower(embed_paragram, vocab)
print("FastText : ")
add_lower(embed_fasttext, vocab)
print("Glove : ")
oov_glove = check_coverage(vocab_low, embed_glove)
print("Paragram : ")
oov_paragram = check_coverage(vocab_low, embed_paragram)
print("FastText : ")
oov_fasttext = check_coverage(vocab_low, embed_fasttext)
oov_glove[:100]
contraction_mapping = {"don't":"do not","ain't": "is not", "aren't": "are not","can't": "cannot", "cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
def known_contractions(embed):
    known = []
    for contract in contraction_mapping:
        if contract in embed:
            known.append(contract)
    return known
print("- Known Contractions -")
print("   Glove :")
print(known_contractions(embed_glove))
print("   Paragram :")
print(known_contractions(embed_paragram))
print("   FastText :")
print(known_contractions(embed_fasttext))
def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text
df['treated_text'] = df['lowered_text'].apply(lambda x: clean_contractions(x, contraction_mapping))
vocab = build_vocab(df['treated_text'])
print("Glove : ")
oov_glove = check_coverage(vocab, embed_glove)
print("Paragram : ")
oov_paragram = check_coverage(vocab, embed_paragram)
print("FastText : ")
oov_fasttext = check_coverage(vocab, embed_fasttext)
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&â¦™'    
def unknown_punct(embed, punct):
    unknown = ''
    for p in punct:
        if p not in embed:
            unknown += p
            unknown += ' '
    return unknown
print("Glove :")
print(unknown_punct(embed_glove, punct))
print("Paragram :")
print(unknown_punct(embed_paragram, punct))
print("FastText :")
print(unknown_punct(embed_fasttext, punct))
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi'}
def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])
    
    for p in punct:
        text = text.replace(p, f' {p} ')
    
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text
df['treated_text'] = df['treated_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
vocab = build_vocab(df['treated_text'])
print("Glove : ")
oov_glove = check_coverage(vocab, embed_glove)
print("Paragram : ")
oov_paragram = check_coverage(vocab, embed_paragram)
print("FastText : ")
oov_fasttext = check_coverage(vocab, embed_fasttext)
oov_fasttext[:900]
mispell_dict = {'dieam': 'die',
                'stresseed':'stressed',
                'lowi': 'low i',
                'havei':'have i',
                'worthlesness':'worthlessness',
                'nightsi':'nights i',
                'downrecovering':'down recovering',
                'deferrin':'deferring',
                'helplessi':'helpless i',
                'foind':'found',
                'lowt':'low',
                'helplessstill':'helpless still',
                'birthnow':'birth now',
                'patners':'partners',
                'motivationsuicidal':'motivation suicidal',
                'includingmy':'including my',
                'hopelessnesss':'hopelessness',
                'desserted':'deserted',
                'dillusioned':'disllusioned',
                'issolated':'isolated',
                'undestands':'understands',
                'oldself':'old self',
                'ponographic':'pornographic',
                'unworthyness':'unworthiness',
                'selfworth':'self worth',
                'worldm':'world',
                'incidencesof':'incidences of',
                'weathernow': 'weather now',
                'benefitto':'benefit to',
                'frustratedi':'frustrated i',
                'worhtless':'worthless',
                'doto':'do to',
                'messnow':'mess now',
                'negativecurrently':'negative currently',
                'schoolfee':'school fee',
                'existd':'existed',
                'lonelycurrently':'lonely currently',
                'diserted':'deserted',
                'schoolfees':'school fees',
                'drinnking':'drinking',
                'alccohol':'alcohol',
                'addidcted':'addicted',
                'depressioni':'depression i',
                'childlish':'childish',
                'lonelyi':'lonely i',
                'stresssed':'stressed',
                'deteroriating':'deteriorating',
                'liferight':'life right',
                'thingof':'thing of',
                'whren':'when',
                'everythingi':'everything i',
                'insomia':'insomnia',
                'dizzines':'dizziness',
                'hopelessfor':'hopeless for',
                'confusednow':'confused now',
                'isolatednow':'isolated now',
                'lonelynow':'lonely now',
                'negletion':'neglection',
                'sucidal':'suicidal',
                'psycologically':'psychologically',
                'avoiod':'avoid',
                'cornerfeeling':'corner feeling',
                'hatrednow':'hatred now',
                'mediataton':'medication',
                'frorge':'forget',
                'drugsnow':'drugs now'}
def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x
df['treated_text'] = df['treated_text'].apply(lambda x: correct_spelling(x, mispell_dict))
vocab = build_vocab(df['treated_text'])
print("Glove : ")
oov_glove = check_coverage(vocab, embed_glove)
print("Paragram : ")
oov_paragram = check_coverage(vocab, embed_paragram)
print("FastText : ")
oov_fasttext = check_coverage(vocab, embed_fasttext)
df.shape
df.head()
train_data = df[:length]
test_data = df[length:]
train_data.head()
test_data.head()
train_data = train_data.drop(['text', 'lowered_text'], axis =1)
train_data.head()
test_data = test_data.drop(['label','text','lowered_text'], axis =1)
test_data.head()
train_data.to_csv("cleaned(925)_train.csv", index = False)
test_data.to_csv("cleaned_(925)test.csv", index =False)