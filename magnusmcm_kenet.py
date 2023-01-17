from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import he_normal, he_uniform, glorot_normal, glorot_uniform
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model
from keras.layers import Dense, Embedding, Bidirectional, CuDNNGRU, GlobalAveragePooling1D
from keras.layers import CuDNNLSTM, GlobalMaxPooling1D, concatenate, Input, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

from tqdm import tqdm
from statistics import mean, median, stdev
from numpy import amax, amin

import gc
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import operator 
import re
import os
import math
batch_size = 1024
epochs = 18
current_embd = "Glove"
question_length =  100
max_eval_size = 15000
#max_features = 50000

# Cesc: Seed utilitzada pel K-Fold
DATA_SPLIT_SEED = 2018
K_FOLDS = 5
K_FOLD_EPOCHS = int(epochs/K_FOLDS)

seed_nb=14
np.random.seed(seed_nb)
tf.set_random_seed(seed_nb)
paraules_prohibides = ['2g1c', '2 girls 1 cup', 'acrotomophilia', 'alabama hot pocket', 'alaskan pipeline', 'anal', 'anilingus', 'anus',
                       'apeshit', 'arsehole', 'ass', 'asshole', 'assmunch', 'auto erotic', 'autoerotic', 'babeland', 'baby batter',
                       'baby juice', 'ball gag', 'ball gravy', 'ball kicking', 'ball licking', 'ball sack', 'ball sucking', 'bangbros',
                       'bareback', 'barely legal', 'barenaked', 'bastard', 'bastardo', 'bastinado', 'bbw', 'bdsm', 'beaner', 'beaners',
                       'beaver cleaver', 'beaver lips', 'bestiality', 'big black', 'big breasts', 'big knockers', 'big tits', 'bimbos',
                       'birdlock', 'bitch', 'bitches', 'black cock', 'blonde action', 'blonde on blonde action', 'blowjob', 'blow job',
                       'blow your load', 'blue waffle', 'blumpkin', 'bollocks', 'bondage', 'boner', 'boob', 'boobs', 'booty call',
                       'brown showers', 'brunette action', 'bukkake', 'bulldyke', 'bullet vibe', 'bullshit', 'bung hole', 'bunghole',
                       'busty', 'butt', 'buttcheeks', 'butthole', 'camel toe', 'camgirl', 'camslut', 'camwhore', 'carpet muncher',
                       'carpetmuncher', 'chocolate rosebuds', 'circlejerk', 'cleveland steamer', 'clit', 'clitoris', 'clover clamps',
                       'clusterfuck', 'cock', 'cocks', 'coprolagnia', 'coprophilia', 'cornhole', 'coon', 'coons', 'creampie', 'cum',
                       'cumming', 'cunnilingus', 'cunt', 'darkie', 'date rape', 'daterape', 'deep throat', 'deepthroat', 'dendrophilia',
                       'dick', 'dildo', 'dingleberry', 'dingleberries', 'dirty pillows', 'dirty sanchez', 'doggie style', 'doggiestyle',
                       'doggy style', 'doggystyle', 'dog style', 'dolcett', 'domination', 'dominatrix', 'dommes', 'donkey punch',
                       'double dong', 'double penetration', 'dp action', 'dry hump', 'dvda', 'eat my ass', 'ecchi', 'ejaculation',
                       'erotic', 'erotism', 'escort', 'eunuch', 'faggot', 'fecal', 'felch', 'fellatio', 'feltch', 'female squirting',
                       'femdom', 'figging', 'fingerbang', 'fingering', 'fisting', 'foot fetish', 'footjob', 'frotting', 'fuck',
                       'fuck buttons', 'fuckin', 'fucking', 'fucktards', 'fudge packer', 'fudgepacker', 'futanari', 'gang bang',
                       'gay sex', 'genitals', 'giant cock', 'girl on', 'girl on top', 'girls gone wild', 'goatcx', 'goatse', 'god damn',
                       'gokkun', 'golden shower', 'goodpoop', 'goo girl', 'goregasm', 'grope', 'group sex', 'g-spot', 'guro', 'hand job',
                       'handjob', 'hard core', 'hardcore', 'hentai', 'homoerotic', 'honkey', 'hooker', 'hot carl', 'hot chick', 'how to kill',
                       'how to murder', 'huge fat', 'humping', 'incest', 'intercourse', 'jack off', 'jail bait', 'jailbait', 'jelly donut',
                       'jerk off', 'jigaboo', 'jiggaboo', 'jiggerboo', 'jizz', 'juggs', 'kike', 'kinbaku', 'kinkster', 'kinky', 'knobbing',
                       'leather restraint', 'leather straight jacket', 'lemon party', 'lolita', 'lovemaking', 'make me come', 'male squirting',
                       'masturbate', 'menage a trois', 'milf', 'missionary position', 'motherfucker', 'mound of venus', 'mr hands', 'muff diver',
                       'muffdiving', 'nambla', 'nawashi', 'negro', 'neonazi', 'nigga', 'nigger', 'nig nog', 'nimphomania', 'nipple', 'nipples',
                       'nsfw images', 'nude', 'nudity', 'nympho', 'nymphomania', 'octopussy', 'omorashi', 'one cup two girls', 'one guy one jar',
                       'orgasm', 'orgy', 'paedophile', 'paki', 'panties', 'panty', 'pedobear', 'pedophile', 'pegging', 'penis', 'phone sex',
                       'piece of shit', 'pissing', 'piss pig', 'pisspig', 'playboy', 'pleasure chest', 'pole smoker', 'ponyplay', 'poof',
                       'poon', 'poontang', 'punany', 'poop chute', 'poopchute', 'porn', 'porno', 'pornography', 'prince albert piercing',
                       'pthc', 'pubes', 'pussy', 'queaf', 'queef', 'quim', 'raghead', 'raging boner', 'rape', 'raping', 'rapist', 'rectum',
                       'reverse cowgirl', 'rimjob', 'rimming', 'rosy palm', 'rosy palm and her 5 sisters', 'rusty trombone', 'sadism',
                       'santorum', 'scat', 'schlong', 'scissoring', 'semen', 'sexo', 'sexy', 'shaved beaver', 'shaved pussy',
                       'shemale', 'shibari', 'shit', 'shitblimp', 'shitty', 'shota', 'shrimping', 'skeet', 'slanteye', 'slut', 's&m',
                       'smut', 'snatch', 'snowballing', 'sodomize', 'sodomy', 'spic', 'splooge', 'splooge moose', 'spooge', 'spread legs',
                       'spunk', 'strap on', 'strapon', 'strappado', 'strip club', 'style doggy', 'suck', 'sucks', 'suicide girls', 'sultry women',
                       'swastika', 'swinger', 'tainted love', 'taste my', 'tea bagging', 'threesome', 'throating', 'tied up', 'tight white',
                       'tit', 'tits', 'titties', 'titty', 'tongue in a', 'topless', 'tosser', 'towelhead', 'tranny', 'tribadism', 'tub girl',
                       'tubgirl', 'tushy', 'twat', 'twink', 'twinkie', 'two girls one cup', 'undressing', 'upskirt', 'urethra play', 'urophilia',
                       'vagina', 'venus mound', 'vibrator', 'violet wand', 'vorarephilia', 'voyeur', 'vulva', 'wank', 'wetback', 'wet dream',
                       'white power', 'wrapping men', 'wrinkled starfish', 'xx', 'xxx', 'yaoi', 'yellow showers', 'yiffy', 'zoophilia', '🖕']

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
test['target']=-1
df = pd.concat([train ,test])
del train, test; gc.collect(); time.sleep(5)
def load_embed(file):
    def get_coefs(word,*arr): 
        return word, np.asarray(arr, dtype='float32')
    
    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in tqdm(open(file)) if len(o)>100)
    else:
        embeddings_index = dict(get_coefs(*o.split(" ")) for o in tqdm(open(file, encoding='latin')))
        
    return embeddings_index
embed_glove = load_embed('../input/embeddings/glove.840B.300d/glove.840B.300d.txt')
embed_paragram = load_embed('../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt')
#embed_fasttext = load_embed('../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec')
my_embedding_matrix = dict()

for k1,v1 in embed_glove.items():
    my_val = v1
    if k1 in embed_paragram.keys():
            my_val = (v1 + embed_paragram[k1])/2
    my_embedding_matrix[k1] = my_val

for k1,v1 in embed_paragram.items():
    if k1 not in embed_glove.keys():
        my_embedding_matrix[k1] = v1
print(len(my_embedding_matrix))
print(len(embed_glove))
print(len(embed_paragram))
print(my_embedding_matrix["dog"][0])
print(embed_glove["dog"][0])
print(embed_paragram["dog"][0])
#if current_embd == "Glove":
#    glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
#    print("Extracting GloVe embedding")
#    embed_glove = load_embed(glove)
#elif current_embd == "Paragram":
#    paragram =  '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
#    print("Extracting Paragram embedding")
#    embed_paragram = load_embed(paragram)
#elif current_embd == "FastText":
#    wiki_news = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
#    print("Extracting FastText embedding")
#    embed_fasttext = load_embed(wiki_news)
df['size'] = df['question_text'].str.len()
print(mean(df['size']))
print(median(df['size']))
print(stdev(df['size']))
print(amax(df['size']))
print(amin(df['size']))
df = df.drop(['size'], axis=1)
# Funció per crear el Vocabulari
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
vocab = build_vocab(df['question_text'])
print(vocab["Apple"])
# Convertir en minuscules les paraules
df['question_text'] = df['question_text'].apply(lambda x: x.lower())
# Afegim les paraules amb minuscules als altres diccionaris
def add_lower(embedding, vocab):
    count = 0
    for word in vocab:
        if word in embedding and word.lower() not in embedding:  
            embedding[word.lower()] = embedding[word]
            count += 1
#for w in paraules_prohibides:
#    print(w)

#for q in df['question_text'].values:
#    for w in paraules_prohibides:
#        if w in q:
#            df['conte_paraula_prohibida']=1
for index, row in df.iterrows():
    for w in paraules_prohibides:
        if w in row['question_text']:
            row['conte_paraula_prohibida']=1
            break;
if "sexo" in paraules_prohibides:
    print("OK")
if "casa" in paraules_prohibides:
    print("OK2")    
print(list(df))
for index, row in df.iterrows():
    if row['conte_paraula_prohibida']==1 & index < 20:
        print (row['question_text'])
#print(my_embedding_matrix)

#if current_embd == "Glove":
#    add_lower(embed_glove, vocab)
#elif current_embd == "Paragram":
#    add_lower(embed_paragram, vocab)
#elif current_embd == "FastText":
#    add_lower(embed_fasttext, vocab)

#add_lower(embed_glove, vocab)
#add_lower(embed_paragram, vocab)
add_lower(my_embedding_matrix, vocab)
print(len(my_embedding_matrix))
print(len(vocab))
print(type(my_embedding_matrix))
print(type(vocab))
my_embedding_matrix = {k: v for k, v in my_embedding_matrix.items() if k in vocab}
print(my_embedding_matrix["dog"][0])
del embed_paragram, embed_glove, vocab; gc.collect(); time.sleep(5)
# Contraccions
# Cesc: He afegit canvis en monedes

#contraction_mapping = {"euros" : "eur", "dollars": "usd", "ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

#Ho desfem perquè sembla que no xuta
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
# Netejar contraccions
def clean_contractions(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text
df['question_text'] = df['question_text'].apply(lambda x: clean_contractions(x, contraction_mapping))
# Caracters especials de puntuació
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

# Cesc: Afegim uns quants caracters mes que he trobat#
#mes_punct = "[,.:)(-!?|;$&/[]>%=#*+\\•~@£·_{}©^®`<→°€™›♥←×§″′Â█½à…“★”–●â►−¢²¬░¶↑±¿▾═¦║―¥▓—‹─▒：¼⊕▼▪†■’▀¨▄♫☆é¯♦¤▲è¸¾Ã⋅‘∞∙）↓、│（»，♪╩╚³・╦╣╔╗▬❤ïØ¹≤‡√])"
df['size'] = df['question_text'].str.len()
print(mean(df['size']))
print(median(df['size']))
print(stdev(df['size']))
print(amax(df['size']))
print(amin(df['size']))
# Reemplacament dels caracters especials
# Cesc: He afegit canvis en monedes
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "eur","$": "usd",  "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
#punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
# Neteja dels caracters especials
def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])

    for p in punct:
        text = text.replace(p, f' {p} ')
        #text = text.replace(p, '')
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    
    return text
df['question_text'] = df['question_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
# Cesc: Ho apliquem també amb la nova llista de caracters raros
#df['question_text'] = df['question_text'].apply(lambda x: clean_special_chars(x, mes_punct, punct_mapping))
# Reemplaçament d'errors ortogràfics
mispell_dict = {'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled', 'labour': 'labor', 'organisation': 'organization', 'wwii': 'world war 2', 'citicise': 'criticize', 'youtu ': 'youtube ', 'Qoura': 'Quora', 'sallary': 'salary', 'Whta': 'What', 'narcisist': 'narcissist', 'howdo': 'how do', 'whatare': 'what are', 'howcan': 'how can', 'howmuch': 'how much', 'howmany': 'how many', 'whydo': 'why do', 'doI': 'do I', 'theBest': 'the best', 'howdoes': 'how does', 'mastrubation': 'masturbation', 'mastrubate': 'masturbate', "mastrubating": 'masturbating', 'pennis': 'penis', 'Etherium': 'Ethereum', 'narcissit': 'narcissist', 'bigdata': 'big data', '2k17': '2017', '2k18': '2018', 'qouta': 'quota', 'exboyfriend': 'ex boyfriend', 'airhostess': 'air hostess', "whst": 'what', 'watsapp': 'whatsapp', 'demonitisation': 'demonetization', 'demonitization': 'demonetization', 'demonetisation': 'demonetization'}
# Correcció d'errors ortogràfics
def correct_spelling(x, dic):
    for word in dic.keys():
        x = x.replace(word, dic[word])
    return x
df['question_text'] = df['question_text'].apply(lambda x: correct_spelling(x, mispell_dict))
df['size'] = df['question_text'].str.len()
print(mean(df['size']))
print(median(df['size']))
print(stdev(df['size']))
print(amax(df['size']))
print(amin(df['size']))
df = df.drop(['size'],axis=1)
total_set_qid = df['qid'].values
total_set_X = df["question_text"].fillna("_na_").values
total_set_y = df['target'].values
## Tokenize the sentences
tokenizer = Tokenizer(num_words=None, filters='')
tokenizer.fit_on_texts(list(total_set_X))
total_set_X = tokenizer.texts_to_sequences(total_set_X)
## Pad the sentences 
total_set_X = pad_sequences(total_set_X, maxlen=question_length)
# TODO: provar amb opcio padding='post' per posar els zeros a la dreta
df = pd.concat([pd.DataFrame(total_set_qid) ,pd.DataFrame(total_set_X), pd.DataFrame(total_set_y)], axis=1, keys=["qid", "question_text", "target"])
def embedding_matrix_creator(embeddings_index):
    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    word_index = tokenizer.word_index
    nb_words = len(word_index) #min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words+1, embed_size))
    for word, i in word_index.items():
        #if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    return embedding_matrix
print(len(my_embedding_matrix))
#print(my_embedding_matrix["apple"])
print(type(my_embedding_matrix))
#emb_matrix_fasttext = embedding_matrix_creator(embed_fasttext)
#emb_matrix_glove = embedding_matrix_creator(embed_glove)
#print(len(emb_matrix_fasttext))
#print(len(emb_matrix_glove))
emb_matrix = embedding_matrix_creator(my_embedding_matrix)
#if current_embd == "Glove":
#    emb_matrix = emdedding_matrix_creator(embed_glove)
#    del embed_glove
#elif current_embd == "Paragram":
#    emb_matrix = emdedding_matrix_creator(embed_paragram)
#    del embed_paragram
#elif current_embd == "FastText":
#    emb_matrix = emdedding_matrix_creator(embed_fasttext)
#    del embed_fasttext

# Fem la mitjana de tots els 3 embeddings
#emb_matrix = np.mean([ emdedding_matrix_creator(embed_glove),  
#                       emdedding_matrix_creator(embed_paragram),
#                       emdedding_matrix_creator(embed_fasttext)], axis=1)
#del embed_glove; del embed_paragram; del embed_fasttext; gc.collect(); time.sleep(10)
# Creem el train set i el eval set
train_df, val_df = train_test_split(df[df.target[0]!=-1], test_size=0.1)

train_X = np.array(train_df["question_text"])
train_y = np.array(train_df["target"])

val_X = np.array(val_df["question_text"])
val_y = np.array(val_df["target"])
# Creem el test set
test_df = df[df.target[0]==-1]
#test_df = test_df.drop('target', axis=1)

test_X=np.array(test_df["question_text"])
#print(test_X.shape)
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None
    
    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim
def make_old_model(embedding_matrix, embed_size=300, loss='binary_crossentropy'):
    inp    = Input(shape=(question_length,))
    x      = Embedding(embedding_matrix.shape[0], embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x      = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)
    x      = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
    avg_pl = GlobalAveragePooling1D()(x)
    max_pl = GlobalMaxPooling1D()(x)
    concat = concatenate([avg_pl, max_pl])
    dense  = Dense(64, activation="relu")(concat)
    drop   = Dropout(0.1)(concat)
    output = Dense(1, activation="sigmoid")(concat)
    
    model  = Model(inputs=inp, outputs=output)
    model.compile(loss=loss, optimizer=Adam(lr=0.0001), metrics=['accuracy', f1])
    return model
#model = make_model(emb_matrix)
def model_lstm_gru_atten(embedding_matrix, embed_size=300, loss='binary_crossentropy'):
    inp = Input(shape=(question_length,))
    x = Embedding(embedding_matrix.shape[0], embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.1,seed=seed_nb)(x)
    x = Bidirectional(CuDNNLSTM(64, kernel_initializer=glorot_uniform(seed=seed_nb), return_sequences=True))(x)
    y = Bidirectional(CuDNNGRU(40,kernel_initializer=glorot_uniform(seed=seed_nb), return_sequences=True))(x)

    atten_1 = Attention(question_length)(x) 
    atten_2 = Attention(question_length)(y)
    avg_pool = GlobalAveragePooling1D()(y)
    max_pool = GlobalMaxPooling1D()(y)

    conc = concatenate([atten_1, atten_2, avg_pool, max_pool])
    conc = Dense(16,kernel_initializer=he_uniform(seed=seed_nb),  activation="relu")(conc)
    conc = Dropout(0.1,seed=seed_nb)(conc)
    outp = Dense(1,kernel_initializer=he_uniform(seed=seed_nb),  activation="sigmoid")(conc)    

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss=loss, optimizer=Adam(lr=0.0001), metrics=['accuracy', f1])
    return model
#model = model_lstm_gru_atten(emb_matrix)
model = make_old_model(emb_matrix)
model.summary()
checkpoints = ModelCheckpoint('weights.hdf5', monitor="val_f1", mode="max", verbose=True, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_f1', factor=0.1, patience=2, verbose=1, min_lr=0.000001)
# Trobar el threshold més óptim
def tweak_threshold(pred, truth):
    thresholds = []
    scores = []
    print("Threshold: Valor")
    for thresh in np.arange(0.01, 1.01, 0.01):
        thresh = np.round(thresh, 2)
        thresholds.append(thresh)
        score = f1_score(truth, (pred>thresh).astype(int))
        print(thresh, ": ", score)
        scores.append(score)
    return np.max(scores), thresholds[np.argmax(scores)]
#print(history.history)
#plt.plot(history.history['acc'])
#plt.plot(history.history['f1'])
df_X =  df[df.target[0]!=-1]["question_text"]
df_y =  df[df.target[0]!=-1]["target"]

# Cesc: Afegim la separacio de K-Folds
train_meta = np.zeros(df_y.shape)
test_meta = np.zeros(test_X.shape[0])

#K_FOLDS = 4
#K_FOLD_EPOCHS = 1 #int(epochs/K_FOLDS)
my_splits = list(StratifiedKFold(n_splits=K_FOLDS,
                                  shuffle=True,
                                  random_state=DATA_SPLIT_SEED).split(df_X, df_y))

for idx, (train_idx, valid_idx) in enumerate(my_splits):
    print("======== K-FOLD: {0} ==========".format(idx))
    train_X = df_X.iloc[train_idx]
    train_y = df_y.iloc[train_idx]
    val_X = df_X.iloc[valid_idx]
    val_y = df_y.iloc[valid_idx]
#    model = model_lstm_gru_atten(emb_matrix)
    model = make_old_model(emb_matrix)
#   pred_val_y, pred_test_y, best_score = train_pred(model, X_train, y_train, X_val, y_val, epochs = 8, callback = [clr,])
    history = model.fit(train_X, train_y, batch_size=batch_size, epochs=K_FOLD_EPOCHS, validation_data=[val_X, val_y], callbacks=[checkpoints, reduce_lr])
    model.load_weights('weights.hdf5')
    eval_pred = model.predict(val_X, batch_size=batch_size, verbose=1)
    test_pred = model.predict(test_X, batch_size=batch_size, verbose=1)
   # pred_val_y = model.predict([val_X], batch_size=batch_size, verbose=0)
    train_meta[valid_idx] = eval_pred#.reshape(-1)
    test_meta += test_pred.reshape(-1) / len(my_splits)
score_val, best_thresh = tweak_threshold(train_meta, df_y)
print("=====================================")
print(f"Scored {round(score_val, 4)} for threshold {best_thresh} with untreated texts on validation data")
# Imprimim la submission en un fitxer
y_te = (np.array(test_meta) > best_thresh).astype(np.int)
qid = test_df["qid"].values
#submit_df = pd.DataFrame({"qid": test_df["qid"], "prediction": y_te})
#submit_df = pd.concat([pd.DataFrame(qid),pd.DataFrame(y_te)], axis = 1, keys=["qid", "prediction"])
submit_df = pd.concat([pd.DataFrame(qid, columns=['qid']),pd.DataFrame(y_te, columns=['prediction'])], axis = 1)
submit_df.to_csv("submission_cesc.csv", index=False)
