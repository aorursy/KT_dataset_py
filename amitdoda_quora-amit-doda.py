# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)maxle



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import keras

import operator 
df_train=pd.read_csv('../input/train.csv')

df_test=pd.read_csv('../input/test.csv')
print ("Train data target 1")

print (df_train[df_train['target']==1].count())



print ("Train data target 0")

print (df_train[df_train['target']==0].count())
all_phrases=df_train[df_train.target != 2]

all_words = []

for t in all_phrases.question_text:

    all_words.append(t)

all_words[:4]
all_text = pd.Series(all_words).str.cat(sep=' ')
from wordcloud import WordCloud, STOPWORDS

wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(all_text)

plt.figure(figsize=(12,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
neg_phrases = df_train[df_train.target == 1]

neg_words = []

for t in neg_phrases.question_text:

    neg_words.append(t)

neg_words[:4]
neg_text = pd.Series(neg_words).str.cat(sep=' ')

neg_text[:100]
from wordcloud import WordCloud, STOPWORDS

wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(neg_text)

plt.figure(figsize=(12,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
pos_phrases = df_train[df_train.target == 0]

pos_words = []

for t in pos_phrases.question_text:

    pos_words.append(t)

pos_words[:4]
pos_text = pd.Series(pos_words).str.cat(sep=' ')

pos_text[:100]
from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(pos_text)

plt.figure(figsize=(12,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
df_train['length'] = df_train['question_text'].str.count(' ') + 1

df_test['length'] = df_test['question_text'].str.count(' ') + 1
print (df_train[df_train['target']==1]['length'].median())

print (df_train[df_train['target']==0]['length'].median())
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



    print('Found embeddings for {:.3%} of vocab'.format(len(known_words) / len(vocab)))

    print('Found embeddings for  {:.3%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))

    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]

    return unknown_words
def add_lower(embedding, vocab):

    count = 0

    for word in vocab:

        if word in embedding and word.lower() not in embedding:  

            embedding[word.lower()] = embedding[word]

            count += 1

    print(f"Added {count} words to embedding")
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }

puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 

 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 

 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 

 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 

 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

mispell_dict = {'examinaton': 'examination',

                'undergraduation': 'under graduation',

                'fiancé': 'fiance',

                'qoura': 'quora',

                'bhakts': 'followers',

                'quorans': 'quora users',

                'brexit': 'Britain exit',

                'cryptocurrencies': 'cryptocurrency',

                'colour': 'color',

                'centre': 'center',

                'favourite': 'favorite',

                'travelling': 'traveling',

                'counselling': 'counseling',

                'theatre': 'theater',

                'cancelled': 'canceled',

                'labour': 'labor',

                'organisation': 'organization',

                'wwii': 'world war 2',

                'citicise': 'criticize',

                'youtu ': 'youtube ',

                'Qoura': 'Quora',

                'sallary': 'salary',

                'Whta': 'What',

                'narcisist': 'narcissist',

                'howdo': 'how do',

                'whatare': 'what are',

                'howcan': 'how can',

                'howmuch': 'how much',

                'howmany': 'how many',

                'whydo': 'why do',

                'doI': 'do I',

                'theBest': 'the best',

                'howdoes': 'how does',

                'mastrubation': 'masturbation',

                'mastrubate': 'masturbate',

                "mastrubating": 'masturbating',

                'pennis': 'penis',

                'Etherium': 'Ethereum',

                'narcissit': 'narcissist',

                'bigdata': 'big data',

                '2k17': '2017',

                '2k18': '2018',

                'qouta': 'quota',

                'exboyfriend': 'ex boyfriend',

                'airhostess': 'air hostess',

                "whst": 'what',

                'watsapp': 'whatsapp',

                'demonitisation': 'demonetization',

                'demonitization': 'demonetization',

                'demonetisation': 'demonetization',

                'pokémon': 'pokemon',

                'paralizing': 'paralising',

                'perfeccionism': 'perfectionism',

                'depreciaton': 'depreciation',

                'abvicable': 'abdicable',

                'catanation': 'catenation',

                'leasership': 'leadership',

                'webassembly': 'web assembly',

                'fortitide': 'fortitude',

                'withdrow': 'withdraw',

                'bomblasts': 'bomb blasts',

                'engineerer': 'engineer',

                'citycarclean': 'city car clean',

                'billionsites': 'billion sites',

                'willhandjob': 'will hand job',

                'fireguns': 'fire guns',

                'justeat': 'just eat',

                'ubereats': 'uber eats',

                'doinformation': 'do information',

                'freshersworld': 'freshers world',

                'topicwise': 'topic wise',

                'excitee': 'excited',

                'bengalore': 'bangalore',

                'proproetor': 'proprietor',

                'migeration': 'migration',

                'ejectulate': 'Ejaculate',

                'glucoze': 'glucose',

                'whatapp': 'whatsapp',

                'sumup': 'sum up',

                'besic': 'basic',

                'experienceed': 'experienced',

                'feminisam': 'feminism',

                'kayboard': 'keyboard',

                'retructuring': 'restructuring',

                'becomd': 'become',

                'preidct': 'predict',

                'statups': 'startups',

                'superbrains': 'super brains',

                'becoome': 'become',

                'gwroth': 'growth',

                'wakeupnow': 'wake up now',

                'headpone': 'headphone',

                'industiry': 'industry',

                'arichtecture': 'architecture',

                'simlarity': 'similarity',

                'walmartlabs': 'walmart labs',

                'thunderstike': 'thunder stike',

                'maintanable': 'maintainable',

                'diffferently': 'differently',

                'careamics': 'ceramics',

                'sinnister': 'sinister',

                'quoras': 'quora',

                'breakimg': 'breaking',

                'surggery': 'surgery',

                'whatwill': 'what will',

                'adhaar': 'identity',

                'aidentity': 'identity',

                'upwork': 'up work',

                'alshamsi': 'al shamsi',

                'litecoin': 'cryptocurrency ',

                'chapterwise': 'chapter wise',

                'blockchains': 'blockchain',

                'flipcart': 'flipkart',

               'Terroristan': 'terrorist Pakistan',

                'terroristan': 'terrorist Pakistan',

                'BIMARU': 'Bihar, Madhya Pradesh, Rajasthan, Uttar Pradesh',

                'Hinduphobic': 'Hindu phobic',

                'hinduphobic': 'Hindu phobic',

                'Hinduphobia': 'Hindu phobic',

                'hinduphobia': 'Hindu phobic',

                'Babchenko': 'Arkady Arkadyevich Babchenko faked death',

                'Boshniaks': 'Bosniaks',

                'Dravidanadu': 'Dravida Nadu',

                'mysoginists': 'misogynists',

                'MGTOWS': 'Men Going Their Own Way',

                'mongloid': 'Mongoloid',

                'unsincere': 'insincere',

                'meninism': 'male feminism',

                'jewplicate': 'jewish replicate',

                'unoin': 'Union',

                'daesh': 'Islamic State of Iraq and the Levant',

                'Kalergi': 'Coudenhove-Kalergi',

                'Bhakts': 'Bhakt',

                'bhakts': 'Bhakt',

                'Tambrahms': 'Tamil Brahmin',

                'Pahul': 'Amrit Sanskar',

                'SJW': 'social justice warrior',

                'SJWs': 'social justice warrior',

                'incel': ' involuntary celibates',

                'incels': ' involuntary celibates',

                'emiratis': 'Emiratis',

                'weatern': 'western',

                'westernise': 'westernize',

                'Pizzagate': 'Pizzagate conspiracy theory',

                'naïve': 'naive',

                'Skripal': 'Sergei Skripal',

                'Remainers': 'British remainer',

                'remainers': 'British remainer',

                'bremainer': 'British remainer',

                'antibrahmin': 'anti Brahminism',

                'HYPSM': 'Harvard, Yale, Princeton, Stanford, MIT',

                'HYPS': 'Harvard, Yale, Princeton, Stanford',

                'kompromat': 'compromising material',

                'Tharki': 'pervert',

                'tharki': 'pervert',

                'mastuburate': 'masturbate',

                'Zoë': 'Zoe',

                'indans': 'Indian',

                'xender': 'gender',

                'Naxali ': 'Naxalite ',

                'Naxalities': 'Naxalites',

                'Bathla': 'Namit Bathla',

                'Mewani': 'Indian politician Jignesh Mevani',

                'clichéd': 'cliche',

                'cliché': 'cliche',

                'clichés': 'cliche',

                'Wjy': 'Why',

                'Fadnavis': 'Indian politician Devendra Fadnavis',

                'Awadesh': 'Indian engineer Awdhesh Singh',

                'Awdhesh': 'Indian engineer Awdhesh Singh',

                'Khalistanis': 'Sikh separatist movement',

                'madheshi': 'Madheshi',

                'BNBR': 'Be Nice, Be Respectful',

                'Bolsonaro': 'Jair Bolsonaro',

                'XXXTentacion': 'Tentacion',

                'Padmavat': 'Indian Movie Padmaavat',

                'Žižek': 'Slovenian philosopher Slavoj Žižek',

                'Adityanath': 'Indian monk Yogi Adityanath',

                'Brexit': 'British Exit',

                'Brexiter': 'British Exit supporter',

                'Brexiters': 'British Exit supporters',

                'Brexiteer': 'British Exit supporter',

                'Brexiteers': 'British Exit supporters',

                'Brexiting': 'British Exit',

                'Brexitosis': 'British Exit disorder',

                'brexit': 'British Exit',

                'brexiters': 'British Exit supporters',

                'jallikattu': 'Jallikattu',

                'fortnite': 'Fortnite ',

                'Swachh': 'Swachh Bharat mission campaign ',

                'Quorans': 'Quoran',

                'Qoura ': 'Quora ',

                'quoras': 'Quora',

                'Quroa': 'Quora',

                'QUORA': 'Quora',

                'narcissit': 'narcissist',

                # extra in sample

                'Doklam': 'Tibet',

                'Drumpf': 'Donald Trump fool',

                'Drumpfs': 'Donald Trump fools',

                'Strzok': 'Hillary Clinton scandal',

                'rohingya': 'Rohingya ',

                'wumao': 'cheap Chinese stuff',

                'wumaos': 'cheap Chinese stuff',

                'Sanghis': 'Sanghi',

                'Tamilans': 'Tamils',

                'biharis': 'Biharis',

                'Rejuvalex': 'hair growth formula',

                'Feku': 'Fake',

                'deplorables': 'deplorable',

                'muhajirs': 'Muslim immigrant',

                'Gujratis': 'Gujarati',

                'Chutiya': 'Fucker',

                'Chutiyas': 'Fucker',

                'thighing': 'masturbate',

                '卐': 'Nazi Germany',

                'Pribumi': 'Native Indonesian',

                'Gurmehar': 'Gurmehar Kaur Indian student activist',

                'Novichok': 'Soviet Union agents',

                'Khazari': 'Khazars',

                'Demonetization': 'demonetization',

                'demonetisation': 'demonetization',

                'demonitisation': 'demonetization',

                'demonitization': 'demonetization',

                'demonetisation': 'demonetization',

                'cryptocurrencies': 'cryptocurrency',

                'Hindians': 'North Indian who hate British',

                'vaxxer': 'vocal nationalist ',

                'remoaner': 'remainer ',

                'bremoaner': 'British remainer ',

                'Jewism': 'Judaism',

                'Eroupian': 'European',

                'WMAF': 'White male married Asian female',

                'moeslim': 'Muslim',

                'cishet': 'cisgender and heterosexual person',

                'Eurocentric': 'Eurocentrism ',

                'Jewdar': 'Jew dar',

                'Asifa': 'abduction, rape, murder case ',

                'marathis': 'Marathi',

                'Trumpanzees': 'Trump chimpanzee fool',

                'Crimean': 'Crimea people ',

                'atrracted': 'attract',

                'LGBT': 'lesbian, gay, bisexual, transgender',

                'Boshniak': 'Bosniaks ',

                'Myeshia': 'widow of Green Beret killed in Niger',

                'demcoratic': 'Democratic',

                'raaping': 'rape',

                'Dönmeh': 'Islam',

                'feminazism': 'feminism nazi',

                'langague': 'language',

                'Hongkongese': 'HongKong people',

                'hongkongese': 'HongKong people',

                'Kashmirians': 'Kashmirian',

                'Chodu': 'fucker',

                'penish': 'penis',

                'micropenis': 'tiny penis',

                'Madridiots': 'Real Madrid idiot supporters',

                'Ambedkarite': 'Dalit Buddhist movement ',

                'ReleaseTheMemo': 'cry for the right and Trump supporters',

                'harrase': 'harass',

                'Barracoon': 'Black slave',

                'Castrater': 'castration',

                'castrater': 'castration',

                'Rapistan': 'Pakistan rapist',

                'rapistan': 'Pakistan rapist',

                'Turkified': 'Turkification',

                'turkified': 'Turkification',

                'Dumbassistan': 'dumb ass Pakistan',

                'facetards': 'Facebook retards',

                'rapefugees': 'rapist refugee',

                'superficious': 'superficial',

                # extra from kagglers

                'colour': 'color',

                'centre': 'center',

                'favourite': 'favorite',

                'travelling': 'traveling',

                'counselling': 'counseling',

                'theatre': 'theater',

                'cancelled': 'canceled',

                'labour': 'labor',

                'organisation': 'organization',

                'wwii': 'world war 2',

                'citicise': 'criticize',

                'youtu ': 'youtube ',

                'sallary': 'salary',

                'Whta': 'What',

                'narcisist': 'narcissist',

                'narcissit': 'narcissist',

                'howdo': 'how do',

                'whatare': 'what are',

                'howcan': 'how can',

                'howmuch': 'how much',

                'howmany': 'how many',

                'whydo': 'why do',

                'doI': 'do I',

                'theBest': 'the best',

                'howdoes': 'how does',

                'mastrubation': 'masturbation',

                'mastrubate': 'masturbate',

                'mastrubating': 'masturbating',

                'pennis': 'penis',

                'Etherium': 'Ethereum',

                'bigdata': 'big data',

                '2k17': '2017',

                '2k18': '2018',

                'qouta': 'quota',

                'exboyfriend': 'ex boyfriend',

                'airhostess': 'air hostess',

                'whst': 'what',

                'watsapp': 'whatsapp',

                # extra

                'bodyshame': 'body shaming',

                'bodyshoppers': 'body shopping',

                'bodycams': 'body cams',

                'Cananybody': 'Can any body',

                'deadbody': 'dead body',

                'deaddict': 'de addict',

                'Northindian': 'North Indian ',

                'northindian': 'north Indian ',

                'northkorea': 'North Korea',

                'Whykorean': 'Why Korean',

                'koreaboo': 'Korea boo ',

                'Brexshit': 'British Exit bullshit',

                'shithole': 'shithole ',

                'shitpost': 'shit post',

                'shitslam': 'shit Islam',

                'shitlords': 'shit lords',

                'Fck': 'Fuck',

                'fck': 'fuck',

                'Clickbait': 'click bait ',

                'clickbait': 'click bait ',

                'mailbait': 'mail bait',

                'healhtcare': 'healthcare',

                'trollbots': 'troll bots',

                'trollled': 'trolled',

                'trollimg': 'trolling',

                'cybertrolling': 'cyber trolling',

                'sickular': 'India sick secular ',

                'suckimg': 'sucking',

                'Idiotism': 'idiotism',

                'Niggerism': 'Nigger',

                'Niggeriah': 'Nigger'}

glove = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

paragram =  '../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt'

wiki_news = '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'







def load_embed(file):

    def get_coefs(word,*arr): 

        return word, np.asarray(arr, dtype='float32')

    

    if file == '../input/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec':

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file) if len(o)>100)

    else:

        embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='latin'))

        

    return embeddings_index

glove_embeddings = load_embed(glove)

print (len(glove_embeddings))
train = df_train['question_text']

test = df_test['question_text']

df = pd.concat([train ,test])



vocab = build_vocab(df)
print("oov : Glove ")

oov = check_coverage(vocab, glove_embeddings)



add_lower(glove_embeddings, vocab)



print("oov : ")

oov = check_coverage(vocab, glove_embeddings)

def known_contractions(embed):

    known = []

    for contract in contraction_mapping:

        if contract in embed:

            known.append(contract)

    return known



def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text
def clean_special_chars(text, punct, puncts, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])

    

    for p in punct:

        text = text.replace(p, f' {p} ')

    

    for p in puncts:

        text = text.replace(p, f' {p} ')

        

    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last

    for s in specials:

        text = text.replace(s, specials[s])

    

    return text
def correct_spelling(x, dic):

    for word in dic.keys():

        x = x.replace(word, dic[word])

    return x
import re

def clean_numbers(x):

    x = re.sub('[0-9]{5,}', '#####', x)

    x = re.sub('[0-9]{4}', '####', x)

    x = re.sub('[0-9]{3}', '###', x)

    x = re.sub('[0-9]{2}', '##', x)

    return x
# Lowering

df_train['question_text'] = df_train['question_text'].apply(lambda x: x.lower())

# Contractions

df_train['question_text'] = df_train['question_text'].apply(lambda x: clean_contractions(x, contraction_mapping))

# Special characters

df_train['question_text'] = df_train['question_text'].apply(lambda x: clean_special_chars(x, punct, puncts, punct_mapping))

# Spelling mistakes

df_train['question_text'] = df_train['question_text'].apply(lambda x: correct_spelling(x, mispell_dict))

# Clean Numbers

df_train['question_text'] = df_train['question_text'].apply(lambda x: clean_numbers(x))
# Lowering

df_test['question_text'] = df_test['question_text'].apply(lambda x: x.lower())

# Contractions

df_test['question_text'] = df_test['question_text'].apply(lambda x: clean_contractions(x, contraction_mapping))

# Special characters

df_test['question_text'] = df_test['question_text'].apply(lambda x: clean_special_chars(x, punct,puncts, punct_mapping))

# Spelling mistakes

df_test['question_text'] = df_test['question_text'].apply(lambda x: correct_spelling(x, mispell_dict))

# clean numbers

df_test['question_text'] = df_test['question_text'].apply(lambda x: clean_numbers(x))
train = df_train['question_text']

test = df_test['question_text']

df = pd.concat([train ,test])



vocab = build_vocab(df)



print("oov : ")

oov = check_coverage(vocab, glove_embeddings)
#oov
#train_Y=df_train['target']

#train_X=df_train['question_text']

from sklearn.model_selection import train_test_split

#X_train, X_test2, y_train, y_test2 = train_test_split(train_X, train_Y, test_size=0.04, random_state=123)

#X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.045, random_state=123)



train_df, test2_df = train_test_split(df_train, test_size=0.04, random_state=123)

train_df, valid_df = train_test_split(train_df, test_size=0.00001, random_state=123)



train_df=train_df.reset_index(drop=True)

valid_df=valid_df.reset_index(drop=True)

test2_df=test2_df.reset_index(drop=True)



X_train=train_df['question_text'].values

y_train=train_df['target'].values





X_test2=test2_df['question_text'].values

y_test2=test2_df['target'].values





X_valid=valid_df['question_text'].values

y_valid=valid_df['target'].values





X_test=df_test['question_text'].values







print (X_train.shape)

print (y_train.shape)





print (X_valid.shape)

print (y_valid.shape)







print (X_test2.shape)

print (y_test2.shape)





print (X_test.shape)





print (train_df.shape)

print (valid_df.shape)

print (test2_df.shape)
X_len_train=train_df['length'].values

X_len_test2=test2_df['length'].values

X_len_valid=valid_df['length'].values

X_len_test=df_test['length'].values



print (X_len_train.shape)

print (X_len_test2.shape)

print (X_len_valid.shape)

print (X_len_test.shape)
embedding_dim = 300 # how big is each word vector

max_features = 120000 # how many unique words to use (i.e num rows in embedding vector)

maxlen = 70 # max number of words in a question to use



from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=max_features)

#tokenizer = Tokenizer()

tokenizer.fit_on_texts(X_train.tolist() + X_valid.tolist()+ X_test2.tolist() + X_test.tolist())
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

print (vocab_size)
#tokenizer.word_index
#maxlen=70
def index_to_matrix(embeddings_index,word_index):

    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    for word, i in word_index.items():

        embedding_vector = embeddings_index.get(word)

        if embedding_vector is not None:

            # words not found in embedding index will be all-zeros.

            embedding_matrix[i] = embedding_vector

    return (embedding_matrix)
glove_embedding_matrix=index_to_matrix(glove_embeddings,tokenizer.word_index)

embedding_matrix=glove_embedding_matrix
from keras.models import Model

from keras.layers import Dense, Embedding, Bidirectional, CuDNNGRU,CuDNNLSTM, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, Input, Dropout, Add

from keras.optimizers import Adam

from keras.models import Sequential

from keras import layers

import keras.callbacks

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint
def make_model(embedding_matrix, maxlen, embed_size=300, loss='binary_crossentropy'):

    inp    = Input(shape=(maxlen,))

    inp2   = Input(shape=(1,))

    x      = Embedding(vocab_size, embed_size, weights=[embedding_matrix], trainable=False)(inp)

    #x      = Bidirectional(CuDNNGRU(256, return_sequences=True))(x)

    #x      = Bidirectional(CuDNNGRU(128, return_sequences=True))(x)

    x      = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

    x      = Dropout(0.2)(x)

    x      = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

    #x      = Bidirectional(CuDNNLSTM(32, return_sequences=True))(x)

    #x      = Attention(maxlen)(x)

    #x      = Bidirectional(CuDNNLSTM(32, return_sequences=True))(x)

    x      = Dropout(0.2)(x)

    #x      = Bidirectional(CuDNNLSTM(32, return_sequences=True))(x)

    #x      = Dropout(0.25)(x)

    avg_pl = GlobalAveragePooling1D()(x)

    max_pl = GlobalMaxPooling1D()(x)

    concat = concatenate([avg_pl, max_pl])

    #add=Add()([concat, inp2])

    #concat = concatenate([avg_pl, max_pl,inp2])      # using sentence length as one of the feature.

    dense1  = Dense(32, activation="relu")(concat)

    #dense1  = Dense(32, activation="relu")(concat)

    concat = concatenate([dense1, inp2])

    #drop1   = Dropout(0.2)(concat)

    #dense2  = Dense(8, activation="relu")(drop1)

    #drop2   = Dropout(0.1)(dense2)

    #dense3  = Dense(8, activation="relu")(dense1)

    output = Dense(1, activation="sigmoid")(concat)

    

    #model  = Model(inputs=inp, outputs=output)

    model = Model(inputs=[inp,inp2], outputs=output)

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    #model.compile(optimizer=Adam(lr=0.0001),loss='binary_crossentropy',metrics=['accuracy'])

    

    return model
model = make_model(embedding_matrix,maxlen=70)
model.summary()
import matplotlib.pyplot as plt 

plt.style.use('ggplot')



def plot_history(history):

    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]

    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]

    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]

    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    

    if len(loss_list) == 0:

        print('Loss is missing in history')

        return 

    

    ## As loss always exists

    epochs = range(1,len(history.history[loss_list[0]]) + 1)

    

    ## Loss

    plt.figure(1)

    for l in loss_list:

        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    for l in val_loss_list:

        plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    

    plt.title('Loss')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()

    

    ## Accuracy

    plt.figure(2)

    for l in acc_list:

        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    for l in val_acc_list:    

        plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')



    plt.title('Accuracy')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.legend()

    plt.show()
from sklearn.model_selection import StratifiedKFold

seed = 7

n_splits=5

np.random.seed(seed)

kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
from keras.preprocessing.sequence import pad_sequences

i=1

maxlength={}

maxl={}

for train, valid in kfold.split(X_train, y_train):

    #print ((train))

    if i <=n_splits:

        maxl[i]=X_len_train[train].max()

        print (X_len_train[train])

        maxlength[i]=int(np.quantile(X_len_train[train],0.999))

        print ("Running Fold", i, "/", n_splits)

        print ("split 99.9 percentile length",maxlength[i],"split max length",maxl[i])

        modelname=str("Model") + str(i)

        #print (train)

        #print (valid)

        train_data=[X_train[j] for j in train]

        valid_data=[X_train[k] for k in valid]

        

        #print (train_data[0])

        #print (valid_data[0])

        

        train_data = tokenizer.texts_to_sequences(train_data)

        valid_data = tokenizer.texts_to_sequences(valid_data)

        train_data = pad_sequences(train_data, padding='post', maxlen=maxlength[i])

        valid_data = pad_sequences(valid_data, padding='post', maxlen=maxlength[i])

        #print (train_data[0])

        #print (valid_data[0])

        #x_test2 = pad_sequences(x_test2, padding='post', maxlen=maxlen)

        #x_test = pad_sequences(x_test, padding='post', maxlen=maxlen)

        #print (x_train[train][:10])

        #print ('before')

        model = make_model(embedding_matrix,maxlength[i])

        

        #print ('after')        

        #print ("train", x_train[train][:1])

        #print ("train", y_train[train][:100])

        print ("train", y_train[train].sum())

        print ("validation",y_train[valid].sum())

        checkpointer = ModelCheckpoint(filepath=modelname,monitor='val_loss', mode='auto', verbose = 1, save_best_only=True)

        history = model.fit([train_data,X_len_train[train]], y_train[train],

                        epochs=5,

                        validation_data=([valid_data,X_len_train[valid]], y_train[valid]),

                        batch_size=512,callbacks=[checkpointer])

        plot_history(history)

        i=i+1
#plot_history(history)
y_pred={}

y_pred_test={}



count=0

for i in np.arange(1, n_splits+1, 1):

    try:

        model.load_weights(str("Model") + str(i))

        print (str("Model") + str(i))

        import sklearn

        from sklearn.metrics import f1_score

        #y_pred[i] = model.predict(x_test2, batch_size=512, verbose=1)

        #y_pred_test[i] = model.predict(x_test, batch_size=512, verbose=1)

        

        x_test2 = tokenizer.texts_to_sequences(X_test2)

        x_test = tokenizer.texts_to_sequences(X_test)

        x_test2 = pad_sequences(x_test2, padding='post', maxlen=maxlength[i])

        x_test = pad_sequences(x_test, padding='post', maxlen=maxlength[i])

        

        y_pred[i] = model.predict([x_test2,X_len_test2], batch_size=512, verbose=1)

        y_pred_test[i] = model.predict([x_test,X_len_test], batch_size=512, verbose=1)

       

        

        model_f1_score={}

        for thresh in np.arange(0.1, 0.91, 0.01):

            thresh = np.round(thresh, 2)

            #print("F1 score at threshold {0} is {1}".format(thresh, sklearn.metrics.f1_score(y_valid, (y_pred>=thresh).astype(int))))

            model_f1_score[thresh]=sklearn.metrics.f1_score(y_test2, (y_pred[i]>=thresh).astype(int))

               

        model_cutoff=max(model_f1_score, key=model_f1_score.get)

        print("Max F1 score  is {1} found at threshold {0}".format(model_cutoff, model_f1_score[model_cutoff]))

        count=count+1

        #y_pred_final=y_pred_final + y_pred[i]

    except:

        pass



print ('count is ', count)

y_pred_final={}

y_pred_test_final={}

for i in np.arange(1, count+1, 1):

    if (i == 1):

        y_pred_final=y_pred[i]

        y_pred_test_final=y_pred_test[i]

        #print (i)

    else:

        y_pred_final=y_pred_final + y_pred[i]

        y_pred_test_final=y_pred_test_final + y_pred_test[i] 

        #print(i)

    

#print (count)

y_pred_final=y_pred_final/count

y_pred_test_final=y_pred_test_final/count

print ("Final Model on hold out") 

model_f1_score={}

for thresh in np.arange(0.1, 0.91, 0.01):

    thresh = np.round(thresh, 2)

    #print("F1 score at threshold {0} is {1}".format(thresh, sklearn.metrics.f1_score(y_valid, (y_pred>=thresh).astype(int))))

    model_f1_score[thresh]=sklearn.metrics.f1_score(y_test2, (y_pred_final>=thresh).astype(int))



model_cutoff=max(model_f1_score, key=model_f1_score.get)

print("Max F1 score  is {1} found at threshold {0}".format(model_cutoff, model_f1_score[model_cutoff]))
y_pred_test2_final_class = (y_pred_final >= model_cutoff).astype(int) 

print (y_test2.sum())

print (y_pred_test2_final_class.sum())



import sklearn

from sklearn.metrics import f1_score

print(sklearn.metrics.f1_score(y_test2, y_pred_test2_final_class)) 



print ('classification report')

from sklearn.metrics import classification_report

print(classification_report(y_test2, y_pred_test2_final_class))



print ('Confusion matrix')

print(sklearn.metrics.confusion_matrix(y_test2, y_pred_test2_final_class)) 



from sklearn.metrics import roc_auc_score

print ('roc score')

print(sklearn.metrics.roc_auc_score(y_test2, y_pred_test2_final_class))
df=pd.DataFrame(columns=['text','y_actual', 'y_pred','y_pred_prob','length'])
df['text'] = X_test2

df['y_pred'] =y_pred_test2_final_class

df['y_pred_prob'] =y_pred_final

df['length'] =X_len_test2

df['y_actual'] =y_test2
df[(df.y_actual != df.y_pred ) & (df.y_pred_prob >=(model_cutoff-0.3)) & (df.y_pred_prob <=(model_cutoff + 0.3))]['text'].values
FN_phrases = df[(df.y_actual == 1) & (df.y_pred == 0)]

FN_phrases.shape



FN_words = []

for t in FN_phrases.text:

    FN_words.append(t)

FN_words[:10]



FN_text = pd.Series(FN_words).str.cat(sep=' ')

FN_text[:100]



from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(FN_text)

plt.figure(figsize=(12,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
from collections import Counter

counts = Counter(FN_text.split())

print(counts)
FP_phrases = df[(df.y_actual == 0) & (df.y_pred == 1)]

FP_phrases.shape





FP_words = []

for t in FP_phrases.text:

    FP_words.append(t)

#print (FP_words[:10])



FP_text = pd.Series(FP_words).str.cat(sep=' ')

#print (FP_text[:100])



from wordcloud import WordCloud

wordcloud = WordCloud(width=1600, height=800, max_font_size=200).generate(FP_text)

plt.figure(figsize=(12,10))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()

from collections import Counter

counts = Counter(FP_text.split())

print(counts)
df_test['prediction']= (y_pred_test_final >= model_cutoff).astype(int) 
#df_test
df_test=df_test.drop(['question_text'], axis=1)

df_test=df_test.drop(['length'], axis=1)
df_test.to_csv(r'submission.csv', index = False)