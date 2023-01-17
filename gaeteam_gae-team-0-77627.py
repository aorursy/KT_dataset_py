import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1)
import matplotlib.pyplot as plt
import random
import time

tweets = pd.read_csv('../input/train.csv')
tweets.columns
tweets = tweets.dropna()
tweets[['party', 'text']].head(15)
x = tweets.groupby('party').size().index.values
y = tweets.groupby('party').size().values

plt.bar(x,y)
x = tweets.groupby('party').size().index.values
y = tweets.groupby('party')['retweet_count'].mean().values

plt.bar(x,y)
x = tweets.groupby('party').size().index.values
y = tweets.groupby('party')['favorite_count'].mean().values

plt.bar(x,y)
from langdetect import detect_langs
from langdetect import detect
idioma = []
idioma2 = []
idioms = {'ca':1, 'es':0}
idioms2 = {'ca':0, 'es':1}

idioma2
count = 0
for i in range(len(tweets)):
    try:
        lang = detect(tweets['text'].values[i])
        idioma.append(idioms[lang])
        idioma2.append(idioms2[lang])
        
    except: 
        count +=1
        idioma.append(0)
        idioma2.append(0)

tweets['Català'] = idioma
tweets['Castellà'] = idioma2
x = tweets.groupby('party').size().index.values
y = tweets.groupby('party')['Català'].mean().values

plt.bar(x,y)
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from stop_words import get_stop_words

tweets_train = tweets.dropna()
text_train = tweets['text']
text_test = pd.read_csv('../input/test.csv')['text']

filters = ['.', ',', '"', "'", ':', ';', '(', ')', '[', ']',
           '{', '}','?', '!',"''", "\n", '’', '’’'
           ,'“', '¿', '¡', '”', '``', '...', '@', 'q', 'xq', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '#']
stop_words = get_stop_words('italian')+get_stop_words('spanish')+get_stop_words('english')+get_stop_words('catalan')+get_stop_words('french')+get_stop_words('german')
def tokenizing(words, filters):
    true_tokens = []
    for elem in words:
        try:
            tokenized = word_tokenize(elem)
            filtered = [word.lower() for word in tokenized if  word not in filters and word.lower() not in stop_words
                        and len(word) < 25]
            true_tokens.append(filtered)
        except: print(elem)
    return true_tokens
text_tokenized = tokenizing(text_train, filters)
test_tokenized = tokenizing(text_test, filters)
from keras.preprocessing import text
n_words = 20000
tokenizer = text.Tokenizer(num_words = n_words, filters='"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n¿¡')
tokenizer.fit_on_texts(text_tokenized+test_tokenized)
traintok = tokenizer.texts_to_sequences(text_tokenized)
testtok = tokenizer.texts_to_sequences(test_tokenized)
labels = tweets['party'].values
tokenizer.word_index
dicc = {}
for name, age in tokenizer.word_index.items():    
        dicc[age] = name
    
dicc
print(len(text_tokenized+test_tokenized))
print(len(text_tokenized))
print(len(traintok))
print(len(dicc))
def most_freq_words(text, nb_words, party):
    #counts = np.zeros(15024+1)
    counts = np.zeros(20428+1)
    for i, tweet in enumerate(text):
        if labels[i] == party:
            for elem in tweet:
                counts[elem] += 1
        else: pass
        
    most_frequent = counts.argsort()[::-1]
    
    
    for j in range(nb_words):
        
        print("The {} word is {}. It appears {} times.".format(j+1, dicc[most_frequent[j]], counts[most_frequent[j]]))
            
            
    
most_freq_words(traintok, 50, 'jxcat')
features_cs = [['sánchez'], ['cataluña', 'catalanes'], ['españoles', 'españa'], ['separatistas', 'separatismo', 'nacionalistas'],
              ['torra', 'puigdemont'], ['ciudadanos'], ['ley', 'justicia', 'derechos'], ['democracia', 'libertad'], 
              ['golpe'], ['moncloa'], ['psoe'], ['víctimas'], ['convivencia']]

features_pp = [['ppcatalunya', 'ppopular'],['cup'], ['badalona', 'vecinos', 'alcadesa', 'ayuntamiento'], 
               ['marianorajoy', 'albiol_xg', 'Albiol'], ['gobierno'], ['sanchezcastejon'], 
               ['independentismo', 'independentistas', 'radical'], ['quimtorraipla']] 

features_psc = [['via', 'vía'],
 ['elperiodico', 'periódico'],
 ['👇🏼'],  ['lavanguardia'], ['pscbarcelona','socialistes_cat', 'psc'],
 ['colau'], ['editorial'], ['el_pais']]

features_comuns = [['barcelona', 'bcn'],
 ['més'],['pp'],['persones'],['gràcies'],['gran'],['país'],['social', 'valors', 'solidaritat'],['totes']]

features_erc = [['esquerra_erc'],
 ['avui'],
 ['junqueras'],
 ['companys'],
 ['independència'],
 ['república', 'republicans', 'repúbliques'],
 ['democràcia', 'llibertat'],
 ['espanyol', 'l\'estat'],
 ['referèndum'],
 ['diàleg'],
 ['155'],
 ['llei', 'justícia'],
 ['catalunya', 'catalans']]


features_jxcat = [
 ['pau'],
 ['compromís'],
 ['exiliats'],
 ['catalana'],
    ['president'],
 ['presos', 'presó'],
 ['nostres'],
 ['🎗'],
 ['poble', 'país'],
 ['jordi'],
 ['endavant'],
 ['lliures'],
 ['mesos'],
 ['jordialapreso']]
total_features = features_comuns+features_cs+features_erc+features_jxcat+features_pp+features_psc
X = np.zeros((len(tweets), len(total_features)+4))
X[:,:4] = tweets[['favorite_count', 'retweet_count', 'Català', 'Castellà']].values
for i, tweet in enumerate(text_tokenized):
    for word in tweet:
        for j, feature in enumerate(total_features):
            if word in feature: X[i,j+4] += 1
                
                
y = tweets['party']
from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[:,:4])

X[:,:4] = X_scaled
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.2, random_state=42)


GBC = ensemble.GradientBoostingClassifier(random_state=42)
GBC.fit(X_train3,y_train3)

yhat_GBC_val = GBC.predict(X_test3)
print("Validation accuracy Gradient Boosting Classifier:", metrics.accuracy_score(y_test3,yhat_GBC_val))


GBC.fit(X,y)
yhat = GBC.predict(X)
print("Train accuracy Gradient Boosting Classifier:", metrics.accuracy_score(y,yhat))
test = pd.read_csv('../input/test.csv')
test = test.dropna()
idioma = []
idioma2 = []
idioms = {'ca':1, 'es':0}
idioms2 = {'ca':0, 'es':1}

idioma2
count = 0
for i in range(len(test)):
    try:
        lang = detect(test['text'].values[i])
        idioma.append(idioms[lang])
        idioma2.append(idioms2[lang])
        
    except: 
        count +=1
        idioma.append(0)
        idioma2.append(0)

test['Català'] = idioma
test['Castellà'] = idioma2
X_test = np.zeros((len(test), 69))
X_test[:,:4] = test[['favorite_count', 'retweet_count', 'Català', 'Castellà']].values
X_test_scaled = scaler.fit_transform(X_test[:,:4])

X_test[:,:4] = X_test_scaled
for i, tweet in enumerate(test_tokenized):
    for word in tweet:
        for j, feature in enumerate(total_features):
            if word in feature: X_test[i,j+4] += 1
preds_bons = GBC.predict(X_test)
preds_true = np.insert(np.array(preds_bons),584, 'jxcat')
test_sub = pd.read_csv('../input/test.csv')
test_sub
submisions = pd.DataFrame(test_sub['Id'])
submisions['Prediction'] = preds_true
submisions
submisions.to_csv('submission.csv', index=False)
