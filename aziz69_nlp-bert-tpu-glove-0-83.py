import pandas as pd
import warnings
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
#from sklearn.ensemble import StackingClassifier


from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report,confusion_matrix

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import f1_score, make_scorer
f1 = make_scorer(f1_score , average='macro')

warnings.simplefilter('ignore')
data_train = pd.read_csv('../input/nlp-getting-started/train.csv')
data_test = pd.read_csv('../input/nlp-getting-started/test.csv')
print(len(data_train))
data_train.head()
data_train['target'].value_counts()
print(len(data_test))
data_test.head()
data_test.isnull().sum()
#I concatenate both training and testing data to avoid doing the same operations twice plus we need it for vectorization
data_combined = pd.concat([data_train,data_test],ignore_index=True)  
print(len(data_combined))
data_combined.head()
data_combined.drop(['id','target'],axis=1,inplace=True)
#Total Nan Values
data_combined.isnull().sum()
def clean_text_round1(text):
    '''Make text lowercase, remove text in square brackets, remove punctuation, remove words containing numbers and removing weird characters from tweets'''
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[()[\]{}\''',.``?:;!&^]','',text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('#','',text)
    text = re.sub('û*','',text)
    text = re.sub('ûó*','',text)
    text = re.sub('ò*','',text)
    text = re.sub(r'[^\w]', ' ', text)
    text = re.sub('[^a-zA-Z]+', ' ', text)
    return text
round1 = lambda x: clean_text_round1(x)
data_clean = pd.DataFrame(data_combined.text.apply(round1))
data_clean.head()
stop_words = set(stopwords.words('english'))
data_clean['text_tokenized']= data_clean['text'].map(lambda x : word_tokenize(x))
data_clean.head()
def removing_stop_words(text_tokenized) :
    text_tokenized = [w for w in text_tokenized if w not in stop_words]
    for w in text_tokenized :
        w = w.strip()
    return text_tokenized
data_clean['text_tokenized']= data_clean['text_tokenized'].map(lambda x : removing_stop_words(x))
data_clean.head()
#special thanks to this kernel for the amazing slang fix :
#https://www.kaggle.com/nmaguette/up-to-date-list-of-slangs-for-text-preprocessing

abbreviations = {
    "$" : " dollar ",
    "€" : " euro ",
    "4ao" : "for adults only",
    "a.m" : "before midday",
    "a3" : "anytime anywhere anyplace",
    "aamof" : "as a matter of fact",
    "acct" : "account",
    "adih" : "another day in hell",
    "afaic" : "as far as i am concerned",
    "afaict" : "as far as i can tell",
    "afaik" : "as far as i know",
    "afair" : "as far as i remember",
    "afk" : "away from keyboard",
    "app" : "application",
    "approx" : "approximately",
    "apps" : "applications",
    "asap" : "as soon as possible",
    "asl" : "age, sex, location",
    "atk" : "at the keyboard",
    "ave." : "avenue",
    "aymm" : "are you my mother",
    "ayor" : "at your own risk", 
    "b&b" : "bed and breakfast",
    "b+b" : "bed and breakfast",
    "b.c" : "before christ",
    "b2b" : "business to business",
    "b2c" : "business to customer",
    "b4" : "before",
    "b4n" : "bye for now",
    "b@u" : "back at you",
    "bae" : "before anyone else",
    "bak" : "back at keyboard",
    "bbbg" : "bye bye be good",
    "bbc" : "british broadcasting corporation",
    "bbias" : "be back in a second",
    "bbl" : "be back later",
    "bbs" : "be back soon",
    "be4" : "before",
    "bfn" : "bye for now",
    "blvd" : "boulevard",
    "bout" : "about",
    "brb" : "be right back",
    "bros" : "brothers",
    "brt" : "be right there",
    "bsaaw" : "big smile and a wink",
    "btw" : "by the way",
    "bwl" : "bursting with laughter",
    "c/o" : "care of",
    "cet" : "central european time",
    "cf" : "compare",
    "cia" : "central intelligence agency",
    "csl" : "can not stop laughing",
    "cu" : "see you",
    "cul8r" : "see you later",
    "cv" : "curriculum vitae",
    "cwot" : "complete waste of time",
    "cya" : "see you",
    "cyt" : "see you tomorrow",
    "dae" : "does anyone else",
    "dbmib" : "do not bother me i am busy",
    "diy" : "do it yourself",
    "dm" : "direct message",
    "dwh" : "during work hours",
    "e123" : "easy as one two three",
    "eet" : "eastern european time",
    "eg" : "example",
    "embm" : "early morning business meeting",
    "encl" : "enclosed",
    "encl." : "enclosed",
    "etc" : "and so on",
    "faq" : "frequently asked questions",
    "fawc" : "for anyone who cares",
    "fb" : "facebook",
    "fc" : "fingers crossed",
    "fig" : "figure",
    "fimh" : "forever in my heart", 
    "ft." : "feet",
    "ft" : "featuring",
    "ftl" : "for the loss",
    "ftw" : "for the win",
    "fwiw" : "for what it is worth",
    "fyi" : "for your information",
    "g9" : "genius",
    "gahoy" : "get a hold of yourself",
    "gal" : "get a life",
    "gcse" : "general certificate of secondary education",
    "gfn" : "gone for now",
    "gg" : "good game",
    "gl" : "good luck",
    "gmt" : "greenwich mean time",
    "gmta" : "great minds think alike",
    "gn" : "good night",
    "g.o.a.t" : "greatest of all time",
    "goat" : "greatest of all time",
    "goi" : "get over it",
    "gps" : "global positioning system",
    "gr8" : "great",
    "gratz" : "congratulations",
    "gyal" : "girl",
    "h&c" : "hot and cold",
    "hp" : "horsepower",
    "hr" : "hour",
    "hrh" : "his royal highness",
    "ht" : "height",
    "ibrb" : "i will be right back",
    "ic" : "i see",
    "icq" : "i seek you",
    "icymi" : "in case you missed it",
    "idc" : "i do not care",
    "idgadf" : "i do not give a damn fuck",
    "idgaf" : "i do not give a fuck",
    "idk" : "i do not know",
    "ie" : "that is",
    "i.e" : "that is",
    "ifyp" : "i feel your pain",
    "IG" : "instagram",
    "iirc" : "if i remember correctly",
    "ilu" : "i love you",
    "ily" : "i love you",
    "imho" : "in my humble opinion",
    "imo" : "in my opinion",
    "imu" : "i miss you",
    "iow" : "in other words",
    "irl" : "in real life",
    "j4f" : "just for fun",
    "jic" : "just in case",
    "jk" : "just kidding",
    "jsyk" : "just so you know",
    "l8r" : "later",
    "lb" : "pound",
    "lbs" : "pounds",
    "ldr" : "long distance relationship",
    "lmao" : "laugh my ass off",
    "lmfao" : "laugh my fucking ass off",
    "lol" : "laughing out loud",
    "ltd" : "limited",
    "ltns" : "long time no see",
    "m8" : "mate",
    "mf" : "motherfucker",
    "mfs" : "motherfuckers",
    "mfw" : "my face when",
    "mofo" : "motherfucker",
    "mph" : "miles per hour",
    "mr" : "mister",
    "mrw" : "my reaction when",
    "ms" : "miss",
    "mte" : "my thoughts exactly",
    "nagi" : "not a good idea",
    "nbc" : "national broadcasting company",
    "nbd" : "not big deal",
    "nfs" : "not for sale",
    "ngl" : "not going to lie",
    "nhs" : "national health service",
    "nrn" : "no reply necessary",
    "nsfl" : "not safe for life",
    "nsfw" : "not safe for work",
    "nth" : "nice to have",
    "nvr" : "never",
    "nyc" : "new york city",
    "oc" : "original content",
    "og" : "original",
    "ohp" : "overhead projector",
    "oic" : "oh i see",
    "omdb" : "over my dead body",
    "omg" : "oh my god",
    "omw" : "on my way",
    "p.a" : "per annum",
    "p.m" : "after midday",
    "pm" : "after midday",
    "poc" : "people of color",
    "pov" : "point of view",
    "pp" : "pages",
    "ppl" : "people",
    "prw" : "parents are watching",
    "ps" : "postscript",
    "pt" : "point",
    "ptb" : "please text back",
    "pto" : "please turn over",
    "qpsa" : "what happens", #"que pasa",
    "ratchet" : "rude",
    "rbtl" : "read between the lines",
    "rlrt" : "real life retweet", 
    "rofl" : "rolling on the floor laughing",
    "roflol" : "rolling on the floor laughing out loud",
    "rotflmao" : "rolling on the floor laughing my ass off",
    "rt" : "retweet",
    "ruok" : "are you ok",
    "sfw" : "safe for work",
    "sk8" : "skate",
    "smh" : "shake my head",
    "sq" : "square",
    "srsly" : "seriously", 
    "ssdd" : "same stuff different day",
    "tbh" : "to be honest",
    "tbs" : "tablespooful",
    "tbsp" : "tablespooful",
    "tfw" : "that feeling when",
    "thks" : "thank you",
    "tho" : "though",
    "thx" : "thank you",
    "tia" : "thanks in advance",
    "til" : "today i learned",
    "tl;dr" : "too long i did not read",
    "tldr" : "too long i did not read",
    "tmb" : "tweet me back",
    "tntl" : "trying not to laugh",
    "ttyl" : "talk to you later",
    "u" : "you",
    "u2" : "you too",
    "u4e" : "yours for ever",
    "utc" : "coordinated universal time",
    "w/" : "with",
    "w/o" : "without",
    "w8" : "wait",
    "wassup" : "what is up",
    "wb" : "welcome back",
    "wtf" : "what the fuck",
    "wtg" : "way to go",
    "wtpa" : "where the party at",
    "wuf" : "where are you from",
    "wuzup" : "what is up",
    "wywh" : "wish you were here",
    "yd" : "yard",
    "ygtr" : "you got that right",
    "ynk" : "you never know",
    "zzz" : "sleeping bored and tired"
}
def convert_abbrev(text):
    for word in text : 
        a = word
        word = abbreviations[word.lower()] if word.lower() in abbreviations.keys() else word
        l = word.split(' ')
        text = text + l
        text.remove(a)
    return text
data_clean['text_tokenized']=data_clean['text_tokenized'].map(lambda x: convert_abbrev(x))
data_clean.head()
words_cloud_data=data_clean.copy()
words_cloud_data["word_cloud_text"]=words_cloud_data["text_tokenized"].map(lambda x : ' '.join(x))
words_cloud_data = words_cloud_data[0:7612]
words_cloud_data['target']=data_train['target']
words_cloud_data.drop(['text','text_tokenized'],axis=1,inplace=True)
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wc = WordCloud(stopwords=stop_words, background_color="white", colormap="Dark2",
               max_font_size=150, random_state=42)

plt.rcParams['figure.figsize'] = [16, 6]

target_1 =""
target_0 =""

for i in range(len(words_cloud_data)) :
    if words_cloud_data["target"][i]== 1 :
        target_1 = target_1 + words_cloud_data["word_cloud_text"][i]+" "
    if words_cloud_data["target"][i] == 0 :
        target_0 = target_0 + words_cloud_data["word_cloud_text"][i]+ " "


wc.generate(target_1)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()
plt.rcParams['figure.figsize'] = [16, 6]
wc.generate(target_0)
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()
#Some words are very frequent in both target classes, removing them would improve model's score
def remove_confusing_words(text) :
    for i in text :
        if i == 'amp' or i == 'new' or i=='people' :
            text.remove(i)
    return text
data_clean['text_tokenized']=data_clean['text_tokenized'].map(lambda x: remove_confusing_words(x))
data_clean.head()
data_clean['location']=data_combined['location']
data_clean['keyword']=data_combined['keyword']
data_clean.head()
#dataset containing a good number of cities and their countries that will be usefull for us
cities_data = pd.read_csv('../input/worldcities/worldcities.csv')
cities_data.head()
cities_data.drop(['city','lat','lng','admin_name','capital','population','id','iso2','iso3'],axis=1,inplace=True)
cities_data.head()
cities_dict = {}

for i in range(len(cities_data)) :
    if cities_data['country'][i] in cities_dict :
        cities_dict[cities_data['country'][i]].append(cities_data['city_ascii'][i].lower())
    else :
        cities_dict[cities_data['country'][i]] = list()
        cities_dict[cities_data['country'][i]].append(cities_data['city_ascii'][i].lower())
cities_to_countries = {}
for i,j in cities_dict.items() :
    for element in j :
        cities_to_countries[element]=i.lower()    
print(cities_to_countries)        
countries_names = list(cities_dict.keys())
countries_names_min = [] 
for i in countries_names :
    countries_names_min.append(i.lower())    
print(countries_names_min)    
cities  = [] 
for j in cities_dict.values() : 
    cities = cities + j
print(cities)    
keywords=list(data_clean['keyword'].value_counts().index) # creating a list of existing keywords
locations=list(data_clean['location'].value_counts().index) #creating a list of existing locations
countries_names_min = countries_names_min+locations #adding existing locations to the ones from the dataset
def isNaN(num):
    return num != num
data_clean['Location_not_Nan']=0
data_clean['Keyword_not_Nan']=0
for i in range(len(data_clean)) :
    if isNaN(data_clean['location'][i]) :
        data_clean['Location_not_Nan'][i]=0
    else :
        data_clean['Location_not_Nan'][i]=1
    if isNaN(data_clean['keyword'][i]) :
        data_clean['Keyword_not_Nan'][i]=0
    else :
        data_clean['Keyword_not_Nan'][i]=1
data_clean.head()
for i in range(len(data_clean)):
    if isNaN(data_clean['location'][i]) :
        for j in data_clean['text_tokenized'][i] :
            if j in countries_names_min :
                data_clean['location'][i]=j
            elif j in cities :
                data_clean['location'][i]=cities_to_countries[j]
            else :
                data_clean['location'][i]="NoLocation"
            if j in keywords :
                data_clean['keyword'][i] = j
data_clean['location'].fillna(data_clean['location'].mode()[0],inplace=True)
data_clean['location']=data_clean['location'].map(lambda x : x.lower())
data_clean['keyword'].fillna(data_clean['keyword'].mode()[0],inplace=True)
def changing_location(text) :
    if text == 'usa' or text == 'new york' or text == 'gainesville/tampa, fl' or text =='glendale, ca' or text == 'harbour heights, fl' or text =='new jersey' :
        text = 'united states'
    elif text == 'london' or text == 'brentwood uk' :
        text = 'uk'
    return text

def changing_keyword(text) :
    text = re.sub('^.*fire.*$','fire',text)
    text = re.sub('^.*storm.*$','storm',text)
    text = re.sub('^.*emergency.*$','emergency',text)
    text = re.sub('^.*disaster.*$','disaster',text)
    text = re.sub('^.*collapse.*$','collapse',text)
    text = re.sub('^.*bombing.*$','bombing',text)
    text = re.sub('^.*bomb.*$','bomb',text)
    text = re.sub('^.*zone.*$','zone',text)
    text = re.sub('^.*bagging.*$','bagging',text)

    return text

data_clean['location']=data_clean['location'].map(lambda x : changing_location(x))
data_clean['keyword']=data_clean['keyword'].map(lambda x: changing_keyword(x))
frequency_map_location = data_clean['location'].value_counts().to_dict()
frequency_map_keyword = data_clean['keyword'].value_counts().to_dict()

print(frequency_map_location)

print(frequency_map_keyword)
data_clean['location'] = data_clean['location'].map(frequency_map_location)
data_clean['keyword'] = data_clean['keyword'].map(frequency_map_keyword)
data_clean.head()
#I chose lancaster stemmer because it's the heaviest stemmer between porter stemmer and snowballstemmer
from nltk.stem import LancasterStemmer
lancaster = LancasterStemmer()
def lancaster_stemming(text_tokenized) :
    text_tokenized = [lancaster.stem(w) for w in text_tokenized]
    return ' '.join(text_tokenized)
data_clean['text_tokenized']=data_clean['text_tokenized'].map(lambda x:lancaster_stemming(x))
data_clean.head()
def words_in_tweet(text) :
    l = text.split(' ')
    return len(l)
data_clean['tweet_length']=data_clean['text_tokenized'].map(lambda x: words_in_tweet(x))
def avg_length(text) :
    l = text.split(' ')
    len1 = 0
    for i in l :
        len1 += len(i)
    return len1 / len(l) 
data_clean['avg_word_length'] = data_clean['text_tokenized'].map(lambda x:avg_length(x))
data_clean['unique_words']=data_clean['text_tokenized'].map(lambda x:len(set(x.split(' '))))
data_clean.head()
def count_vect(data_clean):
    cv = CountVectorizer(stop_words='english')
    data_cv = cv.fit_transform(data_clean.text_tokenized)
    data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
    data_dtm.index = data_clean.index
    return data_dtm

data_vectorized = count_vect(data_clean[['text_tokenized']])
data_vectorized.head() 
data_clean1 = pd.concat([data_clean,data_vectorized],axis=1)
data_clean1.shape
data_clean_train1 = data_clean1[0:7613]
data_clean_test1 = data_clean1[7613:]
data_clean_train1['prediction_target']=data_train['target']
data_clean_train1.drop(['text','text_tokenized'],axis=1,inplace=True)
data_clean_test1.drop(['text','text_tokenized'],axis=1,inplace=True)
features_to_encode = ['keyword','location','tweet_length','unique_words']

for col in features_to_encode :
    means = data_clean_train1.groupby(col)['prediction_target'].mean()
    data_clean_train1[col] = data_clean_train1[col].map(means)
    data_clean_test1[col] = data_clean_test1[col].map(means)   
data_clean_train1.head()
y= data_clean_train1['prediction_target']
X= data_clean_train1.drop('prediction_target',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=30)
import numpy as np
def select_model(x,y,model):
    scores=cross_val_score(model,x,y,cv=5,scoring='f1')
    acc=np.mean(scores)
    return acc
gaus_NB = GaussianNB()
print(select_model(X,y,gaus_NB))
multi_NB = MultinomialNB()
print(select_model(X,y,multi_NB))
data_clean_test1['tweet_length'].fillna(data_clean_test1['tweet_length'].mean(),inplace=True)
multi_NB.fit(X,y)
predictions = multi_NB.predict(data_clean_test1)
nb_pred = pd.DataFrame(predictions,columns=['target'])
nb_pred.insert(0,'id',data_test['id'])
nb_pred.to_csv("MysubmissionNB.csv",index=False)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
print('Accuracy = {:.2f}'.format(logreg.score(X_test, y_test)))
mlp = MLPClassifier()
mlp = mlp.fit(X_train,y_train)
mlp_pred = mlp.predict(X_test)
print(classification_report(y_test, mlp_pred))
print('Accuracy = {:.2f}'.format(mlp.score(X_test, y_test)))
rfc = RandomForestClassifier()
rfc = rfc.fit(X_train,y_train)
rfc_pred = rfc.predict(X_test)
print(classification_report(y_test, rfc_pred))
ada = AdaBoostClassifier()
ada = ada.fit(X_train,y_train)
ada_pred = ada.predict(X_test)
print(classification_report(y_test, ada_pred))
gbc = GradientBoostingClassifier()
gbc= gbc.fit(X_train,y_train)
gbc_pred = gbc.predict(X_test)
print(classification_report(y_test, gbc_pred))
etc = ExtraTreesClassifier()
etc = etc.fit(X_train,y_train)
etc_pred = etc.predict(X_test)
print(classification_report(y_test, etc_pred))
#from sklearn.ensemble import StackingClassifier
from mlxtend.classifier import StackingClassifier


#estimators = [
 #  ('rf', rfc),
 #   ('mlp', make_pipeline(StandardScaler(),
 #                      mlp  )),
 #   ('multi_nb',multi_NB),
 #   ('ada',ada),
 #   ('etc',etc)
 #]
clf = StackingClassifier(classifiers = [ada,etc,mlp,rfc,multi_NB],meta_classifier=logreg)
#clf = StackingClassifier(
#    estimators=estimators, final_estimator=logreg)
clf.fit(X,y)
predictions = clf.predict(data_clean_test1)
stacking_pred = pd.DataFrame(predictions,columns=['target'])
stacking_pred.insert(0,'id',data_test['id'])
stacking_pred.to_csv("MysubmissionStack.csv",index=False)
#0.8036
from sklearn.model_selection import RandomizedSearchCV

model = XGBClassifier()

parameters = {
    'n_estimators' : [600],
    'learning_rate' : [0.4],
    'max_depth' : [6],
    'booster' : ['gbtree'],
    'n_jobs' : [-1],
    'objective' : ['binary:logistic']
}
random_cv = RandomizedSearchCV(estimator = model ,
                               param_distributions = parameters,
                               cv = 5,n_iter=50,scoring=f1,verbose=5,return_train_score=True)
random_cv.fit(X,y)
random_cv.best_estimator_
predictions = random_cv.predict(data_clean_test1)
xgbpred = pd.DataFrame(predictions,columns=['target'])
xgbpred.insert(0,'id',data_test['id'])
xgbpred.to_csv("MysubmissionXGB.csv",index=False)
model = CatBoostClassifier()
print(select_model(X,y,model))
model = LGBMClassifier(n_estimators=300)
print(select_model(X,y,model))
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing import text, sequence
from keras import utils
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras import regularizers

model = Sequential()
model.add(Dense(512, activation='relu', input_dim=(12359),
                kernel_regularizer=regularizers.l2(0.01),
                activity_regularizer=regularizers.l1(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=10**-8, decay=0.0001, amsgrad=False)
model.compile(optimizer= adam,
              loss='binary_crossentropy',
              metrics=['accuracy'])
print(model.summary())
hist = model.fit(X, y,
                    batch_size=16,
                    epochs=20,
                    verbose=1)
%matplotlib inline

history = pd.DataFrame(hist.history)
plt.figure(figsize=(12,12))
plt.plot(history["loss"],label='Train Loss')
plt.plot(history["val_loss"],label='Validation Loss')
plt.title("Loss as function of epoch");
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
predictions_val = model.predict(data_clean_test1)
predictions_val = np.where(predictions_val>0.5, 1, 0)
df_predictions = pd.DataFrame(predictions_val,columns=['target'])
df_predictions.insert(0,'id',data_test['id'])
df_predictions.to_csv("MysubmissionDNN.csv",index=False)
from nltk.tokenize import word_tokenize
import gensim
from tqdm.notebook import tqdm
import pandas as pd
import numpy as np
from nltk.corpus import stopwords

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Model , Sequential

from tensorflow.keras.layers import Embedding , LSTM , Dense , SpatialDropout1D , Dropout
from tensorflow.keras.initializers import Constant 
from tensorflow.keras.optimizers import Adam
data_train = pd.read_csv('../input/nlp-getting-started/train.csv')
data_test = pd.read_csv('../input/nlp-getting-started/test.csv')
sub = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
data_train.drop(['id','location'],axis=1,inplace=True)
data_test.drop(['id','location'],axis=1,inplace=True)
data_train.head()
data_train['text'] = data_train['text'].map(round1)
data_test['text'] = data_test['text'].map(round1)
data_train['text_tokenized']= data_train['text'].map(lambda x : word_tokenize(x))
data_test['text_tokenized']= data_test['text'].map(lambda x : word_tokenize(x))
data_train['text_tokenized']=data_train['text_tokenized'].map(lambda x: convert_abbrev(x))
data_test['text_tokenized']=data_test['text_tokenized'].map(lambda x: convert_abbrev(x))
data_train['text']=data_train['text_tokenized'].map(lambda x : ' '.join(x))
data_test['text']=data_test['text_tokenized'].map(lambda x : ' '.join(x))
def remove_URL(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)

def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text) 

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punct(text):
    table=str.maketrans('','',string.punctuation)
    return text.translate(table)
data_train['text'] = data_train['text'].map(lambda x:remove_URL(x))
data_train['text'] = data_train['text'].map(lambda x:remove_emoji(x))
data_train['text'] = data_train['text'].map(lambda x:remove_punct(x))
data_train['text'] = data_train['text'].map(lambda x:remove_html(x))
data_test['text'] = data_test['text'].map(lambda x:remove_URL(x))
data_test['text'] = data_test['text'].map(lambda x:remove_emoji(x))
data_test['text'] = data_test['text'].map(lambda x:remove_punct(x))
data_test['text'] = data_test['text'].map(lambda x:remove_html(x))
df = pd.concat([data_train,data_test])
def create_corpus(df) :
    corpus = []
    for tweet in tqdm(df['text']) :
        words = [word.lower() for word in word_tokenize(tweet) ]
        corpus.append(words)
    return corpus    
corpus = create_corpus(df)
embedding_dict = {}

with open('../input/glove-global-vectors-for-word-representation/glove.6B.100d.txt','r') as f :
    for line in f :
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:],'float32')
        embedding_dict[word] = vectors
f.close()
MAX_LEN = 50
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
sequences = tokenizer.texts_to_sequences(corpus)

tweet_pad = pad_sequences(sequences , maxlen=MAX_LEN,truncating = 'post' , padding='post')
word_index = tokenizer.word_index
print('Number of unique words :',len(word_index))
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words,100))

for word , i in tqdm(word_index.items()) :
    if i < num_words :
        emb_vec = embedding_dict.get(word)
        if emb_vec is not None :
            embedding_matrix[i] = emb_vec
model = Sequential()

embedding = Embedding(num_words,100,embeddings_initializer=Constant(embedding_matrix),input_length=MAX_LEN , trainable=False)
model.add(embedding)
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))

optimizer = Adam(lr=3e-4)

model.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
train = tweet_pad[:data_train.shape[0],:]
test = tweet_pad[data_train.shape[0]:,:]
history = model.fit(train,data_train['target'],epochs=10,batch_size=4)
predictions_glove = model.predict(test)
sub['target'] = predictions_glove.round().astype(int)
sub.to_csv('submission_Glove.csv',index=False)
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ModelCheckpoint 

import tensorflow_hub as hub
import tokenization
def bert_encoder(texts,tokenizer,max_len=512) :
    
    all_tokens = [] 
    all_masks = []
    all_segments = []
    
    for text in texts :
        
        text = tokenizer.tokenize(text)
        text = text[:max_len-2] # so that we can add cls and sep tokens
        input_sequence = ["[CLS]"]+text+["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        
        pad_mask = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_mask)
        all_segments.append(segment_ids)
        
        
    return np.array(all_tokens) , np.array(all_masks) , np.array(all_segments)
def bert_model(bert_layer , max_len=512) :
    
    input_word_ids = Input(shape=(max_len,) ,dtype=tf.int32 , name = 'input_word_ids' )
    input_word_masks = Input(shape=(max_len,),dtype=tf.int32,name='input_mask')
    input_word_segments = Input(shape=(max_len,),dtype=tf.int32,name='input_segments')
    
    _,sequence_outputs = bert_layer([input_word_ids,input_word_masks,input_word_segments])
    
    
    clf_output = sequence_outputs[:,0,:]
    dense_layer1 = Dense(256,activation='relu')(clf_output)
    dense_layer1 = Dropout(0.3)(dense_layer1)
    
    if Dropout_num == 0 :
        out = Dense(1,activation = 'sigmoid')(dense_layer1)
    else :
        X = Dropout(Dropout_num)(dense_layer1)
        out = Dense(1, activation='sigmoid')(X)
        
    model = Model(inputs = [input_word_ids,input_word_masks,input_word_segments] , outputs = out)   
    
    model.compile(optimizer=Adam(lr=learning_rate),loss='binary_crossentropy',metrics=['accuracy'])
    
    return model
#https://tfhub.dev/tensorflow/albert_en_xxlarge/1     #
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"   
bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_input = bert_encoder(data_train.text.values,tokenizer ,max_len = 160)
test_input = bert_encoder(data_test.text.values,tokenizer, max_len = 160)
train_labels = data_train.target.values
Dropout_num = 0.3
learning_rate = 6e-6

model_bert = bert_model(bert_layer,max_len = 160)
model_bert.summary()
checkpoint = ModelCheckpoint('model_BERT.h5' , monitor='val_loss',save_best_only=True)

epochs = 3
batch_size = 16

history = model_bert.fit(
    train_input,train_labels,
    validation_split = 0.2,
    epochs = epochs,
    batch_size = batch_size,
    callbacks = [checkpoint]

)
model_bert.load_weights('model_BERT.h5')
predictions = model_bert.predict(test_input)
sub['target'] = predictions.round().astype(int)
sub.to_csv('submission_bert.csv',index=False)
from transformers import TFAutoModel, AutoTokenizer

