# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# From Scratch
import nltk
dir(nltk)
from nltk.corpus import stopwords

stopwords.words('english')[0:500:25]
rawData=open('/kaggle/input/nlp-data-set/SMSSpamCollection.tsv').read() 

rawData[0:500]
parsedData=rawData.replace("\t", "\n").split("\n")

parsedData[0:5]
label_list=parsedData[0::2]

text_list=parsedData[1::2]
print(label_list[0:5])

print(text_list[0:5])
print(len(label_list))

print(len(text_list))
print(label_list[-5:])
fullCorpus=pd.DataFrame({'label' : label_list[:-1], 'body_list' : text_list})

fullCorpus.head()
dataset=pd.read_csv('/kaggle/input/nlp-data-set/SMSSpamCollection.tsv' , sep='\t', header = None)

dataset.head()
dataset.columns=['label', 'body_list']

dataset.head()
# shape #5568

print("Input data has {} rows and {} columns".format(len(fullCorpus), len(fullCorpus.columns)))
# how many ham and spam

print("Out of the {} rows, {} are spam, {} are ham". format(len(fullCorpus), 

                                                            len(fullCorpus[fullCorpus['label']=='spam']),

                                                            len(fullCorpus[fullCorpus['label']=='ham'])))
# missing data

print("Number of missing label {}".format(fullCorpus['label'].isnull().sum()))

print("Number of missing text {}".format(fullCorpus['body_list'].isnull().sum()))
import re
re_test = 'This is a made up string to test 2 different regex method'

re_test_messy =  'This is     a made up      string to test 2       different regex method'

re_test_messy1 = 'This-is-a-made/up.string*to>>>>test----2""""""different-regex-method'
# splitting a sentence into a list of words

# 1st method

re.split('\s', re_test)
re.split('\s', re_test_messy)
re.split('\s+', re_test_messy)
re.split('\s+', re_test_messy1)
re.split('\W+', re_test_messy1)
# Second method

re.findall('\S+', re_test_messy)
re.findall('\S+', re_test)
re.findall('\S+', re_test_messy1)
re.findall('\w+', re_test_messy1)
pep8_test = 'I try to follow PEP8 guidelines'

pep7_test = 'I try to follow PEP7 guidelines'

peep8_test = 'I try to follow PEEP8 guidelines'
re.findall('[a-z]+' , pep8_test )
re.findall('[A-Z]+' , pep8_test )
re.findall('[A-Z0-9]+' , pep8_test )
re.findall('[A-Z]+[0-9]+' , pep8_test )
re.findall('[A-Z]+[0-9]+' , pep7_test )
re.findall('[A-Z]+[0-9]+' , peep8_test )
re.sub('[A-Z]+[0-9]+', 'PEP8 Python Styleguide',pep8_test )
re.sub('[A-Z]+[0-9]+', 'PEP8 Python Styleguide',pep7_test )
re.sub('[A-Z]+[0-9]+', 'PEP8 Python Styleguide',peep8_test )
pd.set_option('display.max_colwidth', 100)

data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)

data.columns = ['label', 'body_text']

data.head()
# How does cleaned up version look like

data_cleaned=pd.read_csv("/kaggle/input/cleaned-data/SMSSpamCollection_cleaned.tsv" , sep='\t')

data_cleaned
import string

string.punctuation
"I like NLP." == "I like NLP"
def remove_punc(text):

    text_no_punc = [char for char in text if char not in string.punctuation]

    return text_no_punc

data['body_text_clean']=data['body_text'].apply(lambda x : remove_punc(x))

data.head()
def remove_punc(text):

    text_no_punc = "".join([char for char in text if char not in string.punctuation])

    return text_no_punc

data['body_text_clean']=data['body_text'].apply(lambda x : remove_punc(x))

data.head()
def tokenize(text):

    tokens = re.split("\W+", text)

    return tokens

data['body_text_tokenize']=data['body_text_clean'].apply(lambda x : tokenize(x.lower()))

data.head()
'NLP'== 'nlp'
stopword = nltk.corpus.stopwords.words('english')
def remove_stopwords(tokenized_list):

    text = [ word for word in tokenized_list if word not in stopword]

    return text

data['body_text_nostop']=data['body_text_tokenize'].apply(lambda x : remove_stopwords(x))

data.head()
ps=nltk.PorterStemmer()
dir(ps)
print(ps.stem('grows'))

print(ps.stem('growing'))

print(ps.stem('grow'))
print(ps.stem('run'))

print(ps.stem('running'))

print(ps.stem('runner'))
#import re

#import string

pd.set_option('display.max_colwidth', 100)



stopwords=nltk.corpus.stopwords.words('english')

data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)

data.columns = ['label', 'body_text']

data.head()
def clean_text(text):

    text = "".join([word for word in text if word not in string.punctuation])

    tokens=re.split('\W+', text)

    text = [word for word in tokens if word not in stopwords]

    return text

data['body_text_nostop']=data['body_text'].apply(lambda x : clean_text(x.lower()))

data.head()
def stemming(tokenized_text):

    text = [ps.stem(word) for word in tokenized_text]

    return text

data['body_text_stemmed']=data['body_text_nostop'].apply(lambda x : stemming(x))

data.head()
#ps=nltk.PorterStemmer()

wn=nltk.WordNetLemmatizer()

#import re

#import string

#stopwords=nltk.corpus.stopwords.words('english')
dir(wn)
print(ps.stem('meanness'))

print(ps.stem('meaning'))
print(wn.lemmatize('meanness'))

print(wn.lemmatize('meaning'))
print(ps.stem('goose'))

print(ps.stem('geese'))
print(wn.lemmatize('goose'))

print(wn.lemmatize('geese'))
data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)

data.columns = ['label', 'body_text']

data.head()
def clean_text(text):

    text = "".join([word for word in text if word not in string.punctuation])

    tokens=re.split('\W+', text)

    text = [word for word in tokens if word not in stopwords]

    return text

data['body_text_nostop']=data['body_text'].apply(lambda x : clean_text(x.lower()))

data.head()
def lemmatizing(tokenized_text):

    text = [wn.lemmatize(word) for word in tokenized_text]

    return text

data['body_text_lemmatized']=data['body_text_nostop'].apply(lambda x : lemmatizing(x))

data.head()
# import string

# import re

#import nltk

pd.set_option('display.max_colwidth', 100)



stopwords=nltk.corpus.stopwords.words('english')

ps=nltk.PorterStemmer()

    

data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)

data.columns = ['label', 'body_text']

data.head()
def clean_text(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    tokens=re.split('\W+', text)

    text = [ps.stem(word) for word in tokens if word not in stopwords]

    return text

#data['body_text_nostop']=data['body_text'].apply(lambda x : clean_text(x.lower()))

#data.head()
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(analyzer = clean_text)

x_counts=count_vect.fit_transform(data['body_text'])

print(x_counts.shape)

print(count_vect.get_feature_names())
data_sample=data[0:20]

count_vect_sample = CountVectorizer(analyzer = clean_text)

x_counts_sample=count_vect_sample.fit_transform(data_sample['body_text'])

print(x_counts_sample.shape)

print(count_vect_sample.get_feature_names())
x_counts_sample
df=pd.DataFrame(x_counts_sample.toarray())

df.head()
df.columns = count_vect_sample.get_feature_names()

df.head()
# import string

# import re

#import nltk

pd.set_option('display.max_colwidth', 100)



stopwords=nltk.corpus.stopwords.words('english')

ps=nltk.PorterStemmer()

    

data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)

data.columns = ['label', 'body_text']

data.head()
def clean_text(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    tokens=re.split('\W+', text)

    text = " ".join([ps.stem(word) for word in tokens if word not in stopwords])

    return text

data['cleaned_text']=data['body_text'].apply(lambda x : clean_text(x))

data.head()
from sklearn.feature_extraction.text import CountVectorizer

ngram_vect = CountVectorizer(ngram_range=(2,2)) # 1,2,3 = unigram, bigram, trigram

x_counts=ngram_vect.fit_transform(data['cleaned_text'])

print(x_counts.shape)

print(ngram_vect.get_feature_names())
data_sample= data[0:20]

ngram_vect_sample = CountVectorizer(ngram_range=(2,2)) # 1,2,3 = unigram, bigram, trigram

x_counts_sample=ngram_vect_sample.fit_transform(data_sample['cleaned_text'])

print(x_counts_sample.shape)

print(ngram_vect_sample.get_feature_names())
df=pd.DataFrame(x_counts_sample.toarray())

df.columns = ngram_vect_sample.get_feature_names()

df.head()
# import string

# import re

#import nltk

pd.set_option('display.max_colwidth', 100)



stopwords=nltk.corpus.stopwords.words('english')

ps=nltk.PorterStemmer()

    

data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)

data.columns = ['label', 'body_text']

data.head() 
def clean_text(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    tokens=re.split('\W+', text)

    text = [ps.stem(word) for word in tokens if word not in stopwords]

    return text
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect=TfidfVectorizer(analyzer=clean_text)

x_tfidf=tfidf_vect.fit_transform(data['body_text'])

print(x_tfidf.shape)

print(tfidf_vect.get_feature_names())
data_sample= data[0:20]

tfidf_vect_sample=TfidfVectorizer(analyzer=clean_text)

x_tfidf_sample=tfidf_vect_sample.fit_transform(data_sample['body_text'])

print(x_tfidf_sample.shape)

print(tfidf_vect_sample.get_feature_names())
df=pd.DataFrame(x_tfidf_sample.toarray())

df.columns = tfidf_vect_sample.get_feature_names()

df.head()
data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)

data.columns = ['label', 'body_text']

data.head()
# Create feature for text message length

data['body_len'] = data['body_text'].apply(lambda x : len(x) - x.count(" "))

data.head()
# Create feature for % of text that is  punctuation

import string

def count_punc(text):

    count = sum([1 for char in text if char in string.punctuation])

    return round(count/(len(text) - text.count(" ")),3)*100

data['punc%']= data['body_text'].apply(lambda x : count_punc(x))

data.head()
# Evalute new features

import matplotlib.pyplot as plt

%matplotlib inline
bins= np.linspace(0,200,40)

plt.hist(data[data['label'] == 'spam']['body_len'], bins, alpha=0.5, normed=True, label='spam')

plt.hist(data[data['label'] == 'ham']['body_len'], bins, alpha=0.5, normed=True, label='ham')

plt.legend(loc='best')

plt.show()
bins= np.linspace(0,50,40)

plt.hist(data[data['label'] == 'spam']['punc%'], bins, alpha=0.5, normed=True, label='spam')

plt.hist(data[data['label'] == 'ham']['punc%'], bins, alpha=0.5, normed=True, label='ham')

plt.legend(loc='best')

plt.show()
bins= np.linspace(0,200,40)

plt.hist(data['body_len'], bins)

plt.title('Body Length Distribution')

plt.show()
bins= np.linspace(0,50,40)

plt.hist(data['punc%'], bins)

plt.title('Punctuation Length Distribution')

plt.show()
for i in [1,2,3,4,5]:

    plt.hist((data['punc%']) ** (1/i), bins=40)

    plt.title('Transformation : 1/{}'.format(str(i)))

    plt.show()
import nltk

import re 

import string

from sklearn.feature_extraction.text import TfidfVectorizer



stopwords= nltk.corpus.stopwords.words('english')

data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)

data.columns = ['label', 'body_text']



def count_punc(text):

    count = sum([1 for char in text if char in string.punctuation])

    return round(count/(len(text) - text.count(" ")),3)*100



data['body_len']= data['body_text'].apply(lambda x : len(x) - x.count(" "))

data['punc%']= data['body_text'].apply(lambda x : count_punc(x))





def clean_text(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    tokens= re.split('\W+', text)

    text = [ps.stem(word) for word in tokens if word not in stopwords]

    return text



tfidf_vect = TfidfVectorizer(analyzer=clean_text)

x_tfidf = tfidf_vect.fit_transform(data['body_text'])



x_features = pd.concat([data['body_len'], data['punc%'], pd.DataFrame(x_tfidf.toarray())], axis =1)

x_features.head()
from sklearn.ensemble import RandomForestClassifier
print(dir(RandomForestClassifier))

print(RandomForestClassifier())
from sklearn.model_selection import KFold, cross_val_score
rf = RandomForestClassifier(n_jobs=-1) #n_jobs will execute all the decesion tree parallel

K_Fold = KFold(n_splits=5)

cross_val_score(rf, x_features, data['label'], cv=K_Fold, scoring = 'accuracy', n_jobs=-1)
import nltk

import re 

import string

from sklearn.feature_extraction.text import TfidfVectorizer



stopwords= nltk.corpus.stopwords.words('english')

data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)

data.columns = ['label', 'body_text']



def count_punc(text):

    count = sum([1 for char in text if char in string.punctuation])

    return round(count/(len(text) - text.count(" ")),3)*100



data['body_len']= data['body_text'].apply(lambda x : len(x) - x.count(" "))

data['punc%']= data['body_text'].apply(lambda x : count_punc(x))





def clean_text(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    tokens= re.split('\W+', text)

    text = [ps.stem(word) for word in tokens if word not in stopwords]

    return text



tfidf_vect = TfidfVectorizer(analyzer=clean_text)

x_tfidf = tfidf_vect.fit_transform(data['body_text'])



x_features = pd.concat([data['body_len'], data['punc%'], pd.DataFrame(x_tfidf.toarray())], axis =1)

x_features.head()
from sklearn.metrics import precision_recall_fscore_support as score 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_features, data['label'], test_size = 0.2)
rf=RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1)

rf_model = rf.fit(x_train, y_train)
sorted(zip(rf_model.feature_importances_, x_train.columns), reverse=True)[0:10]
y_pred=rf_model.predict(x_test)

precision, recall, fscore, support = score(y_test, y_pred, pos_label = 'spam', average='binary')
print('precision: {} / recall: {} / accuracy: {}'. format(round(precision, 3), round(recall, 3), 

                                                         round((y_pred==y_test).sum() / len(y_pred),3)))
import nltk

import re 

import string

from sklearn.feature_extraction.text import TfidfVectorizer



stopwords= nltk.corpus.stopwords.words('english')

data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)

data.columns = ['label', 'body_text']



def count_punc(text):

    count = sum([1 for char in text if char in string.punctuation])

    return round(count/(len(text) - text.count(" ")),3)*100



data['body_len']= data['body_text'].apply(lambda x : len(x) - x.count(" "))

data['punc%']= data['body_text'].apply(lambda x : count_punc(x))





def clean_text(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    tokens= re.split('\W+', text)

    text = [ps.stem(word) for word in tokens if word not in stopwords]

    return text



tfidf_vect = TfidfVectorizer(analyzer=clean_text)

x_tfidf = tfidf_vect.fit_transform(data['body_text'])



x_features = pd.concat([data['body_len'], data['punc%'], pd.DataFrame(x_tfidf.toarray())], axis =1)

x_features.head()
from sklearn.metrics import precision_recall_fscore_support as score 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_features, data['label'], test_size = 0.2)
def train_RF(n_est, depth):

    rf=RandomForestClassifier(n_estimators=n_est, max_depth=depth, n_jobs=-1)

    rf_model=rf.fit(x_train, y_train)

    y_pred=rf_model.predict(x_test)

    precision, recall, fscore, support = score(y_test, y_pred, pos_label='spam', average='binary')

    print('Est: {} / Depth: {} ---- Precision : {} / Recall : {} / Accuracy : {}'.format(

         n_est, depth, round(precision,3), round(recall, 3), round((y_pred==y_test).sum() / len(y_pred),3)))
for n_est in [10, 50, 100]:

    for depth in [10 , 20 , 30, None]:

        train_RF(n_est, depth)
import nltk

import re 

import string

from sklearn.feature_extraction.text import TfidfVectorizer



stopwords= nltk.corpus.stopwords.words('english')

data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)

data.columns = ['label', 'body_text']



def count_punc(text):

    count = sum([1 for char in text if char in string.punctuation])

    return round(count/(len(text) - text.count(" ")),3)*100



data['body_len']= data['body_text'].apply(lambda x : len(x) - x.count(" "))

data['punc%']= data['body_text'].apply(lambda x : count_punc(x))





def clean_text(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    tokens= re.split('\W+', text)

    text = [ps.stem(word) for word in tokens if word not in stopwords]

    return text



tfidf_vect = TfidfVectorizer(analyzer=clean_text)

x_tfidf = tfidf_vect.fit_transform(data['body_text'])

x_tfidf_feat = pd.concat([data['body_len'], data['punc%'], pd.DataFrame(x_tfidf.toarray())], axis =1)



count_vect = CountVectorizer(analyzer=clean_text)

x_count = count_vect.fit_transform(data['body_text'])

x_count_feat = pd.concat([data['body_len'], data['punc%'], pd.DataFrame(x_count.toarray())], axis =1)



x_count_feat.head()
from sklearn.model_selection import GridSearchCV
rf=RandomForestClassifier()

param = {'n_estimators' : [10, 150, 130],

        'max_depth' : [30, 60, 90 , None ] }

gs =GridSearchCV(rf, param, cv=5, n_jobs = -1 )

gs_fit=gs.fit(x_tfidf_feat, data['label'])

pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]

    
rf=RandomForestClassifier()

param = {'n_estimators' : [10, 150, 130],

        'max_depth' : [30, 60, 90 , None ] }

gs =GridSearchCV(rf, param, cv=5, n_jobs = -1 )

gs_fit=gs.fit(x_count_feat, data['label'])

pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]
import nltk

import re 

import string

from sklearn.feature_extraction.text import TfidfVectorizer



stopwords= nltk.corpus.stopwords.words('english')

data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)

data.columns = ['label', 'body_text']



def count_punc(text):

    count = sum([1 for char in text if char in string.punctuation])

    return round(count/(len(text) - text.count(" ")),3)*100



data['body_len']= data['body_text'].apply(lambda x : len(x) - x.count(" "))

data['punc%']= data['body_text'].apply(lambda x : count_punc(x))





def clean_text(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    tokens= re.split('\W+', text)

    text = [ps.stem(word) for word in tokens if word not in stopwords]

    return text



tfidf_vect = TfidfVectorizer(analyzer=clean_text)

x_tfidf = tfidf_vect.fit_transform(data['body_text'])



x_features = pd.concat([data['body_len'], data['punc%'], pd.DataFrame(x_tfidf.toarray())], axis =1)

x_features.head()
from sklearn.ensemble import GradientBoostingClassifier
print(dir(GradientBoostingClassifier))

print(GradientBoostingClassifier())
def train_GB(est, max_depth, lr):

    GB = GradientBoostingClassifier(n_estimators=est, max_depth=max_depth, learning_rate=lr)

    GB_model = GB.fit(x_train, y_train)

    y_pred = GB_model.predict(x_test)

    precision, recall, fscore, support = score(y_test, y_pred, pos_label = 'spam', average = 'binary')

    print('Est : {} / Max_Depth : {} / LR : {} ----- Precision : {} / Recall : {} /  Accuracy : {}'.format(est, max_depth, lr, 

                                                                     round(precision,3),round(recall,3),round((y_pred==y_test).sum() / len(y_pred),3)))

    
for n_est in [50,100,150]:

    for max_depth in [3, 7 ,11, 15]:

        for lr in [0.01, 0.1, 1]:

            train_GB(n_est,max_depth,lr)
import nltk

import re 

import string

from sklearn.feature_extraction.text import TfidfVectorizer



stopwords= nltk.corpus.stopwords.words('english')

data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)

data.columns = ['label', 'body_text']



def count_punc(text):

    count = sum([1 for char in text if char in string.punctuation])

    return round(count/(len(text) - text.count(" ")),3)*100



data['body_len']= data['body_text'].apply(lambda x : len(x) - x.count(" "))

data['punc%']= data['body_text'].apply(lambda x : count_punc(x))





def clean_text(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    tokens= re.split('\W+', text)

    text = [ps.stem(word) for word in tokens if word not in stopwords]

    return text



tfidf_vect = TfidfVectorizer(analyzer=clean_text)

x_tfidf = tfidf_vect.fit_transform(data['body_text'])

x_tfidf_feat = pd.concat([data['body_len'], data['punc%'], pd.DataFrame(x_tfidf.toarray())], axis =1)



count_vect = CountVectorizer(analyzer=clean_text)

x_count = count_vect.fit_transform(data['body_text'])

x_count_feat = pd.concat([data['body_len'], data['punc%'], pd.DataFrame(x_count.toarray())], axis =1)



x_count_feat.head()
gb = GradientBoostingClassifier()

param = {'n_estimators' : [100, 150], 'max_depth' : [7, 11, 15], 'learning_rate' : [0.1]}



gs= GridSearchCV(gb. param, cv=5, n_jobs= -1)

cv_fit = gs.fit(x_tfidf_feat , data['label'])

pd.DataFrame(cv_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]
gb = GradientBoostingClassifier()

param = {'n_estimators' : [100, 150], 'max_depth' : [7, 11, 15], 'learning_rate' : [0.1]}



gs= GridSearchCV(gb. param, cv=5, n_jobs= -1)

cv_fit = gs.fit(x_count_feat , data['label'])

pd.DataFrame(cv_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]
import nltk

import re 

import string

from sklearn.feature_extraction.text import TfidfVectorizer



stopwords= nltk.corpus.stopwords.words('english')

data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)

data.columns = ['label', 'body_text']



def count_punc(text):

    count = sum([1 for char in text if char in string.punctuation])

    return round(count/(len(text) - text.count(" ")),3)*100



data['body_len']= data['body_text'].apply(lambda x : len(x) - x.count(" "))

data['punc%']= data['body_text'].apply(lambda x : count_punc(x))





def clean_text(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    tokens= re.split('\W+', text)

    text = [ps.stem(word) for word in tokens if word not in stopwords]

    return text
x_train, x_test, y_train, y_test = train_test_split(data[['body_text','body_len','punc%']], data['label'] ,test_size = 0.2)
tfidf_vect = TfidfVectorizer(analyzer=clean_text)

tfidf_vect_fit = tfidf_vect.fit(x_train['body_text'])



tfidf_train = tfidf_vect_fit.transform(x_train['body_text'])

tfidf_test = tfidf_vect_fit.transform(x_test['body_text'])



x_train_vect = pd.concat([x_train[['body_len','punc%']].reset_index(drop=True),

         pd.DataFrame(tfidf_train.toarray())], axis = 1)



x_test_vect = pd.concat([x_test[['body_len','punc%']].reset_index(drop=True),

         pd.DataFrame(tfidf_test.toarray())], axis = 1)

x_train_vect.head()
import time
rf=RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)



start=time.time()

rf_model=rf.fit(x_train_vect, y_train)

end=time.time()

fit_time = (end-start)



start=time.time()

y_pred = rf_model.predict(x_test_vect)

end=time.time()

pred_time = (end-start)



precision, recall, fscore, support = score(y_test, y_pred, pos_label='spam', average='binary')

print('Fit_time: {} / Predict_time : {} / Precision : {} / Recall : {} / Accuracy : {}'.format(round(fit_time,3),round(pred_time,3),round(precision,3), round(recall, 3), 

                                                                round((y_pred==y_test).sum() / len(y_pred),3)))



gb = GradientBoostingClassifier(n_estimators=150, max_depth=11)

start=time.time()

gb_model=gb.fit(x_train_vect, y_train)

end=time.time()

fit_time = (end-start)



start=time.time()

y_pred = gb_model.predict(x_test_vect)

end=time.time()

pred_time = (end-start)



precision, recall, fscore, support = score(y_test, y_pred, pos_label='spam', average='binary')

print('Fit_time: {} / Predict_time : {} / Precision : {} / Recall : {} / Accuracy : {}'.format(round(fit_time,3),round(pred_time,3),round(precision,3), round(recall, 3), 

                                                                round((y_pred==y_test).sum() / len(y_pred),3)))
