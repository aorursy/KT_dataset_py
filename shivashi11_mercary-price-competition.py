%matplotlib inline

import warnings

warnings.filterwarnings("ignore")







import sqlite3

import pandas as pd

import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from nltk.stem.porter import PorterStemmer



import re

# Tutorial about Python regular expressions: https://pymotw.com/2/re/

import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer



from gensim.models import Word2Vec

from gensim.models import KeyedVectors

import pickle

from collections import Counter



from tqdm import tqdm

import os



from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# word cloud library

from wordcloud import WordCloud



# matplotlib

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir())
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_data = pd.read_csv("/kaggle/input/mercari-price-suggestion-challenge/train.tsv", sep='\t') 

print("Shape of train_data",train_data.shape)



test_data = pd.read_csv("/kaggle/input/mercari-price-suggestion-challenge/test_stg2.tsv", sep='\t') 

train_data.head(3)



train_data.describe().T
test_data.shape
sum(train_data['price']==0)
### 874 items are free
plt.subplot(1,2,1)

(train_data['price']).hist(bins=50, figsize=(20,10), range=[0,250], edgecolor = 'white',grid=False)

plt.xlabel('price')

plt.ylabel('frequency')



plt.subplot(1,2,2)

np.log(train_data['price']).hist(bins=50, figsize=(20,10), range=[0,7], edgecolor = 'white', grid=False)

plt.xlabel('log of price')

plt.ylabel('frequency')



plt.plot();

## thus ditribution of price is log-normal
# Shipping
sns.countplot(train_data['shipping'])

plt.title('train shipping')
sns.countplot(test_data['shipping'])

plt.title('test shipping')
### thus we can say train and test data for shipping has same distribution
price_ship0 = train_data[train_data['shipping']==1]['price'].values

price_ship1 = train_data[train_data['shipping']==0]['price'].values
plt.figure(figsize=(15,8), edgecolor = 'black')

plt.hist(price_ship0, bins = 50, range=[0,250], alpha = 1, color = 'green', label = 'ship = 0')

plt.hist(price_ship1, bins = 50, range=[0,250],alpha = 0.7, color = 'blue', label = 'ship = 1')

plt.plot();

### generally the higher priced elements has shipping 1 i.e. paid by seller
# item condition
sns.countplot(train_data['item_condition_id'])
sns.countplot(test_data['item_condition_id'])
sns.jointplot('item_condition_id','price',train_data, ratio = 3)
### thus as item condition id increases price generally decreases
# brand name
train_data['brand_name'].nunique()
sum(train_data['brand_name'].isnull())
train_data.fillna('Nobrand',inplace = True)
plt.figure(figsize = (15,7))

sns.barplot(list(train_data['brand_name'].value_counts()[:20].index),train_data['brand_name'].value_counts()[:20], )

plt.xticks(rotation = 45)

plt.title("Brands with their number of products")

plt.plot();
### these are costlier brands
# category_names
def sep_in_cat(x):

    try:

        if(len(x.split('/'))<3):

            return (x.split('/')[0],'','')

        return x.split('/')

    except:

        return ("No label","No label","No label")
train_data['cat1'], train_data['cat2'], train_data['cat3'] = zip(*train_data['category_name'].apply(lambda x: sep_in_cat(x)))
test_data['cat1'], test_data['cat2'], test_data['cat3'] = zip(*test_data['category_name'].apply(lambda x: sep_in_cat(x)))
plt.figure(figsize = (15,7))

sns.barplot(list(train_data['cat1'].value_counts()[:20].index),train_data['cat1'].value_counts()[:20], )

plt.xticks(rotation = 45)

plt.plot();
### approx 6k points have no value
train_data['cat2'].nunique()
plt.figure(figsize = (15,7))

sns.barplot(list(train_data['cat2'].value_counts()[:20].index),train_data['cat2'].value_counts()[:20], )

plt.xticks(rotation = 45)

plt.plot();
train_data['cat3'].nunique()
plt.figure(figsize = (15,7))

sns.barplot(list(train_data['cat3'].value_counts()[:20].index),train_data['cat3'].value_counts()[:20], )

plt.xticks(rotation = 45)

plt.plot();
# Preprocessing
import re



def decontracted(phrase):

    # specific

    phrase = str(phrase)

    phrase = re.sub(r"won't", "will not", phrase)

    phrase = re.sub(r"can\'t", "can not", phrase)



    # general

    phrase = re.sub(r"n\'t", " not", phrase)

    phrase = re.sub(r"\'re", " are", phrase)

    phrase = re.sub(r"\'s", " is", phrase)

    phrase = re.sub(r"\'d", " would", phrase)

    phrase = re.sub(r"\'ll", " will", phrase)

    phrase = re.sub(r"\'t", " not", phrase)

    phrase = re.sub(r"\'ve", " have", phrase)

    phrase = re.sub(r"\'m", " am", phrase)

    return phrase
stopwords= ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\

            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \

            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\

            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \

            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \

            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \

            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\

            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\

            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\

            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \

            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\

            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\

            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \

            'won', "won't", 'wouldn', "wouldn't"]
from tqdm import tqdm



# tqdm is for printing the status bar

def clean_para(row):

    preprocessed_item_description = []

    sentance = row

    sent = decontracted(sentance)

    sent = sent.replace('\\r', ' ')

    sent = sent.replace('\\"', ' ')

    sent = sent.replace('\\n', ' ')

    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)

    # https://gist.github.com/sebleier/554280

    sent = ' '.join(e for e in sent.split() if e not in stopwords)

    preprocessed_item_description.append(sent.lower().strip())

    return preprocessed_item_description[0]

train_data['item_description'] = train_data['item_description'].apply(lambda x:clean_para(x))

test_data['item_description'] = test_data['item_description'].apply(lambda x:clean_para(x))
train_data['name'] = train_data['name'].apply(lambda x:clean_para(x))

test_data['name'] = test_data['name'].apply(lambda x:clean_para(x))
def clean_simple(i):

    temp = ""

    # consider we have text like this "Math & Science, Warmth, Care & Hunger"

    for j in i.split(','): # it will split it in three parts ["Math & Science", "Warmth", "Care & Hunger"]

        if 'The' in j.split(): # this will split each of the catogory based on space "Math & Science"=> "Math","&", "Science"

            j=j.replace('The','') # if we have the words "The" we are going to replace it with ''(i.e removing 'The')

        j = j.replace(' ','') # we are placeing all the ' '(space) with ''(empty) ex:"Math & Science"=>"Math&Science"

        temp +=j.strip()+" "#" abc ".strip() will return "abc", remove the trailing spaces

        temp = temp.replace('&','_')

        temp = temp.replace('-','_')

        temp = temp.replace('+','_')

        

    return temp.strip()
train_data['cat1'] = train_data['cat1'].apply(lambda x:clean_simple(x))

train_data['cat2'] = train_data['cat2'].apply(lambda x:clean_simple(x))

train_data['cat3'] = train_data['cat3'].apply(lambda x:clean_simple(x))



test_data['cat1'] = test_data['cat1'].apply(lambda x:clean_simple(x))

test_data['cat2'] = test_data['cat2'].apply(lambda x:clean_simple(x))

test_data['cat3'] = test_data['cat3'].apply(lambda x:clean_simple(x))



train_data['brand_name'] = train_data['brand_name'].apply(lambda x:clean_simple(str(x)))

test_data['brand_name'] = test_data['brand_name'].apply(lambda x:clean_simple(str(x)))

train_data.drop(columns = ['category_name'], inplace = True)

test_data.drop(columns = ['category_name'], inplace = True)
#import h5py



#train_data.to_hdf("train_data_preprocessed.h5",key="train")



#test_data.to_hdf("test_data_preprocessed.h5",key="test")
train_data.head()
y=train_data['price'].values

train_data.drop(['price'], axis=1, inplace=True)      # drop project is approved columns  



x=train_data
from sklearn.model_selection import train_test_split



x_train,x_cv,y_train,y_cv= train_test_split(x,y,test_size=0.3,random_state=0)



print("Shape of train",x_train.shape,y_train.shape)

print("Shape of cv",x_cv.shape,y_cv.shape)

print("Shape of test",test_data.shape)
## OHE of categorical data
# OHE of subject category

from sklearn.feature_extraction.text import CountVectorizer

vectorizercat1 = CountVectorizer(ngram_range = (1,2),)

vectorizercat1.fit(x_train['cat1'].values) # fit has to happen only on train data





# we use the fitted CountVectorizer to convert the text to vector

x_train_bow_cat1 = vectorizercat1.transform(x_train['cat1'].values)

x_cv_bow_cat_1 = vectorizercat1.transform(x_cv['cat1'].values)

x_test_bow_cat_1 = vectorizercat1.transform(test_data['cat1'].values)



print("After vectorizations")

print(x_train_bow_cat1.shape, y_train.shape)

print(x_cv_bow_cat_1.shape, y_cv.shape)

print(x_test_bow_cat_1.shape)



print("="*100)





vectorizercat2 = CountVectorizer(ngram_range = (1,3),)

vectorizercat2.fit(x_train['cat2'].values) # fit has to happen only on train data





# we use the fitted CountVectorizer to convert the text to vector

x_train_bow_cat2 = vectorizercat2.transform(x_train['cat2'].values)

x_cv_bow_cat2 = vectorizercat2.transform(x_cv['cat2'].values)

x_test_bow_cat2 = vectorizercat2.transform(test_data['cat2'].values)



print("After vectorizations")

print(x_train_bow_cat2.shape, y_train.shape)

print(x_cv_bow_cat2.shape, y_cv.shape)

print(x_test_bow_cat2.shape)



print("="*100)



vectorizercat3 = CountVectorizer(ngram_range = (1,4),)

vectorizercat3.fit(x_train['cat3'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector

x_train_bow_cat3 = vectorizercat3.transform(x_train['cat3'].values)

x_cv_bow_cat3 = vectorizercat3.transform(x_cv['cat3'].values)

x_test_bow_cat3 = vectorizercat3.transform(test_data['cat3'].values)



print("After vectorizations")

print(x_train_bow_cat3.shape, y_train.shape)

print(x_cv_bow_cat3.shape, y_cv.shape)

print(x_test_bow_cat3.shape)



print("="*100)
#brand name

vectorizercat1 = CountVectorizer(ngram_range = (1,2))

vectorizercat1.fit(x_train['brand_name'].values) # fit has to happen only on train data





# we use the fitted CountVectorizer to convert the text to vector

x_train_bow_brand_name = vectorizercat1.transform(x_train['brand_name'].values)

x_cv_bow_brand_name = vectorizercat1.transform(x_cv['brand_name'].values)

x_test_bow_brand_name = vectorizercat1.transform(test_data['brand_name'].values)



print("After vectorizations")

print(x_train_bow_brand_name.shape, y_train.shape)

print(x_cv_bow_brand_name.shape, y_cv.shape)

print(x_test_bow_brand_name.shape)



print("="*100)
## TFIDF
# item description

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer8 = TfidfVectorizer(ngram_range = (1,3), max_features = 80000)



cleaned_item_description_xtr_tfidf = vectorizer8.fit_transform(x_train['item_description'])

print("Shape of matrix after one hot encodig ",cleaned_item_description_xtr_tfidf.shape)

cleaned_item_description_xcv_tfidf = vectorizer8.transform(x_cv['item_description'])

print("Shape of matrix after one hot encodig ",cleaned_item_description_xcv_tfidf.shape)





cleaned_item_description_xtest_tfidf = vectorizer8.transform(test_data['item_description'])

print("Shape of matrix after one hot encodig ",cleaned_item_description_xtest_tfidf.shape)



print("After vectorizations")

print(cleaned_item_description_xtr_tfidf.shape, y_train.shape)

print(cleaned_item_description_xcv_tfidf.shape, y_cv.shape)

print(cleaned_item_description_xtest_tfidf.shape)



print("="*100)
#name

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer9 = TfidfVectorizer(ngram_range = (1,2), max_features = 50000)



clean_names_xtr_tfidf = vectorizer9.fit_transform(x_train['name'])



clean_names_xcv_tfidf = vectorizer9.transform(x_cv['name'])



clean_names_xtest_tfidf = vectorizer9.transform(test_data['name'])



print("After vectorizations")

print(clean_names_xtr_tfidf.shape, y_train.shape)

print(clean_names_xcv_tfidf.shape, y_cv.shape)

print(clean_names_xtest_tfidf.shape)



print("="*100)
# shipping and item_condition

from sklearn.preprocessing import MinMaxScaler # NORMALIZE



mnn=MinMaxScaler()

X_train_std = mnn.fit_transform(x_train[["item_condition_id","shipping" ]])

X_cv_std = mnn.fit_transform(x_cv[[ "item_condition_id","shipping" ]])

X_test_std = mnn.transform(test_data[[ "item_condition_id","shipping" ]])

print(X_train_std.shape)

print(X_cv_std.shape)

print(X_test_std.shape)
from scipy.sparse import hstack

X_train = hstack((x_train_bow_cat1 ,x_train_bow_cat2 ,x_train_bow_cat3 ,x_train_bow_brand_name,cleaned_item_description_xtr_tfidf, clean_names_xtr_tfidf, X_train_std)).tocsr()



X_cv =    hstack((x_cv_bow_cat_1 ,x_cv_bow_cat2 ,x_cv_bow_cat3 ,x_cv_bow_brand_name, cleaned_item_description_xcv_tfidf, clean_names_xcv_tfidf, X_cv_std)).tocsr()



X_test = hstack((x_test_bow_cat_1,x_test_bow_cat2 ,x_test_bow_cat3 ,x_test_bow_brand_name, cleaned_item_description_xtest_tfidf,clean_names_xtest_tfidf, X_test_std)).tocsr()



print("Final Data matrix")

print(X_train.shape, y_train.shape)

print(X_cv.shape, y_cv.shape)

print(X_test.shape)

print("="*100)
def rmsle(real, predicted):

    sum=0.0

    for x in range(len(predicted)):

        if predicted[x]<0 or real[x]<0: #check for negative values

            continue

        p = np.log(predicted[x]+1)

        r = np.log(real[x]+1)

        sum = sum + (p - r)**2

    return (sum/len(predicted))**0.5
from sklearn.linear_model import SGDRegressor



model = SGDRegressor()

model.fit(X_train, y_train)
train_preds = model.predict(X_train)

cv_preds = model.predict(X_cv)

test_preds = model.predict(X_test)
print(rmsle(y_train,train_preds),"    ",rmsle(y_cv,cv_preds))
result = pd.DataFrame({'test_id' : range(0,len(test_preds)),

                       'price' : test_preds})
result.shape




result.to_csv("submission.csv", index = False)