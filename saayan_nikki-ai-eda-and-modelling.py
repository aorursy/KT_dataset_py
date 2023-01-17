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
import warnings

warnings.filterwarnings("ignore")

#Visualisation

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

import seaborn as sns

#nlp

import nltk

import re

from nltk.corpus import stopwords 

stop_words =set (stopwords.words('english'))

from textblob import TextBlob

from sklearn.feature_extraction import text

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer



#ml

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score, classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold
def text_process(data):

    str1=str(data)

    str1 = re.sub(" \d+", "number", str1) #Digits Replacement

    str1 = re.sub('[^\w\s]','', str1)  # Punctuation Removal

    words = re.split("\n|,|;| ",str1) 

    words = [item.lower() for item in words]

    Keys=[word_text for word_text in words if word_text not in stop_words] 

    Keys = ' '.join(Keys).lower()

    return Keys
def wordcloud(text):

    stopwords = set(STOPWORDS)

    #stopwords.update(['NaN'])

    wordcloud = WordCloud(

                              background_color='black',

                              stopwords=stopwords,

                              max_words=300,

                              max_font_size=30, 

                              random_state=42

                             ).generate(" ".join(text))



    print(wordcloud)

    fig = plt.figure(1)

    plt.rcParams['figure.figsize']=(20,20)

    plt.imshow(wordcloud)

    plt.axis('off')
def lemmatize_text(text):

    # Instantiate the Word tokenizer & Word lemmatizer

    w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

    lemmatizer = nltk.stem.WordNetLemmatizer()

    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
def sentiment_analyser(text):

    return text.apply(lambda Text: pd.Series(TextBlob(Text).sentiment.polarity))



train = pd.read_csv("/kaggle/input/nikkiai-np-challenge/train (3) (1) (3) (2).csv")

test = pd.read_csv("/kaggle/input/nikkiai-np-challenge/test (3) (1) (3) (2).csv")
train.head().T
train.info()
train['review']=train['Review Text'].combine_first(train['Review Title'])

train['review']
train.sample(10).T
train['review']=train['review'].apply(text_process)

plt.rcParams['figure.figsize']=(20,20)

wordcloud(train['review'])

train['lem_review']=train['review'].apply(lemmatize_text)
# Applying function to reviews

train['Polarity'] = sentiment_analyser(train['review'])
train["review_len"]= train['review'].apply(len)
g = sns.FacetGrid(train,col='Star Rating')

g.map(plt.hist,'review_len')
plt.rcParams['figure.figsize']=(10,10)

sns.boxplot(y='review_len', data=train, x='Star Rating')
g = sns.FacetGrid(train,col='Star Rating')

g.map(plt.hist,'Polarity')
train['Star Rating'].value_counts()
rating_df=train.groupby('Star Rating')
rating_df['review_len'].describe()
sns.heatmap(train.corr(), cmap='coolwarm', annot=True)
train.drop(labels='App Version Name',axis=1,inplace=True)
train.sample(10).T # data preview brfore vectorization
cvec = CountVectorizer(min_df=.005, max_df=.9, ngram_range=(1,2), tokenizer=lambda doc: doc, lowercase=False)

cvec.fit(train['lem_review'])
pd.DataFrame(cvec.vocabulary_.items(),

             cvec.vocabulary_.values(),columns=['Word','Occurrence']).sort_values(by='Occurrence', ascending=False).head(25)
tfidf = TfidfVectorizer(token_pattern='(?ui)\\b\\w*[a-z]+\\w*\\b',sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),max_features=200)



features = tfidf.fit_transform(train['review']).toarray()

print(features.shape)

df2 = pd.DataFrame(features, columns=tfidf.get_feature_names())



"""Col_sum=df2.apply(lambda r: df2.sum()[r.name]) #Text Feature

Col_Max=Col_sum.sort_values().tail(500)

Text_Features=Col_Max.index

print(Text_Features)"""
df_Model = pd.concat([train,df2], axis=1)
drop_col=['id', 'Review Text', 'Review Title', 'Star Rating',

       'review', 'lem_review']

X=df_Model.drop(labels=drop_col,axis=1,inplace=False)

y=df_Model['Star Rating']

X['App Version Code'][X["App Version Code"].isna()]=0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42,stratify=y)
clf = RandomForestClassifier()



# Choose some parameter combinations to try

parameters = {'n_estimators': [25], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy'],

              'max_depth': [15], 

              #'min_samples_split': [5,10,15],

              #'min_samples_leaf': [20,100],

              'random_state' : [25]

             }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train, y_train)


predictions = clf.predict(X_test)

print(predictions)

print(classification_report(y_test, predictions))
from sklearn.model_selection import cross_val_score

cross_val_score(clf, X_train, y_train, cv=10, scoring='accuracy')
#Quick Check Model

def Model(Model,train_x=X_train,val_x=X_test,train_y=y_train,val_y=y_test):

    clf= Model()

    clf.fit(train_x,train_y)

    print("Model:",Model,"\n",classification_report(val_y,clf.predict(val_x)))
test.head().T
test['review']=test['Review Text'].combine_first(test['Review Title'])

test['review']=test['review'].apply(text_process)

test['lem_review']=test['review'].apply(lemmatize_text)

test['Polarity'] = sentiment_analyser(test['review'])

test["review_len"]= test['review'].apply(len)
features1 = tfidf.transform(test['review']).toarray()

test_tfidf = pd.DataFrame(features1, columns=tfidf.get_feature_names())

df_test = pd.concat([test,test_tfidf], axis=1)

df_test['App Version Code'][df_test["App Version Code"].isna()] = 0

Score_X=df_test.drop(['id', 'Review Text', 'Review Title','review', 'lem_review','App Version Name'],axis=1)

Score_X['App Version Code'][Score_X['App Version Code'].isna()] = 0
prediction= pd.DataFrame(clf.predict(Score_X))

result=pd.concat([test['id'],prediction],axis=1)

result.to_csv("predictions.csv",index=False)
