import pandas as pd

import numpy  as np
df=pd.read_csv('../input/nlp-getting-started/train.csv')

df
df.info()
df.keyword.value_counts(dropna=False)
df.location.value_counts(dropna=False)
df.drop(labels='location',inplace=True,axis=1)
df.isnull().sum()
df[df['target']==1].isnull().sum()
df['keyword'].fillna(df[df['target']==1].keyword.mode(), inplace=True)
df.isnull().sum()
df[df['target']==0].isnull().sum()
df[df['target']==1].isnull().sum()
df.target.value_counts()
df.dropna(inplace=True)
df.isnull().sum()
df.info()
X=df.drop('target',axis=1)

y=df['target']
X
X.drop(labels='id',inplace=True,axis=1)
y
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
X.head()
# Removing punctuations

X=X.iloc[:,0:2]

X.replace("[^a-zA-Z]"," ",regex=True, inplace=True)



# Renaming column names for ease of access

list1= [i for i in range(2)]

new_Index=[str(i) for i in list1]

X.columns= new_Index

X.head(5)
# Convertng headlines to lower case

for index in new_Index:

    X[index]=X[index].str.lower()

X.head(1)
' '.join(str(x) for x in X.iloc[1,0:2])
tweet = []### Combining both the columns

for row in range(0,len(X.index)):

    tweet.append(' '.join(str(x) for x in X.iloc[row,0:3]))
tweet[0]
len(tweet)
import re

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

corpus = []

for i in range(len(tweet)):

    review = re.sub('[^a-zA-Z]', ' ', tweet[i])

    review = review.lower()

    review = review.split() 

    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]

    review = ' '.join(review)

    corpus.append(review)
corpus[0]
len(corpus)
## implement BAG OF WORDS

countvector=CountVectorizer(ngram_range=(2,2))

traindataset=countvector.fit_transform(corpus)
countvector
traindataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(traindataset, y, test_size=0.33, random_state=42)
X_train.shape
from sklearn.ensemble import RandomForestClassifier



# implement RandomForest Classifier

randomclassifier=RandomForestClassifier(n_estimators=200,criterion='entropy')

randomclassifier.fit(X_train,y_train)
predictions = randomclassifier.predict(X_test)

predictions
## Import library to check accuracy

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score



matrix=confusion_matrix(y_test,predictions)

print(matrix)

score=accuracy_score(y_test,predictions)

print(score)

report=classification_report(y_test,predictions)

print(report)
from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()

mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)

y_pred
cm=confusion_matrix(y_test,y_pred)

print(cm)

acc=accuracy_score(y_test,y_pred)

print(acc)

rep=classification_report(y_test,y_pred)

print(rep)
from sklearn.model_selection import cross_val_score

nb_Kfold_accu = cross_val_score(estimator = mnb, X = X_train, y = y_train, cv = 10)

nb_Kfold_accu=nb_Kfold_accu.mean()

print("Accuracy: {:.2f} %".format(nb_Kfold_accu*100))
test=pd.read_csv('../input/nlp-getting-started/test.csv')

test
test.isnull().sum()
test.drop(labels=['location'],axis=1,inplace=True)
test.keyword.value_counts()
test['keyword'].fillna('deluged',inplace=True)
test.info()
df2=test.copy()
test.drop(labels=['id'],axis=1,inplace=True)
# Removing punctuations

test=test.iloc[:,0:2]

test.replace("[^a-zA-Z]"," ",regex=True, inplace=True)



# Renaming column names for ease of access

list2= [j for j in range(2)]

new_Index1=[str(j) for j in list2]

test.columns= new_Index1

test.head(5)
' '.join(str(x) for x in test.iloc[1,0:2])
test_tweet = []

for row in range(0,len(test.index)):

    test_tweet.append(' '.join(str(x) for x in test.iloc[row,0:2]))
test_tweet[1]
ps = PorterStemmer()

test_corpus = []

for i in range(len(test_tweet)):

    review1 = re.sub('[^a-zA-Z]', ' ',test_tweet[i])

    review1 = review1.lower()

    review1 = review1.split() 

    review1 = [ps.stem(word) for word in review1 if not word in stopwords.words('english')]

    review1 = ' '.join(review1)

    test_corpus.append(review1)
test_corpus[0]
test_dataset = countvector.transform(test_corpus)

predict1 = randomclassifier.predict(test_dataset)

predict1
sub= pd.DataFrame({'id':df2.id,

                      'target':predict1})



sub.to_csv("outcome.csv",index=False,header = True)