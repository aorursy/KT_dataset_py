import warnings

warnings.filterwarnings('ignore')

import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

import seaborn as sns
df1=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

df1.head()
df2=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

df2.head()
print('shape of training data is ',df1.shape)

print('shape of testing data is ',df2.shape)
print(df1.info())
print(df2.info())
df1.isna().sum()
df2.isna().sum()
df1.describe()
df2.describe()
sns.countplot(df1['target'],palette='rainbow')

plt.show()
df1['Data']='train'

df2['Data']='test'
df2['target']=np.nan ## for concat the dataset
df=pd.concat([df1,df2])

df.head()
df.tail()
print('checking shape of main dataset',df.shape)
df.info()
df.isnull().sum()
df['keyword'].nunique()
df['keyword'].value_counts()
df['location'].nunique()
df['location'].mode()
df['length']=df['text'].apply(len)

df.head()
df.describe()
plt.figure(figsize=(12,4))

g = sns.FacetGrid(df,col='target')

g.map(plt.hist,'length')

plt.show()
sns.boxplot(x='target',y='length',data=df,palette='rainbow')

plt.show()
stars = df.groupby('target').mean()

stars
stars.corr()
df1=df[df['Data']=='train']

df2=df[df['Data']=='test']
## droping unwanted column



df1=df1.drop('Data',axis=1)

df2=df2.drop(['Data','target'],axis=1)
print('shape of training data is ',df1.shape)

print('shape of testing data is ',df2.shape)
#Vectorizing the sentences;remove stop words

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(stop_words='english')
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
X=df1['text']

y=df1['target']

X1=df2['text']
### fitting the dataset 



vect.fit(X,y)
vect.vocabulary_
X_train_transformed= vect.transform(X)

X_test_transformed= vect.transform(X1)
### fitting the datset



mnb.fit(X_train_transformed, y)
### predicitng the ans 



y_pred_class = mnb.predict(X_test_transformed)

y_pred_class
len(y_pred_class)
results = pd.DataFrame({'id':df2['id'], 'target':y_pred_class})

results.to_csv("target_mnb.csv", index = False)
from sklearn.tree import DecisionTreeClassifier



dtc = DecisionTreeClassifier ( )
dtc.fit(X_train_transformed, y)



y_pred1 = dtc.predict(X_test_transformed)

y_pred1
results = pd.DataFrame({'id':df2['id'], 'target':y_pred1})

results.to_csv("target_dtc.csv", index = False)
from sklearn.svm import SVC

svc = SVC(C=1500)
svc.fit(X_train_transformed, y)



y_pred2 = svc.predict(X_test_transformed)

y_pred2
results = pd.DataFrame({'id':df2['id'], 'target':y_pred2})

results.to_csv("target_svc.csv", index = False)
from sklearn.feature_extraction.text import  TfidfTransformer



from sklearn.pipeline import Pipeline
pipeline = Pipeline([

    ('bow', CountVectorizer()),  # strings to token integer counts

    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier

])
X=df1['text']

y=df1['target']

X1=df2['text']
pipeline.fit(X,y)
predictions = pipeline.predict(X1)

predictions
results = pd.DataFrame({'id':df2['id'], 'target':predictions})

results.to_csv("target_pipe_mnb.csv", index = False)
pipeline = Pipeline([

    ('bow', CountVectorizer()),  # strings to token integer counts

    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ('classifier', DecisionTreeClassifier()),  # train on TF-IDF vectors w/ Naive Bayes classifier

])
pipeline.fit(X,y)



prediction = pipeline.predict(X1)

prediction





results = pd.DataFrame({'id':df2['id'], 'target':prediction})

results.to_csv("target_pipe_dtc.csv", index = False)
from sklearn.linear_model import LogisticRegression
pipeline = Pipeline([

    ('bow', CountVectorizer()),  # strings to token integer counts

    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores

    ('classifier', LogisticRegression()),  # train on TF-IDF vectors w/ Naive Bayes classifier

])
pipeline.fit(X,y)



prediction_lr = pipeline.predict(X1)

prediction_lr





results = pd.DataFrame({'id':df2['id'], 'target':prediction_lr})

results.to_csv("target_pipe_dtc.csv", index = False)
# Importing Count Vectorizer...

from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()

# Example of Count Vectorizer

text = ["Jack and Jill went up the Hill!"]

count_vectorizer.fit(text) # fit function helps learn a vocabulary about the text

transformed = count_vectorizer.transform(text) # encodes the text/doc into a vector
print(transformed.shape)

print(type(transformed))

print(transformed.toarray())
print(count_vectorizer.vocabulary_)
# Test with another word

print(count_vectorizer.transform(["HELLO GUYS"]).toarray()) # able to recognize the word in upper case | Location is 2

print(count_vectorizer.transform(["AND"]).toarray()) # Loc is 0 as per above vocabulary

print(count_vectorizer.transform(["up"]).toarray())

print(count_vectorizer.transform(["SWAPNIL WAGH"]).toarray()) # No words found and hence all 0
# lets get the count of first 5 tweets

exmple  = count_vectorizer.fit_transform(df1["text"][0:5])



print(exmple[0].todense().shape)

print(exmple[0].todense())
print(list(count_vectorizer.vocabulary_))

print("Unique Words are: ")

print(np.unique(list(count_vectorizer.vocabulary_)))
# Train Set

tweets = count_vectorizer.fit_transform(df1["text"]) # Transformed the Train Tweets

y=df1['target']
# Test Set

testtweets = count_vectorizer.transform(df2["text"])
from sklearn.linear_model import RidgeClassifier

from sklearn.model_selection import cross_val_score

ridge = RidgeClassifier()
print(cross_val_score(ridge, tweets, y, cv = 5, scoring ="f1").mean())
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier

lr=LogisticRegression()

svc=SVC()

svc1 = SVC(C=1500)

dt=DecisionTreeClassifier()

mnb=MultinomialNB()

gbm = GradientBoostingClassifier()

rf = RandomForestClassifier()

vc = VotingClassifier(estimators = [("rf", rf), ("ridge", ridge), ("GBM", gbm),

                                    ("lr", lr), ("svc", svc), ("svc1", svc1),

                                    ("dt", dt), ("mnb", mnb)])
vc.fit(tweets, y)
test=vc.predict(testtweets)

test
results = pd.DataFrame({'id':df2['id'], 'target':test})

results.to_csv("target_test.csv", index = False)
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier

gbm = GradientBoostingClassifier()

rf = RandomForestClassifier()

vc = VotingClassifier(estimators = [("rf", rf), ("ridge", ridge), ("GBM", gbm)])
vc.fit(tweets, df1.target)
solution = pd.DataFrame({"id": df2.id, "target": vc.predict(testtweets)})

solution.to_csv("VC Model.csv", index=False) 