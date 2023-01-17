
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


df=pd.read_csv('../input/fake-news-data-for-classification/fake-news-data.csv')
df.head(5)
df.shape
df.info()
df.isnull().sum()
import seaborn as sns
sns.countplot(df['label'])
df.value_counts(df['label'])
x=df.title.values
y=df.label.values
y
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.20)
from sklearn.feature_extraction.text import CountVectorizer
cb=CountVectorizer()
x_train=cb.fit_transform(xtrain)
x_train.toarray()
x_test=cb.transform(xtest)
x_test.toarray()
from sklearn.naive_bayes import MultinomialNB
mn=MultinomialNB()
mn.fit(x_train,ytrain)
mn.score(x_test,ytest)
news=["Kerry to go to Paris in gesture of sympathy"]
test_news=cb.transform(news)
mn.predict(test_news)