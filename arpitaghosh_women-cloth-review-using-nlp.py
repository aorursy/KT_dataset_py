import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('white')
import matplotlib.pyplot as plt
%matplotlib inline
import os
print(os.listdir("../input/womens-ecommerce-clothing-reviews"))
df = pd.read_csv("..//input//womens-ecommerce-clothing-reviews//Womens Clothing E-Commerce Reviews.csv")
df.head()
df.drop(df.columns[0], axis = 1,inplace=True)
df.head()
df.info()
df.describe()
#type(df['Review Text'])
df['Review Text']=df['Review Text'].astype(str)
df['Review Length']=df['Review Text'].apply(len)
df.head(10)
g = sns.FacetGrid(df,col='Rating')
g.map(plt.hist,'Review Length')
sns.boxplot(x='Rating',y='Review Length',data=df,palette='rainbow')
sns.countplot(x='Rating',data=df,palette='rainbow')
ratings = df.groupby('Rating').mean()
ratings
ratings.corr()
sns.heatmap(ratings.corr(),cmap='coolwarm',annot=True)
df_part = df[(df.Rating==1) | (df.Rating==5)]
X = df_part['Review Text'].astype(str)

y = df_part['Rating']
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)
predictions = nb.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])
X = df_part['Review Text']
y = df_part['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))