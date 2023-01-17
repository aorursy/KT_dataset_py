import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
yelp = pd.read_csv("../input/yelp-reviews-dataset/yelp.csv")
yelp.head()
yelp.info()
yelp.describe()
yelp["text length"] = yelp["text"].apply(len)
yelp.head()
###
g = sns.FacetGrid(yelp, col="stars")
g.map(plt.hist, "text length")
sns.boxplot(x="stars", y="text length", data=yelp)
sns.countplot(yelp["stars"])
stars = yelp.groupby("stars").mean()
stars
stars.corr()
sns.heatmap(stars.corr(), cmap="coolwarm", annot=True)
yelp_class = yelp[(yelp["stars"] == 1) | (yelp["stars"] == 5)]
yelp_class.head()
X = yelp_class["text"] 
y = yelp_class["stars"]
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ("bow", CountVectorizer()),
    ("tfidf", TfidfTransformer()),
    ("model", MultinomialNB())
])
X = yelp_class["text"]
y = yelp_class["stars"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
pipeline.fit(X_train, y_train)
new_pred = pipeline.predict(X_test)
print(confusion_matrix(y_test, new_pred))
print(classification_report(y_test, new_pred))