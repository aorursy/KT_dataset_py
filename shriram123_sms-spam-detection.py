# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/spam.csv', encoding='latin-1')

data.columns = ["label", "text", "A1", "A2", "A3"]

data.head()
data = data.drop(columns=["A1", "A2", "A3"])

data.head()
sns.countplot(data["label"])
data.groupby('label').describe()
from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



spamtextdf = data[data.label == "spam"].text

hamtextdf = data[data.label == "ham"].text



spamtext = " ".join(text for text in spamtextdf)

hamtext = " ".join(text for text in hamtextdf)

stopwords = stopwords.words('english')



wordcloudspam = WordCloud(stopwords=stopwords, background_color="white").generate(spamtext)

wordcloudham = WordCloud(stopwords=stopwords, background_color="white").generate(hamtext)
plt.imshow(wordcloudspam, interpolation='bilinear')

plt.axis("off")

plt.show()
plt.imshow(wordcloudham, interpolation='bilinear')

plt.axis("off")

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

X_train,X_test,y_train,y_test = train_test_split(data["text"],data["label"], test_size = 0.2, random_state = 42)

from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer()
vect.fit(X_train)

feature_train = vect.transform(X_train)

feature_test = vect.transform(X_test)

print(feature_train.toarray())
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import RandomForestClassifier

prediction = dict()
model = LogisticRegression()

model.fit(feature_train,y_train)

prediction["Logistics"]= model.predict(feature_test)

accuracy_score(y_test,prediction["Logistics"])
model = SVC(gamma=1)

model.fit(feature_train,y_train)

prediction["svc"]= model.predict(feature_test)

accuracy_score(y_test,prediction["svc"])
model = MultinomialNB(alpha=0.01)

model.fit(feature_train,y_train)

prediction["mnb"]= model.predict(feature_test)

accuracy_score(y_test,prediction["mnb"])
model = RandomForestClassifier(n_estimators=15, random_state=42)

model.fit(feature_train,y_train)

prediction["rf"]= model.predict(feature_test)

accuracy_score(y_test,prediction["rf"])
accuracy = {"LogisticRegression":0.967713004484305, "SVM":0.9820627802690582, "naiveBays":0.9838565022421525, "randomForest":0.9748878923766816}
accuracy_df = pd.Series(accuracy)

accuracy_df.plot(kind='bar')