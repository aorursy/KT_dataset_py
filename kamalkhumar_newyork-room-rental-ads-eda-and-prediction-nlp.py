# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import train_test_split

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

import missingno as miss

from collections import Counter

import matplotlib.pyplot as plt

import plotly.express as px
df = pd.read_csv("/kaggle/input/newyork-room-rentalads/room-rental-ads.csv")
df.head()
df.shape
df.dtypes
df.isna().sum()
miss.bar(df)

plt.show()
df.dropna(how="any", inplace=True)

df = df.reset_index(drop=True)
df["Vague/Not"].value_counts()
df.rename(columns = {"Vague/Not":"Target"},inplace = True)

df.Target = df.Target.astype("int").astype("category")

df
df.Description.sample(10)
#check for duplicates



len(df[df.duplicated()])
#drop duplicates



df = df.drop_duplicates(subset=['Description'])

print(df.head())

print(df.shape)
#normalization



import re

import spacy

nlp = spacy.load('en')



def normalize(msg):

    

    msg = re.sub('[^A-Za-z]+', ' ', msg) #remove special character and intergers

    doc = nlp(msg)

    res=[]

    for token in doc:

        if(token.is_stop or token.is_punct or token.is_currency or token.is_space or len(token.text) <= 2): #word filteration

            pass

        else:

            res.append(token.lemma_.lower())

    return res
df["Description"] = df["Description"].apply(normalize)

df.head()
words_collection = Counter([item for sublist in df['Description'] for item in sublist])

freq_word_df = pd.DataFrame(words_collection.most_common(20))

freq_word_df.columns = ['frequently_used_word','count']



freq_word_df.style.background_gradient(cmap='YlGnBu', low=0, high=0, axis=0, subset=None)
fig = px.scatter(freq_word_df, x="frequently_used_word", y="count", color="count", title = 'Frequently used words - Scatter plot')

fig.show()
df["Description"] = df["Description"].apply(lambda m : " ".join(m))
from sklearn.feature_extraction.text import TfidfVectorizer #vectorize the string

c = TfidfVectorizer(ngram_range=(1,2))

mat=pd.DataFrame(c.fit_transform(df["Description"]).toarray(),columns=c.get_feature_names(),index=None)

mat
from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import BernoulliNB

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix
clfs = {

    'mnb': MultinomialNB(),

    'gnb': GaussianNB(),

    'svm1': SVC(kernel='linear'),

    'svm2': SVC(kernel='rbf'),

    'svm3': SVC(kernel='sigmoid'),

    'mlp1': MLPClassifier(),

    'mlp2': MLPClassifier(hidden_layer_sizes=[100, 100]),

    'ada': AdaBoostClassifier(),

    'dtc': DecisionTreeClassifier(),

    'rfc': RandomForestClassifier(),

    'gbc': GradientBoostingClassifier(),

    'lr': LogisticRegression()

}
train_x, train_y, test_x, test_y = train_test_split(mat, df['Target'], test_size=0.3)
accuracy_scores = dict()



for clf_name in clfs:

    

    clf = clfs[clf_name]

    clf.fit(train_x, test_x)

    y_pred = clf.predict(train_y)

    accuracy_scores[clf_name] = accuracy_score(y_pred, test_y)

    print(clf_name, accuracy_scores[clf_name])
accuracy_scores = dict(sorted(accuracy_scores.items(), key = lambda kv:(kv[1], kv[0]), reverse= True))

h = list(accuracy_scores.keys())[0]

print("Classifier with high accuracy --> ",clfs[h])

print("With the accuracy of",accuracy_scores[h])
cm = confusion_matrix(clfs[h].predict(train_y), test_y)

print(cm)
#graph with confusion matrix



group_names = ["True Neg","False Pos","False Neg","True Pos"]

group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]

group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

sns.heatmap(cm, annot=labels, fmt="", cmap='Blues')

plt.show()
fig,ax=plt.subplots(figsize=(10,5))

sns.regplot(y=test_y,x=clfs[h].predict(train_y),marker="*")

plt.show()