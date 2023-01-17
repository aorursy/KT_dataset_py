# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/Texas Last Statement - CSV.csv", encoding="latin1")
df.head()
df["Gender"] = 0
genderList = [1 if i == 1.0 else 0 for i in df.MaleVictim]
df.Gender = genderList
df.drop(["Execution","TDCJNumber","PreviousCrime","Codefendants","NumberVictim","WhiteVictim","HispanicVictim","BlackVictim","VictimOther Races",],axis=1, inplace=True)
df.drop(["FemaleVictim","MaleVictim"],axis=1,inplace=True)

df.head()
df.dropna(inplace=True)

df.AgeWhenReceived = df.AgeWhenReceived.astype(int)
df.EducationLevel = df.EducationLevel.astype(int)
df.head()
plt.figure(figsize=(12,7))
sns.boxplot(y=df.Age)
plt.show()

print("Minimum Age is {}".format(df.Age.min()))
print("Maximum Age is {}".format(df.Age.max()))
print("Average Age is {:.1f}".format(df.Age.mean()))
df.corr()
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), linewidths=2, annot=True)
plt.show()
educationList= sorted(list(zip(df.EducationLevel.value_counts().index, df.EducationLevel.value_counts().values)))
eduYear, eduCount = zip(*educationList)
eduYear, eduCount = list(eduYear), list(eduCount)

plt.figure(figsize=(16,10))
plt.xlabel("Education Level")
plt.ylabel("Number of Offender")
plt.title("Number of Offender According To Their Education Level")
sns.barplot(x=eduYear,y=eduCount)
plt.show()
plt.figure(figsize=(16,10))
plt.xlabel("Last Name")
plt.ylabel("Frequency")
plt.title("Most 10 Frequent Last Names")
sns.barplot(x=df.LastName.value_counts()[:11].index,y=df.LastName.value_counts()[:11].values, palette="cubehelix")
plt.show()
plt.figure(figsize=(16,10))
plt.xlabel("First Name")
plt.ylabel("Frequency")
plt.title("Most 10 Frequent First Names")
sns.barplot(x=df.FirstName.value_counts()[:11].index,y=df.FirstName.value_counts()[:11].values, palette="gist_ncar_r")
plt.show()
import squarify

plt.figure(figsize=(15,8))
squarify.plot(sizes=df.Race.value_counts().values, label=df.Race.value_counts().index, color=["#17A096","#CBC015","#E4595D", "#979797"], alpha=.8 )
plt.axis('off')
plt.show()
print(df.Race.value_counts())
# We use regular expression to delete non-alphabetic characters on data.
import re

first_lastStatement = df.LastStatement[0]
lastStatement = re.sub("[^a-zA-Z]"," ",first_lastStatement)
print(lastStatement)
# Since upper and lower characters are (e.g a - A) evaluated like they are different each other by computer we make turn whole characters into lowercase.

lastStatement = lastStatement.lower()
print(lastStatement)
import nltk  # Natural Language Tool Kit

nltk.download("stopwords")  # If you dont't have that module this line will download it.
nltk.download('punkt') # It's necessary to import the module

from nltk.corpus import stopwords # We are importing 'stopwords'

lastStatement = nltk.word_tokenize(lastStatement) # We tokenized the statement

print(lastStatement)
# We will remove words like 'the', 'or', 'and', 'is' etc.

lastStatement = [i for i in lastStatement if not i in set(stopwords.words("english"))]
print(lastStatement)
# e.g: loved => love

nltk.download('wordnet') # It can be necessary
import nltk as nlp

lemmatization = nlp.WordNetLemmatizer()
lastStatement = [lemmatization.lemmatize(i) for i in lastStatement]

print(lastStatement)
# Now we turn our lastStatement list into sentence again

lastStatement = " ".join(lastStatement)

print(lastStatement)
statementList = list()

for statement in df.LastStatement:
    statement = re.sub("[^a-zA-Z]"," ",statement)
    statement = statement.lower()
    statement = nltk.word_tokenize(statement)
    statement = [i for i in statement if not i in set(stopwords.words("english"))]
    statement = [lemmatization.lemmatize(i)for i in statement]
    statement = " ".join(statement)
    statementList.append(statement)
statementList
from sklearn.feature_extraction.text import CountVectorizer

max_features = 600
count_vectorizer = CountVectorizer(max_features=max_features) 
sparce_matrix = count_vectorizer.fit_transform(statementList)
sparce_matrix = sparce_matrix.toarray()

print("Most Frequent {} Words: {}".format(max_features, count_vectorizer.get_feature_names()))
y = df.iloc[:,9].values # gender column
x = sparce_matrix
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.05, random_state = 42)

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train, y_train)

# Prediction
y_pred = nb.predict(x_test)
print("Accuracy: {:.2f}%".format(nb.score(y_pred.reshape(-1,1), y_test)*100))
df2 = pd.DataFrame(sparce_matrix)
df2["Age"] = df.Age
df2["Age"].fillna((df2["Age"].mean()),inplace=True)

# Normalization
x2 = (df2 - np.min(df2)) / (np.max(df2) - np.min(df2)).values
x_train2, x_test2, y_train2, y_test2 = train_test_split(x2, y, test_size = 0.05, random_state = 42)

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train2, y_train2)

# Prediction
y_pred2= nb.predict(x_test2)
print("Accuracy: {:.2f}%".format(nb.score(y_pred2.reshape(-1,1), y_test2)*100))
