import pandas as pd

import numpy as np

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
df.describe()
df.columns
df.tail(10)
df.dropna(inplace=True)

df.drop(axis=1, columns="Id", inplace=True)
df.head()
sns.pairplot(df, hue='income')
df['income'].value_counts().plot(kind='bar')
df['sex'].value_counts().plot(kind='bar')
df['hours.per.week'].plot(kind='hist')
sns.scatterplot(df['education.num'], df['hours.per.week'], hue=df['income'])
sns.countplot(y=df['occupation'], hue=df['income'])
sns.countplot(y=df['race'], hue=df['income'])
sns.countplot(x=df['education.num'], hue=df['race'])
testAdult = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
testAdult.head(3)
testAdult.dropna(inplace=True)

testAdult.drop(axis='1', columns=['Id','education'], inplace=True)
adult = df.drop(axis='1', columns='education')
from sklearn import preprocessing
df.columns
qualiVars = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
adult[qualiVars] = adult[qualiVars].apply(preprocessing.LabelEncoder().fit_transform)
testAdult[qualiVars] = testAdult[qualiVars].apply(preprocessing.LabelEncoder().fit_transform)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
Xtrain = adult.iloc[:,0:-1]

Ytrain = adult.income
scores = []

for i in range(1,50):

    knn = KNeighborsClassifier(n_neighbors=i)

    scores.append(np.mean(cross_val_score(knn, Xtrain, Ytrain, cv=10)))
scores
plt.figure()

plt.plot(range(1,50), scores, color='blue',linestyle='dashed', marker='o', markerfacecolor='red')

plt.xlabel("K")

plt.ylabel("Cross_val score")
plt.figure(figsize=(10,6))

plt.plot(range(10,30), scores[10:30], color='blue',linestyle='dashed', marker='o', markerfacecolor='red')

plt.xlabel("K")

plt.ylabel("Cross_val score")
K = 18
knn = KNeighborsClassifier(n_neighbors=K)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xtrain,Ytrain,test_size=0.20)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("K=18\n")

print(confusion_matrix(y_test, pred))

print(classification_report(y_test, pred))

print('Accuracy: ',accuracy_score(y_test, pred))
knn = KNeighborsClassifier(n_neighbors=K)

knn.fit(Xtrain,Ytrain)

YTestAdult = knn.predict(testAdult)
YTestAdult
resultSubmission = pd.DataFrame(YTestAdult)
resultSubmission.index.name = 'Id'
resultSubmission.rename({0:'income'}, axis=1, inplace=True)
resultSubmission
#resultSubmission.to_csv("resultSubmission.csv")