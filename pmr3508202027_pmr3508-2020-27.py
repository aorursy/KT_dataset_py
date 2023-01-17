import pandas as pd

import sklearn as sk

import numpy as np







training = pd.read_csv("../input/adult-pmr3508/train_data.csv", index_col=['Id'], na_values="?")



testing = pd.read_csv("../input/adult-pmr3508/test_data.csv", index_col=['Id'], na_values="?")

training.head() 
testing.head() 
training.info()
training.describe()
print(len(training))

print(len(testing))
training.drop_duplicates(keep='first', inplace=True)

print(len(training))
cTraining = training.copy()

cTesting = testing.copy()
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.preprocessing import LabelEncoder

view_training = cTraining.dropna().apply(LabelEncoder().fit_transform)

le = LabelEncoder()



mask = np.triu(np.ones_like(view_training.corr(), dtype=np.bool))

plt.figure(figsize=(12,12))

sns.heatmap(view_training.corr(),square = True, annot=True, vmin=-1, vmax=1, cmap="Pastel2")

plt.show()
plt.figure(figsize=(10,8))

sns.countplot(x="workclass", data=training, palette='Spectral')

plt.show()
total = len(training["workclass"])

private = 100*(training["workclass"].value_counts()["Private"])/total

msg = "Private working class represents %.2f"%private

print(msg+"%")



total = len(training["workclass"])

self = 100*(training["workclass"].value_counts()["Self-emp-not-inc"])/total

msg = "Self-emp-not-inc class represents %.2f"%self

print(msg+"%")
plt.figure(figsize=(8,8))

sns.countplot(x="sex", data=training, palette='Spectral')

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(y="native.country", data=training, palette='Spectral')

plt.show()
total = len(training["native.country"])

private = 100*(training["native.country"].value_counts()["United-States"])/total

msg = "United States represents %.2f"%private

print(msg+"%")
plt.figure(figsize=(12,8))

sns.countplot(y="marital.status", data=training, palette='Spectral')

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(y="race", data=training, palette='Spectral')

plt.show()
total = len(training["race"])

white = 100*(training["race"].value_counts()["White"])/total

msg = "White represent %.2f"%white

print(msg+"%")



total = len(training["race"])

b = 100*(training["race"].value_counts()["Black"])/total

msg = "Balck represent %.2f"%b

print(msg+"%")
plt.figure(figsize=(12,8))

sns.countplot(y="occupation", data=training, palette='Spectral')

plt.show()
plt.figure(figsize=(12,8))

sns.countplot(y="relationship", data=training, palette='Spectral')

plt.show()
drop_cols = ['education', 'native.country', 'fnlwgt', 'income', 'race', 'workclass']

Ytraining = cTraining['income']

Xtraining = cTraining.drop(columns=drop_cols)
from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy='most_frequent') 

Xtraining[['occupation']] = imputer.fit_transform(Xtraining[['occupation']])
from sklearn.preprocessing import StandardScaler



Xtraining = pd.get_dummies(Xtraining)

Xtraining = StandardScaler().fit_transform(Xtraining)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score #scores of cross validation
kList = np.array([3, 10, 20, 25, 30, 40, 50, 60])

scoreList = np.array([]) #empty list to store mean scores for each K



for K in kList:

    knn = KNeighborsClassifier(n_neighbors=K)

    scores = cross_val_score(knn, Xtraining, Ytraining, cv=5) 

    mean_score = scores.mean() #computes mean of cross-validation scores

    scoreList = np.append(scoreList, mean_score)

    
for K, S in zip(kList, scoreList):

    print("K = %d ; Score = %5f"%(K,S))
newscoreList = np.array([]) #empty list to store mean scores for each K

newkList = []

for K in range(25,35):

    newkList.append(K)

    knn = KNeighborsClassifier(n_neighbors=K)

    scores = cross_val_score(knn, Xtraining, Ytraining, cv=5) 

    mean_score = scores.mean() #computes mean of cross-validation scores

    newscoreList = np.append(newscoreList, mean_score)
for K, S in zip(newkList, newscoreList):

    print("K = %d ; Score = %5f"%(K,S))
Xtesting = cTesting.drop(columns=['education', 'native.country', 'fnlwgt', 'race', 'workclass'])

Xtesting[['occupation']] = imputer.transform(Xtesting[['occupation']])
Xtesting = pd.get_dummies(Xtesting)
Xtesting = StandardScaler().fit_transform(Xtesting)
knn = KNeighborsClassifier(n_neighbors=34)
knn.fit(Xtraining,Ytraining)
prediction = knn.predict(Xtesting)
results = pd.DataFrame()

results[0] = cTesting.index

results[1] = prediction

results.columns = ['id','income']
results.head()
results.to_csv('submission.csv',index = False)