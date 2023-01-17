import pandas as pd

import numpy as np 

import sklearn

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

#Train data parsing

adult = pd.read_csv("../input/adult-pmr3508/train_data.csv",

        names=[

        "Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?",

         header=0)
#Replacing NaN with most present values in each column

nadult = adult.apply(lambda x:x.fillna(x.value_counts().index[0]))
nadult.describe()
#Using seaborn pairplot do get correlation between two vars and income

sns.pairplot(nadult,vars=['Age', 'Education-Num', 'Hours per week'], hue="Target", palette="Set2", diag_kind="kde", height=2.5)
plt.figure(figsize=(10,8))

adultNumTarget = adult.copy()

adultNumTarget["Target"] =adultNumTarget["Target"].replace(["<=50K",">50K"],[0,1])

adultNumTarget=adultNumTarget.drop("Id",1)

sns.heatmap(adultNumTarget.dropna().corr(), vmin=-1, vmax=1, annot=True, cmap='viridis')

plt.show()
#Testing data parsing

testAdult = pd.read_csv("../input/adult-pmr3508/test_data.csv",

        names=[

        "Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?",

         skiprows=1)
#Replacing NaN with most present values in each column

nTestAdult = testAdult.apply(lambda x:x.fillna(x.value_counts().index[0]))
#Preprocessing data by replacing words with numbers

numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)
numAdult.describe()
#Normalizing values

numAdult[["Hours per week"]] = RobustScaler().fit_transform(numAdult[["Hours per week"]])

numAdult[["Age","Education-Num","Workclass"]] = StandardScaler().fit_transform(numAdult[["Age","Education-Num","Workclass"]])
#Definition of target values and data columns

Yadult = numAdult.Target

Xadult = numAdult[["Age",  "Education-Num", "Marital Status",

        "Occupation", "Race","Relationship", "Workclass",  "Sex", "Capital Gain", "Capital Loss",

        "Hours per week"]]
from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsClassifier



bestK=-1

bestMean=-1



#Testing train performance with different K nearest neighbors from 28 to 33

for i in range(25,36):

    knn = KNeighborsClassifier(n_neighbors=i)

    scores = cross_val_score(knn, Xadult, Yadult, cv=10, n_jobs=-1)

    mean = np.mean(scores)

    print(i,":",mean)

    if(mean > bestMean):

        bestMean = mean

        bestK = i
print("Best k:"+str(bestK))
#Generating model with best performance K value

knn = KNeighborsClassifier(n_neighbors=bestK)

knn.fit(Xadult,Yadult)
#Preparing test data

numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)

numTestAdult[['Hours per week']] = RobustScaler().fit_transform(numTestAdult[['Hours per week']])

numTestAdult[["Age","Education-Num","Workclass"]] = StandardScaler().fit_transform(numTestAdult[["Age","Education-Num","Workclass"]])



XtestAdult = numTestAdult[["Age",  "Education-Num", "Marital Status",

        "Occupation", "Race","Relationship", "Workclass",  "Sex", "Capital Gain", "Capital Loss",

        "Hours per week"]]



#Predict using generated model

YtestPred = knn.predict(XtestAdult)
#generating dict with IDxResult

matches = np.where(YtestPred==0,"<=50K",">50K")

final = dict(enumerate(x.rstrip() for x in matches))
#generating CSV

result = pd.DataFrame(final.items(), columns=['Id', 'income'])

result.to_csv (r'submission.csv', index = False, header=True)