import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as mno

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression



%matplotlib inline
data = pd.read_csv("../input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv")

print(data.info())

data.sample(10)
mno.matrix(data, figsize = (20, 6))
x = data.iloc[:, 2:12]

y = data.iloc[:,13]
col_names_int = list(x.select_dtypes(int).columns)

le=LabelEncoder()

y = pd.DataFrame(le.fit_transform(np.array(y.values.ravel())), columns=["BankCrisis"])

for col in col_names_int:

    x[col] = le.fit_transform(x[col].astype(str))

    y = pd.DataFrame(le.fit_transform(np.array(y.values.ravel())), columns=["RainTomorrow"])

df1= x[col_names_int]
col_names = list(x.select_dtypes(float).columns)

scaler = StandardScaler()

df2 = scaler.fit_transform(x.select_dtypes(float))

df2 = pd.DataFrame(df2, columns=col_names)
#join features

x = pd.concat([df1, df2], axis=1)
resultsagg = pd.DataFrame()
results30=[]

for i in range(30): 

    kf = KFold(10, shuffle=True, random_state=i)

    results = []

    for l_train, l_valid in kf.split(x):

        x_train, x_valid = x.iloc[l_train], x.iloc[l_valid] 

        y_train, y_valid = y.iloc[l_train], y.iloc[l_valid]



        knn = KNeighborsClassifier(n_neighbors=10)

        knn.fit(x_train, y_train.values.ravel())

        y_pred = knn.predict(x_valid)

        acc = accuracy_score(y_valid.values.ravel(), y_pred)

        results.append(acc)

    

    results30.append(np.mean(results))



resultsagg["KNN"]=results30
sns.distplot(results30)
results30=[]

for i in range(30):

    kf = KFold(10, shuffle=True, random_state=i)

    results = []

    for l_train, l_valid in kf.split(x):

        x_train, x_valid = x.iloc[l_train], x.iloc[l_valid] 

        y_train, y_valid = y.iloc[l_train], y.iloc[l_valid]



        log = LogisticRegression(random_state=i, solver='liblinear')

        log.fit(x_train, y_train.values.ravel())

        y_pred = log.predict(x_valid)

        acc = accuracy_score(y_valid.values.ravel(), y_pred)

        results.append(acc)

    results30.append(np.mean(results))



resultsagg["LogisticRegression"]=results30
sns.distplot(resultsagg["LogisticRegression"])
results30=[]

for i in range(30):

    kf = KFold(10, shuffle=True, random_state=i)

    results = []

    for l_train, l_valid in kf.split(x):

        x_train, x_valid = x.iloc[l_train], x.iloc[l_valid] 

        y_train, y_valid = y.iloc[l_train], y.iloc[l_valid]



        rf = RandomForestClassifier(n_estimators=20, n_jobs=-1, random_state=i)

        rf.fit(x_train, y_train.values.ravel())

        y_pred = rf.predict(x_valid)

        acc = accuracy_score(y_valid.values.ravel(), y_pred)

        results.append(acc)

    results30.append(np.mean(results))



resultsagg["RandomForest"]=results30
sns.distplot(resultsagg["RandomForest"])
results30=[]

for i in range(30):

    kf = KFold(10, shuffle=True, random_state=i)

    results = []

    for l_train, l_valid in kf.split(x):

        x_train, x_valid = x.iloc[l_train], x.iloc[l_valid] 

        y_train, y_valid = y.iloc[l_train], y.iloc[l_valid]



        nb = GaussianNB()

        nb.fit(x_train, y_train.values.ravel())

        y_pred = nb.predict(x_valid)

        acc = accuracy_score(y_valid.values.ravel(), y_pred)

        results.append(acc)

    results30.append(np.mean(results))



resultsagg["NaiveBayes"]=results30
sns.distplot(resultsagg["NaiveBayes"])
print("KNN:",  resultsagg["KNN"].mean())

print("RandomForest:",  resultsagg["RandomForest"].mean())

print("LogisticRegression:",  resultsagg["LogisticRegression"].mean())

print("NaiveBayes:",  resultsagg["NaiveBayes"].mean())
resultsagg
def ranking_model(results_aggregate):

    ranking = pd.DataFrame(columns=results_aggregate.columns)

    for i in range(results_aggregate.shape[0]):

        ranking.loc[i, resultsagg.iloc[i].rank(ascending=False).index]=resultsagg.iloc[i].rank(ascending=False)

    return ranking
ranking_model(resultsagg)