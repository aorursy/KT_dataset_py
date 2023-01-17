import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing as prep

import numpy as np
adult = pd.read_csv("../input/adult-pmr3508/train_data.csv", names=[

        "Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?",skiprows=[0])

adult.shape
testAdult = pd.read_csv("../input/adult-pmr3508/test_data.csv", names=[

        "Id","Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?",skiprows=[0])

testAdult.shape
adult.head()
adult.describe()
adult.isnull().sum()
print('Occupation:\n')

print(adult['Occupation'].describe())



print('\n\nWorkclass:\n')

print(adult['Workclass'].describe())



print('\n\nCountry:\n')

print(adult['Country'].describe())
moda = adult['Workclass'].describe().top

adult['Workclass'] = adult['Workclass'].fillna(moda)



moda = adult['Country'].describe().top

adult['Country'] = adult['Country'].fillna(moda)



moda = adult['Occupation'].describe().top

adult['Occupation'] = adult['Occupation'].fillna(moda)
adult.isnull().sum()
testAdult.isnull().sum()
print('Occupation:\n')

print(testAdult['Occupation'].describe())



print('\n\nWorkclass:\n')

print(testAdult['Workclass'].describe())



print('\n\nCountry:\n')

print(testAdult['Country'].describe())
moda = testAdult['Workclass'].describe().top

testAdult['Workclass'] = testAdult['Workclass'].fillna(moda)



moda = testAdult['Country'].describe().top

testAdult['Country'] = testAdult['Country'].fillna(moda)



moda = testAdult['Occupation'].describe().top

testAdult['Occupation'] = testAdult['Occupation'].fillna(moda)
fig, axes = plt.subplots(nrows = 3, ncols = 2)

plt.tight_layout(pad = .4, w_pad = .5, h_pad = 1.)



sex = adult.groupby(["Sex", "Target"]).size().unstack()

sex["sum"] = adult.groupby("Sex").size()

sex = sex.sort_values("sum", ascending = False)[["<=50K", ">50K"]]

sex.plot(kind = "bar", stacked = True, ax = axes[0,0], figsize = (20, 15))



relationship = adult.groupby(["Relationship", "Target"]).size().unstack()

relationship["sum"] = adult.groupby("Relationship").size()

relationship = relationship.sort_values("sum", ascending = False)[["<=50K", ">50K"]]

relationship.plot(kind = "bar", stacked = True, ax = axes[0, 1])



education = adult.groupby(["Education", "Target"]).size().unstack()

education["sum"] = adult.groupby("Education").size()

education = education.sort_values("sum", ascending = False)[["<=50K", ">50K"]]

education.plot(kind = "bar", stacked = True, ax = axes[1, 0])



occupation = adult.groupby(["Occupation", "Target"]).size().unstack()

occupation["sum"] = adult.groupby("Occupation").size()

occupation = occupation.sort_values("sum", ascending = False)[["<=50K", ">50K"]]

occupation.plot(kind = "bar", stacked = True, ax = axes[1, 1])



workclass = adult.groupby(["Workclass", "Target"]).size().unstack()

workclass["sum"] = adult.groupby("Workclass").size()

workclass = workclass.sort_values("sum", ascending = False)[["<=50K", ">50K"]]

workclass.plot(kind = "bar", stacked = True, ax = axes[2, 0])



race = adult.groupby(["Race", "Target"]).size().unstack()

race["sum"] = adult.groupby("Race").size()

race = race.sort_values("sum", ascending = False)[["<=50K", ">50K"]]

race.plot(kind = "bar", stacked = True, ax = axes[2, 1])
adult["Country"].value_counts()
nNumber = ["Workclass", "Occupation", "Race", "Sex", "Relationship", "Martial Status"]



adult[nNumber] = adult[nNumber].apply(prep.LabelEncoder().fit_transform)



testAdult[nNumber] = testAdult[nNumber].apply(prep.LabelEncoder().fit_transform)



adult.head()
atr_num = ["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]
x_train = adult[atr_num]
y_train = adult.Target
x_test = testAdult[atr_num]
knn = KNeighborsClassifier(algorithm = "auto", leaf_size = 30, metric = "manhattan",

           metric_params = None, n_jobs = 1, n_neighbors = 5, p = 2,

           weights = "uniform")
score = np.mean(cross_val_score(knn, x_train, y_train, cv=10))

score
knn.fit(x_train, y_train)

YtestePred = knn.predict(x_test)

YtestePred
atr_cat = ["Workclass", "Occupation", "Race", "Sex", "Relationship", "Martial Status"]

atr = atr_num + atr_cat
x_train = adult[atr]
x_test = testAdult[atr]
knn = KNeighborsClassifier(algorithm = "auto", leaf_size = 30, metric = "manhattan",

           metric_params = None, n_jobs = 1, n_neighbors = 5, p = 2,

           weights = "uniform")
score = np.mean(cross_val_score(knn, x_train, y_train, cv=10))

score
knn.fit(x_train, y_train)

YtestePred = knn.predict(x_test)

YtestePred
scores = 0



k_max = 0

for k in range(20,35):

    knn = KNeighborsClassifier(n_neighbors = k, metric = "manhattan")

    score = np.mean(cross_val_score(knn, x_train, y_train, cv=10))

    

    if score > scores:

        scores = score

        k_max = k

        



knn = KNeighborsClassifier(n_neighbors = k_max, metric = "manhattan")

print("O melhor k é %d com score de %.4f" %(k_max, scores))
knn.fit(x_train, y_train)

YtestePred = knn.predict(x_test)
predict = pd.DataFrame(testAdult)

predict = predict.drop('Age', axis=1)

predict = predict.drop("Workclass", axis=1)

predict = predict.drop("Education-Num", axis=1)

predict = predict.drop("Martial Status", axis=1)

predict = predict.drop("Occupation", axis=1)

predict = predict.drop("Race", axis=1)

predict = predict.drop("Sex", axis=1)

predict = predict.drop("Capital Gain", axis=1)

predict = predict.drop("Capital Loss", axis=1)

predict = predict.drop("Hours per week", axis=1)

predict = predict.drop("Country", axis=1)

predict = predict.drop("fnlwgt", axis=1)

predict = predict.drop("Relationship", axis=1)

predict = predict.drop("Education", axis=1)

predict = predict.drop("Id", axis=1)

predict["income"] = YtestePred

predict.head()
predict.to_csv("prediction.csv", index = True, index_label = 'Id')