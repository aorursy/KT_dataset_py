import pandas as pd

import numpy as np



adult = pd.read_csv("../input/adult-census-income/adult.csv", names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

adult_test = pd.read_csv("../input/us-census-data/adult-test.csv", names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")

adult_training = pd.read_csv("../input/us-census-data/adult-training.csv", names=[

        "Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",

        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",

        "Hours per week", "Country", "Target"],

        sep=r'\s*,\s*',

        engine='python',

        na_values="?")



# data prep



# remove first line

adult=adult[1:]

adult_test=adult_test[1:]



# remove missing data

nadult = adult.dropna()

nTestAdult = adult_test.dropna()

nTrainingAdult = adult_training.dropna()
# random forest



# data prep

from sklearn import preprocessing



numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)

numTrainingAdult = nTrainingAdult.apply(preprocessing.LabelEncoder().fit_transform)



XtestAdult = numTestAdult.iloc[:,0:14]

YtestAdult = numTestAdult.Target

XtrainingAdult = numTrainingAdult.iloc[:,0:14]

YtrainingAdult = numTrainingAdult.Target



# train model

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import accuracy_score



rf = RandomForestRegressor(n_estimators = 400, random_state = 42)

rf.fit(XtrainingAdult, YtrainingAdult)

predictions = rf.predict(XtestAdult)



predictions = np.array(predictions)

YtestAdult = np.array(YtestAdult)



y = []

for i in predictions:

    if i >= 0.5:

        y.append(1)

    else:

        y.append(0)

y = np.array(y)

accuracy_score(YtestAdult, y)
# linear regression



# data prep

numAdult = nadult.apply(preprocessing.LabelEncoder().fit_transform)

numTestAdult = nTestAdult.apply(preprocessing.LabelEncoder().fit_transform)

numTrainingAdult = nTrainingAdult.apply(preprocessing.LabelEncoder().fit_transform)

Xadult = numAdult.iloc[:,0:14]

Yadult = numAdult.Target

XtestAdult = numTestAdult.iloc[:,0:14]

YtestAdult = numTestAdult.Target

XtrainingAdult = numTrainingAdult.iloc[:,0:14]

YtrainingAdult = numTrainingAdult.Target





YtestAdult = np.array(YtestAdult)



# train model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(XtrainingAdult, YtrainingAdult)

y_pred = regressor.predict(XtestAdult)



y = []

for i in y_pred:

    if i >= 0.5:

        y.append(1)

    else:

        y.append(0)



y = np.array(y)

accuracy_score(y, YtestAdult)
#k-means

from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score



# sem seleção de atributos



# data prep

Xadult = numAdult.iloc[:,0:14]

Xadult = np.array(Xadult)

Yadult = numAdult.Target

Yadult = np.array(Yadult)



# model

km = KMeans(n_clusters = 2, init = 'random')

km.fit(Xadult)

y_km = km.predict(Xadult)

y_km = np.array(y_km)

a = accuracy_score(Yadult,y_km)

print(a)



# com seleção de atributos



#data prep

Xadult1 = nadult[["Age","Education-Num","Capital Gain", "Capital Loss", "Hours per week"]]

Yadult1 = numAdult.Target



# model

km1 = KMeans(n_clusters = 2, init = 'random')

km1.fit(Xadult1)

y1_km = km1.predict(Xadult1)

y1_km = np.array(y1_km)

b = accuracy_score(Yadult1,y1_km)

print(b)