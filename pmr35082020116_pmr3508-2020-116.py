import pandas as pd

import sklearn

import matplotlib.pyplot as plt

import numpy as np



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
dtrain = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",



        sep=r'\s*,\s*',

        engine='python',

        na_values="?")
dtrain.shape
dtrain.describe()
dtrain.head()
dtrain["native.country"].value_counts()
dtrain["age"].value_counts().plot(kind="bar", figsize=(18,6))
dtrain["sex"].value_counts().plot(kind="pie")
dtrain["education"].value_counts().plot(kind="barh")
dtrain["occupation"].value_counts().plot(kind="barh")
dtest=pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

                     

                     sep=r'\s*,\s*',

                     engine="python",

                     na_values="?")
dtest.shape
dtest.head()
dtest
pd.crosstab(dtrain["age"],dtrain["income"],normalize="index").plot(figsize=(7,4))
pd.crosstab(dtrain["workclass"],dtrain["income"],normalize="index").plot.barh(stacked=True, figsize=(10,4))
pd.crosstab(dtrain["education.num"],dtrain["income"],normalize="index").plot.barh(stacked=True)
pd.crosstab(dtrain["marital.status"],dtrain["income"],normalize="index").plot.barh()
pd.crosstab(dtrain["occupation"],dtrain["income"],margins=True,normalize="index").plot.barh(stacked="True")
pd.crosstab(dtrain["relationship"],dtrain["income"],margins=True,normalize="index").plot.barh()
pd.crosstab(dtrain["race"],dtrain["income"],normalize="index").plot.barh()
pd.crosstab(dtrain["sex"],dtrain["income"],normalize="index").plot.barh(stacked="True")
pd.crosstab(dtrain["capital.gain"],dtrain["income"]).plot(figsize=(10,5))
pd.crosstab(dtrain["capital.loss"],dtrain["income"]).plot(figsize=(10,5))
pd.crosstab(dtrain["hours.per.week"],dtrain["income"]).plot(figsize=(12,5))
pd.crosstab(dtrain["native.country"],dtrain["income"]).plot.bar(figsize=(12,5))
ids = []

nums = []



work_id = ["Private","Self-emp-not-inc","Local-gov","State-gov","Self-emp-inc","Federal-gov","Without-pay","Never-worked"]

work_num = ["22696","2541","2093","1297","1116","960","14","7"]

ids += work_id

nums += work_num



marit_id = ["Married-civ-spouse","Never-married","Divorced","Separated","Widowed","Married-spouse-absent","Married-AF-spouse"]

marit_num = ["14976","10682","4443","1025","993","418","23"]

ids += marit_id

nums += marit_num



occu_id = ["Prof-specialty","Craft-repair","Exec-managerial","Adm-clerical","Sales","Other-service","Machine-op-inspct","Transport-moving","Handlers-cleaners","Farming-fishing","Tech-support","Protective-serv","Priv-house-serv","Armed-Forces"]

occu_num = ["4140","4099","4066","3769","3650","3295","2002","1597","1370","994","928","649","149","9"]

ids += occu_id

nums += occu_num



rela_id = ["Husband","Not-in-family","Own-child","Unmarried","Wife","Other-relative"]

rela_num = ["13193","8304","5068","3446","1568","981"]

ids += rela_id

nums += rela_num



race_id = ["White","Black","Asian-Pac-Islander","Amer-Indian-Eskimo","Other"]

race_num = ["27815","3124","1039","311","271"]

ids += race_id

nums += race_num



sex_id = ["Male","Female"]

sex_num = ["21789","10771"]

ids += sex_id

nums += sex_num

def freq1(dado):

    for i in range(len(nums)):

        if dado == ids[i]:

            return nums[i]

    return dado
def freq2(dado):

    for i in range(len(nums)):

        if dado == ids[i]:

            return nums[i]

    return 0
dtrain["age"] = dtrain["age"].apply(freq1)

dtrain["workclass"] = dtrain["workclass"].apply(freq1)

dtrain["marital.status"] = dtrain["marital.status"].apply(freq1)

dtrain["occupation"] = dtrain["occupation"].apply(freq1)

dtrain["relationship"] = dtrain["relationship"].apply(freq1)

dtrain["race"] = dtrain["race"].apply(freq1)

dtrain["sex"] = dtrain["sex"].apply(freq1)



dtest["age"] = dtest["age"].apply(freq1)

dtest["workclass"] = dtest["workclass"].apply(freq1)

dtest["marital.status"] = dtest["marital.status"].apply(freq1)

dtest["occupation"] = dtest["occupation"].apply(freq1)

dtest["relationship"] = dtest["relationship"].apply(freq1)

dtest["race"] = dtest["race"].apply(freq1)

dtest["sex"] = dtest["sex"].apply(freq1)
ndtrain = dtrain.dropna()

ndtrain
ndtest = dtest

ndtest.shape
datat=["age","workclass","education.num","marital.status","occupation","relationship","race","sex","capital.gain","capital.loss","hours.per.week"]

data= datat.copy()



data.remove("hours.per.week")

data.remove("sex")

data.remove("race")

#data.remove("relationship")

data.remove("marital.status")

data.remove("workclass")

data.remove("occupation")

Xdtrain = ndtrain[data]
Ydtrain = ndtrain.income
Xdtest = ndtest[data]
knn = KNeighborsClassifier(n_neighbors=28, p=1)
scores = cross_val_score(knn, Xdtrain, Ydtrain, cv=13)
scores
#For tests

#accuracy = np.mean(scores)

#accuracy
knn.fit(Xdtrain, Ydtrain)
Xdtest
dtest
Ytestp = knn.predict(Xdtest)
Ytestp
accuracy = np.mean(scores)

accuracy
id_index = pd.DataFrame({'Id' : list(range(len(Ytestp)))})

income = pd.DataFrame({'income' : Ytestp})

result = income

result
result.to_csv("submission.csv", index = True, index_label = 'Id')