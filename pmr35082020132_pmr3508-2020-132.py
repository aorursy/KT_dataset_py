import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
train = pd.read_csv("../input/adult-pmr3508/train_data.csv",sep=',',
        engine='python',
        na_values="?")
test = pd.read_csv("../input/adult-pmr3508/test_data.csv",
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
train.shape
train.head()
test.shape
test.head()
train = train.dropna()
test = test.dropna()
train.describe()
def percent(colum):
    return colum*100//float(colum[-1])
targetxeducationnum = pd.crosstab(train["age"],train["income"],margins=True)
targetxeducationnum.apply(percent,axis=1).plot()
targetxeducationnum = pd.crosstab(train["workclass"],train["income"],margins=True)
targetxeducationnum.apply(percent,axis=1).plot()
targetxeducationnum = pd.crosstab(train["education.num"],train["income"],margins=True)
targetxeducationnum.apply(percent,axis=1).plot()
targetxeducationnum = pd.crosstab(train["marital.status"],train["income"],margins=True)
targetxeducationnum.apply(percent,axis=1).plot()
targetxeducationnum = pd.crosstab(train["relationship"],train["income"],margins=True)
targetxeducationnum.apply(percent,axis=1).plot()
targetxeducationnum = pd.crosstab(train["race"],train["income"],margins=True)
targetxeducationnum.apply(percent,axis=1).plot()
targetxeducationnum = pd.crosstab(train["sex"],train["income"],margins=True)
targetxeducationnum.apply(percent,axis=1).plot()
targetxeducationnum = pd.crosstab(train["hours.per.week"],train["income"],margins=True)
targetxeducationnum.apply(percent,axis=1).plot()
not_number = ["Male", "Female", "Asian-Pac-Islander", "White", "Black", "Amer-Indian-Eskimo", "Other", "Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
def num_func(label):
    for i in range(len(not_number)):
        if label == not_number[i]:
            return i
    return label
#função aplicada para raça:
train["race"] = train["race"].apply(num_func)
test["race"] = test["race"].apply(num_func)
#função aplicada para sexo:
train["sex"] = train["sex"].apply(num_func)
test["sex"] = test["sex"].apply(num_func)
#função aplicada para estado civil:
train["marital.status"] = train["marital.status"].apply(num_func)
test["marital.status"] = test["marital.status"].apply(num_func)
train.head()
Xtrain = train[["age", "education.num", "marital.status", "race", "sex", "hours.per.week"]]
Ytrain = train.income
Xtest = test[["age", "education.num", "marital.status", "race", "sex", "hours.per.week"]]
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=28, p=1)
scores = cross_val_score(knn, Xtrain, Ytrain, cv=10)
scores
knn.fit(Xtrain, Ytrain)
YtestPred = knn.predict(Xtest)
accuracy = np.mean(scores)
accuracy
id_index = pd.DataFrame({'Id' : list(range(len(YtestPred)))})
income = pd.DataFrame({'income' : YtestPred})
result = income
result
result.to_csv("submission.csv", index = True, index_label = 'Id')