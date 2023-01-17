import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline

from IPython.display import display

import statistics

import re
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

submission = pd.read_csv("../input/titanic/gender_submission.csv")



raw_train = train.copy()

raw_test = test.copy()
print(train.shape)

print(test.shape)

print(submission.shape)
train.head()
test.head()
submission.head()
train.describe()
test.describe()
train.info()
test.info()
train["Title"] = train.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

print(train["Title"].value_counts())

test["Title"] = test.Name.apply(lambda x: re.search(' ([A-Z][a-z]+)\.', x).group(1))

print(test["Title"].value_counts())

train["Title"].isnull().sum()
title_list = {

        "Capt":       "Officer",

        "Col":        "Officer",

        "Major":      "Officer",

        "Dr":         "Officer",

        "Rev":        "Officer",

        "Jonkheer":   "Royalty",

        "Don":        "Royalty",

        "Sir" :       "Royalty",

        "Countess":   "Royalty",

        "Dona":       "Royalty",

        "Lady" :      "Royalty",

        "Mme":        "Mrs",

        "Ms":         "Mrs",

        "Mrs" :       "Mrs",

        "Mlle":       "Miss",

        "Miss" :      "Miss",

        "Mr" :        "Mr",

        "Master" :    "Master"

}



train["Title"] = train.Title.map(title_list)

test["Title"] = test.Title.map(title_list)

train["Title"].isnull().sum()

train[train["Title"].isnull()]
print(train.groupby("Title")["Survived"].mean())
age_group = train.groupby(["Sex","Pclass","Title"])["Age"]

print(age_group.median())
train["Survived"].plot(figsize=(15,5))
train["Age"].plot.hist()

train["Age"].hist()

train["Age"].hist(grid=True)
train[["Age","Survived"]].boxplot(by="Survived")
pd.concat([train.isnull().sum(), test.isnull().sum()], keys=["train","test"], axis=1, sort=False)
cabin_no = train[train["Cabin"].isnull()]["Survived"].value_counts()

cabin_no[1] / sum(cabin_no) * 100
cabin_yes = train[train["Cabin"].isnull() == False]["Survived"].value_counts()

cabin_yes[1] / sum(cabin_yes) * 100
train["Age"].isnull().sum()
def fill_missing_value(df):

    df = df.copy()

    # set before data

    before = df.isnull().sum()

    ####### fill missing value #######

    #df["Age"] = df["Age"].fillna(df["Age"].median())

    df.loc[df.Age.isnull(), "Age"] = df.groupby(["Sex","Pclass","Title"]).Age.transform("median")



    df["Embarked"] = df["Embarked"].fillna("S")

    

    df["Fare"] = df["Fare"].fillna(-0.5)

    

    df.loc[df["Cabin"].isnull() == False, "Cabin"] = 1

    df.loc[df["Cabin"].isnull(), "Cabin"] = 0

    ##################################

    # set after data

    after = df.isnull().sum()

    # display results

    print(pd.concat([before, after], keys=["before","after"], axis=1))

    return df



print("-- train")

train = fill_missing_value(train)

print("-- test")

test = fill_missing_value(test)



train["Cabin"].value_counts()
train.describe()
test.describe()
train[["Survived","Pclass"]].corr()

train.corr()
train[["Survived","Pclass"]].plot.scatter(x="Pclass", y="Survived", figsize=(5,5))
pd.crosstab(train['Survived'], train['Sex'], margins=True)
train["Age"]
# bining

"""

def bining_age(df):

    bining = pd.cut(df["Age"], range(0,90,10), labels=range(0,8))

    display(bining[:5])

    return bining

"""

def bining_age(df):

    interval = (0, 5, 12, 18, 25, 35, 60, 120)

    cats = ['Babies', 'Children', 'Teen', 'Student', 'Young', 'Adult', 'Senior']

    bining = pd.cut(df.Age, interval, labels=cats)

    return bining



ag = bining_age(train)

mag = bining_age(train[train["Sex"]=="male"])

fag = bining_age(train[train["Sex"]=="female"])
def crosstab_age(bining, data):

    survived = pd.DataFrame({"Age": bining,"Survived": data["Survived"]})

    crosstab = pd.crosstab(survived["Age"], survived["Survived"], margins=True)

    crosstab.loc[:,"Survival Rate"] = round(crosstab.loc[:,1] / crosstab.loc[:,"All"] * 100)

    display(crosstab)

    return crosstab

ct = crosstab_age(ag, train)

mct = crosstab_age(mag, train[train["Sex"]=="male"])

fct = crosstab_age(fag, train[train["Sex"]=="female"])
def create_bar_age(ct, title):

    p1 = plt.bar(range(0,8), ct.loc[:,"Survival Rate"][:8], color="blue", tick_label=range(10,90,10))

    p2 = plt.bar(range(0,8), 100-ct.loc[:,"Survival Rate"][:8], bottom=ct.loc[:,"Survival Rate"][:8], color="gray", tick_label=range(10,90,10))

    plt.legend((p1[0], p2[0]), ("Alive", "Dead"))

    plt.title(title)

    plt.show()



create_bar_age(ct, "Total")

create_bar_age(mct, "Male")

create_bar_age(fct, "Female")
male_y_rate = len(train[(train.Sex == 'male') & (train.Survived == 1)]) / len(train[train.Sex == 'male'])

female_y_rate = len(train[(train.Sex == 'female') & (train.Survived == 1)]) / len(train[train.Sex == 'female'])

sex_y_rate = [male_y_rate, female_y_rate]

plt.bar([0,1], sex_y_rate, tick_label=['male', 'female'], width=0.5)
def update_feature(df):

    df = df.copy()

    # display before data

    display(df.head())

    ######### update feature #########



    #interval = range(0,90,10)

    interval = (0, 5, 12, 18, 25, 35, 60, 120)

    age_bining = pd.cut(df["Age"], interval, labels=False)

    df["Age_c"] = age_bining

    ##################################

    # display after data

    display(df.head())

    display(df.info())

    display(df.describe())

    return df



print("-- train")

train = update_feature(train)

print()

print("-- test")

test = update_feature(test)
display(train["Embarked"].value_counts())

plt.figure(figsize=(12,5))

sns.countplot(x="Embarked", hue="Survived", data=train, palette="hls")

plt.show()
plt.figure(figsize=(12,10))

plt.subplot(2,1,2)

sns.swarmplot(x='Age_c',y="Fare",data=train, hue="Survived", palette="hls")

plt.subplots_adjust(hspace = 0.5, top = 0.9)
plt.figure(figsize=(12,5))

sns.distplot(train[train.Survived == 0]["Fare"], bins=50, color='r')

sns.distplot(train[train.Survived == 1]["Fare"], bins=50, color='g')

plt.show()
quant = (-1, 0, 8, 15, 31, 600)

label_quants = ['NoInf', 'quart_1', 'quart_2', 'quart_3', 'quart_4']

train["Fare_c"] = pd.cut(train.Fare, quant, labels=label_quants)

test["Fare_c"] = pd.cut(test.Fare, quant, labels=label_quants)

print(pd.crosstab(train.Fare_c, train.Survived))



plt.figure(figsize=(12,5))

sns.countplot(x="Fare_c", hue="Survived", data=train, palette="hls")

plt.show()
# family = SibSp + Parch + me

train["family"] = train["SibSp"] + train["Parch"] + 1

test["family"] = test["SibSp"] + test["Parch"] + 1

sns.countplot(x='family', data=train, hue="Survived")

plt.show()
train[:5]
drop_colomns = ["Name","Age","Ticket","Fare","SibSp","Parch"]

x_train = train.drop(drop_colomns, axis=1)

x_test = test.drop(drop_colomns, axis=1)

x_train[:5]
dummies_colomns = ["Title","Sex","Embarked","Age_c","Fare_c"]

prefix_colomns = ["Name","Sex","Emb","Age","Fare"]

x_train = pd.get_dummies(x_train, columns=dummies_colomns, prefix=prefix_colomns, drop_first=True)

x_test = pd.get_dummies(x_test, columns=dummies_colomns, prefix=prefix_colomns, drop_first=True)

display(x_train[:5])

display(x_train.info())
x_train["Cabin"].value_counts()
plt.figure(figsize=(15,12))

sns.heatmap(x_train.astype(float).corr(), vmax=1.0, annot=True)

plt.show()
x_train[:5]
y_train = x_train["Survived"].values
x_train = x_train.drop(["Survived","PassengerId"], axis=1)

x_test = x_test.drop(["PassengerId"], axis=1)



print(x_train.shape)

print(x_test.shape)
y_train[:5]
feature_nums = len(x_train.columns)

feature_nums
feature_columns = x_train.columns

feature_columns
x_train[:5]
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.fit_transform(x_test)
x_train[:2]
from sklearn.model_selection import StratifiedKFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier



algs = [

    RandomForestClassifier,

    DecisionTreeClassifier,

    LogisticRegression,

    LinearSVC,

    SVC,

    KNeighborsClassifier,

    GaussianNB,

    Perceptron,

    SGDClassifier

]



def cross_validate(clf, data, label):

    skf = StratifiedKFold(n_splits=10)

    scores = []

    for train_ix, test_ix in skf.split(data,label):

        clf.fit(data[train_ix], label[train_ix])

        score = clf.score(data[test_ix], label[test_ix])

        scores.append(score)

    return np.mean(scores)



results = {}

for alg in algs:

    clf = alg()

    score = cross_validate(clf, x_train, y_train)

    alg_name = str(type(clf)).split("'")[1].split(".")[-1]

    results[alg_name] = score



print(pd.Series(results).sort_values(ascending=False))
import sklearn

from sklearn.metrics import fbeta_score, make_scorer

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier as RFC



parameters = {

    "n_estimators":[i for i in range(10,100,10)],

    "criterion":["gini","entropy"],

    "max_depth":[i for i in range(1,6,1)],

    'min_samples_split': [2, 4, 10,12,16],

    "random_state":[3],

}

scorer = make_scorer(fbeta_score, beta=0.5)

# create model

clf = sklearn.model_selection.GridSearchCV(RFC(), parameters,cv=5,n_jobs=-1)

clf_fit=clf.fit(x_train, y_train)



predictor=clf_fit.best_estimator_
prediction=predictor.predict(x_train)

table=sklearn.metrics.confusion_matrix(y_train,prediction)

tn,fp,fn,tp=table[0][0],table[0][1],table[1][0],table[1][1]

print("ACC\t{0:.3f}".format((tp+tn)/(tp+fp+fn+tn))) # 正確率: accuracy

print("TPR\t{0:.3f}".format(tp/(tp+fn))) # 再現率: recall

print("PPV\t{0:.3f}".format(tp/(tp+fp))) # 適合率: precision

print("SPC\t{0:.3f}".format(tn/(tn+fp))) # 特異性: specificity

print("F1\t{0:.3f}".format((2*tp)/(2*tp+fp+fn))) # F値: F1 score

     

print(sorted(predictor.get_params(True).items()))
features = feature_columns

importances = predictor.feature_importances_

indices = np.argsort(importances)



plt.figure(figsize=(6,6))

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), features[indices])

plt.show()
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)



X = x_train

Y = y_train



for cv_train, cv_test in kfold.split(X, Y):



    prediction = predictor.predict(X[cv_test])

    table = sklearn.metrics.confusion_matrix(Y[cv_test], prediction)

    tn,fp,fn,tp = table[0][0],table[0][1],table[1][0],table[1][1]

    print("ACC\t{0:.3f}".format((tp+tn)/(tp+fp+fn+tn)))

    

    expect = pd.DataFrame(np.array(prediction).reshape(-1, 1))

    anser = pd.DataFrame(np.array(Y[cv_test]).reshape(-1, 1))

    mistake = (expect + anser) % 2



    ids = pd.DataFrame(cv_test.reshape(-1, 1), columns=["PassengerId"])

    df = pd.concat([mistake, ids], axis=1)

    mistake_ids = df.loc[df[0]==1]

    mistake_data = pd.merge(mistake_ids, train)

    

    dead = mistake_data.loc[mistake_data["Survived"]==0]

    print("死んでるのに生きてるとされた: {0}".format(len(dead)))

    display(dead)

    

    plt.figure(figsize=(12,10))

    plt.subplot(2,1,2)

    sns.swarmplot(x='Age',y="Fare",data=dead, hue="Sex", palette="hls")

    plt.subplots_adjust(hspace = 0.5, top = 0.9)

    plt.show()

    

    alive = mistake_data.loc[mistake_data["Survived"]==1]

    print("生きてるのに死んでるとされた: {0}".format(len(alive)))

    display(alive)

    

    plt.figure(figsize=(12,10))

    plt.subplot(2,1,2)

    sns.swarmplot(x='Age',y="Fare",data=alive, hue="Sex", palette="hls")

    plt.subplots_adjust(hspace = 0.5, top = 0.9)

    plt.show()

    

    break

y_pred=predictor.predict(x_test)

submission = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": y_pred

    })

submission.to_csv('submission.csv', index=False)