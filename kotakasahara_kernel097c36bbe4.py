import pandas as pd

from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

import random

from sklearn.model_selection import KFold

import lightgbm as lgb

from lightgbm import LGBMRegressor

import numpy as np

from sklearn.metrics import mean_squared_error
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_country = pd.read_csv("../input/country_info.csv")

df_country = df_country.replace(",", ".", regex=True)



n = 0

for i in df_country.columns:

    if n >= 2:

        df_country[i] = df_country[i].astype(float)

    n = n + 1



df_train = pd.merge(df_train, df_country, on= 'Country', how='left')

df_test = pd.merge(df_test, df_country, on='Country', how='left')
rawcolumns = df_train.columns
#for i in df_train.columns:

#    if df_train[i].dtype == "object":

#        continue

#    plt.figure(figsize=[7,7])

#    plt.hist(df_train[i], bins=50, color='r', label='train', alpha=0.5, density=True)

#    if i in df_test.columns:

#        plt.hist(df_test[i], bins=50, color='b', label='test', alpha=0.5, density=True)

#    plt.savefig("hist/"+ i + ".png") 

#    plt.close()
kf = KFold(n_splits=4, random_state=71, shuffle=True)

df_train["fold"] = range(0, len(df_train))

split = kf.split(df_train)

n=0

for i, j in split:

    df_train["fold"].iloc[j] = n

    n = n + 1
columns_hasnull = []

for col in rawcolumns:

    if df_train[col].isnull().sum() > 0:

        columns_hasnull.append(col)
def MissingColumns(train, test, columns):

    trainout = train.copy()

    testout = test.copy()

    for col in columns:

        name = col + "_Miss"

        trainout[name] = list(train[col].isnull())

        trainout = trainout.drop(col,axis=1)

        testout[name] = list(testout[col].isnull())

        testout = testout.drop(col, axis=1)

    return trainout, testout
train_Miss, test_Miss = MissingColumns(df_train[columns_hasnull], df_test[columns_hasnull], columns_hasnull)
for i in columns_hasnull:

    if df_train[i].dtype == "float64":

        df_train[i].fillna(df_train[i].median(), inplace=True)

        df_test[i].fillna(df_train[i].median(), inplace=True)

    if df_train[i].dtype == "object":

        df_train[i].fillna("Kuhaku", inplace=True)

        df_test[i].fillna("Kuhaku", inplace=True)
#df_train = df_train.join(train_Miss)

#df_test = df_test.join(test_Miss)
textlist = [

    "DevType",

    "CommunicationTools",

    "FrameworkWorkedWith"

]
devtype = []

for i in list(df_train["DevType"]):

    devtype.extend(i.split(";"))

for i in set(devtype):

    test = []

    for j in df_train["DevType"]:

        if j.find(i) >= 0:

            test.append(1)

        else:

            test.append(0)

    test2 = []

    for j in df_test["DevType"]:

        if j.find(i) >= 0:

            test2.append(1)

        else:

            test2.append(0)

    df_train[i] = test

    df_test[i] = test2
comtype = []

for i in list(df_train["CommunicationTools"]):

    comtype.extend(i.split(";"))

for i in set(comtype):

    test = []

    for j in df_train["CommunicationTools"]:

        if j.find(i) >= 0:

            test.append(1)

        else:

            test.append(0)

    test2=[]

    for j in df_test["CommunicationTools"]:

        if j.find(i) >= 0:

            test2.append(1)

        else:

            test2.append(0)

    df_train[i] = test

    df_test[i] = test2
fm = []

for i in list(df_train["FrameworkWorkedWith"]):

    fm.extend(i.split(";"))

for i in set(fm):

    test = []

    for j in df_train["FrameworkWorkedWith"]:

        if j.find(i) >= 0:

            test.append(1)

        else:

            test.append(0)

    test2 = []

    for j in df_test["FrameworkWorkedWith"]:

        if j.find(i) >= 0:

            test2.append(1)

        else:

            test2.append(0)

    df_train[i] = test

    df_test[i] = test2
col_cate = []

for i in rawcolumns:

    if df_train[i].dtype == "object":

        if i in textlist:

            continue

        col_cate.append(i)
def TargetEncoder(traindf, testdf, category, target):

    trainout = pd.DataFrame()

    testout = pd.DataFrame()

    for col in category:

        trainout_temp = pd.DataFrame()

        testout_temp = pd.DataFrame()

        n= 0

        name = col + "_TargetEncoder"

        # training

        for i in set(traindf["fold"]):

            label_mean = traindf[traindf["fold"] != i].groupby(col)[target].mean()

            if n == 0:

                trainout_temp = traindf[traindf["fold"] == i][col].map(label_mean)

            else:

                trainout_temp = trainout_temp.append(traindf[traindf["fold"] == i][col].map(label_mean))

            n = n + 1

        trainout[name] = trainout_temp

        trainout[name].fillna(trainout[name].median(), inplace=True)

        

        # test

        label_mean = traindf.groupby(col)[target].mean()

        testout[name] = testdf[col].map(label_mean)

        testout[name].fillna(trainout[name].median(), inplace=True)

    return trainout, testout



def CountEncoder(traindf, testdf, category):

    trainout = pd.DataFrame()

    testout = pd.DataFrame()

    for val in category:

        newname = val + "_CountEncoder"

        count = traindf.groupby(val)[val].count()

        trainout[newname] = traindf.groupby(val)[val].transform('count')

        testout[newname] = traindf.groupby(val)[val].transform('count')

        trainout[newname].fillna(trainout[newname].median(), inplace=True)

        testout[newname].fillna(trainout[newname].median(), inplace=True)

    return trainout, testout
train_TE, test_TE = TargetEncoder(df_train, df_test, col_cate, "ConvertedSalary")
#train_CE, test_CE = CountEncoder(df_train, df_test, textlist)
df_train = df_train.join([train_TE])

df_test = df_test.join([train_TE])
dellist = col_cate

dellist.extend(textlist)

#dellist.append("ConvertedSalary")
X_train = df_train.drop(dellist, axis = 1).drop("fold", axis = 1)

y_train = df_train.ConvertedSalary

X_train = X_train.drop("ConvertedSalary", axis=1)

X_test = df_test.drop(dellist, axis = 1)
y_pred_test = np.zeros(len(X_test))

scores = []

for i, (train_ix, test_ix) in enumerate(kf.split(X_train, y_train)):

    X_tr, y_tr = X_train.values[train_ix], y_train.values[train_ix]

    X_te, y_te = X_train.values[test_ix], y_train.values[test_ix]

    

    clf = LGBMRegressor(

        learning_rate = 0.05,

        num_leaves=31,

        colsample_bytree=0.9,

        subsample=0.9,

        n_estimators=9999,

        random_state=71,

        importance_type='gain'

    )

    

    clf.fit(X_tr, y_tr, early_stopping_rounds=200, eval_metric='RMSLE', eval_set=[(X_te, y_te)], verbose=100)

    y_pred = clf.predict(X_te)

    score = mean_squared_error(y_te, y_pred)

    

    y_pred_test += clf.predict(X_test)

    scores.append(score)

    print('CV Score of Fold_%d is %f' % (i, score))

    

y_pred_test /=4
y_pred_test2 = np.zeros(len(X_test))

scores = []

for i, (train_ix, test_ix) in enumerate(kf.split(X_train, y_train)):

    X_tr, y_tr = X_train.values[train_ix], y_train.values[train_ix]

    X_te, y_te = X_train.values[test_ix], y_train.values[test_ix]

    

    clf = LGBMRegressor(

        learning_rate = 0.05,

        num_leaves=31,

        colsample_bytree=0.9,

        subsample=0.9,

        n_estimators=9999,

        random_state=50,

        importance_type='gain'

    )

    

    clf.fit(X_tr, y_tr, early_stopping_rounds=200, eval_metric='RMSLE', eval_set=[(X_te, y_te)], verbose=100)

    y_pred = clf.predict(X_te)

    score = mean_squared_error(y_te, y_pred)

    

    y_pred_test += clf.predict(X_test)

    scores.append(score)

    print('CV Score of Fold_%d is %f' % (i, score))

    

y_pred_test2 /=4
y_pred_test3 = np.zeros(len(X_test))

scores = []

for i, (train_ix, test_ix) in enumerate(kf.split(X_train, y_train)):

    X_tr, y_tr = X_train.values[train_ix], y_train.values[train_ix]

    X_te, y_te = X_train.values[test_ix], y_train.values[test_ix]

    

    clf = LGBMRegressor(

        learning_rate = 0.05,

        num_leaves=31,

        colsample_bytree=0.9,

        subsample=0.9,

        n_estimators=9999,

        random_state=55,

        importance_type='gain'

    )

    

    clf.fit(X_tr, y_tr, early_stopping_rounds=200, eval_metric='RMSLE', eval_set=[(X_te, y_te)], verbose=100)

    y_pred = clf.predict(X_te)

    score = mean_squared_error(y_te, y_pred)

    

    y_pred_test += clf.predict(X_test)

    scores.append(score)

    print('CV Score of Fold_%d is %f' % (i, score))

    

y_pred_test3 /=4
y_pred_test4 = np.zeros(len(X_test))

scores = []

for i, (train_ix, test_ix) in enumerate(kf.split(X_train, y_train)):

    X_tr, y_tr = X_train.values[train_ix], y_train.values[train_ix]

    X_te, y_te = X_train.values[test_ix], y_train.values[test_ix]

    

    clf = LGBMRegressor(

        learning_rate = 0.05,

        num_leaves=31,

        colsample_bytree=0.9,

        subsample=0.9,

        n_estimators=9999,

        random_state=56,

        importance_type='gain'

    )

    

    clf.fit(X_tr, y_tr, early_stopping_rounds=200, eval_metric='RMSLE', eval_set=[(X_te, y_te)], verbose=100)

    y_pred = clf.predict(X_te)

    score = mean_squared_error(y_te, y_pred)

    

    y_pred_test += clf.predict(X_test)

    scores.append(score)

    print('CV Score of Fold_%d is %f' % (i, score))

    

y_pred_test4 /=4
output = pd.read_csv("../input/sample_submission.csv")
y_pred_all = []

for i in range(len(y_pred_test)):

    y_pred_all.append((y_pred_test[i] + y_pred_test2[i] + y_pred_test3[i] + y_pred_test4[i])/4)
output["ConvertedSalary"] = y_pred_all
output.to_csv("submission.csv", index=False)
output["ConvertedSalary"].mean()
df_train["ConvertedSalary"].mean()