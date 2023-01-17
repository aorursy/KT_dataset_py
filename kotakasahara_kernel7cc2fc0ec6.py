import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from datetime import datetime 

import category_encoders as ce

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import PowerTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm_notebook as tqdm

import random

random.seed(20)

import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

from statistics import mean, median,variance,stdev

import lightgbm as lgb

from lightgbm import LGBMClassifier
#df = pd.read_csv('../input/train.csv')

#df = df.sample(n=30000, random_state=20)

#df_train, df_test = train_test_split(df, test_size=0.2, random_state=20)
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_gdp = pd.read_csv("../input/US_GDP_by_State.csv")

df_state = pd.read_csv("../input/statelatlong.csv")

df_spi = pd.read_csv("../input/spi.csv")
year = []

for y in list(df_train["issue_d"]):

    year.append(int(y[4:8]))

df_train["year"] = year



year_test = []

for y in list(df_test["issue_d"]):

    year_test.append(int(y[4:8]))

df_test["year"] = year_test
years = [2007, 2008, 2009, 2010, 2011, 2012]

add_list = []

for y in years:

    for state in set(df_gdp["State"]):

        add = []

        add.append(state)

        add.append(int(df_gdp["State & Local Spending"][df_gdp.year == 2013][df_gdp.State == state]))

        add.append(float(df_gdp["Gross State Product"][df_gdp.year == 2013][df_gdp.State == state]))

        add.append(float(df_gdp["Real State Growth %"][df_gdp.year == 2013][df_gdp.State == state]))

        add.append(float(df_gdp["Population (million)"][df_gdp.year == 2013][df_gdp.State == state]))

        add.append(y)

        add_list.append(add)

        df_gdp.append(pd.DataFrame(add_list, columns=["State", "State & Local Spending", "Gross State Product", "Real State Growth %", "Population (million)", "year"]))



df_gdp = pd.merge(df_gdp, df_state, left_on = "State", right_on = "City", how='left')



date_spi = []

for d in list(df_spi['date']):

    date_temp = datetime.strptime(d, '%d-%b-%y')

    date_spi.append(date_temp.strftime('%b-%Y'))

df_spi["date"] = date_spi

df_spi = df_spi.groupby("date").mean()



df_train = pd.merge(df_train, df_gdp, left_on=['addr_state','year'], right_on = ['State_y','year'], how ='left')

df_test = pd.merge(df_test, df_gdp, left_on=['addr_state','year'], right_on = ['State_y','year'], how ='left')
df_train = pd.merge(df_train, df_spi, left_on='issue_d', right_on ='date', how = 'left')

df_test = pd.merge(df_test, df_spi, left_on='issue_d', right_on ='date', how = 'left')
#y_train = df_train.loan_condition

#y_test = df_test.loan_condition

#df_train = df_train.drop(["loan_condition", "issue_d"], axis=1)

#df_test = df_test.drop(["issue_d"], axis=1)
df_train["loan_sal_ratio"] = df_train["loan_amnt"]/df_train["annual_inc"]

df_test["loan_sal_ratio"] = df_test["loan_amnt"]/df_test["annual_inc"]

df_train["loan_sal_ratio"] = df_train["loan_sal_ratio"].replace(np.inf, 1)

df_test["loan_sal_ratio"] = df_test["loan_sal_ratio"].replace(np.inf, 1)
df_train["collections_12_mths_ex_med*mths_since_last_major_derog"] = df_train["collections_12_mths_ex_med"]*df_train["mths_since_last_major_derog"]

df_train["mths_since_last_major_derog/open_acc"] = df_train["mths_since_last_major_derog"]/df_train["open_acc"]

df_train["loan_amnt*dti"] = df_train["loan_amnt"]*df_train["dti"]

df_train["pub_rec/revol_util"] = df_train["pub_rec"]/df_train["revol_util"]

df_train["pub_rec/installment"] = df_train["pub_rec"]/df_train["installment"]

df_train["pub_rec/loan_amnt"] = df_train["pub_rec"]/df_train["loan_amnt"]

df_train["pub_rec/total_acc"] = df_train["pub_rec"]/df_train["total_acc"]

df_train["dti*installment"] = df_train["dti"]*df_train["installment"]

df_train["total_acc*pub_rec"] = df_train["total_acc"]*df_train["pub_rec"]

df_train["collections_12_mths_ex_med*mths_since_last_major_derog"] = df_train["collections_12_mths_ex_med*mths_since_last_major_derog"].replace(np.inf,1).replace(-np.inf,1)

df_train["mths_since_last_major_derog/open_acc"] = df_train["mths_since_last_major_derog/open_acc"].replace(np.inf,1).replace(-np.inf,1)

df_train["loan_amnt*dti"] = df_train["loan_amnt*dti"].replace(np.inf,1).replace(-np.inf,1)

df_train["pub_rec/revol_util"] = df_train["pub_rec/revol_util"].replace(np.inf,1).replace(-np.inf,1)

df_train["pub_rec/installment"] = df_train["pub_rec/installment"].replace(np.inf,1).replace(-np.inf,1)

df_train["pub_rec/loan_amnt"] = df_train["pub_rec/loan_amnt"].replace(np.inf,1).replace(-np.inf,1)

df_train["pub_rec/total_acc"] = df_train["pub_rec/total_acc"].replace(np.inf,1).replace(-np.inf,1)

df_train["dti*installment"] = df_train["dti*installment"].replace(np.inf,1).replace(-np.inf,1)

df_train["total_acc*pub_rec"] = df_train["total_acc*pub_rec"].replace(np.inf,1).replace(-np.inf,1)
df_test["collections_12_mths_ex_med*mths_since_last_major_derog"] = df_test["collections_12_mths_ex_med"]*df_test["mths_since_last_major_derog"]

df_test["mths_since_last_major_derog/open_acc"] = df_test["mths_since_last_major_derog"]/df_test["open_acc"]

df_test["loan_amnt*dti"] = df_test["loan_amnt"]*df_test["dti"]

df_test["pub_rec/revol_util"] = df_test["pub_rec"]/df_test["revol_util"]

df_test["pub_rec/installment"] = df_test["pub_rec"]/df_test["installment"]

df_test["pub_rec/loan_amnt"] = df_test["pub_rec"]/df_test["loan_amnt"]

df_test["pub_rec/total_acc"] = df_test["pub_rec"]/df_test["total_acc"]

df_test["dti*installment"] = df_test["dti"]*df_test["installment"]

df_test["total_acc*pub_rec"] = df_test["total_acc"]*df_test["pub_rec"]

df_test["collections_12_mths_ex_med*mths_since_last_major_derog"] = df_test["collections_12_mths_ex_med*mths_since_last_major_derog"].replace(np.inf,1).replace(-np.inf,1)

df_test["mths_since_last_major_derog/open_acc"] = df_test["mths_since_last_major_derog/open_acc"].replace(np.inf,1).replace(-np.inf,1)

df_test["loan_amnt*dti"] = df_test["loan_amnt*dti"].replace(np.inf,1).replace(-np.inf,1)

df_test["pub_rec/revol_util"] = df_test["pub_rec/revol_util"].replace(np.inf,1).replace(-np.inf,1)

df_test["pub_rec/installment"] = df_test["pub_rec/installment"].replace(np.inf,1).replace(-np.inf,1)

df_test["pub_rec/loan_amnt"] = df_test["pub_rec/loan_amnt"].replace(np.inf,1).replace(-np.inf,1)

df_test["pub_rec/total_acc"] = df_test["pub_rec/total_acc"].replace(np.inf,1).replace(-np.inf,1)

df_test["dti*installment"] = df_test["dti*installment"].replace(np.inf,1).replace(-np.inf,1)

df_test["total_acc*pub_rec"] = df_test["total_acc*pub_rec"].replace(np.inf,1).replace(-np.inf,1)
columns_raw = df_train.columns

columns_raw_test = df_test.columns
df_train['fold'] = df_train.loan_amnt.apply( lambda x: random.randint(0,5))
columns = df_train.columns

columns_hasnull = []

for col in columns:

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
col_miss = ["emp_length", "title", "emp_title"]

train_Miss, test_Miss = MissingColumns(df_train[col_miss], df_test[col_miss],col_miss)
for i in columns_hasnull:

    if df_train[i].dtype == "float64":

        df_train[i].fillna(df_train[i].median(), inplace=True)

        df_test[i].fillna(df_train[i].median(), inplace=True)

    if df_train[i].dtype == "object":

        df_train[i].fillna("Kuhaku", inplace=True)

        df_test[i].fillna("Kuhaku", inplace=True)
df_train = df_train.join(train_Miss)

df_test = df_test.join(test_Miss)
def OrdinalEncoder(traindf, testdf, category):

    oe = ce.OrdinalEncoder(cols=category, return_df=True)

    names = list(map(lambda x: x + "_OrdinalEncoder", category))

    trainout = oe.fit_transform(traindf[category])

    testout = oe.transform(testdf[category])

    trainout.columns = names

    testout.columns = names

    for name in names:

        trainout[name].fillna(trainout[name].median(), inplace=True)

        testout[name].fillna(trainout[name].median(), inplace=True)

    return trainout, testout



def OneHotEncoder(traindf, testdf, category):

    ohe = ce.OneHotEncoder(cols=category, handle_unknown='impute')

    trainout = ohe.fit_transform(traindf[category])

    testout = ohe.transform(testdf[category])

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
col_CE = [

#"earliest_cr_line",

]

train_CE, test_CE = CountEncoder(df_train, df_test, col_CE)
col_OE = [

#    "initial_list_status",

#    "zip_code",

#    "earliest_cr_line",

]

train_OE, test_OE = OrdinalEncoder(df_train, df_test, col_OE)
col_TE = [

        "initial_list_status",

    "zip_code",

    "earliest_cr_line",

"title",

"grade",

"sub_grade",

"emp_length",

"home_ownership",

"purpose",

"addr_state"

]

train_TE, test_TE = TargetEncoder(df_train, df_test, col_TE, "loan_condition")
for i in list(df_train.columns):

    if df_train[i].dtype!="object":

        print(i)
def StandardScale(train, test, columns):

    scaler = StandardScaler()

    scaler.fit(train[columns])

    trainout = train[columns].copy()

    testout = test[columns].copy()

    trainout[columns] = scaler.transform(train[columns])

    testout[columns] = scaler.transform(test[columns])

    names = list(map(lambda x: x + "_StandardScale", columns))

    trainout.columns = names

    testout.columns = names

    for name in names:

        trainout[name].fillna(trainout[name].median(), inplace=True)

        testout[name].fillna(trainout[name].median(), inplace=True)

    return trainout, testout



def MinMaxScale(train, test, columns):

    scaler = MinMaxScaler()

    scaler.fit(train[columns])

    trainout = train[columns].copy()

    testout = test[columns].copy()

    trainout[columns] = scaler.transform(trainout[columns])

    testout[columns] = scaler.transform(testout[columns])

    names = list(map(lambda x: x + "_MinMaxScale", columns))

    trainout.columns = names

    testout.columns = names

    for name in names:

        trainout[name].fillna(trainout[name].median(), inplace=True)

        testout[name].fillna(trainout[name].median(), inplace=True)

    return trainout, testout



def LogTransform(train, test, columns):

    trainout = train[columns].copy()

    testout = test[columns].copy()

    for col in columns:

        logval = []

        for i in list(train[col]):

            tes = np.sign(i)*np.log(abs(i))

            logval.append(tes)

        trainout[col] = logval

        

        logval_test = []

        for i in list(test[col]):

            tes = np.sign(i)*np.log(abs(i))

            logval_test.append(tes)

        testout[col] = logval_test

    names = list(map(lambda x: x + "_Log", columns))

    trainout.columns = names

    testout.columns = names

    for name in names:

        trainout[name].fillna(trainout[name].median(), inplace=True)

        testout[name].fillna(trainout[name].median(), inplace=True)

    return trainout, testout



def BoxCoxTransform(train, test, columns):

    pt = PowerTransformer(method='yeo-johnson')

    pt.fit(train[columns])

    trainout = train[columns].copy()

    testout = test[columns].copy()

    trainout[columns] = pt.transform(train[columns])

    testout[columns] = pt.transform(test[columns])

    names = list(map(lambda x: x + "_BoxCox", columns))

    trainout.columns = names

    testout.columns = names

    for name in names:

        trainout[name].fillna(trainout[name].median(), inplace=True)

        testout[name].fillna(trainout[name].median(), inplace=True)

    return trainout, testout
col_SS = [

    "loan_amnt",

    "installment",

    "dti",

    "delinq_2yrs",

    "inq_last_6mths",

    "pub_rec",

    "revol_util",

    "State & Local Spending",

    "Gross State Product",

    "Population (million)",

    "close",

    "tot_coll_amt",

    "loan_sal_ratio",

    "mths_since_last_major_derog",

    "collections_12_mths_ex_med*mths_since_last_major_derog",

    #"mths_since_last_major_derog/open_acc",

    #"loan_amnt*dti",

    #"pub_rec/revol_util",

    #"pub_rec/installment",

    #"pub_rec/loan_amnt",

    #"pub_rec/total_acc",

    #"dti*installment",

    #"total_acc*pub_rec",

]

train_SS, test_SS = StandardScale(df_train, df_test, col_SS)
col_log = [

    "annual_inc",

    "open_acc",

    "revol_bal",

    "total_acc",

    "tot_cur_bal",

    "Real State Growth %"

]

train_log, test_log = LogTransform(df_train, df_test, col_log)
col_MinMax = [

    "mths_since_last_record"

]

train_MinMax, test_MinMax = MinMaxScale(df_train, df_test, col_MinMax)
col_BoxCox = [

 "mths_since_last_delinq"

]

train_BoxCox, test_BoxCox = BoxCoxTransform(df_train, df_test, col_BoxCox)
col_text = [

    "emp_title"

]
def TFIDF(trainlist, testlist):

    features = 30

    vec = TfidfVectorizer(min_df = 1, max_features = features)

    alltxt = trainlist + testlist

    vec.fit(alltxt)

    trainout = pd.DataFrame(vec.transform(trainlist).toarray())

    testout = pd.DataFrame(vec.transform(testlist).toarray())

    names = []

    for f in range(features):

        txt = "txt_" + str(f)

        names.append(txt)

    trainout.columns = names

    testout.columns = names

    return trainout, testout
train_TXT, test_TXT = TFIDF(list(df_train["emp_title"]), list(df_test["emp_title"]))
df_train.columns
df_train = df_train.join([

    train_CE,

    train_OE,

    train_TE,

    train_SS,

    train_log,

    train_MinMax,

    train_BoxCox,

    train_TXT

])

df_test = df_test.join([

    test_CE,

    test_OE,

    test_TE,

    test_SS,

    test_log,

    test_MinMax,

    test_BoxCox,

    test_TXT

])
X_train = df_train.drop(columns_raw, axis = 1).drop("fold", axis = 1)

y_train = df_train.loan_condition

X_test = df_test.drop(columns_raw_test, axis = 1)
X = pd.concat([X_train, X_test])

y = pd.DataFrame(np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))]))

skf = StratifiedKFold(n_splits=3, random_state=71, shuffle=True)

df_split = pd.DataFrame

n = 0

for train_ix, test_ix in skf.split(X,y):

    X_train_, y_train_ = X.iloc[train_ix], y.iloc[train_ix]

    X_val, y_val = X.iloc[test_ix], y.iloc[test_ix]

    

    clf = LGBMClassifier(boosting_type='gbdt', 

                         class_weight=None,

                         colsample_bytree=0.71,

                        importance_type="split",

                        learning_rate=0.05,

                        max_depth=-1,

                        min_child_samples=20,

                        min_child_weight=0.001,

                        min_split_gain=0.0,

                        n_estimators=9999,

                        n_jobs=-1,

                        num_leaves=31,

                        objective=None,

                        random_state=71, 

                         reg_alpha=1.0,

                        reg_lambda=1.0,

                        silent=True,

                        subsample=0.9, subsample_for_bin=200000,

                        subsample_freq=0)

    clf.fit(X_train_, y_train_, early_stopping_rounds=200,eval_metric='auc',eval_set=[(X_val,y_val)])

    y_pred = clf.predict_proba(X_val)[:,1]



    df_temp = X_val.copy()

    df_temp["y_val"] = list(y_val[0])

    df_temp["y_pred"] = list(y_pred)

    print("test pred ", roc_auc_score(y_val, y_pred))

    if n == 0:

        df_split = df_temp

    else:

        df_split = pd.concat([df_split, df_temp])

    n = n+1
train_index = df_split[df_split["y_val"]==0].drop("y_val", axis=1).sort_values(by=["y_pred"], ascending=False)[:round(len(df_split)*0.25)].index
X_train = df_train.drop(train_index).drop(columns_raw, axis =1).drop("fold", axis=1)

y_train = df_train.drop(train_index).loan_condition

X_val = df_train.iloc[train_index].drop(columns_raw, axis = 1).drop("fold", axis=1)

y_val = df_train.iloc[train_index].loan_condition
clf = LGBMClassifier(boosting_type='gbdt', 

                         class_weight=None,

                         colsample_bytree=0.71,

                        importance_type="split",

                        learning_rate=0.05,

                        max_depth=-1,

                        min_child_samples=20,

                        min_child_weight=0.001,

                        min_split_gain=0.0,

                        n_estimators=9999,

                        n_jobs=-1,

                        num_leaves=31,

                        objective=None,

                        random_state=71, 

                         reg_alpha=1.0,

                        reg_lambda=1.0,

                        silent=True,

                        subsample=0.9, subsample_for_bin=200000,

                        subsample_freq=0)

#X_train_, X_val, y_train_, y_val = train_test_split(X_train, y_train, random_state=42)

clf.fit(X_train, y_train, early_stopping_rounds=200,eval_metric='auc',eval_set=[(X_val,y_val)])
y_pred = clf.predict_proba(X_test)[:,1]

submission = pd.read_csv("../input/sample_submission.csv")

submission.loan_condition = y_pred

submission.to_csv("submission.csv", index=False)