# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from sklearn.model_selection import LeaveOneOut

from statsmodels import api as sm

from sklearn import linear_model

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from sklearn import preprocessing

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv') 

df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

tr_len = len(df_train)

te_len = len(df_test)
print("covid_train")

print(df_train.shape)

print(df_train.columns)

print(df_train.head())

print("covid_test")

print(df_test.shape)

print(df_test.columns)
df_train['Days'] = pd.to_datetime(df_train['Date']).sub(pd.Timestamp('2020-01-21')).dt.days

df_test['Days'] = pd.to_datetime(df_test['Date']).sub(pd.Timestamp('2020-01-21')).dt.days
df_all = pd.concat([df_train,df_test],axis=0)

df_all = df_all.reset_index()

print(df_all)

#print("Original features:\n", list(df_all.columns), "\n")

dummies = pd.get_dummies(df_all["Country_Region"])

#print("Features after get_dummies:\n", list(dummies.columns))

dummies

df_all_trans = pd.concat([df_all, dummies],axis=1)
all_length = len(df_all_trans)

date_num = df_all_trans["Date"].values

df_all_trans = df_all_trans.drop(columns=['Date', 'Country_Region'])

#date_num
for i in range(all_length):

    a,b,c = date_num[i].split("-")

    num_date = int(a)*10000+int(b)*100+int(c)

    #print(num_date)

    if i==0:

        num_dates = num_date

    else:

        num_dates = np.vstack((num_dates,num_date))

df_temp = pd.DataFrame(num_dates, columns=["Date_transer"])

df_end = pd.concat([df_all_trans, df_temp],axis=1)
train = df_end[0:tr_len]

test = df_end[tr_len:]

print(test)

train_lavel = pd.concat([train["Fatalities"],train["ConfirmedCases"]],axis=1)

train_lavel

x_data_1 = train.drop(columns=["Fatalities","ConfirmedCases"])

x_data_1 = x_data_1.drop(columns=["Province_State","ForecastId","Id"])

x_data_2 = train.drop(columns=["ConfirmedCases","Province_State","ForecastId","Id"])



test_pre = test.drop(columns=["Fatalities","ConfirmedCases"])

test_pre = test_pre.drop(columns=["Province_State","ForecastId","Id"])

last_test = test_pre.values





x_data = x_data_1.values

x_data_2 = x_data_2.values

y_data_1 = train["Fatalities"]

y_data_1 = y_data_1.values

y_data_2 = train["ConfirmedCases"]

y_data_2 = y_data_2.values
print(test.shape)

print(x_data.shape)
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

import xgboost as xgb
def XGBReg(X_tra,y_tra,X_eval,y_eval):

    #xgboostモデルの作成

    #reg = xgb.XGBRegressor()

    # ハイパーパラメータ探索

    #reg_cv = GridSearchCV(reg, {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}, verbose=1)

    #reg_cv.fit(X_tra, y_tra)

    #print(reg_cv.best_params_, reg_cv.best_score_)



# 改めて最適パラメータで学習

    #reg = xgb.XGBRegressor(**reg_cv.best_params_)

    reg = xgb.XGBRegressor(max_depth = 4, n_estimators = 100)

    reg.fit(X_tra, y_tra)

    y_pred = reg.predict(X_eval)

    

    return y_pred
y_ = train["Fatalities"].values
df_all = pd.concat([df_train,df_test],axis=0)

print(df_all)

#print("Original features:\n", list(df_all.columns), "\n")

dummies = pd.get_dummies(df_all["Country_Region"])

#print("Features after get_dummies:\n", list(dummies.columns))

dummies

df_all_trans = pd.concat([df_all, dummies],axis=1)

#df_all = pd.concat([df_train, df_test])

df_all_trans
n=5

kf = KFold(n_splits=n,shuffle = True)

#skf = StratifiedKFold(n_splits=10)

kunikunosaku = []

#STANDARD = True

STANDARD = False



for train_index, eval_index in kf.split(x_data,y_data_1):

    X_tra, X_eval = x_data[train_index],x_data[eval_index]

    y_tra, y_eval = y_data_1[train_index], y_data_1[eval_index]

    print(train_index)

    print(train_index.shape)

    print(eval_index)

    print(eval_index.shape)

    

    if STANDARD==True:

        #標準化

        #scaler = MinMaxScaler(feature_range=(0, 1))

        scaler = StandardScaler()

        scaler.fit(X_tra)

        X_tra = scaler.transform(X_tra)

        X_eval = scaler.transform(X_eval)

        

        #教師なし

        #untamed = TSNE(n_components=2, random_state=0)

        untamed = PCA(n_components=100)

        untamed.fit(X_tra)

        ev_ratio = untamed.explained_variance_ratio_

        print("sum = ",sum(ev_ratio))

        

        

        

        X_tra = untamed.transform(X_tra)

        X_eval = untamed.transform(X_eval)

    

    #clf = linear_model.LassoCV()

    #result = clf.fit(X_tra,y_tra)

    #y_pred = clf.predict(X_eval)

    y_pred = XGBReg(X_tra,y_tra,X_eval,y_eval)

    plt.scatter(y_eval, y_pred)

    print(y_pred.shape)

    connection = np.vstack([eval_index,y_pred])

    print(connection.shape)

    kunikunosaku.append(connection)

for i in range(n):

    kunikunosaku_np = kunikunosaku[i].transpose(1, 0)

    clone = pd.DataFrame(kunikunosaku_np)

    clone.columns = ['Row', 'Fatalities']

    clone = clone.set_index('Row')

    if i==0:

        connectior = clone

    else:

        connectior = pd.concat([connectior, clone],axis=1)

    

connectior = connectior.fillna(0)

Fata_predict = connectior.sum(axis=1)

Fata_predict

clonez = pd.DataFrame(Fata_predict)

clonez=clonez.reset_index()

clonez = clonez.drop(columns=["Row"])

clonez

data_plus = pd.concat([x_data_1,clonez],axis=1)

data_plus = data_plus.drop(columns=["index"])

add_data = data_plus.values
n=5

kf = KFold(n_splits=n,shuffle = True)

#skf = StratifiedKFold(n_splits=10)

predict_ans = []

true_ans = []

#STANDARD = True



for train_index, eval_index in kf.split(add_data,y_data_2):

    X_tra, X_eval = add_data[train_index],add_data[eval_index]

    y_tra, y_eval = y_data_2[train_index], y_data_2[eval_index]

    

    if STANDARD==True:

        #標準化

        #scaler = MinMaxScaler(feature_range=(0, 1))

        scaler = StandardScaler()

        scaler.fit(X_tra)

        X_tra = scaler.transform(X_tra)

        X_eval = scaler.transform(X_eval)

        

        untamed = PCA(n_components=100)



        

        untamed.fit(X_tra)

        ev_ratio = untamed.explained_variance_ratio_

        print("sum = ",sum(ev_ratio))



        X_tra = untamed.transform(X_tra)

        X_eval = untamed.transform(X_eval)

        

    y_pred = XGBReg(X_tra,y_tra,X_eval,y_eval)

    

    #clf = linear_model.LassoCV()

    #result = clf.fit(X_tra,y_tra)

    #y_pred = clf.predict(X_eval)

    

    plt.scatter(y_eval, y_pred)

    print(y_pred.shape)
reg = xgb.XGBRegressor(max_depth = 4, n_estimators = 100)

STANDARD=False



if STANDARD==True:

    scaler = StandardScaler()

    scaler.fit(x_data)

    x_data = scaler.transform(x_data)

    last_test = scaler.transform(last_test)

    untamed = PCA(n_components=100)

    untamed.fit(x_data)

    x_data = untamed.transform(x_data)

    last_test = untamed.transform(last_test)



reg.fit(x_data,y_data_1)

print(x_data.shape)

print(last_test.shape)

y_pred_1 = reg.predict(last_test)

y_pred_1 = y_pred_1.reshape([13459,1])

cl = pd.DataFrame(y_pred_1)

cl.columns = ["Fatalities"]
test_pre = test_pre.reset_index()

STANDARD==True
data_pl = pd.concat([test_pre,cl],axis=1)



data_pl = data_pl.drop(columns=["index"])

add_data_last = data_pl.values

add_data_last.shape



if STANDARD==True:

    scaler = StandardScaler()

    scaler.fit(x_data_2)

    x_data_2 = scaler.transform(x_data_2)

    add_data_last = scaler.transform(add_data_last)

    untamed = PCA(n_components=100)

    untamed.fit(x_data_2)

    x_data_2 = untamed.transform(x_data_2)

    add_data_last = untamed.transform(add_data_last)
reg = xgb.XGBRegressor(max_depth = 4, n_estimators = 100)

reg.fit(x_data_2,y_data_2)

y_pred_2 = reg.predict(add_data_last)

y_pred_2 = y_pred_2.reshape([13459,1])

cler = pd.DataFrame(y_pred_2)
result = np.hstack([y_pred_2,y_pred_1])

result.shape
result_csv = pd.DataFrame(result)

result_csv
df_test["ForecastId"]

#result_csv

df_end = pd.concat([df_test["ForecastId"], result_csv],axis=1)
df_end.columns = ['ForecastId', 'ConfirmedCases', 'Fatalities']
df_end
df_end.to_csv('submission.csv', index=False)