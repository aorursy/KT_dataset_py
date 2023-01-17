# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/vgsales.csv")
df.head(3)
df.isnull().sum()
df=df.fillna(0)
plt.figure(figsize=(10,5))

sns.countplot(df['Genre'])
plt.figure(figsize=(15,5))

sns.countplot(df['Platform'])
platGenre = pd.crosstab(df.Platform,df.Genre)

platGenre.head(5)
platGenreTotal =platGenre.sum(axis=1).sort_values(ascending = False)

plt.figure(figsize=(10,15))

sns.barplot(x=platGenreTotal.values,y=platGenreTotal.index)
df.head(3)
pub=df.groupby('Publisher')['Publisher'].count().sort_values(ascending = False).head(15)

sns.barplot(x=pub.values,y=pub.index)
yr=df.groupby('Year')['Year'].count().sort_values(ascending = False).head(15)

plt.plot(yr)
plt.figure(figsize=(10,5))

sns.pointplot(x=yr.index ,y=yr.values)
from sklearn import model_selection, preprocessing

for c in df.columns:

    if df[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(df[c].values)) 

        df[c] = lbl.transform(list(df[c].values))

        #x_train.drop(c,axis=1,inplace=True)
df.head(3)
corr=df.corr()

corr = (corr)

plt.figure(figsize=(14,14))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws=



{'size': 15},

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.title('Heatmap of Correlation Matrix')
plt.figure(figsize=(10,5))

sns.regplot(x='Rank',y='Global_Sales',data=df)
plt.figure(figsize=(10,5))

sns.regplot(x='NA_Sales',y='Global_Sales',data=df)
df.head(3)
df.dtypes
#Train-Test split

from sklearn.model_selection import train_test_split

label = df.pop('Global_Sales')

data_train, data_test, label_train, label_test = train_test_split(df, label, test_size = 0.2, random_state = 200)
data_train.shape,data_test.shape
import xgboost as xgb

from sklearn.model_selection import KFold, train_test_split, GridSearchCV
xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}
dtrain = xgb.DMatrix(data_train, label_train)
cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

    verbose_eval=50, show_stdv=False)
num_boost_rounds = len(cv_output)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)
fig, ax = plt.subplots(1, 1, figsize=(8, 13))

xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
dtest=xgb.DMatrix(data_test)
y_predict = model.predict(dtest)

out = pd.DataFrame({'Actual_Global_Sales': label_test, 'predict_Global_Sales': y_predict,'Diff' :(label_test-y_predict)})

out[['Actual_Global_Sales','predict_Global_Sales','Diff']].head(5)
sns.regplot(out['predict_Global_Sales'],out['Diff'])
sns.regplot(out['Actual_Global_Sales'],out['Diff'])
data_train.head(3)
lr_data_train=data_train[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']]

lr_data_test=data_test[['NA_Sales','EU_Sales','JP_Sales','Other_Sales']]

lr_label_train=label_train

lr_label_test=label_test
#Linear Regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(lr_data_train, lr_label_train)

lr_score_train = lr.score(lr_data_train, lr_label_train)

print("Training score: ",lr_score_train)

lr_score_test = lr.score(lr_data_test, lr_label_test)

print("Testing score: ",lr_score_test)
y_pre = lr.predict(lr_data_test)
out_lr = pd.DataFrame({'Actual_Global_Sales': lr_label_test, 'Predict_Global_Sales': y_pre,'Diff' :(lr_label_test-y_pre)})

out_lr[['Actual_Global_Sales','Predict_Global_Sales','Diff']].head(5)
out_lr.shape
sns.regplot(out_lr['Predict_Global_Sales'],out_lr['Diff'])
#Ensemble XGBOOST & LINEAR REGRESSOR for train data

en_dtest=xgb.DMatrix(data_train)

y_xgb_pred = model.predict(en_dtest)



y_lr_pred = lr.predict(lr_data_train)



Ensemble=pd.DataFrame({'XGBOOST':y_xgb_pred ,'LINEAR_REG':y_lr_pred ,'GLOBAL_SALES':lr_label_train})

Ensemble.head(5)
corr=Ensemble.corr()

corr = (corr)

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

sns.plt.title('Heatmap of Correlation Matrix')
#Ensemble XGBOOST & LINEAR REGRESSOR for test data

en_dtest_1=xgb.DMatrix(data_test)

y_xgb_pred_1 = model.predict(en_dtest_1)



y_lr_pred_1 = lr.predict(lr_data_test)



Ensemble_test_with_actual=pd.DataFrame({'XGBOOST':y_xgb_pred_1 ,'LINEAR_REG':y_lr_pred_1,'ACTUAL_SALES':lr_label_test})
Ensemble_test=Ensemble_test_with_actual[['XGBOOST','LINEAR_REG']]
Ensemble_test.shape,Ensemble.shape
Ensemble_test.head(3)
#Train-Test split

from sklearn.model_selection import train_test_split

label = Ensemble.pop('GLOBAL_SALES')

X_train, X_test, Y_train, Y_test = train_test_split(Ensemble, label, test_size = 0.3, random_state = 200)
from sklearn import linear_model

clf = linear_model.Lasso(alpha=1e-4)

clf.fit(X_train,Y_train)

tr_scr=clf.score(X_train,Y_train)

print("Training score: ",tr_scr)

ts_scr=clf.score(X_test,Y_test)

print("Testing score: ",ts_scr)

ensm_prd=clf.predict(Ensemble_test)
lr_label_test.shape
Output=pd.DataFrame({'LINEAR_REGRASSOR':Ensemble_test['LINEAR_REG'],'XGBOOST':Ensemble_test['XGBOOST'],'ENSEMBLE':ensm_prd,'ACTUAL_PRICE':Ensemble_test_with_actual['ACTUAL_SALES']})
Output.head(10)