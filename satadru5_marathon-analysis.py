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

import xgboost as xgb

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/MarathonData.csv")
df.head(3)
sns.countplot(df['CATEGORY'])
sns.countplot(df['Category'])
sns.countplot(df['CrossTraining'])
sns.countplot(df['Marathon'])
df.duplicated('Name')
names=df.Name.value_counts()

names[names > 1]
df[df.Name =='Tomas Drabek']
df.head(3)
plt.plot(df['MarathonTime'])
plt.hist(df['MarathonTime'])
df['MarathonTime'].skew()
df['km4week'].skew()
#hot encoding

from sklearn import model_selection, preprocessing

for c in df.columns:

    if df[c].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(df[c].values)) 

        df[c] = lbl.transform(list(df[c].values))

        #x_train.drop(c,axis=1,inplace=True)
df.head(3)
y_train = df['MarathonTime']

x_train = df.drop(["MarathonTime"], axis=1)
x_train.head(2)
x_train['km4week'].skew()
x_train['sp4week']=np.log(x_train.sp4week)
x_train['sp4week'].skew()
x_train.head(3)
#create test and training data

from sklearn.model_selection import train_test_split

data_train, data_test, label_train, label_test = train_test_split(x_train, y_train, test_size = 0.2, random_state = 42)
#Check with XGBOOST Model

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
#without log transform

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,

    verbose_eval=50, show_stdv=False)
num_boost_rounds = len(cv_output)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)
fig, ax = plt.subplots(1, 1, figsize=(8, 13))

xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
#without log transformation

dtest=xgb.DMatrix(data_test)
#without log

y_predict = model.predict(dtest)

out = pd.DataFrame({'Actual_Time': label_test, 'predict_Time': y_predict,'Diff' :(label_test-y_predict)})

out[['Actual_Time','predict_Time','Diff']].head(5)
#log transformation

sns.regplot(out['predict_Time'],out['Diff'])
#Check with Random Forest

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 100 , oob_score = True, random_state = 42)

rf.fit(data_train, label_train)

rf_score_train = rf.score(data_train, label_train)

print("Training score: ",rf_score_train)

rf_score_test = rf.score(data_test, label_test)

print("Testing score: ",rf_score_test)
importance = rf.feature_importances_

importance = pd.DataFrame(importance, index=data_train.columns, 

                          columns=["Importance"])
importance=importance.sort_values(ascending=False,by="Importance")
importance
x = range(importance.shape[0])

y = importance.ix[:, 0]

plt.bar(x, y, align="center")

plt.show()
#Gridsearch

from sklearn.grid_search import GridSearchCV

model = RandomForestRegressor(random_state=30)

param_grid = { "n_estimators"      : [250, 300,500],

           "max_features"      : [3, 5],

           "max_depth"         : [10, 20],

           "min_samples_split" : [2, 4]}

grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=2)
grid_search.fit(data_train, label_train)

print (grid_search.best_params_)
#Check with Random Forest with tuned parameter

from sklearn.ensemble import RandomForestRegressor

rf_tn = RandomForestRegressor(n_estimators = 500 ,max_depth=10,min_samples_split=2, oob_score = True, random_state = 42)

rf_tn.fit(data_train, label_train)

rf_tn_score_train = rf_tn.score(data_train, label_train)

print("Training score: ",rf_tn_score_train)

rf_tn_score_test = rf.score(data_test, label_test)

print("Testing score: ",rf_tn_score_test)
grid_search.best_score_