import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv("../input/sf-crime/train.csv.zip", parse_dates=['Dates'])

test = pd.read_csv("../input/sf-crime/test.csv.zip", parse_dates=['Dates'], index_col='Id')

sampleSubmission = pd.read_csv("../input/sf-crime/sampleSubmission.csv.zip")
train.head(3)
test.head(3)
train.info()
fe_name = list(test)

df_train = train[fe_name]

df = pd.concat((df_train,test))
target = train['Category']
target.value_counts()
from sklearn.preprocessing import LabelEncoder



LB = LabelEncoder() 

target = LB.fit_transform(target)

print(LB.classes_)
target
date = pd.to_datetime(df['Dates'])

df['Date'] = date.dt.date

df['Year'] = date.dt.year

df['Month'] = date.dt.month

df['Day'] = date.dt.day

df['Hour'] = date.dt.hour
df.drop("Dates", axis = 1, inplace = True)
year = df.groupby('Year').count().iloc[:,0]

month = df.groupby('Month').count().iloc[:,0]

hour = df.groupby('Hour').count().iloc[:,0]

dayofweek = df.groupby('DayOfWeek').count().iloc[:, 0]



figure, axs = plt.subplots(2,2, figsize = (15,10))



sns.barplot(x=year.index, y= year,ax = axs[0][0])

sns.barplot(x=month.index, y= month,ax = axs[0][1])

sns.barplot(x=hour.index, y= hour,ax = axs[1][0])

sns.barplot(x=dayofweek.index, y= dayofweek,ax = axs[1][1])
date = df.groupby('Date').count().iloc[:, 0]
sns.kdeplot(data=date, shade=True)

plt.axvline(x=date.median(), ymax=0.95, linestyle='--')

plt.annotate('Median ' + str(date.median()),xy =(date.median(), 0.005))
lb = LabelEncoder()

df['PdDistrict'] = lb.fit_transform(df["PdDistrict"])
df["PdDistrict"].value_counts()
sns.countplot(df["PdDistrict"])
lb = LabelEncoder()

df['PdDistrict'] = lb.fit_transform(df["PdDistrict"])
df['DayOfWeek'] = lb.fit_transform(df["DayOfWeek"])
df["Address"].value_counts().head(20)
df['block'] = df['Address'].str.contains('block', case=False)

df['ST'] = df['Address'].str.contains('ST', case=False)
df['block'] = lb.fit_transform(df["block"])

df['ST'] = lb.fit_transform(df["ST"])
df.drop("Address", axis = 1, inplace = True)
print(df["X"].min(), df["X"].max())

print(df["Y"].min(), df["Y"].max())
print(len(df.loc[df["X"] >= -120.5, "X"]))

print(len(df.loc[df["Y"] >= 90, "Y"]))
X_median = df[df["X"] < -120.5]["X"].median()

Y_median = df[df["Y"] < 90]["Y"].median()

df.loc[df["X"] >= -120.5, "X"] = X_median

df.loc[df["Y"] >= 90, "Y"] = Y_median
df["X+Y"] = df["X"] + df["Y"]

df["X-Y"] = df["X"] - df["Y"]
df.drop("Date", axis = 1, inplace = True)
new_train = df[:train.shape[0]]

new_test = df[train.shape[0]:]
new_train.head()
import lightgbm as lgb



train_data = lgb.Dataset(new_train, label=target, categorical_feature=["PdDistrict", "DayOfWeek"])

params = {'boosting':'gbdt',

          'objective':'multiclass',

          'num_class':39,

          'max_delta_step':0.9,

          'min_data_in_leaf': 21,

          'learning_rate': 0.4,

          'max_bin': 465,

          'num_leaves': 41,

          'verbose' : 1}

bst = lgb.train(params, train_data, 120)
predictions = bst.predict(new_test)
submission = pd.DataFrame(predictions,columns=LB.inverse_transform(np.linspace(0, 38, 39, dtype='int16')),index=new_test.index)

#submission.to_csv('LGB.csv', index_label='Id')
import xgboost as xgb

train_xgb = xgb.DMatrix(new_train, label=target)

test_xgb  = xgb.DMatrix(new_test)
params = {

    'max_depth': 4,  

    'eta': 0.3,  

    'silent': 1, 

    'objective': 'multi:softprob', 

    'num_class': 39,

}



xg = xgb.cv(params, train_xgb, nfold=3, early_stopping_rounds=10, metrics='mlogloss', verbose_eval=True) 
train_xgb = xgb.train(params, train_xgb, 10)

pred_xgb = train_xgb.predict(test_xgb)
submission1 = pd.DataFrame(pred_xgb,columns=LB.inverse_transform(np.linspace(0, 38, 39, dtype='int16')),index=new_test.index)

#submission1.to_csv('XGB.csv', index_label='Id')
ensemble = 0.9*predictions + 0.1*pred_xgb
sub = pd.DataFrame(ensemble,columns=LB.inverse_transform(np.linspace(0, 38, 39, dtype='int16')),index=new_test.index)

sub.to_csv('submission.csv', index_label='Id')