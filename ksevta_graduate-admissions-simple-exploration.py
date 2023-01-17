import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
import warnings
warnings.simplefilter('ignore')
import missingno as msn
os.listdir('../input/')
data_1 = pd.read_csv('../input/Admission_Predict.csv')
data_2 = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
print("Data Columns: ",np.array(data_1.columns.tolist()))
#original columns has trailing sapaces in LOR and Chance of Admit. changed Serial no. with Id
cols = ['Id', 'GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
       'LOR', 'CGPA', 'Research', 'Chance of Admit']
data_1.columns = cols
data_2.columns = cols
data_1.head()
data_2.head()
print("data_1 length: ",len(data_1))
print("data_2 length: ",len(data_2))
(data_1 != data_2[:len(data_1)]).sum()
data = data_2.copy()
msn.matrix(data,figsize=(10,5))
data.describe()
data.info()
## numerical and categorical columns
num_cols = ['GRE Score','TOEFL Score', 'CGPA']
cat_cols = ['University Rating', 'SOP','LOR']
bin_cols = ['Research']
g = sns.pairplot(data,hue='Research',vars=num_cols,diag_kind='kde')
g = sns.catplot(x='Research',y='CGPA',data=data,kind='box')
g.fig.set_size_inches([10,5])
plt.figure(figsize=(10,5))
sns.distplot(data[data['Research'] == 0]['CGPA'],label='0',bins=20)
sns.distplot(data[data['Research'] == 1]['CGPA'],label='1',bins=20)
plt.legend()
plt.show()
print("Num of students with research experience: ",len(data[data.Research == 1]))
print("Num of students with NO research experience: ",len(data[data.Research == 0]))
print("Avg CGPA of students with research experience: ", data[data.Research == 1]['CGPA'].mean())
print("Avg CGPA of students with NO research experience: ", data[data.Research == 0]['CGPA'].mean())
CGPA_thre = 7.5
print("Num of students with CGPA >= {}: {}".format(CGPA_thre,len(data[data.CGPA >= CGPA_thre])))
print("Num of students with CGPA < {}: {}".format(CGPA_thre,len(data[data.CGPA < CGPA_thre])))
print("percentage of students with CGPA >= {} and go for research: {}".format(CGPA_thre,
                                                                                   len(data[(data.CGPA>=CGPA_thre)&(data.Research == 1)])/len(data[data.CGPA >= CGPA_thre])))
print("percentage of students with CGPA < {} and go for research: {}".format(CGPA_thre,
                                                                                   len(data[(data.CGPA<CGPA_thre)&(data.Research == 1)])/len(data[data.CGPA < CGPA_thre])))
fig,[axs1,axs2] = plt.subplots(ncols=2,figsize=(20,5))
sns.scatterplot(x='GRE Score',y='CGPA',data=data,hue='SOP',ax=axs1,size='SOP')
sns.scatterplot(x='GRE Score',y='CGPA',data=data,hue= 'LOR',ax=axs2,size='LOR')
fig.show()
fig,axs = plt.subplots(nrows=2,ncols=2,figsize=(20,10))
sns.countplot(x='LOR',data=data,ax=axs[0,0])
sns.countplot(x='SOP',data=data,ax=axs[0,1])
sns.countplot(x='University Rating',data=data,ax=axs[1,0])
sns.countplot(x='Research',data=data,ax=axs[1,1])
fig.show()
_,axs = plt.subplots(ncols=3,figsize=(30,5))
sns.scatterplot(x='CGPA',y='Chance of Admit',data=data,hue='Research',ax=axs[0])
sns.scatterplot(x='GRE Score',y='Chance of Admit',data=data,hue='Research',ax=axs[1])
sns.scatterplot(x='TOEFL Score',y='Chance of Admit',data=data,hue='Research',ax=axs[2])
sns.set_style('whitegrid')
g = sns.catplot(x='SOP',y='Chance of Admit',data=data,kind='swarm',col='Research')
g.fig.set_size_inches([20,5])
g.set_xticklabels("")
sns.set_style('whitegrid')
g = sns.catplot(x='LOR',y='Chance of Admit',data=data,kind='swarm',col='Research')
g.fig.set_size_inches([20,5])
g.set_titles("")
g = sns.relplot(x='CGPA',y='Chance of Admit',data=data,col='Research',hue='University Rating',size='University Rating')
g.fig.set_size_inches([20,5])
g = sns.catplot(x='Research',y='Chance of Admit',data=data,kind='box')
g.fig.set_size_inches([10,5])
cols = ['GRE Score','TOEFL Score','SOP','LOR','CGPA','Chance of Admit']
g = sns.heatmap(data[cols].corr(),annot=True,xticklabels=cols,yticklabels=cols,fmt=".2f")
g.figure.set_size_inches([10,10])
X_cols = ['GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research']
target = 'Chance of Admit'
X = data[X_cols]
y = data['Chance of Admit']
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from xgboost import plot_importance
rfc = RandomForestRegressor(max_depth=3)
xgb = XGBRegressor()
rfc.fit(X,y)
xgb.fit(X,y)
df_feature_imp = pd.DataFrame()
df_feature_imp['feature_name'] = X.columns
df_feature_imp['xgb'] = xgb.feature_importances_
df_feature_imp['rfc'] = rfc.feature_importances_
df_feature_imp
_,axs = plt.subplots(ncols=2,sharey=True,figsize=(10,5))
sns.barplot(x='xgb',y='feature_name',data=df_feature_imp,ax=axs[0])
sns.barplot(x='rfc',y='feature_name',data=df_feature_imp,ax=axs[1])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
xgb = XGBRegressor(max_depth=3,learning_rate=0.1)
xgb.fit(X_train,y_train)
from sklearn.metrics import mean_squared_error
print("Using XGB")
print("RMSE score on training data: ",np.sqrt(mean_squared_error(y_true=y_train,y_pred=xgb.predict(X_train))))
print("RMSE score on testing data: ",np.sqrt(mean_squared_error(y_true=y_test,y_pred=xgb.predict(X_test))))
rfc = RandomForestRegressor(max_depth=3)
rfc.fit(X_train,y_train)
print("Using rfc")
print("RMSE score on training data: ",np.sqrt(mean_squared_error(y_true=y_train,y_pred=rfc.predict(X_train))))
print("RMSE score on testing data: ",np.sqrt(mean_squared_error(y_true=y_test,y_pred=rfc.predict(X_test))))
df_eval_train = pd.DataFrame()
df_eval_test = pd.DataFrame()

df_eval_train['rfc'] = rfc.predict(X_train)
df_eval_train['xgb'] = xgb.predict(X_train)
df_eval_train['target'] = y_train.values

df_eval_test['rfc'] = rfc.predict(X_test)
df_eval_test['xgb'] = xgb.predict(X_test)
df_eval_test['target'] = y_test.values
df_eval_train.sort_values('target',inplace=True)
df_eval_train.reset_index(drop=True,inplace=True)
_,axs = plt.subplots(2,2,figsize=(20,10))
axs[0,0].plot(df_eval_train['xgb'])
axs[0,0].plot(df_eval_train['target'])
axs[0,1].plot(df_eval_train['rfc'])
axs[0,1].plot(df_eval_train['target'])

axs[1,0].plot(df_eval_test['target'])
axs[1,0].plot(df_eval_test['xgb'],'o')
axs[1,1].plot(df_eval_test['target'])
axs[1,1].plot(df_eval_test['rfc'],'o')
axs[0,0].legend(['target','XGB'])
axs[0,1].legend(['target','RFC'])

axs[0,0].set_ylabel('Train')
axs[1,0].set_ylabel('Test')

axs[1,0].set_xlabel('XGB')
axs[1,1].set_xlabel('RFC')
df_eval_test['rfc_diff'] = np.square(df_eval_test['target']-df_eval_test['rfc'])
df_eval_test['xgb_diff'] = np.square(df_eval_test['target']-df_eval_test['xgb'])
plt.figure(figsize=(10,5))
plt.plot(df_eval_test['rfc_diff'],'o')
plt.plot(df_eval_test['xgb_diff'],'o')
plt.legend(['RFC','XGB'])