# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.plotting import scatter_matrix
import sys
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
def load_data(data_path,data_fname):
    csv_path = os.path.join(data_path, data_fname)
    return pd.read_csv(csv_path)
filename=filenames[0]
if (filenames[1].find('region')>0):
    filename=filenames[1]
    
df_reg = load_data(dirname,filename)
df_reg.head()
df_reg.shape
df_reg.info()
df_reg.describe(include='all')
df_reg=df_reg.drop('Country',axis=1)
df_reg=df_reg.drop('SNo',axis=1)
df_reg.columns
df_reg.Date.value_counts()
df_reg.Date=pd.to_datetime(df_reg.Date)
df_reg.info()
df_reg_conv=df_reg.loc[:,['RegionCode','RegionName']]
sum_devstd=0
for reg in df_reg_conv.RegionName.value_counts().index:
    print (reg)
    print ('Coded as:')
    print(str(df_reg_conv.loc[df_reg_conv.RegionName==reg].mean()[0]))
    sum_dev_std=+df_reg_conv.loc[df_reg_conv.RegionName==reg].std()[0]
print ('Total Dev Std:')
print(str(sum_dev_std))
    
    
df_reg=df_reg.drop('RegionName',axis=1)
df_reg.set_index('Date')
df_reg.TotalPositiveCases.plot()
train_set,test_set=train_test_split(df_reg,test_size=0.2, random_state=10)
train_set.plot(kind='scatter',x='Longitude',y='Latitude')
train_set.loc[df_reg_conv.RegionCode==19].loc[:,['Latitude','Longitude']].mean()
train_set.plot(kind="scatter", x="Longitude", y="Latitude", alpha=0.4,
    s=train_set['TotalPositiveCases'], figsize=(10,13),
    c='TestsPerformed', cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.legend()
corr_mx=train_set.iloc[:,1:].corr()
col=train_set.iloc[:,1:].columns
sns.set(font_scale=1.5)
hm=sns.heatmap(corr_mx,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':6},yticklabels=col,xticklabels=col)
corr_mx=train_set.iloc[:,5:].corr()
col=train_set.iloc[:,5:].columns
hm=sns.heatmap(corr_mx,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':9},yticklabels=col,xticklabels=col)
col=['TotalHospitalizedPatients','IntensiveCarePatients','CurrentPositiveCases','TotalPositiveCases','HomeConfinement','Recovered','Deaths','TestsPerformed']
axes =scatter_matrix(train_set[col],figsize=(14,10))

for ax in axes.flatten():
    ax.xaxis.label.set_rotation(90)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha('right')

plt.tight_layout()
plt.gcf().subplots_adjust(wspace=0, hspace=0)
plt.show()
corr_mx.NewPositiveCases.sort_values(ascending=False)
train_set_m=train_set.iloc[:,4:]
test_set_m=test_set.iloc[:,4:]
train_set_m=train_set_m.drop('Deaths',axis=1)
train_set_m=train_set_m.drop('Recovered',axis=1)
test_set_m=test_set_m.drop('Deaths',axis=1)
test_set_m=test_set_m.drop('Recovered',axis=1)
y_train_set_m=train_set_m.iloc[:,5]
y_test_set_m=test_set_m.iloc[:,5]
X_train_set_m=train_set_m.drop('NewPositiveCases',axis=1)
X_test_set_m=test_set_m.drop('NewPositiveCases',axis=1)
X_train_set_m.describe()
lin_reg=LinearRegression()
lin_reg.fit(X_train_set_m_std,y_train_set_m)
y_test_predict=lin_reg.predict(X_test_set_m)
lin_rmse=np.sqrt(mean_squared_error(y_test_set_m,y_test_predict))
print(lin_rmse)
rnd_clf=RandomForestRegressor(n_estimators=30)
rnd_clf.fit(X_train_set_m,y_train_set_m)
y_test_predict_rf=rnd_clf.predict(X_test_set_m)
lin_rmse_rf=np.sqrt(mean_squared_error(y_test_set_m,y_test_predict_rf))
print(lin_rmse_rf)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
param_grid=[{'n_estimators':[3,30,50,80,100]}]
grid_search=GridSearchCV(rnd_clf,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(X_train_set_m,y_train_set_m)
grid_search.best_estimator_
cvres=grid_search.cv_results_
for mean_score,params in zip(cvres["mean_test_score"],cvres["params"]):
    print(np.sqrt(-mean_score),params)
rnd_clf=RandomForestRegressor(n_estimators=80)
rnd_clf.fit(X_train_set_m,y_train_set_m)
y_test_predict_rf=rnd_clf.predict(X_test_set_m)
lin_rmse_rf=np.sqrt(mean_squared_error(y_test_set_m,y_test_predict_rf))
print(lin_rmse_rf)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(rnd_clf,X_train_set_m,y_train_set_m,scoring="neg_mean_squared_error",cv=10)

score_rmse=np.sqrt(-scores)
print(score_rmse)
print(score_rmse.mean())
print(score_rmse.std())