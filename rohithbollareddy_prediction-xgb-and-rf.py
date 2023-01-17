# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/car-price-estimation/datasets_33080_43333_car data.csv')

df.info()
print('Transmission types :',df['Transmission'].unique())
print('Fuel type: ',df['Fuel_Type'].unique())
print('Seller Type: ',df['Seller_Type'].unique())

#missing values
df.isnull().any()
df.describe()
df['Current_year']=2020
df['Age_of_Car']=df['Current_year']-df['Year']
df.drop(['Year','Current_year'],axis=1,inplace=True)
df.drop(['Car_Name'],axis=1,inplace=True)
df=pd.get_dummies(df,drop_first=True)
#correlation
correlation=df.corr
sns.pairplot(df)
corr_mat=df.corr()
top_corr=corr_mat.index
plt.figure(figsize=(10,10))
sns.heatmap(df[top_corr].corr(),annot=True,cmap='Spectral_r')
df.columns
X=df.iloc[:,1:]
Y=df['Selling_Price']
from sklearn.ensemble import ExtraTreesRegressor
et=ExtraTreesRegressor()
et.fit(X,Y)
print(et.feature_importances_)
imp_fea=pd.Series(et.feature_importances_,index=X.columns)
imp_fea.nlargest(5).plot(kind='barh')
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
from xgboost import XGBRegressor
xgb=XGBRegressor()
n_estimators=[int(x) for x in np.linspace(start=100,stop=1500,num=12)]
max_features=['auto','sqrt']
max_depth=[int(x) for x in np.linspace(5,30,num=6)]
min_sample_split=[2,4,6,10,100]
min_sample_leaf=[1,2,5,10]
from sklearn.model_selection import RandomizedSearchCV
random_grid={'n_estimators':n_estimators,
             'max_features':max_features,
             'max_depth':max_depth,
             'min_sample_split':min_sample_split,
              'min_sample_leaf':min_sample_leaf}
random_grid_rf={'n_estimators':n_estimators,
             'max_features':max_features,
             'max_depth':max_depth}
print(random_grid)
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid_rf,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2,
                                random_state=42, n_jobs = 1)
xgb_random = RandomizedSearchCV(estimator = xgb, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2,
                                random_state=42, n_jobs = 1)
rf_random.fit(X_train,Y_train)
xgb_random.fit(X_train,Y_train)
pred_rf=rf_random.predict(X_test)
pred=xgb_random.predict(X_test)
sns.distplot(Y_test-pred)
sns.distplot(Y_test-pred)