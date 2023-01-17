# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_audi=pd.read_csv('../input/used-car-dataset-ford-and-mercedes/audi.csv')

pd.set_option('display.max_columns',None)

df_audi.head()
df_audi.describe()
df_audi.isnull().sum()
filt=df_audi['engineSize']==0

df_audi.loc[filt,'engineSize']=1.9
df_audi['engineSize'].unique()
df=pd.get_dummies(df_audi)
df.columns
sns.countplot('fuelType',hue='transmission',data=df_audi)

plt.show()
sns.lineplot(x='year',y='price',data=df_audi)

plt.show()
sns.barplot(x='fuelType',y='price',data=df_audi)

plt.show()
sns.boxplot(x='transmission',y='price',data=df_audi)

plt.show()
sns.distplot(df_audi['price'])

plt.show()
sns.boxplot(df_audi['mpg'])

plt.show()
sns.distplot(df_audi['mpg'])

plt.show()
df.drop(columns=['model_ A1','transmission_Automatic','fuelType_Diesel'],axis=0,inplace=True)
corr=df.corr()



corr=pd.DataFrame(corr)



filt=corr['price']>0.35

corr.loc[filt]
filt=corr['price']<(-0.35)

corr.loc[filt]
col=['year','price','tax','engineSize','model_ Q7','mileage','mpg','transmission_Manual']
df=df[col]

corr_new=df.corr()



sns.heatmap(corr_new,annot=True,cmap='coolwarm')

plt.show()
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X=df.drop(columns='price',axis=0)

y=df['price']

y=y.values.reshape(-1,1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)

sc_y = StandardScaler()

y_train = sc_y.fit_transform(y_train)

y_test = sc_y.fit_transform(y_test)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
reg=RandomForestRegressor()

param_grid = { 

            "n_estimators"      : [10,20,30,40,50],

            "max_features"      : ["auto", "sqrt", "log2"],

            }



grid = GridSearchCV(estimator=reg, param_grid=param_grid, n_jobs=-1, cv=5)

grid.fit(X_train, y_train)

print(grid.best_params_)
reg=RandomForestRegressor(n_estimators=40,max_features='log2')

reg.fit(X_train,y_train)

y_pred=reg.predict(X_test)
from sklearn.metrics import mean_squared_error,r2_score



forest_rmse = np.sqrt(mean_squared_error(y_test,y_pred))

forest_r2score = r2_score(y_test,y_pred)

print("R2 score is ", forest_r2score)

print("rmse is ", forest_rmse )