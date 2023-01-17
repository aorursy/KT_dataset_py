# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno

plt.style.use("classic")

from scipy import stats

pd.set_option("display.max_columns",None)

pd.set_option("display.max_rows",None)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/car-price-prediction/CarPrice_Assignment.csv")

df.head()
df.info()
#Checking Null Values

n = msno.bar(df,color='gold')
plt.figure(figsize=(8,6))

sns.jointplot(x = 'carlength', y = 'price',data= df,kind = 'kde',color='coral')

plt.show()
plt.figure(figsize=(8,6))

fig,ax = plt.subplots(2,3,figsize=(10,8))

sns.regplot(x = 'carlength', y = 'price',data= df,color='coral',ax=ax[0][0])

sns.regplot(x = 'wheelbase', y = 'price',data= df,color='coral',ax=ax[0][1])

sns.regplot(x = 'enginesize', y = 'price',data= df,color='coral',ax=ax[0][2])

sns.regplot(x = 'horsepower', y = 'price',data= df,color='coral',ax=ax[1][0])

sns.countplot(x='fueltype',hue = 'enginetype', data= df,ax=ax[1][1])

sns.countplot(x='doornumber',hue = 'carbody', data= df,ax=ax[1][2])



plt.tight_layout()

plt.show()
plt.figure(figsize=(8,6))

sns.distplot(df['price'],fit = stats.norm,color='coral')

plt.show()
df['price'] = np.log(df['price'].values)

plt.figure(figsize=(8,6))

sns.distplot(df['price'],fit = stats.norm,color='coral')

plt.show()
plt.figure(figsize=(10,9))

sns.boxplot(data=df,palette='Set3')

plt.xticks(rotation=90)

plt.show()
plt.figure(figsize=(8,6))

fig,ax = plt.subplots(3,3,figsize=(10,8))

sns.distplot(df['compressionratio'], fit = stats.norm,color='coral',ax=ax[0][0])

sns.distplot(df['enginesize'], fit = stats.norm,color='coral',ax=ax[0][1])

sns.distplot(df['horsepower'], fit = stats.norm,color='coral',ax=ax[0][2])

sns.distplot(df['wheelbase'], fit = stats.norm,color='coral',ax=ax[1][0])

sns.distplot(df['carwidth'], fit = stats.norm,color='coral',ax=ax[1][1])

sns.distplot(df['curbweight'], fit = stats.norm,color='coral',ax=ax[1][2])

sns.distplot(df['citympg'], fit = stats.norm,color='coral',ax=ax[2][0])

sns.distplot(df['highwaympg'], fit = stats.norm,color='coral',ax=ax[2][1])

sns.distplot(df['stroke'], fit = stats.norm,color='coral',ax=ax[2][2])



plt.tight_layout()

plt.show()
plt.rcParams['figure.figsize']=(10,8)

plt.style.use("classic")

color = ['yellowgreen','gold','lightskyblue','coral','pink','orange']

explode = [0,0,0,0.01,0,0.4]

df['symboling'].value_counts().plot.pie(y='symboling',explode=explode,colors=color,startangle=50,shadow=True,autopct='%0.1f%%')
plt.rcParams['figure.figsize']=(8,6)

sns.countplot(x='enginelocation',hue = 'doornumber', data= df)

plt.show()
plt.rcParams['figure.figsize'] =(9,8)

sns.catplot(x="carbody", hue="fueltype", col="doornumber",

                data=df, kind="count",

                height=6, aspect=.7,palette='Set3')

plt.show()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['CarName'] = le.fit_transform(df['CarName'])

df['fueltype'] = le.fit_transform(df['fueltype'])

df['aspiration'] = le.fit_transform(df['aspiration'])

df['doornumber'] = le.fit_transform(df['doornumber'])

df['carbody'] = le.fit_transform(df['carbody'])

df['drivewheel'] = le.fit_transform(df['drivewheel'])

df['enginelocation'] = le.fit_transform(df['enginelocation'])

df['enginetype'] = le.fit_transform(df['enginetype'])

df['cylindernumber'] = le.fit_transform(df['cylindernumber'])

df['fuelsystem'] = le.fit_transform(df['fuelsystem'])
from sklearn.model_selection import train_test_split

x = df.drop(['car_ID','price'],axis=1)

y = df['price']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from mlxtend import regressor

from sklearn import metrics,linear_model,model_selection

from sklearn.metrics import r2_score,mean_squared_error,mean_squared_log_error

from sklearn.model_selection import cross_val_score
lg =LGBMRegressor(n_estimators=1500)

lg.fit(x_train,y_train)
y_pred_lg = lg.predict(x_test)

lgbmetrics = pd.DataFrame({'Model': 'LightGBM',

                          'r2score':r2_score(y_test,y_pred_lg),

                          'MSE': metrics.mean_squared_error(y_test,y_pred_lg),

                           'RMSE': np.sqrt(metrics.mean_squared_error(y_test,y_pred_lg)),

                           'MSLE': metrics.mean_squared_log_error(y_test,y_pred_lg),

                           'RMSLE':np.sqrt(metrics.mean_squared_log_error(y_test,y_pred_lg))             

                          },index=[1])



lgbmetrics
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=15,criterion='mse',random_state=25)

rf.fit(x_train,y_train)
rf_pred = rf.predict(x_test)

rfMetrics = pd.DataFrame({'Model': 'Random Forest',

                          'r2score':r2_score(y_test,rf_pred),

                          'MSE': metrics.mean_squared_error(y_test,rf_pred),

                           'RMSE': np.sqrt(metrics.mean_squared_error(y_test,rf_pred)),

                           'MSLE': metrics.mean_squared_log_error(y_test,rf_pred),

                           'RMSLE':np.sqrt(metrics.mean_squared_log_error(y_test,rf_pred))             

                          },index=[2])



rfMetrics
frames = [lgbmetrics,rfMetrics]

TrainingResult = pd.concat(frames)

TrainingResult.style.background_gradient(cmap='Blues')
rf_pred = np.exp(rf_pred)

y_test = np.exp(y_test)

actualvspredicted = pd.DataFrame({"Actual":y_test,"Predicted":rf_pred})

actualvspredicted.head().style.background_gradient(cmap='Blues')
plt.figure(figsize=[8,6])

sns.regplot(actualvspredicted['Predicted'],actualvspredicted['Actual'],truncate=False)

plt.title('Actual vs Predicted')

plt.xlabel('Predicted')

plt.ylabel('Actual')