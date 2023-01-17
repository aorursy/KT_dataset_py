# Load Libraries

import pandas as pd

import numpy as np



# Data Visualization

import matplotlib.pyplot as plt

import seaborn as sns



# Pre Processing

from sklearn.preprocessing import StandardScaler, MinMaxScaler



# Model Selection

from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV



# Model

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor



# Metrics

from sklearn.metrics import mean_squared_error, r2_score



# Feature Selection

from sklearn.feature_selection import SelectKBest, chi2



# Warnings 

import warnings as ws

ws.filterwarnings('ignore')



# Save Model

import dill



sns.set_style('whitegrid')

pd.pandas.set_option('display.max_columns',None)
# Read Dataset

data = pd.read_csv('../input/brasilian-houses-to-rent/houses_to_rent_v2.csv')

data.head()
print('Shape of the data : {}'.format(data.shape))
data.columns.groupby(data.dtypes)
data['floor'].value_counts()
data['floor'] = pd.to_numeric(data['floor'].replace({'-':0}))
data.isnull().sum()
data.nunique()
data.hist(bins = 25, figsize = (10,10))

plt.show()
def outlier_detection(feature):

    plt.figure(figsize = (15,3))

    # Box Plot

    plt.subplot(1,3,1)

    sns.boxplot(x = feature,showfliers = False)

    plt.title('Box Plot')

    

    # Distribution Plot

    plt.subplot(1,3,2)

    sns.distplot(feature, kde = False)

    plt.title('Distribution Plot')

    

    # After Log Distriburion plot

    plt.subplot(1,3,3)

    sns.distplot(np.log(feature+1), kde = False)

    plt.title('Distribution Plot after log')

    

    min = feature.min()

    

    result = pd.DataFrame(columns = ['Min','25%','50%','75%','95%','Max'])

    result.loc[0] = [feature.min(),np.quantile(feature,.25),np.quantile(feature,.50),np.quantile(feature,.75),np.quantile(feature,.95), feature.max()]

    

    print(result)

# Area

outlier_detection(data['area'])
# hoa (R$)

outlier_detection(data['hoa (R$)'])
data['hoa (R$)'][data['hoa (R$)'] == 0].count()
#rent amount

outlier_detection(data['rent amount (R$)'])
outlier_detection(data['property tax (R$)'])
data['property tax (R$)'][data['property tax (R$)'] == 0].count()
outlier_detection(data['total (R$)'])
# Area

outlier_detection(data['floor'])
['city', 'floor', 'animal', 'furniture']
# City

sns.countplot(x = 'city', data = data)
# Floor

sns.countplot(x = 'animal', data = data)
sns.countplot(x = 'furniture', data = data)
sns.pairplot(data)
def pair_plot(feature, hue = None):

    plt.figure(figsize = (15,3))

    plt.subplot(1,2,1)

    sns.scatterplot(x = data['total (R$)'], y = feature, hue = hue)

    plt.title('Pairplot')

    

    plt.subplot(1,2,2)

    sns.scatterplot( x = np.log(data['total (R$)']), y = np.log(feature+1), hue = hue)

    plt.title('Pairplot after log transform')

    
pair_plot(data['area'], data['city'])
pair_plot(data['rent amount (R$)'], data['city'])
pair_plot(data['hoa (R$)'], data['city'])
pair_plot(data['property tax (R$)'], data['city'])
pair_plot(data['fire insurance (R$)'], data['city'])
pair_plot(data['floor'], data['city'])
sns.boxplot(x = 'animal', y = 'total (R$)', data = data, showfliers = False)
sns.boxplot(x = 'furniture', y = 'total (R$)', data = data, showfliers = False)
sns.boxplot(x = 'city', y = 'total (R$)', data = data, showfliers = False)
sns.boxplot(x = 'rooms', y = 'total (R$)', data = data, showfliers = False)
sns.boxplot(x = 'bathroom', y = 'total (R$)', data = data, showfliers = False)
sns.boxplot(x = 'parking spaces', y = 'total (R$)', data = data, showfliers = False)
# Correlation Map

corr = data.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

# Heatmap

plt.figure(figsize=(15, 10))

sns.heatmap(corr, annot = True, fmt = '.2f', mask = mask, linewidths = 2, cmap = 'YlGnBu')

plt.plot()
df['animal'] = df['animal'].replace({'not acept':0, 'acept':1})

df['furniture'] = df['furniture'].replace({'not furnished':0, 'furnished':1})
df['city'] = pd.get_dummies(df['city'], drop_first= True)
X = df.drop(['total (R$)'], axis = 1)

Y = df['total (R$)']



x_train, x_test,y_train, y_test = train_test_split(X,Y, test_size = 0.3, random_state = 7)
# Models

# Models

models = []

models.append(('LR', LinearRegression()))

models.append(('LASSO', Lasso()))

models.append(('Ridge', Ridge()))

models.append(('EN', ElasticNet()))
results = []

names = []



col = ['Model', 'MSE Mean','MSE Std']

model_result = pd.DataFrame(columns = col)

i= 0

for name, model in models:

    kfold = KFold(n_splits = 10, random_state = 42)

    cv_results = cross_val_score(model, x_train, y_train, cv = kfold, scoring = 'neg_mean_squared_error')

    

    results.append(cv_results)

    names.append(name)

    

    model_result.loc[i] = [name, cv_results.mean(), cv_results.std()]

    i +=1



    

plt.figure(figsize = (7,4))

sns.boxplot(x = names , y= results, showfliers = False)

plt.tight_layout()

plt.title('MSE')

plt.show()



model_result.sort_values('MSE Mean',ascending = False)

model_result
lr = LinearRegression()

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

mse = mean_squared_error(y_test, y_pred)

print('RMSE Value is : {:.5f}'.format(np.sqrt(mse)))
y_test_log = np.log(y_test+1)

y_pred_log = np.log(y_pred+1)

sns.regplot(x = y_pred_log, y = y_test_log)
err = y_test - y_pred

sns.distplot(err)