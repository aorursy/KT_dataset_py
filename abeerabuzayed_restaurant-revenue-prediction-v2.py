# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#TFI (tab food investments) has provided a dataset with 137 restaurants in the training set, and a test set of 100000 restaurants..

data =  pd.read_csv('../input/restaurant-revenue-prediction/train.csv')

test_data = pd.read_csv('../input/restaurant-revenue-prediction/test.csv')
data.describe()
data.head()
data.columns
data.dtypes
data.info()
data['Type'].unique()
data['City Group'].unique()
data['City'].unique()
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(14,6))

plt.subplot(1,2,1)

sns.boxplot(data.revenue)

plt.subplot(1,2,2)

sns.distplot(data.revenue, bins=20, kde=False)

plt.show()
#City distribution

data["City"].value_counts().plot(kind='bar')
data["Type"].value_counts().plot(kind='bar')
data["City Group"].value_counts().plot(kind='bar')
# Crrelation between revenue and feature (p)s

def numFeaturePlot():

    features=(data.loc[:,'P1':'P37']).columns.tolist()

    plt.figure(figsize=(35,18))

    j=1

    while j<len(features):

        col=features[j-1]

        plt.subplot(6,6,j)

        sorted_grp = data.groupby(col)["revenue"].sum().sort_values(ascending=False).reset_index()

        x_val = sorted_grp.index

        y_val = sorted_grp['revenue'].values

        plt.scatter(x_val, y_val)

        plt.xticks(rotation=60)

        plt.xlabel(col, fontsize=20)

        plt.ylabel('Revenue', fontsize=20)

        j+=1    

    plt.tight_layout()

    plt.show()

numFeaturePlot()
# This method helps in understanding the correlation between the different features and the Revenue.

def featureCatPlot(col):

    

    plt.figure(figsize=(15,6))

    i=1

    if not data[col].dtype.name=='int64' and not data[col].dtype.name=='float64':

        plt.subplot(1,2,i)

        sns.boxplot(x=col,y='revenue',data=data)

        plt.xticks(rotation=60)

        plt.ylabel('Revenue')

        i+=1 

        plt.subplot(1,2,i)

        mean=data.groupby(col)['revenue'].mean()

        level=mean.sort_values().index.tolist()

        data[col]=data[col].astype('category')

        data[col].cat.reorder_categories(level,inplace=True)

        data[col].value_counts().plot()

        plt.xticks(rotation=60)

        plt.xlabel(col)

        plt.ylabel('Counts')       

        plt.show()
featureCatPlot('City Group')

# Splitting 01/31/2018 as 01, 31, 2018

train_date=data['Open Date'].str.split('/', n = 2, expand = True)

data['month']=train_date[0]

data['days']=train_date[1]

data['year']=train_date[2]



test_date=test_data['Open Date'].str.split('/', n = 2, expand = True)

test_data['month']=test_date[0]

test_data['days']=test_date[1]

test_data['year']=test_date[2]

data['month']

featureCatPlot('month')
data.sort_values('revenue', ascending=False)[:20]
top_6= data.sort_values('revenue', ascending=False)[:20]

plt.figure(figsize=(13,12))

plt.title("The top 6 resturants")

sns.barplot(x=top_6['City'], y=top_6['revenue'])



best_month= data.sort_values('revenue', ascending=False)[:20]

plt.figure(figsize=(13,12))



sns.barplot(x=best_month['month'], y=best_month['revenue'])

plt.xticks(rotation=60)
best_type= data.sort_values('revenue', ascending=False)



plt.figure(figsize=(13,12))



sns.barplot(x=best_type['Type'], y=best_type['revenue'])
data.isnull().sum()
data
# Get list of categorical variables



new= data[data.columns[~data.columns.isin(['Open Date','days','year','month'])]]



numerical_features = new.select_dtypes([np.number]).columns.tolist()

categorical_features = new.select_dtypes(exclude = [np.number,np.datetime64]).columns.tolist()

categorical_features

from sklearn import preprocessing

from sklearn.model_selection import train_test_split
#data = data.drop('Id', axis=1)

#test_data = test_data.drop('Id', axis=1)

y= data.revenue



x_train = data[data.columns[~data.columns.isin(['Open Date','revenue'])]]  #train features to be fit in model

x_test = test_data[test_data.columns[~test_data.columns.isin(['Open Date'])]]  #test features

from sklearn.preprocessing import LabelEncoder

# Processing the categorical columns to provide vector form of feature

class DataFrameProcess:

    def __init__(self,df,col):

        self.df =df

        self.col=col

    def dataEncoding(self):

        if self.df[self.col].dtype.name == 'object' or self.df[self.col].dtype.name == 'category':

            le = LabelEncoder()

            self.df[self.col] = le.fit_transform(self.df[self.col])    





def data_transform(df):  

    for col in df.columns:

        data_prcs = DataFrameProcess(df,col)

        data_prcs.dataEncoding()  

data_transform(x_train) 

data_transform(x_test)
x_train.head(5)
#X_train, X_valid, y_train, y_valid = train_test_split(x_train, y, train_size=0.8, test_size=0.2,

 #                                                               random_state=0)
from xgboost import XGBRegressor

from sklearn.ensemble import GradientBoostingRegressor



gbRegr = GradientBoostingRegressor(max_depth=3, random_state=42)

gbRegr.fit(x_train, y)

prediction_rr = gbRegr.predict(x_test)

test_label=pd.read_csv('../input/restaurant-revenue-prediction/sampleSubmission.csv')  # test target

test_label.head(10)
from sklearn.metrics import mean_squared_error

from math import sqrt

label_list=test_label['Prediction'].tolist()
print('Root Mean squared error {}'.format(sqrt(mean_squared_error(label_list, prediction_rr))))

from sklearn import ensemble



params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,

          'learning_rate': 0.05, 'loss': 'ls'}

GBR = ensemble.GradientBoostingRegressor(**params)



GBR.fit(x_train, y)

preds_GBR = GBR.predict(x_test)



GradientBoostingRegressor_RMSE= sqrt(mean_squared_error(label_list, preds_GBR))



print('Root Mean squared error {}'.format(GradientBoostingRegressor_RMSE))
parameters = [{'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], 

                     'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 

                                       0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

                    }]

from sklearn.model_selection import GridSearchCV

gsearch = GridSearchCV(estimator=XGBRegressor(),

                       param_grid = parameters, 

                       scoring='neg_mean_absolute_error',

                       n_jobs=4,cv=3)



gsearch.fit(x_train,y)

gsearch.best_params_, gsearch.best_score_
final_model = XGBRegressor(n_estimators=gsearch.best_params_.get('n_estimators'), 

                           learning_rate=gsearch.best_params_.get('learning_rate'), 

                           n_jobs=4)
final_model.fit(x_train, y)
preds_test = final_model.predict(x_test)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from math import sqrt



rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(x_train, y)

rf_val_predictions = rf_model.predict(x_test)

RMSE = sqrt(mean_squared_error(label_list,rf_val_predictions))

print(RMSE)



submission = pd.DataFrame({

        "Id": test_data["Id"],

        "Prediction": rf_val_predictions

    })

submission.to_csv('submission.csv',header=True, index=False)

print('done')