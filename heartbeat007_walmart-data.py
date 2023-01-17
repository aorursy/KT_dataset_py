import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold
train = pd.read_csv('../input/train.csv')

train = train[['Store','Dept','Date','Weekly_Sales','IsHoliday']]
test = pd.read_csv('../input/test.csv')
train.head()
test.head()
feature = pd.read_csv('../input/features.csv')
feature.head()
## we have to merge train already has the isHoliday so we have to drop the isHoliday in feature

feature = feature.drop('IsHoliday',1)



store = pd.read_csv('../input/stores.csv',header=0)
store.head()
## merging the train feature and the store



dataset=train.merge(store,how='left').merge(feature,how='left')
dataset.head()
def scatter(dataset,column):

    plt.figure()

    plt.scatter(dataset[column],dataset['Weekly_Sales'])

    plt.ylabel('WEEKLY SALES')

    plt.xlabel(column)

    
scatter(dataset, 'Fuel_Price')

def plot(dataset,column):

    plt.figure()

    plt.plot(dataset[column],dataset['Weekly_Sales'])

    plt.ylabel('WEEKLY SALES')

    plt.xlabel(column)

    
plot(dataset, 'Fuel_Price')
plt.plot(dataset['Weekly_Sales'],dataset['Date'])
## can plot the weekly sales based on the DEPT and The Store



def feature_based_result_plot(feture1,feture2):

    

    ## we plot based every two combination related to weekly sales

    ## not everything is compltable

    for x,data in dataset.groupby([feture1,feture2]):

        plt.title(x)

        plt.scatter(range(len(data)),group['Weekly_Sales'])

        plt.show()

        

        
#feature_based_result_plot('Store','Dept')
## Whats the problm using classification???

## cause these are continuous value
# Data Manupulation

d={True:1,False:0}
dataset['IsHoliday']=dataset['IsHoliday'].map(d)
dataset.head()
######## very useful

dataset = pd.get_dummies(dataset, columns=["Type"])
## ok now the markdown is is anothre pain 

## huge ammount of missing data

dataset[['MarkDown1','MarkDown2','MarkDown3','MarkDown4', 'MarkDown5']] = dataset[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].fillna(0)
dataset.head()
dataset['Month']=pd.to_datetime(dataset['Date']).dt.month
dataset.head()
#we gonna drop the Date column its useless now

dataset = dataset.drop('Date',1)



X = dataset.drop('Weekly_Sales',1)

Y= dataset['Weekly_Sales']
X=np.array(X)

Y=np.array(Y)
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.3)

## now prepare the test dataset

dataset_test=test.merge(store,how='left').merge(feature,how='left')
d={True:1,False:0}

dataset_test['IsHoliday']=dataset_test['IsHoliday'].map(d)
dataset_test.head()
dataset_test = pd.get_dummies(dataset_test, columns=["Type"])
dataset_test[['MarkDown1','MarkDown2','MarkDown3','MarkDown4', 'MarkDown5']] = dataset_test[['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']].fillna(0)
dataset_test['Month']=pd.to_datetime(dataset_test['Date']).dt.month
dataset_test = dataset_test.drop('Date',1)
dataset_test.head()
## this dataset is the all the value except the weekly_sale just train the model and predict the y value with this
test=np.array(dataset_test)
## thats it now apply regression algorithm

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor
ML=[]

M=['SVR','KNeighborsRegressor','MLPRegressor','LinearRegression','RandomForestRegressor']

Z=[SVR(),KNeighborsRegressor(),MLPRegressor(),LinearRegression(),RandomForestRegressor()]

for model in Z:

    model.fit(X_train,y_train)      ## training the model this could take a little time

    accuracy=model.score(X_test,y_test)    ## comparing result with the test data set

    ML.append(accuracy)   ## saving the accuracy



d={'Accuracy':ML,'Algorithm':M}

df1=pd.DataFrame(d)
df1