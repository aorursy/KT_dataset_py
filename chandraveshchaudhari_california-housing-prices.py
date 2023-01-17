## import important library : we will import other libraries when we need them

#better way is to speculate what you will need and import exactly that thing for saving memory usage.import os

import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
os.listdir()
data= pd.read_csv('/kaggle/input/housing.csv')
data.head()
data.count()
data.total_bedrooms.isnull().sum()
data.loc[data.total_bedrooms.isnull()]
data.ocean_proximity.unique()
data.ocean_proximity.value_counts()
data.groupby(['ocean_proximity'])['total_bedrooms'].sum()
data.groupby(['ocean_proximity'])['total_bedrooms'].mean()
x=4937435/9136

x
import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
corr = data.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corr,cmap="YlGnBu")
data.groupby(['ocean_proximity'])['total_bedrooms'].median()
data.ocean_proximity.head()
## DATA CLEANING

data.loc[(data.total_bedrooms.isnull())&(data.ocean_proximity=='<1H OCEAN'), 'total_bedrooms']=438.0

data.loc[(data.total_bedrooms.isnull())&(data.ocean_proximity=='INLAND'),'total_bedrooms']= 423.0

data.loc[(data.total_bedrooms.isnull())&(data.ocean_proximity=='ISLAND'),'total_bedrooms']= 512.0

data.loc[(data.total_bedrooms.isnull())&(data.ocean_proximity=='NEAR BAY'),'total_bedrooms']= 423.0

data.loc[(data.total_bedrooms.isnull())&(data.ocean_proximity=='NEAR OCEAN'),'total_bedrooms']= 464.0
data.total_bedrooms.isnull().sum()
# next 3 figures are learned from https://www.kaggle.com/manisood001/california-housing-optimised-modelling kernel 

plt.figure(figsize=(10,5))

sns.distplot(data['median_house_value'],color='green')

plt.show()
plt.figure(figsize=(10,10))



plt.scatter(data['population'],data['median_house_value'],c=data['median_house_value'], s=data['median_income']*10)

plt.colorbar

plt.title('population vs house value' )

plt.xlabel('population')

plt.ylabel('house value')

plt.plot()
# s=size of circles, c= color of circles

plt.figure(figsize=(15,15))

plt.scatter(data['longitude'],data['latitude'],c=data['median_house_value'],s=data['population']/10,cmap='viridis')

plt.colorbar()

plt.xlabel('longitude')

plt.ylabel('latitude')

plt.title('house price on basis of geo-coordinates')

plt.show()
plt.figure(figsize=(10,10))



sns.stripplot(data=data,x='ocean_proximity',y='median_house_value',jitter=0.3)
data = pd.get_dummies(data)
data.head()
y=data.pop('median_house_value')
data.isnull().sum()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.33, random_state=42)
X_train.head()
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.tree import DecisionTreeRegressor

from sklearn.neural_network import MLPRegressor   

from sklearn.ensemble import  GradientBoostingRegressor
import warnings

warnings.filterwarnings('ignore')
model= [LinearRegression(), DecisionTreeRegressor() ,   Lasso(), Ridge(),  MLPRegressor(), GradientBoostingRegressor()  ]

name = ['LinearRegression','DecisionTreeRegressor','Lasso','Ridge','MLPRegressor','GradientBoostingRegressor']

SCORE= []

TESTING=[]

RSME=[]

for ku in model:

    #ku will be replaced with each model like as first one is LogisticRegression()

    algorithm = ku.fit(X_train,y_train)

    print(ku)

    #now 'algorithm' will be fitted by API with above line and next line will check score with data training and testing

    predict_ku=ku.predict(X_test)

    print('RSME: {:.4f}'.format(np.sqrt(mean_squared_error(y_test,predict_ku))))

    score=cross_val_score(ku,X_train,y_train,cv=10,scoring='neg_mean_squared_error')

    ku_score_cross=np.sqrt(-score)

    

    print('mean: {:.2f} and std:{:.2f}'.format(np.mean(ku_score_cross),np.std(ku_score_cross)))

    print('---'*10)

    print('training set accuracy: {:.2f}'.format(algorithm.score(X_train,y_train)))

    print('test set accuracy: {:.2f}'.format(algorithm.score(X_test,y_test)))

    print('---'*30)

    #Now we are making a dataframe where by each loop the dataframe is added by SCORE,TESTING

    RSME.append(np.sqrt(mean_squared_error(y_test,predict_ku)))

    SCORE.append(algorithm.score(X_train,y_train))

    TESTING.append(algorithm.score(X_test,y_test))

models_dataframe=pd.DataFrame({'training score':SCORE,'testing score':TESTING,'RSME':RSME},index=name)
models_dataframe
asendingtraining = models_dataframe.sort_values(by='RSME', ascending=False)

asendingtraining 
asendingtraining['RSME'].plot.barh(width=0.8)

plt.title('RSME')

fig=plt.gcf()

fig.set_size_inches(8,8)

plt.show()