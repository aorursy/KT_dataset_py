import os

dir='../input/automobiledata'

os.chdir(dir)

import pandas as pd

from pandas import DataFrame,Series

import numpy as np
data=pd.read_csv('Car.csv',sep=',')
#Row and Columns:

data.shape
#Top 5 values of data:

data.head()
#Type of each attributes:

data.dtypes
#Summary Statistics:

data.describe()
#Checking missing value:

data.isnull().sum()
#Retrieving only the car brand or manufacturing company names:

data['CarName']=data['CarName'].str.split(" ").str[0]
#Checking unique values of car companies:

data['CarName'].unique()
#Fixing and replacing the data each car brands: (removing spelling error)

data['CarName']=data['CarName'].str.replace("Nissan","nissan")

data['CarName']=data['CarName'].str.replace("vokswagen","volkswagen")

data['CarName']=data['CarName'].str.replace("porcshce","porsche")

data['CarName']=data['CarName'].str.replace("toyouta","toyota")

data['CarName']=data['CarName'].str.replace("vw","volkswagen")

data['CarName']=data['CarName'].str.replace("maxda","mazda")

#Summary statistics of PRICE across each type of car company:

data.groupby('CarName')['price'].describe()
#Summary statistics of PRICE across fueltype:

data.groupby('fueltype')['price'].describe()
#Summary statistics of PRICE across aspiration:

data.groupby('aspiration')['price'].describe()
#Summary statistics of PRICE across number of doors in car:

data.groupby('doornumber')['price'].describe()
#Summary statistics of PRICE across bodytype of car:

data.groupby('carbody')['price'].describe()
#Summary statistics of PRICE across drivewheel:

data.groupby('drivewheel')['price'].describe()
#Summary statistics of PRICE across engine location:

data.groupby('enginelocation')['price'].describe()
#Summary statistics of PRICE across engine type:

data.groupby('enginetype')['price'].describe()
#Summary statistics of PRICE across cylinder number:

data.groupby('cylindernumber')['price'].describe()
#Summary statistics of PRICE across fuel system:

data.groupby('fuelsystem')['price'].describe()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
#Visualization of Price distribution:

sns.distplot(data['price'],color='blue')
data['CarName'].value_counts().plot(kind='bar', figsize=(15,5))

plt.title("Number of vehicles by car company")

plt.ylabel('Number of vehicles')

plt.xlabel('Company')

plt.show
#Visualization of AVERAGE PRICE across each continuous attributes:

plt.figure(figsize=(15,15))

plt.subplot(3,3,1)

plt.scatter(data['price'],data['enginesize'],color='blue')

plt.title("Price across enginesize ")

plt.subplot(3,3,2)

plt.scatter(data['price'],data['boreratio'],color='magenta')

plt.title("Price across boreratio")

plt.subplot(3,3,3)

plt.scatter(data['price'],data['wheelbase'],color='green')

plt.title("Price across wheelbase ")

plt.subplot(3,3,4)

plt.scatter(data['price'],data['stroke'],color='yellow')

plt.title("Price across stroke ")

plt.subplot(3,3,5)

plt.scatter(data['price'],data['compressionratio'],color='purple')

plt.title("Price across compressionratio ")

plt.subplot(3,3,6)

plt.scatter(data['price'],data['horsepower'],color='pink')

plt.title("Price across horsepower")

plt.subplot(3,3,7)

plt.scatter(data['price'],data['peakrpm'],color='brown')

plt.title("Price across peakrpm ")

plt.subplot(3,3,8)

plt.scatter(data['price'],data['citympg'],color='orange')

plt.title("Price across citympg ")

plt.subplot(3,3,9)

plt.scatter(data['price'],data['highwaympg'],color='gold')

plt.title("Price across highwaympg")

plt.tight_layout()

plt.show()
#Visualization of AVERAGE PRICE across each categorical variable:

from matplotlib import style

plt.figure(figsize=(15,15))

plt.subplot(3,3,1)

data.groupby('fueltype')['price'].mean().plot(kind='bar')

plt.title("Average price across fueltype")

plt.subplot(3,3,2)

data.groupby('aspiration')['price'].mean().plot(kind='bar')

plt.title("Average price across aspiration")

plt.subplot(3,3,3)

data.groupby('doornumber')['price'].mean().plot(kind='bar')

plt.title("Average price across doornumber")

plt.subplot(3,3,4)

data.groupby('carbody')['price'].mean().plot(kind='bar')

plt.title("Average price across carbody")

plt.subplot(3,3,5)

data.groupby('drivewheel')['price'].mean().plot(kind='bar')

plt.title("Average price across drivewheel")

plt.subplot(3,3,6)

data.groupby('enginelocation')['price'].mean().plot(kind='bar')

plt.title("Average price across enginelocation")

plt.subplot(3,3,7)

data.groupby('fuelsystem')['price'].mean().plot(kind='bar')

plt.title("Average price across fuelsystem")

plt.subplot(3,3,8)

data.groupby('enginetype')['price'].mean().plot(kind='bar')

plt.title("Average price across enginetype")

plt.subplot(3,3,9)

data.groupby('cylindernumber')['price'].mean().plot(kind='bar')

plt.title("Average price across cylindernumber")

plt.tight_layout()

style.use('classic')

plt.show()
#Unique types of fuel:

data['fueltype'].unique()
#Unique types of aspiration:

data['aspiration'].unique()
#Unique number of doors present in car:

data['doornumber'].unique()
#Unique types of carbody:

data['carbody'].unique()
#Unique wheel feature present:

data['drivewheel'].unique()
#Unique types of engine:

data['enginetype'].unique()
#Number of cylinder:

data['cylindernumber'].unique()
#Type of fuel system:

data['fuelsystem'].unique()
#Unique type of engine location:

data['enginelocation'].unique()
#Conversion of categorical variable to numeric type, without using HOT ENCODING:

conversion={"carbody": {'convertible':1,'hatchback':2,'sedan':3,'wagon':4,'hardtop':5}, 

            "drivewheel": {'rwd':1,'fwd':2,'4wd':3},

            "enginetype": {'dohc':1,'ohcv':2,'ohc':3,'l':4,'rotor':5,'ohcf':6,'dohcv':7},

            "cylindernumber": {'four':4,'six':6,'five':5,'three':3,'twelve':12,'two':2,'eight':8},

            "fuelsystem": {'mpfi':1,'2bbl':2,'mfi':3,'1bbl':4,'spfi':5,'4bbl':6,'idi':7,'spdi':8},

            "enginelocation": {'front':1,'rear':0},

            "fueltype": {'gas':1,'diesel':2},

            "aspiration": {'std':1,'turbo':0},

            "doornumber": {'two':2,'four':4}}

data.replace(conversion,inplace=True)
#Conversion of CarName to numeric feature using cat.codes:

data['CarName']=data['CarName'].astype('category') #converting object into category

data['Carcompany']=data['CarName'].cat.codes #encoding numerical values 
#Top 5 data values after conversion:

data.head()
#Using Pearson Correlation - finding correlation:

plt.figure(figsize=(20,20))

correlation=data.corr()

sns.heatmap(correlation,annot=True)

plt.show()
#Correlation with target variable:

cor_target=abs(correlation["price"])

#Selecting highly correlated features:

relevantfeatures=cor_target[cor_target>0.4] #conidering values greater than 40% of correlation

relevantfeatures
#Checking correlation acroos each highly correlated independent variables:

a=data[['drivewheel','wheelbase','carlength','carwidth','curbweight','enginesize','boreratio','horsepower','citympg','highwaympg']].corr()

a
#Using Pearson Correlation for the above correlated features:

plt.figure(figsize=(9,4))

sns.heatmap(a,annot=True)

plt.show()
#Dropping independent variables that are highly correlating against each other or tending to show same significances:

dat=data.drop(['carlength','highwaympg','citympg','boreratio','drivewheel','car_ID','CarName'],axis=1)
import sklearn.linear_model as linear_model

import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection

train,test=model_selection.train_test_split(dat,test_size=0.3)
#Intialising the dependent variable, independent variables and fitting the model:

x=train.drop(['price'],axis=1) # Independent variable

y=train['price'] #Target variable

#Linear Regression model:

model=linear_model.LinearRegression(fit_intercept=True)

model.fit(x,y)
import statsmodels.api as sm

#Compute with statsmodels, by adding intercept manually:

x1=sm.add_constant(x) 

model=sm.OLS(y,x1).fit()
#Summary stats of the model, determining the stats value and feature importance, to obtain better model prediction:

model.summary()
#Intialising the dependent variable and independent variables based on the significant variable obtained from previous model:

x=train[['fueltype','aspiration','curbweight','enginesize','carbody','horsepower','compressionratio','enginelocation','peakrpm','cylindernumber','fuelsystem','stroke','Carcompany']]

y=train['price']
#Compute with statsmodels, by adding intercept manually:

x1=sm.add_constant(x)

model=sm.OLS(y,x1).fit()
#Summary stats of the model, determining the stats value and feature importance, to obtain better model prediction:

model.summary()
#Intialising the dependent variable and independent variables based on the significant variable obtained from previous model:

x=train[['aspiration','enginesize','curbweight','symboling','enginelocation','fuelsystem','stroke','peakrpm','compressionratio','horsepower','Carcompany']]

y=train['price']
#Compute with statsmodels, by adding intercept manually:

x1=sm.add_constant(x)

model=sm.OLS(y,x1).fit()
#Summary stats of the model, determining the stats value and feature importance, to obtain better model prediction:

model.summary()
#Intialising the dependent variable and independent variables based on the significant variable obtained from previous model:

x=train[['enginesize','curbweight','enginelocation','fuelsystem','stroke','peakrpm','compressionratio','horsepower','Carcompany']]

y=train['price']
#Compute with statsmodels, by adding intercept manually:

x1=sm.add_constant(x)

model=sm.OLS(y,x1).fit()
#Summary stats of the model, determining the stats value and feature importance, to obtain better model prediction:

model.summary()
#Intialising the dependent variable and independent variables based on the significant variable obtained from previous model:

x=train[['enginesize','curbweight','enginelocation','stroke','peakrpm','compressionratio','horsepower','Carcompany']]

y=train['price']
#Compute with statsmodels, by adding intercept manually:

x1=sm.add_constant(x)

model=sm.OLS(y,x1).fit()
#Summary stats of the model, determining the stats value and feature importance, to obtain better model prediction:

model.summary()
#From the above model summary we find the most sognificant features:

# The most significant features are:

# --- enginesize, enginelocation, curbweight, stroke, peakrpm, compressionratio, horsepower and carcompan(brand name of car)
#Intialising the test x and y values based on the above model created:

test_x=test[['enginesize','curbweight','enginelocation','stroke','peakrpm','compressionratio','horsepower','Carcompany']]

test_y=test['price']
#Predicting the model across the train set:

pred=model.predict(x1)
#Predicting the model across the test set:

x2=sm.add_constant(test_x)

y_pred=model.predict(x2)
#Visualization of actual vs predicted values:

plt.scatter(test_y,y_pred)

plt.title("Actual vs Predicted values")

plt.xlabel("Actual values")

plt.ylabel("Predicted values")

plt.show
#Determining the MSE value of our above fit model:

from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, y_pred))  
#Calulating Rsquared value for the test set of actual and predicted values:

from sklearn.metrics import r2_score

r2_score(test_y,y_pred) # The value obtaine is close the value of R2 obtained in the above fitted model summary
#Visualization of predicted values of test dataset:

sns.distplot(y_pred)

plt.xlabel("Differences of Price")

plt.ylabel("Count")

plt.plot()
#Visualization of the above model fit- Partial Regression Plot:

fig,ax=plt.subplots(figsize=(20,20))

fig=sm.graphics.plot_partregress_grid(model,fig=fig) # Identifies the best  fit olsline