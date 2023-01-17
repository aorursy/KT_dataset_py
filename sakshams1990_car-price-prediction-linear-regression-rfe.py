#Importing library for dataframe

import pandas as pd

import numpy as np

#Importing library for suppressing warnings

import warnings

warnings.filterwarnings('ignore')

#Importing library for data visualization

import matplotlib.pyplot as plt

import seaborn as sns

#Importing library for train-test data split

from sklearn.model_selection import train_test_split

#Importing library for rescaling the features

from sklearn.preprocessing import MinMaxScaler

# Importing library to calculate r-squared

from sklearn.metrics import r2_score

#Importing RFE and Linear Regression for building model

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

#Importing statsmodel for adding a constant

import statsmodels.api as sm
#Reading CarPrice csv file from local

car_price = pd.read_csv("../input/CarPrice_Assignment.csv")
#Checking first 5 records of the dataset

car_price.head()
# To display all the columns

pd.set_option('display.max_columns',50)
# To check the records of first 5 in the dataset

car_price.head()
#To see the number of rows and columns of the dataset

car_price.shape
# To check if there are any missing entries.

car_price.info()
# To check the statistical parameters of the numerical columns

car_price.describe()
# Dropping car models from the CarName column to have only the car brand  names

car_price['CarName']=car_price['CarName'].apply(lambda x : x.split(' ')[0])
# To see check first 5 records of the dataframe

car_price.head()
#To see what are the unique car brand names

car_price['CarName'].unique()
# Renaming CarName values in order to have consistency over car brand names

def car_name_replace(a,b):

    car_price.CarName.replace(a,b,inplace=True)

    

car_name_replace('maxda','mazda')

car_name_replace('porcshce','porsche')

car_name_replace('toyouta','toyota')

car_name_replace('vw','volkswagen')

car_name_replace('vokswagen','volkswagen')

car_name_replace('Nissan','nissan')
#To check the unique car brand names

car_price.CarName.unique()
# To count number of cars for each brand

fig, ax = plt.subplots(figsize = (15,5))

df=sns.countplot(car_price['CarName'],order = car_price['CarName'].value_counts().index)

df.set_xlabel('Car Brands')

df.set_ylabel('Count of Cars')

df.set_title('Car Brand vs No of Cars')

plt.xticks(rotation = 90)

plt.show()

plt.tight_layout()
# Plotting barplots for categorical data vs avg price

df=pd.DataFrame(car_price.groupby(['CarName'])['price'].mean().sort_values(ascending=False))

df.plot.bar()

plt.title('Car Company vs Average Price')

plt.show()



df=pd.DataFrame(car_price.groupby(['fueltype'])['price'].mean().sort_values(ascending=False))

df.plot.bar()

plt.title('Fuel Type vs Average Price')

plt.show()



df=pd.DataFrame(car_price.groupby(['carbody'])['price'].mean().sort_values(ascending=False))

df.plot.bar()

plt.title('Car Body vs Average Price')

plt.show()
# Plotting barplots for categorical data vs avg price

plt.figure(figsize=(15,10))

df=pd.DataFrame(car_price.groupby(['drivewheel'])['price'].mean().sort_values(ascending=False))

df.plot.bar()

plt.title('Drive Wheel vs Average Price')

plt.show()



df=pd.DataFrame(car_price.groupby(['aspiration'])['price'].mean().sort_values(ascending=False))

df.plot.bar()

plt.title('Aspiration vs Average Price')

plt.show()



df=pd.DataFrame(car_price.groupby(['doornumber'])['price'].mean().sort_values(ascending=False))

df.plot.bar()

plt.title('No. of doors vs Average Price')

plt.show()
# Plotting barplots for categorical data vs avg price

df=pd.DataFrame(car_price.groupby(['enginelocation'])['price'].mean().sort_values(ascending=False))

df.plot.bar()

plt.title('Engine Location vs Average Price')

plt.show()



df=pd.DataFrame(car_price.groupby(['enginetype'])['price'].mean().sort_values(ascending=False))

df.plot.bar()

plt.title('Engine Types vs Average Price')

plt.show()



df=pd.DataFrame(car_price.groupby(['cylindernumber'])['price'].mean().sort_values(ascending=False))

df.plot.bar()

plt.title('No. of Cylinder vs Average Price')

plt.show()
# Plotting barplots for categorical data vs avg price

df=pd.DataFrame(car_price.groupby(['fuelsystem'])['price'].mean().sort_values(ascending=False))

df.plot.bar()

plt.title('Fuel System vs Average Price')

plt.show()



df=pd.DataFrame(car_price.groupby(['symboling'])['price'].mean().sort_values(ascending=False))

df.plot.bar()

plt.title('Symboling vs Average Price')

plt.show()
# Checking statistical parameters of the column price of the dataframe

sns.distplot(car_price['price'])

car_price['price'].describe()
#Wheel base vs Price 

df=sns.scatterplot(x='wheelbase',y='price',data=car_price)

df.set_xlabel('Wheelbase')

df.set_ylabel('Price')

plt.show()
#Plotting Car Size features with the price of the cars.

df=sns.scatterplot(x='curbweight',y='price',data=car_price)

df.set_xlabel('Curb Weight')

df.set_ylabel('Price')

plt.show()



df=sns.scatterplot(x='carlength',y='price',data=car_price)

df.set_xlabel('Car Length')

df.set_ylabel('Price')

plt.show()



df=sns.scatterplot(x='carwidth',y='price',data=car_price)

df.set_xlabel('Car Width')

df.set_ylabel('Price')

plt.show()



df=sns.scatterplot(x='carheight',y='price',data=car_price)

df.set_xlabel('Car height')

df.set_ylabel('Price')

plt.show()
#Engine features vs price

df=sns.scatterplot(x='enginesize',y='price',data=car_price)

df.set_xlabel('Engine Size')

df.set_ylabel('Price')

plt.show()



df=sns.scatterplot(x='boreratio',y='price',data=car_price)

df.set_xlabel('Bore Ratio')

df.set_ylabel('Price')

plt.show()



df=sns.scatterplot(x='stroke',y='price',data=car_price)

df.set_xlabel('Stroke')

df.set_ylabel('Price')

plt.show()



df=sns.scatterplot(x='compressionratio',y='price',data=car_price)

df.set_xlabel('Compression Ratio')

df.set_ylabel('Price')

plt.show()



df=sns.scatterplot(x='horsepower',y='price',data=car_price)

df.set_xlabel('Horse Power')

df.set_ylabel('Price')

plt.show()



df=sns.scatterplot(x='peakrpm',y='price',data=car_price)

df.set_xlabel('Peak RPM')

df.set_ylabel('Price')

plt.show()
# Creating a common column mileage on putting a formula of citympg*0.55+ highwaympg*0.45

car_price['mileage']=car_price['citympg']*0.55 + car_price['highwaympg']*0.45
#Mileage vs Car prices

df=sns.scatterplot(x='mileage',y='price',data=car_price)

df.set_xlabel('Mileage')

df.set_ylabel('Price')

plt.show()
#Categorizing the price of the cars into 3 categories - Economy,Standard,Luxury

car_price['pricecategory']=car_price['price'].apply(lambda x : "Economy" if x < 10000 else ("Standard" if 10000<=x<30000 else "Luxury"))
# Showing random 10 data with different brand category

car_price.sample(10)
#The model based on the analysis done on the variables with respective to the price of the car

price_lr=car_price[['fueltype',

'carbody',

'drivewheel',

'aspiration',

'enginelocation',

'enginetype',

'cylindernumber',

'fuelsystem',

'wheelbase',

'curbweight',

'carlength',

'carwidth',

'carheight',

'enginesize',

'boreratio',

'horsepower',

'pricecategory',

'mileage',

'price']]
price_lr.head()
# Visualising the numerical columns using pair plot

plt.figure(figsize=(15,15))

sns.pairplot(price_lr)

plt.show()
# Plotting box plots for Categorical Variables

plt.figure(figsize=(15,25))

plt.subplot(5,2,1)

sns.boxplot(x='fueltype',y='price',data=price_lr)



plt.subplot(5,2,2)

sns.boxplot(x='carbody',y='price',data=price_lr)



plt.subplot(5,2,3)

sns.boxplot(x='drivewheel',y='price',data=price_lr)



plt.subplot(5,2,4)

sns.boxplot(x='aspiration',y='price',data=price_lr)



plt.subplot(5,2,5)

sns.boxplot(x='enginelocation',y='price',data=price_lr)



plt.subplot(5,2,6)

sns.boxplot(x='enginetype',y='price',data=price_lr)



plt.subplot(5,2,7)

sns.boxplot(x='cylindernumber',y='price',data=price_lr)



plt.subplot(5,2,8)

sns.boxplot(x='fuelsystem',y='price',data=price_lr)



plt.subplot(5,2,9)

sns.boxplot(x='pricecategory',y='price',data=price_lr)
# Creating dummy variables to change categorical variable into numerical variable
cylinder_no = pd.get_dummies(price_lr['cylindernumber'], drop_first = True)

price_lr = pd.concat([price_lr, cylinder_no], axis = 1)
fuel_type = pd.get_dummies(price_lr['fueltype'], drop_first = True)

price_lr = pd.concat([price_lr, fuel_type], axis = 1)
car_body = pd.get_dummies(price_lr['carbody'], drop_first = True)

price_lr = pd.concat([price_lr, car_body], axis = 1)
drive_wheel = pd.get_dummies(price_lr['drivewheel'], drop_first = True)

price_lr = pd.concat([price_lr, drive_wheel], axis = 1)
asp = pd.get_dummies(price_lr['aspiration'], drop_first = True)

price_lr = pd.concat([price_lr, asp], axis = 1)
engine_location = pd.get_dummies(price_lr['enginelocation'], drop_first = True)

price_lr = pd.concat([price_lr, engine_location], axis = 1)
engine_type = pd.get_dummies(price_lr['enginetype'], drop_first = True)

price_lr = pd.concat([price_lr, engine_type], axis = 1)
fuel_system = pd.get_dummies(price_lr['fuelsystem'], drop_first = True)

price_lr = pd.concat([price_lr, fuel_system], axis = 1)
price_category = pd.get_dummies(price_lr['pricecategory'], drop_first = True)

price_lr = pd.concat([price_lr, price_category], axis = 1)
price_lr.head()
price_lr=price_lr.drop(['fueltype','carbody','drivewheel','aspiration','enginelocation','enginetype','cylindernumber','fuelsystem','pricecategory'],axis=1)
price_lr.head()
#To check the number of rows and columns of the new dataframe created for linear regression

price_lr.shape
#Splitting data into train data set and test data set

np.random.seed(0)

df_train , df_test = train_test_split(price_lr , train_size = 0.7, random_state = 100)
#Feature Scaling using MinMax scaler

scaler = MinMaxScaler()

num_vars=['wheelbase','curbweight','carlength','carwidth','carheight','enginesize','boreratio','horsepower','mileage','price']

df_train[num_vars]=scaler.fit_transform(df_train[num_vars])
df_train.head()
df_train.describe()
#Correlation using heatmap

plt.figure(figsize=(30,30))

sns.heatmap(df_train.corr(),annot=True,cmap='YlGnBu')

plt.show()
# Dividing data into X and Y

y_train = df_train.pop('price')

X_train = df_train
#Running RFE with the output number of variables equal to 10

lm = LinearRegression()

lm.fit(X_train,y_train)

#Running RFE

rfe = RFE(lm,10)

rfe = rfe.fit(X_train,y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]

col
X_train.columns[~rfe.support_]
#Creating X_train dataframe with RFE selected variables

X_train_rfe = X_train[col]
#Adding a constant variable

X_train_rfe= sm.add_constant(X_train_rfe)
#Running the linear model

lm = sm.OLS(y_train,X_train_rfe).fit()
#Summary of the linear model

print(lm.summary())
#Calculating VIF for the model

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['Features'] = X_train_rfe.columns

vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
#dropping enginesize due to high p-value and VIF

X_train_new = X_train_rfe.drop(['enginesize'],axis=1) 
#Adding a constant variable and creating a new model

X_train_lm = sm.add_constant(X_train_new)

lm = sm.OLS(y_train,X_train_new).fit()

print(lm.summary())
#Removing boreratio due to high p-value

X_train_new1=X_train_new.drop(['boreratio'],axis=1)
#Adding a constant variable

X_train_lm = sm.add_constant(X_train_new1)

lm = sm.OLS(y_train,X_train_lm).fit()

print(lm.summary())
#Dropping curbweight due to high p -value and creating a new model



X_train_new2=X_train_new1.drop(['curbweight'],axis=1)

X_train_lm = sm.add_constant(X_train_new2)

lm = sm.OLS(y_train,X_train_lm).fit()

print(lm.summary())
# Dropping the feature const from the dataframe

X_train_new2=X_train_new2.drop(['const'],axis=1)
#Since all the p-values are less than 0.05, we will check for the VIFs for the new model

vif = pd.DataFrame()

X = X_train_new2

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
#Removing carwidth due to high VIF value and rebuilding the model

X_train_new3=X_train_new2.drop(['carwidth'],axis=1)

X_train_lm = sm.add_constant(X_train_new3)

lm = sm.OLS(y_train,X_train_lm).fit()

print(lm.summary())
#Since all the p-values are less than 0.05, we will check for the VIFs for the new model

vif = pd.DataFrame()

X = X_train_new3

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
# Need to check if the errors are also normally distributed

y_train_price = lm.predict(X_train_lm)
#Plotting distribution plot for the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price),bins=20)

fig.suptitle('Error Terms',fontsize=20)

plt.xlabel('Errors',fontsize=18)
#Scaling the test set

num_vars=['wheelbase','curbweight','carlength','carwidth','carheight','enginesize','boreratio','horsepower','mileage','price']

df_test[num_vars] = scaler.transform(df_test[num_vars])
y_test = df_test.pop('price')

X_test = df_test
#Creating X_test_new containing only those columns which were used to train dataset

X_test_new = X_test[['horsepower','four','rwd','dohcv','rotor','Luxury']]
#Adding a constant to the X test

X_test_new = sm.add_constant(X_test_new)
#Predicting y based on X test data set

y_pred = lm.predict(X_test_new)
#Calculating r2 score between y_test and y_pred

from sklearn.metrics import r2_score

r2_score(y_test,y_pred)
fig = plt.figure()

plt.scatter(y_test, y_pred)

fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 

plt.xlabel('y_test', fontsize = 18)                          # X-label

plt.ylabel('y_pred', fontsize = 16)