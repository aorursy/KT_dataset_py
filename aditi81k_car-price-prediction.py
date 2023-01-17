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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.linear_model import LinearRegression



# reading the dataset

cars = pd.read_csv("/kaggle/input/car-price-prediction/CarPrice_Assignment.csv")

cars.head()
cars.shape
cars.describe()
cars.info()
#Splitting company name from CarName column

CompanyName = cars['CarName'].apply(lambda a : a.split(' ')[0])

cars.insert(3,"CompanyName",CompanyName)

cars.drop(['CarName'],axis=1,inplace=True)

print(cars.head())
cars.CompanyName.unique()
cars.CompanyName = cars.CompanyName.str.lower()



def replace_name(a,b):

    cars.CompanyName.replace(a,b,inplace=True)



replace_name('maxda','mazda')

replace_name('porcshce','porsche')

replace_name('toyouta','toyota')

replace_name('vokswagen','volkswagen')

replace_name('vw','volkswagen')



cars.CompanyName.unique()
#Checking for duplicates

cars.loc[cars.duplicated()]
cars.columns
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

plt.title('Car Price Distribution Plot')

sns.distplot(cars.price)



plt.subplot(1,2,2)

plt.title('Car Price Spread')

sns.boxplot(y=cars.price)



plt.show()
print(cars.price.describe(percentiles = [0.25,0.50,0.75,0.85,0.90,1]))
plt.figure(figsize=(25, 6))



plt.subplot(1,3,1)

plt1 = cars.CompanyName.value_counts().plot('bar')

plt.title('Companies Histogram')

plt1.set(xlabel = 'Car company', ylabel='Frequency of company')



plt.subplot(1,3,2)

plt1 = cars.fueltype.value_counts().plot('bar')

plt.title('Fuel Type Histogram')

plt1.set(xlabel = 'Fuel Type', ylabel='Frequency of fuel type')



plt.subplot(1,3,3)

plt1 = cars.carbody.value_counts().plot('bar')

plt.title('Car Type Histogram')

plt1.set(xlabel = 'Car Type', ylabel='Frequency of Car type')



plt.show()
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

plt.title('Symboling Histogram')

sns.countplot(cars.symboling, palette=("cubehelix"))



plt.subplot(1,2,2)

plt.title('Symboling vs Price')

sns.boxplot(x=cars.symboling, y=cars.price, palette=("cubehelix"))



plt.show()
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

plt.title('Engine Type Histogram')

sns.countplot(cars.enginetype, palette=("Blues_d"))



plt.subplot(1,2,2)

plt.title('Engine Type vs Price')

sns.boxplot(x=cars.enginetype, y=cars.price, palette=("PuBuGn"))



plt.show()



df = pd.DataFrame(cars.groupby(['enginetype'])['price'].mean().sort_values(ascending = False))

df.plot.bar(figsize=(8,6))

plt.title('Engine Type vs Average Price')

plt.show()
plt.figure(figsize=(25, 6))



df = pd.DataFrame(cars.groupby(['CompanyName'])['price'].mean().sort_values(ascending = False))

df.plot.bar()

plt.title('Company Name vs Average Price')

plt.show()



df = pd.DataFrame(cars.groupby(['fueltype'])['price'].mean().sort_values(ascending = False))

df.plot.bar()

plt.title('Fuel Type vs Average Price')

plt.show()



df = pd.DataFrame(cars.groupby(['carbody'])['price'].mean().sort_values(ascending = False))

df.plot.bar()

plt.title('Car Type vs Average Price')

plt.show()
plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

plt.title('Door Number Histogram')

sns.countplot(cars.doornumber, palette=("plasma"))



plt.subplot(1,2,2)

plt.title('Door Number vs Price')

sns.boxplot(x=cars.doornumber, y=cars.price, palette=("plasma"))



plt.show()



plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

plt.title('Aspiration Histogram')

sns.countplot(cars.aspiration, palette=("plasma"))



plt.subplot(1,2,2)

plt.title('Aspiration vs Price')

sns.boxplot(x=cars.aspiration, y=cars.price, palette=("plasma"))



plt.show()
def plot_count(x,fig):

    plt.subplot(4,2,fig)

    plt.title(x+' Histogram')

    sns.countplot(cars[x],palette=("magma"))

    plt.subplot(4,2,(fig+1))

    plt.title(x+' vs Price')

    sns.boxplot(x=cars[x], y=cars.price, palette=("magma"))

    

plt.figure(figsize=(15,20))



plot_count('enginelocation', 1)

plot_count('cylindernumber', 3)

plot_count('fuelsystem', 5)

plot_count('drivewheel', 7)



plt.tight_layout()
def scatter(x,fig):

    plt.subplot(5,2,fig)

    plt.scatter(cars[x],cars['price'])

    plt.title(x+' vs Price')

    plt.ylabel('Price')

    plt.xlabel(x)



plt.figure(figsize=(10,20))



scatter('carlength', 1)

scatter('carwidth', 2)

scatter('carheight', 3)

scatter('curbweight', 4)



plt.tight_layout()
def pp(x,y,z):

    sns.pairplot(cars, x_vars=[x,y,z], y_vars='price',size=4, aspect=1, kind='scatter')

    plt.show()



pp('enginesize', 'boreratio', 'stroke')

pp('compressionratio', 'horsepower', 'peakrpm')

pp('wheelbase', 'citympg', 'highwaympg')
np.corrcoef(cars['carlength'], cars['carwidth'])[0, 1]
#Fuel economy

cars['fueleconomy'] = (0.55 * cars['citympg']) + (0.45 * cars['highwaympg'])

cars['fueleconomy']
#Binning the Car Companies based on avg prices of each Company.

cars['price'] = cars['price'].astype('int')

temp = cars.copy()

table = temp.groupby(['CompanyName'])['price'].mean()

temp = temp.merge(table.reset_index(), how='left',on='CompanyName')

bins = [0,10000,20000,40000]

cars_bin=['Budget','Medium','Highend']

cars['carsrange'] = pd.cut(temp['price_y'],bins,right=False,labels=cars_bin)

cars.head()
plt.figure(figsize=(8,6))



plt.title('Fuel economy vs Price')

sns.scatterplot(x=cars['fueleconomy'],y=cars['price'],hue=cars['drivewheel'])

plt.xlabel('Fuel Economy')

plt.ylabel('Price')



plt.show()

plt.tight_layout()
plt.figure(figsize=(25, 6))



df = pd.DataFrame(cars.groupby(['fuelsystem','drivewheel','carsrange'])['price'].mean().unstack(fill_value=0))

df

df.plot.bar()

plt.title('Car Range vs Average Price')

plt.show()
cars_lr = cars[['price','fueltype', 'aspiration','carbody', 'drivewheel','wheelbase','curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'boreratio','horsepower','fueleconomy', 'carlength','carwidth', 'carsrange']]

cars_lr.head()
sns.pairplot(cars_lr)

plt.show()
# Defining the map function

def dummies(x,df):

    temp = pd.get_dummies(df[x], drop_first = True)

    df = pd.concat([df, temp], axis = 1)

    df.drop([x], axis = 1, inplace = True)

    return df

# Applying the function to the cars_lr



cars_lr = dummies('fueltype',cars_lr)

cars_lr = dummies('aspiration',cars_lr)

cars_lr = dummies('carbody',cars_lr)

cars_lr = dummies('drivewheel',cars_lr)

cars_lr = dummies('enginetype',cars_lr)

cars_lr = dummies('cylindernumber',cars_lr)

cars_lr = dummies('carsrange',cars_lr)

cars_lr.head()
cars_lr.shape
from sklearn.model_selection import train_test_split



np.random.seed(0)

df_train, df_test = train_test_split(cars_lr, train_size = 0.7, test_size = 0.3, random_state = 100)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fueleconomy','carlength','carwidth','price']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
df_train.head()
df_train.describe()
#Correlation using heatmap

plt.figure(figsize = (30, 25))

sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")

plt.show()
#Dividing data into X and y variables

y_train = df_train.pop('price')

X_train = df_train
from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm 

from statsmodels.stats.outliers_influence import variance_inflation_factor
lm = LinearRegression()

lm.fit(X_train,y_train)

rfe = RFE(lm, 10)

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
X_train.columns[rfe.support_]
X_train_rfe = X_train[X_train.columns[rfe.support_]]

X_train_rfe.head()
def build_model(X,y):

    X = sm.add_constant(X) #Adding the constant

    lm = sm.OLS(y,X).fit() # fitting the model

    print(lm.summary()) # model summary

    return X

    

def checkVIF(X):

    vif = pd.DataFrame()

    vif['Features'] = X.columns

    vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    vif['VIF'] = round(vif['VIF'], 2)

    vif = vif.sort_values(by = "VIF", ascending = False)

    return(vif)
X_train_new = build_model(X_train_rfe,y_train)
X_train_new = X_train_rfe.drop(["twelve"], axis = 1)
X_train_newX_train_new = build_model(X_train_new,y_train)
X_train_new = X_train_new.drop(["fueleconomy"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
X_train_new = build_model(X_train_new,y_train)
#Calculating the Variance Inflation Factor

checkVIF(X_train_new)
X_train_new = X_train_new.drop(["curbweight"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
checkVIF(X_train_new)
X_train_new = X_train_new.drop(["sedan"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
checkVIF(X_train_new)
X_train_new = build_model(X_train_new,y_train)
checkVIF(X_train_new)
#Dropping dohcv to see the changes in model statistics

X_train_new = X_train_new.drop(["dohcv"], axis = 1)

X_train_new = build_model(X_train_new,y_train)

checkVIF(X_train_new)
lm = sm.OLS(y_train,X_train_new).fit()

y_train_price = lm.predict(X_train_new)

# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)   
#Scaling the test set

num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fueleconomy','carlength','carwidth','price']

df_test[num_vars] = scaler.fit_transform(df_test[num_vars])
#Dividing into X and y

y_test = df_test.pop('price')

X_test = df_test
# Now let's use our model to make predictions.

X_train_new = X_train_new.drop('const',axis=1)

# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[X_train_new.columns]



# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
# Making predictions

y_pred = lm.predict(X_test_new)
from sklearn.metrics import r2_score 

r2_score(y_test, y_pred)
#EVALUATION OF THE MODEL

# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16) 
print(lm.summary())