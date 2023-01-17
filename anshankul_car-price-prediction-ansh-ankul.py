#------------------------------------------Data Preprocessing--------------------------------------------------

#importing the libraries

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np 

import pandas as pd 
#importing the dataset

dataset = pd.read_csv('../input/car-data/CarPrice_Assignment.csv')
dataset.shape
dataset.info()
dataset.describe()
#Data Cleaning

#Splitting company name from CarName column



CompanyName = dataset['CarName'].apply(lambda x : x.split(' ')[0])

dataset.insert(3,"CompanyName",CompanyName)

dataset.drop(['CarName'],axis=1,inplace=True)

dataset.head()
dataset.CompanyName.unique()
#Correcting the wrong spellings

dataset.CompanyName = dataset.CompanyName.str.lower()



def replace_name(a,b):

    dataset.CompanyName.replace(a,b,inplace=True)



replace_name('maxda','mazda')

replace_name('porcshce','porsche')

replace_name('toyouta','toyota')

replace_name('vokswagen','volkswagen')

replace_name('vw','volkswagen')



dataset.CompanyName.unique()
#Checking for duplicates

dataset.loc[dataset.duplicated()]
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

plt.title('Car Price Distribution Plot')

sns.distplot(dataset.price)



plt.subplot(1,2,2)

plt.title('Car Price Spread')

sns.boxplot(y=dataset.price)



plt.show()
dataset.price.describe(percentiles = [0.25,0.50,0.75,0.85,0.90,1])
plt.figure(figsize=(25, 6))



plt.subplot(1,3,1)

plt1 = dataset.CompanyName.value_counts().plot(kind='bar')

plt.title('Companies Histogram')

plt1.set(xlabel = 'Car company', ylabel='Frequency of company')



plt.subplot(1,3,2)

plt1 = dataset.fueltype.value_counts().plot(kind='bar')

plt.title('Fuel Type Histogram')

plt1.set(xlabel = 'Fuel Type', ylabel='Frequency of fuel type')



plt.subplot(1,3,3)

plt1 = dataset.carbody.value_counts().plot(kind='bar')

plt.title('Car Type Histogram')

plt1.set(xlabel = 'Car Type', ylabel='Frequency of Car type')



plt.show()
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

plt.title('Symboling Histogram')

sns.countplot(dataset.symboling, palette=("bright"))



plt.subplot(1,2,2)

plt.title('Symboling vs Price')

sns.boxplot(x=dataset.symboling, y=dataset.price, palette=("bright"))



plt.show()
plt.figure(figsize=(25, 6))



df = pd.DataFrame(dataset.groupby(['CompanyName'])['price'].mean().sort_values(ascending = False))

df.plot.bar()

plt.title('Company Name vs Average Price')

plt.show()



df = pd.DataFrame(dataset.groupby(['fueltype'])['price'].mean().sort_values(ascending = False))

df.plot.bar()

plt.title('Fuel Type vs Average Price')

plt.show()



df = pd.DataFrame(dataset.groupby(['carbody'])['price'].mean().sort_values(ascending = False))

df.plot.bar()

plt.title('Car Type vs Average Price')

plt.show()
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

plt.title('Engine Type Histogram')

sns.countplot(dataset.enginetype, palette=("bright"))



plt.subplot(1,2,2)

plt.title('Engine Type vs Price')

sns.boxplot(x=dataset.enginetype, y=dataset.price, palette=("bright"))



plt.show()



df = pd.DataFrame(dataset.groupby(['enginetype'])['price'].mean().sort_values(ascending = False))

df.plot.bar(figsize=(8,6))

plt.title('Engine Type vs Average Price')

plt.show()
plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

plt.title('Door Number Histogram')

sns.countplot(dataset.doornumber, palette=("bright"))



plt.subplot(1,2,2)

plt.title('Door Number vs Price')

sns.boxplot(x=dataset.doornumber, y=dataset.price, palette=("bright"))



plt.show()



plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

plt.title('Aspiration Histogram')

sns.countplot(dataset.aspiration, palette=("bright"))



plt.subplot(1,2,2)

plt.title('Aspiration vs Price')

sns.boxplot(x=dataset.aspiration, y=dataset.price, palette=("bright"))



plt.show()
def plot_count(x,fig):

    plt.subplot(4,2,fig)

    plt.title(x+' Histogram')

    sns.countplot(dataset[x],palette=("bright"))

    plt.subplot(4,2,(fig+1))

    plt.title(x+' vs Price')

    sns.boxplot(x=dataset[x], y=dataset.price, palette=("bright"))

    

plt.figure(figsize=(15,20))



plot_count('enginelocation', 1)

plot_count('cylindernumber', 3)

plot_count('fuelsystem', 5)

plot_count('drivewheel', 7)



plt.tight_layout()
def scatter(x,fig):

    plt.subplot(5,2,fig)

    plt.scatter(dataset[x],dataset['price'])

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

    sns.pairplot(dataset, x_vars=[x,y,z], y_vars='price',height=4, aspect=1, kind='scatter')

    plt.show()



pp('enginesize', 'boreratio', 'stroke')

pp('compressionratio', 'horsepower', 'peakrpm')

pp('wheelbase', 'citympg', 'highwaympg')
# Creating a new feature based on inference from above plots

dataset['fueleconomy'] = (0.60 * dataset['citympg']) + (0.40 * dataset['highwaympg'])



dataset["brand_category"] = dataset['price'].apply(lambda x : "Budget" if x < 10000 

                                                     else ("Mid_Range" if 10000 <= x < 20000

                                                           else ("Luxury")))
plt.figure(figsize=(8,6))



plt.title('Fuel economy vs Price')

sns.scatterplot(x=dataset['fueleconomy'],y=dataset['price'],hue=dataset['drivewheel'])

plt.xlabel('Fuel Economy')

plt.ylabel('Price')



plt.show()

plt.tight_layout()



plt1 = sns.scatterplot(x = 'horsepower', y = 'price', hue = 'brand_category', data = dataset)

plt1.set_xlabel('Horsepower')

plt1.set_ylabel('Price of Car ($)')

plt.show()
attributes = dataset[['fueltype', 'aspiration', 'carbody', 'drivewheel', 'wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginetype'

       , 'enginesize',  'boreratio', 'horsepower', 'price', 'brand_category', 'fueleconomy']]



attributes.head()
#visualising most of the attributes

plt.figure(figsize=(15,15))

sns.pairplot(attributes)

plt.show()
#Droppping non-important features according to the plots

dataset = dataset.drop(["highwaympg"], axis = 1)

dataset = dataset.drop(["citympg"], axis = 1)

dataset = dataset.drop(["car_ID"], axis = 1)

dataset = dataset.drop(["doornumber"], axis = 1)

dataset = dataset.drop(["cylindernumber"], axis = 1)

dataset = dataset.drop(["enginelocation"], axis = 1)
#Handling Categorical Data

# Defining the map function

def dummies(x,df):

    temp = pd.get_dummies(df[x], drop_first = True)

    df = pd.concat([df, temp], axis = 1)

    df.drop([x], axis = 1, inplace = True)

    return df



dataset = dummies('CompanyName',dataset)

dataset = dummies('fueltype',dataset)

dataset = dummies('fuelsystem',dataset)

dataset = dummies('aspiration',dataset)

dataset = dummies('carbody',dataset)

dataset = dummies('drivewheel',dataset)

dataset = dummies('enginetype',dataset)

dataset = dummies('brand_category',dataset)
#Splitting into training and test set

from sklearn.model_selection import train_test_split

np.random.seed(0)

df_train, df_test = train_test_split(dataset, train_size = 0.8, test_size = 0.2, random_state = 100)
#Feature Scaling

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','fueleconomy','carlength','carwidth','price']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])
#Correlation using heatmap

plt.figure(figsize = (30, 25))

sns.heatmap(df_train.corr(), annot = True, cmap="flag")

plt.show()
#Dividing data into X and y variables

y_train = df_train.pop('price')

X_train = df_train
#Including RFE

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm 

from statsmodels.stats.outliers_influence import variance_inflation_factor
lm = LinearRegression()

lm.fit(X_train,y_train)

rfe = RFE(lm, 10)

rfe = rfe.fit(X_train, y_train)
X_train_rfe = X_train[X_train.columns[rfe.support_]]
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
X_train_new = X_train_rfe.drop(["hardtop"], axis = 1)
X_train_new = build_model(X_train_new,y_train)
lm = sm.OLS(y_train,X_train_new).fit()

y_train_price = lm.predict(X_train_new)

# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)   

#Calculating the Variance Inflation Factor

checkVIF(X_train_new)
#Scaling the test set

num_vars = ['symboling','wheelbase', 'curbweight', 'enginesize', 'boreratio','stroke','compressionratio','peakrpm', 'horsepower','fueleconomy','carlength','carwidth','carheight','price']

df_test[num_vars] = scaler.fit_transform(df_test[num_vars])
#Dividing into X and y

y_test = df_test.pop('price')

X_test = df_test



X_train_new = X_train_new.drop('const',axis=1)

X_test_new = X_test[X_train_new.columns]
X_test_new = sm.add_constant(X_test_new)
# Making predictions

y_pred = lm.predict(X_test_new)
from sklearn.metrics import r2_score 

r2_score(y_test, y_pred)

#Evaluation



# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)