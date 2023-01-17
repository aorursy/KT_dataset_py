# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb
dataset = pd.read_csv('../input/car-data/CarPrice_Assignment.csv')

dataset.head()
dataset.shape
dataset.describe()
#Splitting company name from CarName column



CompanyName = dataset['CarName'].apply(lambda x : x.split(' ')[0])

dataset.insert(3,"CompanyName",CompanyName)

dataset.drop(['CarName'],axis=1,inplace=True)

dataset.head()
#data cleaning and correcting



def replace_name(a,b):

    dataset.CompanyName.replace(a,b,inplace=True)



replace_name('maxda','mazda')

replace_name('porcshce','porsche')

replace_name('toyouta','toyota')

replace_name('vokswagen','volkswagen')

replace_name('vw','volkswagen')
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

plt.title('Car Price Distribution Plot')

sb.distplot(dataset.price)



plt.subplot(1,2,2)

plt.title('Car Price Spread')

sb.boxplot(y=dataset.price)



plt.show()
X = dataset.iloc[: ,:-1].values

#df = pd.DataFrame(X)

y = dataset.iloc[:, 25].values



#visualising categorical data



fig, ax = plt.subplots(figsize = (15,5))

plt1 = sb.countplot(dataset['CompanyName'], order=pd.value_counts(dataset['CompanyName']).index,)

plt1.set(xlabel = 'Brand', ylabel= 'Count of Cars')

#xticks(rotation = 90)

plt.show()

plt.tight_layout()
dataset_comp_avg_price = dataset[['CompanyName','price']].groupby("CompanyName", as_index = False).mean()

plt1 = dataset_comp_avg_price.plot(x = 'CompanyName', kind='bar',legend = False, sort_columns = True, figsize = (15,3))

plt1.set_xlabel("CompanyName")

plt1.set_ylabel("Avg Price (Dollars)")

plt.show()




plt.figure(figsize=(25, 6))







plt.subplot(1,2,1)

plt1 = dataset.fueltype.value_counts().plot(kind = 'bar')

plt.title('Fuel Type Histogram')

plt1.set(xlabel = 'Fuel Type', ylabel='Frequency of fuel type')



plt.subplot(1,2,2)

plt1 = dataset.carbody.value_counts().plot(kind = 'bar')

plt.title('Car Type Histogram')

plt1.set(xlabel = 'Car Type', ylabel='Frequency of Car type')



plt.show()
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

plt.title('Symboling Histogram')

sb.countplot(dataset.symboling)



plt.subplot(1,2,2)

plt.title('Symboling vs Price')

sb.boxplot(x=dataset.symboling, y=dataset.price)



plt.show()
plt.figure(figsize=(20,8))



plt.subplot(1,2,1)

plt.title('Engine Type Histogram')

sb.countplot(dataset.enginetype, palette=("PuBuGn"))









df = pd.DataFrame(dataset.groupby(['enginetype'])['price'].mean().sort_values(ascending = False))

df.plot.bar(figsize=(8,6))

plt.title('Engine Type vs Average Price')

plt.show()
df = pd.DataFrame(dataset.groupby(['fueltype'])['price'].mean().sort_values(ascending = False))

df.plot.bar()

plt.title('Fuel Type vs Average Price')

plt.show()



df = pd.DataFrame(dataset.groupby(['carbody'])['price'].mean().sort_values(ascending = False))

df.plot.bar()

plt.title('Car Type vs Average Price')

plt.show()
plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

plt.title('Door Number Histogram')

sb.countplot(dataset.doornumber, palette=("RdBu"))



plt.subplot(1,2,2)

plt.title('Door Number vs Price')

sb.boxplot(x=dataset.doornumber, y=dataset.price, palette=("RdBu"))



plt.show()



plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

plt.title('Aspiration Histogram')

sb.countplot(dataset.aspiration, palette=("RdBu"))



plt.subplot(1,2,2)

plt.title('Aspiration vs Price')

sb.boxplot(x=dataset.aspiration, y=dataset.price, palette=("RdBu"))



plt.show()
def plott(x,fig):

    plt.subplot(4,2,fig)

    plt.title(x+' Histogram')

    sb.countplot(dataset[x])

    plt.subplot(4,2,(fig+1))

    plt.title(x+' vs Price')

    sb.boxplot(x=dataset[x], y=dataset.price)

    

plt.figure(figsize=(15,20))



plott('enginelocation', 1)

plott('cylindernumber', 3)

plott('fuelsystem', 5)

plott('drivewheel', 7)



plt.tight_layout()
#Visualising numerical data





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
def other_attributes(x,y,z):

    sb.pairplot(dataset, x_vars=[x,y,z], y_vars='price',height=4, aspect=1, kind='scatter')

    plt.show()

    

    

other_attributes('enginesize', 'boreratio', 'stroke')

other_attributes('compressionratio', 'horsepower', 'peakrpm')

other_attributes('wheelbase', 'citympg', 'highwaympg')

#adding a new feature.

dataset['mileage'] = dataset['citympg']*0.55 + dataset['highwaympg']*0.45

dataset.head()
#setting up levels for price.

dataset["brand_category"] = dataset['price'].apply(lambda x : "Budget" if x < 10000 

                                                     else ("Mid_Range" if 10000 <= x < 20000

                                                           else ("Luxury")))

dataset.head()



#bivariate analysis of mileage and price with company name

plt1 = sb.scatterplot(x = 'mileage', y = 'price', hue = 'brand_category', data = dataset)

plt1.set_xlabel('Mileage')

plt1.set_ylabel('Price of Car (Dollars)')

plt.show()
plt1 = sb.scatterplot(x = 'horsepower', y = 'price', hue = 'brand_category', data = dataset)

plt1.set_xlabel('Horsepower')

plt1.set_ylabel('Price of Car ($)')

plt.show()
plt1 = sb.scatterplot(x = 'mileage', y = 'price', hue = 'fueltype', data = dataset)

plt1.set_xlabel('Mileage')

plt1.set_ylabel('Price of Car ($)')

plt.show()
attributes = dataset[['fueltype', 'aspiration', 'carbody', 'drivewheel', 'wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginetype',

       'cylindernumber', 'enginesize',  'boreratio', 'horsepower', 'price', 'brand_category', 'mileage']]



attributes.head()
#visualising most of the attributes

plt.figure(figsize=(15,15))

sb.pairplot(attributes)

plt.show()
# Defining the map function

def dummies(x,df):

    temp = pd.get_dummies(df[x], drop_first = True)

    df = pd.concat([df, temp], axis = 1)

    df.drop([x], axis = 1, inplace = True)

    return df

# Applying the function to the  attributes



attributes = dummies('fueltype',attributes)

attributes = dummies('aspiration',attributes)

attributes = dummies('carbody',attributes)

attributes = dummies('drivewheel',attributes)

attributes = dummies('enginetype',attributes)

attributes = dummies('cylindernumber',attributes)

attributes = dummies('brand_category',attributes)

attributes.head()
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(attributes, y, test_size = 0.20, random_state = 0)



from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

num_vars = ['wheelbase', 'curbweight', 'enginesize', 'boreratio', 'horsepower','mileage','carlength','carwidth','price']

X_train[num_vars] = scaler.fit_transform(X_train[num_vars])
X_train.head()
X_train.describe()
plt.figure(figsize = (30, 25))

sb.heatmap(X_train.corr(), annot = True, cmap="RdBu")

plt.show()
y_train = X_train.pop('price')

X_train_new = X_train
import statsmodels.api as sm

model = sm.OLS(y_train, X_train_new.astype(float)).fit()

model.summary()

def build_model(X,y):

    X = sm.add_constant(X) #Adding the constant

    lm = sm.OLS(y,X).fit() # fitting the model

    print(lm.summary()) # model summary

    return X

X_train_new = build_model(X_train.astype(float),y_train)
X_train_new = X_train.drop(['rwd'], axis = 1)

X_train_new = build_model(X_train_new.astype(float),y_train)
X_train_new = X_train_new.drop(['two'], axis = 1)

X_train_new = X_train_new.drop(['rotor'], axis = 1)

X_train_new = X_train_new.drop(['carlength'], axis = 1)

X_train_new = X_train_new.drop(['carwidth'], axis = 1)

X_train_new = build_model(X_train_new.astype(float),y_train)

X_train_new = X_train_new.drop(['ohcv'], axis = 1)

X_train_new = X_train_new.drop(['curbweight'], axis = 1)

X_train_new = X_train_new.drop(['wheelbase'], axis = 1)

X_train_new = build_model(X_train_new.astype(float),y_train)
X_train_new = X_train_new.drop(['mileage'], axis = 1)



X_train_new = build_model(X_train_new.astype(float),y_train)
X_train_new = X_train_new.drop(['wagon'], axis = 1)

X_train_new = X_train_new.drop(['sedan'], axis = 1)



X_train_new = build_model(X_train_new.astype(float),y_train)
X_train_new = X_train_new.drop(['four'], axis = 1)



X_train_new = build_model(X_train_new.astype(float),y_train)
X_train_new = X_train_new.drop(['five'], axis = 1)

X_train_new = X_train_new.drop(['ohc'], axis = 1)

X_train_new = X_train_new.drop(['ohcf'], axis = 1)



X_train_new = build_model(X_train_new.astype(float),y_train)
X_train_new = X_train_new.drop(['l'], axis = 1)

X_train_new = X_train_new.drop(['dohcv'], axis = 1)

X_train_new = X_train_new.drop(['three'], axis = 1)





X_train_new = build_model(X_train_new.astype(float),y_train)
lm = sm.OLS(y_train,X_train_new).fit()

y_train_price = lm.predict(X_train_new)
# Plot the histogram of the error terms

fig = plt.figure()

sb.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)   
num_vars = ['turbo', 'enginesize', 'boreratio','gas','fwd','hardtop', 'horsepower','price']

X_test[num_vars] = scaler.fit_transform(X_test[num_vars])
#Dividing into X and y

y_test = X_test.pop('price')

XX_test = X_test
# Now let's use our model to make predictions.

X_train_new = X_train_new.drop('const',axis=1)

# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = XX_test[X_train_new.columns]



# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
y_pred = lm.predict(X_test_new.astype(float))
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