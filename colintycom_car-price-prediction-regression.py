import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
#Read in the Data Dictionary provided to give clearer insight into the dataset's variables

df_dict = pd.read_excel('../input/car-data/Data Dictionary - carprices.xlsx')

df_dict.rename(columns={"Unnamed: 7": "Column_Name", "Unnamed: 11": "Description"}, inplace=True)

df_dict = df_dict.drop(df_dict.index[0:3])

df_dict = df_dict.drop(df_dict.index[26:28])

df_dict = df_dict.filter(["Column_Name","Description"])

pd.set_option('display.max_colwidth', -1)

df_dict
#Read in CarPrice_Assignment csv dataset

df = pd.read_csv('../input/car-data/CarPrice_Assignment.csv')

print(df.shape)

df.head()
#Look closer at the CarPrices dataset using describe() and info()

df.describe()
df.info()
#When looking at the CarName column, you can see that the Manufacturer is at the beginning of the values. We want to extract this as a new column

Manufacturer = df['CarName'].apply(lambda x : x.split(' ')[0])

Manufacturer.unique()
Manufacturer = Manufacturer.replace('maxda', 'mazda')

Manufacturer = Manufacturer.replace('Nissan', 'nissan')

Manufacturer = Manufacturer.replace('porcshce', 'porsche')

Manufacturer = Manufacturer.replace('toyouta', 'toyota')

Manufacturer = Manufacturer.replace('vokswagen', 'volkswagen')

Manufacturer = Manufacturer.replace('vw', 'volkswagen')

Manufacturer.unique()
df.insert(1,'manufacturer',Manufacturer)

df.head()
#Since there are so many unique Car Names compared to the overall size of the dataset, I will remove 'CarName' now that we have Manufacturer to be used instead.

df.drop(['CarName'],axis=1,inplace=True)

df.head()
#Looking into the dependent variable "price"

plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

plt.title('Price Boxplot')

sns.boxplot(y="price", data=df)



plt.subplot(1,2,2)

plt.title('Price Distribution')

sns.distplot(df["price"])
#Looking at how the different manufacturers compare with against their prices

ax  = sns.boxplot(x="manufacturer", y="price", data=df)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
#Getting counts of each manufacturer as well

ax = sns.countplot(x="manufacturer", data=df)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
df.select_dtypes(include=['object'])
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

plt.title('Fuel Type')

sns.countplot(x="fueltype", data=df)



plt.subplot(1,3,2)

plt.title('Aspiration')

sns.countplot(x="aspiration", data=df)



plt.subplot(1,3,3)

plt.title('Number of Doors')

sns.countplot(x="doornumber", data=df)



plt.show()
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

plt.title('Carbody')

sns.countplot(x="carbody", data=df)



plt.subplot(1,3,2)

plt.title('Drivewheel')

sns.countplot(x="drivewheel", data=df)



plt.subplot(1,3,3)

plt.title('Engine Location')

sns.countplot(x="enginelocation", data=df)



plt.show()
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

plt.title('Engine Type')

sns.countplot(x="enginetype", data=df)



plt.subplot(1,3,2)

plt.title('Cylinder Number')

sns.countplot(x="cylindernumber", data=df)



plt.subplot(1,3,3)

plt.title('Fuel System')

sns.countplot(x="fuelsystem", data=df)



plt.show()
plt.figure(figsize=(18,5))



plt.subplot(1,3,1)

plt.title('Fuel Type')

sns.boxplot(x="fueltype", y="price", data=df)



plt.subplot(1,3,2)

plt.title('Aspiration')

sns.boxplot(x="aspiration", y="price", data=df)



plt.subplot(1,3,3)

plt.title('Number of Doors')

sns.boxplot(x="doornumber", y="price", data=df)



plt.show()
plt.figure(figsize=(18,5))



plt.subplot(1,3,1)

plt.title('Carbody')

sns.boxplot(x="carbody", y="price", data=df)



plt.subplot(1,3,2)

plt.title('Drivewheel')

sns.boxplot(x="drivewheel", y="price", data=df)



plt.subplot(1,3,3)

plt.title('Engine Location')

sns.boxplot(x="enginelocation", y="price", data=df)



plt.show()
plt.figure(figsize=(18,5))



plt.subplot(1,3,1)

plt.title('Engine Type')

sns.boxplot(x="enginetype", y="price", data=df)



plt.subplot(1,3,2)

plt.title('Cylinder Number')

sns.boxplot(x="cylindernumber", y="price", data=df)



plt.subplot(1,3,3)

plt.title('Fuel System')

sns.boxplot(x="fuelsystem", y="price", data=df)



plt.show()
df.select_dtypes(include=['int64', 'float64'])
#Looking to understand the symboling variable

plt.figure(figsize=(18,5))



plt.subplot(1,3,1)

sns.boxplot(x="symboling", y="price", data=df)



plt.subplot(1,3,2)

sns.countplot(x="symboling", data=df)



plt.subplot(1,3,3)

sns.scatterplot(x="symboling", y="price", data=df)
#Comparing the other numerical variables to price

plt.figure(figsize=(20,5))



plt.subplot(1,3,1)

plt.title('Wheel Base')

sns.scatterplot(x="wheelbase", y="price", data=df)



plt.subplot(1,3,2)

plt.title('Engine Size')

sns.scatterplot(x="enginesize", y="price", data=df)



plt.subplot(1,3,3)

plt.title('Curb Weight')

sns.scatterplot(x="curbweight", y="price", data=df)



plt.show()
plt.figure(figsize=(20,5))



plt.subplot(1,3,1)

plt.title('Car Length')

sns.scatterplot(x="carlength", y="price", data=df)



plt.subplot(1,3,2)

plt.title('Car Width')

sns.scatterplot(x="carwidth", y="price", data=df)



plt.subplot(1,3,3)

plt.title('Car Height')

sns.scatterplot(x="carheight", y="price", data=df)



plt.show()
plt.figure(figsize=(20,5))



plt.subplot(1,3,1)

plt.title('Bore Ratio')

sns.scatterplot(x="boreratio", y="price", data=df)



plt.subplot(1,3,2)

plt.title('Stroke')

sns.scatterplot(x="stroke", y="price", data=df)



plt.subplot(1,3,3)

plt.title('Compression Ratio')

sns.scatterplot(x="compressionratio", y="price", data=df)



plt.show()
plt.figure(figsize=(23,5))



plt.subplot(1,4,1)

plt.title('Horsepower')

sns.scatterplot(x="horsepower", y="price", data=df)



plt.subplot(1,4,2)

plt.title('Peak RPM')

sns.scatterplot(x="peakrpm", y="price", data=df)



plt.subplot(1,4,3)

plt.title('City MPG')

sns.scatterplot(x="citympg", y="price", data=df)



plt.subplot(1,4,4)

plt.title('Highway MPG')

sns.scatterplot(x="highwaympg", y="price", data=df)



plt.show()
sns.pairplot(df[['price','wheelbase','enginesize','curbweight','carlength','carwidth','boreratio','stroke','horsepower','citympg','highwaympg']])

plt.show()
sns.heatmap(df[['price','wheelbase','enginesize','curbweight','carlength','carwidth','boreratio','stroke','horsepower','citympg','highwaympg']].corr())

plt.show()
#As mentioned above, since citympg and highwaympg are so strongly correlated and when you think about it logically these 2 variables 

    #are both related to the car's fuel economy and how many mpg the car can drive.

    

#I will combine these 2 variables into one by taking the adding the values and dividing by 2. This will give a rough "average mpg".



df["avgmpg"] = (df["citympg"]+df["highwaympg"])/2

df["avgmpg"].head(10)

df_cars = df[['price', 'manufacturer', 'fueltype', 'aspiration','carbody','drivewheel','enginetype','cylindernumber','fuelsystem','wheelbase','enginesize','curbweight','carlength','carwidth','boreratio','stroke','horsepower','avgmpg']]

df_cars
df_dummy = pd.get_dummies(df_cars['manufacturer'])

df_cars = pd.concat([df_cars, df_dummy], axis = 1)

df_cars.drop('manufacturer', axis = 1, inplace=True)



df_dummy = pd.get_dummies(df_cars['fueltype'])

df_cars = pd.concat([df_cars, df_dummy], axis = 1)

df_cars.drop('fueltype', axis = 1, inplace=True)



df_dummy = pd.get_dummies(df_cars['aspiration'])

df_cars = pd.concat([df_cars, df_dummy], axis = 1)

df_cars.drop('aspiration', axis = 1, inplace=True)



df_dummy = pd.get_dummies(df_cars['carbody'])

df_cars = pd.concat([df_cars, df_dummy], axis = 1)

df_cars.drop('carbody', axis = 1, inplace=True)



df_dummy = pd.get_dummies(df_cars['drivewheel'])

df_cars = pd.concat([df_cars, df_dummy], axis = 1)

df_cars.drop('drivewheel', axis = 1, inplace=True)



df_dummy = pd.get_dummies(df_cars['enginetype'])

df_cars = pd.concat([df_cars, df_dummy], axis = 1)

df_cars.drop('enginetype', axis = 1, inplace=True)



df_dummy = pd.get_dummies(df_cars['cylindernumber'])

df_cars = pd.concat([df_cars, df_dummy], axis = 1)

df_cars.drop('cylindernumber', axis = 1, inplace=True)



df_dummy = pd.get_dummies(df_cars['fuelsystem'])

df_cars = pd.concat([df_cars, df_dummy], axis = 1)

df_cars.drop('fuelsystem', axis = 1, inplace=True)



print(df_cars.shape)
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



y = df_cars['price']

X = df_cars.drop(['price'], axis = 1)



X_train_org, X_test_org, y_train, y_test = train_test_split(X, y, random_state = 0)



scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train_org)

#.fit_transform first fits the original data and then transforms it

X_test = scaler.transform(X_test_org)



print("X_train shape: ", X_train.shape)

print("y_train shape: ", y_train.shape)

print("X_test shape: ", X_test.shape)

print("y_test shape: ", y_test.shape)
from sklearn.linear_model import LinearRegression

from sklearn import metrics





lreg = LinearRegression()

lreg.fit(X_train, y_train)

print("R2 Training Score: ", lreg.score(X_train, y_train))

print("R2 Testing Score: ", lreg.score(X_test, y_test))
print(lreg.intercept_)

lreg.coef_
test_predict = lreg.predict(X_test)

test_predict = pd.DataFrame(test_predict,columns=['Predicted_Price'])

test_predict['Predicted_Price'] = round(test_predict['Predicted_Price'],2)



y_test_index = y_test.reset_index()

y_test_index = y_test_index.drop(columns='index', axis = 1)

test_predict = pd.concat([y_test_index, test_predict], axis = 1)

test_predict.head(15)
plt.figure(figsize=(15,5))



plt.subplot(1,2,1)

sns.scatterplot(x="price", y="Predicted_Price", data=test_predict)



plt.subplot(1,2,2)

sns.regplot(x="price", y="Predicted_Price", data=test_predict)
from  sklearn.linear_model import Lasso



#Testing different alpha values for the L1 regularization

alpha_range = [0.01, 0.1, 1, 10, 100]

train_score_list = []

test_score_list = []



for alpha in alpha_range: 

    lasso = Lasso(alpha)

    lasso.fit(X_train,y_train)

    train_score_list.append(lasso.score(X_train,y_train))

    test_score_list.append(lasso.score(X_test, y_test))
#Comparing different alpha values to see which produces the best scores

%matplotlib inline

import matplotlib.pyplot as plt

plt.plot(alpha_range, train_score_list, c = 'g', label = 'Train Score')

plt.plot(alpha_range, test_score_list, c = 'b', label = 'Test Score')

plt.xscale('log')

plt.legend(loc = 3)

plt.xlabel(r'$\alpha$')
lasso = Lasso(100)

lasso.fit(X_train,y_train)

print(lasso.score(X_train,y_train))

print(lasso.score(X_test,y_test))
test_predict = lasso.predict(X_test)

test_predict = pd.DataFrame(test_predict,columns=['Predicted_Price'])

test_predict['Predicted_Price'] = round(test_predict['Predicted_Price'],2)

y_test_index = y_test.reset_index()

y_test_index = y_test_index.drop(columns='index', axis = 1)

test_predict = pd.concat([y_test_index, test_predict], axis = 1)

print(test_predict.head(15))

sns.regplot(x="price", y="Predicted_Price", data=test_predict)
from  sklearn.linear_model import Ridge



#Testing different alpha values for the L2 regularization

x_range = [0.01, 0.1, 1, 10, 100]

train_score_list = []

test_score_list = []



for alpha in x_range: 

    ridge = Ridge(alpha)

    ridge.fit(X_train,y_train)

    train_score_list.append(ridge.score(X_train,y_train))

    test_score_list.append(ridge.score(X_test, y_test))
%matplotlib inline

import matplotlib.pyplot as plt

plt.plot(x_range, train_score_list, c = 'g', label = 'Train Score')

plt.plot(x_range, test_score_list, c = 'b', label = 'Test Score')

plt.xscale('log')

plt.legend(loc = 3)

plt.xlabel(r'$\alpha$')
print(train_score_list)

print(test_score_list)
ridge = Ridge(1)

ridge.fit(X_train,y_train)

test_predict = ridge.predict(X_test)

test_predict = pd.DataFrame(test_predict,columns=['Predicted_Price'])

test_predict['Predicted_Price'] = round(test_predict['Predicted_Price'],2)

y_test_index = y_test.reset_index()

y_test_index = y_test_index.drop(columns='index', axis = 1)

test_predict = pd.concat([y_test_index, test_predict], axis = 1)

test_predict.head(15)
sns.regplot(x="price", y="Predicted_Price", data=test_predict)
#Re-show X and y (y = price, X = all other variables from df_cars)

print(X.shape)

X.head()
#Scale the X dataset

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)
print(y.shape)

y[:5]
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



nn_list = list(range(1,51))



param_grid = {'n_neighbors': nn_list}



grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5)

grid_search.fit(X_scaled, y)

print(grid_search.score(X_scaled,y))

print("Best parameters: {}".format(grid_search.best_params_))

knn_reg = KNeighborsRegressor(n_neighbors=23)

knn_reg.fit(X_scaled, y)

knn_reg.score(X_scaled, y)
param_grid = {'alpha': [0.01,0.1,1,10,100]}



grid_search = GridSearchCV(Lasso(), param_grid, cv=5)

grid_search.fit(X_scaled, y)

print("Best parameters: {}".format(grid_search.best_params_))
lasso = Lasso(alpha=100)

lasso.fit(X_scaled, y)

lasso.score(X_scaled, y)
param_grid = {'alpha': [0.01,0.1,1,10,100]}



grid_search = GridSearchCV(Ridge(), param_grid, cv=5)

grid_search.fit(X_scaled, y)

print("Best parameters: {}".format(grid_search.best_params_))
ridge = Ridge(alpha=1)

ridge.fit(X_scaled, y)

ridge.score(X_scaled,y)