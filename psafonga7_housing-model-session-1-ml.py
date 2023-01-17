# some imports



from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))



# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)

 

# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



# Common imports

import numpy as np

import os



# to make this notebook's output stable across runs

np.random.seed(42)



# To plot pretty figures

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)

plt.rc('font', size=12) 

plt.rc('figure', figsize = (12, 5))



# Settings for the visualizations

import seaborn as sns

sns.set_style("whitegrid")

sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2,'font.family': [u'times']})



import pandas as pd

pd.set_option('display.max_rows', 55)

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', 50)



#geopandas for geolocation

#import geopandas



# Ignore useless warnings (see SciPy issue #5998)

import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")



# create output folder

if not os.path.exists('output'):

    os.makedirs('output')

if not os.path.exists('output/session1'):

    os.makedirs('output/session1')
## load data

train_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/train_set.csv',index_col=0) 

test_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/test_set.csv',index_col=0) 



# print a summary of the data in Melbourne data

train_set.describe()
test_set.describe()
# print the top elements from the dataset

train_set.head()
# print the dataset size

print("There is", train_set.shape[0], "samples")

print("Each sample has", train_set.shape[1], "features")
# print the top elements from the dataset

train_set.head()
# As it can be seen the database contains several features, some of them numerical and some of them are categorical.

# It is important to check each of the to understand it.

len(train_set['Postcode'].unique())
len(test_set['Postcode'].unique())
# we can see the type of each features as follows

train_set.dtypes
# print those categorical features

train_set.select_dtypes(include=['object']).describe()
# print those categorical features

test_set.select_dtypes(include=['object']).describe()
# We can check how many different type there is in the dataset using the folliwing line

train_set["Type"].value_counts()

#h = house / casa 

#u = unit / apartamento

#t = town housing / casa adosada

sns.countplot(y="Type", data=train_set, color="c")
# We can check how many different type there is in the dataset using the folliwing line

train_set["Postcode"].value_counts()
test_set["Postcode"].value_counts()
# We can check how many different type there is in the dataset using the folliwing line

train_set["Regionname"].value_counts()
test_set["Regionname"].value_counts()
sns.countplot(x="Suburb", data=train_set, color="c", order = train_set['Suburb'].value_counts().index )
# We can check how many different type there is in the dataset using the folliwing line

train_set["CouncilArea"].value_counts()
sns.countplot(y="CouncilArea", data=train_set, color="c", order = train_set['CouncilArea'].value_counts().index)
# We can check how many different type there is in the dataset using the folliwing line

train_set["Regionname"].value_counts()
sns.countplot(y="Regionname", data=train_set, color="c", order = train_set['Regionname'].value_counts().index)
sns.distplot(train_set["Price"])

plt.show()


print("Skewness: %f" % train_set['Price'].skew())

print("Kurtosis: %f" % train_set['Price'].kurt())
#Using Pearson Correlation to see possible correlations between all data



plt.figure(figsize=(12,10))

cor = train_set.corr()[['Price']].sort_values(by='Price', ascending=False)

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
sns.distplot(train_set["YearBuilt"])

plt.show()
plt.scatter(x=train_set["Price"], y=train_set["YearBuilt"] )

plt.xlabel('Price')

plt.ylabel('YearBuilt');

plt.show()
plt.scatter(x=train_set["Postcode"], y=train_set["Price"] )

plt.xlabel('Postcode')

plt.ylabel('Price');

plt.show()
plt.scatter(x=train_set["Price"], y=train_set["Regionname"] )

plt.xlabel('Price')

plt.ylabel('Regionname');

plt.show()
plt.scatter(x=train_set["Price"], y=train_set["Suburb"] )

plt.xlabel('Price')

plt.ylabel('Suburb');

plt.show()
plt.scatter(x=train_set["Type"], y=train_set["Price"] )

plt.xlabel('BuildingArea')

plt.ylabel('Price');

plt.show()
plt.scatter(x=train_set["Date"], y=train_set["Price"] )

plt.xlabel('BuildingArea')

plt.ylabel('Price');

plt.show()
plt.scatter(x=train_set["Distance"], y=train_set["Price"] )

plt.xlabel('BuildingArea')

plt.ylabel('Price');

plt.show()
#Using Pearson Correlation to see possible correlations between the data and the price



plt.figure(figsize=(12,10))

cor = train_set.corr()[['Price']].sort_values(by='Price', ascending=False)

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
#Using Pearson Correlation to see possible correlations between the data



plt.figure(figsize=(12,10))

cor = train_set.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.coolwarm)

plt.show()
plt.scatter(x=train_set["Lattitude"], y=train_set["Price"])

plt.xlabel('Rooms')

plt.ylabel('Predicted price');

plt.show()
plt.scatter(x=train_set["Longtitude"], y=train_set["Price"])

plt.xlabel('Longtitude')

plt.ylabel('Predicted price');

plt.show()
missing_val_count_by_column = (train_set.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
train_set.count()
test_set.count()
missing_val_count_by_column = (test_set.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
## the features



features = ['Rooms','Landsize', 'BuildingArea', 'YearBuilt']

## DEFINE YOUR FEATURES

X = train_set[features].fillna(0)

y = train_set[['Price']]



## the model

# KNeighborsRegressor

from sklearn import neighbors

n_neighbors = 3 # you can modify this paramenter (ONLY THIS ONE!!!)

model = neighbors.KNeighborsRegressor(n_neighbors)



## fit the model

model.fit(X, y)



## predict training set

y_pred = model.predict(X)



## Evaluate the model and plot it

from sklearn.metrics import mean_squared_error, r2_score

print("----- EVALUATION ON TRAIN SET ------")

print("RMSE",np.sqrt(mean_squared_error(y, y_pred)))

print("R^2: ",r2_score(y, y_pred))





plt.scatter(y, y_pred)

plt.xlabel('Price')

plt.ylabel('Predicted price');

plt.show()



## predict the test set and generate the submission file

X_test = test_set[features].fillna(0)

y_pred = model.predict(X_test)



df_output = pd.DataFrame(y_pred)

df_output = df_output.reset_index()

df_output.columns = ['index','Price']



df_output.to_csv('output/session1/baseline.csv',index=False)
# load data 

train_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/train_set.csv',index_col=0) 

test_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/test_set.csv',index_col=0) 



missing_val_count_by_column = (test_set.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
missing_val_count_by_column = (train_set.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
# create a copy of train set to clean, modify and upgrade the data

Z = train_set.copy()

#cliping





#train_set["New Age"]= data["Age"].clip(lower = lower, upper = upper) 

## clipping

Z["Rooms"]= Z["Rooms"].clip( upper = 10)

Z["Bedroom2"]= Z["Bedroom2"].clip( upper = 10)

Z["YearBuilt"]= Z["YearBuilt"].fillna(1950)

Z["Price"]= Z["Price"].clip( upper = 4000000)

# change labels

change_labels = {"Type":     {"h": 1.0, "t": 0.6, "u": 0.2}}



Z.replace(change_labels, inplace=True)







## new features space

Z['RegionMean'] = Z.groupby('Regionname')['Price'].transform('mean')

Z['PostcodeMean'] = Z.groupby('Postcode')['Price'].transform('mean')

Z['CouncilAreaMean'] = Z.groupby('CouncilArea')['Price'].transform('mean')





#most detailed besides lat/long (issues wit hgeopanda, as well as data quite corrupt) is suburb mean

#Z['SuburbMean'] = Z.groupby('Suburb')['Price'].transform('mean')

Z['SuburbMean'] = Z['Suburb'].copy()

Z['SuburbMean'].replace(train_set.groupby('Suburb')['Price'].mean(), inplace=True)



#fix suburbmean empty values

Z["SuburbMean"] = pd.to_numeric(Z['SuburbMean'], downcast='float', errors='coerce')

#replace NaN with gloabal mean in case of missing values

Z["SuburbMean"]= Z["SuburbMean"].fillna(1.078470e+06)



Z.to_csv('output/session1/Ztrain_clean.csv',index=False)



#train_set['Location'] =  [Point(xy) for xy in zip(train_set['Longtitude'], train_set['Lattitude'])]

#train_set['Location'] =  geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(train_set.Longtitude, train_set.Lattitude))

#Z['Location'] = list(zip(Z['Longtitude'], Z['Lattitude']))





missing_val_count_by_column = (Z.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
missing_val_count_by_column = (Z.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
Z.groupby('Regionname')['Price'].mean()
Z.groupby('Suburb')['Price'].mean()
Z.groupby('Postcode')['Price'].mean().describe()
set(test_set['Postcode'].unique()).issubset(set(train_set['Postcode'].unique()))
set(train_set['Postcode'].unique()).issubset(set(test_set['Postcode'].unique()))
#values un train_set missing in test_set

pd.DataFrame({'Postcode': np.setdiff1d(test_set['Postcode'], train_set['Postcode'])})

test_set.loc[test_set['Postcode'] == 3044]

pd.concat([test_set['Postcode'],  train_set['Postcode']]).drop_duplicates(keep=False)
#values un test:set missing in train_set

df1 = pd.DataFrame({'PostcodeaMissing': np.setdiff1d(test_set['Postcode'], train_set['Postcode'])})

print(df1)
#values un test:set missing in train_set

df2 = pd.DataFrame({'PostcodeaMissing': np.setdiff1d(train_set['Postcode'], test_set['Postcode'])})

print(df2)
df3 = pd.DataFrame({'Suburb': np.setdiff1d(test_set['Suburb'], train_set['Suburb'])})

print(df3)
missingSuburbs = pd.DataFrame({'Suburb': np.setdiff1d(train_set['Suburb'], test_set['Suburb'])})

print(missingSuburbs)
Z.groupby('Regionname')['Price'].mean()
Z.describe()
allfeatures = ['SuburbMean','Rooms', 'Type', 'Price', 'Distance','Bathroom', 'RegionMean', 'Postcode', 'Bedroom2', 'Car', 

               'Landsize', 'BuildingArea', 'YearBuilt', 'PostcodeMean', 'Lattitude', 'Longtitude', 'CouncilAreaMean']

Z = Z[allfeatures]
#Using Pearson Correlation to see possible correlations between the data and the price



plt.figure(figsize=(12,10))

cor = Z.corr()[['Price']].sort_values(by='Price', ascending=False)

sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)

plt.show()
#Using Pearson Correlation to see possible correlations between the data and price



plt.figure(figsize=(12,10))

cor = Z.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.coolwarm)

plt.show()
# TODO normalize and invert values of negative correlations

# normalize: create a scaler object

#from sklearn.preprocessing import minmax_scale

# fit and transform the data

#Z = minmax_scale(Z.astype(np.float64)).fillna(0)



missing_val_count_by_column = (Z.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
## the features

#allfeatures = ['SuburbMean','Rooms', 'Type', 'Price', 'Distance','Bathroom', 'RegionMean', 'Postcode', 'Bedroom2','Car','Landsize', 'BuildingArea', 'YearBuilt', 'PostcodeMean', 'Lattitude', 'Longtitude']

#features = ['Rooms','Landsize', 'BuildingArea', 'YearBuilt']

#features = ['SuburbMean','Rooms', 'Type', 'Price', 'Distance','Bathroom', 'RegionMean','Car','Landsize', 'PostcodeMean']



features = ['SuburbMean','RegionMean','Rooms', 'Type','Bathroom', 'Car']



## DEFINE YOUR FEATURES



# normalize: create a scaler object

#from sklearn.preprocessing import minmax_scale

# fit and transform the data

y = train_set[['Price']]



X = Z[features].fillna(0)

X.to_csv('output/session1/Xtrain.csv',index=False)

## the model

# KNeighborsRegressor

from sklearn import neighbors

n_neighbors = 5# you can modify this paramenter (ONLY THIS ONE!!!)

model = neighbors.KNeighborsRegressor(n_neighbors)



## fit the model

model.fit(X, y)



## predict training set

y_pred = model.predict(X)



## Evaluate the model and plot it

from sklearn.metrics import mean_squared_error, r2_score



# R^2 (coefficient of determination) regression score function.

#Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). 

#A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.



#Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors).

#Residuals are a measure of how far from the regression line data points are; RMSE is a measure 

#of how spread out these residuals are. In other words, it tells you how concentrated the data is around the line of best fit. 



print("----- EVALUATION ON TRAIN SET ------")

print("RMSE",np.sqrt(mean_squared_error(y, y_pred)))

print("R^2: ",r2_score(y, y_pred))





plt.scatter(y, y_pred)

plt.xlabel('Price')

plt.ylabel('Predicted price');

plt.show()
train_set.describe()
# load data 

train_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/train_set.csv',index_col=0) 

test_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/test_set.csv',index_col=0) 



features = ['SuburbMean','RegionMean','Rooms', 'Type','Bathroom', 'Car']



## predict the test set and generate the submission file

#test_set['SuburbMean'] = train_set.groupby('Suburb')['Price'].transform('mean')

#test_set['RegionMean'] = train_set.groupby('Regionname')['Price'].transform('mean')

#test_set['PostcodeMean'] = train_set.groupby('Postcode')['Price'].transform('mean')



#train_set["New Age"]= data["Age"].clip(lower = lower, upper = upper)



## clipping

test_set["Rooms"]= test_set["Rooms"].clip( upper = 10)

test_set["Bedroom2"]= test_set["Bedroom2"].clip( upper = 10)

test_set["YearBuilt"]= test_set["YearBuilt"].fillna(1950)



# create new series

test_set['SuburbMean'] = test_set['Suburb'].copy()

test_set['PostcodeMean'] = test_set['Postcode'].copy()

test_set['CouncilAreaMean'] = test_set['CouncilArea'].copy()

test_set['RegionMean'] = test_set['Regionname'].copy()



# change labels

test_set['SuburbMean'].replace(train_set.groupby('Suburb')['Price'].mean(), inplace=True)

#test_set['PostcodeMean'].replace(train_set.groupby('Postcode')['Price'].mean(), inplace=True)

test_set['RegionMean'].replace(train_set.groupby('Regionname')['Price'].mean(), inplace=True)

#test_set['CouncilAreaMean'].replace(train_set.groupby('CouncilArea')['Price'].mean(), inplace=True)







change_labels = {"Type":     {"h": 1.0, "t": 0.6, "u": 0.2}}

test_set.replace(change_labels, inplace=True)





#fix suburbmean empty values

test_set["SuburbMean"] = pd.to_numeric(test_set['SuburbMean'], downcast='float', errors='coerce')

#replace NaN with gloabal mean

test_set["SuburbMean"]= test_set["SuburbMean"].fillna(1.078470e+06)



#fix postalcode empty values

#pd.to_numeric(test_set['Suburb'], downcast='float', errors='coerce')

#replace NaN with gloabal mean

#test_set["Suburb"]= test_set["Suburb"].fillna(1.078470e+06)



#we'll use suburb mean. In case is not avaible, we'll use postcode 

#test_set.replace(train_set.groupby('Suburb')['Price'].mean(), inplace=True)



#eleven = lambda x: (len(x['col1']) >= 11)

#test_set['Suburb'] = test_set['Suburb'].map(lambda x:  train_set.groupby('Suburb')['Price'].mean() if test_set['Suburb'].in(train_set.groupby('Suburb')) else train_set.groupby('Regionname')['Price'].mean())

#train_set.groupby('Regionname')['Price'].mean(), inplace=True

#map(lambda x: True if x % 2 == 0 else False, range(1, 11))

#test_set['Suburb'].map(lambda x: train_set.groupby('Suburb')['Price'].mean())

#test_set['Suburb'] = test_set['Suburb'].astype(float)





#test_set['Suburb']=df['Hourly Rate'].mask(pd.isnan, df['Daily Rate'])

#pd.to_numeric(s, errors='coerce').fillna(0, downcast='infer')

#test_set['Suburb'].fillna(train_set.groupby('Postcode')['Price'].mean())





#test_set.describe()
#test_set['SuburbMean'].describe()
#test_set.to_csv('output/session1/test_set_clean.csv',index=True)
#test_set["SuburbMean"].unique().dtype
#test_set.dtypes
#test_set.describe()
#test_set
missing_val_count_by_column = (test_set.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
missing_val_count_by_column = (test_set.isna().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
## predict the test set and generate the submission file

X_test = test_set[features].fillna(0)

X_test.to_csv('output/session1/Xtest.csv',index=False)

y_pred = model.predict(X_test)



df_output = pd.DataFrame(y_pred)

df_output = df_output.reset_index()

df_output.columns = ['index','Price']



df_output.to_csv('output/session1/prediction.csv',index=False)
X_test
#df_output.describe()
#train_set['Price'].describe()