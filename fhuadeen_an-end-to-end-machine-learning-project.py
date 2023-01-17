# import relevant libraries

import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

%matplotlib inline
# import dataset

data = '../input/airbnb.xlsx'



# read dataset to pandas dataframe

pd.set_option('display.max_columns', 100)

df = pd.read_excel(data)



# view first 20 rows of the dataset

df.head(20)
# import relevant library

from sklearn.model_selection import train_test_split



# set random seed

seed = 234



# split out the test and train sets from the data

Train, Test = train_test_split(df, test_size=.33, random_state=seed)
# save Test set to local machine

#Test.to_csv('airbnbTest.csv')
# save Train set to local machine

#Train.to_csv('airbnbTrain.csv')
# analyse the percentage missing values of each variable from the original datasets

df.isnull().mean()
len(df)
# Analyse descriptive stats of numerical variables

df.describe()
df.hist()

plt.show()
# the distribution of beds. Plotted with plt so as not to tamper with the missing values yet

plt.hist(df['Beds'])

plt.show()
# the distribution of Number Of Reviews

sns.distplot(df['Number Of Reviews'], kde=True).set_title('Distribution of Number of Review')

plt.show()
# the distribution of Price

#plt.hist(df['Price'])

sns.distplot(df['Price'], kde=True).set_title('Distribution of Price')

plt.show()
# the distribution of Review Scores Rating

plt.hist(df['Review Scores Rating'])

plt.show()
# the distribution of Price

sns.boxplot(df['Beds']).set_title('Boxplot of Beds')

plt.show()
# the distribution of Number Of Reviews

sns.boxplot(df['Number Of Reviews']).set_title('Boxplot of Number of Review')

plt.show()
# the distribution of Price

sns.boxplot(df['Price']).set_title('Boxplot of Price')

plt.show()
# the distribution of Review Scores Rating

sns.boxplot(df['Review Scores Rating']).set_title('Boxplot of Review Scores Rating')

plt.show()
# Faceting Price and Room Type

sns.boxplot(df['Price'], df['Room Type'])

plt.show()
# Faceting Price and Neighbourhood (for some reason, yet unknown, Seaborn is unable to plot this graph)

sns.boxplot(df['Price'], df['Borough'])

plt.show()
# Association between Beds and Price

sns.scatterplot('Beds', 'Price', data=df)

plt.show()
# Association between Number Of Reviews and Price

sns.scatterplot('Number Of Reviews', 'Price', data=df)

plt.show()
# Association between Review Scores Rating and Price

sns.scatterplot('Review Scores Rating', 'Price', data=df)

plt.show()
df.corr(method='pearson')
# Property Type

# House, Condominium, Townhouse, Villa, Bungalow, Chalet, Castle will be converted to House.

# Dorm, Camper/RV, Treehouse, Tent, Hut and Lighthouse will be converted to Mini_house.

# Apartment, Bed & Breakfast, Other, Loft will be converted to Apartment.

# Boat, Cabin will be converted to Conveyance

Train['Property Type'].unique().tolist()
# Room Type

# all Entire home/apt will be converted to Private room

Train['Room Type'].unique().tolist()
# I will like to make prediction with the review scores rating being in continuous form so as to see the effect of 1 change in 

# reviews scores rating on price

Train['Review Scores Rating'].unique().tolist()
# Use the year values of 'Host Since' and convert it to age of Host to see if there's an effect in the number of years an

# host has been on airbnb in determining price of a home.

# Assumes it's year 2015 and so the oldest host will have an age of 8 and so on
# remove irrelevant features

Train.drop(['Host Id', 'Name', 'Review Scores Rating (bin)', 'Zipcode', 'Number of Records'], axis=1, inplace=True)

Train.head(4)
# Convert Host Since to Years (named it Age already)

Train['Age'] = Train['Host Since'].map(lambda x: x.year)

Train.head()
# fill up Age attribute NAs with the mode value 2015 to allow for easier calculation later

Train.Age.fillna(Train.Age.mode()[0], inplace=True)

Train.head()
# convert values in Age attribute to integer

Train.Age = Train.Age.fillna(0.0).astype(int)

Train.head(3)
# convert the years to age in number of years. Latest year in data is 2015

Train.Age = 2015 - Train.Age

Train.head(3)
# drop the 'Host Since' variable

Train.drop('Host Since', axis=1, inplace=True)

Train.head(2)
Train.isnull().sum()
# For Property Type

# House, Condominium, Townhouse, Villa, Bungalow, Chalet, Castle will be converted to House (They are all similar to being just a house). 

# Dorm, Camper/RV, Treehouse, Tent, Hut and Lighthouse will be converted to Mini_house.

# Apartment, Bed & Breakfast, Other, Loft will be converted to Apartment.

# Boat, Cabin will be converted to Conveyance

#df['Propety_Type'] = 

Train['Property Type'].replace({'Condominium' : 'House', 'Townhouse' : 'House', 'Villa' : 'House', 'Bungalow' : 'House', 'Chalet' : 'House', 'Castle' : 'House',

                                               'Dorm' : 'Mini_house', 'Camper/RV' : 'Mini_house', 'Treehouse' : 'Mini_house', 'Tent' : 'Mini_house', 'Hut' : 'Mini_house', 'Lighthouse' : 'Mini_house',

                                               'Bed' : 'Apartment', 'Bed & Breakfast' : 'Apartment', 'Other' : 'Apartment', 'Loft': 'Apartment',

                                               'Boat' : 'Conveyance', 'Cabin' : 'Conveyance'}, inplace=True)

Train['Property Type'].unique().tolist()
# For Room Type

# all Entire home/apt will be converted to Private room

Train['Room Type'].replace({'Entire home/apt' : 'Private room'}, inplace=True)

Train['Room Type'].unique().tolist()
# Checking correlation between predictors and the target (train set)

trainCorr = Train.corr()

trainCorr.Price.sort_values(ascending=False)
# look deeper into Beds (train set)

Train.plot(kind='scatter', x='Beds', y='Price', alpha=.1)

plt.show()
# Label encode all categorical variables to prepare for estimating missing values

Train['Borough'].replace({'Brooklyn' : 1, 'Manhattan' : 2, 'Queens' : 3, 'Bronx' : 4, 'Staten Island' : 5}, inplace=True)

Train['Room Type'].replace({'Private room' : 1, 'Shared room' : 2}, inplace=True)

Train['Property Type'].replace({'Apartment' : 1, 'House' : 2, 'Mini_house' : 3, 'Conveyance' : 4}, inplace=True)

Train.head()
# Fill up missing values with mode and median.

# Mode for Property Type as it is a categorical variable

# Mode for Beds (it's the same value as the median)

# Median for Review Scores Rating. Due to the high amount of outliers, median is better for central tendency.

Train['Property Type'].fillna(Train['Property Type'].mode()[0], inplace=True)

Train['Beds'].fillna(Train['Beds'].mode()[0], inplace=True)

Train['Review Scores Rating'].fillna(Train['Review Scores Rating'].median(), inplace=True)
# check for null values in the data set

Train.isnull().sum()
# Identify and remove possible outliers from the data

# converting Train set to arrays

trainA = Train.values # trainA means train set arrays



# import relevant library

from sklearn.neighbors import LocalOutlierFactor



# instantiate function

lof = LocalOutlierFactor()

ohat = lof.fit_predict(trainA)



# select non outliers

mask = ohat != -1

trainNO = trainA[mask, :] #trainNO means train set with no outliers



print(trainNO.shape)
# we can convert it back to pandas dataframe

trainNO = pd.DataFrame(trainNO, columns=list(Train.columns))

trainNO.head(4)
# Split out predictors from target variable

trainX = trainNO.drop(['Price'], axis=1)

trainY = trainNO['Price']

trainY.head()
# split out numerical predictors from trainX

numFeats = ['Beds', 'Number Of Reviews', 'Review Scores Rating', 'Age'] #numFeats means numerical feature columns

trainNum = trainX.loc[:, numFeats]

trainNum.head() # trainNum means trainset numerical predictors
# split out categorical predictors from trainX

catFeats = trainX.columns.drop(numFeats) # catFeats means categorical f

trainCat = trainX.loc[:, catFeats]

trainCat.head() # trainCat means train set's categorical predictors
# Pipeline for numerical predictors



# import relevant libraries

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer



# instantiate Pipeline

numPipe = Pipeline([('ImputeNum', SimpleImputer(strategy='median')), ('Scaler', StandardScaler())])
# Pipeline for categorical predictors



# import relevant library

from sklearn.preprocessing import OneHotEncoder



# instantiate Pipeline

catPipe = Pipeline([('ImputeCat', SimpleImputer(strategy='most_frequent')), ('OneHot', OneHotEncoder())])
# Combine the two pipelines using Sklearn's ColumnTransformer



from sklearn.compose import ColumnTransformer



fullPipe = ColumnTransformer([('Nums', numPipe, numFeats), ('Cats', catPipe, catFeats)])
# run pipeline on trainX set

fullPipe.fit(trainX, trainY)

trainXPrep = fullPipe.transform(trainX) # trainXPrep means prepared train predictors

trainXPrep
# import relevant libraries

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet

from sklearn.tree import DecisionTreeRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



# define the pipeline

models = []



models.append(('LR', LinearRegression()))

models.append(('Lasso', Lasso()))

models.append(('EN', ElasticNet()))

models.append(('KNN', KNeighborsRegressor()))

models.append(('SVR', SVR()))

models.append(('CART', DecisionTreeRegressor()))



# empty lists to store model names and results

results = []

names = []



# Kfold and cross validation parameters

seed = 234

num_fold = 10

num_repeat = 3

scoring = 'neg_mean_squared_error'



for name, model in models:

    kfold = RepeatedStratifiedKFold(n_splits=num_fold, n_repeats=num_repeat, random_state=seed)

    cv_results = cross_val_score(model, trainXPrep, trainY, cv=kfold, scoring=scoring, n_jobs=-1)

    

    # append each model results into empty lists created

    results.append(cv_results)

    names.append(name)

    

    # output to evaluate results

    rmse = np.sqrt(-cv_results)

    see_results = '%s: %f (%f)' % (name, rmse.mean(), rmse.std())

    print(see_results)
# To use KFold instead of RepeatedStratifiedKFold

from sklearn.model_selection import KFold



# define models

kmodels = []



kmodels.append(('LR', LinearRegression()))

kmodels.append(('Lasso', Lasso()))

kmodels.append(('EN', ElasticNet()))

kmodels.append(('KNN', KNeighborsRegressor()))

kmodels.append(('SVR', SVR()))

kmodels.append(('CART', DecisionTreeRegressor()))



# empty lists to store model names and results

kresults = []

knames = []



# Kfold and cross validation parameters

seed = 234

num_fold = 10

scoring = 'neg_mean_squared_error'



for kname, kmodel in kmodels:

    kfold_cv = KFold(n_splits=num_fold, random_state=seed)

    kcv_results = cross_val_score(kmodel, trainXPrep, trainY, cv=kfold_cv, scoring=scoring)

    

    # append each model results into empty lists created

    kresults.append(kcv_results)

    knames.append(kname)

    

    # output to evaluate results

    rmse = np.sqrt(-kcv_results)

    see_kresults = '%s: %f (%f)' % (kname, rmse.mean(), rmse.std())

    print(see_kresults)
# Try some Ensemble Techniques

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor





emodels = []



emodels.append(('RF', RandomForestRegressor()))

emodels.append(('GB', GradientBoostingRegressor()))



# empty lists to store model names and results

eresults = []

enames = []



# Kfold and cross validation parameters

seed = 234

num_fold = 10

scoring = 'neg_mean_squared_error'



for ename, emodel in emodels:

    ekfold = KFold(n_splits=num_fold, random_state=seed)

    ecv_results = cross_val_score(emodel, trainXPrep, trainY, cv=kfold_cv, scoring=scoring)

    

    # append each model results into empty lists created

    eresults.append(ecv_results)

    enames.append(ename)

    

    # output to evaluate results

    ermse = np.sqrt(-ecv_results)

    see_eresults = '%s: %f (%f)' % (ename, ermse.mean(), ermse.std())

    print(see_eresults)
# import relevant libraries

from sklearn.model_selection import GridSearchCV



# set parameters

estimators = [100, 150, 200, 250]

loss = ['ls', 'quantile']

max_feat = ['sqrt', 'log2']

param_grid = dict(loss=loss, n_estimators=estimators, max_features=max_feat)



# instantiate model

model = GradientBoostingRegressor()



# Kfold and cross validation parameters

seed = 234

num_fold = 10

scoring = 'neg_mean_squared_error'



# gridsearchCV

kfold = KFold(n_splits=num_fold, random_state=seed)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)

grid_result = grid.fit(trainXPrep, trainY)



# print result

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']

stds = grid_result.cv_results_['std_test_score']

params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):

    print("%f (%f) with: %r" % (mean, stdev, param))
# Create Age variable from Host Since

Test['Age'] = Test['Host Since'].map(lambda x: x.year) # create variable with age and populate with year of Host Since

Test['Age'].fillna(Test['Age'].mode()[0], inplace=True) # fill up missing values with mode if any

Test['Age'] = Test['Age'].fillna(0.0).astype(int) # convert values to integers

Test['Age'] = 2015 - Test['Age']



# Reduce/merge categories of Property Type and Room Type

Test['Property Type'].replace({'Condominium' : 'House', 'Townhouse' : 'House', 'Villa' : 'House', 'Bungalow' : 'House', 'Chalet' : 'House', 'Castle' : 'House',

                                               'Dorm' : 'Mini_house', 'Camper/RV' : 'Mini_house', 'Treehouse' : 'Mini_house', 'Tent' : 'Mini_house', 'Hut' : 'Mini_house', 'Lighthouse' : 'Mini_house',

                                               'Bed' : 'Apartment', 'Bed & Breakfast' : 'Apartment', 'Other' : 'Apartment', 'Loft': 'Apartment',

                                               'Boat' : 'Conveyance', 'Cabin' : 'Conveyance'}, inplace=True)

Test['Room Type'].replace({'Entire home/apt' : 'Private room'}, inplace=True)



# Replace categories in Borough, Property Type and Room Type with numbers

Test['Borough'].replace({'Brooklyn' : 1, 'Manhattan' : 2, 'Queens' : 3, 'Bronx' : 4, 'Staten Island' : 5}, inplace=True)

Test['Room Type'].replace({'Private room' : 1, 'Shared room' : 2}, inplace=True)

Test['Property Type'].replace({'Apartment' : 1, 'House' : 2, 'Mini_house' : 3, 'Conveyance' : 4}, inplace=True)



# Drop irrelevant variables - 'Host Id', 'Host Since', 'Name', 'Review Scores Rating (bin)', 'Zipcode', 'Number of Records'

Test.drop(['Host Id', 'Host Since', 'Name', 'Review Scores Rating (bin)', 'Zipcode', 'Number of Records'], axis=1, inplace=True)

Test.head(4)
# Split out Test predictors from test target

testX = Test.drop(['Price'], axis=1)

testY = Test['Price']

testX.head()
# Transform Test predictors

testXPrep = fullPipe.transform(testX) # testXPrep means prepared test predictors

testXPrep
from sklearn.metrics import mean_squared_error



finalModel = grid_result.best_estimator_





predictions = finalModel.predict(testXPrep)



finalMSE = mean_squared_error(testY, predictions)

finalRMSE = np.sqrt(finalMSE)



print(f' MSE is {finalMSE} and RMSE is {finalRMSE}')
from sklearn.externals import joblib



joblib.dump([fullPipe, finalModel], 'airbnbModel.sav', compress=1)