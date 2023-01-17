#import libraries for pre-processing

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd

from pandas.plotting import scatter_matrix

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import numpy as np



from dateutil.parser import parse

from datetime import datetime

from scipy.stats import norm



# import all what you need for machine learning

import sklearn

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import cross_validate

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import RobustScaler
#load data

#housing= pd.read_csv('C:/Users/EAMC/Desktop/melbourne-housing-market/Melbourne_housing_FULL.csv', sep=',')

housing= pd.read_csv('../input/Melbourne_housing_FULL.csv')
# visualize the first two rows for every column

housing.head()
#Check the type of variables

housing.info()
#change Postcode

housing['Postcode']= housing['Postcode'].astype('object')

#change Bathroom and car to integer. Before the transformation I need to convert all the missing values to 0. 

#lastly, change Propertycount to category

housing['Propertycount']= housing['Propertycount'].astype('object')
sns.lmplot(data= housing, x='Bedroom2', y='Rooms')
#drop Bedroom2

housing= housing.drop(['Bedroom2'], axis=1)
#check basic statistics

housing.describe()
# check number of bathrooms

housing['Bathroom'].value_counts()
housing.loc[housing.Bathroom>7].head()
#check building area 

housing.loc[housing.BuildingArea<1].head()
housing['BuildingArea'].loc[housing.BuildingArea<1].count()
#use the unary operator ~ to delete the rows

housing = housing[~(housing['BuildingArea'] < 1)]  

#check the deletion

housing['BuildingArea'].loc[housing.BuildingArea<1].count()
#it is important now to reset the index, otherwise I will have some missing rows in my dataframe, which may be troublesome later.

housing = housing.reset_index()
sns.boxplot(data = housing, y = 'BuildingArea')
housing.loc[housing.BuildingArea>40000]
#replace outlier building area

housing['BuildingArea'].replace(44515.0, 445, inplace=True)
# check YearBuilt > 2018

print(housing['YearBuilt'].loc[housing.YearBuilt>2018])

#replace 2106 with 2016 and 2019 with 2018

housing['YearBuilt'].replace([2106, 2019], [2016, 2018], inplace=True)
# check missing data

housing.isnull().sum()
# We will save the "cured" data columns in variables

#first with the mean

priceWithMean = housing['Price'].fillna(housing['Price'].mean())

BAWithMean = housing['BuildingArea'].fillna(housing['BuildingArea'].mean())



#now with the median

priceMedian = housing['Price'].fillna(housing['Price'].median())

BAMedian = housing['BuildingArea'].fillna(housing['BuildingArea'].median())
missVIDsJoint = housing['Price'].isnull() | housing['BuildingArea'].isnull()

# missVIDsJoint now has a True for items that are missing an Age or a Fare value
# create a dictionary to indicate different colors, missing values will be orange

colorChoiceDict = {True: (1.0, 0.55, 0.0, 1.0), False: (0.11, 0.65, 0.72, 0.1)}



# create a column with color values using list comprehension

colorCol = [colorChoiceDict[val] for val in missVIDsJoint]
plt.style.use('ggplot')



f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(19, 8))

ax1.scatter(BAWithMean, priceWithMean, c = colorCol, linewidth=0)

ax1.set_title('MV with mean')

ax1.set_xlabel('Building Area')

ax1.set_ylabel('Price')

ax2.scatter(BAMedian, priceMedian, c = colorCol, linewidth=0)

ax2.set_title('MV with median')

ax2.set_xlabel('Building Area')

ax2.set_ylabel('Price')
housing['Price']= housing['Price'].fillna(housing['Price'].median())

housing['Landsize']= housing['Landsize'].fillna(housing['Landsize'].median())

#Similarly, fill the only missing value of  all the other numeric values

housing['Distance'] = housing['Distance'].fillna(housing['Distance'].median())

housing['BuildingArea']= housing['BuildingArea'].fillna(housing['BuildingArea'].median())

housing['Lattitude']= housing['Lattitude'].fillna(housing['Lattitude'].median())

housing['Longtitude']= housing['Longtitude'].fillna(housing['Longtitude'].median())

housing['YearBuilt']= housing['YearBuilt'].fillna(housing['YearBuilt'].median())

housing.isnull().sum()
housing['Bathroom']= housing['Bathroom'].fillna(housing['Bathroom'].mode()[0])

housing['Car']= housing['Car'].fillna(housing['Car'].mode()[0])

housing['CouncilArea']= housing['CouncilArea'].fillna(housing['CouncilArea'].mode()[0])

housing['Regionname']= housing['Regionname'].fillna(housing['Regionname'].mode()[0])

housing['Propertycount']= housing['Propertycount'].fillna(housing['Propertycount'].mode()[0])

housing['Postcode']= housing['Postcode'].fillna(housing['Postcode'].mode()[0])

housing.isnull().sum()
housing['Bathroom'] = pd.to_numeric(housing['Bathroom']).round(0).astype(int)

housing['Car'] = pd.to_numeric(housing['Car']).round(0).astype(int)
# create additional columns filled with 0 values

housing["isOutlierPrice"] = 0 

housing["isOutlierDistance"] = 0



# save the mean and standard deviation in variables

meanPrice = housing['Price'].mean()

stdDevPrice = housing['Price'].std()



meanDistance = housing['Distance'].mean()

stdDevDistance = housing['Distance'].std()



#mark outliers as 

housing['isOutlierPrice'] = np.where(abs(housing['Price'] - meanPrice) > 5 * stdDevPrice, 1, 0)

housing['isOutlierDistance'] = np.where(abs(housing['Distance'] - meanDistance) > 5 * stdDevDistance, 1, 0)
#create a function to compute the percentage of missing values

def percent(nom, denom):

    res= (nom*100)/denom

    print("%.3f%%" % round(res,3))



#percentage of MV for Price

percent(housing["isOutlierPrice"].value_counts()[1], housing["isOutlierPrice"].value_counts()[0])   
#percentage of MV for Landsize

percent(housing["isOutlierDistance"].value_counts()[1], housing["isOutlierDistance"].value_counts()[0])   
# This part helps us to generate a color array with different colors for the 1D outliers we compute



# first create an empty list

colorColumn = []

# we make use of the HEX color codes to use nicely distinguisable colors

for i in range(len(housing)):

    if housing["isOutlierPrice"][i]== 1:

        colorColumn.append("#D06B36") # orange color

    elif housing["isOutlierDistance"][i] == 1:

        colorColumn.append("#40A0C9") # a blueish color

    else:

        colorColumn.append("#B9BCC0") # gray



plt.figure(figsize=(15,10))

plt.xlabel('Price')

plt.suptitle('Price vs. Distance')

plt.ylabel('Distance')

plt.scatter(housing.Distance, housing.Price , c = colorColumn, s = 50, linewidth=0)
#take just the price outlier

housing.iloc[:,:19][housing.Price > 11000000]
# We now get a part of the data frame as a numpy matrix to use in scipy

housing.dropna()

columnValues = housing.as_matrix(["Price", "Distance"])



# In order to generate a "mean vector", we use the mean values already computed above.

# Notice that we make use of the reshape() function to get the mean vector in a compatible shape

# as the data values.

meanVector = np.asarray([meanPrice, meanDistance]).reshape(1,2)



# We make us of the scipy function which does the computations itself.

# Alternatively, one can provide a covariance matrix that is computed outside as a parameter.

# In cases where robustness of the covariance matrix is the issue, this can be a good option.



# first import the spatial subpackage from scipy

from scipy import spatial

mahalanobisDistances = spatial.distance.cdist(columnValues, meanVector, 'mahalanobis')[:,0]



# We create a new figure where we use a color mapping and use the computed mahalanobis distances 

# as the mapping value

plt.figure(figsize=(15,10))

plt.xlabel('Distance')

plt.suptitle('Price & Distance')

plt.ylabel('Price')

plt.scatter(housing.Distance, housing.Price , c = mahalanobisDistances, cmap = plt.cm.Greens, s = 50, linewidth=0)
housing['houseAge'] = 2018-housing['YearBuilt']
#create the new column data restructuring the original Date column with pd.to_datetime

housing['data'] = pd.to_datetime(housing['Date'])
# calculate day of year

housing['doy'] = housing['data'].dt.dayofyear

# Create year

housing['Year'] = housing['data'].dt.year



#to divide by season it's better to use the day of the year instead of the months

spring = range(80, 172)

summer = range(172, 264)

fall = range(264, 355)

# winter = everything else



daje = []

for i in housing['doy']:

    if i in spring:

        season = 'spring'

    elif i in summer:

        season = 'summer'

    elif i in fall:

        season = 'fall'

    else:

        season = 'winter'

    daje.append(season)   



#add the resulting column to the dataframe (after transforming it as a Series)

housing['season']= pd.Series(daje)
housing.info()
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing = train_set.copy()
#create my palette

myPal = ["#1E90FF", "#FFD700", "#00EEEE", "#668B8B", "#EAEAEA", "#FF3030"]

sns.set_palette(myPal)

sns.palplot(sns.color_palette())
l= ['Bathroom', 'Rooms', 'Car', 'season', 'Year']

for i in l:

    plt.figure()

    sns.countplot(x=i, data=housing)
l= [housing['Price'], housing['Distance'], housing['BuildingArea'], housing['houseAge'], housing['Propertycount']]

for i in l:

    plt.figure(figsize=(11,6))

    sns.distplot(i, fit=norm, kde=False)
plt.figure(figsize=(15,10))

sns.distplot(housing['BuildingArea'], fit=norm, bins=120, kde=False)

plt.xlim(0,1000)
# Suplots of categorical features v price

sns.set_style('darkgrid')

f, axes = plt.subplots(2,2, figsize = (15,15))



# Plot [0,0]

sns.boxplot(data = housing, x = 'season', y = 'Price', ax = axes[0, 0])

axes[0,0].set_xlabel('Season')

axes[0,0].set_ylabel('Price')

axes[0,0].set_title('Season & Price')



# Plot [0,1]

sns.violinplot(data = housing, x = 'Year', y = 'Price', ax = axes[0, 1])

axes[0,1].set_xlabel('Year')

axes[0,1].set_ylabel('Price')

axes[0,1].set_title('Year & Price')



# Plot [1,0]

sns.boxplot(x = 'Type', y = 'Price', data = housing, ax = axes[1,0])

axes[1,0].set_xlabel('Type')

axes[1,0].set_ylabel('Price')

axes[1,0].set_title('Type & Price')



# Plot [1,1]

sns.boxplot(x = 'Rooms', y = 'Price', data = housing, ax = axes[1,1])

axes[1,1].set_xlabel('Rooms')

axes[1,1].set_ylabel('Price')

axes[1,1].set_title('Rooms & Price')
#use shape to count the number of rows of the database grouped by day using a pivot table.

housing.pivot_table('Price', index='data', aggfunc='sum').shape
import calendar

# create new column storing the month of each operation

housing['month'] = housing['data'].dt.month

#use group by (alternative to pivot_table) to have the total value of houses sold per month

by_month= housing.groupby('month')['Price'].sum()

#plot figure

plt.figure(figsize=(15,10))

plt.plot(by_month, color="red")

plt.xlabel('Month')

plt.suptitle('Price by months')

plt.ylabel('Price')

plt.xticks(np.arange(13), calendar.month_name[0:13], rotation=20)
#create index month-year

housing['month_year'] = housing['data'].dt.to_period('M')

#use groupby to compute the price for each available month, then store the result in a dataframe

by_year_month= pd.Series.to_frame(housing.groupby('month_year')['Price'].sum())

#draw graph

fig, ax = plt.subplots(figsize=(15,10))

by_year_month.plot(ax=ax, xticks=by_year_month.index, rot=45)

ax.set_xticklabels(by_year_month.index)
#select only the data we are interested in

attributes= ['Price', 'Distance', 'Bathroom', 'Rooms', 'Car', 'Landsize', 'BuildingArea', 'houseAge', 'Lattitude', 'Longtitude', 

             'Year', 'Propertycount']

h= housing[attributes]



#whitegrid

sns.set_style('whitegrid')

#compute correlation matrix...

corr_matrix=h.corr(method='spearman')

#...and show it with a heatmap

#first define the dimension

plt.figure(figsize=(20,15))



# Generate a mask for the upper triangle

mask = np.zeros_like(corr_matrix, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0, vmax=1, vmin =-1, annot=True,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Scatterplot

f, axes = plt.subplots(figsize = (15,10))

plt.subplot(221)

sns.regplot(data= housing, x='BuildingArea', y='Price')

plt.subplot(222)

sns.regplot(data= housing, x='houseAge', y='Price')

plt.subplot(223)

sns.regplot(data= housing, x='Rooms', y='Price')

plt.subplot(224)

sns.regplot(data= housing, x='Distance', y='Price')
import warnings

with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    # Price and distance

    sns.jointplot(y='Price', x='Rooms', data=housing, kind='hex', gridsize=15)
sns.lmplot(data= housing, x='houseAge', y= 'Price', hue= 'Type')
sns.lmplot(data= housing, x='BuildingArea', y= 'Price', hue='Type')
sns.lmplot(data= housing, x='Rooms', y= 'Price', hue='Type')
housing.plot(kind="scatter", x="Longtitude", y="Lattitude", alpha=0.4,

c=housing.Price, cmap=plt.get_cmap("jet"), label= 'Price by location', figsize=(15,10)) 

plt.ylabel("Latitude", fontsize=14)



plt.legend(fontsize=14)
housing['Price_cut']= housing['Price'].loc[housing.Price<3500000]
f, axes = plt.subplots(1,2, figsize = (12,7))

# Plot [0,0] full price

housing['Price'].hist(ax = axes[0])

axes[0].set_title('BEFORE CUT')

axes[0].set_xlabel('Price')

# Plot [0,1] price cut

housing['Price_cut'].hist(ax = axes[1])

axes[1].set_xlabel('Price')

axes[1].set_title('AFTER CUT')
housing.plot(kind="scatter", x="Longtitude", y="Lattitude", alpha=0.4,

c=housing.Price_cut, cmap=plt.get_cmap("jet"), label= 'Price by location', figsize=(15,10)) 

plt.ylabel("Latitude", fontsize=14)



plt.legend(fontsize=14)
housing.info()
#dummy variable

hD= pd.get_dummies(housing, columns= ['Type', 'Regionname', 'season'])

#drop useless variables

hD= hD.drop(['Suburb', 'Address', 'Method', 'SellerG', 'Date', 'Postcode', 'CouncilArea', 'isOutlierPrice', 'isOutlierDistance',

            'YearBuilt', 'data', 'doy', 'month', 'month_year', 'Price_cut'], axis=1)

#check variables

hD.info()
#create x and y variables

X = hD.drop("Price", axis=1)

Y = hD["Price"].copy()

#transform to array size

#feature scaling

scaler = RobustScaler()

hD= scaler.fit_transform(hD.astype(np.float64))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = .20, random_state= 42)
#fit decision tree

tree = DecisionTreeRegressor()

tree.fit(x_train, y_train)

#fit random forest

forest = RandomForestRegressor(n_jobs=-1)

forest.fit(x_train, y_train)

#fit regression

lin_reg = LinearRegression(n_jobs=-1)

lin_reg.fit(x_train, y_train)
models= [('lin_reg', lin_reg), ('random forest', forest), ('decision tree', tree)]

from sklearn.metrics import mean_squared_error

for i, model in models:    

    predictions = model.predict(x_train)

    MSE = mean_squared_error(y_train, predictions)

    RMSE = np.sqrt(MSE)

    msg = "%s = %.2f" % (i, round(RMSE, 2))

    print('RMSE of', msg)
for i, model in models:

    # Make predictions on train data

    predictions = model.predict(x_train)

    # Performance metrics

    errors = abs(predictions - y_train)

    # Calculate mean absolute percentage error (MAPE)

    mape = np.mean(100 * (errors / y_train))

    # Calculate and display accuracy

    accuracy = 100 - mape    

    #print result

    msg = "%s= %.2f"% (i, round(accuracy, 2))

    print('Accuracy of', msg,'%')
models= [('lin_reg', lin_reg), ('forest', forest), ('dt', tree)]

scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']



#for each model I want to test three different scoring metrics. Therefore, results[0] will be lin_reg x MSE, 

# results[1] lin_reg x MSE and so on until results [8], where we stored dt x r2



results= []

metric= []

for name, model in models:

    for i in scoring:

        scores = cross_validate(model, x_train, y_train, scoring=i, cv=10, return_train_score=True)

        results.append(scores)
#this is an example of the stored results

results[8]
#THIS IS FOR Linear regression

#if you change signa and square the Mean Square Error you get the RMSE, which is the most common metric to accuracy

LR_RMSE_mean = np.sqrt(-results[0]['test_score'].mean())

LR_RMSE_std= results[0]['test_score'].std()

# note that also here I changed the sign, as the result is originally a negative number for ease of computation

LR_MAE_mean = -results[1]['test_score'].mean()

LR_MAE_std= results[1]['test_score'].std()

LR_r2_mean = results[2]['test_score'].mean()

LR_r2_std = results[2]['test_score'].std()



#THIS IS FOR RF

RF_RMSE_mean = np.sqrt(-results[3]['test_score'].mean())

RF_RMSE_std= results[3]['test_score'].std()

RF_MAE_mean = -results[4]['test_score'].mean()

RF_MAE_std= results[4]['test_score'].std()

RF_r2_mean = results[5]['test_score'].mean()

RF_r2_std = results[5]['test_score'].std()



#THIS IS FOR DT

DT_RMSE_mean = np.sqrt(-results[6]['test_score'].mean())

DT_RMSE_std= results[6]['test_score'].std()

DT_MAE_mean = -results[7]['test_score'].mean()

DT_MAE_std= results[7]['test_score'].std()

DT_r2_mean = results[8]['test_score'].mean()

DT_r2_std = results[8]['test_score'].std()
modelDF = pd.DataFrame({

    'Model'       : ['Linear Regression', 'Random Forest', 'Decision Trees'],

    'RMSE_mean'    : [LR_RMSE_mean, RF_RMSE_mean, DT_RMSE_mean],

    'RMSE_std'    : [LR_RMSE_std, RF_RMSE_std, DT_RMSE_std],

    'MAE_mean'   : [LR_MAE_mean, RF_MAE_mean, DT_MAE_mean],

    'MAE_std'   : [LR_MAE_std, RF_MAE_std, DT_MAE_std],

    'r2_mean'      : [LR_r2_mean, RF_r2_mean, DT_r2_mean],

    'r2_std'      : [LR_r2_std, RF_r2_std, DT_r2_std],

    }, columns = ['Model', 'RMSE_mean', 'RMSE_std', 'MAE_mean', 'MAE_std', 'r2_mean', 'r2_std'])



modelDF.sort_values(by='r2_mean', ascending=False)
sns.factorplot(x= 'Model', y= 'RMSE_mean', data= modelDF, kind='bar', legend='True')
from sklearn.model_selection import GridSearchCV



param_grid = [

{'n_estimators': [10, 25], 'max_features': [5, 10], 

 'max_depth': [10, 50, None], 'bootstrap': [True, False]}

]



grid_search_forest = GridSearchCV(forest, param_grid, cv=10, scoring='neg_mean_squared_error')

grid_search_forest.fit(x_train, y_train)
#now let's how the RMSE changes for each parameter configuration

cvres = grid_search_forest.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
#find the best model of grid search

grid_search_forest.best_estimator_
# Performance metrics

grid_best= grid_search_forest.best_estimator_.predict(x_train)

errors = abs(grid_best - y_train)

# Calculate mean absolute percentage error (MAPE)

mape = np.mean(100 * (errors / y_train))

# Calculate and display accuracy

accuracy = 100 - mape    

#print result

print('The best model from grid-search has an accuracy of', round(accuracy, 2),'%')
#RMSE

grid_mse = mean_squared_error(y_train, grid_best)

grid_rmse = np.sqrt(grid_mse)

print('The best model from the grid search has a RMSE of', round(grid_rmse, 2))
from sklearn.model_selection import RandomizedSearchCV

from pprint import pprint



# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 20, stop = 200, num = 5)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(1, 45, num = 3)]

# Minimum number of samples required to split a node

min_samples_split = [5, 10]



# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split}



pprint(random_grid)
# Use the random grid to search for best hyperparameters



# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = forest, param_distributions = random_grid, n_iter = 10, cv = 10, verbose=2, random_state=42, n_jobs = -1, scoring='neg_mean_squared_error')

# Fit the random search model

rf_random.fit(x_train, y_train)
#now let's how the RMSE changes for each parameter configuration

cvres2 = rf_random.cv_results_

for mean_score, params in zip(cvres2["mean_test_score"], cvres2["params"]):

    print(np.sqrt(-mean_score), params)
# best random model 

rf_random.best_estimator_
# best combination of parameters of random search

rf_random.best_params_
rf_random.best_estimator_
# Performance metrics (MAPE)

random_best= rf_random.best_estimator_.predict(x_train)

errors = abs(random_best - y_train)

# Calculate mean absolute percentage error (MAPE)

mape = np.mean(100 * (errors / y_train))

# Calculate and display accuracy

accuracy = 100 - mape    

#print result

print('The best model from the randomized search has an accuracy of', round(accuracy, 2),'%')
#this is the RMSE

final_mse = mean_squared_error(y_train, random_best)

final_rmse = np.sqrt(final_mse)

print('The best model from the randomized search has a RMSE of', round(final_rmse, 2))
# extract the numerical values of feature importance from the grid search

importances = rf_random.best_estimator_.feature_importances_



#create a feature list from the original dataset (list of columns)

# What are this numbers? Let's get back to the columns of the original dataset

feature_list = list(X.columns)



#create a list of tuples

feature_importance= sorted(zip(importances, feature_list), reverse=True)



#create two lists from the previous list of tuples

df = pd.DataFrame(feature_importance, columns=['importance', 'feature'])

importance= list(df['importance'])

feature= list(df['feature'])



#see df

print(df)
# Set the style

plt.style.use('bmh')

# list of x locations for plotting

x_values = list(range(len(feature_importance)))



# Make a bar chart

plt.figure(figsize=(15,10))

plt.bar(x_values, importance, orientation = 'vertical')

# Tick labels for x axis

plt.xticks(x_values, feature, rotation='vertical')

# Axis labels and title

plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
final_model = rf_random.best_estimator_

# Predicting test set results

final_pred = final_model.predict(x_test)

final_mse = mean_squared_error(y_test, final_pred)

final_rmse = np.sqrt(final_mse)

print('The final RMSE on the test set is', round(final_rmse, 2))
#calculate accuracy

errors = abs(final_pred - y_test)

# Calculate mean absolute percentage error (MAPE)

mape = np.mean(100 * (errors / y_test))

# Calculate and display accuracy

accuracy = 100 - mape    

#print result

print('The best model achieves on the test set an accuracy of', round(accuracy, 2),'%')
max_depths = np.linspace(1, 50, 50, endpoint=True)



train_results = []

test_results = []



for i in max_depths:

    dt = RandomForestRegressor(max_depth=i)

    dt.fit(x_train, y_train)    

    #compute accuracy for train data

    housing_tree = dt.predict(x_train)

    errors = abs(housing_tree - y_train)

    # Calculate mean absolute percentage error (MAPE)

    mape = 100 * (errors / y_train)

    # Calculate and display accuracy

    accuracy = 100 - np.mean(mape)

    #append results of accuracy

    train_results.append(accuracy)

    

    #now again for test data

    housing_tree = dt.predict(x_test)

    errors = abs(housing_tree - y_test)

    # Calculate mean absolute percentage error (MAPE)

    mape = 100 * (errors / y_test)

    # Calculate and display accuracy

    accuracy = 100 - np.mean(mape)

    #append results of accuracy

    test_results.append(accuracy)

    

from matplotlib.legend_handler import HandlerLine2D

line1, = plt.plot(max_depths, train_results, 'b', label='Train accuracy')

line2, = plt.plot(max_depths, test_results, 'r', label= 'Test accuracy')



plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})

plt.ylabel('Accuracy score')

plt.xlabel('Tree depth')