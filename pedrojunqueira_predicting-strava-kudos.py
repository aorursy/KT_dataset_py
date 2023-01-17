# import packages to use in the notebook

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import os

from datetime import datetime

import seaborn as sns

# load file

os.getcwd()
ride = pd.read_csv('../input/strava_data.csv')
# only analysing rides so drop all runs imported 

ride = ride[ride['type']=="Ride"]
# df dimension

ride.shape
ride.tail()
# Check 1 ride detail

ride.iloc[1035,:]
# feature information

ride.info()
#check columns names

ride.columns
# Function to calculate missing values by column

def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
missing_values_table(ride)
#drop not usefull columns

# ids, booleans, objects and etc no data 

ride.drop(columns=['Unnamed: 0','id', 'external_id', 'upload_id','name', 'athlete','map', 'trainer', 'commute', 'manual', 'private',

       'flagged','device_watts','has_kudoed','max_watts',

       'weighted_average_watts','gear_id' ,'type','workout_type','photo_count'],axis=1,inplace=True)



# check what is left

ride.info()
# check the object columns which is text and will be hard to sort e.g dates as a string does not make sense

ride.iloc[:,6:11].head()
# transform to date time

ride['start_date']= ride['start_date'].apply(lambda x: datetime.strptime(x[0:19],'%Y-%m-%d %X'))
ride['start_date_local']= ride['start_date_local'].apply(lambda x: datetime.strptime(x[0:19],'%Y-%m-%d %X'))
# create a city atribute

ride['city']= ride['timezone'].apply(lambda x: x.split('/')[-1])
# expand Lat and Lon for start

s_lat_lon = ride['start_latlng'].str.split(',',expand=True).rename(index=int, columns={0: "s_lat", 1: "s_lon"})
# expand Lat and Lon for end

e_lat_lon = ride['end_latlng'].str.split(',',expand=True).rename(index=int, columns={0: "e_lat", 1: "e_lon"})

#Add to the main frame

ride =pd.concat([ride,s_lat_lon],axis=1)
#Add to the main frame

ride =pd.concat([ride,e_lat_lon],axis=1)
# drop original columns

ride.drop(columns=['start_latlng','end_latlng','timezone'],axis=1,inplace=True)
# Check data frame

ride.head()
#fix the lat lon columns 

#remove de [] from lat lon and convert to float

ride['s_lat'] = ride['s_lat'].str.replace('[','').astype(float)

ride['e_lat'] = ride['e_lat'].str.replace('[','').astype(float)

ride['s_lon'] = ride['s_lon'].str.replace(']','').astype(float)

ride['e_lon'] = ride['e_lon'].str.replace(']','').astype(float)
# check if all numeric

ride.info()
#Check head

ride.head()
# ckeck final null values

ride.isna().sum()
# only 3 value. will replace with with average

ride = ride.fillna(ride.mean())

# no values nulls

ride.isna().sum()
# final look at columns

ride.describe()
#Analyse histogram of all variables



ride.hist(bins=50,figsize=(20,15))
# Plotting Scatter Matrix

# just numeric excluding latitude and city

g=sns.pairplot(data=ride.iloc[:,:-5]);

g.fig.set_size_inches(20,20)

#Correlation against kudos

ride.corr()['kudos_count'].sort_values(ascending=False)[1:]
ride.drop(['comment_count'],axis=1,inplace=True)
# visualise kudos overtime



pd.pivot_table(ride,values='kudos_count',index='start_date_local',aggfunc=np.mean).sort_index().plot()
ride.shape # before dropping rows
ride = ride[ride['start_date_local'] > datetime(2017,6,1)]
ride.shape # 30% dataset reduced
# plot again

pd.pivot_table(ride,values='kudos_count',index='start_date_local',aggfunc=np.mean).sort_index().plot()
ride.corr().loc[:,'kudos_count'].sort_values(ascending=False)[1:]
corr = ride.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(15,12))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(240, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)



plt.title('Correlation Matrix Plot')
selected_columns = ride.loc[:,['max_speed',              

'moving_time',             

'kilojoules',     

'total_elevation_gain',    

'distance',

'elev_high',

'total_photo_count',

'average_watts',

'athlete_count','kudos_count']]      
# pair plot the non correlated and in order of importance

g=sns.pairplot(data=selected_columns);

g.fig.set_size_inches(20,20)
# plot geographical data

# only adelaide

fig, ax = plt.subplots(figsize=(8,8))

plt.style.use('ggplot')

data = ride[ride['city']=='Adelaide']

plt.scatter(y=data['e_lat'],x=data['e_lon'])

#ride[ride['city']=='Adelaide'].plot(kind='scatter',y='s_lat',x='s_lon',c='moving_time')

# check all current selected variables

ride.info()
ride.drop(columns=['e_lon','e_lat','s_lat','s_lon','start_date','start_date_local'],axis=1,inplace=True)
ride.info()
ride = pd.concat([ride,pd.get_dummies(ride['city'])],axis=1)

ride.drop('city',axis=1,inplace=True)
# split sets

from sklearn.model_selection import train_test_split

X = ride.drop('kudos_count',axis=1).values

y = ride['kudos_count'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(len(X_train)+len(X_test))
# feature scalling

from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()

X_train_std = stdsc.fit_transform(X_train)

X_test_std = stdsc.transform(X_test)

# train model

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train_std,y_train)
# select 3 first elements of train set

some_data = X_train_std[:3]

some_predictions = lr.predict(some_data)

some_labels = y_train[:3]

print(f'some predictions{some_predictions} and some labels{some_labels}')
# calculate rmse

from sklearn.metrics import mean_squared_error

kudos_predictions_train = lr.predict(X_train_std)

kudos_predictions_test = lr.predict(X_test_std)

train_mse_lr = mean_squared_error(y_train,kudos_predictions_train)

train_rmse_lr = np.sqrt(train_mse_lr)



print(f'mse train {train_mse_lr:.2f}')

print(f'rmse train {train_rmse_lr:.2f}')
# Cross validation code

from sklearn.model_selection import cross_val_score

scores = cross_val_score(lr, X_train_std,y_train,scoring='neg_mean_squared_error',cv=10)

lr_rmse_scores = np.sqrt(-scores)



def display_scores(scores):

    print('Scores: ', scores)

    print('Mean: ', scores.mean())

    print('Standard deviarion: ', scores.std())
# mse Scores

display_scores(scores)



#rmse Scores

display_scores(lr_rmse_scores)
# decision tree regressor

from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=3)

tree.fit(X_train_std,y_train)

kudos_predictions_train_tree = tree.predict(X_train_std)

kudos_predictions_test_tree = tree.predict(X_test_std)

train_mse_tree = mean_squared_error(y_train,kudos_predictions_train_tree)

train_rmse_tree = np.sqrt(train_mse_tree)

print(f'mse train {train_mse_tree:.2f}')

print(f'rmse train {train_rmse_tree:.2f}')
# Cross Validation tree



scores = cross_val_score(tree, X_train_std,y_train,scoring='neg_mean_squared_error',cv=10)

tree_rmse_scores = np.sqrt(-scores)
# mse Scores

display_scores(scores)



#rmse Scores

display_scores(tree_rmse_scores)
# random forest  regressor

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(n_estimators=1000,criterion='mse',random_state=1,n_jobs=-1)

forest.fit(X_train_std,y_train)

kudos_predictions_train_forest = forest.predict(X_train_std)

kudos_predictions_test_forest = forest.predict(X_test_std)

train_mse_forest = mean_squared_error(y_train,kudos_predictions_train_forest)

train_rmse_forest = np.sqrt(train_mse_forest)

print(f'mse train {train_mse_forest:.2f}')

print(f'rmse train {train_rmse_forest:.2f}')
# Cross Validation Forest



scores = cross_val_score(forest, X_train_std,y_train,scoring='neg_mean_squared_error',cv=10)

forest_rmse_scores = np.sqrt(-scores)
# mse Scores

display_scores(scores)



#rmse Scores

display_scores(forest_rmse_scores)
# getting the parameters from the first training

forest.get_params()
from sklearn.model_selection import RandomizedSearchCV

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 5)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}





print(random_grid)
# do a random search first for some guidance and find best hyperoarameters

rf = RandomForestRegressor()



# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)



# Fit the random search model

rf_random.fit(X_train_std, y_train)



rf_random.best_params_
from sklearn.model_selection import GridSearchCV



# Create the parameter grid 

param_grid = {

    'bootstrap': [True],

    'max_depth': [ 10, 10],

    'max_features': [2, 3],

    'min_samples_leaf': [1, 2],

    'min_samples_split': [2, 4],

    'n_estimators': [100, 500, 1000],

     'n_jobs': [-1]

}



# Create a based model

rf = RandomForestRegressor()



# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 10, scoring='neg_mean_squared_error')

# fit

grid_search.fit(X_train_std,y_train)
# get best params

grid_search.best_params_
# get best estimator

grid_search.best_estimator_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'],cvres['params']):

    print(np.sqrt(-mean_score), params)
# finally train best estimator and get the rmse for best parameter



forest = grid_search.best_estimator_

forest.fit(X_train_std,y_train)



# predict and calculate rmse

kudos_predictions_train_forest = forest.predict(X_train_std)

kudos_predictions_test_forest = forest.predict(X_test_std)

train_mse_forest = mean_squared_error(y_train,kudos_predictions_train_forest)

train_rmse_forest = np.sqrt(train_mse_forest)



print(f'mse train {train_mse_forest:.2f}')

print(f'rmse train {train_rmse_forest:.2f}')
# Cross Validation Forest with best estimator



scores = cross_val_score(forest, X_train_std,y_train,scoring='neg_mean_squared_error',cv=10)

forest_rmse_scores = np.sqrt(-scores)



#rmse Scores

display_scores(forest_rmse_scores)

# random forest  regressor



forest = RandomForestRegressor(n_estimators=1000,criterion='mse',random_state=1,n_jobs=-1)

forest.fit(X_train_std,y_train)



kudos_predictions_train_forest = forest.predict(X_train_std)

kudos_predictions_test_forest = forest.predict(X_test_std)

train_mse_forest = mean_squared_error(y_train,kudos_predictions_train_forest)

train_rmse_forest = np.sqrt(train_mse_forest)



print(f'mse train {train_mse_forest:.2f}')

print(f'rmse train {train_rmse_forest:.2f}')
# Evaluate Residuals Plain Regression



fig, ax = plt.subplots(figsize=(8,8))

plt.style.use('ggplot')

plt.scatter(kudos_predictions_train,kudos_predictions_train-y_train,c='blue',marker='o',label='traing data')

plt.scatter(kudos_predictions_test,kudos_predictions_test-y_test,c='lightgreen',marker='s',label='test data')

plt.title('Regression Residuals')

plt.xlabel('predicted values')

plt.ylabel('residuals')

plt.legend(loc='upper left')

plt.hlines(y=0,xmin=0,xmax=80)

plt.xlim([0,80])

plt.show()



# lets import another metric that in the case would be intersting analyse

from sklearn.metrics import r2_score

r2_train = r2_score(y_train,kudos_predictions_train)

r2_test = r2_score(y_test,kudos_predictions_test)





print(f'r2 train {r2_train:.2f}')

print(f'r2 test {r2_test:.2f}')



# Also do the test evaluation of rmse



test_mse_lr = mean_squared_error(y_test,kudos_predictions_test)

test_rmse_lr = np.sqrt(test_mse_lr)



print(f'rmse train {train_rmse_lr:.2f}')

print(f'rmse test {test_rmse_lr:.2f}')



# Evaluate Residuals Decision Tree



fig, ax = plt.subplots(figsize=(8,8))

plt.style.use('ggplot')

plt.scatter(kudos_predictions_train_tree,kudos_predictions_train_tree-y_train,c='blue',marker='o',label='traing data')

plt.scatter(kudos_predictions_test_tree,kudos_predictions_test_tree-y_test,c='lightgreen',marker='s',label='test data')

plt.title('Decision Tree Regression Residuals')

plt.xlabel('predicted values')

plt.ylabel('residuals')

plt.legend(loc='upper left')

plt.hlines(y=0,xmin=0,xmax=80)

plt.xlim([0,80])

plt.show()

np.max(kudos_predictions_test_tree)
r2_train = r2_score(y_train,kudos_predictions_train_tree)

r2_test = r2_score(y_test,kudos_predictions_test_tree)



print(f'r2 train {r2_train:.2f}')

print(f'r2 test {r2_test:.2f}')



# Also do the test evaluation of rmse



test_mse_tree = mean_squared_error(y_test,kudos_predictions_test_tree)

test_rmse_tree = np.sqrt(test_mse_tree)



print(f'rmse train {train_rmse_tree:.2f}')

print(f'rmse test {test_rmse_tree:.2f}')



# Evaluate Residuals for random forest



fig, ax = plt.subplots(figsize=(8,8))

plt.style.use('ggplot')

plt.scatter(kudos_predictions_train_forest,kudos_predictions_train_forest-y_train,c='blue',marker='o',label='traing data')

plt.scatter(kudos_predictions_test_forest,kudos_predictions_test_forest-y_test,c='lightgreen',marker='s',label='test data')

plt.title('Random Forest Regression Residuals')

plt.xlabel('predicted values')

plt.ylabel('residuals')

plt.legend(loc='upper left')

plt.hlines(y=0,xmin=0,xmax=80)

plt.xlim([0,80])

plt.show()

r2_train = r2_score(y_train,kudos_predictions_train_forest)

r2_test = r2_score(y_test,kudos_predictions_test_forest)



print(f'r2 train {r2_train:.2f}')

print(f'r2 test {r2_test:.2f}')





# Also do the test evaluation of rmse



test_mse_forest = mean_squared_error(y_test,kudos_predictions_test_forest)

test_rmse_forest = np.sqrt(test_mse_forest)



print(f'rmse train {train_rmse_forest:.2f}')

print(f'rmse test {test_rmse_forest:.2f}')

# evaluating feature importances

feat_labels = ride.drop('kudos_count',axis=1).columns

importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for feature in range(X_train.shape[1]):

    print(f'{feature} - {feat_labels[indices[feature]]} {importances[indices[feature]]:.2f}')
# plotting feature importance

plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(10,5))

plt.title('Feature Importance',fontsize=15)

plt.bar(range(X_train.shape[1]),importances[indices],color='lightblue',align='center')

plt.xticks(range(X_train.shape[1]),feat_labels[indices],rotation=90)

plt.xlim([-1,X_train.shape[1]])

plt.tight_layout()

# list of features in order of importance

feature_importance = [feat_labels[indices[feature]] for feature in range(X_train.shape[1]) ]
feature_importance
# collecting rmses for training and validation CV from the most import feature to the least important feature



train_rmse_s = []

validation_rmse_s = []

for i in range(len(feature_importance)):

    col = feature_importance[:i+1]

    

    X = ride.loc[:,col].values

    y = ride['kudos_count'].values



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



    stdsc = StandardScaler()

    X_train_std = stdsc.fit_transform(X_train)

    X_test_std = stdsc.transform(X_test)



    forest = RandomForestRegressor(n_estimators=1000,criterion='mse',random_state=1,n_jobs=-1)

    forest.fit(X_train_std,y_train)



    kudos_predictions_train_forest = forest.predict(X_train_std)

    kudos_predictions_test_forest = forest.predict(X_test_std)

    train_mse = mean_squared_error(y_train,kudos_predictions_train_forest)

    train_rmse = np.sqrt(train_mse)

    train_rmse_s.append(train_rmse)

    scores = cross_val_score(forest, X_train_std,y_train,scoring='neg_mean_squared_error',cv=10)

    forest_rmse_scores = np.sqrt(-scores)

    validation_rmse = forest_rmse_scores.mean()

    validation_rmse_s.append(validation_rmse)

    

    

plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(10,7))

plt.title('rmse for trainig and test from adding feature by importance',fontsize=15)

plt.plot(range(1,len(feature_importance)+1),train_rmse_s,color='lightblue',label='rmse train',lw=3,marker='o')

for i in range(1,len(feature_importance)):

    plt.annotate(f'{train_rmse_s[i]:.2f}',

            xy=(i+1, train_rmse_s[i]), xycoords='data')

plt.plot(range(1,len(feature_importance)+1),validation_rmse_s,color='lightgreen',label='rmse validation',lw=3,marker='o')

for i in range(1,len(feature_importance)):

    plt.annotate(f'{validation_rmse_s[i]:.2f}',

            xy=(i+1, validation_rmse_s[i]), xycoords='data')

plt.xlabel("Number of important features")

plt.ylabel("rmse")

plt.xlim(1,max(range(1,len(feature_importance)+1)))

plt.legend(loc='upper right')

plt.tight_layout()

# select top 8 features



top8 = feature_importance[:8]



# pre-process the data



X = ride.loc[:,top8].values

y = ride['kudos_count'].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)



stdsc = StandardScaler()

X_train_std = stdsc.fit_transform(X_train)

X_test_std = stdsc.transform(X_test)



# Fit model

forest = RandomForestRegressor(n_estimators=1000,criterion='mse',random_state=1,n_jobs=-1)

forest.fit(X_train_std,y_train)



# Predict and calculate rmse

kudos_predictions_train_forest = forest.predict(X_train_std)

kudos_predictions_test_forest = forest.predict(X_test_std)

train_mse_forest = mean_squared_error(y_train,kudos_predictions_train_forest)

test_mse_forest = mean_squared_error(y_test,kudos_predictions_test_forest)

train_rmse_forest = np.sqrt(train_mse_forest)

test_rmse_forest = np.sqrt(test_mse_forest)

r2_train = r2_score(y_train,kudos_predictions_train_forest)

r2_test = r2_score(y_test,kudos_predictions_test_forest)





print(f'rmse train {train_rmse_forest:.2f}')

print(f'rmse test {test_rmse_forest:.2f}')

print(f'r2 train {r2_train:.2f}')

print(f'r2 test {r2_test:.2f}')

from sklearn.model_selection import learning_curve



train_sizes, train_scores, test_scores = learning_curve(forest,X_train_std,y_train,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=1)

# plot learning curve

plt.style.use('default')

train_mean = np.mean(train_scores, axis=1)

train_std = np.std(train_scores, axis=1)

test_mean = np.mean(test_scores, axis=1)

test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='b',marker='o',markersize=5, label='training accuracy')

plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')

plt.plot(train_sizes, test_mean, color='g',marker='s',linestyle='--',markersize=5, label='validation accuracy')

plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std,alpha=0.15,color='g')
# plot validation curve

from sklearn.model_selection import validation_curve



param_range = [1,10,200,500,800,1000]

train_scores, test_scores = validation_curve(forest,X_train_std,y_train,param_range=param_range,cv=10, param_name='n_estimators')





train_mean = np.mean(train_scores,axis=1)

train_std = np.std(train_scores,axis=1)

test_mean = np.mean(test_scores,axis=1)

test_std = np.std(test_scores,axis=1)

plt.plot(param_range, train_mean, color='b',marker='o',markersize=5, label='training accuracy')

plt.fill_between(param_range,train_mean+train_std,train_mean-train_std,alpha=0.15,color='blue')

plt.plot(param_range, test_mean, color='g',marker='s',linestyle='--',markersize=5, label='validation accuracy')

plt.fill_between(param_range,test_mean+test_std,test_mean-test_std,alpha=0.15,color='g')
