#Importing important libriries

import pandas as pd, numpy as np

import matplotlib.pyplot as plt

import os #Provides functions for interacting with the operating system 

%matplotlib inline
#We check the current working directory

os.getcwd()
#We check the parent working directory name.

print(os.listdir("../input"))
merc_train = pd.read_csv("../input/mercedesbenz-greener-manufacturing/train.csv")

merc_test = pd.read_csv("../input/mercedesbenz-greener-manufacturing/test.csv")
#We check the shape of each Dataset

#Each Dataset comprises of 4209 attributes, 378 indexes for train dataset and 377 indexes for test dataset

merc_train.shape, merc_test.shape
#We check the nature of our columns by viewing the first five rows

merc_train.head()
#We check the data type. We have the pandas DataFrame

type(merc_train)
#Summary statistics. The summary statistics for the 8 columns could not be displayed since it is categorical.

#From X0 to X9 its look like we have the categorical indexes as they were noot included in the summary statistics.

merc_train.describe()
#The nature of our columns

#We have one float which is our target variable y, It seems like we have 8 categorical variables, and from X10 to X385

#We have integers.

merc_train.info()
merc_train.dtypes
#We observe the nature of our indexes by printing the numerical and categorical features size of the columns.

#From X0 to X8 are categorical variables. The target variable is continuous. The other indexes are binary nominal data type. 

#The numeric(binary) - presence/absence of the car feature.



numerical_features = merc_train.select_dtypes(include=[np.number]).columns

categorical_features = merc_test.select_dtypes(include=[np.object]).columns



print('Numerical feature size: {}'.format(len(numerical_features)))

print('Categorical feature size: {}'.format(len(categorical_features)))
#We see what are those columns after column 10

merc_train.columns[10:]
#We check the columns with unique values

np.unique(merc_train.columns[10:])
#We check if they are real binary

np.unique(merc_train[merc_train.columns[10:]])
merc_train.loc[:, (merc_train !=0).any(axis=0)]
(merc_train !=0).any(axis=0)
#Checking for null values in each column

#There is no null values in either training or testing datasets



merc_train.isna().any()
merc_test.isna().any()
#Lets get an overview and some statistics of a datasets especially on the variable y.

print(merc_train['y'].describe())
#Distribution plot. We draw distribution plot as sns. 

import seaborn as sns

plt.figure(figsize=(12,6))

plt.hist(merc_train['y'], bins=50, color='b')

plt.xlabel('testing time in secs')
plt.figure(figsize=(12,8))



sns.distplot(merc_train.y.values, bins=50, kde=True)

plt.xlabel('y value', fontsize=12)
#Simple plotting y visualize some outlier. One of the cars is taking 270s for testing(which is an outlier).

#The majority of the cars are taking around 75 to 150 seconds for testing.

plt.figure(figsize=(15,6))

plt.plot(merc_train['y'])
#pdf and cdf



counts, bin_edges = np.histogram(merc_train['y'], bins=10, density=True)

plt.xlabel('y')

pdf = counts/(sum(counts))

print("pdf=",pdf);

print("bin_edges=",bin_edges);

cdf = np.cumsum(pdf)

print("cdf=",cdf)

plt.plot(bin_edges[1:],pdf);

plt.plot(bin_edges[1:], cdf)
#boxplot it view some statistical values. The minimum value is around 73 seconds, 

#the median is 100s(most cars test around this time). The maximun is approximately 140s.

#we have one outlier at above 260s.

sns.boxplot(y='y', data=merc_train)

plt.figure(figsize=(15,9))

plt.show()
#Violin plot gives pdf(probabilty density plot) along with boxplot in it.

#We can observe that most of the cars are tested around 73s to 150s looking at the blue shade. 

#After that we can expect some extreme values and probably some outliers.

sns.violinplot(y='y', data=merc_train, size=8)

plt.show()
numerics = ['int16','int32','int64','float16','float32','float64']

objects = ['object']
merc_train_num = merc_train.select_dtypes(include=numerics)

merc_train_cat = merc_train.select_dtypes(include=objects)
print(merc_train_num.shape, merc_train_cat.shape)



print('*****************************************************************')

print(merc_train_cat.columns)

print('*****************************************************************')

print(merc_train_num.columns)
merc_test_num = merc_test.select_dtypes(include=numerics)

merc_test_cat = merc_test.select_dtypes(include=objects)
print(merc_test_num.shape, merc_test_cat.shape)



print('******************************************************************')

print(merc_test_cat.columns)

print('******************************************************************')

print(merc_test_num.columns)
for col_name in merc_train_cat.columns:

    print('The unique values in '+col_name+' are:', merc_train_cat[col_name].nunique())

    print(merc_train_cat[col_name].unique())

    print('*********************************************')
for col_name in merc_test_cat.columns:

    print('The unique values in '+col_name+' are:', merc_test_cat[col_name].nunique())

    print(merc_test_cat[col_name].unique())

    print('*********************************************')
#We look at the change of y fro each of this category index X0, X2, X3...

#We do this to analyze the outliers and drop them to make a match of our train and test datasets

cols=['X0','X1','X2','X3','X4','X5','X6','X8']

for col in cols:

    

    plt.figure(figsize=(16,6))

    

    sns.boxplot(x=col, y='y', data=merc_train)

    

    plt.xlabel(col, fontsize=10)

    plt.title('Distibution of y variable')

    plt.ylabel('y', fontsize=10)

    plt.xticks(fontsize=8)

    plt.yticks(fontsize=8) 
#As we observed the boxplot with categorical distribution of y. There might be some outliers because of small single dots above

#150.



plt.figure(figsize=(12,6))

sns.violinplot(merc_train['y'].values)
cols=['X0','X1','X2','X3','X4','X5','X6','X8']



for col in cols:

    plt.figure(figsize=(16,6))

    sns.violinplot(x=col, y='y', data=merc_train, height=15)

    plt.show()
from numpy import percentile
# calculate interquartile range

q25, q75 = percentile(merc_train.loc[:,'y'], 25), percentile(merc_train.loc[:,'y'], 75)

iqr = q75 - q25
print(q25,q75)
iqr
# calculate the outlier cutoff

cut_off = iqr * 1.5

lower, upper = q25 - cut_off, q75 + cut_off
print(lower,upper)
# identify outliers

outliers = [x for x in merc_train.loc[:,'y'] if x < lower or x > upper]
outliers
print('Identified outliers: %d' % len(outliers))
outliers_removed = [x for x in merc_train.loc[:,'y'] if x >= lower or x <= upper]
outliers_removed
#We firstly append the two datasets (train and test)



#With `ignore_index` set to True:



#>>> df.append(df2, ignore_index=True)

merc_train.append(merc_test, ignore_index=True)
#We assign the DataFrame as 'merc'

merc = merc_train.append(merc_test, ignore_index=True)
merc=pd.get_dummies(merc)
#Checking the indexes of the converted dataset

merc.index
train, test = merc[0:len(merc_train)], merc[len(merc_train):]
train.shape, test.shape
merc.head()
#Seperate the features and response column

X_train_1 = train.drop(['y','ID'], axis=1)

y_train_1 = train['y']



X_test_1 = test.drop(['y','ID'], axis=1)
from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X_train_1,y_train_1, test_size=0.30, random_state=101)
from sklearn.tree import DecisionTreeRegressor

dtree = DecisionTreeRegressor()
%%time

dtree.fit(X_train, y_train)
y_pred = dtree.predict(X_train)
y_pred
score = r2_score(y_train, y_pred)
error = mean_squared_error(y_train,y_pred)
error
score
y_pred = dtree.predict(X_test)
score = r2_score(y_test, y_pred)
score
#instatiate the Random Forest Regressor

from sklearn.ensemble import RandomForestRegressor 

rf_reg = RandomForestRegressor(n_estimators=50) 
X_train, X_test, y_train, y_test = train_test_split(X_train_1,y_train_1, test_size=0.25, random_state=4)
%%time

#fit the data

rf_reg.fit(X_train,y_train)
#from sklearn.metrics import r2_score, mean_squared_error





#predict (training samples)

y_pred = rf_reg.predict(X_train)



print('\nTraining score :')

print('Mean square error : %2.f'% mean_squared_error(y_train,y_pred))

print('R2 score: %2.f' %r2_score(y_train,y_pred))



#predict (testing samples)

y_pred = rf_reg.predict(X_test)



print('\nTesting score :')

print('Mean square error : %2.f'% mean_squared_error(y_test,y_pred))

print('R2 score: %2.f' %r2_score(y_test,y_pred))
# Use the forest's predict method on the test data

predictions = rf_reg.predict(X_test)
# Calculate the absolute errors

errors = abs(predictions - y_test)
# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
# Calculate mean absolute percentage error (MAPE)

mape = 100 * (errors / y_test)
# Calculate and display accuracy

accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
#From sklearn we load Gradient Boosting

from sklearn.ensemble import GradientBoostingRegressor
%%time

GB_regressor = GradientBoostingRegressor()

GB_regressor.fit(X_train,y_train)
y_predictions = GB_regressor.predict(X_train)
y_predictions
print('Score of Model:', r2_score(y_train,y_predictions))

print('Mean square error:', mean_squared_error(y_train, y_predictions))
y_predictions = GB_regressor.predict(X_test)
y_predictions
print('Score of Model:', r2_score(y_test,y_predictions))

print('Mean square error:', mean_squared_error(y_test,y_predictions))
from xgboost import XGBRegressor
%%time

xgb_regressor = XGBRegressor()

xgb_regressor.fit(X_train, y_train)
y_prediction2 = xgb_regressor.predict(X_train)
y_prediction2
print('Score of Model :', r2_score(y_train,y_prediction2))

print('Mean square error :', mean_squared_error(y_train,y_prediction2))
y_prediction3 = xgb_regressor.predict(X_test)
y_prediction3
print('Score of Model :', r2_score(y_test,y_prediction3))

print('Mean square error :', mean_squared_error(y_test,y_prediction3))
# Hyper Parametezation using RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# We evaluate the params using "mae"

from sklearn.metrics import mean_absolute_error
# Prepare the dict of parameters to use

xgb_params = { "max_depth" : [3,4,5,6,7,8,9],

                "learning_rate" : [0.05,0.10,0.15,0.20,0.25,0.30],

                "verbosity" : [0,1,3],

                "min_child_weights" : [1,3,5,7],

                "gamma" : [0.0,0.1,0.2,0.3,0.4],

                "colsample_bytree" : [0.3,0.4,0.5,0.7],

                "n_estimators" : [100]

               }
def timer(start_time = None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod ((datetime.now()-start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('n\ Time taken: %i hours %i modules and %s seconds.' %(thour, tmin, round(tsec,2)))
class Timer:

    def _init_(self):

        self.start = time.time()

        def restart(self):

            self.start = time.time()

            m,s = divmod(end_self.start, 60)

            h,m = divmod(m,60)

            time_str = "%02d:%02d" %(h,m,s)

            return time_str
#Instantiate the xgboost regressor model

regressor=XGBRegressor()
#Call the RandomizdSearchCV



random_search = RandomizedSearchCV(regressor,param_distributions=xgb_params,n_iter=5, scoring='neg_mean_squared_error', 

                                   n_jobs=-1,cv=5,verbose=3)
# Fitting the model into a timer



from datetime import datetime

start_time = timer(None)

random_search.fit(X_train,y_train)

timer(start_time)
# Parameter setting that gave the best results on the hold out data.

random_search.best_estimator_
# Gives the parameter setting for the best model, that gives the highest mean score.

random_search.best_params_
# Alternate code for the above task.

random_search.cv_results_['params'][random_search.best_index_]
regressor=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.5, gamma=0.3,

             importance_type='gain', learning_rate=0.05, max_delta_step=0,

             max_depth=4, min_child_weight=1, min_child_weights=5, missing=None,

             n_estimators=100, n_jobs=1, nthread=None, objective='reg:linear',

             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,

             seed=None, silent=None, subsample=1, verbosity=1)
from sklearn.model_selection import cross_val_score
score = cross_val_score(regressor, X_train, y_train, cv=10)
score
score.mean()