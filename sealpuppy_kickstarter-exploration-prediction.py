import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.model_selection import GridSearchCV
import squarify
import time
import os
ksdf = pd.read_csv('../input/ks-projects-201801.csv', sep=',', engine='python')
ksdf.head()
ksdf.describe()
ksdf.isnull().sum()
# Check missing values in the column "name"
ksdf[pd.isnull(ksdf['name'])].index
ksdf[ksdf.index == 166851]
ksdf[ksdf.index == 307234]
# Delete usdf pledged column
ksdf.drop(['usd pledged'], axis = 1, inplace = True)
# Add pledged_ratio column
ksdf['pledged_ratio'] = ksdf['usd_pledged_real']/ ksdf['usd_goal_real']
def year_cut(string):
    return string[0:4]

def month_cut(string):
    return string[5:7]

ksdf['year'] = ksdf['launched'].apply(year_cut)
ksdf['month'] = ksdf['launched'].apply(month_cut)

ksdf['year'] = ksdf['year'].astype(int)
ksdf['month'] = ksdf['month'].astype(int)
ksdf['time'] = (ksdf['year'].values - 2009)*12 + (ksdf['month']).astype(int)
print (ksdf.columns)
ksdf['year'].value_counts()
ksdf_year = {}
for year in range(2009, 2019):
    ksdf_year[year] = ksdf[ksdf['year'] == year]['year'].count()
ksdf_year = pd.Series(ksdf_year)
ksdf_year = pd.DataFrame(ksdf_year)
ksdf_year = ksdf_year.rename(columns = {0: "counts"})
ksdf_year
ksdf['state'].value_counts()
squarify.plot(sizes=[197719,133956, (38779 + 3562 + 2799 + 1846)], 
              label=["Failed (52.22%)", "Successful (35.38%)", "Others (10.24%)",], color=["blue","red","green"], alpha=.4 )
plt.title('State', fontsize = 20)
plt.axis('off')
plt.show()
success_timely = []

for year in range(2009, 2019):
    success = len (ksdf[(ksdf['year'] == year) & (ksdf['state'] == 'successful')]['state'])
    overall = len (ksdf[ksdf['year'] == year]['year'])
    ratio = success/ overall
    success_timely.append(ratio)
    print ("Year = ",year, ratio * 100, '%')
ksdf[ksdf['year'] == 2018]['state'].value_counts()
ksdf_year['success_ratio'] = success_timely
ksdf_year.head
ksdf[ksdf['year'] == 2017]['backers'].count()
backers_year = {}
for year in range(2009, 2019):
    backers_count = ksdf[ksdf['year'] == year]['backers'].sum()
    backers_year[year] = backers_count

ksdf_year['backers'] = pd.Series(backers_year)
ksdf_year
# Cross-year proposed projects
sns.set_style("whitegrid")
sns.barplot(ksdf_year['counts'].index, y= ksdf_year['counts'] ,
            palette="Blues_d", saturation = 0.5)
sns.despine(right = True, top = True)
# Cross-year success ratio
sns.set_style("whitegrid")
sns.barplot(ksdf_year['success_ratio'].index, y= ksdf_year['success_ratio'], data = ksdf_year,
            palette="Blues_d", saturation = 0.5)
sns.despine(right = True, top = True)
sns.set_style("whitegrid")
sns.barplot(ksdf_year['backers'].index, y= ksdf_year['backers'] ,
            palette="Blues_d", saturation = 0.5)
sns.despine(right = True, top = True)
sum_pledged = ksdf['usd_pledged_real'].sum()
print (sum_pledged)
ksdf['usd_pledged_real'].describe()
# Ratio of successful/ failed / others
success_pledged = ksdf[ksdf['state'] == "successful"]['usd_pledged_real'].sum()
fail_pledged = ksdf[ksdf['state'] == 'failed']['usd_pledged_real'].sum()
others_pledged = (ksdf[ksdf['state'] == 'canceled']['usd_pledged_real'].sum() +
                  ksdf[ksdf['state'] == 'undefined']['usd_pledged_real'].sum() +
                  ksdf[ksdf['state'] == 'live']['usd_pledged_real'].sum() +
                  ksdf[ksdf['state'] == 'suspended']['usd_pledged_real'].sum())

print (success_pledged, success_pledged/ sum_pledged * 100, '%')
print (fail_pledged, fail_pledged/ sum_pledged * 100, '%')
print (others_pledged, others_pledged/ sum_pledged * 100, '%')
squarify.plot(sizes=[3036889045.99, 261108466.05, 132263736.79], 
              label=["Successful (88.53%)", "Failed (7.61%)", "Others (3.86%)",], color=["red","blue", "green"], alpha=.4 )
plt.title('Pledged Amount', fontsize = 20)
plt.axis('off')
plt.show()
success_projects = ksdf[ksdf['state'] == 'successful']['state'].count()
fail_projects  = ksdf[ksdf['state'] == 'failed']['state'].count()
others_projects  = (
    ksdf[ksdf['state'] == 'canceled']['state'].count() +
    ksdf[ksdf['state'] == 'live']['state'].count() +
    ksdf[ksdf['state'] == 'undefined']['state'].count() +
    ksdf[ksdf['state'] == 'suspended']['state'].count())

print ("Average pledged amount per successful project = ",success_pledged/success_projects)
print ("Average pledged amount per failed project = ",fail_pledged/ fail_projects)
print ("Average pledged amount per other project = ",others_pledged/ others_projects)
sns.set_style("whitegrid")
sns.barplot(["Successful", "Failed", "Others"],
            y= [22670.7952312, 1320.60381678, 2814.96055825],
            palette = "Set1",
            saturation = 0.5)
sns.despine(right = True, top = True)
ksdf['country'].unique()
ksdf['country'].value_counts()
sns.countplot(ksdf['country'], palette = 'Set1', order = ksdf['country'].value_counts().index)
sns.despine(bottom = True, left = True)
us = ksdf[ksdf['country'] == "US"]['country'].count()
print (us/len(ksdf['country']) * 100, "%")
pledged_sum = {}
for category in list(set(ksdf['main_category'])):
    amount = ksdf[ksdf['main_category'] == category]['usd_pledged_real'].sum()
    pledged_sum[category] = amount

# Create dataframe
cate_df = pd.Series(pledged_sum)
cate_df = pd.DataFrame(cate_df)
cate_df = cate_df.rename(columns = {0:"pledged_sum"})

cate_df.head()
cate_count = {}
for category in list(set(ksdf['main_category'])):
    count = ksdf[ksdf['main_category'] == category]['main_category'].count()
    cate_count[category] = count
    
cate_df['count'] = pd.Series(cate_count)

cate_df.head()
cate_df['average_amount'] = cate_df['pledged_sum']/ cate_df['count']
cate_df.head()
success = {}
for category in list(set(ksdf['main_category'])):
    success_count = len(ksdf[(ksdf['main_category'] == category) & 
         (ksdf['state'] == "successful")])
    success[category] = success_count

cate_df["success_count"] = pd.Series(success)
cate_df.head()
cate_df["success_rate"] = cate_df['success_count']/ cate_df['count']
cate_df.head()
# pledged_sum plot
cate_df = cate_df.sort_values('pledged_sum',  ascending = False)
plt.subplots(figsize = (20,5))
sns.set_style("whitegrid")
sns.barplot(cate_df['pledged_sum'].index, y= cate_df['pledged_sum'] ,
            palette="Set1",saturation = 0.5)
sns.despine(right = True, top = True)
# avarage amount plot
cate_df = cate_df.sort_values('average_amount',  ascending = False)
plt.subplots(figsize = (20,5))
sns.set_style("whitegrid")
sns.barplot(cate_df['average_amount'].index, y= cate_df['average_amount'] ,
            palette="Set1",saturation = 0.5)
sns.despine(right = True, top = True)
# count plot
cate_df = cate_df.sort_values('count',  ascending = False)
plt.subplots(figsize = (20,5))
sns.set_style("whitegrid")
sns.barplot(cate_df['count'].index, y= cate_df['count'] ,
            palette="Set1",saturation = 0.5)
sns.despine(right = True, top = True)
# success rate plot
cate_df = cate_df.sort_values('success_rate',  ascending = False)
plt.subplots(figsize = (20,5))
sns.set_style("whitegrid")
sns.barplot(cate_df['success_rate'].index, y= cate_df['success_rate'] ,
            palette="Set1",saturation = 0.5)
sns.despine(right = True, top = True)
back_cate = {}

for category in set(ksdf['main_category']):
    backers = ksdf[ksdf['main_category'] == category]['backers'].sum()
    back_cate[category] = backers

backers = pd.Series(back_cate)
cate_df['backers'] = backers
cate_df = cate_df.sort_values('backers',  ascending = False)
plt.subplots(figsize = (20,5))
sns.set_style("whitegrid")
sns.barplot(cate_df['backers'].index, y= cate_df['backers'] ,
            palette="Set1",saturation = 0.5)
sns.despine(right = True, top = True)
ksdf['backers'].quantile(list(np.arange(0,1,0.01))).plot(grid = 0, color = '#055968')
sns.set_style("whitegrid")
sns.kdeplot(ksdf['backers'])
sns.despine(right = True, top = True)
# Select not zero-pledged projects
non_zero = ksdf[ksdf['usd_pledged_real'] != 0]
print (non_zero.shape)
# Define X and Y
X = ksdf[ksdf['usd_pledged_real'] != 0]['backers'].values
Y = ksdf[ksdf['usd_pledged_real'] != 0]['usd_pledged_real'].values

print (X.shape)
print (Y.shape)
X = X.reshape(326134,1)
Y = Y.reshape(326134,1)
# Model fitting and visualization
regr = linear_model.LinearRegression()
regr.fit(np.log(X+1), np.log(Y+1))

plt.scatter(np.log(X+1), np.log(Y+1))
plt.plot(np.log(X+1), regr.predict(np.log(X+1)), color='red', linewidth=3)
plt.show()
# Results: error and parameters

Y_pred = regr.predict(np.log(X+1))
Y_true = np.log(Y+1)

print ("error = ", sklearn.metrics.mean_squared_error(Y_true, Y_pred))
print ("coefficient = ", regr.coef_)
print ("intercept = ", regr.intercept_)
print (ksdf['state'].value_counts())
print ('')
print ("ksdf.shape = ", ksdf.shape)
def state_change(cell_value):
    if cell_value == 'successful':
        return 1
    
    elif cell_value == 'failed':
        return 0
    
    else:
        return 'del'
ksdf['state'] = ksdf['state'].apply(state_change)
print (ksdf[ksdf['state'] == 1].shape)
print (ksdf[ksdf['state'] == 0].shape)
print (ksdf[ksdf['state'] == 'del'].shape)
print (ksdf[ksdf['state'] == 1].shape[0] + ksdf[ksdf['state'] == 0].shape[0])
ksdf_rf = ksdf.drop(ksdf[ksdf['state'] == 'del'].index)
print (ksdf_rf.shape)
ksdf_rf = pd.concat([
                  ksdf_rf['main_category'],
                  ksdf_rf['time'],
                  ksdf_rf['state']], axis = 1
                 )

print (ksdf_rf.shape)
train, test = sklearn.model_selection.train_test_split(ksdf_rf, test_size = 0.3, random_state = 42)

print ("Train shape = ", train.shape, ",", len(train)/ len(ksdf_rf) * 100, "%")
print ("Test shape = ", test.shape, ",", len(test)/ len(ksdf_rf) * 100, "%")
X_train = pd.concat(
    [
     pd.get_dummies(train['main_category'], prefix = 'main_category'),
     train["time"]
    ], axis=1)

Y_train = train['state']
X_test = pd.concat(
    [
     pd.get_dummies(test['main_category'], prefix = 'main_category'),
     test["time"]
    ], axis=1)

Y_test = test['state']
X_train = X_train.astype(int)
Y_train = Y_train.astype(int)
X_test = X_test.astype(int)
Y_test = Y_test.astype(int)

print (X_train.shape)
print (Y_train.shape)
print (X_test.shape)
print (Y_test.shape)
for_record = {
    'baseline':{},
    'best_random1':{},
    'best_random2':{},
    'best_random3':{},
    'grid1':{},
    'grid2':{}
}
start = time.time()
rf = RandomForestClassifier(random_state = 42)
rf.fit(X_train, Y_train)
end = time.time()
sec = end - start
Y_pred = rf.predict(X_train)
for_record['baseline']['params'] = rf.get_params()
for_record['baseline']['time'] = sec
for_record['baseline']["train_score"] = rf.score(X_train, Y_train)
for_record['baseline']['f1'] = f1_score(Y_train, Y_pred, average = 'weighted')
for_record['baseline']['test_score'] = rf.score(X_test, Y_test)
# Number of trees in random forest
n_estimators = [int(x) for x in range(2, 100, 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in range(2,50,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
param_rand1 = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random1 = RandomizedSearchCV(estimator = rf, param_distributions = param_rand1, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random1.fit(X_train, Y_train)

# Use best random parameters to train a new model
start = time.time()
rf_rand1 = RandomForestClassifier(bootstrap = rf_random1.best_params_['bootstrap'],
                                 max_depth = rf_random1.best_params_['max_depth'],
                                 max_features = rf_random1.best_params_['max_features'],
                                 min_samples_leaf = rf_random1.best_params_['min_samples_leaf'],
                                 min_samples_split = rf_random1.best_params_['min_samples_split'],
                                 n_estimators = rf_random1.best_params_['n_estimators'],
                                 random_state = 42, n_jobs = -1)

rf_rand1.fit(X_train, Y_train)
end = time.time()
rand1_time = end - start
Y_pred = rf_rand1.predict(X_train)

for_record['best_random1']['params'] = rf_rand1.get_params()
for_record['best_random1']['time'] = rand1_time
for_record['best_random1']["train_score"] = rf_rand1.score(X_train, Y_train)
for_record['best_random1']['f1'] = f1_score(Y_train, Y_pred, average = 'weighted')
for_record['best_random1']['test_score'] = rf_rand1.score(X_test, Y_test)
# # Number of trees in random forest
# n_estimators = [int(x) for x in range(55, 65, 2)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in range(2, 20, 2)]
# # Minimum number of samples required to split a node
# min_samples_split = [2,5,10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1,2,4]
# # Method of selecting samples for training each tree
# bootstrap = [True,False]
# # Create the random grid
# param_grid1 = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}

# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_grid1 = GridSearchCV(estimator = rf, param_grid = param_grid1, cv = 3, verbose=2, n_jobs = -1)
# # Fit the random search model
# rf_grid1.fit(X_train, Y_train)

# # Use best random parameters to train a new model
# start = time.time()
# rf_grid1 = RandomForestClassifier(bootstrap = rf_grid1.best_params_['bootstrap'],
#                                  max_depth = rf_grid1.best_params_['max_depth'],
#                                  max_features = rf_grid1.best_params_['max_features'],
#                                  min_samples_leaf = rf_grid1.best_params_['min_samples_leaf'],
#                                  min_samples_split = rf_grid1.best_params_['min_samples_split'],
#                                  n_estimators = rf_grid1.best_params_['n_estimators'],
#                                  random_state = 42, n_jobs = -1)

# rf_grid1.fit(X_train, Y_train)
# end = time.time()
# grid1_time = end - start
# Y_pred = rf_grid1.predict(X_train)

# for_record['grid1']['params'] = rf_grid1.get_params()
# for_record['grid1']['time'] = grid1_time
# for_record['grid1']["train_score"] = rf_grid1.score(X_train, Y_train)
# for_record['grid1']['f1'] = f1_score(Y_train, Y_pred, average = 'weighted')
# for_record['grid1']['test_score'] = rf_grid1.score(X_test, Y_test)
# Number of trees in random forest
n_estimators = [int(x) for x in range(100, 200, 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in range(50,100,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
param_random2 = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random2 = RandomizedSearchCV(estimator = rf, param_distributions = param_random2, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random2.fit(X_train, Y_train)

# Use best random parameters to train a new model
start = time.time()
rf_rand2 = RandomForestClassifier(bootstrap = rf_random2.best_params_['bootstrap'],
                                 max_depth = rf_random2.best_params_['max_depth'],
                                 max_features = rf_random2.best_params_['max_features'],
                                 min_samples_leaf = rf_random2.best_params_['min_samples_leaf'],
                                 min_samples_split = rf_random2.best_params_['min_samples_split'],
                                 n_estimators = rf_random2.best_params_['n_estimators'],
                                 random_state = 42, n_jobs = -1)

rf_rand2.fit(X_train, Y_train)
end = time.time()
rand2_time = end - start
Y_pred = rf_rand2.predict(X_train)
for_record['best_random2']['params'] = rf_rand2.get_params()
for_record['best_random2']['time'] = rand2_time
for_record['best_random2']["train_score"] = rf_rand2.score(X_train, Y_train)
for_record['best_random2']['f1'] = f1_score(Y_train, Y_pred, average = 'weighted')
for_record['best_random2']['test_score'] = rf_rand2.score(X_test, Y_test)
# # Number of trees in random forest
# n_estimators = [int(x) for x in range(100, 150, 2)]
# # Number of features to consider at every split
# max_features = 'auto'
# # Maximum number of levels in tree
# max_depth = [int(x) for x in range(2, 20, 2)]
# # Minimum number of samples required to split a node
# min_samples_split = 5
# # Minimum number of samples required at each leaf node
# min_samples_leaf = 2
# # Method of selecting samples for training each tree
# bootstrap = False
# # Create the random grid
# param_grid1 = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# Number of trees in random forest
n_estimators = [int(x) for x in range(201, 300, 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in range(100,150,10)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
param_random3 = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random3 = RandomizedSearchCV(estimator = rf, param_distributions = param_random3, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random3.fit(X_train, Y_train)

# Use best random parameters to train a new model
start = time.time()
rf_rand3 = RandomForestClassifier(bootstrap = rf_random3.best_params_['bootstrap'],
                                 max_depth = rf_random3.best_params_['max_depth'],
                                 max_features = rf_random3.best_params_['max_features'],
                                 min_samples_leaf = rf_random3.best_params_['min_samples_leaf'],
                                 min_samples_split = rf_random3.best_params_['min_samples_split'],
                                 n_estimators = rf_random3.best_params_['n_estimators'],
                                 random_state = 42, n_jobs = -1)

rf_rand3.fit(X_train, Y_train)
end = time.time()
rand3_time = end - start
Y_pred = rf_rand3.predict(X_train)
for_record['best_random3']['params'] = rf_rand3.get_params()
for_record['best_random3']['time'] = rand3_time
for_record['best_random3']["train_score"] = rf_rand3.score(X_train, Y_train)
for_record['best_random3']['f1'] = f1_score(Y_train, Y_pred, average = 'weighted')
for_record['best_random3']['test_score'] = rf_rand3.score(X_test, Y_test)
print (for_record['best_random1']['test_score'])
print (for_record['best_random2']['test_score'])
print (for_record['best_random3']['test_score'])
