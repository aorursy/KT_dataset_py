import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import statsmodels.api as sm

from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn import linear_model, ensemble
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tools.eval_measures import mse, rmse
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import time
%matplotlib inline

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

# suppress warnings about "value is trying to be set on a copy of a slice from a DataFrame"
pd.options.mode.chained_assignment = None  # default='warn'

# function to visualize the predictions compared to the test set
# also print out several fit metrics
# only using RMS error, but including other metrics to be on the safe side

def fitter_metrics(y_actual, y_pred):
    plt.scatter(y_actual, y_pred)
    plt.plot(y_actual, y_actual, color="red")
    plt.xlabel("true values")
    plt.ylabel("predicted values")
    plt.title("Life expectancy: true and predicted values")
    plt.show()

    print("Mean absolute error of the prediction is: {}".format(mean_absolute_error(y_actual, y_pred)))
    print("Mean squared error of the prediction is: {}".format(mse(y_actual, y_pred)))
    print("Root mean squared error of the prediction is: {}".format(rmse(y_actual, y_pred)))
    print("Mean absolute percentage error of the prediction is: {}".format(np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100))

# using a random number seed for the train-test split
# set it here, in case we want to re-run with a different seed number
rand_seed = 173

# *** read in file from Kaggle ***
who_df = pd.read_csv('/kaggle/input/who-national-life-expectancy/who_life_exp.csv')

# *** otherwise read from local version of file ***
#who_df = pd.read_csv('who_life_exp.csv')

print('Data set has {} countries for the years {} to {}\n'.format(who_df['country_code'].nunique(),
                                                                  who_df['year'].min(),who_df['year'].max()))

print(who_df.info())
region_names = who_df['region'].unique()
fig, axs = plt.subplots(2, 3)
fig.set_size_inches(14.0, 8.0)
for ireg, region in enumerate(region_names):
    ix = ireg//3
    iy = ireg%3
    axs[ix, iy].set_title(region)
    temp_df = who_df[who_df['region'] == region]
    for country in temp_df['country'].unique():
        axs[ix, iy].plot(temp_df[temp_df['country']==country].year, temp_df[temp_df['country']==country].life_expect)
    axs[ix, iy].set_xlabel("year")
    axs[ix, iy].set_ylabel("life expectancy")
plt.tight_layout()
plt.show()
# This identified which country had the 2010 dip in the Americas plot
#who_df[(who_df['region'] == "Americas") & (who_df['life_expect'] < 65)].head(30)

# print(who_df[who_df['country_code'] == 'HTI'].head(20))

who_df = who_df[~((who_df['country_code'] == 'HTI') & (who_df['year'] == 2010))]
# We haven't done any data cleaning, so make sure to remove null values
X = who_df[['life_expect', 'infant_mort', 'age1-4mort', 'adult_mortality']].copy()
X = X.dropna(axis=0)

Y = X['life_expect']
X_m = X[['adult_mortality', 'infant_mort', 'age1-4mort']]
X_m['infant2'] = X_m['infant_mort'].pow(2)
X_m['youth2'] = X_m['age1-4mort'].pow(2)
X_m['adult2'] = X_m['adult_mortality'].pow(2)
X_m = sm.add_constant(X_m)
X_train, X_test, y_train, y_test = train_test_split(X_m, Y, test_size = 0.2, random_state = rand_seed)

# We fit an OLS model using statsmodels
results_ols = sm.OLS(y_train, X_train).fit()

# We print the summary results.
print(results_ols.summary())
# We are making predictions here
y_ols = results_ols.predict(X_test)

fitter_metrics(y_test, y_ols)
plt.scatter(who_df["life_expect"], who_df["une_life"])
plt.plot(who_df["life_expect"], who_df["life_expect"], color="red")
plt.xlabel('GHO life expect')
plt.ylabel('UNESCO life expect')
plt.show()

del who_df['une_life']
del who_df['adult_mortality']
del who_df['infant_mort']
del who_df['age1-4mort']
del who_df['une_infant']
del who_df['life_exp60']
clean_df = who_df.copy()

#print("NaN for polio")
#print(clean_df[clean_df['polio'].isnull()].head(50))

# missing vaccine information for 2000 and 2001
indices = clean_df[(clean_df['country_code'] == 'TLS') & (clean_df['year'] < 2002)].index
clean_df.drop(indices , inplace=True)

#print("NaN for alcohol")
#print(clean_df[clean_df['alcohol'].isnull()].head(50))

for country in clean_df['country_code'].unique():
    num_na = clean_df[clean_df['country_code'] == country]['alcohol'].isnull()
    if (num_na.any()):
        print("  for feature \"alcohol\" ",country," is missing data for", num_na.sum(),"years")

indices = clean_df[((clean_df['country_code'] == 'SSD') | (clean_df['country_code'] == 'SDN'))].index
clean_df.drop(indices , inplace=True)

indices = clean_df[((clean_df['country_code'] == 'SRB') | (clean_df['country_code'] == 'MNE')) &
                 (clean_df['year'] < 2006)].index
clean_df.drop(indices , inplace=True)
#print("NaN for gni ppp")
#print(clean_df[clean_df['une_gni'].isnull()].head(50))

indices = clean_df[((clean_df['country_code'] == 'SOM') | (clean_df['country_code'] == 'SYR') | (clean_df['country_code'] == 'PRK'))].index
clean_df.drop(indices , inplace=True)

plt.scatter(clean_df["gni_capita"], clean_df["une_gni"])
plt.plot(clean_df["gni_capita"], clean_df["gni_capita"], color="red")
plt.xlabel('GHO GNI per capita')
plt.ylabel('UNESCO GNI per capita')
plt.show()

clean_df['gni_scale'] = (clean_df['une_gni'] / clean_df['gni_capita'])
plt.scatter(clean_df["gni_capita"], clean_df["gni_scale"])
plt.plot([0.0, 122000.0], [1.0, 1.0], color="red")
plt.xlabel('GHO GNI per capita')
plt.ylabel('UNESCO GNI / GHO GNI')
plt.show()

print('Mean UNESCO/ GHO value for GNI > 40K : ', clean_df[clean_df['gni_capita'] > 40000]['gni_scale'].mean())
print('Mean UNESCO/ GHO value for GNI > 80K : ', clean_df[clean_df['gni_capita'] > 80000]['gni_scale'].mean())

# Not using a scale factor
# use GHO value when the UNESCO value is missing; possible that both are null
clean_df['une_gni'] = np.where(clean_df['une_gni'].notnull(), clean_df['une_gni'], clean_df['gni_capita'])
del clean_df['gni_scale']
del clean_df['gni_capita']
# to interpolate the missing values
clean_df = clean_df.groupby('country').apply(lambda group: group.interpolate(method='linear', limit_direction='both'))

# Montenegro is missing info for gghe-d and che_gdp; Albania is missing che_gdp
clean_df = clean_df[~((clean_df['country_code'] == 'ALB') | (clean_df['country_code'] == 'MNE'))]

country_list = clean_df['country_code'].unique()
column_list = list(clean_df.columns)

gone_all = dict()
gone_some = dict()

for col in column_list:
    for country in country_list:
        num_na = clean_df[clean_df['country_code'] == country][col].isnull()
        if (num_na.all()):
            gone_all[col] = gone_all.get(col, 0) + 1
        if (num_na.any()):
            gone_some[col] = gone_some.get(col, 0) + 1
    if col in gone_some:
        print("Feature",col,"has",gone_all[col],"countries with no data, ",gone_some[col],"with some missing data.")
#print(clean_df.corr())

plt.figure(figsize=(20,10))
sn.heatmap(clean_df.corr(), annot=True)
plt.show()
from sklearn.decomposition import PCA
pca = PCA(n_components=1)

# Standardizing the features
X = clean_df[['measles', 'polio', 'diphtheria']]
X = StandardScaler().fit_transform(X)

# want the principal component vector to be positively correlated with increasing vaccination rates
# for this particular fit by the code, that requires multiplying by -1
principalComponents = pca.fit_transform(X)
clean_df['vaccination'] = -principalComponents

print('Variance accounted for by pca:', pca.explained_variance_ratio_)

plt.figure(figsize=(12,4))
plt.subplot(1, 3, 1)
plt.scatter(clean_df['vaccination'], clean_df['measles'])
plt.xlabel('vac pca')
plt.ylabel('measles')

plt.subplot(1, 3, 2)
plt.scatter(clean_df['vaccination'], clean_df['polio'])
plt.xlabel('vac pca')
plt.ylabel('polio')

plt.subplot(1, 3, 3)
plt.scatter(clean_df['vaccination'], clean_df['diphtheria'])
plt.xlabel('vac pca')
plt.ylabel('diphtheria')

plt.tight_layout()
plt.show()

del clean_df['measles']
del clean_df['polio']
del clean_df['diphtheria']
target_feature = "life_expect"

num_features = list(clean_df.columns)
num_features.remove(target_feature)
num_features.remove('country')
num_features.remove('country_code')
num_features.remove('region')
num_features.remove('year')

print("Target feature:",target_feature)
print("Numeric  features:",num_features)

print("Plotting target feature:",target_feature)
plt.hist(clean_df[target_feature])
plt.show()

for feat in num_features:
    print("Plotting feature:",feat)
    plt.figure(figsize=(12,4))
    plt.subplot(1, 2, 1)
    plt.hist(clean_df[feat])
    plt.xlabel(feat)

    plt.subplot(1, 2, 2)
    plt.scatter(clean_df[feat], clean_df[target_feature])
    plt.xlabel(feat)
    plt.ylabel("life expectancy")
    plt.show()
no_log_df = clean_df.copy()

feature_log = ['age5-19obesity', 'doctors', 'une_gni']

for feat in feature_log:
    clean_df[feat] = np.log1p(clean_df[feat])

    print("Plotting feature:",feat)
    plt.figure(figsize=(12,4))
    plt.subplot(1, 2, 1)
    plt.hist(clean_df[feat])
    plt.xlabel(feat)

    plt.subplot(1, 2, 2)
    plt.scatter(clean_df[feat], clean_df[target_feature])
    plt.xlabel(feat)
    plt.ylabel("life expectancy")
    plt.show()
# make a copy before dropping any more features and countries
# this will save some work later on, when I want to try a
# different selection of features

df_before_remove = clean_df.copy()

# drop hepatitis, hospitals, hiv, poverty, spend edu, literacy, school
remove_list = ['hepatitis', 'hospitals', 'une_hiv', 'une_poverty', 'une_edu_spend', 'une_literacy', 'une_school']
for col_name in remove_list:
    del clean_df[col_name]
    del no_log_df[col_name]
    num_features.remove(col_name)

print('Remaining features:',len(num_features), num_features)
print("\n",clean_df['country_code'].nunique(),"countries to be analyzed")

# drop remaining NaN rows; should be zero, but running it just in case I missed a stray value somewhere
clean_df = clean_df.dropna(axis=0)
no_log_df = no_log_df.dropna(axis=0)
# create a table of the model metrics, to allow easier comparison
model_perform = pd.DataFrame(columns = ['model', 'rms_error', 'time'])

X = clean_df[num_features]
y = list(clean_df[target_feature])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = rand_seed)

# standardize the features before fitting
# not all models require this, but it will make life easier to do it for all of them
sc = StandardScaler()
X_train_sc = pd.DataFrame(sc.fit_transform(X_train), columns=num_features)
X_test_sc = pd.DataFrame(sc.transform(X_test), columns=num_features)
# fit using statsmodels OLS
X_train_sc2 = sm.add_constant(X_train_sc)
results_ols = sm.OLS(y_train, X_train_sc2).fit()

# We print the summary results.
print(results_ols.summary())

# We are making predictions here
X_test_sc2 = sm.add_constant(X_test_sc)
y_ols = results_ols.predict(X_test_sc2)

fitter_metrics(y_test, y_ols)
model = linear_model.LinearRegression()
score = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='neg_root_mean_squared_error')
print('Array of cross_val_score results:',score)
print("Unweighted Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() ))

elapse_time = time.time()
y_pred = cross_val_predict(model, X_train_sc, y_train, cv=5)
elapse_time = time.time() - elapse_time
model_perform = model_perform.append({'model': 'OLS (before log transform)', 'rms_error': -score.mean(), 'time': elapse_time}, ignore_index=True)

fitter_metrics(y_train, y_pred)
y2 = no_log_df[target_feature]
X2 = no_log_df[num_features]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size = 0.2, random_state = 173)

model = make_pipeline(StandardScaler(), linear_model.LinearRegression())
score = cross_val_score(model, X2_train, y2_train, cv=5, scoring='neg_root_mean_squared_error')
print('Array of cross_val_score results:',score)
print("Unweighted Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() ))

elapse_time = time.time()
y_pred = cross_val_predict(model, X2_train, y2_train, cv=5)
elapse_time = time.time() - elapse_time
model_perform = model_perform.append({'model': 'OLS (after log transform)', 'rms_error': -score.mean(), 'time': elapse_time}, ignore_index=True)

fitter_metrics(y2_train, y_pred)
model = linear_model.ElasticNet()
score = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='neg_root_mean_squared_error')
print('Array of cross_val_score results:',score)
print("Unweighted Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() ))

elapse_time = time.time()
y_pred = cross_val_predict(model, X_train_sc, y_train, cv=5)
elapse_time = time.time() - elapse_time
model_perform = model_perform.append({'model': 'ElasticNet', 'rms_error': -score.mean(), 'time': elapse_time}, ignore_index=True)

fitter_metrics(y_train, y_pred)
# Failed to converge with default of max_iter=100
model = linear_model.HuberRegressor(max_iter=1200)
score = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='neg_root_mean_squared_error')
print('Array of cross_val_score results:',score)
print("Unweighted Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() ))

elapse_time = time.time()
y_pred = cross_val_predict(model, X_train_sc, y_train, cv=5)
elapse_time = time.time() - elapse_time
model_perform = model_perform.append({'model': 'Huber Linear', 'rms_error': -score.mean(), 'time': elapse_time}, ignore_index=True)

fitter_metrics(y_train, y_pred)
# default is n_neighbors=5
model = KNeighborsRegressor()
score = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='neg_root_mean_squared_error')
print('Array of cross_val_score results:',score)
print("Unweighted Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() ))

elapse_time = time.time()
y_pred = cross_val_predict(model, X_train_sc, y_train, cv=5)
elapse_time = time.time() - elapse_time
model_perform = model_perform.append({'model': 'KNeighbors', 'rms_error': -score.mean(), 'time': elapse_time}, ignore_index=True)

fitter_metrics(y_train, y_pred)
# 
model = SVR()
score = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='neg_root_mean_squared_error')

print('Array of cross_val_score results:',score)
print("Unweighted Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() ))

elapse_time = time.time()
y_pred = cross_val_predict(model, X_train_sc, y_train, cv=5)
elapse_time = time.time() - elapse_time
model_perform = model_perform.append({'model': 'Support Vector', 'rms_error': -score.mean(), 'time': elapse_time}, ignore_index=True)

fitter_metrics(y_train, y_pred)
# The default value of n_estimators changed from 10 to 100 in version 0.22

model = ensemble.RandomForestRegressor()
score = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='neg_root_mean_squared_error')
print('Array of cross_val_score results:',score)
print("Unweighted Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() ))

elapse_time = time.time()
y_pred = cross_val_predict(model, X_train_sc, y_train, cv=5)
elapse_time = time.time() - elapse_time
model_perform = model_perform.append({'model': 'Random Forest', 'rms_error': -score.mean(), 'time': elapse_time}, ignore_index=True)

fitter_metrics(y_train, y_pred)
# The default value of n_estimators changed from 10 to 100 in version 0.22

model = ensemble.RandomForestRegressor(max_depth=4)
score = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='neg_root_mean_squared_error')
print('Array of cross_val_score results:',score)
print("Unweighted Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() ))

elapse_time = time.time()
y_pred = cross_val_predict(model, X_train_sc, y_train, cv=5)
elapse_time = time.time() - elapse_time
model_perform = model_perform.append({'model': 'Random Forest (depth 4)', 'rms_error': -score.mean(), 'time': elapse_time}, ignore_index=True)

fitter_metrics(y_train, y_pred)
# default n_estimators=100, max_depth=3

model = ensemble.GradientBoostingRegressor()
score = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='neg_root_mean_squared_error')
print('Array of cross_val_score results:',score)
print("Unweighted Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() ))

elapse_time = time.time()
y_pred = cross_val_predict(model, X_train_sc, y_train, cv=5)
elapse_time = time.time() - elapse_time
model_perform = model_perform.append({'model': 'Gradient Boost', 'rms_error': -score.mean(), 'time': elapse_time}, ignore_index=True)

fitter_metrics(y_train, y_pred)
print(model_perform.round(decimals=3).head(20))
params = {'n_estimators': [100, 200, 300, 500],
          'max_depth': [3, 4, 5, 6, 7]}

model = ensemble.GradientBoostingRegressor()
clf = GridSearchCV(model, params)
clf.fit(X_train_sc, y_train)

optimize_df = pd.DataFrame.from_dict(clf.cv_results_)
del optimize_df['params']
#rank_col = optimize_df.pop("rank_test_score")
#optimize_df = optimize_df.insert(1, rank_col.name, rank_col)
optimize_df.round(decimals=3).head(20)
params = {'n_estimators': 300,
          'max_depth': 6}

model = ensemble.GradientBoostingRegressor(**params)
score = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='neg_root_mean_squared_error')
print('Array of cross_val_score results:',score)
print("Unweighted Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() ))

y_pred = cross_val_predict(model, X_train_sc, y_train, cv=5)

fitter_metrics(y_train, y_pred)
params = {'n_estimators': 300,
          'max_depth': 6}

# Initialize and fit the model.
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train_sc, y_train)

feature_importance = clf.feature_importances_

# Make importances relative to max importance.
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance depth 6')

params = {'n_estimators': 300,
          'max_depth': 6}

X_no_h2o = X_train_sc.copy()
del X_no_h2o['basic_water']

clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_no_h2o, y_train)

feature_importance = clf.feature_importances_

# Make importances relative to max importance.
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_no_h2o.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance w/o basic_water')

plt.subplots_adjust(left=0.5, right=1.1)
plt.tight_layout()
plt.show()
params = {'n_estimators': 300,
          'max_depth': 6}

model = ensemble.GradientBoostingRegressor(**params)
score = cross_val_score(model, X_no_h2o, y_train, cv=5, scoring='neg_root_mean_squared_error')
print('Array of cross_val_score results:',score)
print("Unweighted Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() ))

y_pred = cross_val_predict(model, X_no_h2o, y_train, cv=5)

fitter_metrics(y_train, y_pred)
params = {'n_estimators': 300,
          'max_depth': 4}

# Initialize and fit the model.
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train_sc, y_train)
score = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='neg_root_mean_squared_error')


feature_importance = clf.feature_importances_

# Make importances relative to max importance.
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance depth 4')

params = {'n_estimators': 300,
          'max_depth': 8}

clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train_sc, y_train)

feature_importance = clf.feature_importances_

# Make importances relative to max importance.
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance depth 8')

plt.subplots_adjust(left=0.5, right=1.1)
plt.tight_layout()
plt.show()
# use the copy before we dropped features and countries
clean_df = df_before_remove.copy()

more_features = list(clean_df.columns)
more_features.remove(target_feature)

#remove_list = ['hepatitis', 'une_hiv', 'une_poverty', 'une_edu_spend', 'une_literacy',
#               'gni_capita', 'une_pop']
#for col_name in remove_list:
#    del clean_df[col_name]

# drop remaining NaN rows
clean_df = clean_df.dropna(axis=0)

print(clean_df['country_code'].nunique(),"countries to be analyzed")
list_use_country = clean_df['country_code'].unique()

# make another dataframe with the coutries we just dropped
unclean_df = df_before_remove.copy()
unclean_df = unclean_df[~unclean_df.country_code.isin(list_use_country)]

# drop remaining NaN rows
#unclean_df = unclean_df.dropna(axis=0)

print(unclean_df['country_code'].nunique(),"countries excluded")

more_features.remove('country')
more_features.remove('country_code')
more_features.remove('region')
more_features.remove('year')

print("Target feature:",target_feature)
print("Larger set of numeric  features:",len(more_features), more_features)
Y3 = clean_df[target_feature]
X3 = clean_df[more_features]

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, Y3, test_size = 0.2, random_state = rand_seed)

params = {'n_estimators': 300,
          'max_depth': 6}

model = make_pipeline(StandardScaler(), ensemble.GradientBoostingRegressor(**params))
score = cross_val_score(model, X3_train, y3_train, cv=5, scoring='neg_root_mean_squared_error')
print('Array of cross_val_score results:',score)
print("Unweighted Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() ))

y_pred = cross_val_predict(model, X3_train, y3_train, cv=5)

fitter_metrics(y3_train, y_pred)
Y4 = clean_df[target_feature]
X4 = clean_df[num_features]

X4_train, X4_test, y4_train, y4_test = train_test_split(X4, Y4, test_size = 0.2, random_state = rand_seed)
# 
params = {'n_estimators': 300,
          'max_depth': 6}

model = make_pipeline(StandardScaler(), ensemble.GradientBoostingRegressor(**params))
score = cross_val_score(model, X4_train, y4_train, cv=5, scoring='neg_root_mean_squared_error')
print(score)
print("Unweighted Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() ))

y_pred = cross_val_predict(model, X4_train, y4_train, cv=5)

fitter_metrics(y4_train, y_pred)
Y5 = unclean_df[target_feature]
X5 = unclean_df[num_features]

X5_train, X5_test, y5_train, y5_test = train_test_split(X5, Y5, test_size = 0.2, random_state = rand_seed)
# 
params = {'n_estimators': 300,
          'max_depth': 6}

model = make_pipeline(StandardScaler(), ensemble.GradientBoostingRegressor(**params))
score = cross_val_score(model, X5_train, y5_train, cv=5, scoring='neg_root_mean_squared_error')
print(score)
print("Unweighted Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() ))

y_pred = cross_val_predict(model, X5_train, y5_train, cv=5)

fitter_metrics(y5_train, y_pred)
params = {'n_estimators': 300,
          'max_depth': 6}

# Initialize and fit the model.
clf = ensemble.GradientBoostingRegressor(**params)
clf.fit(X_train_sc, y_train)

predict_test = clf.predict(X_test_sc)
fitter_metrics(y_test, predict_test)

feature_importance = clf.feature_importances_

# Make importances relative to max importance.
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_test.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()