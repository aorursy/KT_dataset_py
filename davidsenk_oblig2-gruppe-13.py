import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
train = pd.read_csv('../input/dat158-2020/housing_data.csv')

test = pd.read_csv('../input/dat158-2020/housing_test_data.csv')

sample_submission = pd.read_csv('../input/dat158-2020/sample_submission.csv')
train.head(10)
train.describe()
train.hist(bins=25, figsize=(20,15))

plt.show()
import seaborn as sns

sns.heatmap(train.corr(), annot=True, fmt='.2g', cmap='coolwarm')
train['income_cat'] = pd.cut(train['median_income'], bins=[0,2,4,6,8,np.inf], labels=[1,2,3,4,5])

train['income_cat'].hist(color='skyblue', ec='black', grid=False)

plt.show()
# Provides train/test indices to split data in train/test sets.

from sklearn.model_selection import StratifiedShuffleSplit



sss = StratifiedShuffleSplit(random_state=42, n_splits=1, test_size=0.2)

for train_index, test_index in sss.split(train, train['income_cat']):

    strat_train_set = train.loc[train_index]

    strat_test_set = train.loc[test_index]

strat_test_set['income_cat'].value_counts() / len(strat_test_set)
compare_props = pd.DataFrame({

    'Overall': train['income_cat'].value_counts() / len(train),

    'Stratified': strat_test_set['income_cat'].value_counts() / len(strat_test_set)

}).sort_index()

compare_props['Stratified % error'] = 100 * compare_props['Stratified'] / compare_props['Overall'] - 100

compare_props
for set_ in (strat_train_set, strat_test_set):

    set_.drop('income_cat', axis=1, inplace=True)
import matplotlib as mpl

sns.set(rc={'figure.figsize':(10,10), 'axes.facecolor':'white'})
sns.scatterplot(train['longitude'], train['latitude'], hue=train['median_house_value'], s=50, palette='BuGn')

regplot = sns.regplot(data=train, x='longitude', y='latitude', scatter=False)

regplot.figure.colorbar(plt.cm.ScalarMappable

                        (norm = mpl.colors.Normalize

                         (vmin=0, vmax=train['median_house_value'].max(), clip=False), cmap='BuGn'),label='Median House Value')
ax = sns.catplot(x='ocean_proximity', y='median_house_value', kind='boxen', data=train)

for axes in ax.axes.flat:

    axes.set_xticklabels(axes.get_xticklabels(), rotation=40, horizontalalignment='right')
# xlabel vises ikke.

train.plot(kind='scatter', y='median_house_value', x='median_income', 

           s='housing_median_age', 

           title='Boligens alder basert på median house value og median income', 

           figsize=(10,10), alpha=0.5, c='housing_median_age', cmap=plt.get_cmap('jet'), colorbar=True)
train.plot(kind='scatter', x='median_income', y='median_house_value', 

           s=train['housing_median_age'] > 25, title='Age > 25', figsize=(10,10), alpha=0.5, 

           c='housing_median_age', cmap=plt.get_cmap('jet'), colorbar=True)

train.plot(kind='scatter', x='median_income', y='median_house_value', 

           s=train['housing_median_age'] < 25, title='Age < 25', 

           figsize=(10,10), alpha=0.5, c='housing_median_age', 

           cmap=plt.get_cmap('jet'), colorbar=True)
corr_matrix = train.corr()

corr_matrix['median_house_value'].sort_values(ascending=False)
#train = the whole dataset except the column 'median_house_value'.

train_set = strat_train_set.drop('median_house_value', axis=1)

#Copies the column 'median_house_value' into train_labels.

train_labels = strat_train_set['median_house_value'].copy()
train_set.info()
from sklearn.impute import SimpleImputer



imputer = SimpleImputer(strategy='median')

train_num = train_set.drop('ocean_proximity', axis=1)

imputer.fit(train_num)



#Median for all columns except ocean_prox.

imputer.statistics_
X = imputer.transform(train_num)

train_tr = pd.DataFrame( X,  columns=train_num.columns)

train_tr['total_bedrooms'].count
from sklearn.preprocessing import OneHotEncoder



onehot_encoder = OneHotEncoder()

ocean_cat_encoded = onehot_encoder.fit_transform(train_set[['ocean_proximity']])

ocean_cat_encoded.toarray()
onehot_encoder.categories_
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='median')),

    ('std_scaler', StandardScaler())

])

train_tr = num_pipeline.fit_transform(train_num)
test_id = test['Id'].copy()

test.drop('Id', axis=1, inplace=True)
from sklearn.compose import ColumnTransformer



num_attr = list(train_num)

cat_attr = ['ocean_proximity']



full_pipeline = ColumnTransformer([

    ('num', num_pipeline, num_attr),

    ('cat', OneHotEncoder(), cat_attr)

])

train_prepared = full_pipeline.fit_transform(train_set)

submission = full_pipeline.transform(test)
#Predicting using linear regression.

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



lin_reg = LinearRegression()

lin_reg.fit(train_prepared, train_labels)

train_pred = lin_reg.predict(train_prepared)

lin_mse = mean_squared_error(train_labels, train_pred)

lin_rmse = np.sqrt(lin_mse)

print(lin_rmse)
from sklearn.ensemble import RandomForestRegressor

for_reg = RandomForestRegressor(random_state=42)

for_reg.fit(train_prepared, train_labels)

train_pred1 = for_reg.predict(train_prepared)

lin_rmse1 = np.sqrt(mean_squared_error(train_labels, train_pred1))

print(lin_rmse1)
from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon=1.5)

svm_reg.fit(train_prepared, train_labels)

train_pred2 = svm_reg.predict(train_prepared)

svr_mse = mean_squared_error(train_labels, train_pred2)

print(np.sqrt(svr_mse))
import lightgbm as lgb

lgb_reg = lgb.LGBMRegressor()

lgb_reg.fit(train_prepared, train_labels, eval_metric='rmse', verbose=2000)

train_pred3 = lgb_reg.predict(train_prepared)

lgb_mse = mean_squared_error(train_labels, train_pred3)

print(np.sqrt(lgb_mse))
from sklearn.feature_selection import SelectKBest, f_regression



pipeline_reg = Pipeline([

    ('selector', SelectKBest(f_regression)),

    ('model', lgb.LGBMRegressor())

])
from sklearn.model_selection import GridSearchCV



grid_search = GridSearchCV(

    estimator = pipeline_reg,

    param_grid = {'selector__k':[12], 'model__n_estimators':np.arange(494,495,1),

                  'model__max_bin':[1000]},

    cv=100,

    verbose=3,

    scoring='neg_mean_squared_error', 

    return_train_score=True

)

grid_search.fit(train_prepared, train_labels)
cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)

print(grid_search.best_params_)
final_model = grid_search.best_estimator_

final_predict = final_model.predict(submission)


output = pd.DataFrame({

    'Id': test_id,

    'median_house_value': final_predict

})

output.head(10)
output.to_csv('submission.csv', index=False)