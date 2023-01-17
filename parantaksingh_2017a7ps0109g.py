# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('/kaggle/input/bits-f464-l1/train.csv')

dataset.head()
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')



import warnings

def ignore_warn(*args, **kwargs):

    pass

# ignore warnings from sklearn and seaborn

warnings.warn = ignore_warn



from scipy import stats

from scipy.stats import norm, skew



pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

# limiting to 3 decimal points
dataset_id = dataset['id']



y = dataset['label']



dataset = dataset.drop(['label'],axis=1)

dataset.drop("id", axis = 1, inplace = True)



dataset.head()
# analysis on the agent rating -> label

sns.distplot(y, fit=norm);



# get the fitted parameters used by the function

(mu, sigma) = norm.fit(y)

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



# plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')



plt.ylabel('Frequency')

plt.title('Rating')



# QQ plot | to compare probability distributions

fig = plt.figure()

res = stats.probplot(y, plot=plt)

plt.show()
# as linear models prefer normally distribution, try transformation

y = np.log1p(y)



# check the new distribution

sns.distplot(y, fit=norm)



# new parameters

(mu, sigma) = norm.fit(y)

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



# plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.ylabel('Frequency')

plt.title('Rating')



# QQ-plot

fig = plt.figure()

res = stats.probplot(y, plot=plt)

plt.show()
# Feature Engineering



# seeing unique values

dtype_unique = pd.concat([dataset.dtypes, dataset.nunique()], axis = 1)

dtype_unique.columns = ["dtype","unique_vals"]



for ind in dtype_unique.index:

    print(ind, dtype_unique["unique_vals"][ind])



# dropping all columns which have just 1 unique value

dataset.drop(['b10', 'b12', 'b26', 'b61', 'b81'], axis=1, inplace=True)
# dataset['b25'] = dataset['b25'].astype('category')

# dataset['b28'] = dataset['b28'].astype('category')

# dataset['b32'] = dataset['b32'].astype('category')

# dataset['b33'] = dataset['b33'].astype('category')

# dataset['b41'] = dataset['b41'].astype('category')

# # dataset['b42'] = dataset['b42'].astype('category')

# dataset['b57'] = dataset['b57'].astype('category')

# # dataset['b74'] = dataset['b74'].astype('category')

# # dataset['b93'] = dataset['b93'].astype('category')



# # getting dummy categorical features

# dataset = pd.get_dummies(dataset)

# print(dataset.shape)# getting dummy categorical features



# '''

# did not improve the RMSE, avoided changing these variables

# '''
# # skewed features

# numeric_feats = dataset.dtypes[dataset.dtypes != "category"].index



# # check skew of all numerical features

# skewed_feats = dataset[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

# print("\nSkew in numerical features: \n")

# skewness = pd.DataFrame({'Skew' :skewed_feats})

# skewness.head()



# # through analysis, we discovered that correcting the skew through box cox messes up the power of predictive model
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X = dataset

X = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split



# split into 20% test and 80% training

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestRegressor



regressor = RandomForestRegressor(n_estimators=300, 

                                  random_state=0, 

                                  n_jobs=-1, 

                                  oob_score=True, 

                                  bootstrap=True)

# n_jobs = -1, we'll use all processors

# oob_score = out of bag



regressor.fit(X_train,y_train)
# sort features by decreasing order of importance



importances = regressor.feature_importances_

std = np.std([regressor.feature_importances_ for tree in regressor.estimators_],

            axis=0)

indices = np.argsort(importances)
plt.figure(figsize=(20,60))

plt.title("Feature importances")

plt.barh(range(X_train.shape[1]), importances[indices],

       color="r", xerr=std[indices], align="center")



plt.yticks(range(X_train.shape[1]), indices)

plt.ylim([-1, X_train.shape[1]])

plt.show()
from sklearn.feature_selection import SelectFromModel



sfm = SelectFromModel(regressor, threshold=0.0015)



# train the selector

sfm.fit(X_train, y_train)
from sklearn import metrics



X_important_train = sfm.transform(X_train)

X_important_test = sfm.transform(X_test)



# create a new random forest

regressor_imp = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1,

                                     oob_score=True,

                                     bootstrap=True)



# train the new regressor

regressor_imp.fit(X_important_train,y_train)



# predict values

Y_pred = np.expm1(regressor_imp.predict(X_important_test))



# see RMSE

print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(np.expm1(y_test),Y_pred)))



# print R^2 metric

print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(regressor_imp.score(X_important_train, y_train), 

                                                                                             regressor_imp.oob_score_,

                                                                                             regressor_imp.score(X_important_test, y_test)))
# '''

# I also tried stacked regression while testing individual models, did not yield satisfactory result

# '''

# # LASSO

# lasso = make_pipeline(RobustScaler(), 

#                       Lasso(alpha =0.0005, random_state=1))

# # Elastic Net

# ENet = make_pipeline(RobustScaler(), 

#                      ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

# # Random Forest Regressor

# RF = RandomForestRegressor(n_estimators=100, criterion='mse', 

#                            max_depth=4, min_samples_split=10, 

#                            min_samples_leaf=15, max_features='sqrt', 

#                            random_state=0)

# # Gradient Boosting Regression

# # with huber loss that makes it robust to outliers

#     GBoost = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,

#                                    max_depth=4, max_features='sqrt',

#                                    min_samples_leaf=15, min_samples_split=10, 

#                                    loss='huber', random_state =5)



# class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

#     def __init__(self, models):

#         self.models = models

        

#     # we define clones of the original models to fit the data in

#     def fit(self, X, y):

#         self.models_ = [clone(x) for x in self.models]

        

#         # Train cloned base models

#         for model in self.models_:

#             model.fit(X, y)



#         return self

    

#     #Now we do the predictions for cloned models and average them

#     def predict(self, X):

#         predictions = np.column_stack([

#             model.predict(X) for model in self.models_

#         ])

#         return np.mean(predictions, axis=1)