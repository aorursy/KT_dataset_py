# Core

import pandas as pd

import numpy as np



# Data Visualisation

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



pd.options.display.max_columns = None

pd.options.display.max_rows = 80



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_path = '/kaggle/input/restaurantrevenue/train.csv'

test_path = '/kaggle/input/restaurantrevenue/test.csv'



data_train = pd.read_csv(train_path)

data_test = pd.read_csv(test_path)

IDtest = data_test['Id']
data_train.head()
# Numerical columns

num_col = data_train.select_dtypes(exclude=['object']).drop(['Id'], axis=1).columns
num_col
# Categorical columns

cat_col = data_train.select_dtypes(include=['object']).columns
cat_col
# Describe data train

data_train[num_col].describe().round(decimals=2)
# Plot skew

fig = plt.figure(figsize=(12,18))

for i in range(len(num_col)):

    fig.add_subplot(10,4,i+1)

    sns.distplot(data_train[num_col[i]], kde_kws={'bw': 0.1})

    plt.title('Skew : %.2f' % data_train[num_col[i]].skew())

    

plt.tight_layout()

plt.show()
# Revenue with log

sns.distplot(np.log(data_train['revenue']))

plt.title('Skew : %.2f' % np.log(data_train['revenue']).skew())
# Univariate analysis - boxplot

fig = plt.figure(figsize=(12,18))

for i in range(len(num_col)):

    fig.add_subplot(10,4,i+1)

    sns.boxplot(y=data_train[num_col[i]])

    

plt.tight_layout()

plt.show()
# Bivariate analysis - scatterplot

fig = plt.figure(figsize=(12,18))

for i in range(len(num_col)):

    fig.add_subplot(10,4,i+1)

    sns.scatterplot(data_train[num_col[i]], data_train['revenue'])

    

plt.tight_layout()

plt.show()
# Correlation

correlation = data_train[num_col].corr()



f, ax = plt.subplots(figsize=(14,12))

plt.title('Correlation of numerical attributes', size=16)

sns.heatmap(correlation)

plt.show()
correlation['revenue'].sort_values(ascending=False)
# Missing value

data_train.isna().sum()
data_train[cat_col].describe()
# Bivariate analysis - box plot

f, ax = plt.subplots(figsize=(12,8))

sns.boxplot(y=data_train['revenue'], x=data_train['Type'])

plt.xticks(rotation=40)

plt.show()
# Bivariate analysis - box plot

f, ax = plt.subplots(figsize=(12,8))

sns.boxplot(y=data_train['revenue'], x=data_train['City Group'])

plt.xticks(rotation=40)

plt.show()
# Transform Open date to age

from datetime import datetime



def count_years(open_date):

    date_parse = datetime.strptime(open_date, '%m/%d/%Y')

    date_now = datetime.now()

    return date_now.year - date_parse.year
open_years = []

for i in data_train['Open Date']:

    open_years.append(count_years(i))



df_open_years = pd.DataFrame({ 'open_years' : open_years } )

group_years = df_open_years['open_years'].value_counts()
# Barplot open years

sns.barplot(x=group_years.index, y=group_years.values)
# City

city_most = data_train['City'].value_counts()[data_train['City'].value_counts() > 2].index
city_transform = []



for i in data_train['City']:

    if i in city_most:

        city_transform.append(i)

    else:

        city_transform.append('other')

        

df_city_transform = pd.DataFrame({ 'city_transform' : city_transform } )

group_city = df_city_transform['city_transform'].value_counts()
# Barplot cgroup city

sns.barplot(x=group_city.index, y=group_city.values)
data_train_copy = data_train.copy()

data_train_copy['revenue_log'] = np.log(data_train_copy['revenue'])
data_train_copy.head()
transformed_corr = data_train_copy.corr()

plt.figure(figsize=(12,10))

sns.heatmap(transformed_corr)
attr_select = ['Open Date', 'City', 'City Group', 'Type', 'P2', 'P6', 'P13', 'P28', 'P29', 'revenue_log']

train_select = data_train_copy[attr_select]
# Transform Open Year

open_years = []

for i in train_select['Open Date']:

    open_years.append(count_years(i))

    

df_open_years = pd.DataFrame({ 'open_years' : open_years } )

group_years = df_open_years['open_years'].value_counts()



# Transform City

city_most = train_select['City'].value_counts()[train_select['City'].value_counts() > 2].index

city_transform = []



for i in train_select['City']:

    if i in city_most:

        city_transform.append(i)

    else:

        city_transform.append('other')

        

df_city_transform = pd.DataFrame({ 'city_transform' : city_transform } )

group_city = df_city_transform['city_transform'].value_counts()



train_final = pd.concat([train_select, df_open_years, df_city_transform], axis=1).drop(['Open Date', 'City'], axis=1)
train_final.head()
# Preprare data

X = train_final.drop(['revenue_log'], axis=1)

y = train_final['revenue_log']



X = pd.get_dummies(X)



X = np.array(X)

y = np.array(y)
from sklearn.model_selection import GridSearchCV, cross_val_score, learning_curve

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import Lasso

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.neighbors import KNeighborsRegressor

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
random_state = 2

classifiers = []

classifiers.append(Lasso(random_state=random_state))

classifiers.append(LinearRegression())

classifiers.append(Ridge(random_state=random_state))

classifiers.append(ElasticNet(random_state=random_state))

classifiers.append(KNeighborsRegressor())

classifiers.append(SVR())

classifiers.append(RandomForestRegressor(random_state=random_state))

classifiers.append(GradientBoostingRegressor())

classifiers.append(AdaBoostRegressor(random_state = random_state))

classifiers.append(DecisionTreeRegressor())

classifiers.append(XGBRegressor())





cv_results = []

for classifier in classifiers :

    cv_results.append(cross_val_score(classifier, X, y, scoring='neg_mean_squared_error', cv =10, n_jobs=4))



cv_means = []

cv_std = []

for cv_result in cv_results:

    cv_means.append(cv_result.mean())

    cv_std.append(cv_result.std())



cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["Lasso","LinearRegression","Ridge",

"ElasticNet","KNeighborsRegressor","SVR","RandomForestRegressor","GradientBoostingRegressor","AdaBoostRegressor","DecisionTreeRegressor", "XGBRegressor"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})

g.set_xlabel("Mean Accuracy")

g = g.set_title("Cross validation scores")
cv_res.sort_values(ascending=False, by='CrossValMeans')
# SVR

model = SVR()



# Search grid for optimal parameters

ex_param_grid = {

    'kernel': ['rbf'],

    'gamma': [1e-4, 1e-3, 0.01, 0.1],

    'C': [1, 10, 100]

}



gsSVR = GridSearchCV(model, 

                     param_grid = ex_param_grid, 

                     cv=10, 

                     scoring="neg_mean_squared_error")



gsSVR.fit(X, y)

SVR_best = gsSVR.best_estimator_



# Best score

print('SVR')

print('Best score : ', gsSVR.best_score_)

print('Best params : ', gsSVR.best_params_)
# AdaBoostRegressor

model = AdaBoostRegressor(random_state = random_state)



# Search grid for optimal parameters

ex_param_grid = {

    'n_estimators': [50, 100],

    'learning_rate' : [0.01,0.05,0.1,0.3,1],

    'loss' : ['linear', 'square', 'exponential']

}



gsABR = GridSearchCV(model, 

                     param_grid = ex_param_grid, 

                     cv=10, 

                     scoring="neg_mean_squared_error")



gsABR.fit(X, y)

ABR_best = gsABR.best_estimator_



# Best score

print('ABR')

print('Best score : ', gsABR.best_score_)

print('Best params : ', gsABR.best_params_)
# RandomForestRegressor

model = RandomForestRegressor(random_state = random_state)



# Search grid for optimal parameters

ex_param_grid = {

    'n_estimators'      : [10,20,30,40,50],

    'max_features'      : ["auto", "sqrt", "log2"],

    'min_samples_split' : [2,4,8,10,12,14,16]

}



gsRFR = GridSearchCV(model, 

                     param_grid = ex_param_grid, 

                     cv=10, 

                     scoring="neg_mean_squared_error")



gsRFR.fit(X, y)

RFR_best = gsRFR.best_estimator_



# Best score

print('RFR')

print('Best score : ', gsRFR.best_score_)

print('Best params : ', gsRFR.best_params_)
# KNeighborsRegressor

model = KNeighborsRegressor()



# Search grid for optimal parameters

ex_param_grid = {

    'n_neighbors': [4,6,8,10],

    'leaf_size': [30,40,50,60],

    'weights': ['uniform','distance']

}



gsKNR = GridSearchCV(model, 

                     param_grid = ex_param_grid, 

                     cv=10, 

                     scoring="neg_mean_squared_error")



gsKNR.fit(X, y)

KNR_best = gsKNR.best_estimator_



# Best score

print('KNR')

print('Best score : ', gsKNR.best_score_)

print('Best params : ', gsKNR.best_params_)
# GradientBoostingRegressor

model = GradientBoostingRegressor(random_state = random_state)



# Search grid for optimal parameters

ex_param_grid = {

    'learning_rate': [0.25, 0.1, 0.05, 0.01],

    'n_estimators': [50,100,200,300],

    'max_depth': [3,5,7]

}



gsGBR = GridSearchCV(model, 

                     param_grid = ex_param_grid, 

                     cv=10, 

                     scoring="neg_mean_squared_error")



gsGBR.fit(X, y)

GBR_best = gsGBR.best_estimator_



# Best score

print('GBR')

print('Best score : ', gsGBR.best_score_)

print('Best params : ', gsGBR.best_params_)
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
g = plot_learning_curve(SVR_best,"SVR",X, y,cv=10)

g = plot_learning_curve(ABR_best,"AdaBoost",X, y,cv=10)

g = plot_learning_curve(RFR_best,"RandomForest",X, y,cv=10)

g = plot_learning_curve(KNR_best,"KNeighbors",X, y,cv=10)

g = plot_learning_curve(GBR_best,"GradientBoosting",X, y,cv=10)
## Plot Feature Importance of Random Forest



plt.figure(figsize=(10,10))

names_classifiers = [("RFR",RFR_best)]

nclassifier = 0

name = names_classifiers[nclassifier][0]

classifier = names_classifiers[nclassifier][1]

indices = np.argsort(classifier.feature_importances_)[::-1][:40]



train_dummies = pd.get_dummies(train_final.drop(['revenue_log'], axis=1))



g = sns.barplot(y=train_dummies.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h')

g.set_xlabel("Relative importance",fontsize=12)

g.set_ylabel("Features",fontsize=12)

g.tick_params(labelsize=9)

g.set_title(name + " feature importance")   
## Plot Feature Importance of Gradient Boosting



plt.figure(figsize=(10,10))

names_classifiers = [("GBR",GBR_best)]

nclassifier = 0

name = names_classifiers[nclassifier][0]

classifier = names_classifiers[nclassifier][1]

indices = np.argsort(classifier.feature_importances_)[::-1][:40]



train_dummies = pd.get_dummies(train_final.drop(['revenue_log'], axis=1))



g = sns.barplot(y=train_dummies.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h')

g.set_xlabel("Relative importance",fontsize=12)

g.set_ylabel("Features",fontsize=12)

g.tick_params(labelsize=9)

g.set_title(name + " feature importance") 
data_test.head()
attr_select = ['Open Date', 'City', 'City Group', 'Type', 'P2', 'P6', 'P13', 'P28', 'P29']

test_select = data_test[attr_select]
# Transform Open Year

open_years = []

for i in test_select['Open Date']:

    open_years.append(count_years(i))

    

df_open_years = pd.DataFrame({ 'open_years' : open_years } )

group_years = df_open_years['open_years'].value_counts()



# Transform City

city_most = train_select['City'].value_counts()[train_select['City'].value_counts() > 2].index

city_transform = []



for i in test_select['City']:

    if i in city_most:

        city_transform.append(i)

    else:

        city_transform.append('other')

        

df_city_transform = pd.DataFrame({ 'city_transform' : city_transform } )

group_city = df_city_transform['city_transform'].value_counts()



test_final = pd.concat([test_select, df_open_years, df_city_transform], axis=1).drop(['Open Date', 'City'], axis=1)
X_test = pd.get_dummies(test_final).drop(['Type_MB'], axis=1)

test = pd.get_dummies(X_test)
test_type_SVR = pd.Series(SVR_best.predict(test), name="SVR")

test_type_ABR = pd.Series(ABR_best.predict(test), name="ABR")

test_type_RFR = pd.Series(RFR_best.predict(test), name="RFR")

test_type_KNR = pd.Series(KNR_best.predict(test), name="KNR")

test_type_GBR = pd.Series(GBR_best.predict(test), name="GBR")





# Concatenate all classifier results

ensemble_results = pd.concat([test_type_SVR, test_type_ABR, test_type_RFR, test_type_KNR, test_type_GBR],axis=1)



g= sns.heatmap(ensemble_results.corr(),annot=True)
# Using Voting Regressor

from sklearn.ensemble import VotingRegressor



votingR = VotingRegressor(estimators=[('svr', SVR_best), ('abr', ABR_best),

('gbr', GBR_best), ('rfr', RFR_best), ('knr', KNR_best)], n_jobs=4)



votingR = votingR.fit(X, y)
predict_test = pd.Series(np.exp(votingR.predict(test)), name="Prediction")

results = pd.concat([IDtest, predict_test],axis=1)

results.to_csv("my_prediction.csv",index=False)