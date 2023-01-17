# library import

import pandas as pd 

import numpy as np 

import seaborn as sns 

import matplotlib.pyplot as plt

import warnings 

from scipy.stats import norm



warnings.simplefilter(action='ignore', category=FutureWarning)



def coef_plot(model, X, abs_ = False, n = 5):

    if abs_ == False:

        coefs = pd.DataFrame({'name': X.columns, 'coef' : model.coef_}).sort_values('coef', ascending = False)

    else:

        coefs = pd.DataFrame({'name': X.columns, 'coef' : np.abs(model.coef_)}).sort_values('coef', ascending = False)

    plt.figure(figsize = (16, 8))

    sns.pointplot(y="name", x="coef",

                  data=coefs.head(n), ci=None, color = 'C0')

    sns.barplot(y = "name", x= "coef", data=coefs.head(n), ci=None, color = 'C0', alpha = 0.2)

    plt.title('Coeficient Plot')

    plt.tight_layout()
# Use pandas to read in CSV files

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

# row and column number in train set 

print("train rows amd columns", train.shape)

print("test rows amd columns", test.shape)
# plot histograms of all the numeric columns

train.hist(figsize = (12, 7), bins = 40)

plt.tight_layout()

plt.show()
plt.figure(figsize = (12, 6))

train.skew().sort_values().plot(kind = 'bar', color = 'C0')

plt.title('Skew of Variables')
fig, ax = plt.subplots(2, 1, figsize = (12, 8)) 



sns.distplot(train['SalePrice'], fit = norm, ax = ax[0])

sns.distplot(np.log(train['SalePrice']), fit = norm, ax = ax[1])

plt.show()
ax = sns.clustermap(train.corr(method='spearman'), center = np.median(train.corr(method='spearman')))

plt.show()
plt.figure(figsize = (12, 8))

cormat = train.corr(method = 'spearman')[['SalePrice']].sort_values(by = 'SalePrice', ascending = True)[:-1]

sns.heatmap(cormat, annot = True, center = np.median(cormat))

plt.title("Numeric Variable's Correaltion with Sale Price")

plt.show()
train.select_dtypes('object').describe()
plt.figure(figsize = (12, 6))

train['Neighborhood'].value_counts().plot(kind = 'barh', color = 'C0')

plt.xlabel('Count')

plt.title("Woah! That's a Lot of Neighborhoods")

sns.despine()

plt.show()
for col in train.select_dtypes('object').columns:

    top = train[col].value_counts().head(10)

    train[col] = [x if x in top else "other" for x in train[col]]

    test[col] = [x if x in top else "other" for x in test[col]]
train.isna().sum()[train.isna().sum() > 0].sort_values().plot(kind = 'barh', color = 'C0', figsize = (10, 4))

plt.title('Missing Values')

plt.show()
print(pd.get_dummies(train).shape)

print(pd.get_dummies(test).shape)
from sklearn.linear_model import Lasso

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.impute import SimpleImputer



y = train['SalePrice']

X = train.drop(['SalePrice'], 1)



train_objs_num = len(X)

dataset = pd.concat([X, test])

dataset = pd.get_dummies(dataset)

X = dataset[:train_objs_num].copy()

test = dataset[train_objs_num:].copy()
import warnings 

from sklearn.exceptions import ConvergenceWarning





# these warnings are telling me that some of the alpha values are too low, I know this. I am attempting to do a wide grid search.

warnings.filterwarnings(action='ignore', category=ConvergenceWarning,)



alphas = []

scores = []



for i in [1e-6, 1e-4, 1e-2,.1, 1, 5, 10, 20, 30, 50, 100,200,250, 300,400,500,750,1000]:

    

    pipe = Pipeline([

                 ('Imputer', SimpleImputer(strategy = 'most_frequent')),

                 ('scaler', RobustScaler()),

                ('lasso', Lasso(alpha= i, max_iter= 10000))

            ])

    pipe.fit(X, y)

    score = cross_val_score(pipe, X, y, cv = 5)

    # nested loops in python are really ugly and should be avoided by I'm lazy

    for x in score:

        scores.append(x)

        alphas.append(i)

ridge_frame = pd.DataFrame({'alpha': alphas, 'score': scores})



top_alpha = ((ridge_frame.groupby('alpha', as_index=False).mean().sort_values('score')))



top_alpha = top_alpha.iloc[-1:, 0].values

print(top_alpha[0])
plt.figure(figsize = (12, 8))

sns.lineplot('alpha', 'score', data = ridge_frame, ci = "sd")

plt.title('Crossvalidated Alpha vs R^2 Score + or - 1 sd')

plt.axvline(top_alpha, color = "C1")

plt.xscale('log')

plt.show()
pipe = Pipeline([

                  ('Imputer', SimpleImputer(strategy = 'most_frequent')),  

                 ('scaler', RobustScaler()),

                ('lasso', Lasso(alpha= top_alpha))

            ])

pipe.fit(X, y)

score = cross_val_score(pipe, X, y, cv = 5)



rcv = pipe.named_steps['lasso']



coef_plot(rcv, X, abs_ = True, n = 40)

print(np.mean(scores))

print(np.std(scores))

plt.show()
from sklearn.linear_model import Ridge

alphas = []

scores = []



for i in [1e-6, 1e-4, 1e-2,.1, 1, 5, 10, 20, 30, 50, 100,200,250, 300,400,500,750, 1000,1500, 2000,2500, 3000, 5000]:

    

    pipe = Pipeline([

                 ('Imputer', SimpleImputer(strategy = 'most_frequent')),

                 ('scaler', RobustScaler()),

                ('ridge', Ridge(alpha= i, max_iter= 10000))

            ])

    pipe.fit(X, y)

    score = cross_val_score(pipe, X, y, cv = 5)

    # nested loops in python are really ugly and should be avoided by I'm lazy

    for x in score:

        scores.append(x)

        alphas.append(i)

ridge_frame = pd.DataFrame({'alpha': alphas, 'score': scores})



top_alpha = ((ridge_frame.groupby('alpha', as_index=False).mean().sort_values('score')))



top_alpha = top_alpha.iloc[-1:, 0].values
pipe = Pipeline([

                  ('Imputer', SimpleImputer(strategy = 'most_frequent')),  

                 ('scaler', RobustScaler()),

                ('ridge', Ridge(alpha= top_alpha))

            ])

pipe.fit(X, y)

score = cross_val_score(pipe, X, y, cv = 5)



rcv = pipe.named_steps['ridge']



coef_plot(rcv, X, abs_ = True, n = 40)

print(np.mean(scores))

print(np.std(scores))

plt.show()
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV







pipe = Pipeline([('imp', SimpleImputer(strategy = 'most_frequent')),

                 ('scaler', RobustScaler()),

                ('rf', RandomForestRegressor())

            ])

pipe.fit(X, y)

params_rf = {'rf__n_estimators' : [100, 350, 500, 1000, 2000],

            'rf__max_features': ['log2', 'auto', 'sqrt'],

            'rf__min_samples_leaf': [1, 2, 5, 10, 30],

            "rf__min_samples_split": [2, 3, 5, 7,9,11,13,15,17],

            "rf__max_depth" : [None, 1, 3, 5, 7, 9, 11]}



# Import GridSearchCV



# Instantiate grid_rf

grid_rf = RandomizedSearchCV(estimator=pipe,

                       param_distributions=params_rf,

                       cv=5,

                       verbose=0,

                       n_jobs=-1,

                       n_iter = 30)



grid_rf.fit(X,y)

bestmodel = grid_rf.best_estimator_



print(grid_rf.best_score_)
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_squared_log_error



pipe = Pipeline([('imp', SimpleImputer(strategy = 'most_frequent')),

                 ('scaler', RobustScaler()),

                ('gbm', GradientBoostingRegressor())

            ])



pipe.fit(X, y)

params_gbm = {'gbm__n_estimators' : [100, 350, 500, 1000, 2000],

            'gbm__max_features': ['log2', 'auto', 'sqrt'],

            'gbm__min_samples_leaf': [1, 2, 10, 30],

            "gbm__min_samples_split": [2, 3,5,7,9,11],

            "gbm__learning_rate" : [1e-5,1e-4, 1e-3, 1e-2, 0.1, 1]}

# Import GridSearchCV



# Instantiate grid_rf

grid_rf = RandomizedSearchCV(estimator=pipe,

                       param_distributions=params_gbm,

                       cv=5,

                       verbose=0,

                       n_jobs=-1, 

                       n_iter = 30)



grid_rf.fit(X,y)

bestmodel = grid_rf.best_estimator_

print(grid_rf.best_score_)