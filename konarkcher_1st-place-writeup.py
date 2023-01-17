import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



%matplotlib inline
train_df = pd.read_csv('../input/houses_train.csv')

test_df = pd.read_csv('../input/houses_test.csv')

print('Train shape:', train_df.shape)

print('Test shape:', test_df.shape)



train_df.head()
train_df = train_df.drop('id', 1)

test_df = test_df.drop('id', 1)



train_df.head()
def get_month_and_year(date):

    return pd.Series({'year': int(date[:4]), 'month': int(date[4:6])})

    

train_df = pd.concat([train_df, train_df['date'].apply(get_month_and_year)], 1).drop('date', 1)

test_df = pd.concat([test_df, test_df['date'].apply(get_month_and_year)], 1).drop('date', 1)



train_df.head()
def get_dates(df, min_year):

    return df.year * 12 + df.month - min_year * 12



min_year = min(train_df.year.min(), test_df.year.min())

plt.figure(figsize=(12, 8))

plt.title('Month distribution starting from January {}'.format(min_year))



plt.hist(get_dates(train_df, min_year), label='train')

plt.hist(get_dates(test_df, min_year), label='test')

plt.legend();
def add_coord(df):

    x = np.cos(df.lat) * np.cos(df.long)

    y = np.cos(df.lat) * np.sin(df.long) 

    z = np.sin(df.lat) 

    return pd.DataFrame({'x': x, 'y': y, 'z': z})



train_df = pd.concat([train_df, add_coord(train_df)], 1).drop(['lat', 'long'], 1)

test_df = pd.concat([test_df, add_coord(test_df)], 1).drop(['lat', 'long'], 1)



train_df.head()
X_with_ans, y = train_df.drop('price', 1), train_df.price

X_test = test_df



X_with_ans.head()
import seaborn as sns

from scipy import stats

from scipy.stats import norm

import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)



plt.figure(figsize=(18, 6))

plt.subplot(121)

sns.distplot(y, fit=norm);



(mu, sigma) = norm.fit(y)

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.xlabel('Price')

plt.ylabel('Frequency')

plt.title('Price distribution')



plt.subplot(122)

stats.probplot(y, plot=plt)

plt.title('Price QQ-plot');
y = np.log1p(y)



plt.figure(figsize=(18, 6))

plt.subplot(121)

sns.distplot(y, fit=norm);



(mu, sigma) = norm.fit(y)

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

plt.xlabel('Price')

plt.ylabel('Frequency')

plt.title('Price distribution')



plt.subplot(122)

stats.probplot(y, plot=plt)

plt.title('Price QQ-plot');
from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_val_score, GridSearchCV



def mape_exp(y_true, y_pred):

    y_true = np.expm1(y_true)

    y_pred = np.expm1(y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100



mape_scorer = make_scorer(mape_exp, greater_is_better=False)
from lightgbm import LGBMRegressor



def score_model(model, model_name):

    score = cross_val_score(model, X_with_ans.values, y.values, scoring=mape_scorer, cv=3)

    print(f'{model_name} score on folds:', -score)

    print(f'{model_name} mean score:', -score.mean())

    



lgb = LGBMRegressor(n_jobs=-1, subsample_freq=1, subsample=0.9, colsample_bytree=0.7,

                    n_estimators=760, num_leaves=40, random_state=42)

score_model(lgb, 'LightGBM')
from xgboost import XGBRegressor



xgb = XGBRegressor(n_jobs=-1, colsample_bytree=0.7, max_depth=6, learning_rate=0.05,

                   random_state=42, num_leaves=40, subsample=0.9, n_estimators=920)

score_model(xgb, 'XGBoost')
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone

from sklearn.model_selection import KFold



class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # We again fit the data on clones of the original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        

        # Train cloned base models then create out-of-fold predictions

        # that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

   

    #Do the predictions of all base models on the test data and use the averaged predictions as 

    #meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
from sklearn.linear_model import Lasso

from sklearn.preprocessing import RobustScaler

from sklearn.pipeline import make_pipeline



lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, random_state=42))

stack_model = StackingAveragedModels([lgb, xgb], lasso)



score_model(stack_model, 'StackModel')
stack_model.fit(X_with_ans.values, y)

ans = np.expm1(stack_model.predict(X_test.values))

ans[:5]
ans_pd = pd.DataFrame({'index': np.arange(len(ans)) + 1, 'price': ans})

ans_pd.to_csv('submission.csv', index=False)

pd.read_csv('submission.csv').head()