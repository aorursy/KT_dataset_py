# load modules

import pandas as pd

import numpy as np

from sklearn import preprocessing

from sklearn.metrics import r2_score



import matplotlib.pyplot as plt

%matplotlib inline
# load data

train_df  = pd.read_csv('../input/train.csv')

test_df  = pd.read_csv('../input/test.csv')
# get train_y, test ids and unite datasets to perform

train_y = train_df['y']

train_df.drop('y', axis = 1, inplace = True)

test_ids = test_df.ID.values

all_df = pd.concat([train_df,test_df], axis = 0)



# ...one hot encoding of categorical variables

categorical =  ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]

for f in categorical:

    dummies = pd.get_dummies(all_df[f], prefix = f, prefix_sep = '_')

    all_df = pd.concat([all_df, dummies], axis = 1)



# drop original categorical features

all_df.drop(categorical, axis = 1, inplace = True)
# get feature dataset for test and training        

train_X = all_df.drop(["ID"], axis=1).iloc[:len(train_df),:]

test_X = all_df.drop(["ID"], axis=1).iloc[len(train_df):,:]
print(train_X.head())

print(test_X.head())
# Let's perform a cross-validation to find the best combination of alpha and l1_ratio

from sklearn.linear_model import ElasticNetCV, ElasticNet



cv_model = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, .995, 1], eps=0.001, n_alphas=100, fit_intercept=True, 

                        normalize=True, precompute='auto', max_iter=2000, tol=0.0001, cv=5, 

                        copy_X=True, verbose=0, n_jobs=-1, positive=False, random_state=None, selection='cyclic')
cv_model.fit(train_X, train_y)
print('Optimal alpha: %.8f'%cv_model.alpha_)

print('Optimal l1_ratio: %.3f'%cv_model.l1_ratio_)

print('Number of iterations %d'%cv_model.n_iter_)
# train model with best parameters from CV

model = ElasticNet(l1_ratio=cv_model.l1_ratio_, alpha = cv_model.alpha_, max_iter=cv_model.n_iter_, fit_intercept=True, normalize = True)

model.fit(train_X, train_y)
# r2 score on training dataset

print(r2_score(train_y, model.predict(train_X)))
# preds = model.predict(test_X)

# df_sub = pd.DataFrame({'ID': test_ids, 'y': preds})

# df_sub.to_csv('elnet_submission_dummies.csv', index=False)
feature_importance = pd.Series(index = train_X.columns, data = np.abs(model.coef_))



n_selected_features = (feature_importance>0).sum()

print('{0:d} features, reduction of {1:2.2f}%'.format(

    n_selected_features,(1-n_selected_features/len(feature_importance))*100))



feature_importance.sort_values().tail(30).plot(kind = 'bar', figsize = (18,6))