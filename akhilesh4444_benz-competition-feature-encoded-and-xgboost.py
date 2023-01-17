import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
plt.figure(figsize=(10,5))
plt.hist(train['y'])
plt.xlabel('value')
plt.ylabel('frequency')
plt.title('Distribution of target variable')
train_cols = train.columns
train_cols_num = train._get_numeric_data().columns
cat_cols = list(set(train_cols) - set(train_cols_num))
cat_cols
uniq_vals = {}
for col in cat_cols:
    uniq_vals[col] = len(train[col].unique())
cat_cols
train_desc = train.describe()

# Checking for the presence of negative values in the dataset
# - There are no negative values in the dataset
min(train_desc.loc['min',:])


def limits(k):
    upper_limit = k.mean() + 2*k.std()
    lower_limit = k.mean() - 2*k.std()
    std = k.std()
    return (lower_limit,upper_limit)

outlier_indices = []
mask = (train['y'] < limits(train['y'])[0]) | (train['y'] > limits(train['y'])[1])
outlier_indices.extend(train['y'][mask].index.values)
train_cleaned = train.drop(train.index[list(set(outlier_indices))])
train_cleaned
import seaborn as sns

fig,(ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8) = plt.subplots(8,1,figsize = (20,25))
sns.boxplot(data = train_cleaned,x = 'X0', y = 'y',ax = ax1)
sns.boxplot(data = train_cleaned,x = 'X1', y = 'y',ax = ax2)
sns.boxplot(data = train_cleaned,x = 'X2', y = 'y',ax = ax3)
sns.boxplot(data = train_cleaned,x = 'X3', y = 'y',ax = ax4)
sns.boxplot(data = train_cleaned,x = 'X4', y = 'y',ax = ax5)
sns.boxplot(data = train_cleaned,x = 'X5', y = 'y',ax = ax6)
sns.boxplot(data = train_cleaned,x = 'X6', y = 'y',ax = ax7)
sns.boxplot(data = train_cleaned,x = 'X8', y = 'y',ax = ax8)
# Glue train + test
train['eval_set'] = 0; test['eval_set'] = 1
df = pd.concat([train, test], axis=0, copy=True,sort = True)
# Reset index
df.reset_index(drop=True, inplace=True)
df
def add_new_col(x):
    if x not in new_col.keys(): 
        # set n/2 x if is contained in test, but not in train 
        # (n is the number of unique labels in train)
        # or an alternative could be -100 (something out of range [0; n-1]
        return int(len(new_col.keys())/2)
    return new_col[x] # rank of the label

for c in cat_cols:
    # get labels and corresponding means
    new_col = train_cleaned.groupby(c).y.mean().sort_values().reset_index()
    # make a dictionary, where key is a label and value is the rank of that label
    new_col = new_col.reset_index().set_index(c).drop('y', axis=1)['index'].to_dict()
    # add new column to the dataframe
    df[c + '_new'] = df[c].apply(add_new_col)

# # drop old categorical columns
df_new = df.drop(cat_cols, axis=1)

# # show the result
df_new.head()
X = df.drop(list(set(cat_cols)), axis=1)

# Train
X_train = X[X.eval_set == 0]
y_train = X_train.pop('y'); 
X_train = X_train.drop(['eval_set', 'ID'], axis=1)

# Test
X_test = X[X.eval_set == 1]
X_test = X_test.drop(['y', 'eval_set', 'ID'], axis=1)

# Base score
y_mean = y_train.mean()
# Shapes

print('Shape X_train: {}\nShape X_test: {}'.format(X_train.shape, X_test.shape))

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)

# from sklearn.linear_model import LassoCV
# from sklearn.linear_model import Lasso
# from sklearn.linear_model import Ridge
# from sklearn.model_selection import KFold
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import mean_absolute_error,r2_score

# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(pd.DataFrame(X_train),y_train,test_size = 0.30, random_state=50)

# lasso = Lasso(random_state=0)
# alphas = 10**np.linspace(1.2,-3,50)

# k_fold = KFold(5)

# lasso_r2_score = []
# for i in range(0,len(alphas)):
#     print(i)
#     r2_score_k = []
#     for k, (train, val) in enumerate(k_fold.split(X_train, Y_train)):
#         clf = Lasso(alpha=alphas[i])
#         clf.fit(X_train.iloc[train], Y_train.iloc[train])
#         y_pred_val = clf.predict(X_train.iloc[val])
#         k = r2_score(Y_train.iloc[val],y_pred_val)
#         r2_score_k.append(k)
#     m = np.mean(r2_score_k)
#     lasso_r2_score.append(m)

# l = pd.DataFrame(lasso_r2_score)
# l['alphas'] = alphas
# l.columns = ['lasso_r2_score','alphas']
# print('Best chosen alpha value on cross validation(Lasso) is :',alphas[l['lasso_r2_score'].idxmax()])

# # Lasso
# from sklearn.metrics import r2_score
# lasso = Lasso(alpha= 0.023 )
# lasso.fit(X,Y)

# #train
# y_pred_test = lasso.predict(X_actual_test)
# print('r2_score : %0.2f'%r2_score(Y,y_pred_test))


# Test
# Make predictions using the testing set
# y_pred_test = lasso.predict(X_actual_test)
# final_sub = pd.DataFrame(test['ID'])
# final_sub['y'] = list(y_pred_test)
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# xgb_preds = []

# sub = pd.DataFrame()
# test_id = test['ID']
# sub['id'] = test['ID']
# sub['y'] = np.zeros_like(test_id)

# K = 5
# kf = KFold(n_splits = K, random_state = 1, shuffle = True)

# # X_new = new_features

# for i,(train_index,val_index) in enumerate(kf.split(X_train,np.array(y_train))):
#     print('Fold : %d'%i)
# #     print(train_index)
# #     print(val_index)
    
#     x_train, x_valid = X_train[train_index], X_train[val_index]
#     y_train, y_valid = y_train[train_index], y_train[val_index]

#     xgb_params= {
#                     'learning_rate': 0.03,
#                     'objective' : 'reg:linear',
#                     'max_depth' : 4,
#                     'metric': 'rmse',
#                     'subsample': 0.9,
#                     'colsample_bytree': 0.9,
#                     'random_state': 1,
#                     'num_leaves': 15
#                  }
    
#     d_train = xgb.DMatrix(x_train, y_train)
#     d_valid = xgb.DMatrix(x_valid, y_valid)
#     d_test = xgb.DMatrix(np.array(X_test))

#     def xgb_r2_score(preds, dtrain):
#         labels = dtrain.get_label()
#         return 'r2', r2_score(labels, preds)
    
#     watchlist = [(d_train, 'train'), (d_valid, 'valid')]

#     mdl = xgb.train(xgb_params, d_train, 1000, watchlist, early_stopping_rounds=70, feval=xgb_r2_score, maximize=True, verbose_eval=1)

#     print('[Fold %d/%d Prediciton:]' % (i + 1, K))
# #     # Predict on our test data
#     p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)
#     sub['y'] += p_test/K  

xgb_params = {
    'n_trees': 500, 
    'eta': 0.005,
    'max_depth': 3,
    'subsample': 0.9,
    'colsample_bytree': 0.6,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': np.log(y_mean),
    'silent': 1
}

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(X_train, np.array(np.log(y_train)))
dtest = xgb.DMatrix(X_test)

# evaluation metric
def the_metric(y_pred, y):
    y_true = y.get_label()
    return 'r2', r2_score(y_true, y_pred)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=2000, 
                   nfold = 3,
                   early_stopping_rounds=50,
                   feval=the_metric,
                   verbose_eval=100, 
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print('num_boost_rounds=' + str(num_boost_rounds))

# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

# Predict on trian and test
y_train_pred = np.exp(model.predict(dtrain))
y_pred = np.exp(model.predict(dtest))

print('First 5 predicted test values:', y_pred[:5])

output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('sub_16_encoded.csv', index=False)
# from lightgbm import LGBMRegressor

# lgb_params = {
#     'learning_rate': 0.03,
#     'metric': 'rmse',
#     'subsample': 0.9,
#     'colsample_bytree': 0.9,
#     'random_state': 1,
#     'num_leaves': 31
# }
# X_train.shape

# sub.to_csv('sub14_xgb_cv.csv',index = False)