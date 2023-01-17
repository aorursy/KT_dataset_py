import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score as AUC

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score 

from sklearn.preprocessing import LabelEncoder, LabelBinarizer

from sklearn.model_selection import cross_val_score



from scipy import stats

import seaborn as sns

from copy import deepcopy



#model fitting

import xgboost as xgb

import pickle

import sys

import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, make_scorer

from scipy.sparse import csr_matrix, hstack

from sklearn.model_selection import KFold, train_test_split

from xgboost import XGBRegressor



%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.shape
print ('First 20 columns:'), list(train.columns[:20])
print ('Last 20 columns:'), list(train.columns[-20:])
train.describe()
train['loss'].describe()
pd.isnull(train).values.any()
train.info()
cat_features = list(train.select_dtypes(include=['object']).columns)

print ("Categorical features:",len(cat_features))
cont_features = [cont for cont in list(train.select_dtypes(

                 include=['float64', 'int64']).columns) if cont not in ['loss', 'id']]

print ("Continuous features:", len(cont_features))

print(cont_features)
id_col = list(train.select_dtypes(include=['int64']).columns)

print ("A column of int64: ", id_col)
cat_uniques = []

for cat in cat_features:

    cat_uniques.append(len(train[cat].unique()))

    

uniq_values_in_categories = pd.DataFrame.from_items([('cat_name', cat_features), ('unique_values', cat_uniques)])
uniq_values_in_categories.head()
fig, (ax1, ax2) = plt.subplots(1,2)

fig.set_size_inches(16,5)

ax1.hist(uniq_values_in_categories.unique_values, bins=50)

ax1.set_title('Amount of categorical features with X distinct values')

ax1.set_xlabel('Distinct values in a feature')

ax1.set_ylabel('Features')

ax1.annotate('A feature with 326 vals', xy=(322, 2), xytext=(200, 38), arrowprops=dict(facecolor='black'))



ax2.set_xlim(2,30)

ax2.set_title('Zooming in the [0,30] part of left histogram')

ax2.set_xlabel('Distinct values in a feature')

ax2.set_ylabel('Features')

ax2.grid(True)

ax2.hist(uniq_values_in_categories[uniq_values_in_categories.unique_values <= 30].unique_values, bins=30)

ax2.annotate('Binary features', xy=(3, 71), xytext=(7, 71), arrowprops=dict(facecolor='black'))
# Another option is to use Series.value_counts() method, but its

# output is not that nice



uniq_values = uniq_values_in_categories.groupby('unique_values').count()

uniq_values = uniq_values.rename(columns={'cat_name': 'categories'})

uniq_values.sort_values(by='categories', inplace=True, ascending=False)

uniq_values.reset_index(inplace=True)

print (uniq_values)
plt.figure(figsize=(16,8))

plt.plot(train['id'], train['loss'])

plt.title('Loss values per id')

plt.xlabel('id')

plt.ylabel('loss')

plt.legend()

plt.show()
stats.mstats.skew(train['loss']).data
stats.mstats.skew(np.log(train['loss'])).data
fig, (ax1, ax2) = plt.subplots(1,2)

fig.set_size_inches(16,5)

ax1.hist(train['loss'], bins=50)

ax1.set_title('Train Loss target histogram')

ax1.grid(True)

ax2.hist(np.log(train['loss']), bins=50, color='g')

ax2.set_title('Train Log Loss target histogram')

ax2.grid(True)

plt.show()
train[cont_features].hist(bins=50, figsize=(16,12))
plt.subplots(figsize=(16,9))

correlation_mat = train[cont_features].corr()

sns.heatmap(correlation_mat, annot=True)
# Simple data preparation



train_d = train.drop(['id','loss'], axis=1)

test_d = test.drop(['id'], axis=1)



# To make sure we can distinguish between two classes

train_d['Target'] = 1

test_d['Target'] = 0



# We concatenate train and test in one big dataset

data = pd.concat((train_d, test_d))



# We use label encoding for categorical features:

data_le = deepcopy(data) # creates a same copy which can be used for other operations without 

#modifying the dataframe



#`data label encoding`

for c in range(len(cat_features)):

    data_le[cat_features[c]] = data_le[cat_features[c]].astype('category').cat.codes



# We use one-hot encoding for categorical features:

data = pd.get_dummies(data=data, columns=cat_features)
# randomize before splitting them up into train and test sets

data = data.iloc[np.random.permutation(len(data))]

data.reset_index(drop = True, inplace = True)



x = data.drop(['Target'], axis = 1)

y = data.Target



train_examples = 100000



x_train = x[:train_examples]

x_test = x[train_examples:]

y_train = y[:train_examples]

y_test = y[train_examples:]
x.head()

y.head()
# Logistic Regression:

clf = LogisticRegression()

clf.fit(x_train, y_train)

pred = clf.predict_proba(x_test)[:,1]

auc = AUC(y_test, pred)

print("Logistic Regression AUC: ",auc)



# Random Forest, a simple model (100 trees) trained in parallel

clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

clf.fit(x_train, y_train)

pred = clf.predict_proba(x_test)[:,1]

auc = AUC(y_test, pred)

print ("Random Forest AUC: ",auc)



# Finally, CV our results (a very simple 2-fold CV):

scores = cross_val_score(LogisticRegression(), x, y, scoring='roc_auc', cv=2) 

print ("Mean AUC: {:.2%}, std: {:.2%} \n",scores.mean(),scores.std())
# A possibility to use pretrained models to limit the computational time.

USE_PRETRAINED = True
train['log_loss'] = np.log(train['loss'])
train['log_loss'].sample(5)
features = [x for x in train.columns if x not in ['id','loss', 'log_loss']]



cat_features = [x for x in train.select_dtypes(

        include=['object']).columns if x not in ['id','loss', 'log_loss']]

num_features = [x for x in train.select_dtypes(

        exclude=['object']).columns if x not in ['id','loss','log_loss']]



print ("Categorical features:", len(cat_features))

print ("Numerical features:", len(num_features))
ntrain = train.shape[0]



train_x = train[features]

train_y = train['log_loss'] #target variable

test_x = test[features]



for c in range(len(cat_features)):

    train_x[cat_features[c]] = train_x[cat_features[c]].astype('category').cat.codes

    test_x[cat_features[c]] = test_x[cat_features[c]].astype('category').cat.codes

print ("Xtrain:", train_x.shape)

print ("ytrain:", train_y.shape)

print("Xtest:", test_x.shape)
train_x.head()
test_x.head()
pred_model = xgb.XGBRegressor() #As target variable is a continuous variable

pred_model.fit(train_x, train_y)

y_pred = pred_model.predict(test_x)

predictions = [value for value in y_pred]



#loss-prediction log-transformed

print(predictions)



# evaluate predictions

# accuracy = accuracy_score(y_test, predictions)

# print("Accuracy: %.2f%%" % (accuracy * 100.0))
loss_value = np.exp(predictions)

print(loss_value[:5])
train['loss'].sample(5)
def xg_eval_mae(yhat, dtrain):

    y = dtrain.get_label()

    return 'mae', mean_absolute_error(np.exp(y), np.exp(yhat))
dtrain = xgb.DMatrix(train_x, train['log_loss'])
#We use some average set of parameters to make XGBoost work:

xgb_params = {

    'seed': 0,

    'eta': 0.1,

    'colsample_bytree': 0.5,

    'silent': 1,

    'subsample': 0.5,

    'objective': 'reg:linear',

    'max_depth': 5,

    'min_child_weight': 3

}

# to be explored in detail for tuning and optimizing the model
bst_cv1 = xgb.cv(xgb_params, dtrain, num_boost_round=100, nfold=3, seed=0, 

                    feval=xg_eval_mae, maximize=False, early_stopping_rounds=10)



print ('CV score:', bst_cv1.iloc[-1,:]['test-mae-mean'])

#bst_cv1