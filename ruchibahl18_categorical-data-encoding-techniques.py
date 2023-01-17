!pip install logitboost
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from logitboost import LogitBoost

import seaborn as sns

from category_encoders import TargetEncoder, HashingEncoder, LeaveOneOutEncoder

from sklearn.naive_bayes import ComplementNB, GaussianNB, MultinomialNB, BernoulliNB

from sklearn.preprocessing import MinMaxScaler,StandardScaler

import string



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/cat-in-the-dat/train.csv')

test = pd.read_csv('../input/cat-in-the-dat/test.csv')
train.head()
train.shape
train.isna().sum().sort_values(ascending = False)
train.describe(include='all')
# Get list of categorical variables

s = (train.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
target = train['target']
replace_xor = lambda x: 'xor' if x in xor_values else x
print(set(train['ord_4'].unique()))

print(set(test['ord_4'].unique()))

columns_to_test = ['ord_5', 'ord_4', 'ord_3']

for column in columns_to_test:

    xor_values = set(train[column].unique()) ^ set(test[column].unique())

    if xor_values:

        print('Column', column, 'has', len(xor_values), 'XOR values')

        train[column] = train[column].apply(replace_xor)

        test[column] = test[column].apply(replace_xor)

    else:

        print('Column', column, 'has no XOR values')
#train["ord_5a"]=train["ord_5"].str[0]

#train["ord_5b"]=train["ord_5"].str[1]

#train.drop(['ord_5'], axis=1, inplace = True)

#test["ord_5a"]=test["ord_5"].str[0]

#test["ord_5b"]=test["ord_5"].str[1]

#test.drop(['ord_5'], axis=1, inplace = True)
def date_cyc_enc(df, col, max_vals):

    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)

    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)

    return df
map_to_ascii_index = lambda x: string.ascii_letters.index(x)
replace_xor = lambda x: 'xor' if x in xor_values else x
train['merge_col1'] =  train[['nom_0', 'nom_1']].apply(lambda x: ''.join(x), axis=1)

test['merge_col1'] =  test[['nom_0', 'nom_1']].apply(lambda x: ''.join(x), axis=1)



train['merge_col2'] =  train[['nom_1', 'nom_2']].apply(lambda x: ''.join(x), axis=1)

test['merge_col2'] =  test[['nom_1', 'nom_3']].apply(lambda x: ''.join(x), axis=1)



train['merge_col3'] =  train[['nom_2', 'nom_3']].apply(lambda x: ''.join(x), axis=1)

test['merge_col3'] =  test[['nom_2', 'nom_3']].apply(lambda x: ''.join(x), axis=1)



train['merge_col4'] =  train[['nom_3', 'nom_4']].apply(lambda x: ''.join(x), axis=1)

test['merge_col4'] =  test[['nom_3', 'nom_4']].apply(lambda x: ''.join(x), axis=1)
# Binary encoding

train['bin_3'] = [0 if x == 'F' else 1 for x in train['bin_3']]

train['bin_4'] = [0 if x == 'N' else 1 for x in train['bin_4']]



#Hard coded Label encoding

train['ord_1'] = [0 if x == 'Novice' else 1 if x == 'Contributor' else 2 if x == 'Expert' else 3 if x == 'Master' else 4 for x in train['ord_1']]

train['ord_2'] = [0 if x == 'Freezing' else 1 if x == 'Cold' else 2 if x == 'Warm' else 3 if x == 'Hot' else 4 if x == 'Boiling Hot' else 5 for x in train['ord_2']]



# Label encoding via LabelEncoder class

label_encoder = LabelEncoder()

train['ord_3'] = label_encoder.fit_transform(train['ord_3'])

test['ord_3'] = label_encoder.transform(test['ord_3'])



train['ord_4'] = label_encoder.fit_transform(train['ord_4'])

test['ord_4'] = label_encoder.transform(test['ord_4'])



train['ord_5'] = label_encoder.fit_transform(train['ord_5'])

test['ord_5'] = label_encoder.transform(test['ord_5'])



#train['ord_5b'] = label_encoder.fit_transform(train['ord_5b'])



#train['ord_3'] = train['ord_3'].apply(map_to_ascii_index)

#train['ord_4'] = train['ord_4'].apply(map_to_ascii_index)

#train['ord_5'] = label_encoder.fit_transform(train['ord_5'])





#train = train.drop('ord_5b', axis=1)



train = date_cyc_enc(train, 'day', 7)

train = date_cyc_enc(train, 'month', 12)

train.drop(['day', 'month'], axis=1, inplace = True)



#Leave one out encoding high cardinal variables

high_cardinal_vars = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']



#trgt_encoder = TargetEncoder(cols=high_cardinal_vars, smoothing=0, return_df=True)

#hashing_encoder = HashingEncoder(cols = high_cardinal_vars)

loo_encoder = LeaveOneOutEncoder(cols=high_cardinal_vars)

train = loo_encoder.fit_transform(train.drop(['target'], axis = 1), train['target'])



# Same for test data

test['bin_3'] = [0 if x == 'F' else 1 for x in test['bin_3']]

test['bin_4'] = [0 if x == 'N' else 1 for x in test['bin_4']]

test['ord_1'] = [0 if x == 'Novice' else 1 if x == 'Contributor' else 2 if x == 'Expert' else 3 if x == 'Master' else 4 for x in test['ord_1']]

test['ord_2'] = [0 if x == 'Freezing' else 1 if x == 'Cold' else 2 if x == 'Warm' else 3 if x == 'Hot' else 4 if x == 'Boiling Hot' else 5 for x in test['ord_2']]



#test = test.drop('ord_5b', axis=1)



#test['ord_3'] = test['ord_3'].apply(map_to_ascii_index)

#test['ord_4'] = test['ord_4'].apply(map_to_ascii_index)

#test['ord_5b'] = label_encoder.fit_transform(test['ord_5b'])



#for column in ['ord_3', 'ord_4', 'ord_5a']:

#    train[column] = train[column].apply(map_to_ascii_index)

#    test[column] = test[column].apply(map_to_ascii_index)



#For cyclic data we convert it into sin and cosine values

test = date_cyc_enc(test, 'day', 7)

test = date_cyc_enc(test, 'month', 12)

test.drop(['day', 'month'], axis=1, inplace = True)



test = loo_encoder.transform(test)

# One Hot encoding other nominal columns

train_df = pd.get_dummies(train, drop_first=True)



# Same for test data

#test_modified = test.drop(nominal_variables, axis = 1)

test_df = pd.get_dummies(test, drop_first=True)
print(train_df.shape)

print(test_df.shape)
cor = train_df.corr()
f, ax = plt.subplots(figsize=(25, 25))

sns.heatmap(cor, annot=False, ax=ax)
X = train_df.drop(['id'], axis=1)

y = target
x=y.value_counts()

plt.bar(x.index,x)

plt.gca().set_xticks([0,1])

plt.title('distribution of target variable')

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=1)
sm = SMOTE(kind = "regular")

X_tr,y_tr = sm.fit_sample(X_train,y_train)

def draw_roc( actual, probs ):

    fpr, tpr, thresholds = roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()
# Function for comparing different approaches

def score_dataset(X_train, X_test, y_train, y_test, model):

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    draw_roc(y_test, preds)

    return roc_auc_score(y_test, preds)
scaler = StandardScaler()

X_tr = scaler.fit_transform(X_tr)

X_test = scaler.transform(X_test)
lr = LogisticRegression(random_state=0, solver = 'lbfgs')

print('AUC score with Logistic Regression :- ', score_dataset(X_tr, X_test, y_tr, y_test, lr))
rft = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

print('AUC score with RandomForest :- ', score_dataset(X_tr, X_test, y_tr, y_test, rft))
dt = DecisionTreeClassifier(random_state=0)

print('AUC score with Decision Tree :- ', score_dataset(X_tr, X_test, y_tr, y_test, dt))
gaussianNB = GaussianNB(priors=None, var_smoothing=1e-09)

print('AUC score with gausian Naive Bayes :- ', score_dataset(X_tr, X_test, y_tr, y_test, gaussianNB))
# specify number of folds for k-fold CV

n_folds = 5



# parameters to build the model on

param_grid = {

    'max_depth': range(1, 5),

    'min_samples_leaf': range(25, 175, 50),

    'min_samples_split': range(50, 150, 50)

}
# uncomment if you want to see hyper parameter tuning. Although it takes some good amount of time

'''

# instantiate the model

dt = DecisionTreeClassifier()



# fit tree on training data

grid_search_dt = GridSearchCV(estimator = dt, param_grid = param_grid, 

                          cv = n_folds, verbose = 1, n_jobs = -1, scoring="roc_auc")

grid_search_dt.fit(X_tr, y_tr)

'''
# uncomment to see the results

'''

cv_results_dt = pd.DataFrame(grid_search_dt.cv_results_)

# printing the optimal accuracy score and hyperparameters

print("Decison Tree grid search Accuracy : ", grid_search_dt.best_score_)

print(grid_search_dt.best_estimator_)

'''
param_grid['n_estimators']  = range(50, 200, 50)
# uncomment if you want to see hyper parameter tuning. Although it takes some good amount of time

'''

# instantiate the model

rft = RandomForestClassifier(n_jobs= -1)



# fit tree on training data

grid_search_rft = GridSearchCV(estimator = rft, param_grid = param_grid, 

                          cv = n_folds, verbose = 1, n_jobs = -1, scoring="roc_auc")

grid_search_rft.fit(X_tr, y_tr)

'''
# uncomment to see the results

'''

cv_results_rft = pd.DataFrame(grid_search_rft.cv_results_)

# printing the optimal accuracy score and hyperparameters

print("Random Forest grid search Accuracy : ", grid_search_rft.best_score_)

# Best estimators

print(grid_search_rft.best_estimator_)

'''


logit_param_grid = {

    'C': [0.100, 0.150, 0.120, 0.125, 0.130, 0.135, 0.140, 0.145, 0.150]

}



logit_grid = GridSearchCV(estimator = lr, param_grid = logit_param_grid,

                          scoring='roc_auc', cv=5, n_jobs=-1, verbose=0)

logit_grid.fit(X_tr, y_tr)



best_C = logit_grid.best_params_['C']

# best_C = C = 0.125



print('Best C:', best_C)

rft = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                       max_depth=4, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=25, min_samples_split=50,

                       min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=-1,

                       oob_score=False, random_state=None, verbose=0,

                       warm_start=False)

print('AUC score with RandomForest :- ', score_dataset(X_tr, X_test, y_tr, y_test, rft))
dt = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=4,

                       max_features=None, max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=25, min_samples_split=50,

                       min_weight_fraction_leaf=0.0, presort=False,

                       random_state=None, splitter='best')

print('AUC score with Decision Tree :- ', score_dataset(X_tr, X_test, y_tr, y_test, dt))
lr = LogisticRegression(solver='lbfgs', random_state = 0, C=best_C)

print('AUC score with Losgistic Regression :- ', score_dataset(X_tr, X_test, y_tr, y_test, lr))
catboost = CatBoostClassifier(iterations=20,learning_rate=1,depth=2, custom_metric=['AUC'])

print('AUC score with Catboost classifier :- ', score_dataset(X_tr, X_test, y_tr, y_test, catboost))
lboost = LogitBoost(n_estimators=200, random_state=0)

print('AUC score with Logitboost classifier :- ', score_dataset(X_tr, X_test, y_tr, y_test, lboost))
xgboost = XGBClassifier(random_state=0)

print('AUC score with Xgboost classifier :- ', score_dataset(X_tr, X_test, y_tr, y_test, xgboost))
test_df.drop(['id'], axis=1, inplace = True)
test_df = scaler.transform(test_df)
X = scaler.transform(X)
lr = lr.fit(X,y)

dt = dt.fit(X,y)

rft = rft.fit(X,y)

catboost = catboost.fit(X,y)

lboost = lboost.fit(X,y)

gaussianNB = gaussianNB.fit(X,y)
y_test_final_lr = lr.predict(test_df)

y_test_final_dt = dt.predict(test_df)

y_test_final_rft = rft.predict(test_df)

y_test_NB = gaussianNB.predict(test_df)

y_test_final_catboost = catboost.predict(test_df)

y_test_final_logitboost = lboost.predict(test_df)
y_test_prob_lr = lr.predict_proba(test_df)[:, 1]

y_test_prob_dt = dt.predict_proba(test_df)[:, 1]

y_test_prob_rft = rft.predict_proba(test_df)[:, 1]

y_test_prob_NB = gaussianNB.predict_proba(test_df)[:, 1]

y_test_prob_catboost = catboost.predict_proba(test_df)[:, 1]

y_test_prob_logitboost = lboost.predict_proba(test_df)[:, 1]
submission = pd.DataFrame({

        "id": test["id"],

        "target": y_test_prob_lr

    })

submission.to_csv('LogisticRegression.csv',header=True, index=False)
submission = pd.DataFrame({

        "id": test["id"],

        "target": y_test_prob_dt

    })

submission.to_csv('DecisonTree.csv',header=True, index=False)
submission = pd.DataFrame({

        "id": test["id"],

        "target": y_test_prob_rft

    })

submission.to_csv('RandomForest.csv',header=True, index=False)
submission = pd.DataFrame({

        "id": test["id"],

        "target": y_test_prob_NB

    })

submission.to_csv('GaussianNB.csv',header=True, index=False)
submission = pd.DataFrame({

        "id": test["id"],

        "target": y_test_prob_catboost

    })

submission.to_csv('Catboost.csv',header=True, index=False)
submission = pd.DataFrame({

        "id": test["id"],

        "target": y_test_prob_logitboost

    })

submission.to_csv('Logitboost.csv',header=True, index=False)