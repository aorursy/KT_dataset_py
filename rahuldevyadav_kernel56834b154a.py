# Importing graph lib.

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



# lib to read data and mathematical operations 

import pandas as pd

import numpy as np



# Libaries for featureengg. and ML

# Preprocessing Scaling features

from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler,Normalizer

# library for feature selection

from sklearn.model_selection import StratifiedKFold,RepeatedKFold,RepeatedStratifiedKFold

from sklearn.feature_selection import RFECV

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

# for ml model to be used for feature selection

from sklearn.linear_model import LogisticRegression,Lasso

# metrics for evaluate the prediction

from sklearn.metrics import roc_auc_score,make_scorer



# for training model

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, cross_val_score



# to Ignore warnings

import warnings; warnings.simplefilter('ignore')

import pickle
# reading the training data

data = pd.read_csv('../input/dont-overfit-ii/train.csv')

data.head()
# Creating label matrics

target = data.target.values.astype(int)

# removing id and target coulumns for preparing training data 

train_data = data.drop(columns=['id','target'])
# reading and creating test data

test_data = pd.read_csv('../input/dont-overfit-ii/test.csv')

test_data = test_data.drop(columns='id')
# using ref [1]

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.axis('equal')

langs = ['Train Data', 'Test Data']

students = [train_data.shape[0],test_data.shape[0]]

ax.pie(students, labels = langs,autopct='%1.2f%%')

plt.title('Train and Test data distribution')

plt.show()

print('There are "',data.isna().sum().sum(),'" missing values in data')
print("Total training samples \t: {}".format(train_data.shape[0]))

print("Total features \t\t: {}".format(train_data.shape[1]))
print('All features have {} data type'.format(list(train_data.dtypes)[0]))
# ref [2]

# finding correlation 

corr = train_data.corr().abs()

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))



# Find index of feature columns with correlation greater than 0.5 and 0.2

top_50 = [column for column in upper.columns if any(upper[column] > 0.50)]

top_20 = [column for column in upper.columns if any(upper[column] > 0.20)]





print('There {:0.2f}% feature have more than 0.5 correlation and {}% features have more than 0.2 correlation'.format((len(top_50)/train_data.shape[1])*100,(len(top_20)/train_data.shape[1])*100))
corrs = data.corr().abs().unstack().sort_values(kind="quicksort").reset_index()

corrs = corrs[corrs['level_0'] == 'target']

corrs.tail(5)
target_corr = list(corrs.level_1[-31:-1].values)
# target_corr.remove('id')
len(target_corr)
labels, counts = np.unique(target,return_counts=True)

print('Total class labels : {}\n'.format(len(labels)))
data['target'].value_counts()
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.axis('equal')

langs = ['0', '1']

students = [183,67]

ax.pie(students, labels = langs,autopct='%1.2f%%')

plt.title('Target Distribution')

plt.show()
# using ref [3]

print('\t\t\t\tFeature Distribution')

print('-'*100)



plt.figure(figsize=(30, 200))

plt.subplots_adjust(hspace=0.5,wspace=0.1)

for i, col in enumerate(data.columns[2:]):

    plt.subplot(50, 6, i + 1)

    sns.kdeplot(data.loc[data['target'] == 1, col], shade=True, label='1')

    sns.kdeplot(data.loc[data['target'] == 0, col], shade=True, label='0')

    plt.title(col)
def feature_std_norm(train,test,col,scaler=None):

    """

    This function is used to scale features of train and test data.

    INPUT:

        train : Training data (data-frame)

        test  : Test data (data-frame)

        scaler: Scaling method (sklearn preprocessing methods) to be used. The options are as follows:

                a. minMax        : MinMaxScaler 

                b. robustScaler  : RobustScaler 

                c. stdScaler     : StandardScaler

                d. normalization : Normalizer

                e. None(default) : No Scaler selected

    OUTPUT:

        Xtrain : Scaled training data

        Xtest  : Scaled testing data

    """

    if scaler =='minMax':

        print('Scaling Data Using MinMax Scaler ...')

        mm_scaler = MinMaxScaler()

        mm_scaler = mm_scaler.fit(train[col])

        Xtrain = mm_scaler.transform(train[col])

        Xtest = mm_scaler.transform(test[col])

    elif scaler =='robustScaler':

        print('Scaling Data Using RobustScaler Scaler ...')

        rs_scaler = RobustScaler()

        rs_scaler = rs_scaler.fit(train[col])

        Xtrain = rs_scaler.transform(train[col])

        Xtest = rs_scaler.transform(test[col])

    elif scaler =='stdScaler':

        print('Scaling Data Using StandardScaler Scaler ...')

        ss_scaler = StandardScaler()

        ss_scaler = ss_scaler.fit(train[col])

        Xtrain = ss_scaler.transform(train[col])

        Xtest = ss_scaler.transform(test[col])

    elif scaler =='normalization':

        print('Scaling Data Using Normalizing Scaler ...')

        n_scaler = Normalizer()

        n_scaler = n_scaler.fit(train[col])

        Xtrain = n_scaler.transform(train[col])

        Xtest = n_scaler.transform(test[col])

    else:

        print('No scaler selected...')

        

        return train,test

    return pd.DataFrame(Xtrain,columns=col),pd.DataFrame(Xtest,columns=col)
col =list(train_data.columns)
train_mm,test_mm=feature_std_norm(train_data,test_data,col,scaler='minMax')

train_rs,test_rs=feature_std_norm(train_data,test_data,col,scaler='robustScaler')

train_ss,test_ss=feature_std_norm(train_data,test_data,col,scaler='stdScaler')

train_nn,test_nn=feature_std_norm(train_data,test_data,col,scaler='normalization')
features = data[['target','0']]

features['mm_0'] = train_mm['0']

features['rs_0'] = train_rs['0']

features['ss_0'] = train_ss['0']

features['nn_0'] = train_nn['0']
# features.head()

# using ref [3]

print('\t\t\t\tScaled Feature Distribution')

print('-'*100)



plt.figure(figsize=(30, 7))

plt.subplots_adjust(hspace=0.5,wspace=0.1)

for i, col in enumerate(features.columns[1:]):

    plt.subplot(1, 5, i + 1)

    sns.kdeplot(features.loc[features['target'] == 1, col], shade=True, label='1')

    sns.kdeplot(features.loc[features['target'] == 0, col], shade=True, label='0')

    plt.title(col)
# robust_roc_auc = make_scorer(scoring_roc_auc)

from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, r2_score

from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from xgboost import XGBClassifier
# https://www.kaggle.com/enespolat/grid-search-with-logistic-regression

param_model1 ={"C":[0.2, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37], "penalty":["l1"],

               'tol'   : [0.0001, 0.00011, 0.00009],'solver':['liblinear'],'max_iter':[500]}



#https://www.kaggle.com/tilii7/hyperparameter-grid-search-with-xgboost

param_model3 = {'min_child_weight': [1,2,3],

                'learning_rate':[0.01,0.05,0.1,0.5,1],

                'colsample_bytree': [0.2,0.4,0.5],

                'max_depth': [2,3,4,5],

                'n_estimators':[5,10,20,50,100]}
random_state=234587

model1 = LogisticRegression(class_weight='balanced',random_state=random_state) 

model3 = XGBClassifier(objective='binary:logistic',random_state=random_state)
# data

X = train_data.values

y = target

test = test_data.values
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, r2_score, make_scorer
n_fold = 20

folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)

repeated_folds = RepeatedStratifiedKFold(n_splits=20, n_repeats=20, random_state=42)

r2_threshold = 0.185

from imblearn.over_sampling import SMOTE



def train_model(X, X_tests, y, params, text,folds=folds, averaging='usual', model=None,r2_threshold=r2_threshold,feature_selection=True):

    prediction = np.zeros(len(X_tests))

    scores = []

    

    if feature_selection:

        grid_search = GridSearchCV(model, param_grid=params, verbose=0, n_jobs=-1, scoring='roc_auc', cv=20)

        grid_search.fit(X,y)

        feature_selector = RFECV(grid_search.best_estimator_, min_features_to_select=12, scoring='roc_auc',

                                 step=15, verbose=0, cv=20, n_jobs=-1)



        

    print("~"*120)

    print('\t\t\t\t\t',text)

    print('-'*120,'\n')

    print('\t\tVal. scores for each folds and stacking status...')

    print('-'*120)

    print("  fold   | val_mse  |  val_mae  |  val_roc  |  val_r2    ")

    print("----------------------------------------------------------")

    

        

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):

        # print('Fold', fold_n, 'started at', time.ctime())

        X_train, X_valid = X[train_index], X[valid_index]

        y_train, y_valid = y[train_index], y[valid_index]

        

        if feature_selection:

            feature_selector.fit(X_train, y_train)

            X_train  = feature_selector.transform(X_train)

            X_valid  = feature_selector.transform(X_valid)

            X_test   = feature_selector.transform(X_tests)

            model    = feature_selector.estimator_

        

        

        grid_search = GridSearchCV(model, param_grid=params, n_jobs=-1, scoring='roc_auc', cv=20)

        grid_search.fit(X_train, y_train)

#         lsvc = 

            

        model = grid_search.best_estimator_

#         print(model)

        model.fit(X_train, y_train)

        y_pred_valid = model.predict(X_valid).reshape(-1,)

#         score = roc_auc_score(y_valid, y_pred_valid)

        # print(f'Fold {fold_n}. AUC: {score:.4f}.')

        # print('')

        

        val_mse = mean_squared_error(y_valid, y_pred_valid)

        val_mae = mean_absolute_error(y_valid, y_pred_valid)

        val_roc = roc_auc_score(y_valid, y_pred_valid)

        val_r2  = r2_score(y_valid, y_pred_valid)



#         y_pred = model.predict_proba(X_test)[:, 1]

            

#         oof[valid_index] = y_pred_valid.reshape(-1,)

#         scores.append(roc_auc_score(y_valid, y_pred_valid))

#         print('ROC {}: {:.4f}.'.format(fold_n, roc_auc_score(y_valid, y_pred_valid)))

#         if averaging == 'usual':

#             prediction += y_pred

#         elif averaging == 'rank':

#             prediction += pd.Series(y_pred).rank().values

        if val_r2 > r2_threshold:

            message = '<-- OK - Stacking'

            y_pred = model.predict_proba(X_test)[:, 1]

#             oof[valid_index] = y_pred_valid.reshape(-1,)

            score = roc_auc_score(y_valid, y_pred_valid)

            

            scores.append(roc_auc_score(y_valid, y_pred_valid))

            if averaging == 'usual':

                prediction += y_pred

            elif averaging == 'rank':

                prediction += pd.Series(y_pred).rank().values

        else:

            message = '<-- skipping'

            

        print("{:2}       | {:.4f}   |  {:.4f}   |  {:.4f}   |  {:.4f}    \t{}   ".format(fold_n, val_mse, val_mae, val_roc, val_r2,message))

    

    

    

    prediction /= n_fold

    if prediction.sum()>0:

        print('-'*50)

        print('CV mean score of model after folds: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

        print()

        sub = pd.read_csv('../input/dont-overfit-ii/sample_submission.csv')

        sub['target']=prediction

        sub.to_csv('{}.csv'.format(text),index=False)

        

        print('\n Result : Created Submission file - "{}.csv"'.format(text))

        print('_'*120,'\n\n')

    else:

        print('\n Results Discarding the current ML agorithm - because Threshod cretria not meet')

        print('_'*120,'\n\n')

    

    return prediction, scores
_,s1a = train_model(train_rs.values,test_rs.values,y, text='Experiment-1a',params=param_model1, model=model1,feature_selection=True)



_,s1b = train_model(train_rs[target_corr].values,test_rs[target_corr].values,y, text='Experiment-1b',params=param_model1, model=model1,feature_selection=True)

def with_statistics(X):

    statistics = pd.DataFrame()

    statistics['mean']   = X.mean(axis=1)

    statistics['kurt']   = X.kurt(axis=1)

    statistics['mad']    = X.mad(axis=1)

    statistics['median'] = X.median(axis=1)

    statistics['max']    = X.max(axis=1)

    statistics['min']    = X.min(axis=1)

    statistics['skew']   = X.skew(axis=1)

    statistics['sem']    = X.sem(axis=1)

    sin_temp = np.sin(X)

    cos_temp = np.cos(X)

    tan_temp = np.tan(X)

    statistics['mean_sin'] = np.mean(sin_temp, axis=1)

    statistics['mean_cos'] = np.mean(cos_temp, axis=1)

    statistics['mean_tan'] = np.mean(tan_temp, axis=1)

    # Hyperbolic FE

    sinh_temp = np.sinh(X)

    cosh_temp = np.cosh(X)

    tanh_temp = np.tanh(X)

    statistics['mean_sinh'] = np.mean(sin_temp, axis=1)

    statistics['mean_cosh'] = np.mean(cos_temp, axis=1)

    statistics['mean_tanh'] = np.mean(tan_temp, axis=1)

    # Exponents FE

    exp_temp = np.exp(X)

    expm1_temp = np.expm1(X)

    exp2_temp = np.exp2(X)

    statistics['mean_exp'] = np.mean(exp_temp, axis=1)

    statistics['mean_expm1'] = np.mean(expm1_temp, axis=1)

    statistics['mean_exp2'] = np.mean(exp2_temp, axis=1)

    # Polynomial FE

    # X**2

    statistics['mean_x2'] = np.mean(np.power(X,2), axis=1)

    # X**3

    statistics['mean_x3'] = np.mean(np.power(X,3), axis=1)

    # X**4

    statistics['mean_x4'] = np.mean(np.power(X,4), axis=1)

    

    from sklearn.neighbors import NearestNeighbors

    neigh = NearestNeighbors(5, n_jobs=-1)

    neigh.fit(X)



    dists, _ = neigh.kneighbors(X, n_neighbors=5)

    dists = np.delete(dists, 0, 1)

    statistics['minDist'] = dists.mean(axis=1)

    statistics['maxDist'] = dists.max(axis=1)

    statistics['meanDist'] = dists.min(axis=1)



    X = pd.concat([X, statistics], axis=1)

    return X
train_1 = with_statistics(train_rs)

test_1 = test = with_statistics(test_rs)



train_2 = with_statistics(train_ss)

test_2 =  with_statistics(test_ss)



train_3 = with_statistics(train_rs[target_corr])

test_3 = test = with_statistics(test_rs[target_corr])



train_4 = with_statistics(train_ss[target_corr])

test_4 =  with_statistics(test_ss[target_corr])



train_1.head()
train_3.head()
_,s4a = train_model(train_1.values,test_1.values,y, text='Experiment-4a',params=param_model1, model=model1,feature_selection=True)



_,s4b = train_model(train_3.values,test_3.values,y, text='Experiment-4b',params=param_model1, model=model1,feature_selection=True)
