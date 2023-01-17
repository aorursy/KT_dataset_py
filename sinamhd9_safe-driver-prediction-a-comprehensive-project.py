import warnings

warnings.filterwarnings("ignore")

import pandas as pd

import numpy as np



df_train = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv')

df_test = pd.read_csv('../input/porto-seguro-safe-driver-prediction/test.csv')

print('train data size', df_train.shape)

print('test data size', df_test.shape)

display(df_train.head())

display(df_test.head())
import seaborn as sns

import matplotlib.pyplot as plt



sns.set_style('white')

sns.set_palette("Paired")

sns.set(font_scale=1.5)

plt.figure()

sns.countplot(df_train['target'],palette='nipy_spectral')

plt.show()


print('Nan values =', df_train.isnull().sum().sum())

df_missing_train = pd.DataFrame({'column':df_train.columns, 'missing(%)':((df_train==-1).sum()/df_train.shape[0])*100})

df_missing_test = pd.DataFrame({'column':df_test.columns, 'missing(%)':((df_test==-1).sum()/df_test.shape[0])*100})



df_missing_train_nl = df_missing_train.nlargest(7, 'missing(%)')

df_missing_test_nl = df_missing_test.nlargest(7, 'missing(%)')

sns.set_palette(sns.color_palette('nipy_spectral'))

plt.figure(figsize=(16,6))

sns.barplot(data= df_missing_train_nl, x='column', y='missing(%)',palette='nipy_spectral')

plt.title('Missing values (%) in training set')

plt.show()

plt.figure(figsize=(16,6))

sns.barplot(data= df_missing_test_nl, x='column', y='missing(%)',palette='nipy_spectral')

plt.title('Missing values (%) in test set')

plt.show()
plt.figure(figsize=(18,10))

sns.heatmap(df_train==-1, cmap='gray', cbar=False)

plt.title('Missing values in training set')

plt.show()


features = df_train.columns.tolist()

cat_features = [c for c in features if 'cat' in c]

bin_features = [b for b in features if 'bin' in b]

cat_features_df = pd.DataFrame({'Categorical features': cat_features})

bin_features_df = pd.DataFrame({'Binary features': bin_features})



n_row = len(cat_features)

n_col = 2   

n_sub = 1   

fig = plt.figure(figsize=(20,50))

plt.subplots_adjust(bottom=-0.2,top=1.2)

for i in range(len(cat_features)):

    plt.subplot(n_row, n_col, n_sub)

    sns.countplot(x= df_train[cat_features[i]],palette='nipy_spectral')

    n_sub+=1

plt.show()



n_row = len(bin_features)

n_col = 2   

n_sub = 1      

fig = plt.figure(figsize=(20,50))

plt.subplots_adjust(bottom=-0.2,top=1.2)

for i in range(len(bin_features)):

    plt.subplot(n_row, n_col, n_sub)

    sns.countplot(x= df_train[bin_features[i]],palette='nipy_spectral')

    n_sub+=1   

plt.show()
int_features = df_train.select_dtypes(include=['int64']).columns.tolist()



ordinal_features = [o for o in int_features if ('cat' not in o and 'bin' not in o and 'id' not in o and 'target' not in o )]

ord_features_df = pd.DataFrame({'Ordinal features': ordinal_features})



n_row = len(ordinal_features)

n_col = 2   

n_sub = 1      

fig = plt.figure(figsize=(20,50))

plt.subplots_adjust(bottom=-0.2,top=1.2)

for i in range(len(ordinal_features)):

    plt.subplot(n_row, n_col, n_sub)

    sns.countplot(x= df_train[ordinal_features[i]],palette='nipy_spectral')

    n_sub+=1   

plt.show()
cont_features = df_train.select_dtypes(include=['float64']).columns.tolist()

cont_features_df = pd.DataFrame({'Numerical Continuous features': cont_features})

cont_features.remove('ps_calc_01')

cont_features.remove('ps_calc_02')

cont_features.remove('ps_calc_03')



n_row = len(cont_features)

n_col = 2   

n_sub = 1      

fig = plt.figure(figsize=(20,30))

plt.subplots_adjust(bottom=-0.2,top=1.2)

for i in range(len(cont_features)):

    plt.subplot(n_row, n_col, n_sub)

    sns.distplot(df_train[cont_features[i]], kde=False,color='Red')

    n_sub+=1   

plt.show()
a = df_train[cont_features]

plt.figure(figsize=(10,6))

sns.heatmap(a.corr(), annot=True,cmap='nipy_spectral')

plt.title('Pearson Correlation of continuous features')

plt.show()
X_train = df_train.drop(['id', 'target'], axis=1)

y_train = df_train['target']
from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, f1_score, roc_auc_score

from time import time



def metrics(true, preds):

    accuracy = accuracy_score(true, preds)

    recall = recall_score(true, preds)

    f1score = f1_score(true, preds)

    cf = confusion_matrix(true, preds)

    print('accuracy: {}, recall: {}, f1-score: {}'.format(accuracy, recall, f1score))

    print('Confusion matrix', cf)



def gini(true, preds):

    res = 2* roc_auc_score(true, preds) - 1

    return res



X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, random_state=42, test_size=0.2)



def test_clfs(clfs):

    for clf in clfs:

        print('------------------------------------------')

        start = time()

        clf = clf(random_state=42)

        clf.fit(X_tr, y_tr)

        y1_pred = clf.predict(X_te)

        y1_pred_prob = clf.predict_proba(X_te)

        print(str(clf), '\nresults:')

        metrics(y_te, y1_pred)

        print('gini score', gini(y_te, y1_pred_prob[:, 1]))

        end = time()

        print('Processing time', end-start,'s')



classifiers = [LGBMClassifier, XGBClassifier, RandomForestClassifier]

test_clfs(classifiers)
from imblearn.over_sampling import SMOTE

from imblearn.over_sampling import RandomOverSampler

from imblearn.under_sampling import RandomUnderSampler



sm_methods = [SMOTE,RandomUnderSampler,RandomOverSampler]

clf = LGBMClassifier(random_state=42)

for sm in sm_methods:

    print('------------------------------------------')

    sm = sm(random_state=42)

    start = time()

    X_train_resampled, y_train_resampled = sm.fit_sample(X_tr, y_tr)

    clf.fit(X_train_resampled, y_train_resampled)

    y_pred = clf.predict(X_te)

    y_pred_prob = clf.predict_proba(X_te)

    print(str(sm), 'results:')

    print('training resampled shape', X_train_resampled.shape)

    print('value counts in each class', y_train_resampled.value_counts())

    metrics(y_te, y_pred)

    gini_score = gini(y_te, y_pred_prob[:, 1])

    print('Normalized gini score', gini_score)

    end = time()

    print('Processing time', end-start,'s')
from sklearn.model_selection import StratifiedKFold, cross_val_predict



cv = StratifiedKFold(n_splits=5)

scores_gini = []



for train_index, validation_index in cv.split(X_train, y_train):

    train, val = X_train.iloc[train_index], X_train.iloc[validation_index]

    target_train, target_val = y_train.iloc[train_index], y_train.iloc[validation_index]

    clf = LGBMClassifier(random_state=42)

    clf.fit(train, target_train)

    y_val_pred = clf.predict(val)

    y_val_pred_prob = clf.predict_proba(val)

    gini_score = gini(target_val, y_val_pred_prob[:, 1])

    print('gini score:', gini_score)

    scores_gini.append(gini_score)

    

print('mean gini_score: {}'.format(np.mean(scores_gini)))

df_train = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv')

X_train = df_train.drop(['id', 'target'], axis=1)

y_train = df_train['target']

features = df_train.columns.tolist()

cat_features = [c for c in features if 'cat' in c]

X_train = pd.get_dummies(X_train, columns=cat_features, prefix_sep='_', drop_first=True)

# subsample = [0.5, 0.7, 0.9]

# num_leaves = [10, 12, 15, 20, 25, 30]

# learning_rate = [0.1, 0.15, 0.2]

# n_estimators = [50, 100, 150, 200]

# min_child_weight = [0.01, 0.1, 1, 10, 50, 100, 150, 200]

# min_child_samples = [5, 10, 15, 20, 25]



# params = dict(num_leaves=num_leaves, subsample=subsample, learning_rate=learning_rate,

#               n_estimators=n_estimators, min_child_samples=min_child_samples, min_child_weight=min_child_weight)

# clf = GridSearchCV(estimator=LGBMClassifier(random_state=42, n_jobs=-1), param_grid=params, scoring='roc_auc', cv=cv, n_jobs=-1,

#                    verbose=2)

# clf.fit(X_train, y_train)



# print('best params', clf.best_params_)

# print('best score', clf.best_score_)



# means = clf.cv_results_['mean_test_score']

# stds = clf.cv_results_['std_test_score']

# for mean, std, params in zip(means, stds, clf.cv_results_['params']):

#     print("%0.3f (+/-%0.03f) for %r"

#           % (mean, std * 2, params))

best_params = {'learning_rate': 0.1, 'min_child_samples': 5, 'min_child_weight': 100, 'n_estimators': 100, 'num_leaves': 20, 'subsample': 0.5}

print('best parameters', best_params)
scores_gini = []

for train_index, validation_index in cv.split(X_train, y_train):

    train, val = X_train.iloc[train_index], X_train.iloc[validation_index]

    target_train, target_val = y_train.iloc[train_index], y_train.iloc[validation_index]

    clf = LGBMClassifier(random_state=42, n_jobs=-1)

    clf.set_params(**best_params)

    clf.fit(train, target_train)

    y_val_pred = clf.predict(val)

    y_val_pred_prob = clf.predict_proba(val)

    gini_score = gini(target_val, y_val_pred_prob[:, 1])

    print('gini score:', gini_score)

    scores_gini.append(gini_score)



print('mean gini_score: {}'.format(np.mean(scores_gini)))
pd.set_option('display.max_columns', None)

y_preds = y_val_pred_prob[:,1]

results = pd.DataFrame({'target_true':target_val , 'target_pred': y_preds})

b = results.nlargest(10, columns='target_pred')

b2 = results.nlargest(1000, columns='target_pred')

c = results.nsmallest(1000, columns='target_pred')

c_counts_1 = (c['target_true']==1).sum()

b2_counts_1 =  (b2['target_true']==1).sum()

print('Numeber of 1s in 1000 smallest probabilities:', c_counts_1)

print('Number of 1s in 1000 largest probabilities:', b2_counts_1 )

display(b)

display(val.loc[b.index])
from sklearn.feature_selection import SelectFromModel



features = train.columns

importances = clf.feature_importances_

indices = np.argsort(clf.feature_importances_)[::-1]

imp_features= pd.DataFrame({'feature':features[indices], 'importance':importances[indices]})

display(imp_features)

imp_features_50smallest = imp_features.nsmallest(50, 'importance')

features_to_drop = imp_features_50smallest['feature'].tolist()

imp_features_10large = imp_features.nlargest(10, 'importance')

plt.figure(figsize=(15,8))

sns.barplot(data=imp_features_10large, x='feature', y='importance',palette='nipy_spectral')

plt.show()
print(imp_features_10large)

display(val.loc[b.index, imp_features_10large.feature])
features = X_train.columns.tolist()

cat_car_05 = [c for c in features if 'car_05_cat' in c]

calc_features = [c for c in features if 'calc' in c]

X_train.drop(calc_features, axis=1, inplace=True)

X_train.drop(cat_car_05, axis=1, inplace=True)

X_train['ps_reg_03'].replace(-1, X_train['ps_reg_03'].median(), inplace=True)

X_train['ps_car_14'].replace(-1, X_train['ps_car_14'].median(), inplace=True)



scores_gini = []

for train_index, validation_index in cv.split(X_train, y_train):

    train, val = X_train.iloc[train_index], X_train.iloc[validation_index]

    target_train, target_val = y_train.iloc[train_index], y_train.iloc[validation_index]

    clf = LGBMClassifier(random_state=42, n_jobs=-1)

    clf.set_params(**best_params)

    clf.fit(train, target_train)

    y_val_pred = clf.predict(val)

    y_val_pred_prob = clf.predict_proba(val)

    gini_score = gini(target_val, y_val_pred_prob[:, 1])

    print('gini score:', gini_score)

    scores_gini.append(gini_score)

    

print('mean gini_score: {}'.format(np.mean(scores_gini)))
y_preds = y_val_pred_prob[:,1]

results = pd.DataFrame({'target_true':target_val , 'target_pred': y_preds})

b = results.nlargest(10, columns='target_pred')

b2 = results.nlargest(1000, columns='target_pred')

c = results.nsmallest(1000, columns='target_pred')

c_counts_1 = (c['target_true']==1).sum()

b2_counts_1 =  (b2['target_true']==1).sum()

print('Numeber of 1s in 1000 smallest probabilities:', c_counts_1)

print('Number of 1s in 1000 largest probabilities:', b2_counts_1 )



# Same process to df_test before submission

df_test = pd.read_csv('../input/porto-seguro-safe-driver-prediction/test.csv')

X_test = df_test.drop('id', axis=1)

X_test = pd.get_dummies(X_test, columns=cat_features, prefix_sep='_', drop_first=True)

X_test.drop(calc_features, axis=1, inplace=True)

X_test.drop(cat_car_05, axis=1, inplace=True)

X_test['ps_reg_03'].replace(-1, X_test['ps_reg_03'].median(), inplace=True)

X_test['ps_car_14'].replace(-1, X_test['ps_car_14'].median(), inplace=True)

assert(X_train.shape[1]==X_test.shape[1])

clf.fit(X_train, y_train)
# Submission 

preds =  clf.predict_proba(X_test)[:,1]

df_subm = pd.read_csv('../input/porto-seguro-safe-driver-prediction/sample_submission.csv')

df_subm.loc[:,'target'] = preds

display(df_subm)

df_subm.to_csv('submission.csv', index=False)
