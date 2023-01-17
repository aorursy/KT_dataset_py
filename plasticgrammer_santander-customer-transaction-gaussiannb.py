import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

import gc,os,sys



sns.set_style('darkgrid')

pd.options.display.float_format = '{:,.3f}'.format



print(os.listdir("../input"))
%%time

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print(train.shape, test.shape)
for c in train.columns:

    if c not in test.columns: print(c)
null_cnt = train.isnull().sum().sort_values()

print('null count:', null_cnt[null_cnt > 0])
train['target'].value_counts().to_frame().plot.bar()
all_data = train.append(test, sort=False).reset_index(drop=True)

del train, test

gc.collect()



all_data.head()
from sklearn import preprocessing



features = [c for c in all_data.columns if c not in ['target', 'ID_code']]

for feat in features:

    #all_data[feat+'_var'] = all_data.groupby([feat])[feat].transform('var')

    all_data[feat+'_var'] = all_data[feat].var()

    all_data[feat+'_plus'] = all_data[feat] + all_data[feat+'_var']

    all_data[feat+'_minus'] = all_data[feat] - all_data[feat+'_var']

    all_data.loc[:,[feat+'_plus', feat+'_minus']].fillna(0, inplace=True)

    all_data.drop([feat+'_var'], axis=1, inplace=True)



scaler = preprocessing.StandardScaler()

numcols = all_data.drop('target',axis=1).select_dtypes(include='number').columns.values

all_data.loc[:,numcols] = scaler.fit_transform(all_data[numcols])

#all_data.loc[:,features] = scaler.fit_transform(all_data[features])
X_train = all_data[all_data['target'].notnull()].reset_index(drop=True)

X_test = all_data[all_data['target'].isnull()].drop(['target'], axis=1).reset_index(drop=True)

del all_data

gc.collect()



# drop ID_code

X_train.drop(['ID_code'], axis=1, inplace=True)

X_test_ID = X_test.pop('ID_code')



Y_train = X_train.pop('target')



print(X_train.shape, X_test.shape)
from sklearn import metrics

from sklearn.model_selection import StratifiedKFold

from sklearn.naive_bayes import GaussianNB



folds = StratifiedKFold(n_splits=10)

oof_preds = np.zeros(X_train.shape[0])

sub_preds = np.zeros(X_test.shape[0])

for fold_, (trn_, val_) in enumerate(folds.split(X_train, Y_train)):

    trn_x, trn_y = X_train.iloc[trn_], Y_train[trn_]

    val_x, val_y = X_train.iloc[val_], Y_train[val_]



    model = GaussianNB()

    model.fit(trn_x, trn_y)



    oof_preds[val_] = model.predict_proba(val_x)[:,1]

    sub_preds += model.predict_proba(X_test)[:,1] / folds.n_splits
gnb_1 = oof_preds[Y_train > 0]

gnb_0 = oof_preds[Y_train == 0]

plt.hist([np.log(gnb_1), np.log(gnb_0)], color=['b','r'], bins=30, alpha=0.5, histtype='barstacked')

plt.title('GaussianNB visualization')

plt.show()
fpr, tpr, thresholds = metrics.roc_curve(Y_train, oof_preds)

auc = metrics.auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %.3f)'%auc)

plt.legend()

plt.title('ROC curve')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
preds = sub_preds
submission = pd.DataFrame({

    'ID_code': X_test_ID,

    'target': preds

})

submission.to_csv("submission.csv", index=False)
submission['target'].sum()
submission['target'].hist(bins=30, alpha=0.5)

plt.show()