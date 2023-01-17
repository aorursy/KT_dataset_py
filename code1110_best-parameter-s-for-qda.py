# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import GridSearchCV, train_test_split



from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



import seaborn as sns

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



print("Libraries were imported.")
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print("Data were loaded.")

cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]
# parameter to be optimized

params = [{'reg_param': [0.1, 0.2, 0.3, 0.4, 0.5]}]



# 512 models

reg_params = np.zeros(512)

for i in range(512):

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = VarianceThreshold(threshold=2).fit_transform(data[cols])



    train3 = data2[:train2.shape[0]]; 

    

    qda = QuadraticDiscriminantAnalysis()

    clf = GridSearchCV(qda, params, cv=4)

    clf.fit(train3, train2['target'])

    

    reg_params[i] = clf.best_params_['reg_param']

    print("Best reg_param for model " + str(i) + " is " + str(reg_params[i]))
sns.distplot(reg_params)

plt.title("reg_param in QDA")

plt.figure()
oof = np.zeros(len(train))

preds = np.zeros(len(test))



for i in range(512):



    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)



    data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])

    data2 = VarianceThreshold(threshold=2).fit_transform(data[cols])



    train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]



    skf = StratifiedKFold(n_splits=5, random_state=42)

    for train_index, test_index in skf.split(train2, train2['target']):



        clf = QuadraticDiscriminantAnalysis(reg_params[i])

        clf.fit(train3[train_index,:],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]

        preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits



auc = roc_auc_score(train['target'], oof)

print(f'AUC: {auc:.5}')
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv', index=False)