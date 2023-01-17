# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 

from sklearn.model_selection import cross_validate, StratifiedKFold

from sklearn.preprocessing import StandardScaler

from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import SMOTE

from imblearn.ensemble import BalancedBaggingClassifier

from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score,roc_auc_score

pd.pandas.set_option('display.max_columns', None)

%matplotlib inline
trainset=pd.read_csv('../input/minor-project-2020/train.csv')

testset=pd.read_csv('../input/minor-project-2020/test.csv')
sns.countplot(trainset['target'])

plt.show()

print(trainset['target'].value_counts())
y=trainset['target']

X=trainset.drop(['id','target'], axis=1)
scaler=StandardScaler()

X=scaler.fit_transform(X)
from sklearn.linear_model import LogisticRegression

lr= LogisticRegression(max_iter=2000)

clf=BalancedBaggingClassifier(base_estimator=lr)

scoring = ['accuracy', 'precision_macro', 'recall_macro' , 'f1_weighted', 'roc_auc']

scores = cross_validate(clf, X, y, scoring=scoring, cv=5)
clf.fit(X,y)
test=testset.drop(['id'],axis=1)

scaled_test = scaler.fit_transform(test)
pred_test = clf.predict(scaled_test)
ids = testset['id'].values

output = {'id':ids, 'target':pred_test}

output = pd.DataFrame(output)
sns.countplot(output['target'],label="Sum")

plt.show()

print(output['target'].value_counts())
output.to_csv('outputbaggingbalanced.csv',index=False)