# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
test = pd.read_csv('../input/test.csv')

train  = pd.read_csv('../input/train.csv')
tra = train[['Pclass', 'Sex', 'Age', 'Survived']].dropna().sort_values(by='Survived')

tes = test[['Pclass', 'Sex', 'Age']].dropna()

tra['Sex'] = tra['Sex'].apply(lambda x: 1 if x=='male' else 0)# numerical male/female

tes['Sex'] = tes['Sex'].apply(lambda x: 1 if x=='male' else 0)# numerical male/female



clf = SVC()

rf = RandomForestClassifier()

k = KNeighborsClassifier()

clf.fit(tra[['Pclass', 'Sex', 'Age']], tra['Survived'])

rf.fit(tra[['Pclass', 'Sex', 'Age']], tra['Survived'])

k.fit(tra[['Pclass', 'Sex', 'Age']], tra['Survived'])



svmpred = clf.predict(tes);

rfpred = rf.predict(tes);

kpred = k.predict(tes);

np.count_nonzero(svmpred-rfpred), np.count_nonzero(svmpred-kpred), np.count_nonzero(rfpred-kpred)
svm_data = pd.concat([tes, pd.Series(pred).rename('Survived (Predicted)')], axis=1)

svm_data['Sex'] = svm_data['Sex'].apply(lambda x: 'male' if x==1 else 'female')#invert numerical conversion
test.head()
pred_data.head(40)