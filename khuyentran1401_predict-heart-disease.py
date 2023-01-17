# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
disease = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

disease.describe()
disease.info()
from sklearn.model_selection import train_test_split



train, test = train_test_split(disease, test_size = 0.2, random_state = 1)
train.describe()
train.info()
train.head(10)
train['target'].value_counts()
train['target'].value_counts()[1]/train['target'].value_counts()[0]
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(10,10))

sns.heatmap(train.corr(),annot=True)
import seaborn as sns

import matplotlib.pyplot as plt



sns.set_style('whitegrid')

sns.catplot('target','age', data=train, kind='box', palette='viridis')

sns.catplot('target','age', data=train, kind='box', hue ='sex', palette='viridis')
sns.barplot('sex','target',data=train, palette='coolwarm')
sex_compare = train[['sex','target']].groupby('sex').mean()

sex_compare
sex_compare.iloc[0]/sex_compare.iloc[1]
sex_count = train['sex'].value_counts()

sex_count
sex_count.iloc[1]/sex_count.iloc[0]
sns.barplot('cp','target',data=train)
sns.catplot('target','trestbps',data=train, kind='box')
sns.catplot('target','chol',data=train, kind='box')
g = sns.FacetGrid(data=train, col = 'fbs')

g.map(plt.hist,'target', bins=2)
diff_df = pd.DataFrame(train[['target','fbs']])

diff_df['diff'] = np.abs(diff_df['target']-diff_df['fbs'])

np.sum(diff_df['diff'])/diff_df['diff'].count()

g = sns.FacetGrid(data=train, col = 'restecg')

g.map(sns.barplot,'target')
sns.catplot('target','thalach',data=train, kind='box')
g = sns.FacetGrid(data=train, col = 'target')

g.map(plt.hist,'thalach')
g = sns.FacetGrid(data=train, col = 'target')

g.map(plt.hist,'exang', bins=2)
sns.barplot('exang','target',data=train)
sns.catplot('target','oldpeak',data=train, kind= 'box')
g = sns.FacetGrid(data=train, col = 'target')

g.map(plt.hist,'oldpeak')
sns.barplot('slope','target',data=train)
g = sns.FacetGrid(data=train, col = 'target')

g.map(plt.hist,'slope',bins=3)
sns.barplot('target','slope',data=train)
sns.barplot('target','ca',data=train)
g = sns.FacetGrid(data=train, col = 'target')

g.map(plt.hist,'ca',bins=3)
sns.countplot('ca',data=train, hue='target')
sns.countplot('thal',data=train, hue='target')
X_train = train.drop(['target','fbs'],axis=1)

y_train = train['target']



X_test = train.drop(['target','fbs'],axis=1)

y_test = train['target']
from sklearn.linear_model import SGDClassifier



sgd_model = SGDClassifier(random_state=1)

sgd_model.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score

cross_val_score(sgd_model, X_train, y_train).mean()
from sklearn.model_selection import cross_val_predict

from sklearn.metrics import confusion_matrix



sgd_predictions = cross_val_predict(sgd_model, X_train, y_train)

confusion_matrix(y_train, sgd_predictions)

from sklearn.metrics import classification_report



print(classification_report(y_train, sgd_predictions))
from sklearn.ensemble import RandomForestClassifier



forest_clf = RandomForestClassifier(random_state=1)

forest_predictions = cross_val_predict(forest_clf, X_train, y_train )

print(classification_report(y_train, forest_predictions))