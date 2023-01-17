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
data = pd.read_csv("/kaggle/input/udacity-mlcharity-competition/census.csv")

display(data.describe())

display(data.head(n=10))

data.isnull().sum()
data['income'] = data['income'].apply(lambda x: 1 if x=='>50K' else 0)

data = pd.get_dummies(data, columns=['workclass', 'education_level', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'])
# removing label

y = data.pop('income')

display(y.head())

display(data.head())
from sklearn.model_selection import train_test_split

# splitting datasets

X_train, X_test, y_train, y_test = train_test_split(data, y, train_size=0.8)
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier()

model.fit(X_train, y_train)
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve

accuracy_score(y_test, y_predict)
len(y[y == 0]) / len(y)
f1_score(y_test, y_predict)
confusion_matrix(y_test, y_predict)
import matplotlib.pyplot as plt
y_probas = model.predict_proba(X_test)[::,1]

fpr, tpr, _ = roc_curve(y_test, y_probas)

plt.plot(fpr, tpr)

plt.legend(loc=4)

plt.show()
results = pd.DataFrame({'fpr':fpr,'tpr': tpr, 'threshold':_})
results[results['tpr']> .9]
X_submission = pd.read_csv('/kaggle/input/udacity-mlcharity-competition/test_census.csv')

X_submission = pd.get_dummies(X_submission, columns=['workclass', 'education_level', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'])

X_submission.head()

X_submission = X_submission.dropna(axis=1)
y_pred_test = model.predict(X_submission)
y_pred_df = pd.DataFrame(y_pred_test, columns=['income'])
y_pred_df
y_pred_df.to_csv('submission.csv', index_label='id')