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
print(os.listdir("../input/ds2-ds5-competition-1/"))
train = pd.read_csv("../input/ds2-ds5-competition-1/train.csv")

test = pd.read_csv("../input/ds2-ds5-competition-1/test.csv")

submission = pd.read_csv("../input/ds2-ds5-competition-1/sample_submission.csv")
train.info()
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(train.label)

plt.xticks(rotation=45)

plt.show() 
np.unique(train.label)
sns.countplot(train.label)

plt.xticks(rotation=45)

plt.show() 
train[train.label == 0].shape[0] / train.shape[0]
sns.jointplot(x='time', y='label', data=train)

plt.show()
sns.jointplot(x='s1', y='label', data=train)

plt.show()
from sklearn.linear_model import LinearRegression
X = train.copy()

x_cols = ['s'+ str(i) for i in list(range(1,17,1))]

X = X[x_cols]

X.head()
y = train['label']
lm_model = LinearRegression()

lm_model.fit(X, y)
new_X = test[x_cols]

new_y = lm_model.predict(new_X)
submission_lm = submission.copy()

submission_lm['label'] = new_y
submission_lm.shape
submission_lm.head()
submission_lm.to_csv('submission_lm.csv', index=False)
from sklearn.ensemble import RandomForestRegressor
corr = train.corr(method='pearson')

print(corr.label)
plt_cols = ['s'+ str(i) for i in [1, 5, 6, 14]]

plt_cols = ['label'] + plt_cols

plt_cols
sns.pairplot(train[plt_cols])

plt.show()
sub_x_cols = ['s1', 's5', 's6', 's14']

sub_X = train[sub_x_cols]

sub_X.head()
rf_model = RandomForestRegressor()

rf_model.fit(sub_X, y)
new_X = test[sub_x_cols]

new_y = rf_model.predict(new_X)
submission_rf = submission.copy()

submission_rf['label'] = new_y
submission_rf.head()
submission_rf.to_csv('submission_rf.csv', index=False)