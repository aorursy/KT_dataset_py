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
train_df = pd.read_csv('/kaggle/input/minor-project-2020/train.csv')
train_df.head()
train_df.shape
train_df[train_df["target"]==0].shape
from imblearn.over_sampling import SMOTE
sm = SMOTE(sampling_strategy='auto', random_state=7)
oversampled_trainX, oversampled_trainY = sm.fit_sample(train_df.drop('target', axis=1), train_df['target'])
oversampled_train = pd.concat([pd.DataFrame(oversampled_trainY), pd.DataFrame(oversampled_trainX)], axis=1)
temp = oversampled_train[oversampled_train['target']==0].shape
temp1 = oversampled_train[oversampled_train['target']==1].shape
print(temp, temp1)
import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt


fig, ax = plt.subplots(figsize=(20,10))
corr = oversampled_train.corr()
sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)
ax.set_title("Correlation Matrix", fontsize=14)
plt.show()
corr_gt = corr[abs(corr['target'])>=0.035]
corr_gt
col_names = ['col_0','col_1','col_2','col_4','col_5','col_7','col_8','col_11','col_13','col_15','col_16','col_23','col_24','col_35','col_52','col_56','col_57','col_59','col_62','col_72']
X, y = oversampled_train[col_names], oversampled_train['target']
import matplotlib.pyplot as plt
%matplotlib inline


for i in range(20):
    oversampled_train.plot.scatter(x=col_names[i], y='target')


from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X)       #Instantiate the scaler
scaled_X_train = scaler.transform(X)    #Fit and transform the data

scaled_X_train
test_df = pd.read_csv('/kaggle/input/minor-project-2020/test.csv')
x_test = test_df[col_names]
X_test = scaler.transform(x_test)
X_test
from sklearn.linear_model import LogisticRegression

X_train = scaled_X_train
y_train = y

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)

prob = logistic_regression.predict_proba(X_test)
prob

prob.shape
prob = prob[:,1]
prob
results = pd.DataFrame({'id': pd.DataFrame.to_numpy(test_df['id']), 'target': prob})
results
results.to_csv(r'/kaggle/working/submission.csv',index=False)