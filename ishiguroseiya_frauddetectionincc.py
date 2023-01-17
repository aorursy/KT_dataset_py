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
df_train = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/train.csv', index_col=0)

df_test = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/test.csv', index_col=0)
df_train
df_test
import seaborn as sns

from matplotlib import pyplot



sns.set_style("darkgrid")

pyplot.figure(figsize=(31, 31))

sns.heatmap(df_train.corr(), square=True, annot=True)
df_train = df_train[['V3', 'V6', 'V7', 'V10', 'V12', 'V13', 'V16', 'V17', 'Amount', 'Class']]

df_test = df_test[['V3', 'V6', 'V7', 'V10', 'V12', 'V13', 'V16', 'V17', 'Amount']]
df_train
df_test
from sklearn.decomposition import PCA



X_train = df_train.drop('Class', axis=1).values  # 目的変数を除いてndarray化

pca = PCA()  # 次元圧縮なし

pca.fit(X_train) 
sns.set_style("darkgrid")

ev_ratio = pca.explained_variance_ratio_

ev_ratio = np.hstack([0,ev_ratio.cumsum()])

sns.lineplot(data=ev_ratio)
X_train = df_train.drop('Class', axis=1).values

pca = PCA(n_components=2)  # 2次元

pca.fit(X_train)  # 主成分分析

X_train_pca = pca.transform(X_train) 
df_train_pca = pd.DataFrame(X_train_pca, columns=['Comp. 1', 'Comp. 2'], index=df_train.index)

df_train_pca['Class'] = df_train['Class']



sns.set_style("darkgrid")

sns.relplot(data=df_train_pca, x='Comp. 1', y='Comp. 2', hue='Class')
from sklearn.svm import SVC

X_train = df_train.drop('Class', axis=1).values

y_train = df_train['Class'].values



model = SVC()

model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error

predict = model.predict(X_train)

mean_squared_error(predict, y_train)
X_test = df_test.values

predict = model.predict(X_test)
X_test_pca = pca.transform(X_test)

df_test_pca = pd.DataFrame(X_test_pca, columns=['Comp. 1', 'Comp. 2'])

df_test_pca['Class'] = predict



sns.set_style("darkgrid")

sns.relplot(data=df_test_pca, x='Comp. 1', y='Comp. 2', hue='Class')
submit = pd.read_csv('/kaggle/input/1056lab-fraud-detection-in-credit-card/sampleSubmission.csv')

submit['Class'] = predict

submit.to_csv('submission.csv', index=False)