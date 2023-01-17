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
df = pd.read_csv('/kaggle/input/logistic-regression/Social_Network_Ads.csv')

print(df.shape)

print('-'*40)

print(df.info())

print('-'*40)

print(df.describe())

print('-'*40)

print(df.isnull().sum())
df.head()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

df['Gender'] = encoder.fit_transform(df['Gender'])
import seaborn as sns

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20, 10)

sns.heatmap(df.corr(), annot=True)

plt.title('Data Correlation')
g = sns.FacetGrid(df, col='Purchased')

g.map(plt.hist, 'Age')
X = df.iloc[:, [2,3]].values

y = df.iloc[:, -1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

score_LogReg = round(logreg.score(X_train, y_train) * 100, 2)

print('Score ', score_LogReg)

print('Confusion Matrix \n', confusion_matrix(y_test, y_pred))
from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train ,y_train)

y_pred = svc.predict(X_test)

score_SVM = round(svc.score(X_train, y_train) * 100, 2)

print('Score ', acc_SVM)

print('Confusion Matrix \n', confusion_matrix(y_test, y_pred))
df_score = pd.DataFrame({'Model':['Logistic Regression', 'Support Vector Machine'], 'Score':[score_LogReg, score_SVM]})

df_score 