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
import pandas as pd
df=pd.read_csv('../input/drug-classification/drug200.csv')
df.head()
df['Drug'].value_counts()
df['Drug'].replace('DrugY', 'drugY', inplace=True)
df.info()
df.describe()
import seaborn as sns
sns.pairplot(df, hue='Drug')
sns.countplot(x=df['Drug'], data=df)
sns.countplot(x='Sex', hue='Drug', data=df)
df.groupby('Drug')['BP'].value_counts().unstack().plot.bar()
df.groupby('Drug')['Cholesterol'].value_counts().unstack().plot.bar()
df.groupby('Drug')['Sex'].value_counts().unstack().plot.bar()
sns.swarmplot(x='Drug', y='Na_to_K', data=df)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cols = ['Sex', 'BP', 'Cholesterol', 'Drug']
for features in cols:
    df[features] = le.fit_transform(df[features])
from sklearn.model_selection import train_test_split
X = df.drop('Drug', axis=1)
y = df['Drug']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, lr_pred)*100,'%')
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_pred = gbc.predict(X_test)
print(accuracy_score(y_test,gbc_pred)*100,'%')
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_pred = dtc.predict(X_test)
print(accuracy_score(y_test, dtc_pred)*100,'%')
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100, criterion = 'entropy', random_state=22)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(accuracy_score(y_test, rfc_pred)*100,'%')
