
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
        df = pd.read_csv(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df.head()
df.columns
df.shape
df.info()
df.describe()
df.isnull().sum()
for col in df.columns:
    print(col)
    print(df[col].unique())
df=df.drop(['veil-type'], axis=1)
print(df.columns)
print(len(df.columns))
X=df.iloc[:, 1:22]
X.head()
y=df.iloc[:,0]
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(y)
print(y)
X = pd.get_dummies(X)
X
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#scaler.fit(X)
#X = scaler.transform(X)
#X
#scaler yapmamiz gerekiyor mu?
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred))
classifier1 =RandomForestClassifier()
classifier1.fit(X_train, y_train)
y_pred_1=classifier.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred_1))
classifier1.feature_importances_
Importance = pd.DataFrame({"Importance": classifier1.feature_importances_*100},
                         index = X_train.columns)
Importance.sort_values(by = "Importance", 
                       axis = 0, 
                       ascending = False)[0:5]
X_new=X[['odor_n','gill-size_n','stalk-surface-below-ring_k','odor_f','gill-size_b']]
X_new
y
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=1)
classifier2 =RandomForestClassifier()
classifier2.fit(X_train, y_train)
y_pred_2=classifier2.predict(X_test)
print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_pred_2))