# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
df = pd.read_csv('../input/avocado.csv')
#lets check first 10 rows of our data.
df.head(10)
#drop the unnecessary columns.
df = df.drop(['Unnamed: 0', 'Date'], axis = 1)
df.head(10)
#check the correlation matrix.
plt.figure(figsize=(12,6))
sns.heatmap(df.corr(),cmap='coolwarm',annot=True)

df['region'].nunique()
df['type'].nunique()
#there are 54 entries in region so we will drop this entire column as it will be difficult to transfer dummy entries to this column.
df_final=pd.get_dummies(df.drop(['region'],axis=1),drop_first=True)

df_final.head()

X=df_final.iloc[:,1:10]
y=df_final['AveragePrice']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn import preprocessing
from sklearn import utils
lab_enc = preprocessing.LabelEncoder()
y_train = lab_enc.fit_transform(y_train)
y_test = lab_enc.fit_transform(y_test)

from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg = OneVsRestClassifier(logreg, n_jobs=1)
logreg.fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn = OneVsRestClassifier(knn, n_jobs=1)
knn.fit(X_train,y_train)
pred_knn = knn.predict(X_test)

from sklearn.svm import SVC
svc = SVC()
svc = OneVsRestClassifier(svc, n_jobs=1)
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)

from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree = OneVsRestClassifier(decision_tree, n_jobs=1)
decision_tree.fit(X_train, y_train)
pred_tree = decision_tree.predict(X_test)

sns.jointplot(x=y_test, y=pred_logreg, color= 'g')
sns.jointplot(x=y_test, y=pred_knn, color= 'g')
sns.jointplot(x=y_test, y=pred_svc, color= 'g')
sns.jointplot(x=y_test, y=pred_tree, color= 'g')

plt.show()
 
#as knn is giving the best y=x graph for y_test vs ypredicted values we will go with this and modify knn.

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn = OneVsRestClassifier(knn, n_jobs=1)
knn.fit(X_train,y_train)
pred_knn = knn.predict(X_test)
sns.jointplot(x=y_test, y=pred_knn, color= 'g')
plt.plot()
