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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/lendingclubcom-loan-dataset/loan_data.csv')
df.info()
df.head()
df.describe()
plt.figure(figsize=(15,10))

sns.heatmap(df.corr(), annot= True, cmap = 'coolwarm')
plt.figure(figsize=(11,7))

sns.countplot(x='purpose',hue='not.fully.paid',data=df,palette='Set1')

plt.show()
df.drop(['delinq.2yrs','log.annual.inc','dti','days.with.cr.line'],axis=1,inplace=True)

df.head()

cat_feats = ['purpose']
final_data = pd.get_dummies(df,columns=cat_feats,drop_first=True)
final_data.info()
final_data['not.fully.paid'].value_counts()
from imblearn.combine import SMOTETomek

smk = SMOTETomek()
X = final_data.drop('not.fully.paid',axis=1)

y = final_data['not.fully.paid']

X_res,y_res=smk.fit_sample(X,y)
y_res.value_counts()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_res = sc.fit_transform(X_res)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.30, random_state=101)
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)



from sklearn.metrics import classification_report,confusion_matrix



print(classification_report(y_test,predictions))



print(confusion_matrix(y_test,predictions))
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators=800)



rfc.fit(X_train,y_train)
predictions = rfc.predict(X_test)



from sklearn.metrics import classification_report,confusion_matrix



print(classification_report(y_test,predictions))



print(confusion_matrix(y_test,predictions))
