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
trained_data = pd.read_csv('../input/titanic/train.csv')

trained_data.head()
test_data = pd.read_csv('../input/titanic/test.csv')

test_data.head()
trained_data.isnull().sum()
trained_data.drop('Cabin',axis=1,inplace = True)
trained_data.isnull().sum()
df_filter = trained_data

df_filter.head()
df_filter.drop(df_filter[df_filter['Embarked'].isnull()].index,inplace=True)

df_filter.isnull().sum()
df_filter.drop('Name',axis=1,inplace=True)

df_filter
df_filter.drop('Ticket',axis=1,inplace=True)
df_filter
df_filter = pd.get_dummies(df_filter,drop_first=True)

df_filter
df_filter.isnull().sum()
dt= df_filter.loc[df_filter['Age'].notnull()]
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(dt.drop('Age', 1), dt['Age'], test_size = .271, random_state=37)
r = LinearRegression()

r.fit(X_train,y_train)
dtf = df_filter

dtf
a = dtf.loc[dtf['Age'].isnull()]

a=a.drop(['Age'],1)

b= r.predict(a)

b.shape
dtf.loc[dtf['Age'].isnull(),'Age']=b

dtf.head(10)
dtf[dtf['Age']<0]*=-1

dtf[dtf['Age']<0]
dtf.dtypes
import seaborn as sns

sns.clustermap(dtf.corr(),annot=True,figsize=(10,10))
X_tr, X_te, y_tr, y_te = train_test_split(dtf.drop('Survived', 1), dtf['Survived'], test_size = .2, random_state=37)
from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

ndtf=dtf.drop('Survived', 1)

select_feature = SelectKBest(f_classif,k=6).fit(X_tr,y_tr)

mask = select_feature.get_support()

nf = ndtf.columns[mask]

print(select_feature.scores_)

print(nf)
X_tr2 = select_feature.transform(X_tr)

X_te2 = select_feature.transform(X_te)

print(X_tr2.shape,y_tr.shape)

print(X_te2.shape,y_te.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import f1_score,confusion_matrix

from sklearn.metrics import accuracy_score

clf_rf_2 = RandomForestClassifier()      

clr_rf_2 = clf_rf_2.fit(X_tr2,y_tr)

ac_2 = accuracy_score(y_te,clf_rf_2.predict(X_te2))

print('Accuracy is: ',ac_2)

cm_2 = confusion_matrix(y_te,clf_rf_2.predict(X_te2))

sns.heatmap(cm_2,annot=True,fmt="d")
td =test_data

td
td.drop(['Cabin','Name','Ticket'],axis=1,inplace = True)

td =td.fillna(0)
td = pd.get_dummies(td,drop_first=True)

td
feat=['Pclass', 'Age', 'Parch', 'Fare', 'Sex_male', 'Embarked_S']

test_f = pd.get_dummies(td[feat])

predictions = clr_rf_2.predict(test_f)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")