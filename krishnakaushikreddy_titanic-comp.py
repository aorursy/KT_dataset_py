# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
data = pd.read_csv("../input/titanic-extended/train.csv")
df = pd.read_csv('../input/titanic-extended/test.csv')
data.head()
df.head()
df.shape, data.shape
data.isnull().sum(), df.isnull().sum()
data.info(),df.info()
data.Age.loc[data.Age.isna()] = data.Age_wiki
df.Age.loc[df.Age.isna()] = df.Age_wiki
data.Age.loc[data.Age.isna()] = data.Age.mean()
df.Age.loc[df.Age.isna()] = df.Age.mean()
data['Embarked'].value_counts()
data.Embarked.loc[data.Embarked.isna()] = 'S'
df['Embarked'].value_counts()
df.Embarked.loc[df.Embarked.isna()] = 'S'
data['Fare'].describe()
data.Fare.loc[data.Fare.isna()] = 32.20
df['Fare'].describe()
df.Fare.loc[df.Fare.isna()] = 35.20
data = data.iloc[:,:12]
data.head()
df = df.iloc[:,:12]
data.head()
l=[]
import re
for i in data['Name']:
    info=re.search('(\s)([A-za-z]+)',i)
    info.group()
    l.append(info.group(2))
data['Name'] = pd.DataFrame({"Modified Named":l})
data['Name'].value_counts()
data['Survived'].value_counts()
survived = data.loc[(data['Survived'] == 1)]
survived
survived["Embarked"].value_counts()
survived["Sex"].value_counts()
grouped = data.groupby('Survived')
grouped['Embarked'].value_counts(normalize=True).unstack() * 100
import matplotlib.pyplot as plt
import seaborn as sns
sorted_nb = data.groupby(['Survived'])['Age'].median().sort_values()

sns.boxplot(x=data['Survived'], y=data['Age'], order=list(sorted_nb.index))
plt.xticks(rotation=90)
plt.show()
data.drop(['Ticket','Cabin'],axis=1,inplace=True)
data.head()
df.drop(['Ticket','Cabin','WikiId'],axis=1,inplace=True)
df.head()
l=[]
import re
for i in df['Name']:
    info=re.search('(\s)([A-za-z]+)',i)
    info.group()
    l.append(info.group(2))
df['Name'] = pd.DataFrame({"Modified Named":l})
df['Name'].value_counts()
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data.Survived = data.Survived.astype('int64')
sns.pairplot(data[['Survived','Age','SibSp','Parch','Fare']])
from sklearn.preprocessing import LabelEncoder
import numpy as np
values1=np.array(data['Name'])
label_encoder = LabelEncoder()
integer_encoded1 = label_encoder.fit_transform(values1)


values2=np.array(data['Sex'])
label_encoder = LabelEncoder()
integer_encoded2 = label_encoder.fit_transform(values2)


values3=np.array(data['Embarked'])
label_encoder = LabelEncoder()
integer_encoded3 = label_encoder.fit_transform(values3)

data['Name']=pd.DataFrame(integer_encoded1)
data['Sex']=pd.DataFrame(integer_encoded2)
data['Embarked']=pd.DataFrame(integer_encoded3)
data.head()
from sklearn.preprocessing import LabelEncoder
import numpy as np
values1=np.array(df['Name'])
label_encoder = LabelEncoder()
integer_encoded1 = label_encoder.fit_transform(values1)


values2=np.array(df['Sex'])
label_encoder = LabelEncoder()
integer_encoded2 = label_encoder.fit_transform(values2)


values3=np.array(df['Embarked'])
label_encoder = LabelEncoder()
integer_encoded3 = label_encoder.fit_transform(values3)


df['Name']=pd.DataFrame(integer_encoded1)
df['Sex']=pd.DataFrame(integer_encoded2)
df['Embarked']=pd.DataFrame(integer_encoded3)

df.head()
X = data.drop('Survived',axis=1)
y= data.Survived
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
data.corr()
plt.figure(figsize=(12,12))
sns.heatmap(data.corr(),annot=True)
y_predtest=logreg.predict(df)

from sklearn.neighbors import KNeighborsClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=0)
classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(X_train, y_train)
y_predclass = classifier.predict(X_test)
print('Accuracy of KNeighborsClassifier classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))
print(accuracy_score(y_test, y_predclass))
from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
y_preddecison=model.predict(X_test)
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))
print(accuracy_score(y_test, y_preddecison))
