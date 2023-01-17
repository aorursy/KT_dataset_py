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
dataset = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from  matplotlib.pyplot import subplot

%matplotlib inline



from sklearn import svm

from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report,confusion_matrix,mean_squared_error
dataset.head()
dataset.shape
dataset.info()
dataset.describe()
dataset.quality.value_counts()
dataset.isnull().sum()
sns.countplot(dataset['quality'])
plt.figure(figsize=(20,10))

sns.heatmap(dataset.corr(), annot=True,cmap='Blues')

plt.show()
dataset.corr()
sns.barplot(y='fixed acidity', x='quality', data=dataset)
sns.barplot(y='volatile acidity', x='quality', data=dataset)
sns.barplot(y='citric acid', x='quality', data=dataset)
sns.barplot(y='residual sugar', x='quality', data=dataset)
sns.barplot(y='chlorides', x='quality', data=dataset)
sns.barplot(y='free sulfur dioxide', x='quality', data=dataset)
sns.barplot(y='total sulfur dioxide', x='quality', data=dataset)
sns.barplot(y='density', x='quality', data=dataset)
sns.barplot(y='pH', x='quality', data=dataset)
sns.barplot(y='sulphates', x='quality', data=dataset)
sns.barplot(y='alcohol', x='quality', data=dataset)
dataset['quality'].min()
dataset['quality'].max()
values = (2, 6, 9)

qual = ['bad', 'good']

dataset['quality'] = pd.cut(dataset['quality'], bins = values, labels = qual)

dataset.head()
dataset['quality'].value_counts()
label_enc = LabelEncoder()

dataset['quality']=label_enc.fit_transform(dataset['quality'])
X = dataset.drop('quality',axis=1)

y = dataset['quality']
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2,random_state=42)
std_scale = StandardScaler()

xtrain = std_scale.fit_transform(xtrain)

xtest = std_scale.fit_transform(xtest)
model = svm.SVC()

model.fit(xtrain,ytrain)

y0_pred = model.predict(xtest)

print(accuracy_score(ytest,y0_pred))
rf = RandomForestClassifier()

rf.fit(xtrain,ytrain)

y1_pred = rf.predict(xtest)

print(accuracy_score(ytest,y1_pred))
xgb = XGBClassifier(max_depth=3,n_estimators=200,learning_rate=0.5)

xgb.fit(xtrain,ytrain)

y2_pred = xgb.predict(xtest)

print(accuracy_score(ytest,y2_pred))
print(confusion_matrix(ytest,y2_pred))
x=dataset.drop('quality', axis = 1)

y= dataset['quality']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

scaler.fit_transform(x)



x.head()
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)
from sklearn.linear_model import LogisticRegression

classifier_log = LogisticRegression()

model = classifier_log.fit(xtrain,ytrain)



y_pred_log = classifier_log.predict(xtest)



from sklearn.metrics import accuracy_score

print(accuracy_score(y_pred_log, ytest)*100)
from sklearn.tree import DecisionTreeClassifier



# doing pruning to avoid overfitting

classifier_tree=DecisionTreeClassifier(criterion ='gini', splitter = 'random',

                         max_leaf_nodes = 10, min_samples_leaf = 5, 

                         max_depth = 6)

model = classifier_tree.fit(xtrain, ytrain)



y_pred_tree = classifier_tree.predict(xtest)



print(accuracy_score(y_pred_tree, ytest)*100)
print(classification_report(ytest,y2_pred))