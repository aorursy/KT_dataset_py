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
import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.listdir('../input')
df= pd.read_csv('../input/iris/Iris.csv')
df.head()
df.isna().sum()
df['Species'].value_counts()
plt.figure(figsize=(20,5))
plt.subplot(121)
sns.countplot(df['Species'])
plt.subplot(122)
plt.pie(df['Species'].value_counts() ,autopct='%1.1f%%',labels=['Iris-setosa' ,'Iris-virginica ' ,'Iris-versicolor'] );

sns.pairplot(df ,hue = 'Species')
from sklearn.preprocessing import LabelEncoder
en = LabelEncoder()
df['Species'] = en.fit_transform(df[['Species']])
df['Species'].value_counts()
df.sample(5)
X= df.iloc[:,1:5]
Y= df['Species']
X.shape ,Y.shape
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(X,Y,test_size=.30 ,random_state=101)
x_train.shape,y_train.shape
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators= 20 ,random_state = 0)
rf.fit(x_train,y_train)
pred = rf.predict(x_test)
pred

print(accuracy_score(y_test,prediction))
rf.estimators_
print(len(rf.estimators_))
plt.figure(figsize=(20,7))
tree.plot_tree(rf.estimators_[1] ,filled=True)
plt.figure(figsize=(20,7))

for i in range(len(rf.estimators_)):
    tree.plot_tree(rf.estimators_[i])
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,classification_report ,confusion_matrix
model = LogisticRegression()
model.fit(x_train,y_train)
prediction = model.predict(x_test)
print(prediction)
print(accuracy_score(y_test,prediction))
print(classification_report(y_test ,prediction))
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(x_train ,y_train)
pred = clf.predict(x_test)
print(classification_report(y_test ,pred))
from sklearn.model_selection import StratifiedKFold, GridSearchCV
rclf=RandomForestClassifier()

parametre_grid={'n_estimators':[5,10,25,50,100],
               'criterion':['gini','entropy'],
               'max_features':[1,2,3,4],
               'warm_start':[True,False]}

crs_validation= StratifiedKFold(n_splits=10, shuffle=True ,random_state=101)

grid_search=GridSearchCV(rclf,parametre_grid ,cv=crs_validation )

grid_search.fit(x_train.values,y_train.values)
print('Best score{}'.format(grid_search.best_score_))
print('Best Parametrs{}'.format(grid_search.best_params_))
grid_search.best_estimator_
grid_search.predict(x_test)
accuracy_score(y_test,grid_search.predict(x_test))

