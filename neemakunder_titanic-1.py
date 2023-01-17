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
import matplotlib.pyplot as plt
import seaborn as sns
train_data=pd.read_csv('/kaggle/input/titanic/train.csv')
train_data.head()
test_data=pd.read_csv('/kaggle/input/titanic/test.csv')
test_data.head()
print(train_data.shape)
print(train_data.info()) 
print(train_data.Age.isnull().sum())
print(train_data.Cabin.isnull().sum()) 
train_data=train_data.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
train_data=pd.get_dummies(train_data)
display(train_data.head()) 
sns.heatmap(train_data.corr())
# can't use this with age cuz null
#pd.plotting.scatter_matrix(train_data,c=train_data.Survived.values,figsize=(15,15),s=10,cmap='brg',alpha=0.8)
#plt.show() 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

steps= [('imputation',SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scaler',StandardScaler()),
        ('model',RandomForestClassifier())]

pipeline=Pipeline(steps)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 300, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]


parameters={'model__n_estimators': n_estimators,
               'model__max_features': max_features,
               'model__max_depth': max_depth,
               'model__min_samples_split': min_samples_split,
               'model__min_samples_leaf': min_samples_leaf,
               'model__bootstrap': bootstrap}


X=train_data.drop('Survived',axis=1)
y=train_data['Survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


cv=GridSearchCV(pipeline,param_grid=parameters, cv=3)

cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)

y_score=cv.score(X_test,y_test)
print('Accuracy: {}'.format(y_score))
print(classification_report(y_test,y_pred))
print('Tuned model paramters: {}'.format(cv.best_params_))


#output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': y_pred})
#output.to_csv('my_submission.csv', index=False)