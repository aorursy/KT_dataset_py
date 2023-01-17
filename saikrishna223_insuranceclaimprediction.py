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
#Importing required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")
#importing dataset
df = pd.read_csv('../input/sample-insurance-claim-prediction-dataset/insurance3r2.csv')
df1 =  pd.read_csv('../input/sample-insurance-claim-prediction-dataset/insurance2.csv')
df.head()
df1.head()
#Checking for null values
sns.heatmap(df.isnull())
df.shape
df.describe()
df.dtypes
df['age'].unique()
sns.countplot(x='sex',hue='insuranceclaim',data=df)
sns.countplot(x='region',hue='insuranceclaim',data=df)
sns.countplot(x='smoker',hue='insuranceclaim',data=df)
x = df['charges']
sns.distplot(x,kde=False)
x = df['bmi']
sns.distplot(x)
#Plotting correlation
corrmat=df.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(14,8))
#To plot heatmap
import seaborn as sns
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
df.head()
#Assigning labels
X = df.iloc[ :, :8]
y = df.iloc[:, -1:]
X.head()
y.head()
#Feature importance
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)
##To plot feature importances
feat_importances=pd.Series(model.feature_importances_,index=X.columns)
feat_importances.plot(kind='barh')
#Assigning training and testing value
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
#Performing scaling using StandardScalar
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
scaler=sc.fit(X)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
#Using RandomForestClassifier model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100,max_depth=1700,random_state=5)
rf.fit(X_train, y_train)
predict = rf.predict(X_test)

from sklearn import metrics
print('Accuracy :: ',metrics.accuracy_score(y_test,predict))
print('Precision :: ',metrics.precision_score(y_test,predict))
from sklearn.metrics import accuracy_score,confusion_matrix
Rf_cm = confusion_matrix(predict,y_test)
ax = sns.heatmap(Rf_cm,annot=True)
ax.set(xlabel='predict', ylabel='true')
#Hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
model1 = RandomForestClassifier(n_estimators=50, max_depth=10, max_features='log2')

param_grid = {
    'n_estimators' : [50,1000],
    'max_depth': [1,100],
    'max_features': ['auto', 'sqrt', 'log2'],
    'random_state':[0,1000]
}


CV_rfc = GridSearchCV(estimator=model1, param_grid=param_grid, cv= 5)
CV_rfc.fit(X, y)
print (CV_rfc.best_params_)
#Using RandomForestClassifier model using the tuned parameters
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000,max_depth=100,random_state=1000,max_features='log2')
rf.fit(X_train, y_train)
predict = rf.predict(X_test)

from sklearn import metrics
print('Accuracy :: ',metrics.accuracy_score(y_test,predict))
print('Precision :: ',metrics.precision_score(y_test,predict))
from sklearn.metrics import accuracy_score,confusion_matrix
Rf_cm = confusion_matrix(predict,y_test)
ax = sns.heatmap(Rf_cm,annot=True)
ax.set(xlabel='predict', ylabel='true')
