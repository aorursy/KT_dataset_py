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

sns.set_style('whitegrid')
df= pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
sns.jointplot('Glucose','BloodPressure',data=df,kind='hex')
sns.jointplot('Glucose','Insulin',data=df,kind='hex')
sns.jointplot('BMI','Age',data=df,kind='hex')
sns.pairplot(df.drop('Outcome',axis=1))
df.isna().sum()
df.isnull().sum()
from sklearn.model_selection import train_test_split



X = df.drop('Outcome',axis=1)

y = df['Outcome']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=101)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression



lm = LogisticRegression()

lm.fit(X_train,y_train)
pred = lm.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix



print("Confusion Matrix for Logistic Regression")

print(confusion_matrix(y_test,pred))



print('\n\n')

print('Classification Report')

print(classification_report(y_test,pred))
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=5)

knn.fit(X_train,y_train)
knn_pred = knn.predict(X_test)
print('Confusion matrix for KNN with n = 5')

print(confusion_matrix(y_test,knn_pred))



print('\n\n')

print('Classification Report')

print(classification_report(y_test,knn_pred))
error_rate=[]



for i in range(1,50):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))

    

    
plt.figure(figsize=(10,6))

plt.plot(range(1,50),error_rate)
knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(X_train,y_train)

k_pred = knn.predict(X_test)



print('Confusion matrix for KNN with n = 10')

print(confusion_matrix(y_test,k_pred))



print('\n\n')

print('Classification Report')

print(classification_report(y_test,k_pred))
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV



dtree = DecisionTreeClassifier()

#dtree.fit(X_train,y_train)
dtree_params = {'max_leaf_nodes': list(range(2,100)),'min_samples_split':[2,3,4],'max_depth': list(range(1,20,2))}                                     
grid = GridSearchCV(DecisionTreeClassifier(),dtree_params,verbose=3,cv=5)

grid.fit(X_train,y_train)
print('Best Estimators: {}'.format(grid.best_estimator_))

print('Best Parameters: {}'.format(grid.best_params_))

print('Best Score: {}'.format(grid.best_score_))
grid_pred = grid.predict(X_test)
print('confusion matrix of decision tree')

print(confusion_matrix(y_test,grid_pred))



print('\n\n')

print(classification_report(y_test,grid_pred))
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier()

#rfc.fit(X_train,y_train)

#rfc_pred = rfc.predict(X_test)

param_grid = {'min_samples_leaf': [3, 4, 5],'min_samples_split': [8, 10, 12],'n_estimators': [100, 200, 300, 1000]}

grid = GridSearchCV(RandomForestClassifier(),param_grid,verbose=3,cv=5)



grid.fit(X_train,y_train)
print('Best Estimators: {}'.format(grid.best_estimator_))

print('Best Parameters: {}'.format(grid.best_params_))

print('Best Score: {}'.format(grid.best_score_))
grid_pred = grid.predict(X_test)
print('confusion matrix random forest classifier')

print(confusion_matrix(y_test,grid_pred))



print('\n\n')

print(classification_report(y_test,grid_pred))
from sklearn.naive_bayes import GaussianNB



nb = GaussianNB()

nb.fit(X_train,y_train)
nb_pred= nb.predict(X_test)
print('confusion matrix gaussianNB')

print(confusion_matrix(y_test,nb_pred))



print('\n\n')

print(classification_report(y_test,nb_pred))
from sklearn.svm import SVC





sv = SVC()

grid_param = {'C': [0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}

grid = GridSearchCV(SVC(),grid_param,verbose=3)

grid.fit(X_train,y_train)

print('Best Estimators: {}'.format(grid.best_estimator_))

print('Best Parameters: {}'.format(grid.best_params_))

print('Best Score: {}'.format(grid.best_score_))
grid_pred = grid.predict(X_test)
print('confusion matrix SVM with grid search')

print(confusion_matrix(y_test,grid_pred))

print('\n\n')

print(classification_report(y_test,grid_pred))