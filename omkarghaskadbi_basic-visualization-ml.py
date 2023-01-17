# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #For Visualization 

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
iris = pd.read_csv("../input/Iris.csv")

iris.head()
iris.drop('Id',inplace=True,axis=1)
iris.head()
iris.info()
iris.describe()
sns.scatterplot(x = 'SepalLengthCm', y = 'SepalWidthCm', data= iris ,hue='Species')

plt.title('Sepal Length vs Sepal Width')

plt.show()
sns.scatterplot(x = 'PetalLengthCm', y = 'PetalWidthCm', data= iris ,hue='Species')

plt.title('Petal Length vs Petal Width')

plt.show()
sns.scatterplot(x = 'PetalLengthCm', y = 'SepalLengthCm', data= iris ,hue='Species')

plt.title('Petal Length vs Sepal Length')

plt.show()
sns.scatterplot(x = 'PetalWidthCm', y = 'SepalWidthCm', data= iris ,hue='Species')

plt.title('Petal Width vs Sepal Width')

plt.show()
sns.countplot(x = 'Species', data = iris)

plt.show()
sns.heatmap(data = iris.corr(),annot=True)

plt.show()
sns.violinplot(y = 'PetalLengthCm', x = 'Species', data= iris, hue='Species')

plt.show()
sns.pairplot(data= iris, hue='Species',palette='Dark2')
np.unique(iris['Species'])
iris['Species'] = pd.Categorical(iris['Species'])

iris['Species'] = iris['Species'].cat.codes.apply(int)

np.unique(iris['Species'])
iris.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(iris.drop('Species',axis=1))

scaled_features = scaler.transform(iris.drop('Species',axis = 1))

iris_feat = pd.DataFrame(scaled_features,columns=iris.columns[:-1])
iris_feat.head()
from sklearn.model_selection import train_test_split
X = iris_feat 

y = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(solver='lbfgs',multi_class='auto')
log.fit(X_train,y_train)
prediction_lr = log.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,prediction_lr))

print(confusion_matrix(y_test,prediction_lr))

print('Accuracy score is',accuracy_score(y_test,prediction_lr))
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
prediction_dt = dt.predict(X_test)
print(confusion_matrix(y_test,prediction_dt))

print(classification_report(y_test,prediction_dt))

print('Accuracy score is',accuracy_score(y_test,prediction_dt))
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=200)
rf.fit(X_train,y_train)
prediction_rf = rf.predict(X_test)
print(confusion_matrix(y_test,prediction_rf))

print(classification_report(y_test,prediction_rf))

print('Accuracy score is',accuracy_score(y_test,prediction_rf))
from sklearn.svm import SVC
sv = SVC(gamma='auto')
sv.fit(X_train,y_train)
prediction_sv = sv.predict(X_test)
print(confusion_matrix(y_test,prediction_sv))

print(classification_report(y_test,prediction_sv))

print('Accuracy score is',accuracy_score(y_test,prediction_sv))
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,.1,.01,.001,.0001]}
gs = GridSearchCV(SVC(),param_grid,verbose=5)
gs.fit(X_train,y_train)
gs.best_params_ #Checking parameters
gs.best_estimator_
prediction_gs = gs.predict(X_test)
print(confusion_matrix(y_test,prediction_gs))

print(classification_report(y_test,prediction_gs))

print('Accuracy score is',accuracy_score(y_test,prediction_gs))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
prediction_knn = knn.predict(X_test)
print(confusion_matrix(y_test,prediction_knn))

print(classification_report(y_test,prediction_knn))

print('Accuracy score is',accuracy_score(y_test,prediction_knn))
error_rate =[]



for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue',linestyle = 'dashed', marker = 'o', markerfacecolor = 'red',markersize = 10)

plt.show()
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
prediction_knn = knn.predict(X_test)
print(confusion_matrix(y_test,prediction_knn))

print(classification_report(y_test,prediction_knn))

print('Accuracy score is',accuracy_score(y_test,prediction_knn))
import tensorflow as tf
iris_feat.columns #columns of iris_feat 
feat_col = []



for col in iris_feat.columns:

    feat_col.append(tf.feature_column.numeric_column(col))
feat_col
input_fn = tf.estimator.inputs.pandas_input_fn(x = X_train, y = y_train,batch_size= 10, num_epochs= 10, shuffle=True)
classifier =tf.estimator.DNNClassifier(hidden_units=[15,15,15],n_classes=3,feature_columns=feat_col)
classifier.train(input_fn,steps= 50)
pred_fn = tf.estimator.inputs.pandas_input_fn(x = X_test,batch_size=len(X_test),shuffle=False)
prediction_dl =list( classifier.predict(input_fn=pred_fn))
prediction_dl
final_pred = []



for pred in prediction_dl:

    final_pred.append(pred['class_ids'][0])
final_pred
print(confusion_matrix(y_test,final_pred))

print(classification_report(y_test,final_pred))

print('Accuracy score is',accuracy_score(y_test,final_pred))