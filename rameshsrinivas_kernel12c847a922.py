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
# read data from csv

import pandas as pd 

df = pd.read_csv("../input/amd-vs-intel/AMDvIntel.csv")

df.info()
df.head()
df.describe()
# Drop Name and price because it doesnt impact for classification

df = df.drop(['Name','Price'],axis=1)

df.head()
# Analyse all the features

df.hist(bins = 20,figsize=(20,20))

df.plot()
# Analyze how each feature influence the Target variable(IorA)



import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(15,20))



plt.subplot(2,2,1)

sns.barplot(x = df['IorA'], y = df['Cache(M)'])



plt.subplot(2,2,2)

sns.barplot(x = df['IorA'], y = df['Cores'])



plt.subplot(2,2,3)

sns.barplot(x = df['IorA'], y = df['Threads'])



plt.subplot(2,2,4)

sns.barplot(x = df['IorA'], y = df['Speed(GHz)'])



plt.show()
# Analyse individual feature count



plt.figure(figsize=(15,30))



plt.subplot(4,1,1)

df['Cache(M)'].value_counts().plot(kind='bar')



plt.subplot(4,1,2)

df.Cores.value_counts().plot(kind='bar')



plt.subplot(4,1,3)

df.Threads.value_counts().plot(kind='bar')



plt.subplot(4,1,4)

df['Speed(GHz)'].value_counts().plot(kind='bar')





plt.show()
# Analyze how each field in the feature influence the Target variable(IorA)

plt.figure(figsize=(15,30))



plt.subplot(4,1,1)

sns.countplot(x = 'Cache(M)',hue = 'IorA',data = df)



plt.subplot(4,1,2)

sns.countplot(x = 'Cores',hue = 'IorA',data = df)



plt.subplot(4,1,3)

sns.countplot(x = 'Threads',hue = 'IorA',data = df)



plt.subplot(4,1,4)

sns.countplot(x = 'Speed(GHz)',hue = 'IorA',data = df)



plt.show()
# Correlation between the variables

df.corr().style.format("{:.2}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
df.info()
df.columns
# Define X and y

X = df.iloc[:,1:5] 

X.head()
y = df.loc[:,['IorA']]

y.head()
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)
X_train.shape
X_test.shape
y_train.shape
y_test.shape
# Apply Model



# Training the model

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score



model_logr  = LogisticRegression()

model_logr .fit(X_train,y_train)
# Predicting the model

y_predict_log = model_logr.predict(X_test)
# Test accuracy

print('Logistics Test Accuracy:', accuracy_score(y_test,y_predict_log))
# Precision, Recall



from sklearn.metrics import classification_report

print(classification_report(y_test,y_predict_log))
# Confusion Matrix

confusion_matrix(y_test,y_predict_log)
# Training the model

from sklearn.svm import SVC

from sklearn.model_selection import RandomizedSearchCV



parameters = {'kernel': ['rbf'], 'gamma': [0.1,1,5], 'C': [0.1,1,10,100]}



rbf_svc = RandomizedSearchCV(SVC(),parameters).fit(X_train,y_train)
print('Best Parameter',rbf_svc.best_params_)
# Predicting the model

y_predict_svm = rbf_svc.predict(X_test)
# Test accuracy

print('SVM Test Accuracy:', accuracy_score(y_test,y_predict_svm))
# Precision, Recall

print(classification_report(y_test,y_predict_svm))
# Confusion Matrix

confusion_matrix(y_test,y_predict_svm)
# Training the model

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV



classifier_dtg=DecisionTreeClassifier(random_state=42,splitter='best')

parameters=[{'min_samples_split':[2,3,4,5],'criterion':['gini']},{'min_samples_split':[2,3,4,5],'criterion':['entropy']}]



model_dectree=GridSearchCV(estimator=classifier_dtg, param_grid=parameters, scoring='accuracy',cv=10)

model_dectree.fit(X_train,y_train)
print('Best Estimator:',model_dectree.best_estimator_)
print('Best Parameter:',model_dectree.best_params_)
print('Best Score:',model_dectree.best_score_)
# Predicting the model

y_predict_dtree = model_dectree.predict(X_test)
# Test accuracy

print('DT GS Test Accuracy:', accuracy_score(y_test,y_predict_dtree))
# Precision, Recall

print(classification_report(y_test,y_predict_dtree))
# Confusion Matrix

confusion_matrix(y_test,y_predict_dtree)
# Training the model

from sklearn.ensemble import RandomForestClassifier



classifier_rfg=RandomForestClassifier(random_state=33,n_estimators=23)

parameters=[{'min_samples_split':[2,3,4,5],'criterion':['gini','entropy'],'min_samples_leaf':[1,2,3]}]



model_gridrf=GridSearchCV(estimator=classifier_rfg, param_grid=parameters, scoring='accuracy',cv=10)

model_gridrf.fit(X_train,y_train)
print('Best Estimators:',model_gridrf.best_estimator_)
print('Best Parameters:',model_gridrf.best_params_)
print('Best Score:',model_gridrf.best_score_)
# Predict the model

y_predict_rf = model_gridrf.predict(X_test)
# Test accuracy

print('RF Test Accuracy:', accuracy_score(y_test,y_predict_rf))
# Precision, Recall

print(classification_report(y_test,y_predict_rf))
# Confusion Matrix

confusion_matrix(y_test,y_predict_rf)
# Training the model

from sklearn.naive_bayes import BernoulliNB

model_nb = BernoulliNB()

model_nb.fit(X_train,y_train)
# Predict the model

y_predict_nb = model_nb.predict(X_test)
# Test accuracy

print('NB Test Accuracy:', accuracy_score(y_test,y_predict_nb))
# Precision, Recall

print(classification_report(y_test,y_predict_nb))
# Confusion Matrix

confusion_matrix(y_test,y_predict_nb)
# Training the model

from sklearn.neighbors import KNeighborsClassifier

model_knn = KNeighborsClassifier(n_neighbors=6,metric='euclidean') # Maximum accuracy for n=10

model_knn.fit(X_train,y_train)
# Predicting the model

y_predict_knn = model_knn.predict(X_test)
# Test accuracy

print('Test Accuracy:', accuracy_score(y_test,y_predict_knn))
# Precision, Recall

print(classification_report(y_test,y_predict_knn))
# Confusion Matrix

confusion_matrix(y_test,y_predict_knn)
# Training the model

from xgboost import XGBClassifier

model_xgb = XGBClassifier(max_depth=5,

                     n_estimators=100,

                     subsample=.8,

                     learning_rate=0.1,

                     reg_alpha=0,

                     reg_lambda=1,

                     colsample_bynode=0.6,

                     colsample_bytree=0.5,

                     gamma = 0)

model_xgb.fit(X_train,y_train)
# Predicting the model

y_predict_xgb = model_xgb.predict(X_test)
# Test accuracy

print('XGB Test Accuracy:', accuracy_score(y_test,y_predict_xgb))
# Precision, Recall

print(classification_report(y_test,y_predict_xgb))
# Confusion Matrix

confusion_matrix(y_test,y_predict_xgb)
# Training the model

from sklearn.neural_network import MLPClassifier

model_mlp = MLPClassifier(hidden_layer_sizes=(100,100,100),batch_size=10,learning_rate_init=0.01,max_iter=2000,random_state=10)

model_mlp.fit(X_train,y_train)
# Predicting the model

y_predict_mlp = model_mlp.predict(X_test)
# Test accuracy

print('ANN Test Accuracy:', accuracy_score(y_test,y_predict_mlp))
# Precision, Recall

print(classification_report(y_test,y_predict_mlp))
# Confusion Matrix

confusion_matrix(y_test,y_predict_mlp)