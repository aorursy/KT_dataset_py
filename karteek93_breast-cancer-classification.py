# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
cell_df=pd.read_csv("/kaggle/input/cell_samples.csv")
# There are 699 records and for each record we have multiple parameters that is measured.

# Public Source - https://s3-api.us-geo.objectstorage.s... "

# we are using a dataset that has a 9 predictors in each record, 699 records 
cell_df.head()
cell_df.tail()
# Let us drop the ID column as it doesnot influence the output "class".

cell_df.drop('ID',axis=1,inplace=True)
cell_df.shape
#Missing Or Null data points



cell_df.isnull().sum()

cell_df.isna().sum()
cell_df.count()
cell_df.columns
col_names = ['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit', 'Class']

for x in col_names: 

    print(cell_df[x].nunique())
# Let us check whether the dataset is a balanced or imbalanced one.

target_count = cell_df.Class.value_counts()

print('Benign:', target_count[2])

print('Malignant:', target_count[4])
458/241
cell_df.dtypes
# Identify the unwanted rows



cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'],errors='coerce').notnull()]

cell_df['BareNuc']=cell_df['BareNuc'].astype('int')
cell_df.dtypes
cell_df.shape
cell_df.describe().transpose()
cell_df.hist(figsize=(20,12))

plt.show()
cell_df.columns
# Taking out the predictors and predicted variables seperately for the further.



feature_df=cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',

       'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]

x=np.asarray(feature_df)

y=np.asarray(cell_df['Class'])

x[0:5]
# Divide the data as train and test dataset



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=4)

x_train.shape
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(C=100,random_state=0)

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

correct = (y_test == y_pred).sum()

incorrect = (y_test != y_pred).sum()

accuracy = correct / (correct + incorrect) * 100



print('\nPercent Accuracy: %0.1f' %accuracy)
prediction = pd.DataFrame()

prediction['actual'] = y_test

prediction['predicted'] = y_pred

prediction['correct'] = prediction['actual'] == prediction['predicted']



print ('\nDetailed results for first 20 tests:')

print (prediction.head(20))

#Accuracy of our model.

from sklearn.metrics import confusion_matrix

c_logistic=confusion_matrix(y_test,y_pred)

print(c_logistic)

Accuracy_logistic=sum(np.diag(c_logistic))/(np.sum(c_logistic))

Accuracy_logistic
#Evaluation 

from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
from sklearn.naive_bayes import GaussianNB

classifier_naive=GaussianNB()

classifier_naive.fit(x_train, y_train)

y_predict=classifier_naive.predict(x_test)
#Accuracy of our model.

from sklearn.metrics import confusion_matrix

c_naive=confusion_matrix(y_test,y_predict)

print(c_naive)

Accuracy_naive=sum(np.diag(c_naive))/(np.sum(c_naive))

Accuracy_naive
#Evaluation 

from sklearn.metrics import classification_report

print(classification_report(y_test,y_predict))
#modelling SVM



from sklearn import svm

classifier_svm=svm.SVC(kernel='linear',gamma='auto',C=1)

classifier_svm.fit(x_train,y_train)

y_predict=classifier_svm.predict(x_test)
# Confusion matrix and Accuracy of our model.



from sklearn.metrics import confusion_matrix

c_svm=confusion_matrix(y_test,y_predict)

print(c_svm)

Accuracy_svm=sum(np.diag(c_svm))/(np.sum(c_svm))

Accuracy_svm
#Evaluation 

from sklearn.metrics import classification_report

print(classification_report(y_test,y_predict))
from sklearn import svm

classifier_svmk=svm.SVC(kernel='poly',gamma='auto',C=1)

classifier_svmk.fit(x_train,y_train)

y_predict=classifier_svmk.predict(x_test)
#Accuracy of our model.

from sklearn.metrics import confusion_matrix

c_svmk=confusion_matrix(y_test,y_predict)

print(c_svmk)

Accuracy_svmk=sum(np.diag(c_svmk))/(np.sum(c_svmk))

Accuracy_svmk
#Evaluation 

from sklearn.metrics import classification_report

print(classification_report(y_test,y_predict))
# modelling Knn Classifier



from sklearn.neighbors import KNeighborsClassifier



# nothing but we are using euclidean distance

classifier_knn=KNeighborsClassifier(n_neighbors=6,metric="minkowski",p=2)

classifier_knn.fit(x_train,y_train)

y_predict=classifier_knn.predict(x_test)
# lets see the best value of k for which the model is predicting with high accuracy.



n=[]

acc=[]



for i in range(1,27):

    classifier_knn_trail=KNeighborsClassifier(n_neighbors=i,metric="minkowski",p=2)

    classifier_knn_trail.fit(x_train,y_train)

    c_knn_trail=confusion_matrix(y_test,classifier_knn_trail.predict(x_test))

    acc.append(sum(np.diag(c_knn_trail))/(np.sum(c_knn_trail)))

    n.append(i)

n=np.array(n)

acc=np.array(acc)

plt.plot(n,acc)

plt.show()
#Accuracy of our model.

from sklearn.metrics import confusion_matrix

c_knn=confusion_matrix(y_test,y_predict)

print(c_knn)

Accuracy_knn=sum(np.diag(c_knn))/(np.sum(c_knn))

Accuracy_knn
#Evaluation 

from sklearn.metrics import classification_report

print(classification_report(y_test,y_predict))
# Fitting Decision Tree Classification to the Training set

from sklearn.tree import DecisionTreeClassifier

classifier_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)# for gini 0.948905109489051

classifier_tree.fit(x_train, y_train)



# Predicting the Test set results

y_predict = classifier_tree.predict(x_test)
#Accuracy of our model.

from sklearn.metrics import confusion_matrix

c_tree=confusion_matrix(y_test,y_predict)

print(c_tree)

Accuracy_tree=sum(np.diag(c_tree))/(np.sum(c_tree))

Accuracy_tree
#Evaluation 

from sklearn.metrics import classification_report

print(classification_report(y_test,y_predict))
# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

classifier_ensemble = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier_ensemble.fit(x_train, y_train)



# Predicting the Test set results

y_predict = classifier_ensemble.predict(x_test)
#Accuracy of our model.

from sklearn.metrics import confusion_matrix

c_ensemble=confusion_matrix(y_test,y_predict)

print(c_ensemble)

Accuracy_ensemble=sum(np.diag(c_ensemble))/(np.sum(c_ensemble))

Accuracy_ensemble
#Evaluation 

from sklearn.metrics import classification_report

print(classification_report(y_test,y_predict))
!pip install xgboost
# Fitting the XGBoost to the training set

from xgboost import XGBClassifier

classifier_xg=XGBClassifier()

classifier_xg.fit(x_train,y_train)



# Predicting the test results

y_predictor= classifier_xg.predict(x_test)
# Confusion Matrix

#Accuracy of our model.

from sklearn.metrics import confusion_matrix

c_xg=confusion_matrix(y_test,y_predictor)

print(c_xg)

Accuracy_xg=sum(np.diag(c_xg))/(np.sum(c_xg))

Accuracy_xg
#Evaluation 

from sklearn.metrics import classification_report

print(classification_report(y_test,y_predictor))
d={'Accuracy(%)' : [97.08,94.89,96.35,97.08,98.54,96.35,97.81,97.91],'Precision' : [0.96,0.94,0.95,0.96,0.98,0.96,0.97,0.97],'recall' : [0.98,0.96,0.97,0.98,0.99,0.96,0.98,0.98],'F1 Score' : [0.97,0.95,0.96,0.97,0.98,0.96,0.98,0.98]}

Model_metrics = pd.DataFrame(d,index=['Logistic Regression','Naive Bayes','Svm-Linear','Svm-Polynomial','KNN','Decison Tree','Random Forest','XGBoost'])
Model_metrics
y_predict=classifier_knn.predict(np.array([[1,2,2,5,3,4,6,4,8]]))

print(y_predict)
y_predict=classifier_ensemble.predict(np.array([[1,2,2,5,3,4,6,4,8]]))

print(y_predict)
y_predict=classifier_xg.predict(np.array([[1,2,2,5,3,4,6,4,8]]))

print(y_predict)