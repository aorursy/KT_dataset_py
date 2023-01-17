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
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
data=pd.read_csv("/kaggle//input/heart-disease-uci/heart.csv")
data.head()
#age - age in years
#sex - (1 = male; 0 = female)
#cp - chest pain type(#1 = typical angina,#2 = atypical angina,#3 = non — anginal pain,#4 = asymptotic)
#trestbps - resting blood pressure (in mm Hg on admission to the hospital)
#chol - serum cholestoral in mg/dl
#fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
#restecg - resting electrocardiographic results(0 = normal, 1 = having ST-T wave abnormality,2 = left ventricular hyperthrophy)
#thalach - maximum heart rate achieved
#exang - exercise induced angina (1 = yes; 0 = no)
#oldpeak - ST depression induced by exercise relative to rest
#slope - the slope of the peak exercise ST segment
#ca - number of major vessels (0-3) colored by flourosopy
#thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
#target - have disease or not (1=yes, 0=no)
data.shape
#Checking for null values in the dataset
data.isnull().sum()
#We can see there are no null values in the dataset
#To get insights about the dataset.
data.describe()
#We can see in some columns mean<median and mean>median, this indicates presence of outliers
#Importing stats tool to check the z score of each data in the dataset
from scipy import stats
z = np.abs(stats.zscore(data))
print(z)
#The Z-score is the signed number of standard deviations by which the value of an observation or data point is above the mean value of what is being observed or measured.
#In most cases z score lies between -3 and +3 if any data point's z score doesn't fit in this interval, then they are classified as outliers
threshold = 3 #setting the threshold at 3
print(np.where(z > 3)) #finding what all data exceeded the treshold
z[28][4]# Just an example of a data exceeding z=3 since the above result gives a 2-d Array
data= data[(z < 3).all(axis=1)]# Only including data which have z score<3 which is a dataset free of outliers
data.shape#Checking the shape of dataset after outlier removal, we can see it that some rows have been reduced
#We can see people having disease is some what more than and people not having disease
import seaborn as sns
sns.countplot(data['target'])
#We can see that majority all males and they are most affected to heart disease.
sns.countplot(data['sex'])
#Variation of age with target 
sns.countplot(x=data['age'],hue=data['target'])
#We can see variation of gender with target
#We can see among men percentage of having diesease is less compared to female where their percent is higher.
sns.countplot(x=data['sex'],hue=data['target'])
#RSeparating features and target.
y=data['target']
X=data.drop(['target'],axis=1)
X1=X
from sklearn import preprocessing
#Scaling (Standarisation)
X_1=preprocessing.scale(X)
X_1
#Splitting training and test data for X1 for models like KNN, Naive_Bayes, Logistic Regression and SVM
xtrain1,xtest1,ytrain1,ytest1=train_test_split(X_1,y,test_size=0.3,random_state=7)
#Splitting for decision tree, random forest with original data because there is no need for scaling the data for these models as they work on rules
xtrain2,xtest2,ytrain2,ytest2=train_test_split(X,y,test_size=0.3,random_state=7)
#Using KNN Classifier with X_1
#KNN Classifier for various values of n to know which n gives best model
acc=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(xtrain1,ytrain1)
    pred_i=knn.predict(xtest1)
    acc.append(accuracy_score(ytest1,pred_i))
acc
print('In KNN classifier for n={} we got the best accuracy of {}'.format(acc.index(max(acc)),max(acc)))
#Perfromance metrics using KNN best
kn1=KNeighborsClassifier(n_neighbors=12)
kn1.fit(xtrain1,ytrain1)
pred_knn=kn1.predict(xtest1)
print(classification_report(ytest1,pred_knn))
#Now using Logistic Regression
log=LogisticRegression(max_iter=1000)
log.fit(xtrain1,ytrain1)
log_pred=log.predict(xtest1)
print('Using Logisitic Regression we got accuracy of {}'.format(accuracy_score(ytest1,log_pred)))
print(classification_report(ytest1,log_pred))
#Now using Naive Bayes Classifier
naive=GaussianNB()
naive.fit(xtrain1,ytrain1)
naive_pred=naive.predict(xtest1)
print('Using Naive Bayes algorithm we got accuracy of {}'.format(accuracy_score(ytest1,naive_pred)))
print(classification_report(ytest1,naive_pred))
#Now using Decision Tree Classifier
tree=DecisionTreeClassifier(criterion='entropy')
tree.fit(xtrain2,ytrain2)
tree_pred=tree.predict(xtest2)
print('Using Decision Tree algorithm we got accuracy of {}'.format(accuracy_score(ytest2,tree_pred)))
print(classification_report(ytest2,tree_pred))
#Using Random Forest Classifier
clf=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=1,max_depth=10)
clf.fit(xtrain2,ytrain2)
clf_pred=clf.predict(xtest2)
print('Using Random Forest Classifier we got accuracy of {}'.format(accuracy_score(ytest2,clf_pred)))
print(classification_report(ytest2,clf_pred))
#Finding feature importance using feature_importances command with Random forest classifier model.
feature_imp=pd.Series(clf.feature_importances_,index=X1.columns).sort_values(ascending=False)
feature_imp
#Getting a dataset with importance above 0.08 and fitting them
from sklearn.feature_selection import SelectFromModel
feat_sel=SelectFromModel(clf,threshold=0.08)
feat_sel.fit(xtrain2,ytrain2)
#Splittng the data as x_train and x_test to train only important feature
x_train=feat_sel.transform(xtrain2)
x_test=feat_sel.transform(xtest2)
#Using Random Forest only for important features
clf_imp=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=1,max_depth=7)
clf_imp.fit(x_train,ytrain2)
imp_pred=clf_imp.predict(x_test)
print('Using Random forest with for important features we got an accuracy of {}'.format(accuracy_score(ytest2,imp_pred)))
print(classification_report(ytest2,imp_pred))
#Feeding manual parameters to grid search CV and evaluating using SVM
param_grid = {'C': [0.1,1, 5, 8, 10, 50,100, 1000], 
      'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
      'kernel': ['rbf']} 
 
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
grid.fit(xtrain1,ytrain1)

#getting best paramter of the grid search
grid.best_params_
#Accuracy from SVM
grid_svm_pred=grid.predict(xtest1)
print('Using SVM with Grid Search CV we got an accuracy of {}'.format(accuracy_score(ytest1,grid_svm_pred)))
print(classification_report(ytest1,grid_svm_pred))