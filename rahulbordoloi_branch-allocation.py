# Get File
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#downgrading colab 
!pip install -r "requirements.txt"
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
df = pd.read_csv('/kaggle/input/kiit-counselling-dataset/Branch_Allocation_Final.csv',error_bad_lines=False) 
df
sns.pairplot(df)
df['Nationality'].unique()
df.info()
df.describe()
x = df.iloc[:,:5]
x
y=df.iloc[:,-1]
type(y)
#encoding categorical data
from sklearn.preprocessing import LabelEncoder
l = LabelEncoder()
x.iloc[:,-1]=l.fit_transform(x.iloc[:,-1])           #gender
y.iloc[:] = l.fit_transform(y.iloc[:])             #department  
x.iloc[:,1] = l.fit_transform(x.iloc[:,1])         #nationality
x
y
def change(row):
    if row['12th'] >= 65 and row['10th'] >= 75:
      return 1
    else:
      return 0
#adding admission column
x['Adm']=x.apply(change,axis=1)
x
x.head(5)
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x = sc.fit_transform(x)
x
y.shape
y=y.values.reshape(-1,1)
#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size = 0.25, random_state = 0, shuffle = True)
x_train.shape, y_train.shape
x_test.shape, y_test.shape
from sklearn import utils
print(utils.multiclass.type_of_target(x_train.astype('int')))
#fitting random forest classifier to the training set
from sklearn.ensemble import RandomForestClassifier as rfc
classifier = rfc(n_estimators=100,criterion='entropy',random_state=0)
classifier.fit(x_train, y_train)
#predicting the test set results
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix, classification_report

cm=confusion_matrix(y_test, y_pred)
plt.figure(figsize = (5,5))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
print(classification_report(y_test, y_pred))
#applying k-fold cross validation
from sklearn.model_selection import cross_val_score as cvs
accuracies = cvs(estimator=classifier,X=x_train,y=y_train,cv=10)
print(accuracies.mean())
print(accuracies.std())
#fitting logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)
#predicting the test set results
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix, classification_report

cm=confusion_matrix(y_test, y_pred)
plt.figure(figsize = (5,5))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
print(classification_report(y_test, y_pred))
#applying k-fold cross validation
from sklearn.model_selection import cross_val_score as cvs
accuracies = cvs(estimator=classifier,X=x_train,y=y_train,cv=10)
print(accuracies.mean())
print(accuracies.std())
#fitting kernel SVM to the training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(x_train, y_train)
#predicting the test set results
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix, classification_report

cm=confusion_matrix(y_test, y_pred)
plt.figure(figsize = (5,5))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
print(classification_report(y_test, y_pred))
#applying k-fold cross validation
from sklearn.model_selection import cross_val_score as cvs
accuracies = cvs(estimator=classifier,X=x_train,y=y_train,cv=10)
print(accuracies.mean())
print(accuracies.std())
#fitting kernel SVM to the training set
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(x_train, y_train)
#predicting the test set results
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix, classification_report

cm=confusion_matrix(y_test, y_pred)
plt.figure(figsize = (5,5))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
print(classification_report(y_test, y_pred))
#applying k-fold cross validation
from sklearn.model_selection import cross_val_score as cvs
accuracies = cvs(estimator=classifier,X=x_train,y=y_train,cv=10)
print(accuracies.mean())
print(accuracies.std())
from sklearn.neighbors import KNeighborsClassifier as knc
classifier=knc(n_neighbors=10,metric='minkowski',p=2)
classifier.fit(x_train, y_train)
#predicting the test set results
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix, classification_report

cm=confusion_matrix(y_test, y_pred)
plt.figure(figsize = (5,5))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
print(classification_report(y_test, y_pred))
#applying k-fold cross validation
from sklearn.model_selection import cross_val_score as cvs
accuracies = cvs(estimator=classifier,X=x_train,y=y_train,cv=10)
print(accuracies.mean())
print(accuracies.std())
#fitting decision tree classifier to the training set
from sklearn.tree import DecisionTreeClassifier as dtc
classifier = dtc(criterion='entropy' , random_state=0)
classifier.fit(x_train, y_train)
#predicting the test set results
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix, classification_report

cm=confusion_matrix(y_test, y_pred)
plt.figure(figsize = (5,5))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
print(classification_report(y_test, y_pred))
#applying k-fold cross validation
from sklearn.model_selection import cross_val_score as cvs
accuracies = cvs(estimator=classifier,X=x_train,y=y_train,cv=10)
print(accuracies.mean())
print(accuracies.std())
#fitting naive bayes to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
#predicting the test set results
y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix, classification_report

cm=confusion_matrix(y_test, y_pred)
plt.figure(figsize = (5,5))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
print(classification_report(y_test, y_pred))
#applying k-fold cross validation
from sklearn.model_selection import cross_val_score as cvs
accuracies = cvs(estimator=classifier,X=x_train,y=y_train,cv=10)
print(accuracies.mean())
print(accuracies.std())
#fitting decision tree classifier to the training set
from sklearn.tree import DecisionTreeClassifier as dtc
classifier = dtc(criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)
#applying grid search to find the best model and best parameters
from sklearn.model_selection import GridSearchCV as gsv
parameters = [{'splitter':['best','random'],'criterion':['entropy','gini'],'max_depth':['None',2,4,6]}]
grid_search=gsv(estimator=classifier,
                param_grid=parameters,
                scoring='accuracy',
                cv=10,
                n_jobs=1)
grid_search=grid_search.fit(x_train,y_train)
print('best_accuracy=',grid_search.best_score_)
print('best_parameters=',grid_search.best_params_)
#fitting decision tree classifier to the training set
from sklearn.tree import DecisionTreeClassifier as dtc
classifier = dtc(criterion='entropy' ,random_state=0, splitter='best', max_depth=4)
classifier.fit(x_train, y_train)
#predicting the test set results
y_pred=classifier.predict(x_test)
cm=confusion_matrix(y_test, y_pred)
plt.figure(figsize = (10,10))
sns.heatmap(cm, annot=True)
#observating classification report for performance evaluation
print(classification_report(y_test, y_pred))
import pickle
filename = 'branch_allocation.sav'
pickle.dump(dtc, open(filename, 'wb'))
from sklearn.pipeline import Pipeline
pipe = Pipeline([('standard', StandardScaler()),
                    #('boxcox'), stats.boxcox()),
                    ('l-svm', dtc())])
pipe.fit(x_train, y_train)
score = pipe.score(x_test, y_test)
print('Decision Tree pipeline test accuracy: %.3f' % score)
!pip freeze > requirements.txt