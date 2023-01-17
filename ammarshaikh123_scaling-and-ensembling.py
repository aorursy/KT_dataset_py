# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#importing the libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
#reading the file

data =pd.read_csv('../input/diabetes.csv')

data.head()
#Checking for null values



data.isnull().sum()
#since the output is a binary variable with outcome as 1 or 0 lets see how balance the vairable is



sns.countplot(x='Outcome',data=data)

plt.show()
#let us compare the varibales and see if we can find any relation

sns.pairplot(data=data,hue='Outcome',diag_kind='kde')
from sklearn import svm

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split( data.drop('Outcome',axis=1), data.Outcome, test_size=0.33, random_state=42)
temp=[]

classifiers=['Linear Svm','Radial Svm','Logistic Regression','Decision Tree']

models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),DecisionTreeClassifier()]

for i in models:

    model = i

    model.fit(X_train,y_train)

    prediction=model.predict(X_test)

    temp.append(metrics.accuracy_score(prediction,y_test))

models_df=pd.DataFrame(temp,index=classifiers)   

models_df.columns=['Accuracy']

models_df

sns.heatmap(data[data.columns[:8]].corr(),annot=True,cmap='RdYlGn')

fig=plt.gcf()

fig.set_size_inches(10,7)

plt.show()
#let us run random forest classifier and find out which are the top variables contributing in our model



from sklearn.ensemble import RandomForestClassifier 

model= RandomForestClassifier(n_estimators=100,random_state=0)

X=data[data.columns[:8]]

Y=data['Outcome']

model.fit(X,Y)

pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)
#let us standardize top variable we got from above and train our model again 



cols_to_use=data[['Glucose','BMI','Age','DiabetesPedigreeFunction','Outcome']]

from sklearn.preprocessing import StandardScaler #Standardisation

feat=cols_to_use[cols_to_use.columns[:4]]

feat_standard=StandardScaler().fit_transform(feat)# Gaussian Standardisation

x=pd.DataFrame(feat_standard,columns=[['Glucose','BMI','Age','DiabetesPedigreeFunction']])

x['Outcome']=cols_to_use['Outcome']

outcome=x['Outcome']



X_train, X_test, y_train, y_test = train_test_split( x, outcome ,test_size=0.33, random_state=42)
temp=[]

classifiers=['Linear Svm','Radial Svm','Logistic Regression','Decision Tree']

models=[svm.SVC(kernel='linear'),svm.SVC(kernel='rbf'),LogisticRegression(),DecisionTreeClassifier()]

for i in models:

    model = i

    model.fit(X_train,y_train)

    prediction=model.predict(X_test)

    temp.append(metrics.accuracy_score(prediction,y_test))

new_models_df=pd.DataFrame(temp,index=classifiers)   

new_models_df.columns=['Accuracy_new']

new_models_df
new_models_df=new_models_df.merge(models_df,left_index=True,right_index=True,how='left')
new_models_df['Difference']=new_models_df['Accuracy_new'] - new_models_df['Accuracy']
new_models_df
#let us set prameters for our models 



linear_svc=svm.SVC(kernel='linear',C=0.1,gamma=10,probability=True)

radial_svm=svm.SVC(kernel='rbf',C=0.1,gamma=10,probability=True)

lr=LogisticRegression(C=0.1)
from sklearn.ensemble import VotingClassifier #for Voting Classifier



ensemble_lin_rbf=VotingClassifier(estimators=[('Linear_svm', linear_svc), ('Radial_svm', radial_svm)], 

                       voting='soft', weights=[2,1]).fit(X_train,y_train)

print('The accuracy for Linear and Radial SVM is:',ensemble_lin_rbf.score(X_test,y_test))
ensemble_lin_lr=VotingClassifier(estimators=[('Linear_svm', linear_svc), ('Logistic Regression', lr)], 

                       voting='soft', weights=[2,1]).fit(X_train,y_train)

print('The accuracy for Linear SVM and Logistic Regression is:',ensemble_lin_lr.score(X_test,y_test))
X_train, X_test, y_train, y_test = train_test_split( data.drop('Outcome',axis=1), data.Outcome ,test_size=0.33, random_state=42)
ensemble_lin_rbf=VotingClassifier(estimators=[('Linear_svm', linear_svc), ('Radial_svm', radial_svm)], 

                       voting='soft', weights=[2,1]).fit(X_train,y_train)

print('The accuracy for Linear and Radial SVM is:',ensemble_lin_rbf.score(X_test,y_test))
ensemble_lin_lr=VotingClassifier(estimators=[('Linear_svm', linear_svc), ('Logistic Regression', lr)], 

                       voting='soft', weights=[2,1]).fit(X_train,y_train)

print('The accuracy for Linear SVM and Logistic Regression is:',ensemble_lin_lr.score(X_test,y_test))
ensemble_rad_lr_lin=VotingClassifier(estimators=[('Radial_svm', radial_svm), ('Logistic Regression', lr),('Linear_svm',linear_svc)], 

                       voting='soft', weights=[2,1,3]).fit(X_train,y_train)

print('The ensembled model with all the 3 classifiers is:',ensemble_rad_lr_lin.score(X_test,y_test))