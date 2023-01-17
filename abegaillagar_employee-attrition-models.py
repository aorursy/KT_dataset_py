# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sys

import matplotlib.pyplot as plt



%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


filepath='/kaggle/input/Human_Resources_Employee_Attrition.csv'

#data=pd.read_csv(filepath, usecols=[1], engine='python', skipfooter=3)

data=pd.read_csv(filepath, engine='python', skipfooter=3)



data.head()
data.dtypes
#checck for missing values

data.isnull().sum()
features_col=data.columns

print(features_col)
data_features=['satisfaction_level', 'last_evaluation', 'number_of_projects',

       'average_monthly_hours', 'years_at_company', 'work_accident',

       'promotion_last_5years', 'department', 'salary']
#sample histogram

data['department'].hist()

plt.xlabel('department')

plt.ylabel('number of people')
#generate pairplots using seaborn

import seaborn as sn





g=sn.pairplot(x_vars=[ 'last_evaluation', 'number_of_projects',

       'average_monthly_hours', 'years_at_company', 'work_accident', 'left',

       'promotion_last_5years', 'department', 'salary'] ,y_vars='satisfaction_level',data=data,hue='left')

g.fig.set_figheight(6)

g.fig.set_figwidth(15)

#countplots



fig=plt.subplots(figsize=(20,30))

for i , j in enumerate(data_features):

    plt.subplot(5,3,i+1)

    plt.subplots_adjust(hspace=0.5)

    sn.countplot(x=j,data=data,hue='left')
#transform objects to integers

Z=data.salary

Y=data.department

from sklearn import preprocessing

le=preprocessing.LabelEncoder()

salary=le.fit_transform(Z)

department=le.fit_transform(Y)
data['salary_encoded']=salary

data['dept_encoded']=department

data.head()
#drop columns we will not use as feature

data1=data.drop(['left','salary','department'],axis=1)

data1.dtypes
#check for correlation

#tenure is our Y

tenure=data.left

correlations = data1.corrwith(tenure)

correlations.sort_values(inplace=True)

correlations

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(data1,tenure,test_size=0.4,random_state=23)
#some function created:

from sklearn.metrics import accuracy_score





def train_test_model(xtrain,xtest,ytrain,ytest,model,model_name):

    model.fit(xtrain,ytrain)

    ypred=model.predict(xtest)

    sc=accuracy_score(Y_test,ypred)

    print(model_name)

    print('model accuracy:',sc)

    return(ypred)





def same_index(y_predict,ytest):

    Y_pred=[]

    Y_pred2=[]

    val=0

    for i in range(len(y_predict)):

        val=int(y_predict[i])

        Y_pred.append(val)

    #copy index of the Y_test data

    Y_predi=pd.Series(Y_pred,index=ytest.index)

    return(Y_predi)



def show_plot(Y_test,Y_predi):

    plt.plot(Y_test,marker='o',ls='',label='Y_test')

    plt.plot(Y_predi,marker='o',ls='',label='Y_predi')

    plt.legend()

#using KNN



from sklearn.neighbors import KNeighborsClassifier 

model_KNN=KNeighborsClassifier(n_neighbors=25)

resu_KNN=train_test_model(X_train,X_test,Y_train,Y_test,model_KNN,'KNN')

predi_KNN=same_index(resu_KNN,Y_test)

show_plot(Y_test,predi_KNN)
#confusion matrix

from sklearn.metrics import confusion_matrix

cm_KNN=confusion_matrix(Y_test,predi_KNN)

con_KNN=pd.crosstab(Y_test,predi_KNN,rownames=['actual'],colnames=['predicted'])

sn.heatmap(con_KNN,annot=True)

#plt.imshow(cm,cmap='binary',interpolation='None')

plt.show()

cm_KNN
#using RandomforestClassifier

from sklearn.ensemble import RandomForestClassifier 

model_RFC=RandomForestClassifier(n_estimators=100)

resu_RFC=train_test_model(X_train,X_test,Y_train,Y_test,model_RFC,'Random Forest Classifier')

predi_RFC=same_index(resu_RFC,Y_test)

show_plot(Y_test,predi_RFC)


from sklearn.metrics import confusion_matrix

cm_RFC=confusion_matrix(Y_test,predi_RFC)

con_RFC=pd.crosstab(Y_test,predi_RFC,rownames=['actual'],colnames=['predicted'])

sn.heatmap(con_RFC,annot=True)

#plt.imshow(cm,cmap='binary',interpolation='None')

plt.show()

cm_RFC
#importance of our features in the RFC model 



importance=pd.Series(model_RFC.feature_importances_,index=data1.columns)

importance_df=pd.DataFrame(importance)

importance_df.columns=['importance %']

importance_df
ax=importance_df.plot.barh()
#cross validation random forest classifier

from sklearn.model_selection import cross_val_score



RF=RandomForestClassifier(n_estimators=100)

cv_N = 4

scores = cross_val_score(RF, X_train, Y_train, n_jobs=cv_N, cv=cv_N)

print(scores)
#GaussianNB model

from sklearn.naive_bayes import GaussianNB



model_GB=GaussianNB()

resu_GB=train_test_model(X_train,X_test,Y_train,Y_test,model_GB,'Gaussian Naive Bayes')

predi_GB=same_index(resu_GB,Y_test)

show_plot(Y_test,predi_GB)
#cross validation GB

from sklearn.model_selection import cross_val_score



#RF=RandomForestClassifier(n_estimators=100)

cv_N = 4

scores = cross_val_score(model_GB, X_train, Y_train, n_jobs=cv_N, cv=cv_N)

print(scores)