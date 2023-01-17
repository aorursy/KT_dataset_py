# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load libraries 



# Data Analysis libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# disable warnings signals

import warnings

warnings.filterwarnings("ignore")
# Data Gathering from Kaggle

df = pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')

df2 = pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv')
# 3 important preprocess before you start your project 



# 1.Duplication

print(sum(df.duplicated()))

# No duplication



print('\n')



# 2.Missing value

print(df.isna().sum())

# No missing value



print('\n')



# 3.Correct datatype

df.info()

# correct datatype (float and string)
# Data Assessing 



# 1. Head: first 5 rows 

df.head()
# 2.Tail: last 5 rows

df.tail()
# 3.shape: The number of row and column

df.shape

# 310 rows and 7 columns
# 4.column : detail about column informations

df.columns
# 5.describe: statistical value of each columns

df.describe()
# Univariate Visualization (use one variable for visualization)

# Countplot 

ax = sns.countplot(data=df,x='class')

plt.title('Number of each class'.title(),

         fontsize=12,weight='bold')

plt.show()
# Pie chart  

sorted_counts = df['class'].value_counts()

ax=plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 90,

        counterclock = False,pctdistance=0.8 ,wedgeprops = {'width' : 0.4}, autopct='%1.0f%%');

plt.title('Different type of class in percentage')

plt.axis('square');
x = df.iloc[:,:-1]

# x is input data and it means sub features that we're going to use as a factor of orthopedic patients

y = df['class']

# y is output data and it means the result of sub feature combination



# standardization

stand = (x - x.mean()) / (x.std())
df.replace({"Abnormal": 0, "Normal": 1},inplace=True)
# Bivariate Visualization (use 2 variables for visualization)

data = pd.concat([y,stand],axis=1)





data = pd.melt(data,id_vars="class",

                    var_name="features",

                    value_name='value')



plt.subplots(figsize=(15,7))

sns.violinplot(data=data,x='features',y='value',split=True,hue='class',inner='quart')

plt.title('Standardized sub feature wit violinplot'.title(),

          fontsize = 14,weight='bold')



plt.legend(bbox_to_anchor=(1.16,1.02),title="Type",

          fontsize = 13)



plt.xlabel('sub features'.title(),

            fontsize=14,weight="bold")



plt.ylabel('Z-Score'.title(),

          fontsize=14,weight="bold");





#https://stackoverflow.com/questions/43585333/seaborn-countplot-set-legend-for-x-values
data = pd.concat([y,stand],axis=1)





data = pd.melt(data,id_vars="class",

                    var_name="features",

                    value_name='value')



plt.subplots(figsize=(15,7))

sns.pointplot(data=data,x='features',y='value',split=True,hue='class')

plt.title('Standardized sub feature with pointplot'.title(),

          fontsize = 14,weight='bold')



plt.legend(bbox_to_anchor=(1.16,1.02),title="Type",

          fontsize = 13)



plt.xlabel('sub features'.title(),

            fontsize=14,weight="bold")



plt.ylabel('Z-Score'.title(),

          fontsize=14,weight="bold");
# Multivariate Visualization (use more than 2 variables for visualization)

x = df.iloc[:,:-1]

y = df['class']

stand = (x - x.mean()) / (x.std())

#This time we'll not concatnate, but we're goint to merge standardized value into the dataframe

df_new = pd.merge(y,stand, right_index=True, left_index=True)
## Jointplot 

sns.jointplot(data=df_new,x='pelvic_incidence',y='degree_spondylolisthesis',kind="reg");

# pelvic_incidence vs degree_spondylolisthesis
sns.jointplot("pelvic_radius", "lumbar_lordosis_angle", data=df_new,kind="kde", space=0, color="g");

# pelvic_radius vs lumbar_lordosis_angle 
# plot with different size of scatter

plt.subplots(figsize=(12,7))

type_marker =[['0','o'],['1','^']]



for ttype,marker in type_marker:

    sns.regplot(data=df_new,x='pelvic_incidence',y='lumbar_lordosis_angle',x_jitter=0.04,marker=marker,fit_reg=False);



plt.xlabel('pelvic_incidence');

plt.ylabel('lumbar_lordosis_angle')

plt.legend(['Abnormal','Normal'],title='Class',

          fontsize = 13, bbox_to_anchor= (1.2,1))

plt.title("pelvic_incidence vs lumbar_lordosis_angle with different ");
# Load sci-kit learn libraries 



from sklearn.linear_model import LogisticRegression 

# Logistic Regression

from sklearn import svm

# Support vector machine

from sklearn.ensemble import RandomForestClassifier

# Random Forests

from sklearn.neighbors import KNeighborsClassifier

# KNN

from sklearn.naive_bayes import GaussianNB

# Naive Bayes

from sklearn.tree import DecisionTreeClassifier

# Decision Tree



# For split the dataset 

from sklearn.model_selection import train_test_split

from sklearn import metrics 

train,test =train_test_split(df,test_size=0.2,random_state=0,stratify=df['class'])



# Training_data:80% and Test_data:20%



train_X = train[train.columns[:-1]]

train_Y = train[train.columns[-1]]

test_X = test[test.columns[:-1]]

test_Y = test[test.columns[-1]]



# Define X,Y columns of train and test dataset
model = LogisticRegression()

model.fit(train_X,train_Y)

prediction = model.predict(test_X)

print('Accuracy of Support vector machine is',metrics.accuracy_score(prediction,test_Y))
model = svm.SVC(kernel='rbf',gamma=0.1,C=1)

model.fit(train_X,train_Y)

prediction = model.predict(test_X)

print('Accuracy of Support vector machine is',metrics.accuracy_score(prediction,test_Y))
model = svm.SVC(kernel='linear',gamma=0.1,C=1)

model.fit(train_X,train_Y)

prediction = model.predict(test_X)

print('Accuracy of linear Support vector machine is',metrics.accuracy_score(prediction,test_Y))
model = RandomForestClassifier(random_state=0)

model.fit(train_X,train_Y)

prediction = model.predict(test_X)

print('Accuracy of Random Forest is',metrics.accuracy_score(prediction,test_Y))



# If you don't define any parameter value, the number of estimator is by default 100 
model = KNeighborsClassifier()

model.fit(train_X,train_Y)

prediction = model.predict(test_X)

print('Accuracy of K Nearest Neighbors is',metrics.accuracy_score(prediction,test_Y))



# If you don't define any parameter value,by default the k is allocated into k=5
model = GaussianNB()

model.fit(train_X,train_Y)

prediction = model.predict(test_X)

print('Accuracy of Gaussian Naive Bayes is',metrics.accuracy_score(prediction,test_Y))
model = DecisionTreeClassifier()

model.fit(train_X,train_Y)

prediction = model.predict(test_X)

print('Accuracy of Decision Tree is',metrics.accuracy_score(prediction,test_Y))
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

# score evaluation

from sklearn.model_selection import cross_val_predict

# prediction

kfold = KFold(n_splits=20,random_state=10)





# Now we have a Cross Validation model with 10 iterations.In the next part we'll iterate 20 K-Fold Cross validation and get a mean of accuracies

# Advandatage : We don't need to train_test split because if we split data into k_subset we'll automatically (1/k)percentage as a test data size.

# Our case : 1/20 = 5%
# First we need empty list to fill out iterated data



accuracy_mean = []

accuracy_std = []

accuracy = []





x = df.iloc[:,:-1]

y = df['class']





classifiers = ['Logistic Regression','SVM','Linear SVM','Random Forest','KNN','Naive Bayes','Decision Tree']

models =[LogisticRegression(),svm.SVC(kernel='rbf'),svm.SVC(kernel='linear'),RandomForestClassifier(random_state=0),

         KNeighborsClassifier(),GaussianNB(),DecisionTreeClassifier()]





for i in models :

    cv_result = cross_val_score(i,x,y,cv=kfold,scoring='accuracy')

    accuracy_mean.append(cv_result.mean())

    accuracy_std.append(cv_result.std())

    accuracy.append(cv_result)

accuracy_table = pd.DataFrame({'CV mean':accuracy_mean,'CV std':accuracy_std},index=classifiers)

accuracy_table



plt.subplots(figsize=(15,7))

sns.pointplot(x=accuracy_table.index,y=accuracy_table['CV mean'])

plt.title('Cross Validation accuracies mean'.title(),

          fontsize = 14,weight='bold')



plt.xlabel('Classifier'.title(),

            fontsize=14)



plt.ylabel('accuracies mean'.title(),

          fontsize=14);
#SVM with GridSearchCV

from sklearn.model_selection import GridSearchCV

C=[0.05,0.1,0.2,0.3,0.25,0.4,0.5,0.6,0.7,0.8,0.9,1]

gamma=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

kernel=['rbf','linear']

hyper={'kernel':kernel,'C':C,'gamma':gamma}

gd=GridSearchCV(estimator=svm.SVC(),param_grid=hyper,verbose=True)

gd.fit(x,y)

print(gd.best_score_)

print(gd.best_estimator_)
#Random Forest with Random search 

from sklearn.model_selection import RandomizedSearchCV



dict = {'n_estimators':[100,200,300,400,500],

        'max_depth': [3,5,7,10,None],

        'criterion':['gini','entropy'],

        'bootstrap':[True,False],

        'max_leaf_nodes':[3,5,7,10,None]}

gd=RandomizedSearchCV(estimator=RandomForestClassifier(random_state=0),param_distributions=dict)

gd.fit(x,y)

print(gd.best_score_)

print(gd.best_estimator_)
# Soft Voting

from sklearn.ensemble import VotingClassifier

ensemble_lin_rbf=VotingClassifier(estimators=[('KNN',KNeighborsClassifier(n_neighbors=10)),

                                              ('RBF',svm.SVC(probability=True,kernel='rbf',C=0.5,gamma=0.1)),

                                              ('RFor',RandomForestClassifier(n_estimators=500,random_state=0)),

                                              ('LR',LogisticRegression(C=0.05)),

                                              ('DT',DecisionTreeClassifier(random_state=0)),

                                              ('NB',GaussianNB()),

                                              ('svm',svm.SVC(kernel='linear',probability=True))

                                             ], 

                       voting='soft').fit(train_X,train_Y)



# voting parameter : hard or soft 

# hard : majority rule voting

# soft : predicted probabilities,which is recommended for a ensemble of well-calibrated classifiers.

print('The accuracy for ensembled model is:',ensemble_lin_rbf.score(test_X,test_Y))

cross=cross_val_score(ensemble_lin_rbf,x,y, cv = 10,scoring = "accuracy")

# cv is the number of spliting 

print('The cross validated score is',cross.mean())
# Bagging with Decision Tree



# Remember 

'''

train_X = train[train.columns[:-1]]

train_Y = train[train.columns[-1]]

test_X = test[test.columns[:-1]]

test_Y = test[test.columns[-1]]

'''







from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=0,n_estimators=700)

model.fit(train_X,train_Y)

prediction=model.predict(test_X)

print('The accuracy of Decision Tree is :',metrics.accuracy_score(prediction,test_Y))

result =cross_val_score(model,x,y,cv=10,scoring='accuracy')

print('The cross validated score for Decision Tree is :',result.mean())
from sklearn.ensemble import AdaBoostClassifier



model = AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)

# By defalut of base_estimators it's decision tree automatically 

# Learning rate is a tuning parameter in an optimization parameter algoritm that determines the step size at each iteration 

# while moving toward a minimum of a loss fuction. Learning rate shrinks the contribution of each classifier 

result = cross_val_score(model,x,y,cv=10,scoring="accuracy")

print('The cross validated score of AdaBoost is :',result.mean())
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,random_state=0)

result = cross_val_score(model,x,y,cv=10,scoring="accuracy")

print('The cross validated score of Gradient decent Boosting is: ',result.mean())
from xgboost import XGBClassifier

xgboost=XGBClassifier(n_estimators=500,learning_rate=0.1)

result=cross_val_score(xgboost,x,y,cv=10,scoring='accuracy')

print('The cross validated score for XGBoost is:',result.mean())
f,ax = plt.subplots(1,2,figsize=(15,8))



# Deicsion Tree

model =  DecisionTreeClassifier(random_state=0)

model.fit(x,y)

features = pd.Series(

     model.feature_importances_,

    index=x.columns).sort_values(ascending=False)

base_color = sns.color_palette()[0]

sns.barplot(x=features.index,y=features,ax=ax[0],color=base_color)

ax[0].set_title('Decision Tree feature importance')

ax[0].set_xticklabels(ax[0].get_xticklabels(),rotation=45,horizontalalignment='right')





# Random Forest

model = RandomForestClassifier(random_state=0)

model.fit(x,y)

features = pd.Series(

        model.feature_importances_,

    index=x.columns).sort_values(ascending=False)

base_color = sns.color_palette()[1]

sns.barplot(x=features.index,y=features,ax=ax[1],color=base_color)

ax[1].set_title('Random Forest feature importance')

ax[1].set_xticklabels(ax[0].get_xticklabels(),rotation=45,horizontalalignment='right')

f,ax = plt.subplots(1,2,figsize=(15,8))



# AdaBoost

model = AdaBoostClassifier(n_estimators=200,random_state=0,learning_rate=0.1)

model.fit(x,y)

features = pd.Series(

        model.feature_importances_,

    index=x.columns).sort_values(ascending=False)

base_color = sns.color_palette()[0]

sns.barplot(x=features.index,y=features,ax=ax[0],color=base_color)

ax[0].set_title('AdaBoost feature importance')

ax[0].set_xticklabels(ax[0].get_xticklabels(),rotation=45,horizontalalignment='right')







# XGBoost

model = AdaBoostClassifier(n_estimators=500,learning_rate=0.1)

model.fit(x,y)

features = pd.Series(

        model.feature_importances_,

    index=x.columns).sort_values(ascending=False)

base_color = sns.color_palette()[1]

sns.barplot(x=features.index,y=features,ax=ax[1],color=base_color)

ax[1].set_title('XGBoost feature importance')

ax[1].set_xticklabels(ax[0].get_xticklabels(),rotation=45,horizontalalignment='right');