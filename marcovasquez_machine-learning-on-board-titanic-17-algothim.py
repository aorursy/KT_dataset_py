#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline



 #foundational package for scientific computing

import numpy as np



#collection of functions for data processing and analysis modeled after R dataframes with SQL like features

import pandas as pd



#collection of functions for scientific and publication-ready visualization

import matplotlib.pyplot as plt



#Visualization

import seaborn as sns



#collection of functions for scientific computing and advance mathematics

import scipy as sp



#collection of machine learning algorithms

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix



#Common Model Helpers

from sklearn.preprocessing import LabelEncoder

from sklearn import metrics

from sklearn import model_selection

import pylab as pl

from sklearn.metrics import roc_curve



#ignore warnings

import warnings

warnings.filterwarnings('ignore')
datatrain =  pd.read_csv('../input/titanic/train.csv')
datatest =  pd.read_csv('../input/titanic/test.csv')
datatrain.head()
datatest.head()
datatestcopy = datatest.copy()
print(datatrain.shape)

print(datatest.shape)
datatrain.info()
print('Train columns with null values: {} \n' .format( datatrain.isnull().sum()))



print('Test columns with null values: {}'.format( datatest.isnull().sum()))
datatrain.describe()
datatest.describe()
### we are compling the both data with median in Age, mode in Embarked and mediam in Fare



#First datatrain



datatrain['Age'].fillna(datatrain['Age'].median(), inplace = True)

datatrain['Embarked'].fillna(datatrain['Embarked'].mode()[0], inplace = True)

datatrain['Fare'].fillna(datatrain['Fare'].median(), inplace = True)



#Now the datatest 



datatest['Age'].fillna(datatest['Age'].median(), inplace = True)

datatest['Embarked'].fillna(datatest['Embarked'].mode()[0], inplace = True)

datatest['Fare'].fillna(datatest['Fare'].median(), inplace = True)
print('Train columns with null values: {} \n' .format( datatrain.isnull().sum()))

print("*************************************")

print('Test columns with null values: {}'.format( datatest.isnull().sum()))
#Now we are deleting the ['PassengerId','Cabin', 'Ticket'] COLUMNS

drop_column = ['PassengerId','Cabin', 'Ticket']

datatrain.drop(drop_column, axis=1, inplace = True)

datatest.drop(drop_column, axis=1, inplace = True)
datatrain.head()
alltables = [datatrain, datatest]



for dataset in alltables:    

    #Discrete variables

    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1



    dataset['IsAlone'] = 1 #initialize to yes/1 is alone

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1



    #quick and dirty code split title from name: http://www.pythonforbeginners.com/dictionary/python-split

    dataset['Title'] = dataset['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]





    #Continuous variable bins; qcut vs cut: https://stackoverflow.com/questions/30211923/what-is-the-difference-between-pandas-qcut-and-pandas-cut

    #Fare Bins/Buckets using qcut or frequency bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html

    dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)



    #Age Bins/Buckets using cut or value bins: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html

    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)





    

#cleanup rare title names

#print(data1['Title'].value_counts())

stat_min = 10 #while small is arbitrary, we'll use the common minimum in statistics: http://nicholasjjackson.com/2012/03/08/sample-size-is-10-a-magic-number/

title_names = (datatrain['Title'].value_counts() < stat_min) #this will create a true false series with title name as index



#apply and lambda functions are quick and dirty code to find and replace with fewer lines of code: https://community.modeanalytics.com/python/tutorial/pandas-groupby-and-python-lambda-functions/

datatrain['Title'] = datatrain['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

print(datatrain['Title'].value_counts())

print("----------")





#preview data again

datatrain.info()

datatest.info()

datatrain.head()
datatest.head()
sns.countplot(x="Survived", data=datatrain)  # How many people survived
#graph individual features by survival

fig, saxis = plt.subplots(2, 2,figsize=(16,12))



sns.countplot(x='Survived', hue="Embarked", data=datatrain,ax = saxis[0,0])   

sns.countplot(x='Survived', hue="IsAlone", data=datatrain,ax = saxis[0,1])

sns.countplot(x="Survived", hue="Pclass", data=datatrain, ax = saxis[1,0])

sns.countplot(x="Survived", hue="Sex", data=datatrain, ax = saxis[1,1])



fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(14,12))



sns.boxplot(x = 'Pclass', y = 'Fare', hue = 'Survived', data = datatrain, ax = axis1)

axis1.set_title('Pclass vs Fare Survival Comparison')



sns.violinplot(x = 'Pclass', y = 'Age', hue = 'Survived', data = datatrain, split = True, ax = axis2)

axis2.set_title('Pclass vs Age Survival Comparison')



sns.boxplot(x = 'Pclass', y ='FamilySize', hue = 'Survived', data = datatrain, ax = axis3)

axis3.set_title('Pclass vs Family Size Survival Comparison')
#plot distributions of age of passengers who survived or did not survive

a = sns.FacetGrid( datatrain, hue = 'Survived', aspect=4 )

a.map(sns.kdeplot, 'Age', shade= True )

a.set(xlim=(0 , datatrain['Age'].max()))

a.add_legend()
plt.subplots(figsize =(14, 12))

correlation = datatrain.corr()

sns.heatmap(correlation, annot=True,cmap='coolwarm')
#code categorical data

label = LabelEncoder()



for dataset in alltables:    

    dataset['Sex_Code'] = label.fit_transform(dataset['Sex'])

    dataset['Embarked_Code'] = label.fit_transform(dataset['Embarked'])

    dataset['Title_Code'] = label.fit_transform(dataset['Title'])

    dataset['AgeBin_Code'] = label.fit_transform(dataset['AgeBin'])

    dataset['FareBin_Code'] = label.fit_transform(dataset['FareBin'])





#define y variable aka target/outcome

Target = ['Survived']



#define x variables for original features aka feature selection

datatrain_x = ['Sex','Pclass', 'Embarked', 'Title','SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts

datatrain_x_calc = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code','SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation

datatrain_xy =  Target + datatrain_x

print('Original X Y: ', datatrain_xy, '\n')





#define x variables for original w/bin features to remove continuous variables

datatrain_x_bin = ['Sex_Code','Pclass', 'Embarked_Code', 'Title_Code', 'FamilySize', 'AgeBin_Code', 'FareBin_Code']

datatrain_xy_bin = Target + datatrain_x_bin

print('Bin X Y: ', datatrain_xy_bin, '\n')





#define x and y variables for dummy features original

datatrain_dummy = pd.get_dummies(datatrain[datatrain_x])

datatrain_x_dummy = datatrain_dummy.columns.tolist()

datatrain_xy_dummy = Target + datatrain_x_dummy

print('Dummy X Y: ', datatrain_xy_dummy, '\n')



datatrain_dummy.head()



#split train and test data with function defaults





train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = train_test_split(datatrain[datatrain_x_calc], datatrain[Target], random_state = 0)

train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(datatrain[datatrain_x_bin], datatrain[Target] , random_state = 0)



print("DataTrain Shape: {}".format(datatrain.shape))

print("Train1 Shape: {}".format(train1_x_dummy.shape))

print("Test1 Shape: {}".format(test1_x_dummy.shape))

train1_x_dummy.head()
# Decision Tree's

from sklearn.tree import DecisionTreeClassifier



Model = DecisionTreeClassifier()



Model.fit(train1_x_dummy, train1_y_dummy)



y_predL = Model.predict(test1_x_dummy)



# Summary of the predictions made by the classifier

print(classification_report(test1_y_dummy, y_predL))

print(confusion_matrix(test1_y_dummy, y_predL))

# Accuracy score

print('accuracy is',accuracy_score(y_predL,test1_y_dummy))



DT = accuracy_score(y_predL,test1_y_dummy)
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(test1_y_dummy, y_predL)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from sklearn.ensemble import RandomForestClassifier

Model=RandomForestClassifier(max_depth=2)

Model.fit(train1_x_dummy, train1_y_dummy)

y_predR=Model.predict(test1_x_dummy)



# Summary of the predictions made by the classifier

print(classification_report(test1_y_dummy,y_predR))

print(confusion_matrix(y_predR,test1_y_dummy))

#Accuracy Score

print('accuracy is ',accuracy_score(y_predR,test1_y_dummy))



RT = accuracy_score(y_predR,test1_y_dummy)

# LogisticRegression

from sklearn.linear_model import LogisticRegression

Model = LogisticRegression()

Model.fit(train1_x_dummy, train1_y_dummy)



y_predLR = Model.predict(test1_x_dummy)



# Summary of the predictions made by the classifier

print(classification_report(test1_y_dummy, y_predLR))

print(confusion_matrix(test1_y_dummy, y_predLR))

# Accuracy score

print('accuracy is',accuracy_score(y_predLR,test1_y_dummy))



LR = accuracy_score(y_predLR,test1_y_dummy)
# K-Nearest Neighbours

from sklearn.neighbors import KNeighborsClassifier



Model = KNeighborsClassifier(n_neighbors=8)

Model.fit(train1_x_dummy, train1_y_dummy)



y_predKN = Model.predict(test1_x_dummy)



# Summary of the predictions made by the classifier

print(classification_report(test1_y_dummy, y_predKN))

print(confusion_matrix(test1_y_dummy, y_predKN))

# Accuracy score



print('accuracy is',accuracy_score(y_predKN,test1_y_dummy))



KNN = accuracy_score(y_predKN,test1_y_dummy)
# Naive Bayes

from sklearn.naive_bayes import GaussianNB

Model = GaussianNB()

Model.fit(train1_x_dummy, train1_y_dummy)



y_predN = Model.predict(test1_x_dummy)



# Summary of the predictions made by the classifier

print(classification_report(test1_y_dummy, y_predN))

print(confusion_matrix(test1_y_dummy, y_predN))

# Accuracy score

print('accuracy is',accuracy_score(y_predN,test1_y_dummy))



NBB = accuracy_score(y_predN,test1_y_dummy)
# Support Vector Machine

from sklearn.svm import SVC



Model = SVC()

Model.fit(train1_x_dummy, train1_y_dummy)



y_predSVM = Model.predict(test1_x_dummy)



# Summary of the predictions made by the classifier

print(classification_report(test1_y_dummy, y_predSVM))

print(confusion_matrix(test1_y_dummy, y_predSVM))

# Accuracy score



print('accuracy is',accuracy_score(y_predSVM,test1_y_dummy))



SVMm = accuracy_score(y_predSVM,test1_y_dummy)
# Support Vector Machine's 

from sklearn.svm import NuSVC



ModelNU = NuSVC()

ModelNU.fit(train1_x_dummy, train1_y_dummy)



y_predNu = Model.predict(test1_x_dummy)



# Summary of the predictions made by the classifier

print(classification_report(test1_y_dummy, y_predNu))

print(confusion_matrix(test1_y_dummy, y_predNu))

# Accuracy score



print('accuracy is',accuracy_score(y_predNu,test1_y_dummy))



NuS = accuracy_score(y_predNu,test1_y_dummy)
# Linear Support Vector Classification

from sklearn.svm import LinearSVC



Model = LinearSVC()

Model.fit(train1_x_dummy, train1_y_dummy)



y_pred = Model.predict(test1_x_dummy)



# Summary of the predictions made by the classifier

print(classification_report(test1_y_dummy, y_pred))

print(confusion_matrix(test1_y_dummy, y_pred))

# Accuracy score



print('accuracy is',accuracy_score(y_pred,test1_y_dummy))



LSVM = accuracy_score(y_pred,test1_y_dummy)
from sklearn.neighbors import  RadiusNeighborsClassifier

Model=RadiusNeighborsClassifier(radius=148)

Model.fit(train1_x_dummy, train1_y_dummy)

y_pred=Model.predict(test1_x_dummy)



#summary of the predictions made by the classifier

print(classification_report(test1_y_dummy,y_pred))

print(confusion_matrix(test1_y_dummy,y_pred))



#Accouracy score

print('accuracy is ', accuracy_score(test1_y_dummy,y_pred))



RNC = accuracy_score(test1_y_dummy,y_pred)
from sklearn.linear_model import PassiveAggressiveClassifier

Model = PassiveAggressiveClassifier()

Model.fit(train1_x_dummy, train1_y_dummy)



y_pred = Model.predict(test1_x_dummy)



# Summary of the predictions made by the classifier

print(classification_report(test1_y_dummy, y_pred))

print(confusion_matrix(test1_y_dummy, y_pred))

# Accuracy score

print('accuracy is',accuracy_score(y_pred,test1_y_dummy))



PAC = accuracy_score(y_pred,test1_y_dummy)
# BernoulliNB

from sklearn.naive_bayes import BernoulliNB

Model = BernoulliNB()

Model.fit(train1_x_dummy, train1_y_dummy)



y_pred = Model.predict(test1_x_dummy)



# Summary of the predictions made by the classifier

print(classification_report(test1_y_dummy, y_pred))

print(confusion_matrix(test1_y_dummy, y_pred))

# Accuracy score

print('accuracy is',accuracy_score(y_pred,test1_y_dummy))



Ber = accuracy_score(y_pred,test1_y_dummy)
# ExtraTreeClassifier

from sklearn.tree import ExtraTreeClassifier



Model = ExtraTreeClassifier()



Model.fit(train1_x_dummy, train1_y_dummy)



y_pred = Model.predict(test1_x_dummy)



# Summary of the predictions made by the classifier

print(classification_report(test1_y_dummy, y_pred))

print(confusion_matrix(test1_y_dummy, y_pred))

# Accuracy score

print('accuracy is',accuracy_score(y_pred,test1_y_dummy))



ETC = accuracy_score(y_pred,test1_y_dummy)
from sklearn.ensemble import BaggingClassifier

Model=BaggingClassifier()

Model.fit(train1_x_dummy, train1_y_dummy)

y_pred=Model.predict(test1_x_dummy)



# Summary of the predictions made by the classifier

print(classification_report(test1_y_dummy,y_pred))

print(confusion_matrix(y_pred,test1_y_dummy))



#Accuracy Score

print('accuracy is ',accuracy_score(y_pred,test1_y_dummy))



BCC = accuracy_score(y_pred,test1_y_dummy)
from sklearn.ensemble import AdaBoostClassifier

Model=AdaBoostClassifier()

Model.fit(train1_x_dummy, train1_y_dummy)

y_pred=Model.predict(test1_x_dummy)



# Summary of the predictions made by the classifier

print(classification_report(test1_y_dummy,y_pred))

print(confusion_matrix(y_pred,test1_y_dummy))

#Accuracy Score

print('accuracy is ',accuracy_score(y_pred,test1_y_dummy))



AdaB = accuracy_score(y_pred,test1_y_dummy)
from sklearn.ensemble import GradientBoostingClassifier

ModelG=GradientBoostingClassifier()

ModelG.fit(train1_x_dummy, train1_y_dummy)

y_predGR=ModelG.predict(test1_x_dummy)



# Summary of the predictions made by the classifier

print(classification_report(test1_y_dummy,y_predGR))

print(confusion_matrix(y_predGR,test1_y_dummy))



#Accuracy Score

print('accuracy is ',accuracy_score(y_predGR,test1_y_dummy))



GBCC = accuracy_score(y_predGR,test1_y_dummy)
false_positive_rate, true_positive_rate, thresholds = roc_curve(test1_y_dummy, y_predGR)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc



plt.figure(figsize=(10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],linestyle='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

Model=LinearDiscriminantAnalysis()

Model.fit(train1_x_dummy, train1_y_dummy)

y_pred=Model.predict(test1_x_dummy)



# Summary of the predictions made by the classifier

print(classification_report(test1_y_dummy,y_pred))

print(confusion_matrix(y_pred,test1_y_dummy))



#Accuracy Score

print('accuracy is ',accuracy_score(y_pred,test1_y_dummy))



LDAA = accuracy_score(y_pred,test1_y_dummy)
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

Model=QuadraticDiscriminantAnalysis()

Model.fit(train1_x_dummy, train1_y_dummy)

y_pred=Model.predict(test1_x_dummy)



# Summary of the predictions made by the classifier

print(classification_report(test1_y_dummy,y_pred))

print(confusion_matrix(y_pred,test1_y_dummy))



#Accuracy Score

print('accuracy is ',accuracy_score(y_pred,test1_y_dummy))



QDAx = accuracy_score(y_pred,test1_y_dummy)
models = pd.DataFrame({

    'Model': ['Decision Tree', 'Random Forest',

              'LogisticRegression','K-Nearest Neighbours', 'Naive Bayes', 'SVM', 'Nu-Support Vector Classification',

             'Linear Support Vector Classification', 'Radius Neighbors Classifier', 'Passive Aggressive Classifier','BernoulliNB',

             'ExtraTreeClassifier', "Bagging classifier ", "AdaBoost classifier", 'Gradient Boosting Classifier' ,'Linear Discriminant Analysis',

             'Quadratic Discriminant Analysis'],

    'Score': [DT, RT, LR, KNN,NBB,SVMm, NuS,  LSVM , RNC, PAC, Ber, ETC, BCC, AdaB,  GBCC, LDAA, QDAx]})

models.sort_values(by='Score', ascending=False)
plt.subplots(figsize =(14, 12))



sns.barplot(x='Score', y = 'Model', data = models, palette="Set3")



#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html

plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')
#gradient boosting w/full dataset modeling submission score: defaults= 0.75119, tuned= 0.77033

submit_gbc = GradientBoostingClassifier()



submit_gbc.fit(datatrain[datatrain_x_bin], datatrain[Target])



rr = submit_gbc.predict(datatest[datatrain_x_bin])
rr
submission = pd.DataFrame({

        "PassengerId": datatestcopy["PassengerId"],

        "Survived": rr

    })

submission.to_csv('titanic_submission1.csv', index=False)

submission.head()