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
#given skikit-learn estimator object named model, following methods are available



# model.fit() for fitting training data, for supervised learning applications, this accepts X and y arguments (data X and labels y). For unsupervised applications, accepts single argument, data X

# model.predit() given a trained model, predict label of new set of data. Accepts one argument, the new data X_new, or X_test

# model.predict_proba() for classification problems, some estimators also provide this method, returns probability that a new observation has each categorical label. The label with the highest prob. is returned

# model.score() for classification or regression problems, most or all estimators implement a score method. Between 0 and 1, with larger score indicating a better fit
datatrain = pd.read_csv('/kaggle/input/titanic/train.csv')

datatest = pd.read_csv('/kaggle/input/titanic/test.csv')
datatrain.head()
datatest.head()
#creating copy of test df

datatestcopy = datatest.copy()
print(datatrain.shape)

print(datatest.shape)
datatrain.info()
# utilizes the {}.format() function which fills the curly brackets with the contents of the format function



print('Training data columns null count: {} \n'.format(datatrain.isnull().sum()))

print('Test data columns null count: {} \n'.format(datatest.isnull().sum()))
datatrain.describe()
datatest.describe()
#filling gaps with median in Age, mode in Embarked and median in Fate



datatrain['Age'].fillna(datatrain['Age'].median(), inplace = True)

datatrain['Embarked'].fillna(datatrain['Embarked'].mode()[0], inplace = True) #note mode() output can sometimes be an array if there is more than one mode, so will take first element of that array

datatrain['Fare'].fillna(datatrain['Fare'].median(), inplace = True)



datatest['Age'].fillna(datatest['Age'].median(), inplace = True)

datatest['Embarked'].fillna(datatest['Embarked'].mode()[0], inplace = True)

datatest['Fare'].fillna(datatest['Fare'].median(), inplace = True)
# checking work



print('Train columns null values {} \n'.format(datatrain.isnull().sum()))

print('****************************************************************')

print('Test columns null values {} \n'.format(datatest.isnull().sum()))
# dropping unecessary columns



drop = ['PassengerId','Cabin','Ticket']

datatrain.drop(drop, axis = 1, inplace = True)

datatest.drop(drop, axis = 1, inplace = True)
datatrain.head()
alltables = [datatrain,datatest]



for x in alltables:

    #creating discrete variables

    x['FamilySize'] = x['SibSp'] + x['Parch'] + 1

    

    x['IsAlone'] = 1 # 1 is yes/1 is alone

    x['IsAlone'].loc[x['FamilySize'] > 1] = 0 # updating IsAlone column for people that are not 

    

    # quick and dirty split of title from the name

    x['Title'] = x['Name'].str.split(', ', expand = True)[1].str.split('.', expand = True)[0]

    

    # continous variable bins. using qcut instead of cut to split by frequency and not value. Want to get same number of records in each bin, so we use qcut.

    x['FareBin'] = pd.qcut(x['Fare'],4)

    

    x['AgeBin'] = pd.cut(x['Age'].astype(int),5)





#cleaning up rare title name 

stat_min = 10  # common minimum in statistics



#returns boolean series

title_names = datatrain['Title'].value_counts() < stat_min



datatrain['Title'] = datatrain['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)

print(datatrain['Title'].value_counts())

print('*********')

    

    

    

    
datatrain.info()
datatest.info()
datatrain.head()
sns.countplot(x= 'Survived', data= datatrain)
#graphing individual features by survival



fig, saxis = plt.subplots(2,2,figsize=(16,12))



sns.countplot(x= 'Survived', hue='Embarked', data= datatrain, ax=saxis[0,0])

sns.countplot(x= 'Survived', hue = 'IsAlone', data = datatrain, ax = saxis[0,1])

sns.countplot(x= 'Survived', hue = 'Title', data = datatrain, ax = saxis[1,0])

sns.countplot(x= 'Survived', hue = 'Sex', data = datatrain, ax = saxis[1,1])
fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize= (14 ,12))



sns.boxplot(x = 'Pclass', y= 'Fare', hue= 'Survived', data = datatrain, ax= axis1)

axis1.set_title('Pclass vs Fare Survival Comparison')



sns.violinplot(x= 'Pclass', y= 'Age', hue='Survived', data = datatrain, split =True, ax =axis2)

axis2.set_title('Pclass vs Age Survival Comparison')



sns.boxplot(x= 'Pclass', y= 'FamilySize', hue= 'Survived', data= datatrain, ax = axis3)

axis3.set_title('Pclass vs Family Size Survival Comparison')
#plot distributions of age of passengers who survived or did not survive



a = sns.FacetGrid(datatrain, hue= 'Survived', aspect= 4)

a.map(sns.kdeplot, 'Age', shade= True)

a.set(xlim=(0, datatrain['Age'].max ()))

a.add_legend()
plt.subplots(figsize = (14,12))

correlation = datatrain.corr()

sns.heatmap(correlation, annot= True, cmap = 'coolwarm')
label = LabelEncoder()



for x in alltables:

    x['Sex_Code'] = label.fit_transform(x['Sex'])

    x['Embarked_Code'] = label.fit_transform(x['Embarked'])

    x['Title_Code'] = label.fit_transform(x['Title'])

    x['AgeBin_Code'] = label.fit_transform(x['AgeBin'])

    x['FareBin_Code'] = label.fit_transform(x['FareBin'])

    

# define y variable aka target/outcom

Target = ['Survived']



#define x variables for original features aka feature selection



datatrain_x = ['Sex','Pclass','Embarked','Title','SibSp','Parch','Age','Fare','FamilySize','IsAlone'] 

datatrain_x_calc = ['Sex_Code','Pclass','Embarked_Code','Title_Code','SibSp','Parch','Age','Fare'] #coded for algorithm calc

datatrain_xy = Target + datatrain_x

print('Original X Y:', datatrain_xy, '\n')



#define x variables for original w/bin features to remove continous variables



datatrain_x_bin = ['Sex_Code','Pclass','Embarked_Code', 'Title_Code','FamilySize','AgeBin_Code','FareBin_Code']

datatrain_xy_bin = Target + datatrain_x_bin

print('Bin X Y : ', datatrain_xy_bin,'\n')





datatrain_dummy = pd.get_dummies(datatrain[datatrain_x]) # get_dummies converts categorical into dummy variables



datatrain_x_dummy = datatrain_dummy.columns.tolist()

datatrain_xy_dummy = Target + datatrain_x_dummy

print('Dummy X Y: ', datatrain_xy_dummy, '\n')



datatrain_dummy.head()

#split train and test data with function defaults



train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = train_test_split(datatrain[datatrain_x_calc], datatrain[Target],test_size = .25, random_state = 0) #test_size default is 25% already

train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(datatrain[datatrain_x_bin], datatrain[Target], test_size =.25, random_state = 0)



print ('DataTrain Shape: {}'.format(datatrain.shape))

print ('Train1 Shape: {}'.format(train1_x_dummy.shape))

print ('Test1 Shape: {}'.format(test1_x_dummy.shape))

train1_x_dummy.head()
# supervised learning algo mostly used for classification problems, works for both categorical AND continous dependent variables
from sklearn.tree import DecisionTreeClassifier



Model = DecisionTreeClassifier()



Model.fit(train1_x_dummy, train1_y_dummy)



y_predL = Model.predict(test1_x_dummy)



print(classification_report(test1_y_dummy, y_predL))

print(confusion_matrix(test1_y_dummy, y_predL))

print('accuracy is', accuracy_score(y_predL, test1_y_dummy))



DT = accuracy_score(y_predL,test1_y_dummy)
from sklearn.metrics import roc_curve, auc
roc_curve(test1_y_dummy, y_predL) #returns false positive rate, then true positive rate
false_positive_rate, true_positive_rate, thresholds = roc_curve(test1_y_dummy, y_predL)

roc_auc = auc(false_positive_rate, true_positive_rate)

roc_auc
plt.figure(figsize=(10,10))

plt.title('Reciever operating characteristic')

plt.plot(false_positive_rate, true_positive_rate, color = 'red', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0,1],[0,1], linestyle ='--')

plt.axis('tight')

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
#term for an ensemble of decision trees. Each tree gives a classification, then the classification that has the most votes in the forest is chosen by the algorithm



from sklearn.ensemble import RandomForestClassifier



Model = RandomForestClassifier(max_depth = 2)

Model.fit(train1_x_dummy, train1_y_dummy)

y_predR = Model.predict(test1_x_dummy)



# summary of predictions made by classifier



print(classification_report(test1_y_dummy, y_predR))

print(confusion_matrix(y_predR, test1_y_dummy)) #confusion matrix diagnoal is TRUE NEGATIVE,TRUE POSITIVE, etc.





print('Accuracy is ', accuracy_score(y_predR, test1_y_dummy))



RT = accuracy_score(y_predR, test1_y_dummy)
# this is a classification algorithm, not a regression algorithm

# used to estimate discrete values (binary like 1/0, yes/no, true/false) based on a set of independent variables

# predicts probability of occurence of an event by fitting data to a logit function



from sklearn.linear_model import LogisticRegression

Model = LogisticRegression()

Model.fit(train1_x_dummy, train1_y_dummy)



y_predLR = Model.predict(test1_x_dummy)



# summary of predictions



print(classification_report(test1_y_dummy, y_predLR))

print(confusion_matrix(test1_y_dummy, y_predLR))



print('accuracy is', accuracy_score(y_predLR, test1_y_dummy))



LR = accuracy_score(y_predLR, test1_y_dummy)
# can be used for classification as WELL as regression, but mainly classification



from sklearn.neighbors import KNeighborsClassifier



Model = KNeighborsClassifier(n_neighbors= 8)

Model.fit(train1_x_dummy, train1_y_dummy)



y_predKN = Model.predict(test1_x_dummy)



print(classification_report(test1_y_dummy, y_predKN))

print(confusion_matrix(test1_y_dummy, y_predKN))





print('accuracy is', accuracy_score(y_predKN, test1_y_dummy))



KNN = accuracy_score(y_predKN, test1_y_dummy)

# classification technique based on Bayes' theorem with assumption of independence between predictors
from sklearn.naive_bayes import GaussianNB

Model = GaussianNB()

Model.fit(train1_x_dummy, train1_y_dummy)



y_predN = Model.predict(test1_x_dummy)



print(classification_report(test1_y_dummy, y_predN))

print(confusion_matrix(test1_y_dummy, y_predN))



print('accuracy is',accuracy_score(y_predN, test1_y_dummy))



NBB = accuracy_score(y_predN, test1_y_dummy)
# classification method. Plot each data item as a point in n-dimensional space (where n is number of features) with the value of each feature being the value of a particular coordinate

# each coordinate is known as a Support Vector
from sklearn.svm import SVC



Model = SVC()

Model.fit(train1_x_dummy, train1_y_dummy)



y_predSVM = Model.predict(test1_x_dummy)



print(classification_report(test1_y_dummy, y_predSVM))

print(confusion_matrix(test1_y_dummy, y_predSVM))



print('accuracy is',accuracy_score(y_predSVM, test1_y_dummy))



SVM = accuracy_score(y_predSVM, test1_y_dummy)
#Nu-Support Vector Classification, similar to SVC but uses a parameter to control the number of support vectors



from sklearn.svm import NuSVC



Model = NuSVC()

Model.fit(train1_x_dummy, train1_y_dummy)



y_predNu = Model.predict(test1_x_dummy)



print(classification_report(test1_y_dummy, y_predNu))

print(confusion_matrix(test1_y_dummy, y_predNu))



print('accuracy is',accuracy_score(y_predNu, test1_y_dummy))



NuS = accuracy_score(y_predSVM, test1_y_dummy)
# Linear Support Vector Classification

# similar to SVC with parameter kernel = linear, but implemented in terms of liblinear rather than libsvm, so more flexibility in the choice of penalties and loss functions and should scale better to large number of samples



from sklearn.svm import LinearSVC



Model = LinearSVC()

Model.fit(train1_x_dummy, train1_y_dummy)



y_pred = Model.predict(test1_x_dummy)



print(classification_report(test1_y_dummy, y_pred))

print(confusion_matrix(test1_y_dummy, y_pred))



print('accuracy is',accuracy_score(y_pred, test1_y_dummy))



LSVM = accuracy_score(y_pred, test1_y_dummy)
from sklearn.neighbors import RadiusNeighborsClassifier

Model = RadiusNeighborsClassifier(radius=148)

Model.fit(train1_x_dummy, train1_y_dummy)

y_pred = Model.predict(test1_x_dummy)



print(classification_report(test1_y_dummy,y_pred))

print(confusion_matrix(test1_y_dummy, y_pred))



print('accuracy is', accuracy_score(test1_y_dummy, y_pred))



RNC = accuracy_score(test1_y_dummy, y_pred)
# PA algorithm is a margin based online learning algorithm for binary classification



from sklearn.linear_model import PassiveAggressiveClassifier

Model = PassiveAggressiveClassifier()

Model.fit(train1_x_dummy, train1_y_dummy)



y_pred = Model.predict(test1_x_dummy)



print(classification_report(test1_y_dummy,y_pred))

print(confusion_matrix(test1_y_dummy, y_pred))



print('accuracy is', accuracy_score(test1_y_dummy, y_pred))



PAC = accuracy_score(test1_y_dummy, y_pred)
from sklearn.naive_bayes import BernoulliNB

Model = BernoulliNB()

Model.fit(train1_x_dummy,train1_y_dummy)



y_pred = Model.predict(test1_x_dummy)



print(classification_report(test1_y_dummy,y_pred))

print(confusion_matrix(test1_y_dummy, y_pred))



print('accuracy is', accuracy_score(test1_y_dummy, y_pred))

BER = accuracy_score(test1_y_dummy, y_pred)
#ExtraTreeClassifier is ENSEMBLE learning method based on decision tres. Like random forest, randomizes certain decisions and subsets of data to minimize over-learning from the data and overfitting



from sklearn.tree import ExtraTreeClassifier



Model = ExtraTreeClassifier()



Model.fit(train1_x_dummy, train1_y_dummy)



y_pred = Model.predict(test1_x_dummy)



print(classification_report(test1_y_dummy, y_pred))

print(confusion_matrix(test1_y_dummy, y_pred))



print('accuracy is', accuracy_score(y_pred, test1_y_dummy))



ETC = accuracy_score(y_pred, test1_y_dummy)
# Ensemble meta-estimator that fits base classifiers each on random subsets of original dataset and then aggregates individual predictions (either by voting or by averaging) to form final prediction



from sklearn.ensemble import BaggingClassifier

Model = BaggingClassifier()

Model.fit(train1_x_dummy, train1_y_dummy)



y_pred = Model.predict(test1_x_dummy)



print(classification_report(test1_y_dummy, y_pred))

print(confusion_matrix(test1_y_dummy, y_pred))



print('accuracy is', accuracy_score(y_pred, test1_y_dummy))



BCC = accuracy_score(y_pred, test1_y_dummy)
#meta-estimator that begins by fitting a classifier on the original datset and then fits additional copies of the classifier on same dataset but where the weights of incorrectly classified instances are adjusted such that subsequent classifiers focus more on difficult cases

from sklearn.ensemble import AdaBoostClassifier

Model = AdaBoostClassifier()

Model.fit(train1_x_dummy, train1_y_dummy)

y_pred = Model.predict(test1_x_dummy)



print(classification_report(test1_y_dummy, y_pred))

print(confusion_matrix(test1_y_dummy, y_pred))



print('accuracy is', accuracy_score(y_pred,test1_y_dummy))



AdaB = accuracy_score(y_pred,test1_y_dummy)

      
from sklearn.ensemble import GradientBoostingClassifier

Model = GradientBoostingClassifier()

Model.fit(train1_x_dummy, train1_y_dummy)



y_pred = Model.predict(test1_x_dummy)



print(classification_report(test1_y_dummy, y_pred))

print(confusion_matrix(test1_y_dummy, y_pred))



print('accuracy is', accuracy_score(y_pred,test1_y_dummy))



GBC = accuracy_score(y_pred,test1_y_dummy)

      
fpr,tpr,thresholds = roc_curve(test1_y_dummy,y_pred)

roc_auc = auc(fpr, tpr)

roc_auc

plt.figure(figsize= (10,10))

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, color= 'red', label = 'AUC =%0.2f' %roc_auc)

plt.legend(loc= 'lower right')

plt.plot([0,1],[0,1],linestyle='--')

plt.axis('tight')

plt.ylabel('TPR')

plt.xlabel('FPR')
Model.fit(datatrain[datatrain_x_calc],datatrain['Survived'])
y_pred = Model.predict(datatest[datatrain_x_calc])
output = pd.DataFrame({'PassengerId':datatestcopy['PassengerId'], 'Survived':y_pred})
output.to_csv('v4.csv', index= False)
models = pd.DataFrame({'Model':['Decision Tree','Random Forest','LogisticRegression','K-Nearest Neighbors','Naive Bayes', 'SVM','Nu-Support Vector Classification','Linear Support Vector Classification',

                                'Radius Neighbors Classifier','Passive Aggressive Classifier','BernoulliNB','ExtraTreeClassifier','Bagging Classifier','AdaBoost','Gradient Boosting'], 'Score'

                       :[DT, RT, LR, KNN, NBB,SVM,NuS, LSVM, RNC, PAC, BER, ETC, BCC, AdaB, GBC]})



models.sort_values(by='Score', ascending = False)