#%matplotlib inline = show plots in Jupyter Notebook browser

%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

traindata =  pd.read_csv('C:/Users/3arrows/Desktop/ML Projects/Learn Machine Learning ON Board Titanic - 17 Algorithms/titanic/train.csv')
testdata =  pd.read_csv('C:/Users/3arrows/Desktop/ML Projects/Learn Machine Learning ON Board Titanic - 17 Algorithms/titanic/test.csv')
traindata.head()
testdata.head()
print(traindata.shape)

print(testdata.shape)
traindata.info()
testdata.info()
traindata.describe()
traindatacopy = traindata.copy()

testdatacopy = testdata.copy()
print('Train columns with null values: {} \n' .format(traindatacopy.isnull().sum()))



print('Test columns with null values: {}'.format(testdatacopy.isnull().sum()))
traindatacopy['Age'].fillna(traindatacopy['Age'].median(), inplace = True)



testdatacopy['Age'].fillna(testdatacopy['Age'].median(), inplace = True)



drop_column = ['Cabin']

#drop_column = ['PassengerId', 'Ticket']

traindatacopy.drop(drop_column, axis=1, inplace = True)

testdatacopy.drop(drop_column, axis=1, inplace = True)
print('Train columns with null values: {} \n' .format(traindatacopy.isnull().sum()))



print('Test columns with null values: {}'.format(testdatacopy.isnull().sum()))
traindatacopy.head()
testdatacopy.head()
alltables = [traindatacopy, testdatacopy]



for dataset in alltables:    

    #Discrete variables

    dataset['FamilySize'] = dataset ['SibSp'] + dataset['Parch'] + 1



    dataset['IsAlone'] = 1 #initialize to yes/1 is alone

    dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0 # now update to no/0 if family size is greater than 1

traindatacopy.head()
testdatacopy.head()
sns.countplot(x="Survived", data=traindatacopy) 
fig, saxis = plt.subplots(2, 2,figsize=(16,12))



sns.countplot(x='Survived', hue="Embarked", data=traindatacopy,ax = saxis[0,0])   

sns.countplot(x='Survived', hue="IsAlone", data=traindatacopy,ax = saxis[0,1])

sns.countplot(x="Survived", hue="Pclass", data=traindatacopy, ax = saxis[1,0])

sns.countplot(x="Survived", hue="Sex", data=traindatacopy, ax = saxis[1,1])

f,ax=plt.subplots(1,2,figsize=(16,7))

traindatacopy['Survived'][traindatacopy['Sex']=='male'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[0],shadow=True)

traindatacopy['Survived'][traindatacopy['Sex']=='female'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[1],shadow=True)

ax[0].set_title('Survived (male)')

ax[1].set_title('Survived (female)')
a = sns.FacetGrid(traindatacopy, hue = 'Survived', aspect=4 )

a.map(sns.kdeplot, 'Age', shade= True )

a.set(xlim=(0 , traindatacopy['Age'].max()))

a.add_legend()
plt.subplots(figsize =(14, 12))

correlation = traindatacopy.corr()

sns.heatmap(correlation, annot=True,cmap='coolwarm')


#define y variable aka target/outcome

Target = ['Survived']



#define x variables for original features aka feature selection

datatrain_x = ['Sex','Pclass', 'Embarked', 'SibSp', 'Parch', 'Age', 'Fare', 'FamilySize', 'IsAlone'] #pretty name/values for charts

datatrain_x_calc = ['Sex_Code','Pclass', 'SibSp', 'Parch', 'Age', 'Fare'] #coded for algorithm calculation

datatrain_xy =  Target + datatrain_x

print('Original X Y: ', datatrain_xy, '\n')





#define x variables for original w/bin features to remove continuous variables

datatrain_x_bin = ['Pclass',  'FamilySize' ]

datatrain_xy_bin = Target + datatrain_x_bin

print('Bin X Y: ', datatrain_xy_bin, '\n')





#define x and y variables for dummy features original

datatrain_dummy = pd.get_dummies(traindatacopy[datatrain_x])

datatrain_x_dummy = datatrain_dummy.columns.tolist()

datatrain_xy_dummy = Target + datatrain_x_dummy

print('Dummy X Y: ', datatrain_xy_dummy, '\n')



datatrain_dummy.head()
#split train and test data with function defaults

from sklearn.model_selection import train_test_split

from sklearn import model_selection



train1_x_dummy, test1_x_dummy, train1_y_dummy, test1_y_dummy = train_test_split(traindatacopy[datatrain_x_calc], traindatacopy[Target], random_state = 0)

train1_x_bin, test1_x_bin, train1_y_bin, test1_y_bin = model_selection.train_test_split(traindatacopy[datatrain_x_bin], traindatacopy[Target] , random_state = 0)



print("DataTrain Shape: {}".format(traindatacopy.shape))

print("Train1 Shape: {}".format(train1_x_dummy.shape))

print("Test1 Shape: {}".format(test1_x_dummy.shape))

train1_x_dummy.head()
from sklearn.neighbors import KNeighborsClassifier



Model = KNeighborsClassifier(n_neighbors=5).fit(train1_x_dummy, train1_y_dummy)



y_predKN = Model.predict(test1_x_dummy)



print('accuracy is',accuracy_score(y_predKN,test1_y_dummy))



KNN = accuracy_score(y_predKN,test1_y_dummy)
from sklearn.preprocessing import MinMaxScaler



scalar =  MinMaxScaler()



x_scaled = scalar.fit_transform(train1_x_dummy)

y_scaled = scalar.fit_transform(train1_y_dummy)



Model1 = KNeighborsClassifier(n_neighbors=5).fit(x_scaled, train1_y_dummy)



predKN = Model.predict(test1_x_dummy)



print('accuracy is',accuracy_score(predKN,test1_y_dummy))



from sklearn.neighbors import  RadiusNeighborsClassifier

Model=RadiusNeighborsClassifier(radius=148).fit(train1_x_dummy, train1_y_dummy)

y_pred=Model.predict(test1_x_dummy)



print('accuracy is ', accuracy_score(test1_y_dummy,y_pred))



RNC = accuracy_score(test1_y_dummy,y_pred)
from sklearn.naive_bayes import GaussianNB



Model = GaussianNB().fit(train1_x_dummy, train1_y_dummy)



y_predN = Model.predict(test1_x_dummy)



print('accuracy is',accuracy_score(y_predN,test1_y_dummy))



NBB = accuracy_score(y_predN,test1_y_dummy)
from sklearn.naive_bayes import BernoulliNB

Model = BernoulliNB().fit(train1_x_dummy, train1_y_dummy)



y_pred = Model.predict(test1_x_dummy)



print('accuracy is',accuracy_score(y_pred,test1_y_dummy))



Ber = accuracy_score(y_pred,test1_y_dummy)
from sklearn.svm import SVC



Model = SVC().fit(train1_x_dummy, train1_y_dummy)



y_predSVM = Model.predict(test1_x_dummy)



print('accuracy is',accuracy_score(y_predSVM,test1_y_dummy))



SVMm = accuracy_score(y_predSVM,test1_y_dummy)
from sklearn.svm import SVC



Model = SVC(kernel='rbf' , gamma=1).fit(train1_x_dummy, train1_y_dummy)



y_predSVM = Model.predict(test1_x_dummy)



print('accuracy is',accuracy_score(y_predSVM,test1_y_dummy))



SVMrbf = accuracy_score(y_predSVM,test1_y_dummy)
from sklearn.svm import LinearSVC



Model = LinearSVC().fit(train1_x_dummy, train1_y_dummy)



y_pred = Model.predict(test1_x_dummy)



print('accuracy is',accuracy_score(y_pred,test1_y_dummy))



LSVM = accuracy_score(y_pred,test1_y_dummy)
from sklearn.svm import NuSVC



ModelNU = NuSVC().fit(train1_x_dummy, train1_y_dummy)



y_predNu = Model.predict(test1_x_dummy)



print('accuracy is',accuracy_score(y_predNu,test1_y_dummy))



NuS = accuracy_score(y_predNu,test1_y_dummy)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score





Model = DecisionTreeClassifier().fit(train1_x_dummy, train1_y_dummy)



y_predL = Model.predict(test1_x_dummy)



print('accuracy is',accuracy_score(y_predL,test1_y_dummy))



DT = accuracy_score(y_predL,test1_y_dummy)


from sklearn.linear_model import LogisticRegression



Model = LogisticRegression().fit(train1_x_dummy, train1_y_dummy)



y_predLR = Model.predict(test1_x_dummy)



print('accuracy is',accuracy_score(y_predLR,test1_y_dummy))



LR = accuracy_score(y_predLR,test1_y_dummy)
from sklearn.ensemble import RandomForestClassifier

Model=RandomForestClassifier(max_depth=2).fit(train1_x_dummy, train1_y_dummy)

y_predR=Model.predict(test1_x_dummy)



print('accuracy is ',accuracy_score(y_predR,test1_y_dummy))



RT = accuracy_score(y_predR,test1_y_dummy)

from sklearn.tree import ExtraTreeClassifier



Model = ExtraTreeClassifier().fit(train1_x_dummy, train1_y_dummy)



y_pred = Model.predict(test1_x_dummy)



print('accuracy is',accuracy_score(y_pred,test1_y_dummy))



ETC = accuracy_score(y_pred,test1_y_dummy)
from sklearn.ensemble import BaggingClassifier



Model=BaggingClassifier().fit(train1_x_dummy, train1_y_dummy)



y_pred=Model.predict(test1_x_dummy)



print('accuracy is ',accuracy_score(y_pred,test1_y_dummy))



BCC = accuracy_score(y_pred,test1_y_dummy)
from sklearn.ensemble import AdaBoostClassifier



Model=AdaBoostClassifier().fit(train1_x_dummy, train1_y_dummy)



y_pred=Model.predict(test1_x_dummy)



print('accuracy is ',accuracy_score(y_pred,test1_y_dummy))



AdaB = accuracy_score(y_pred,test1_y_dummy)
from sklearn.ensemble import GradientBoostingClassifier



Model=GradientBoostingClassifier().fit(train1_x_dummy, train1_y_dummy)



y_predGR=Model.predict(test1_x_dummy)





print('accuracy is ',accuracy_score(y_predGR,test1_y_dummy))



GBCC = accuracy_score(y_predGR,test1_y_dummy)
models = pd.DataFrame({

    'Model': ['K-Nearest Neighbours','Radius Neighbors Classifier', 'Naive Bayes', 'BernoulliNB', 'Support Vector Machines',

              'Linear Support Vector Classification', 'Nu-Support Vector Classification', 'Decision Tree', 'LogisticRegression',

              'Random Forest', 'ExtraTreeClassifier', "Bagging classifier ", "AdaBoost classifier", 'Gradient Boosting Classifier'],

    'Score': [ KNN, RNC, NBB, Ber, SVMm, LSVM , NuS, DT, LR, RT,ETC, BCC, AdaB,  GBCC]})

models.sort_values(by='Score', ascending=False)
plt.subplots(figsize =(14, 12))



sns.barplot(x='Score', y = 'Model', data = models, palette="Set3")



plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')
submit_gbc = GradientBoostingClassifier().fit(tatrain_x_bin, Target)



rr = submit_gbc.predict(datatrain_x_bin)
rr
submission = pd.DataFrame({

        "PassengerId": testdata["PassengerId"],

        "Survived": rr

    })

submission.to_csv('titanic_submission.csv', index=False)



submission.head(10)