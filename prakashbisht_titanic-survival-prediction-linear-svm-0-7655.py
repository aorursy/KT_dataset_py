#libs for data manipulation

import pandas as pd

import numpy as np



#libs for data visulization & analysis

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#machine learning methods

from sklearn import svm

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
dataset = pd.read_csv('../input/train.csv')

train_x = dataset[['Pclass', 'Sex', 'Age', 'SibSp','Parch', 'Fare', 'Embarked']]

train_y = dataset['Survived'].values 



testdata= pd.read_csv('../input/test.csv')

test_x= testdata[train_x.columns.values] #makes it same size as train_x column wise



print(dataset.shape, train_x.shape, train_y.shape, test_x.shape)

train_x.head(10) #train_x.head()
test_x.head(5)
print('train data description:')

dataset.info(),

print('\n','test data description:')

test_x.info()
dataset.describe()
figure, axis = plt.subplots(1,1,figsize=(20,3))

age = dataset[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()

sns.barplot(x='Age', y='Survived', data= age)
figure = plt.figure(figsize=(13,8))

plt.hist([train_x[dataset['Survived']==1]['Age'].dropna(),train_x[dataset['Survived']==0]['Age'].dropna()], stacked=True, color = ['r','b'], bins = 30,label = ['Survived','Dead'])

plt.xlabel('Age')

plt.ylabel('Passengers Count')

plt.legend()
#filling correlated age individually to NaNs in train set & test set.

age_fill = [train_x['Age'].median(), test_x['Age'].median()]

train_x['Age'].fillna(age_fill[0], inplace=True)

test_x['Age'].fillna(age_fill[1], inplace=True)
print(train_x.shape, train_y.shape)

combined = train_x.append(test_x) #[featureset, test_x]

print(type(combined), combined.shape)
survived_peeps = train_x[dataset['Survived']==1]['Sex'].value_counts()

dead_peeps = train_x[dataset['Survived']==0]['Sex'].value_counts()

df = pd.DataFrame([survived_peeps,dead_peeps])

df.index = ['Survived','Dead']

print(df)

df.plot(kind='bar',stacked=True, figsize=(12,6))
fixSex = combined['Sex'].copy().values

fixSex[fixSex == 'male'] = 0  #changing Categorical(Sex) into numerical analogue

fixSex[fixSex == 'female'] = 1 

fixSex.shape 
grid = sns.FacetGrid(dataset, col ='Sex', size = 3.2, aspect =1.7)

grid.map(sns.barplot, 'Embarked','Survived', alpha= 0.6, ci = None)
combined['Embarked'].fillna(combined['Embarked'].mode()[0], inplace=True) #filling Null vals in Embarked column

fixEmb = combined['Embarked'].copy().values



fixEmb[fixEmb =='S'] = 0 #changing Embarked into numerical analogue

fixEmb[fixEmb =='C'] = 1

fixEmb[fixEmb =='Q'] = 2

fixEmb.shape
#combined['Age'].fillna(sum(age_fill)/2, inplace=True) 

combined['Fare'].fillna(combined['Fare'].median(), inplace=True)

combined.info() #all features have non null values now
print(combined.shape)

combined.head(15) #dataset before being transformed to wholly numeric
allnum_dataset = combined.copy()

allnum_dataset.loc[:,'Sex'] = fixSex #assigning numeric nd arrays to columns that held string vals.

allnum_dataset.loc[:,'Embarked'] = fixEmb

allnum_dataset.head(15) #after numerical feature transformation
X = allnum_dataset.copy()[:891]

test_x=  allnum_dataset.copy()[891:]

X.shape, test_x.shape #final split of training and test data
XtrainV, XtestV, ytrainV, ytestV = train_test_split(X,train_y, test_size = 0.30)

XtrainV.shape, ytrainV.shape, XtestV.shape, ytestV.shape  # X = XtrainV(70%) + XtestV(30%), train_y = ytrainV(70%) +ytestV(30%)
cla_sv =svm.SVC()# svm kernel = rbf

cla_sv
print("size of validation training data:","X inputs->", XtrainV.shape,", y targets-> ", ytrainV.shape)

cla_sv.fit(XtrainV, ytrainV)

print ("\n","expected accuracy on solely on training data basis:",cla_sv.score(XtrainV, ytrainV))
print("size of validation test data:","X inputs->", XtestV.shape,", y targets-> ", ytestV.shape)

ypred = cla_sv.predict(XtestV)

print("\n","expected prediction accuracy:", metrics.accuracy_score(ytestV, ypred))
#Now applying RBF kernel SVM on complete training data

print("size of whole training data:","X inputs->", X.shape,", y targets-> ", train_y.shape)

cla_sv.fit(X, train_y)

print("\n","expected accuracy on solely training data basis:", cla_sv.score(X, train_y))
print("size of whole test data:","X inputs->", test_x.shape,  "y targets->", test_x.shape[0])

target_sv = cla_sv.predict(test_x)

submission_sv= pd.DataFrame({'PassengerId':testdata['PassengerId'].values, 'Survived': target_sv})

#submission_sv.shape

#submission_sv.head(10)

#submitting only linSvm predictions to csv & hiding submissions from all other methods

#submission_sv.to_csv('../output/submission_sv.csv', index=False)
#Now applying linear kernel SVM on complete training data

cla_linsv =svm.SVC(kernel= 'linear')

print("size of whole training data:","X inputs->", X.shape,", y targets-> ", train_y.shape)

cla_linsv.fit(X, train_y)

print("\n","expected accuracy on solely training data basis:", cla_linsv.score(X, train_y))
print("size of whole test data:","X inputs->", test_x.shape,  "y targets->", test_x.shape[0])

target_linsv = cla_linsv.predict(test_x)

submission_linsv= pd.DataFrame({'PassengerId':testdata['PassengerId'].values, 'Survived': target_linsv})
print(submission_linsv.shape)

submission_linsv.to_csv('submission_linsv.csv', index=False)

#submission_linsv.to_csv('../output/submission_linsv.csv', index=False)
#Now using logistic regression on complete training data

cla_log= LogisticRegression()

print("size of whole training data:","X inputs->", X.shape,", y targets-> ", train_y.shape)

cla_log.fit(X, train_y)

print("\n","expected accuracy on solely training data basis:", cla_log.score(X, train_y))
print("size of whole test data:","X inputs->", test_x.shape,  "y targets->", test_x.shape[0])

target_log = cla_log.predict(test_x)

submission_log = pd.DataFrame({'PassengerId':testdata['PassengerId'].values, 'Survived': target_log})
print(submission_log.shape)

#submitting only linSvm predictions to csv & hiding submissions from all other methods

#submission_log.to_csv('../output/submission_log.csv', index=False)
#Now using descision tree on complete training data

cla_dt= DecisionTreeClassifier()

print("size of whole training data:","X inputs->", X.shape,", y targets-> ", train_y.shape)

cla_dt.fit(X, train_y)

print("\n","expected accuracy on solely training data basis:", cla_dt.score(X, train_y))
print("size of whole test data:","X inputs->", test_x.shape,  "y targets->", test_x.shape[0])

target_dt = cla_dt.predict(test_x)

submission_dt = pd.DataFrame({'PassengerId':testdata['PassengerId'].values, 'Survived': target_dt})
print("results through Decision Tree method", submission_log.shape)

#submitting only linSvm predictions to csv & hiding submissions from all other methods

#submission_dt.to_csv('../output/submission_dt.csv', index=False)
#Now using Random forests on complete training data

cla_forest = RandomForestClassifier(n_estimators=100)

print("size of whole training data:","X inputs->", X.shape,", y targets-> ", train_y.shape)

cla_forest.fit(X, train_y)

print("\n","expected accuracy on solely training data basis:", cla_forest.score(X, train_y))
print("size of whole test data:","X inputs->", test_x.shape,  "y targets->", test_x.shape[0])

target_forest = cla_forest.predict(test_x)

submission_forest = pd.DataFrame({'PassengerId':testdata['PassengerId'].values, 'Survived': target_forest})
print("results random forest method", submission_forest.shape)

#submitting only linSvm predictions to csv & hiding submissions from all other methods

submission_forest.to_csv('submission_forest.csv', index=False)
#Model evaluation

models= pd.DataFrame({'Models':['SVM_rbf','SVM_linear','Logistic regression', 'Decision Trees','Random forest'],'Accuracy':[cla_sv.score(X, train_y), cla_linsv.score(X, train_y),cla_log.score(X, train_y),cla_dt.score(X, train_y), cla_forest.score(X, train_y)]})

models.sort_values(by='Accuracy', ascending= False)                                    