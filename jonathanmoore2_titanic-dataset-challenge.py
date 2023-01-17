## Remove unused Machine Learning Packages



# Importing the relevent packages



# Math/Linear algebra

import numpy      as np

import pandas     as pd

import statistics as sts

import math

import scipy      as sp



# Data visulization

import matplotlib.pyplot as plt

import seaborn    as sns

sns.set(style="ticks", color_codes=True)



# Machine Learning Model

from sklearn import linear_model

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
# Import data

trainData = pd.read_csv('../input/titanic/train.csv')

testData  = pd.read_csv('../input/titanic/test.csv')

print(trainData.shape)

print(testData.shape)
# An overview of our columns/data types.

trainData.head(5)
# A basic view of mean, max, etc. of numeric columns

trainData.describe().drop(columns=['PassengerId'])
# Display Number of Null and Unique Values by Columns



trainNull = pd.DataFrame(trainData.isnull().sum())

trainUnique = pd.DataFrame(trainData.nunique())

trainNull.columns = ['Training Null Value Count']

trainUnique.columns = ['Training Unique Value Count']



testNull = pd.DataFrame(testData.isnull().sum())

testUnique = pd.DataFrame(testData.nunique())

testNull.columns = ['Testing Null Value Count']

testUnique.columns = ['Testing Unique Value Count']



table1 = trainNull.join(testNull)

table2 = trainUnique.join(testUnique)



table1.join(table2)
# Passenger Survival Probability Mass Fuction (Pmf).

# Recall: 0 = 'Died', 1 = 'Survived'

sns.countplot(trainData['Survived'])

survivalRatio = round(len(trainData[trainData['Survived']==1])/

                     (len(trainData[trainData['Survived']==1])+len(trainData[trainData['Survived']==0])),2)



print("Survival Rate: {}".format(survivalRatio))



pd.DataFrame({'Count':[trainData['Survived'].value_counts()[0],trainData['Survived'].value_counts()[1]]}, 

             index = ['Died','Lived'])
# Passenger Sex Pmf.

x = trainData.groupby('Sex')['PassengerId'].count()

y = testData.groupby('Sex')['PassengerId'].count()

genderCount = pd.DataFrame({'dataset':['Train','Train','Test','Test'],

                            'sex':['Male','Female','Male','Female'],

                            'count':[x[1],x[0],y[1],y[0]]})



sns.barplot(x="dataset", y="count", hue="sex", data=genderCount) 





trainSexRatio = round(len(trainData[trainData['Sex']=='male'])/

                      (len(trainData[trainData['Sex']=='male'])+len(trainData[trainData['Sex']=='female'])),2)

testSexRatio = round(len(testData[testData['Sex']=='male'])/

                     (len(testData[testData['Sex']=='male'])+len(testData[testData['Sex']=='female'])),2)





d = {'Train': [trainSexRatio], 'Test': [testSexRatio]}

pd.DataFrame(d, index = ['Percentage Male'])
# Passenger Pclass Pmf.

x = trainData.groupby('Pclass')['PassengerId'].count()

y = testData.groupby('Pclass')['PassengerId'].count()



pClassTable = pd.DataFrame({'dataset':['Train','Train','Train','Test','Test','Test'],

                           'Pclass': [1,2,3,1,2,3],

                           'Count':[x.iloc[0],x.iloc[1],x.iloc[2],y.iloc[0],y.iloc[1],y.iloc[2]]})



sns.barplot(x='dataset', y='Count', hue='Pclass',data=pClassTable)



d = pd.DataFrame({'Train':[round(x.iloc[0]/x.sum(),2), round(x.iloc[1]/x.sum(),2), round(x.iloc[2]/x.sum(),2)],

                   'Test':[round(y.iloc[0]/y.sum(),2), round(y.iloc[1]/y.sum(),2), round(y.iloc[2]/y.sum(),2)]},

                index=[1,2,3])

d
# Passenger Age Pmf.

fig_dims = (16, 7)

fig, ax = plt.subplots(figsize=fig_dims)



plt.subplot(2,2,1)

sns.distplot(trainData['Age'],color='blue')

plt.subplot(2,2,2)

sns.distplot(testData['Age'],color='orange')

plt.subplot(2,2,3)

sns.distplot(trainData['Age'],color='blue')

sns.distplot(testData['Age'],color='orange')



d = pd.DataFrame({'Median':[trainData['Age'].median(),testData['Age'].median()]},index=['Train','Test'])

d



# Train data displayed on top Left, Test data displayed on the top Right
# Fare Distribution Check



fig_dims = (16, 7)

fig, ax = plt.subplots(figsize=fig_dims)



plt.subplot(2,2,1)

sns.distplot(trainData['Fare'],color='blue')

plt.subplot(2,2,2)

sns.distplot(testData['Fare'],color='orange')

plt.subplot(2,2,3)

sns.distplot(trainData['Fare'],color='blue')

sns.distplot(testData['Fare'],color='orange')



d = pd.DataFrame({'Median':[trainData['Fare'].median(),testData['Fare'].median()]},index=['Train','Test'])

d



# Train data displayed on top Left, Test data displayed on the top Right
# Total Revenue Generated by the Titanic.

round(trainData['Fare'].sum() + testData['Fare'].sum(), 2)
#Distribution of Sibilings/Spouse Count

sns.countplot(trainData['SibSp'])

pd.DataFrame(trainData['SibSp'].value_counts())
#Distribution of Parent/Children Count

sns.countplot(trainData['Parch'])

pd.DataFrame(trainData['Parch'].value_counts())
# Probably have to move this down too the 3rd section.

sns.barplot(trainData['Parch'],trainData['SibSp'])
#Location Embarked Distribution Check.



x = trainData.groupby('Embarked')['PassengerId'].count()

y = testData.groupby('Embarked')['PassengerId'].count()



embarkedTable = pd.DataFrame({'dataset':['Train','Train','Train','Test','Test','Test'],

                           'Embarked': ['C','Q','S','C','Q','S'],

                           'Count':[x.iloc[0],x.iloc[1],x.iloc[2],y.iloc[0],y.iloc[1],y.iloc[2]]})



sns.barplot(x='dataset', y='Count', hue='Embarked',data=embarkedTable)



# Probability Mass Function Table

d = pd.DataFrame({'Train':[round(x.iloc[0]/x.sum(),2), round(x.iloc[1]/x.sum(),2), round(x.iloc[2]/x.sum(),2)],

                   'Test':[round(y.iloc[0]/y.sum(),2), round(y.iloc[1]/y.sum(),2), round(y.iloc[2]/y.sum(),2)]},

                index=['C','Q','S'])

d
# Extracting passengers 'Title' from 'Name'

trainData['Title'] = trainData.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



x = trainData.groupby('Title')['Survived'].sum()/trainData.groupby('Title')['PassengerId'].count()



survByTitle = pd.DataFrame([trainData.groupby('Title')['Survived'].sum(),x]).T

survByTitle.columns = ['Survived','Survival Ratio']



pd.crosstab(trainData['Title'], trainData['Sex']).join(survByTitle, on='Title').round(2)
trainData[(trainData['Title']=='Dr') & (trainData['Sex']=='female')]
trainData['Title'] = trainData['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



trainData['Title'] = trainData['Title'].replace('Mlle', 'Miss')

trainData['Title'] = trainData['Title'].replace('Ms', 'Miss')

trainData['Title'] = trainData['Title'].replace('Mme', 'Mrs')

trainData = trainData.drop(columns=['Name'])

trainData.head(5)
# Extract Title from names for the test data.

testData['Title'] = testData.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



testData['Title'] = testData['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

testData['Title'] = testData['Title'].replace('Mlle', 'Miss')

testData['Title'] = testData['Title'].replace('Ms', 'Miss')

testData['Title'] = testData['Title'].replace('Mme', 'Mrs')

testData = testData.drop(columns=['Name'])

testData.head(5)
# Title Distribution Check

x = pd.DataFrame(trainData['Title'].value_counts())

y = pd.DataFrame(testData['Title'].value_counts())



sns.barplot(x=x.index,y=x['Title'])



# Probability mass functions

x.columns=['Train']

y.columns=['Test']



round(x.div(891),2).join(round(y.div(418),2))
#isCabin distribution check.

#Note: 0 -> Passenger has a cabin, 1 -> Passenger doesn't have a cabin.



isCabinTrain = trainData.isnull()[['Cabin']].astype(int)

isCabinTest  = testData.isnull()[['Cabin']].astype(int)



sns.countplot(isCabinTrain['Cabin'])



# Ratio Table

x = round(isCabinTrain['Cabin'].sum()/891,2)

y = round(isCabinTest['Cabin'].sum()/418,2)

pd.DataFrame({'Train':x,'Test':y},index=['Percentage of Non-Cabin Passengers'])
# Replacing the old Cabin column with our new isCabin Information.

trainData = trainData.drop(columns=['Cabin']).join(isCabinTrain)

testData = testData.drop(columns=['Cabin']).join(isCabinTest)
# Mean age values age based on title and Pclass.

meanTable = pd.DataFrame(trainData.groupby(['Title','Pclass'])['Age'].mean())

meanTable
# Here we impute Age values based on Mean given title.



for i in range(len(trainData)):

    if math.isnan(trainData.loc[i,'Age']) == True:

        x = trainData.loc[(i,'Pclass')]

        y = trainData.loc[(i,'Title')]

        trainData.loc[i,'Age'] = meanTable.loc[(y,x),'Age']



trainData.isna().sum()
# Same thing for Test Copy.



for i in range(len(testData)):

    if math.isnan(testData.loc[i,'Age']) == True:

        x = testData.loc[(i,'Pclass')]

        y = testData.loc[(i,'Title')]

        testData.loc[i,'Age'] = meanTable.loc[(y,x),'Age']



testData.isna().sum()
#Fill in null fare value.

for i in range(len(testData)):

    if math.isnan(testData.loc[(i,'Fare')]) == True:

        testData.loc[(i, 'Fare')] = testData.describe()['Fare']['mean']

testData.isna().sum()
trainDataClean = trainData.drop(columns = ['Ticket'])

testDataClean = testData.drop(columns = ['Ticket'])

trainDataClean.head()
# Creating Age Bins

bins = [0,10,20,30,40,50,80]

x = pd.cut(trainDataClean['Age'],bins)

trainDataClean['AgeBins'] = x
trainDataDummy = pd.get_dummies(trainDataClean.drop(columns=['AgeBins'])).drop(columns=['Sex_female','Title_Rare'])

trainDataDummy.head()
testDataDummy = pd.get_dummies(testDataClean).drop(columns=['Sex_male','Title_Rare'])

testDataDummy.head()
# Correlation Matrix

fig_dims = (14, 6)

fig, ax = plt.subplots(figsize=fig_dims)



sns.heatmap(trainDataDummy.drop(columns=['PassengerId']).corr(), annot = True, cmap='Blues')
sns.barplot(trainDataClean['Sex'],trainDataClean['Survived'])



x = round(trainDataClean[trainDataClean['Sex']=='male']['Survived'].sum()/

          trainDataClean[trainDataClean['Sex']=='male'].count()['PassengerId'],2)

y = round(trainDataClean[trainDataClean['Sex']=='female']['Survived'].sum()/

          trainDataClean[trainDataClean['Sex']=='female'].count()['PassengerId'],2)



w = pd.DataFrame({'Survival Rate':[x,y]},index = ['Male', 'Female'])

w
sns.barplot(trainDataClean['Pclass'],trainDataClean['Survived'])



x = round(trainDataClean[trainDataClean['Pclass']==1]['Survived'].sum()/

          trainDataClean[trainDataClean['Pclass']==1].count()['PassengerId'],2)

y = round(trainDataClean[trainDataClean['Pclass']==2]['Survived'].sum()/

          trainDataClean[trainDataClean['Pclass']==2].count()['PassengerId'],2)

z = round(trainDataClean[trainDataClean['Pclass']==3]['Survived'].sum()/

          trainDataClean[trainDataClean['Pclass']==3].count()['PassengerId'],2)



w = pd.DataFrame({'Survival Rate':[x,y,z]},index = ['Pclass 1', 'Pclass 2', 'Pclass 3'])

w
#Age Buckets? Age survive graph

#sns.distplot(trainDataClean['Age'])

fig_dims = (13, 5)

fig, ax = plt.subplots(figsize=fig_dims)



y = pd.DataFrame(trainDataClean['AgeBins'].value_counts()).sort_index()



plt.subplot(1,2,1)

sns.barplot(trainDataClean['AgeBins'],trainDataClean['Survived'])

plt.subplot(1,2,2)

sns.barplot(y.index,y['AgeBins'])
fig_dims = (12, 4)

fig, ax = plt.subplots(figsize=fig_dims)



plt.subplot(121)

sns.barplot(trainDataClean['SibSp'],trainDataClean['Survived'])

plt.subplot(122)

sns.barplot(trainDataClean['Parch'],trainDataClean['Survived'])
trainDataClean['FamilySize'] = trainDataClean['SibSp'] + trainDataClean['Parch']

sns.barplot(trainDataClean['FamilySize'],trainDataClean['Survived'])
trainDataClean[['Pclass','Fare']].groupby('Pclass').mean()
g = sns.FacetGrid(trainDataClean, col="Survived")

g = g.map(plt.hist, "Fare")



x = trainDataClean[['Fare','Survived']][trainDataClean['Survived']==0].mean()['Fare']

y = trainDataClean[['Fare','Survived']][trainDataClean['Survived']==1].mean()['Fare']

pd.DataFrame({'Average Fare':[x,y]},index=['Died','Survived'])
# ??? Need relavence justification, donnon, this feels strange to me.

x = trainDataClean[['Pclass','PassengerId']].groupby('Pclass').count()

y = trainDataClean[['Cabin','Pclass']].groupby('Pclass').sum()

z = x.join(y)

w = pd.DataFrame({'Ratio Of Cabins':[round(((216-40)/216),2),round(((216-168)/184),2),round(((491-479)/491),2)]},index = [1,2,3])



#Also didnt mention how this was first class graph, even then its significance is not too impressive.

# This section seems unessasary, should be a simpler test.

sns.barplot(trainDataClean['Cabin'],trainData['Survived'])

plt.xlabel(['Cabin','NoCabin'])

w



# Sum => number of isNull cabins?? ie 216-40 = #of Cabins
# I like this graph. Its useless though.

# Maybe post hawk justification with looking at combining the two collumns.

sns.barplot(trainDataClean['Parch'],trainDataClean['SibSp'])
sns.barplot(trainDataClean['Embarked'],trainDataClean['Survived'])
# Logistic Regression

# Random Forest

# Support Vector Machine

# naive Bayes Classifier

# Descision Tree (??)
# Refresh of the data we are analyzing.

trainDataDummy.head()
# Converting our Dataframe to Array for Regression functions. 

xTrain = np.array(trainDataDummy.drop(columns=['Survived','PassengerId']))

yTrain = np.array(trainDataDummy['Survived'])



xTest = np.array(testDataDummy.drop(columns=['PassengerId']))
#Model 1: statsmodel

import statsmodels.api as sm

logit_model=sm.Logit(yTrain,xTrain)

result=logit_model.fit()

result.summary2()
# Model 2: R General Linear Model.

# Note that I inserted the x_i naming convention from the above model for comparison clarity.



# Coefficients:

#                      Estimate Std. Error t value Pr(>|t|)    

# x0  = (Intercept)   1.3531021  0.3445045   3.928 9.25e-05 ***

# x1  = Pclass       -0.1293291  0.0251023  -5.152 3.18e-07 ***

# x2  = Age          -0.0042975  0.0012202  -3.522  0.00045 ***

# x3  = SibSp        -0.0685433  0.0130027  -5.271 1.71e-07 ***

# x4  = Parch        -0.0506701  0.0179991  -2.815  0.00498 ** 

# x5  = Fare          0.0003787  0.0003203   1.182  0.23740    

# x6  = Cabin        -0.0952522  0.0431424  -2.208  0.02751 *  

# x7  = Sex_male     -0.6116760  0.2266936  -2.698  0.00710 ** 

# x8  = Embarked_C    0.0042827  0.2608641   0.016  0.98691    

# x9  = Embarked_Q   -0.0174467  0.2635731  -0.066  0.94724    

# x10 = Embarked_S   -0.0529766  0.2603837  -0.203  0.83883    

# x11 = Title_Master  0.5280820  0.1123763   4.699 3.03e-06 ***

# x12 = Title_Miss   -0.1013161  0.2143859  -0.473  0.63663    

# x13 = Title_Mr      0.0023926  0.0858493   0.028  0.97777    

# x14 = Title_Mrs     0.0229304  0.2141820   0.107  0.91477 
pd.DataFrame({'Column':['Intercept','Pclass','Age','SibSp','Parch',

                        'Fare','Cabin','Sex_male','Embarked_C','Embarked_Q',

                        'Embarked_S','Title_Master','Title_Miss','Title_Mr','Title_Mrs'],

             'Model 1 P-Value':[np.nan,0,0.0021,0,0.0046,

                                0.2289,0.0312,0.3384,0.0013,0.0042,

                                0.0044,0,0.0802,0.8708,0.0102],

             'Model 2 P-Value':[0,0,0.0004,0,0.0050,

                                0.2374,0.0275,0.0071,0.9869,0.9472,

                                0.8388,0,0.6366,0.9778,0.9148],

             'Model 1: Significant':[np.nan,'Yes','Yes','Yes','Yes',

                                     'No','Yes','No','Yes','Yes',

                                     'Yes','Yes','No','No','Yes'],

             'Model 2: Significant':['Yes','Yes','Yes','Yes','Yes',

                                     'No','Yes','Yes','No','No',

                                     'No','Yes','No','No','No']})
logReg = linear_model.LogisticRegression(max_iter=500)

logReg.fit(xTrain,yTrain)
logReg.score(xTrain,yTrain)
print(logReg.intercept_)

print(logReg.coef_)
yPred = logReg.predict(xTest)
submission = pd.DataFrame({

        "PassengerId": testData["PassengerId"],

        "Survived": yPred

    })

submission.to_csv('submission.csv', index=False)
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(xTrain, yTrain)

Y_pred = random_forest.predict(xTest)

random_forest.score(xTrain, yTrain)

#This score is crazy high
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(xTrain, yTrain)

Y_pred = decision_tree.predict(xTest)

decision_tree.score(xTrain, yTrain)

# Very Suspicious
# Stocastic Gradient Descent

sgd = SGDClassifier()

sgd.fit(xTrain, yTrain)

Y_pred = sgd.predict(xTest)

sgd.score(xTrain, yTrain)
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(xTrain, yTrain)

Y_pred = linear_svc.predict(xTest)

acc_linear_svc = round(linear_svc.score(xTrain, yTrain) * 100, 2)

acc_linear_svc
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(xTrain, yTrain)

Y_pred = gaussian.predict(xTest)

acc_gaussian = round(gaussian.score(xTrain, yTrain) * 100, 2)

acc_gaussian
# K-Nearest Neighbours



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(xTrain, yTrain)

Y_pred = knn.predict(xTest)

acc_knn = round(knn.score(xTrain, yTrain) * 100, 2)

acc_knn