# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random



import seaborn as sb

import matplotlib.pyplot as plot

%matplotlib inline





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

trainDF = pd.read_csv('../input/train.csv')

testDF = pd.read_csv('../input/test.csv')

allData = [trainDF, testDF]

allDF = pd.concat([trainDF, testDF])

totalPassengers = 2224
trainDF.head()
trainDF.info()

print('+'+'-'*40+'+')

testDF.info()

print('+'+'-'*40+'+')

print('+'+'-'*40+'+')

print('Training data comprises %4.2f %% of all data'% (len(trainDF)/totalPassengers*100))

print('Testing data comprises %4.2f %%'%(len(testDF)/totalPassengers*100))

print('+'+'-'*40+'+')

print('+'+'-'*40+'+')
# Get details about numerical features

trainDF.describe()
# Get details about categorical features

trainDF.describe(include=['O'])
trainDF[trainDF.Fare==0].describe(include=['O'])
trainDF[trainDF.Fare==0].describe()
trainDF[trainDF.Age.isnull()].describe()
trainDF[trainDF.Age.isnull()].describe(include=['O'])
trainDF[trainDF.Embarked.isnull()].describe()
trainDF[trainDF.Embarked.isnull()].describe(include=['O'])
trainDF[trainDF.Embarked.isnull()].Name
# Function for sorting on survival. 

# Note this is optimized for discrete variables with only a few possible values

def sortByFeature(df, groupedFeature, sortedFeature='Survived'):

    return df[[groupedFeature, sortedFeature]].groupby([groupedFeature], as_index=False).mean().sort_values(by=sortedFeature, ascending=False)

    

def survivalHistogram(df, x, b=20, size=2.2):

    from seaborn import FacetGrid as grid

    from matplotlib import pyplot as plot

    g = grid(df, col='Survived', size=size,aspect=1.6)

    g.map(plot.hist, x, bins=b)

    

def compoundHistograms(df, numerical, ordinal, b=20, size=2.2):

    from seaborn import FacetGrid as grid

    from matplotlib import pyplot as plot

    g = grid(df, col='Survived', row=ordinal, size=size, aspect=1.6)

    g.map(plot.hist, numerical, bins=b)

    



def infoHist(df, numerical, ordinal, b=20, size=2.2):

    from seaborn import FacetGrid as grid

    from matplotlib import pyplot as plot

    g = grid(df, row=ordinal, size=size, aspect=1.6)

    g.map(plot.hist, numerical,bins=b)

    

def tripleBox(df, numerical, ordinal1, ordinal2, dims=(12,12)):

    from seaborn import boxplot as graph

    from matplotlib import pyplot as plot

    dummy, ax = plot.subplots(figsize=dims)

    graph(x=ordinal1, y=numerical, hue=ordinal2, data=df, ax=ax)

    

def boxPlot(df, ordinal, numerical, survival=False, dims=(12,16)):

    from seaborn import boxplot as graph

    from matplotlib import pyplot as plot

    dummy, ax = plot.subplots(figsize=dims)

    if survival:

        graph(x=ordinal, y=numerical, hue='Survived', data=df, ax=ax)

    else:

        graph(x=ordinal, y=numerical, data=df, ax=ax)

        

def regPlot(df, x, y, dims=(12,12)):

    from seaborn import regplot as rp

    from matplotlib import pyplot as plot

    dummy, ax = plot.subplots(figsize=dims)

    g = rp(x=x, y=y, data=df, ax=ax)
# See average survival rate for each class

sortByFeature(trainDF, 'Pclass')
# Survival rate by sex

sortByFeature(trainDF, 'Sex')
sortByFeature(trainDF, 'Embarked')
survivalHistogram(trainDF, 'Age', 20, 3)
compoundHistograms(trainDF, 'Age', 'Pclass')
compoundHistograms(trainDF, 'Age', 'Sex', 15)
boxPlot(trainDF, 'Survived', 'Age')
survivalHistogram(trainDF, 'Fare', 20, 3)
boxPlot(trainDF, 'Survived', 'Fare')
# Survival rate by number of siblings and spouses aboard

sortByFeature(trainDF, 'SibSp')
# Survival rate by number of parents and children aboard

sortByFeature(trainDF, 'Parch')
# Group Embarked location by average Fare

sortByFeature(trainDF,'Embarked','Fare')
sortByFeature(trainDF, 'Embarked', 'Pclass')
# Graphical representation of above data

boxPlot(trainDF, 'Embarked', 'Fare')
#sortByFeature(trainDF, 'Pclass', 'Fare')

boxPlot(trainDF,'Pclass', 'Fare')
sortByFeature(trainDF, 'Sex', 'Fare')
trainDF[['Fare','Age']].corr()
trainDF[['Fare', 'Pclass']].corr()
boxPlot(trainDF, 'Sex','Age')
boxPlot(trainDF,'Pclass','Age')
boxPlot(trainDF,'SibSp','Age')
boxPlot(trainDF,'Parch','Age')


pd.crosstab(trainDF.Embarked, trainDF.Pclass)
# Extract titles

for df in allData:

    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



#pd.crosstab(trainDF.Title, trainDF.Sex)

pd.crosstab(testDF.Title, testDF.Sex)



sortByFeature(trainDF[trainDF.Sex=='female'],'Title')
sortByFeature(trainDF[trainDF.Sex=='male'],'Title')
sortByFeature(trainDF, 'Title', 'Age')
for df in allData:

    df.loc[df.Sex=='female','Title']=df.loc[df.Sex=='female','Title'].replace(['Lady','Countess','Mme','Dr','Dona'],'Mrs')

    df.loc[df.Sex=='female','Title']=df.loc[df.Sex=='female','Title'].replace(['Ms', 'Mlle'],'Miss')

    df.loc[df.Sex=='male','Title']  =df.loc[df.Sex=='male','Title'].replace(['Capt','Col', 'Don', 'Dr', 'Jonkheer', 'Major', 'Rev', 'Sir'],'Mr')



pd.crosstab(trainDF.Title, trainDF.Sex)
sortByFeature(trainDF, 'Title','Age')
sortByFeature(trainDF,'Title')
# Map Title

titleMap={'Mrs':1,'Miss':2,'Master':3,'Mr':4}

for df in allData:

    df.Title = df.Title.map(titleMap)

pd.crosstab(trainDF.Title, trainDF.Sex)



# Map Sex

sexMap={'female':0,'male':1}

for df in allData:

    df.Sex = df.Sex.map(sexMap)
tripleBox(trainDF,'Age','Title','Pclass')
from sklearn.ensemble import RandomForestRegressor



def completeAge(df, useRegressor=False):

    if useRegressor:

        # Grab pertinent features (only complete ones besides Age)

        ageDF = df[['Age','Sex','Title','Parch','SibSp','Pclass']]

        knownAgeDF = ageDF.loc[(df.Age.notnull()) & (df.Age != 0)]

        unknownAgeDF = ageDF.loc[(df.Age.isnull()) | (df.Age == 0)]

        #print('len(knownAge)   = %d'%(len(knownAgeDF)))

        #print('len(unknownAge) = %d'%(len(unknownAgeDF)))

        # Pull values

        Y = knownAgeDF.values[:,0]

        X = knownAgeDF.values[:,1::]

        rfr = RandomForestRegressor(n_estimators=2000)

        rfr.fit(X, Y)

        predictedAges = rfr.predict(unknownAgeDF.values[:, 1::])

        df.loc[(df.Age.isnull()) | (df.Age == 0),'Age'] = predictedAges

    else:

        ageMap = np.zeros((3,4))

        for p in range(1,ageMap.shape[0]+1):

            for t in range(1,ageMap.shape[1]+1):

                ageMap[p-1][t-1] = trainDF.loc[trainDF.Age.notnull() & (trainDF.Pclass==(p)) & (trainDF.Title==(t)),'Age'].median()

                df.loc[df.Age.isnull() & (df.Pclass==p) & (df.Title==t),'Age'] = ageMap[p-1][t-1]

    return df
for df in allData:

    df = completeAge(df)

len(trainDF[trainDF.Embarked.isnull()])
mostEmbarked = np.zeros((3,1),'str')

for i in range(0,len(mostEmbarked)):

    mostEmbarked[i] = trainDF[trainDF.Pclass==(i+1)].Embarked.dropna().mode()[0]



for df in allData:

    for p in range(1,4):

        df.loc[(df.Pclass==p) & df.Embarked.isnull(),'Embarked'] = mostEmbarked[p-1]

len(trainDF[trainDF.Embarked.isnull()])
tripleBox(trainDF,'Fare','Embarked','Pclass')
print(len(trainDF[trainDF.Embarked.isnull()]))

sortByFeature(trainDF,'Embarked')

embarkMap={'C':1,'Q':2,'S':3}

#embarkMap={1:1,2:3,3:2}

for df in allData:

    df.Embarked = df.Embarked.map(embarkMap)

#for df in allData:

#    df.Title = df.Embarked.map(embarkMap)

#pd.crosstab(trainDF.Embarked, trainDF.Sex)

trainDF[['Embarked','Fare']].corr()
def completeFare(df, useRegressor=False):

    if useRegressor:

        fareDF = df[['Fare','Age','Title','Embarked','SibSp','Parch','Pclass']]

        # Get un/known values

        knownFareDF = fareDF.loc[(fareDF.Fare > 0) & (fareDF.Fare.notnull())]

        unknownFareDF = fareDF.loc[(fareDF.Fare <= 0) | (fareDF.Fare.isnull())]

        Y = knownFareDF.values[:,0]

        X = knownFareDF.values[:,1::]

        rfr = RandomForestRegressor(n_estimators=2000)

        rfr.fit(X, Y)

        predictedFares = rfr.predict(unknownFareDF.values[:,1::])

        df.loc[(df.Fare.isnull()) | (df.Fare <= 0),'Fare'] = predictedFares

    else:

        fareMap = np.zeros((3,3))

        for p in range(1,fareMap.shape[0]+1):

            for e in range(1,fareMap.shape[1]+1):

                fareMap[p-1][e-1] = trainDF.loc[(trainDF.Fare!=0) & (trainDF.Pclass==(p)) & (trainDF.Embarked==(e)),'Fare'].median()

                df.loc[( (df.Fare.isnull()) | (df.Fare==0) ) & (df.Pclass==p) & (df.Embarked==e),'Fare'] = fareMap[p-1][e-1]

            

    return df
for df in allData:

    df = completeFare(df)

trainDF.Fare.hist()

testDF.Fare.hist()
nCuts = 4

#ageCutoffs = list(range(0,80,int(80/nCuts)))

trainDF['AgeBand'], bins = pd.qcut(trainDF.Age, nCuts, retbins=True)

sortByFeature(trainDF, 'AgeBand')

for i in range(0, len(bins)):

    for df in allData:

        if i == nCuts:

            df.loc[(df.Age >= bins[i]),'AgeGroup'] = i

        elif i == 0:

            df.loc[(df.Age < bins[i+1]),'AgeGroup'] = i

        else:

            df.loc[(df.Age >= bins[i]) & (df.Age < bins[i+1]),'AgeGroup'] = i

trainDF.AgeGroup.head()

bins

trainDF[['AgeGroup','Survived']].corr()

sortByFeature(trainDF,'AgeGroup')

survivalHistogram(trainDF, 'AgeGroup')
trainDF['FareBand'], bins = pd.qcut(trainDF.Fare, nCuts, retbins=True)

sortByFeature(trainDF, 'FareBand')

for i in range(0, len(bins)):

    for df in allData:

        if i == nCuts:

            df.loc[(df.Fare >= bins[i], 'FareGroup')] = i

        elif i == 0:

            df.loc[(df.Fare < bins[i+1], 'FareGroup')] = i

        else:

            df.loc[(df.Fare >= bins[i]) & (df.Fare < bins[i+1]), 'FareGroup'] = i

sortByFeature(trainDF,'FareGroup')
def makeFamSizeGroup(df):

    famSizeMap = {0:0,

                  1:1, 2:1, 3:1,

                  4:2, 5:2, 6:2,

                  7:3, 8:3, 9:3, 10:3}

    df['FamSize'] = trainDF.SibSp + trainDF.Parch

    df['FamSizeGroup'] = df.FamSize.map(famSizeMap)



for df in allData:

    df = makeFamSizeGroup(df)
trainDF[['FamSizeGroup','Survived']].corr()

#survivalHistogram(trainDF, 'FamSizeGroup')

#sortByFeature(trainDF,'FamSize')

#print(ceil(3.4))


#Do some PCA on this data to improve generalizability

featureList=['Pclass','AgeGroup','FareGroup','Embarked','Title','Sex','FamSizeGroup','FamSize','Age','Fare']

#featureList = ['Pclass','AgeGroup','FareGroup','Embarked','Title','Sex','FamSizeGroup']

#featureList = ['Pclass','Sex']

fullList = featureList + ['Survived']

#fullList=[list(featureList)[:],'Survived']

trainDF[fullList].corr()



from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

rfc = RandomForestClassifier(n_estimators = 1000, max_features='sqrt')



X_train = trainDF[featureList].values[:,:]

#X_train = trainDF.values[:,:]

Y_train = trainDF['Survived'].values

X_test = testDF[featureList].values[:,:]

rfc = rfc.fit(X_train, Y_train)



features = pd.DataFrame()

features['feature'] = trainDF[featureList].columns

features['importance'] = rfc.feature_importances_

features.sort_values(by='importance', ascending=True, inplace=True)

features.set_index('feature', inplace=True)

features.plot(kind='barh',figsize = (10,10))
#from sklearn.decomposition import PCA

#pca = PCA(0.99)

#pca.fit(X_train)

#X_train = pca.fit_transform(X_train)

#X_test = pca.fit_transform(X_test)
from sklearn.model_selection import cross_val_score

cuts = 5





def ccr(model, X, Y):

    return round(model.score(X, Y)*100, 2)



def printStats(model, X, Y, name, cuts=cuts):

    print('Cross-Validation Scores for %s:'%(name))

    print(cross_val_score(model, X, Y, cv=cuts))

    print('\nMean Correct-Classification rate for %s:'%(name))

    print(ccr(model, X, Y))

# Logistic Regression

from sklearn.linear_model import LogisticRegression

logReg = LogisticRegression()

logReg.fit(X_train, Y_train)

Yp = logReg.predict(X_test)

ccr_logReg = ccr(logReg, X_train, Y_train)

print('Cross-Validation Scores for logReg:')

print(cross_val_score(logReg, X_train, Y_train, cv=cuts))

print('\nMean Correct-Classification rate for logReg:')

print(ccr_logReg)
# SVM

from sklearn import svm

mySVM = svm.SVC()

mySVM.fit(X_train, Y_train) 

Yp_svm = mySVM.predict(X_test)

ccr_svm = ccr(mySVM, X_train, Y_train)

print('Cross-Validation Scores for SVM:')

print(cross_val_score(mySVM, X_train, Y_train, cv=cuts))

print('\nMean Correct-Classification rate for SVM:')

print(ccr_svm)
# kNN

from sklearn.neighbors import KNeighborsClassifier

n_neighbors = 3

knn = KNeighborsClassifier(n_neighbors = n_neighbors)

knn.fit(X_train, Y_train)

Yp_knn = knn.predict(X_test)

ccr_knn = ccr(knn, X_train, Y_train)

printStats(knn, X_train, Y_train, 'KNN')

# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

naiveBayes = GaussianNB()

naiveBayes.fit(X_train, Y_train)

Yp_naiveBayes = naiveBayes.predict(X_test)

printStats(naiveBayes, X_train, Y_train, 'Naive Bayes')
# Perceptron

from sklearn.linear_model import Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Yp_perceptron = perceptron.predict(X_test)

printStats(perceptron, X_train, Y_train, 'Perceptron')
# Linear SVM

from sklearn.svm import LinearSVC



linearSVM = LinearSVC()

linearSVM.fit(X_train, Y_train)

Yp_linearSVM = linearSVM.predict(X_test)

printStats(linearSVM, X_train, Y_train, 'Linear SVM')
# Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Yp_sgd = sgd.predict(X_test)

printStats(sgd, X_train, Y_train, 'Stochastic Gradient Descent')
# Decision Tree

from sklearn.tree import DecisionTreeClassifier



dTree = DecisionTreeClassifier()

dTree.fit(X_train, Y_train)

Yp_dTree = dTree.predict(X_test)

printStats(dTree, X_train, Y_train, 'Decision Tree')

# Random Forest



from sklearn.ensemble import RandomForestClassifier

n_estimators = 100

randForest = RandomForestClassifier(n_estimators = n_estimators)

#randForest = GridSearchCV(RandomForestClassifier(n_estimators = n_estimators),

#                          {"max_depth":[None, 5, 2],

#                           "min_samples_leaf":[2,3,5]})

randForest.fit(X_train, Y_train)

Yp_randForest = randForest.predict(X_test)

printStats(randForest, X_train, Y_train, 'Random Forest')
submission = pd.DataFrame({

        "PassengerId": testDF["PassengerId"],

        "Survived": Yp_randForest

    })

submission.to_csv('submission.csv', index=False)