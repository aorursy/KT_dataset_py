import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn import svm

from sklearn.pipeline import Pipeline

from sklearn.grid_search import GridSearchCV



def predict(trainX, trainY, testX):

    scaler = StandardScaler().fit(trainX)

    trainX = scaler.transform(trainX)

    

    model = svm.SVC(random_state = 1)

    if(trainY.dtype != np.int64):

        print('svr')

        model = svm.SVR()

    else:

        print('svc')

    searchParams = dict(

        C = np.logspace(-3, 2, 10),

        gamma = np.logspace(-3, 2, 10)

    )

    

    search = GridSearchCV(model, param_grid = searchParams, cv = 5)

    search.fit(trainX, trainY)

    

    testX = scaler.transform(testX)

    prediction = search.best_estimator_.predict(testX)

    

    return prediction

    
import numpy as np

import pandas as pd



def convertCategorical(df, column):

    valid = df.ix[df[column].notnull()]

    factor = pd.factorize(valid[column])

    

    for idx, category in enumerate(factor[1]):

        name = column + '_' + category

        df[name] = pd.Series(np.zeros(len(df[column]), dtype = int), index = df.index)

        df.ix[df[column] == category, name] = 1

    df.drop(column, axis = 1, inplace = True)

    

def formatModel(df):

    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)

    df['Sex'] = pd.factorize(df['Sex'])[0]

    df['Embarked'] = pd.factorize(df['Embarked'])[0]

    

    nullColumns = df.ix[:,df.isnull().any()]

    for column in nullColumns:

        #add dummy value for NaN in column

        df.ix[df[column].isnull(), column] = 1e9

        

        #filter columns with null

        train = df.ix[df[column] != 1e9, df.notnull().all()]

        test = df.ix[df[column] == 1e9, df.notnull().all()]

        

        trainX = train.ix[:, train.columns != column].values

        trainY = train.ix[:, column].values

        

        testX = test.ix[:, test.columns != column].values

        

        prediction = predict(trainX, trainY, testX)

        

        df.ix[df[column] == 1e9, column] = prediction

def formatModel2(df):

    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1, inplace = True)

    convertCategorical(df, 'Sex')

    convertCategorical(df, 'Embarked')

    df.ix[df['Fare'].isnull(), 'Fare'] = df['Fare'].mean(skipna = True)

    df.ix[df['Age'].isnull(), 'Age'] = df['Age'].mean(skipna = True)

    
import numpy as np

import pandas as pd



train = pd.read_csv("../input/train.csv")

formatModel2(train)



trainData = train.values

trainX = trainData[:,1:]

trainY = trainData[:,0].astype(int)



test = pd.read_csv('../input/test.csv')

formatModel2(test)

print(test)

testX = test.values

prediction = predict(trainX, trainY, testX)



print(prediction)
test = pd.read_csv('../input/test.csv')

ids = test['PassengerId']

prediction = prediction.astype(int)



result = pd.DataFrame({

    'PassengerId': ids,

    'Survived': prediction

})



result.to_csv('survival.csv', index = False)