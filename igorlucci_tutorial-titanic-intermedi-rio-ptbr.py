import warnings

warnings.filterwarnings("ignore") #Para não poluir o notebook com avisos 



import pandas as pd #Manipulação de Dados

import numpy as np #Álgebra Linear

import matplotlib.pyplot as plt #Visualização de Dados

import seaborn as sns #Visualização de Dados

import pandas_profiling #relatório pronto para entender os dados



# machine learning



from sklearn.linear_model import RidgeCV

import xgboost as xgb

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

test = pd.read_csv("../input/titanic/test (1).csv")

train = pd.read_csv("../input/titanic/train (1).csv")

combine = [train, test]

print(train.shape)

test.shape
train.dtypes
test.dtypes
profile_report = pandas_profiling.ProfileReport(train)

profile_report
train.describe(include=[np.number]).T
train.describe(include='O').T

pd.options.display.max_rows = 100 ##Caso a visualização abaixo fique truncada, é só rodar essa linha de código

def missingValuesInfo(df):

    total = df.isnull().sum().sort_values(ascending=False)

    percent = round(df.isnull().sum().sort_values(ascending=False)/len(df)*100, 2)

    temp = pd.concat([total, percent], axis = 1, keys = ['Total', 'Percent'])

    return temp.loc[temp['Total'] > 0]
missingValuesInfo(train)
#Cabin

train[['Cabin', 'Survived']].groupby(['Cabin'], as_index=False).mean().sort_values(by='Survived', ascending=False)
plt.figure(figsize=(3,3))



train.Cabin.value_counts().plot(kind='bar')

plt.ylabel('Counts')

plt.xlabel('Cabin')

#plt.title('Gender Distribution')

plt.show()
#PCLASS

train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
plt.figure(figsize=(3,3))



train.Pclass.value_counts().plot(kind='bar')

plt.ylabel('Counts')

plt.xlabel('Pclass')

#plt.title('Gender Distribution')

plt.show()
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#Embarked

train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
Target = 'Survived'

train.dropna(axis=0, subset =[Target], inplace=True)
def dropValues(df):

  return df.drop(['Name', 'Cabin', 'Name','Ticket'],axis = 1)
train = dropValues(train)
train.shape
def HandleMissingValues(df):

    # for Object columns fill using 'UNKOWN'

    # for Numeric columns fill using median

    num_cols = [cname for cname in df.columns if df[cname].dtype in ['int64', 'float64']]

    cat_cols = [cname for cname in df.columns if df[cname].dtype == "object"]

    values = {}

    for a in cat_cols:

        values[a] = 'S' #Somente 2 valores nulos e vou colocar o que é mais frequente. 

    for a in num_cols:

        values[a] = df[a].median()



    df.fillna(value=values,inplace=True)   
missingValuesInfo(train)
HandleMissingValues(train)
missingValuesInfo(train)
train.head(3)
train.dtypes.T
train.describe(include='O')
def getObjectColumnsList(df):

    return [cname for cname in df.columns if df[cname].dtype == "object"]



def PerformOneHotEncoding(df, columnsToEncode):

    return pd.get_dummies(df, columns = columnsToEncode)
cat_cols = getObjectColumnsList(train)
print(cat_cols)
train = PerformOneHotEncoding(train, cat_cols)
train.dtypes.T
train.head()
train['Age'] = round(train['Age']).values.astype(np.int64)
train.dtypes
train.head()
HandleMissingValues(test)



test = dropValues(test)



cat_colsTest = getObjectColumnsList(test)



test = PerformOneHotEncoding(test, cat_colsTest)



test['Age'] = round(test['Age']).values.astype(np.int64)

#test['Fare'] = round(test['Fare']).values.astype(np.int64)

#test['Sex_female'] = round(test['Sex_female']).values.astype(np.int64)

#test['Sex_male'] = round(test['Sex_male']).values.astype(np.int64)
test.head()
train.head()
print(train.shape)

test.shape
train_df = train.drop(['PassengerId'], axis=1)

test_df = test

combine = [train_df, test_df]

train_df.shape, test_df.shape
X_train = train_df.drop("Survived", axis=1) #features

Y_train = train_df["Survived"] # label



X_test  = test_df.drop("PassengerId", axis=1).copy() #test



X_train.shape, Y_train.shape, X_test.shape
from sklearn.linear_model import RidgeClassifier

clf = RidgeClassifier().fit(X_train, Y_train)

clf.score(X_train, Y_train)

submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": clf

    })

submission.to_csv('submission.csv', index=False)