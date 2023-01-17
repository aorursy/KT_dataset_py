# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC

from sklearn.multiclass import OneVsRestClassifier

from sklearn.multiclass import OneVsOneClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier



import warnings

warnings.filterwarnings('ignore')
df_train = pd.read_csv('../input/train.csv')

df_train.head()
df_test = pd.read_csv('../input/test.csv')

df_test.head()
#Reading the DataFrame and Clearing the Data

def data_frame(arquivo):

    arquivo_all = '../input/'+arquivo

    df = pd.read_csv(arquivo_all)

    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)    

    df['Age'].fillna(df['Age'].mean(), inplace=True)

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    if arquivo == 'train.csv':

        y = df[df.columns[0]]

        x = df[df.columns[1:]]

        x = pd.get_dummies(x) 

        return x, y

    else:

        x = df[df.columns]

        x = pd.get_dummies(x) 

        return x
X, Y = data_frame('train.csv')

x = data_frame('test.csv')
# Y.head() -> Survived

X.head()
x.head()
#print(Y.isnull().any()) -> Fals

print('Train')

print(X.isnull().any())

print('-'*30)

print('Test')

print(x.isnull().any())
x['Fare'].fillna(x['Fare'].mean(), inplace=True)

print(x.isnull().any())
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, Y)



x_train = np.array(x_train).reshape(len(x_train),10)

x_test = np.array(x_test).reshape(len(x_test),10)

y_train = y_train.values.ravel()

y_test = y_test.values.ravel()
def fit_and_score(x_train, x_validation, y_train, y_validation):

    names = ["OneVsrest", "OneVsOne", "MultinomialNB", "AdaBoost", "LinearRegression", "DecisionTreeRegressor", 

             "AdaBoostRegressor", "GradientBoostingRegressor","LogisticRegression", "RandomForestClassifier",

             "KNeighborsClassifier", "GaussianNB", "Perceptron", "SGDClassifier"]

    models=[OneVsRestClassifier(LinearSVC(random_state=0)), OneVsOneClassifier(LinearSVC(random_state=0)), 

            MultinomialNB(), AdaBoostClassifier(), LinearRegression(), DecisionTreeRegressor(), 

            AdaBoostRegressor(), GradientBoostingRegressor(), LogisticRegression(), RandomForestClassifier(), 

            KNeighborsClassifier(), GaussianNB(), Perceptron(), SGDClassifier()]



    scores_train = []

    scores_validation = []

    for model in models:

        model.fit(x_train, y_train)

        predition = model.predict(x_validation)        

        scores_train.append(model.score(x_train, y_train))

        scores_validation.append(model.score(x_validation, y_validation))



    return names, scores_train, scores_validation
nome, resultado_treino, resultado_validacao = fit_and_score(x_train, x_test, y_train, y_test)



print(" Resultados ")

# for n, r_train, r_validation, a in zip(nome, resultado_treino, resultado_validacao, acuracia):

for n, r_train, r_validation in zip(nome, resultado_treino, resultado_validacao):

    print("_"*30)

    print("Model: {}".format(n))

    print("Score train: {:0.3}".format(r_train))

    print("Score validation: {:0.3}".format(r_validation))



print("\n")
model_OneVsRestClassifier = OneVsRestClassifier(LinearSVC(random_state=0))

model_OneVsOneClassifier = OneVsOneClassifier(LinearSVC(random_state=0))

model_MultinomialNB = MultinomialNB() 

model_AdaBoostClassifier =  AdaBoostClassifier()

model_LogisticRegression = LogisticRegression()

model_RandomForestClassifier = RandomForestClassifier()

model_GaussianNB = GaussianNB()
model_OneVsOneClassifier.fit(X,Y)

model_OneVsRestClassifier.fit(X,Y)

model_MultinomialNB.fit(X,Y)

model_AdaBoostClassifier.fit(X,Y)

model_LogisticRegression.fit(X,Y)

model_RandomForestClassifier.fit(X,Y)

model_GaussianNB.fit(X,Y)
y_OneVsOneClassifier = model_OneVsOneClassifier.predict(x)

y_OneVsRestClassifier = model_OneVsRestClassifier.predict(x)

y_MultinomialNB = model_MultinomialNB.predict(x)

y_AdaBoostClassifier = model_AdaBoostClassifier.predict(x)

y_LogisticRegression = model_LogisticRegression.predict(x)

y_RandomForestClassifier = model_RandomForestClassifier.predict(x)

y_GaussianNB = model_GaussianNB.predict(x)
y_all = y_OneVsOneClassifier + y_OneVsRestClassifier + y_MultinomialNB + y_AdaBoostClassifier

y_all +=  y_LogisticRegression + y_RandomForestClassifier + y_GaussianNB



y_all
n = len(y_all)

for i in range(n):

    if y_all[i] < 5:

        y_all[i] = 0

    else:

        y_all[i] = 1



y_all        
submission = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": y_all

    })



#submission.to_csv('submission.csv', index=False)