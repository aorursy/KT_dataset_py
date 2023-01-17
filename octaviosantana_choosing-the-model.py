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
#Reading the DataFrame and Clearing the Data

def data_frame(arquivo):

    df = pd.read_csv(arquivo)

    df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)    

    df['Age'].fillna(df['Age'].mean(), inplace=True)

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    y = df[df.columns[0]]

    x = df[df.columns[1:]]

    x = pd.get_dummies(x)    

    return x, y
X, Y = data_frame('../input/train.csv')
# Y.head() -> Survived

X.head()
#print(Y.isnull().any()) -> False

print(X.isnull().any())
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, Y)



x_train = np.array(x_train).reshape(len(x_train),10)

x_test = np.array(x_test).reshape(len(x_test),10)

y_train = y_train.values.ravel()

y_test = y_test.values.ravel()
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC

from sklearn.multiclass import OneVsRestClassifier

from sklearn.multiclass import OneVsOneClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LinearRegression

from sklearn import tree
def fit_and_score(x_train, x_validation, y_train, y_validation):

    names = ["OneVsrest", "OneVsOne", "MultinomialNB", "AdaBoost", "LinearRegression", "DecisionTreeRegressor", 

             "AdaBoostRegressor", "GradientBoostingRegressor"]

    models=[OneVsRestClassifier(LinearSVC(random_state=0)), OneVsOneClassifier(LinearSVC(random_state=0)), 

            MultinomialNB(), AdaBoostClassifier(), LinearRegression(), tree.DecisionTreeRegressor(), 

            AdaBoostRegressor(), GradientBoostingRegressor()]



    scores_train = []

    scores_validation = []

    for model in models:

        model.fit(x_train, y_train)

        predition = model.predict(x_validation)        

        scores_train.append(model.score(x_train, y_train))

        scores_validation.append(model.score(x_validation, y_validation))



    return names, scores_train, scores_validation
nome, resultado_treino, resultado_validacao = fit_and_score(x_train, x_test, y_train, y_test)



print(" Results ")

for n, r_train, r_validation in zip(nome, resultado_treino, resultado_validacao):

    print("_"*30)

    print("Model: {}".format(n))

    print("Score train: {:0.3}".format(r_train))

    print("Score validation: {:0.3}".format(r_validation))



print("\n")
print('Winner model is AdaBoostClassifier')