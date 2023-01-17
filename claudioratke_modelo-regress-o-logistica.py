print('Carregando as bibliotecas...')

import numpy as np

import pandas as pd #CSV

from sklearn import cross_validation as cv  

from sklearn.cross_validation import KFold 

from sklearn.tree import DecisionTreeClassifier



print('carregndo as bases de teste e treinamento...')

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64})

print('Tamanho da base de treinamento:', len(train))



print('Limpando os dados...')

def harmonize_data(titanic):

    # Preenchendo os dados em branco

    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].mean())

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].mean())

    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    return titanic



train_data = harmonize_data(train)



#train_features = train_data.DataFrame([train["Age"]]).T



log_model = linear_model.LogisticRegression()



# Train the model

log_model.fit(X = train_features ,

              y = titanic_train["Survived"])



# Check trained model intercept

print(log_model.intercept_)



# Check trained model coefficients

print(log_model.coef_)