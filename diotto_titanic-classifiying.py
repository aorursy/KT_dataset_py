import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn import metrics #accuracy measure



def trataDf(df):

    #Sexo

    df['Sex'] = pd.factorize(df['Sex'].values)[0]

    #Cabine

    df.drop(['Cabin'],inplace=True, axis=1)

    #Corrigindo Idade (NA <- Média)

    df['Age'].fillna(df['Age'].mean(), inplace=True)

    #Corrigindo a Taxa (Para a média)

    df['Fare'].fillna(df['Fare'].mean(), inplace=True)

    #Corrigindo NAs para "S" e transformando em fator

    df['Embarked'].fillna('S', inplace=True)

    df['Embarked'] = pd.factorize(df['Embarked'].values)[0]

    df['SName'] = df['Name'].str.split(',',expand=True)[0]

    df['SName'] = pd.factorize(df['SName'].values)[0]

    return df



#Função idêntica à usada nos modelos de regressão.

def avalia_classificador(clf, kf, X, y, f_metrica):

    metrica = []

    for train, valid in kf:

        x_train = X[train]

        y_train = y[train]

        x_valid = X[valid]

        y_valid = y[valid]

        clf.fit(x_train, y_train)

        y_pred = clf.predict(x_valid)

        metrica.append(f_metrica(y_valid, y_pred))

    return np.array(metrica).mean()
df = pd.read_csv("../input/train.csv")

df = trataDf(df)


y = df['Survived'].values

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked','SName']].values



lr = LogisticRegression()

lr.fit(X, y)
dfTest = pd.read_csv("../input/test.csv")

dfTest = trataDf(dfTest)

XTest = dfTest[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked','SName']].values

predict = lr.predict(XTest)



result = pd.DataFrame()

result['PassengerId'] = dfTest['PassengerId']

result['Survived'] = predict

result.to_csv('result.csv',index=False)
from sklearn.cross_validation import KFold

kf = KFold(y.shape[0], n_folds=5, shuffle=True)

lr = LogisticRegression()

media_acuracia = avalia_classificador(lr, kf, X, y, accuracy_score) 



print('Acurácia: ', media_acuracia, '%')

media_auc = avalia_classificador(lr, kf, X, y, roc_auc_score) 

print('AUC: ', media_auc)