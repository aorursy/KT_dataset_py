import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.model_selection import KFold



#modelos

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.naive_bayes import GaussianNB
#carregar dados iris

file = '../input/iris-train.csv' # caminho absoluto do arquivo

df = pd.read_csv(file, delimiter = ',', index_col='Id')



#variaveis features

X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]



# variavel objetivo

Y = df['Species']
models = []

models.append(('Support Vector Machines - linear', svm.SVC(kernel='linear', random_state=0, gamma=.10, C=2.0) ))

models.append(('Logistic Regression - lbfgs', LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000) ))

models.append(('KNeighborsClassifier', KNeighborsClassifier() ))

models.append(('DecisionTreeClassifier', DecisionTreeClassifier() ))

models.append(('RandomForestClassifier', RandomForestClassifier(n_estimators=100) ))

models.append(('Gaussian Naïve Bayes', GaussianNB()))

# importando cross_val_score

from sklearn.model_selection import cross_val_score


names = []

scores = []

for name, model in models:

    score = (cross_val_score(model, X, Y, cv=10)) # 10 iterações

    names.append(name)

    scores.append(score)



scoresConsolidado = []

for sc in scores:

    scoresConsolidado.append(np.average(sc))



results = pd.DataFrame({'Model': names, 'Score': scoresConsolidado})

results = results.sort_values(by=['Score'], ascending=False) 



print(results)
# separando dados 70% treino 30% teste

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 101)



#selecionando o modelo escolhido (apos testes) para fazer a predição.

model = models[0][1]

filename = models[0][0]



# treinamento com todos os registros da base de treinamento

X_train = df.drop(['Species'], axis=1) # tudo, exceto a coluna alvo

Y_train = df['Species'] # apenas a coluna alvo

model.fit(X_train, Y_train)



#Realizando a previsao arquivo de teste

dfPredict = pd.read_csv('../input/iris-test.csv', delimiter=',', index_col='Id')

xPredict  = dfPredict[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

yPredict  = model.predict(xPredict)

    

# gerar dados de envio (submissão)

submission = pd.DataFrame({

    'Id': xPredict.index,

    'Species': yPredict

})

submission.set_index('Id', inplace=True)

submission.to_csv(filename+'.csv')