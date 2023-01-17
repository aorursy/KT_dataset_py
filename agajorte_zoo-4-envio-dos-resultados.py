# importar pacotes necessários

import numpy as np

import pandas as pd
# importar os pacotes necessários para os algoritmos de classificação

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Perceptron

from sklearn.linear_model import Ridge

from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
# carregar arquivo de dados de treino

train_data = pd.read_csv('../input/zoo-train.csv', index_col='animal_name')
# carregar arquivo de dados de treino

data2 = pd.read_csv('../input/zoo-train2.csv', index_col='animal_name')
# unir ambos os dados de treinamento

train_data = train_data.append(data2)
# carregar arquivo de dados de teste

test_data = pd.read_csv('../input/zoo-test.csv', index_col='animal_name')
# transformar y/n em 0/1



bool_cols = train_data.columns.values.tolist()

bool_cols.remove('legs')

bool_cols.remove('class_type')



for data in [train_data, test_data]:

    for col in bool_cols:

        data[col] = data[col].map({'n': 0, 'y': 1}).astype(int)
# definir dados de treino



X_train = train_data.drop(['class_type'], axis=1) # tudo, exceto a coluna alvo

y_train = train_data['class_type'] # apenas a coluna alvo



print('Forma dos dados de treino:', X_train.shape, y_train.shape)
# definir dados de teste



X_test = test_data # tudo, já que não possui a coluna alvo



print('Forma dos dados de teste:', X_test.shape)
models = []

models.append(('LR', LogisticRegression(random_state=42, solver='lbfgs', multi_class='auto', max_iter=500, C=100)))

models.append(('DT', DecisionTreeClassifier(random_state=42, criterion='gini', max_depth=11)))

models.append(('KNN', KNeighborsClassifier(n_neighbors=1)))

models.append(('SVM', SVC(random_state=42, C=10, gamma=0.1, kernel='rbf')))

models.append(('RF', RandomForestClassifier(random_state=42, max_features='auto', n_estimators=10)))

models.append(('SGD', SGDClassifier(random_state=42, max_iter=100, tol=0.1)))

models.append(('NN', Perceptron(random_state=42, max_iter=100, tol=0.01)))

models.append(('NB', GaussianNB(priors=None, var_smoothing=1e-08)))

models.append(('LSVM', LinearSVC(random_state=42, max_iter=1000, C=10)))

models.append(('ABDT', AdaBoostClassifier(DecisionTreeClassifier(random_state=42), n_estimators=5)))

models.append(('GB', GradientBoostingClassifier(random_state=42, max_depth=3)))

models.append(('MLP', MLPClassifier(random_state=42, solver='lbfgs', alpha=0.1, hidden_layer_sizes=(15,))))

models.append(('LDA', LinearDiscriminantAnalysis(solver='svd')))
for name, model in models:

    print(model, '\n')

    

    # treinar o modelo

    model.fit(X_train, y_train)

    

    # executar previsão usando o modelo

    y_pred = model.predict(X_test)

    

    # gerar dados de envio (submissão)

    submission = pd.DataFrame({

      'animal_name': X_test.index,

      'class_type': y_pred

    })

    submission.set_index('animal_name', inplace=True)



    # gerar arquivo CSV para o envio

    filename = 'zoo-submission-p-%s.csv' % name.lower()

    submission.to_csv(filename)
# verificar conteúdo dos arquivos gerados

!head zoo-submission-p-*.csv