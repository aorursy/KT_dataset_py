from sklearn.datasets import load_wine
wine = load_wine()

data = wine.data
target = wine.target
print(data.shape)
# separando os dados em treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)

# separando o conjunto de treino em validação também
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
# treinando o modelo 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# avaliando o modelo
from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_val)
accuracy_score(y_val, y_pred)
best_model = None
best_accuracy = 0

for k in [1,2,3,4,5]:

    knn = KNeighborsClassifier(n_neighbors = k) # a cada passo, o parâmetro assume um valor
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print('K:', k, '- ACC:', acc)
    
    if acc > best_accuracy:
        best_model = knn
        best_accuracy = acc
        
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print()
print('Melhor modelo:')
print('K:', best_model.get_params()['n_neighbors'], '- ACC:', acc * 100) #corrigir
# embaralhando os dados várias vezes e re-executando o experimento
import numpy as np
from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0) # 5 execuções diferentes com 20% dos dados para teste

acc = []
for train_index, test_index in ss.split(data):
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(data[train_index],target[train_index])
    y_pred = knn.predict(data[test_index])
    acc.append(accuracy_score(y_pred,target[test_index]))

acc = np.asarray(acc) # converte pra numpy pra ficar mais simples de usar média e desvio padrão
print(acc)
print('Acurácia - %.2f +- %.2f' % (acc.mean() * 100, acc.std() * 100))
# utilizando validação cruzada com cross_val_score
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors = 1)
scores = cross_val_score(knn, data, target, cv=5) # 5 execuções diferentes com 20% dos dados para teste

print('Acurácia - %.2f +- %.2f' % (scores.mean() * 100, scores.std() * 100))
# utilizando validação cruzada com KFold
from sklearn.model_selection import KFold
kf = KFold(n_splits = 5)

acc = []
for train_index, test_index in kf.split(data):
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(data[train_index],target[train_index])
    y_pred = knn.predict(data[test_index])
    acc.append(accuracy_score(y_pred,target[test_index]))

acc = np.asarray(acc) # converte pra numpy pra ficar mais simples de usar média e desvio padrão
print(acc)
print('Acurácia - %.2f +- %.2f' % (acc.mean() * 100, acc.std() * 100))
# utilizando validação cruzada com KFold
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits = 5)

acc = []
for train_index, test_index in kf.split(data, target): # precisa passar as classes agora para que a divisão aconteça
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(data[train_index],target[train_index])
    y_pred = knn.predict(data[test_index])
    acc.append(accuracy_score(y_pred,target[test_index]))

acc = np.asarray(acc) # converte pra numpy pra ficar mais simples de usar média e desvio padrão
print('Acurácia - %.2f +- %.2f' % (acc.mean() * 100, acc.std() * 100))
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_r = pca.fit_transform(data)

colors = ['navy', 'turquoise', 'darkorange']
for color, i, target_name in zip(colors, [0, 1, 2], wine.target_names):
    plt.scatter(X_r[target == i, 0], X_r[target == i, 1], color=color, alpha=.8, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA')
# verificando a escala dos atributos
import pandas as pd
df = pd.DataFrame(data)
df.describe()
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

scaler = StandardScaler()
scaler.fit(data)
data_s = scaler.transform(data)

df = pd.DataFrame(data_s)
df.describe()
pca = PCA(n_components=2)
X_r = pca.fit_transform(data_s)

colors = ['navy', 'turquoise', 'darkorange']
for color, i, target_name in zip(colors, [0, 1, 2], wine.target_names):
    plt.scatter(X_r[target == i, 0], X_r[target == i, 1], color=color, alpha=.8, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA')
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits = 5)

acc = []
for train_index, test_index in kf.split(data, target): # precisa passar as classes agora para que a divisão aconteça
    knn = KNeighborsClassifier(n_neighbors = 1)
    
    scaler = StandardScaler()
    train = scaler.fit_transform(data[train_index]) # somente dados de treino no fit
    test = scaler.transform(data[test_index]) # aplica-se transform no teste apenas
    
    knn.fit(train,target[train_index])
    y_pred = knn.predict(test)
    acc.append(accuracy_score(y_pred,target[test_index]))

acc = np.asarray(acc) # converte pra numpy pra ficar mais simples de usar média e desvio padrão
print('Acurácia - %.2f +- %.2f' % (acc.mean() * 100, acc.std() * 100))
# utilizando validação cruzada com cross_val_score
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
pipeline = Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors = 1))])
scores = cross_val_score(pipeline, data, target, cv=5) # 5 execuções diferentes com 20% dos dados para teste

print('Acurácia - %.2f +- %.2f' % (scores.mean() * 100, scores.std() * 100))
# separa-se uma parcela para encontrar os melhores parâmetros (5% do original)
data_gs, data_cv, target_gs, target_cv = train_test_split(data, target, test_size=0.95, random_state=42, stratify=target)

# uma forma automática de StandardScaler + CLF
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())])

# utiliza-se GridSearchCV para achar os melhores parâmetros
from sklearn.model_selection import GridSearchCV
parameters = {'clf__n_neighbors': [1,2,3,4,5], 'clf__weights' : ['uniform','distance']} # quais parâmetros e quais valores serão testados
clf = GridSearchCV(pipeline, parameters, cv=3, iid=False) # clf vai armazenar qual foi a melhor configuração
clf.fit(data_gs, target_gs)

print(clf.best_params_)

# utilizando validação cruzada para avaliar o modelo
scores = cross_val_score(clf.best_estimator_, data_cv, target_cv, cv=5)
print('Acurácia - %.2f +- %.2f' % (scores.mean() * 100, scores.std() * 100))

clf = clf.best_estimator_

from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits = 5)

acc = []
for train_index, test_index in kf.split(data_cv, target_cv): # precisa passar as classes agora para que a divisão aconteça
    
    #scaler = StandardScaler()
    #train = scaler.fit_transform(data_cv[train_index]) # somente dados de treino no fit
    #test = scaler.transform(data_cv[test_index]) # aplica-se transform no teste apenas
    
    clf.fit(data_cv[train_index],target_cv[train_index])
    y_pred = clf.predict(data_cv[test_index])
    acc.append(accuracy_score(y_pred,target_cv[test_index]))

acc = np.array(acc)
print('Acurácia - %.2f +- %.2f' % (acc.mean() * 100, acc.std() * 100))
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
# separando os dados em treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# separando o conjunto de treino em validação também
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
# treinando o modelo 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# avaliando o modelo
from sklearn.metrics import accuracy_score
y_pred = knn.predict(X_val)
accuracy_score(y_val, y_pred)
# Inicialmente obtemos a acurácia de 90.47% com 3 vizinhos no KNN
# Então, nós questionamos se era o melhor número de vizinhos... com isso, criamos um range de 10 para testarmos essa nossa hipótese
best_model = None
best_accuracy = 0

for k in range(1,11):
    knn = KNeighborsClassifier(n_neighbors = k) # a cada passo, o parâmetro assume um valor
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print('K:', k, '- ACC:', acc)
    
    if acc > best_accuracy:
        best_model = knn
        best_accuracy = acc
        
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

best_k = best_model.get_params()['n_neighbors']
print('\nMelhor modelo:')
print('K:', best_k, '- ACC:', acc * 100)
# Vimos que com 6 vizinhos obtemos uma acurácia muito melhor! Manteremos ela para os próximos passos (variável best_k)
# Diante dos estudos apresentados, seguimos na ideia de duas frentes e compararmos diante do nosso best_k: embaralharmos os dados usando ShuffleSplit; e validação cruzada com cross_val_score
# embaralhando os dados várias vezes e re-executando o experimento
import numpy as np
from sklearn.model_selection import ShuffleSplit
ss = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0) # 5 execuções diferentes com 20% dos dados para teste

acc = []
for train_index, test_index in ss.split(X):
    knn = KNeighborsClassifier(n_neighbors = best_k)
    knn.fit(X[train_index],y[train_index])
    y_pred = knn.predict(X[test_index])
    acc.append(accuracy_score(y_pred,y[test_index]))

acc = np.asarray(acc) # converte pra numpy pra ficar mais simples de usar média e desvio padrão
print(acc)
print('Acurácia - %.2f +- %.2f' % (acc.mean() * 100, acc.std() * 100))
# utilizando validação cruzada com cross_val_score
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors = best_k)
scores = cross_val_score(knn, X, y, cv=5) # 5 execuções diferentes com 20% dos dados para teste

print('Acurácia - %.2f +- %.2f' % (scores.mean() * 100, scores.std() * 100))
# Comparando as duas, obtivemos uma acurácia melhor com a validação cruzada (cross_val_score), porém,o erro se apresenta maior.
# Portanto, como essas práticas são mais confiáveis do que utilizar apenas KNN, consideramos até o momento o cross_val_score como melhor acurácia (92.28 +- 1.02)

# Para tirarmos conclusões melhores, decidimos usar a validação cruzada KFold para fins de comparação
# utilizando validação cruzada com KFold
from sklearn.model_selection import KFold
kf = KFold(n_splits = 5)

acc = []
for train_index, test_index in kf.split(X):
    knn = KNeighborsClassifier(n_neighbors = best_k)
    knn.fit(X[train_index],y[train_index])
    y_pred = knn.predict(X[test_index])
    acc.append(accuracy_score(y_pred,y[test_index]))

acc = np.asarray(acc) # converte pra numpy pra ficar mais simples de usar média e desvio padrão
print(acc)
print('Acurácia - %.2f +- %.2f' % (acc.mean() * 100, acc.std() * 100))
# Foi um resultado muito pior devido ao erro de 3.51

# Pelos efeitos da estratificação, vamos utilizar o StratifiedKFold para sanarmos a dúvida
# utilizando validação cruzada com KFold
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits = 5)

acc = []
for train_index, test_index in kf.split(X, y): # precisa passar as classes agora para que a divisão aconteça
    knn = KNeighborsClassifier(n_neighbors = best_k)
    knn.fit(X[train_index],y[train_index])
    y_pred = knn.predict(X[test_index])
    acc.append(accuracy_score(y_pred,y[test_index]))

acc = np.asarray(acc) # converte pra numpy pra ficar mais simples de usar média e desvio padrão
print('Acurácia - %.2f +- %.2f' % (acc.mean() * 100, acc.std() * 100))
# A acurácia em si piorou e obtivemos um erro maior do que o esperado

# Vamos analisar nossos dados para ver possíveis outliers ou dados "desnecessários"
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

pca = PCA(n_components=2)
X_r = pca.fit_transform(X)

colors = ['navy', 'turquoise', 'darkorange']
for color, i, target_name in zip(colors, [0, 1, 2], breast_cancer.target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA')
# Podemos ver claramente que há vários outliers fora do nosso aglomerado de dados

# Vamos utilizar o StandardScaler para padronizarmos nossos dados
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Deixando nossos dados padronizados
scaler = StandardScaler()
scaler.fit(X)
data_s = scaler.transform(X)
# Agora vamos visualizar novamente após a padronização
pca = PCA(n_components=2)
X_r = pca.fit_transform(data_s)

colors = ['navy', 'turquoise', 'darkorange']
for color, i, target_name in zip(colors, [0, 1, 2], breast_cancer.target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA')
# Muito melhor! Podemos notar agora que temos um padrão e não temos dados altamente fora da escala!

# Vamos, então, aplicar novamente nossos conceitos anteriormente realizados para ver se obtemos melhoras
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits = 5)

acc = []
for train_index, test_index in kf.split(X, y): # precisa passar as classes agora para que a divisão aconteça
    knn = KNeighborsClassifier(n_neighbors = best_k)
    
    scaler = StandardScaler()
    train = scaler.fit_transform(X[train_index]) # somente dados de treino no fit
    test = scaler.transform(X[test_index]) # aplica-se transform no teste apenas
    
    knn.fit(train,y[train_index])
    y_pred = knn.predict(test)
    acc.append(accuracy_score(y_pred,y[test_index]))

acc = np.asarray(acc) # converte pra numpy pra ficar mais simples de usar média e desvio padrão
print('Acurácia - %.2f +- %.2f' % (acc.mean() * 100, acc.std() * 100))
# Muito melhor! Por fim, vamos comparar com a validação cruzada (cross_val_score)
# utilizando validação cruzada com cross_val_score
from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline
pipeline = Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier(n_neighbors = best_k))])
scores = cross_val_score(pipeline, X, y, cv=5) # 5 execuções diferentes com 20% dos dados para teste

print('Acurácia - %.2f +- %.2f' % (scores.mean() * 100, scores.std() * 100))
# Curiosamente obtivemos o mesmo resultado! Até então, este tem sido a melhor acurárica obtida

# Para fins de dúvida, vamos aplicar o GridSearch seguido da validação cruzada e StratifiedKFold para compararmos
# separa-se uma parcela para encontrar os melhores parâmetros (5% do original)
data_gs, data_cv, target_gs, target_cv = train_test_split(X, y, test_size=0.95, random_state=42)

# uma forma automática de StandardScaler + CLF
from sklearn.pipeline import Pipeline
pipeline = Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())])

# utiliza-se GridSearchCV para achar os melhores parâmetros
from sklearn.model_selection import GridSearchCV

parameters = {'clf__n_neighbors': range(1,11), 'clf__weights' : ['uniform','distance']} # quais parâmetros e quais valores serão testados
clf = GridSearchCV(pipeline, parameters, cv=3, iid=False) # clf vai armazenar qual foi a melhor configuração
clf.fit(data_gs, target_gs)

print(clf.best_params_)

# utilizando validação cruzada para avaliar o modelo
scores = cross_val_score(clf.best_estimator_, data_cv, target_cv, cv=5)
print('Acurácia com Validação Cruzada - %.2f +- %.2f' % (scores.mean() * 100, scores.std() * 100))

clf = clf.best_estimator_

from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits = 5)

acc = []
for train_index, test_index in kf.split(data_cv, target_cv): # precisa passar as classes agora para que a divisão aconteça
    clf.fit(data_cv[train_index],target_cv[train_index])
    y_pred = clf.predict(data_cv[test_index])
    acc.append(accuracy_score(y_pred,target_cv[test_index]))

acc = np.array(acc)
print('Acurácia com StratifiedKFold - %.2f +- %.2f' % (acc.mean() * 100, acc.std() * 100))
# Obtivemos uma acurácia menor com erro maior!
# Portanto, a melhor acurácia obtida é a de 96.48 +- 0.97 utilizando StratifiedKFold ou Validação Cruzada (cross_val_score) após a padronização dos nossos dados (StandardScaler)!
# Resgatando nossa melhor acurácia
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# separando os dados em treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# separando o conjunto de treino em validação também
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.33, random_state=42)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Deixando nossos dados padronizados
scaler = StandardScaler()
scaler.fit(X)
data_s = scaler.transform(X)
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits = 5)

acc = []
for train_index, test_index in kf.split(X, y): # precisa passar as classes agora para que a divisão aconteça
    knn = KNeighborsClassifier(n_neighbors = best_k)
    
    scaler = StandardScaler()
    train = scaler.fit_transform(X[train_index]) # somente dados de treino no fit
    test = scaler.transform(X[test_index]) # aplica-se transform no teste apenas
    
    knn.fit(train,y[train_index])
    y_pred = knn.predict(test)
    acc.append(accuracy_score(y_pred,y[test_index]))

acc = np.asarray(acc) # converte pra numpy pra ficar mais simples de usar média e desvio padrão
print('Acurácia - %.2f +- %.2f' % (acc.mean() * 100, acc.std() * 100))
# Definindo metodos para calcular precisao, recall e fmedida
def precisao(vp, fp):
    return vp / (vp + fp)

def revocacao(vp, fn):
    return vp / (vp + fn)

def fmedida(vp, fp, fn):
    p = precisao(vp, fp)
    r = revocacao(vp, fn)
    return 2 * (p * r / (p + r))
# Definindo a matriz de contingência
def build_contingence_table(Y, Y_pred_1, Y_pred_2):
    y1_and_y2 = 0
    y1_and_not_y2 = 0
    y2_and_not_y1 = 0
    not_y1_and_not_y2 = 0
    for y, y1, y2 in zip(Y, Y_pred_1, Y_pred_2):
        if y == y1 == y2:
            y1_and_y2 += 1
        elif y != y1 and y != y2:
            not_y1_and_not_y2 += 1
        elif y == y1 and y != y2:
            y1_and_not_y2 += 1
        elif y != y1 and y == y2:
            y2_and_not_y1 += 1
            
    contingency_table = [[y1_and_y2, y1_and_not_y2], 
                         [y2_and_not_y1, not_y1_and_not_y2]]
    
    return contingency_table
from statsmodels.stats.contingency_tables import mcnemar
# Utilizando PCA para visualizarmos a redução de atributos
pca = PCA(n_components=0.999, whiten=True)

X_pca = pca.fit_transform(X)

print('Número original de atributos:', X.shape[1])
print('Número reduzido de atributos:', X_pca.shape[1])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=4000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# calculando a matriz utilizando scikit-learn
from sklearn.metrics import confusion_matrix

vn, fp, fn, vp = confusion_matrix(y_test, y_pred).ravel()
print('VP nos dados originais:', vp)
print('FP nos dados originais:', fp)
print('VN nos dados originais:', vn)
print('FN nos dados originais:', fn)
print('Recall:', revocacao(vp, fn))

print('Acurácia nos dados originais:', accuracy_score(y_test, y_pred))
print('\n\n')

#######

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.33, random_state=42)

model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=4000)

model.fit(X_train, y_train)
y_pred_pca = model.predict(X_test)

# calculando a matriz utilizando scikit-learn
vn, fp, fn, vp = confusion_matrix(y_test, y_pred).ravel()
print('VP nos dados reduzidos (PCA em tudo):', vp)
print('FP nos dados reduzidos (PCA em tudo):', fp)
print('VN nos dados reduzidos (PCA em tudo):', vn)
print('FN nos dados reduzidos (PCA em tudo):', fn)
print('Recall:', revocacao(vp, fn))

print('Acurácia nos dados reduzidos (PCA em tudo):', accuracy_score(y_test, y_pred))
print('\n\n')

#######

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

pca = PCA(n_components=0.95, whiten=True)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=4000)

model.fit(X_train, y_train)
y_pred_pca_parte_certa = model.predict(X_test)

# calculando a matriz utilizando scikit-learn
vn, fp, fn, vp = confusion_matrix(y_test, y_pred).ravel()
print('VP nos dados originais (PCA da parte certa):', vp)
print('FP nos dados originais (PCA da parte certa):', fp)
print('VN nos dados originais (PCA da parte certa):', vn)
print('FN nos dados originais (PCA da parte certa):', fn)
print('Recall:', revocacao(vp, fn))

print('Acurácia nos dados originais (PCA da parte certa):', accuracy_score(y_test, y_pred))
# Vamos agora utilizar a seleção de atributos para compararmos
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import numpy as np

import warnings
warnings.filterwarnings("ignore")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

fvalue_selector = SelectKBest(f_classif, k=20)

X_kbest = fvalue_selector.fit_transform(X_train, y_train)

print('Número original de atributos:', X.shape[1])
print('Número reduzido de atributos:', X_kbest.shape[1])
# Aplicamos o treino, então, em cima dos dados originais e comparamos com o KBest
model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=4000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Acurácia nos dados originais:', accuracy_score(y_test, y_pred))

# calculando a matriz utilizando scikit-learn
vn, fp, fn, vp = confusion_matrix(y_test, y_pred).ravel()
print('VP nos dados originais:', vp)
print('FP nos dados originais:', fp)
print('VN nos dados originais:', vn)
print('FN nos dados originais:', fn)
print('Recall:', revocacao(vp, fn))

print('\n\n')
###

model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=4000)
model.fit(X_kbest, y_train)
X_test_kbest = fvalue_selector.transform(X_test)
y_pred_kbest = model.predict(X_test_kbest)
print('Acurácia nos dados Kbest:', accuracy_score(y_test, y_pred))

# calculando a matriz utilizando scikit-learn
vn, fp, fn, vp = confusion_matrix(y_test, y_pred).ravel()
print('VP nos dados Kbest:', vp)
print('FP nos dados Kbest:', fp)
print('VN nos dados Kbest:', vn)
print('FN nos dados Kbest:', fn)
print('Recall:', revocacao(vp, fn))
# Vamos rodar o McNemmar para verificar se os dois modelos possuem a mesma proporção de erros
contingence_table = build_contingence_table(y_test, y_pred_pca, y_pred_pca_parte_certa)

import pprint

pprint.pprint(contingence_table)
result = mcnemar(contingence_table, exact=True)
    
    
if result.pvalue >= 0.001:
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
else:
    print('statistic=%.3f, p-value=%.3e' % (result.statistic, result.pvalue))

# interpretando o p-value
alpha = 0.05
if result.pvalue > alpha:
    print('Mesma proporção de erros (falhou em rejeitar H0)')
else:
    print('Proporções de erros diferentes (rejeitou H0)')
# Com isso, vemos que falhamos em rejeitar H0, então não é possível afirmar que os modelos tem resultados diferentes.
# Isso afirma a causa de que nosso Recall não teve diferença!
# Vamos agora plotar na curva ROC as três comparações
from sklearn import svm, datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
# Original
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=4000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_pred = model.predict_proba(X_test)[::,1]
fpr_original, tpr_original, _ = metrics.roc_curve(y_test,  y_pred)
auc_original = metrics.roc_auc_score(y_test, y_pred)


# PCA
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

pca = PCA(n_components=0.95, whiten=True)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=4000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

y_pred = model.predict_proba(X_test)[::,1]
fpr_pca, tpr_pca, _ = metrics.roc_curve(y_test,  y_pred)
auc_pca = metrics.roc_auc_score(y_test, y_pred)


# KBest
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

fvalue_selector = SelectKBest(f_classif, k=20)

X_kbest = fvalue_selector.fit_transform(X_train, y_train)

model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=4000)
model.fit(X_kbest, y_train)
X_test_kbest = fvalue_selector.transform(X_test)
y_pred = model.predict(X_test_kbest)

y_pred = model.predict_proba(X_test_kbest)[::,1]
fpr_kbest, tpr_kbest, _ = metrics.roc_curve(y_test,  y_pred)
auc_kbest = metrics.roc_auc_score(y_test, y_pred)
# Plotando
plt.plot(fpr_original,tpr_original,label="Original (area = %0.2f)" % auc_original)
plt.plot(fpr_pca, tpr_pca, lw=1, label="PCA (area = %0.2f)" % auc_pca)
plt.plot(fpr_kbest, tpr_kbest, lw=1, label="KBest( area = %0.2f)" % auc_kbest)

plt.legend(loc=4)
plt.show()
