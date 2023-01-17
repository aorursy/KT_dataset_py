# !conda install scikit-learn=0.22.1 -y
# !conda install -c conda-forge somoclu -y
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('ggplot')

def plot_dt(tree, feature_names, class_names):
    """Função criada para a visualização da árvore de decisão gerada 
    """
    from sklearn.tree import plot_tree
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plot_tree(tree, ax = ax, feature_names=feature_names, class_names=class_names)

def plot_dendrogram(model, **kwargs):
    """Função para a geração de um dendograma de um modelo sklearn.cluster.AgglomerativeClustering
    
    See:
        Função retirada da documentação oficial do scikit-learn
    """
    from scipy.cluster.hierarchy import dendrogram
    
    plt.title('Agrupamento hierárquico')
    
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    plt.xlabel("Índice dos dados (Em parenteses representam a quantidade de elementos no grupo).")
    plt.show()
    

def plot_cm(cm_sklearn, labels):
    """Função para gerar matriz de confusão
    """
    
    import seaborn as sn
    
    df_cm = pd.DataFrame(cm_sklearn, index = [i for i in labels], columns = [i for i in labels])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    
# Configuração das classes
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
_ = le.fit(["d", "h", "o", "s"])
# importando a função para calcular a matriz de confusão
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
train_data = pd.read_csv('https://raw.githubusercontent.com/dataAt/ml-aplicado-dados-espacial/master/src/metodos-supervisionados/dados/forest_type/training.csv')
test_data = pd.read_csv('https://raw.githubusercontent.com/dataAt/ml-aplicado-dados-espacial/master/src/metodos-supervisionados/dados/forest_type/testing.csv')
data_columns = ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'b9']
x_train = train_data[data_columns]
y_train = train_data['class']
x_test = test_data[data_columns]
y_test = test_data['class']
## Definindo classes presentes no conjunto de dados
class_names = y_train.unique().tolist()
class_names
import numpy as np
import pandas as pd
from plotnine import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
# Calculando o índice Kappa
cohen_kappa_score(y_pred, y_test)
# Calculando o Score
accuracy_score(y_pred, y_test)
resultados = {'Kappa': [], 'k': [], 'Score': []}
for k_value in range(1, 15):
    resultados['k'].append(k_value)
    
    neigh = KNeighborsClassifier(n_neighbors=k_value)
    neigh.fit(x_train, y_train)
    
    y_pred = neigh.predict(x_test)
    
    resultados['Kappa'].append(cohen_kappa_score(y_pred, y_test))
    resultados['Score'].append(accuracy_score(y_pred, y_test))

resultados = pd.DataFrame(resultados)

# Visualizando os resultados
resultados.head()
# Transformando os dados em long para facilitar o plot
resultados = resultados.melt('k', var_name = 'Medidas')
(
    ggplot(resultados, aes(x = 'k', y = 'value', color = 'Medidas'))
        + geom_line()
        + theme_bw()
        + facet_grid('~Medidas', space = 'free_y', scales = 'free')
        + scale_x_continuous(breaks = np.arange(1, 17))
        + labs(
            title = 'Métricas de avaliação - KNN',
            x = 'Quantidade de vizinhos (K)',
            y = 'Acurácia'
        )
)
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
# Calculando o índice Kappa
cohen_kappa_score(y_pred, y_test)
# Calculando o Score
accuracy_score(y_pred, y_test)
# Resultados do KNN
y_pred = knn.predict(x_test)
knn_cm = confusion_matrix(y_test, y_pred)
plot_cm(knn_cm, "dhos")
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
# Calculando o índice Kappa
cohen_kappa_score(y_pred, y_test)
# Calculando o Score
accuracy_score(y_test, y_pred)
plot_dt(clf, data_columns, class_names)
# Com a árvore cortada, façamos todo o processo de avaliação e visualização de dados.
clf = DecisionTreeClassifier(max_depth=2)
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
# Calculando o índice Kappa
cohen_kappa_score(y_pred, y_test)
# Calculando o Score
accuracy_score(y_test, y_pred)
plot_dt(clf, data_columns, class_names)
# Resultados da árvore de decisão
y_pred = clf.predict(x_test)
clf_cm = confusion_matrix(y_test, y_pred)
plot_cm(clf_cm, "dhos")
from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder, StandardScaler
enc = OneHotEncoder()

y_train_factor = enc.fit_transform(train_data['class'].values[:, np.newaxis]).toarray()
y_test_factor = enc.fit_transform(test_data['class'].values[:, np.newaxis]).toarray()
n_features = x_train.shape[1]
n_classes = y_train_factor.shape[1]
model = Sequential(name='Modelo1')
model.add(Dense(4, input_dim=n_features, activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train_factor,
                         batch_size=5,
                         epochs=85,
                         verbose=1,
                         validation_data=(x_test, y_test_factor),)
model.evaluate(x_test, y_test_factor, verbose=0)
history = pd.DataFrame(history.history)
history.plot()
# Resultados da RNA Número de épocas 85
y_pred = model.predict_classes(x_test)
model_cm = confusion_matrix(le.transform(y_test.map(lambda x: x.strip())), y_pred)
plot_cm(model_cm, "dhos")
# Variação do número de épocas
model = Sequential(name='Modelo2')
model.add(Dense(4, input_dim=n_features, activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train_factor,
                         batch_size=5,
                         epochs=48,
                         verbose=1,
                         validation_data=(x_test, y_test_factor),)
model.evaluate(x_test, y_test_factor, verbose=0)
history = pd.DataFrame(history.history)
history.plot()
# Resultados da RNA Número de épocas 48
y_pred = model.predict_classes(x_test)
model_cm = confusion_matrix(le.transform(y_test.map(lambda x: x.strip())), y_pred)
plot_cm(model_cm, "dhos")
# Variação do número de épocas
model = Sequential(name='Modelo3')
model.add(Dense(4, input_dim=n_features, activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train_factor,
                         batch_size=5,
                         epochs=100,
                         verbose=1,
                         validation_data=(x_test, y_test_factor),)
model.evaluate(x_test, y_test_factor, verbose=0)
history = pd.DataFrame(history.history)
history.plot()
# Resultados da RNA Número de épocas 100
y_pred = model.predict_classes(x_test)
model_cm = confusion_matrix(le.transform(y_test.map(lambda x: x.strip())), y_pred)
plot_cm(model_cm, "dhos")
# Variação do número de camadas
model = Sequential(name='Modelo4')
model.add(Dense(4, input_dim=n_features, activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(40, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train_factor,
                         batch_size=5,
                         epochs=48,
                         verbose=1,
                         validation_data=(x_test, y_test_factor),)
model.evaluate(x_test, y_test_factor, verbose=0)
history = pd.DataFrame(history.history)
history.plot()
# Resultados da RNA Número camadas
# 4 Camadas ocultas com função ReLU e Softmax
y_pred = model.predict_classes(x_test)
model_cm = confusion_matrix(le.transform(y_test.map(lambda x: x.strip())), y_pred)
plot_cm(model_cm, "dhos")
# Variação do número da camada final com função ReLU
model = Sequential(name='Modelo5')
model.add(Dense(4, input_dim=n_features, activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(n_classes, activation='relu'))
model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
model.summary()
history = model.fit(x_train, y_train_factor,
                         batch_size=5,
                         epochs=48,
                         verbose=1,
                         validation_data=(x_test, y_test_factor),)
model.evaluate(x_test, y_test_factor, verbose=0)
history = pd.DataFrame(history.history)
history.plot()
# Resultados da RNA Número camadas
# 4 Camadas ocultas com função ReLU e Softmax
y_pred = model.predict_classes(x_test)
model_cm = confusion_matrix(le.transform(y_test.map(lambda x: x.strip())), y_pred)
plot_cm(model_cm, "dhos")