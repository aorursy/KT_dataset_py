from sklearn.datasets import load_iris

iris = load_iris()
print(iris.target_names)
data = iris.data # atributos
target = iris.target # classes
print(data[0])
print(iris.target_names[target[0]])
# separando os dados
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.9, shuffle=True, random_state=42)

print(data.shape)
print(X_train.shape)
print(X_test.shape)
# treinando o modelo
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# predizendo
y_pred = knn.predict(X_test)
print(y_pred)
print(y_test)

# comparando com gabarito
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))
import itertools
import numpy as np
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Matriz de confusão',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Gabarito')
    plt.xlabel('Predição')
    plt.tight_layout()


# calcula a matriz de confusão
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=iris.target_names,
                      title='Matriz de confusão sem normalização')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=iris.target_names, normalize=True,
                      title='Matriz de confusão normalizada')

plt.show()
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_r = pca.fit_transform(X_test)

colors = ['navy', 'turquoise', 'darkorange']
for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(X_r[y_test == i, 0], X_r[y_test == i, 1], color=color, alpha=.8, label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
# Coloque seu código aqui
X_train, X_test, y_train, y_test = train_test_split(data, target, 
                                                    test_size=0.5, 
                                                    shuffle=True, 
                                                    random_state=42)
print(data.shape)
print(X_train.shape)
print(X_test.shape)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=iris.target_names,
                      title='Matriz de confusão sem normalização')
knn = KNeighborsClassifier(n_neighbors=3, weights = "uniform", metric = "euclidean")
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)
# Coloque o código aqui
pca = PCA(n_components=2)
X_r = pca.fit_transform(X_train)

for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(X_r[y_train == i, 0], X_r[y_train == i, 1], color=color, alpha=.8, label=target_name)
    
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset - Treino')
pca = PCA(n_components=2)
X_r = pca.fit_transform(X_test)

for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(X_r[y_test == i, 0], X_r[y_test == i, 1], color=color, alpha=.8, label=target_name)
    
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset - Teste')
plt.plot()
pca = PCA(n_components=2)
X_r = pca.fit_transform(data)

for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(X_r[target == i, 0], X_r[target == i, 1], color=color, alpha=.8, label=target_name)
    
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of IRIS dataset')
plt.plot()