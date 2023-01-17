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
# treinando o modelo
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# predizendo
y_pred = knn.predict(X_test)

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
iris = load_iris()
print(iris.target_names)
data = iris.data # atributos
target = iris.target # classes
print(data)
print(target)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.5, shuffle=True, random_state=42)

# treinando o modelo
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# predizendo
y_pred = knn.predict(X_test)

print('y_pred', y_pred)
print('y_test', y_test)

# comparando com gabarito
accuracy_score(y_test, y_pred)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# scaler = StandardScaler() #0.9733333333333334
# scaler = RobustScaler() #0.92
# scaler = MinMaxScaler() # 0.9866666666666667
# data = scaler.fit_transform(data)

k_range = range(1, 26)

scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.5, shuffle=True, random_state=42)

    knn.fit(X_train, y_train)

    #Predict testing set
    y_pred = knn.predict(X_test)
    
    #Check performance using accuracy
    scores.append({'k': k, 'accuracy': accuracy_score(y_test, y_pred)})

print(scores)
from sklearn.model_selection import GridSearchCV

grid_params = {
    'n_neighbors': [3, 5, 6, 7],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'chebyshev']
}

knn = KNeighborsClassifier(n_neighbors=3)

gs = GridSearchCV(knn, grid_params, cv=3, n_jobs=1)

gs_results = gs.fit(X_train, y_train)

print(gs_results.best_score_)
print(gs_results.best_estimator_)
print(gs_results.best_params_)

