from sklearn.datasets import load_breast_cancer
import numpy as np
bc = load_breast_cancer()

data = bc.data
target = bc.target
print("Formato dos dados (linhas x colunas) => ", data.shape)
print("Labels do problema => ", np.unique(target))
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
# separando os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# predizendo os rótulos com o modelo
y_pred = knn.predict(X_train)

# Calculando a acurácia sem o scikit-learn
correct_values = np.sum(y_pred == y_train)
accuracy = correct_values / y_pred.shape[0]
print("Acurácia => ", accuracy)

# avaliando o modelo com o scikit-learn
print('Acurácia:', accuracy_score(y_train, y_pred))
from sklearn.metrics import confusion_matrix
# predizendo os rótulos a partir do modelo
y_pred = knn.predict(X_train)
vp = 0 # verdadeiros positivos
vn = 0 # verdadeiros negativos
fp = 0 # falsos positivos
fn = 0 # falsos negativos

# Percore as predições, 
for pred, true in zip(y_pred, y_train):
    if pred == 1:
        if pred == true:
            vp += 1
        else:
            fp += 1
    else:
        if pred == true:
            vn += 1
        else:
            fn += 1
            
print('VP:', vp)
print('FP:', fp)
print('VN:', vn)
print('FN:', fn)
# Utilizando o scikit-learn para printar a matriz de confusão
print(confusion_matrix(y_train, y_pred))
vn, fp, fn, vp = confusion_matrix(y_train, y_pred).ravel()
print('VP:', vp)
print('FP:', fp)
print('VN:', vn)
print('FN:', fn)

