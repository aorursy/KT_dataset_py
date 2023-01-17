from sklearn.decomposition import PCA
from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target
import matplotlib.pyplot as plt 
plt.gray()
plt.matshow(digits.images[25]) 
plt.show()
print(digits.target[25])
pca = PCA(n_components=0.95, whiten=True)

X_pca = pca.fit_transform(X)

print('Número original de atributos:', X.shape[1])
print('Número reduzido de atributos:', X_pca.shape[1])
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Acurácia nos dados originais:', accuracy_score(y_test, y_pred))

#######

X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.33, random_state=42)

model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Acurácia nos dados reduzidos (PCA em tudo):', accuracy_score(y_test, y_pred))

#######

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

pca = PCA(n_components=0.95, whiten=True)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Acurácia nos dados originais (PCA da parte certa):', accuracy_score(y_test, y_pred))
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import numpy as np

import warnings
warnings.filterwarnings("ignore")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

fvalue_selector = SelectKBest(f_classif, k=20) # Calcula o ANOVA F-value para as amostras
X_kbest = fvalue_selector.fit_transform(X_train, y_train)

print('Número original de atributos:', X.shape[1])
print('Número reduzido de atributos:', X_kbest.shape[1])

###

selected_features = []
map_vector = []
mask = fvalue_selector.get_support()
# print(mask)

# Gera uma imagem no tamanho original indicando as regiões que foram selecionadas pelo kbest
for m, feature in zip(mask, list(range(64))):
    if m:
        selected_features.append(feature)
        map_vector.append(1)
    else:
        map_vector.append(0)

print(selected_features)

map_vector = np.asarray(map_vector)

plt.matshow(map_vector.reshape(8,8)) 
plt.show()

###

model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Acurácia nos dados originais:', accuracy_score(y_test, y_pred))

model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)
model.fit(X_kbest, y_train)
X_test_kbest = fvalue_selector.transform(X_test)
y_pred = model.predict(X_test_kbest)
print('Acurácia nos dados Kbest:', accuracy_score(y_test, y_pred))
import seaborn as sns
v_hist = []
acc = []
for v in range(1,20):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    pca = PCA(n_components=v*0.05, whiten=True)
    X_train = pca.fit_transform(X_train)
    print("#####################################################################")
    print("Variância => ", v*0.05)
    print("Variância (Valor real) => ", pca.explained_variance_ratio_)
    print("Variância (Valor real) => ", np.sum(pca.explained_variance_ratio_))
    print("Quantidade de atributos => ", X_train.shape)
    print("#####################################################################")
    X_test = pca.transform(X_test)

    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    v_hist.append(v*0.05)
    acc.append(accuracy_score(y_test, y_pred))

ax = sns.lineplot(x=np.array(v_hist), y=np.array(acc))
v_hist = []
acc = []
for v in range(1,20):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    pca = PCA(n_components=v, whiten=True)
    X_train = pca.fit_transform(X_train)
    print("#####################################################################")
    print("Variância => ", pca.explained_variance_ratio_)
    print("Variância (Soma) => ", np.sum(pca.explained_variance_ratio_))
    print("Quantidade de atributos => ", v)
    print("#####################################################################")
    X_test = pca.transform(X_test)

    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    v_hist.append(v)
    acc.append(accuracy_score(y_test, y_pred))

ax = sns.lineplot(x=np.array(v_hist), y=np.array(acc))
k_hist = []
acc = []

for k in range(1,30):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    fvalue_selector = SelectKBest(f_classif, k=k)
    X_kbest = fvalue_selector.fit_transform(X_train, y_train)

    model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=2000)
    model.fit(X_kbest, y_train)
    X_test_kbest = fvalue_selector.transform(X_test)
    y_pred = model.predict(X_test_kbest)
    
    k_hist.append(k)
    acc.append(accuracy_score(y_test, y_pred))
    
ax = sns.lineplot(x=np.array(k_hist), y=np.array(acc))