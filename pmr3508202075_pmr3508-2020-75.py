import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
adult_train = pd.read_csv("../input/adult-pmr3508/train_data.csv", names=["Id", "age", "workclass", "fnlwgt", "education", "education num", "marital status",
                                               "occupation", "relationship", "race", "sex", "capital gain", "capital loss",
                                               "hours per week", "country", "income"],
                          sep=r'\s*,\s*', engine='python', na_values="?", skiprows=[0], index_col=['Id'])

adult_train
adult_train.info()
adult_train['occupation'].value_counts().plot(kind="bar")
plt.title('Distribution of Occupation')
print('Elementos Faltantes (Total): {}'.format(adult_train['occupation'].isnull().sum()))
print('Elementos Faltantes (Porcentagem): {:.2f}%'.format(100*adult_train['occupation'].isnull().sum()/adult_train['occupation'].size))
adult_train["age"].value_counts().plot(kind="bar")
plt.title('Age Distribution')
print('Elementos Faltantes (Total): {}'.format(adult_train['age'].isnull().sum()))
print('Elementos Faltantes (Porcentagem): {:.2f}%'.format(100*adult_train['age'].isnull().sum()/adult_train['age'].size))
adult_train["education"].value_counts().plot(kind="bar")
plt.title('Distribution of Education')
print('Elementos Faltantes (Total): {}'.format(adult_train['education'].isnull().sum()))
print('Elementos Faltantes (Porcentagem): {:.2f}%'.format(100*adult_train['education'].isnull().sum()/adult_train['education'].size))
adult_train["country"].value_counts().plot(kind="bar")
plt.title('Distribution of Country')
print('Elementos Faltantes (Total): {}'.format(adult_train['country'].isnull().sum()))
print('Elementos Faltantes (Porcentagem): {:.2f}%'.format(100*adult_train['country'].isnull().sum()/adult_train['country'].size))
adult_train_copy = adult_train.copy()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
adult_train_copy['income'] = le.fit_transform(adult_train_copy['income'])
sns.pairplot(adult_train, diag_kws={'bw':"1.0"}, hue = 'income', palette='magma')
# adult_train['income'] = le.fit_transform(adult_train['income'])
plt.figure(figsize=(10, 8))
sns.heatmap(adult_train_copy.corr(), annot=True, vmin=-1, vmax=1, cmap = 'magma')
plt.figure(figsize=(15,5))
sns.countplot(x="income", hue='marital status', data = adult_train, palette = 'magma')
plt.figure(figsize=(15,5))
sns.countplot(x="income", hue="occupation", data=adult_train, palette="magma")
plt.figure(figsize=(15,5))
sns.countplot(x="income", hue="workclass", data=adult_train, palette="magma")
plt.figure(figsize=(15,5))
sns.countplot(x="income", hue="relationship", data=adult_train, palette="magma")
plt.figure(figsize=(15,5))
sns.countplot(x="income", hue = "sex", data = adult_train, palette = "magma")
plt.figure(figsize=(15,5))
sns.countplot(x="income", hue = "race", data = adult_train, palette = "magma")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
# preenche valores NaN com 0"
adult_train_fill = adult_train.fillna(0)
# classificador com variáveis numéricas e sexo
knn = KNeighborsClassifier(n_neighbors=5)

X = adult_train_fill[["age","education num","sex","capital gain","capital loss","hours per week"]]
Y = adult_train_fill["income"]
# transforma-se a variável sex em uma variável numérica
X["sex-num"] = np.where(X.sex=="Male", 1, 0)
X = X.drop(["sex"], axis = 1)

X
accuracy = cross_val_score(knn, X, Y, cv = 5, scoring="accuracy")
print("Acurácia", accuracy.mean())
k = 1
melhor_k = k
best_accuracy = 0
k_array = []
accuracy_array = []
while k <=30:
    accuracy=cross_val_score(knn, X, Y, cv = 5)
    knn=KNeighborsClassifier(n_neighbors=k)
    k_array.append(k)
    accuracy_array.append(accuracy.mean())
    if accuracy.mean() >= best_accuracy:
        best_accuracy = accuracy.mean()
        best_k = k
        print('K=', k)
        print('Accuracy:', accuracy.mean())
    k = k + 1
    
print("O melhor k é o {0} , com acurácia de {1}".format(best_k,best_accuracy))
adult_test = pd.read_csv("../input/adult-pmr3508/test_data.csv", names=["Id", "age", "workclass", "fnlwgt", "education", "education num", "marital status",
                                               "occupation", "relationship", "race", "sex", "capital gain", "capital loss",
                                               "hours per week", "country", "income"],
                          sep=r'\s*,\s*', engine='python', na_values="?", skiprows=[0], index_col=['Id'])

adult_test
k=best_k
KNN = KNeighborsClassifier(n_neighbors=k)
KNN.fit(X, Y)

adult_test_fill = adult_test.fillna(0)
# mesmo processo de pegar variáveis numéricas para os dados de teste
X_test = adult_test_fill[["age","education num","sex","capital gain","capital loss","hours per week"]]
X_test["sex-int"] = np.where(X_test.sex=="Male", 1, 0)
X_test = X_test.drop(["sex"], axis = 1)
predictions = KNN.predict(X_test)
submission = pd.DataFrame()
submission["income"] = predictions
submission.to_csv("submission.csv",index_label = "Id")
