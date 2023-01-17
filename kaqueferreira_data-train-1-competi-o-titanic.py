import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
# Exibir gráficos dentro do Jupyter Notebook
%matplotlib inline

# Definir tamanho padrão para os gráficos
rcParams['figure.figsize'] = 17, 4
# Realizei processamento de dados com o R adicionando novas colunas de pronome de tratamento 
dataset_original = pd.read_csv('../input/trainn/train.csv')

# Eliminar o identificador dos passageiros
dataset = dataset_original.drop(['PassengerId'], axis=1)
dataset = dataset.drop(['Unnamed: 0'], axis=1)
dataset.head()
dataset.dtypes
# [0] = Quantidade de instâncias
# [1] = Quantidade de atributos
print("O dataset contém ", dataset.shape[0], "instâncias e ", dataset.shape[1], " atributos.")
# .sample() Mostra uma amostra aleatória
# .head() Mostra as primeiras instâncias
# .tail() Mostra as últimas instâncias
dataset.sample(5)
# Somente atributos numéricos são considerados
dataset.describe()
# Quantidade absoluta
totalNulos = (dataset.isnull()).sum()
totalNulos
# Percentual
percentualNulos = (totalNulos / len(dataset)) * 100
percentualNulos
# Apenas atributos int e float
numericos = (dataset.select_dtypes(include=['int64', 'float64'])).columns
# Apenas atributos object (string)
categoricos = (dataset.select_dtypes(include=['object'])).columns

# Não considerar os atributos textuais Name, Ticket e Cabin entre os atributos categóricos 
categoricos = categoricos.drop(['Name', 'Ticket', 'Cabin'])
fig, ax = plt.subplots(ncols=len(numericos), nrows=1)
plt.suptitle("Boxplots dos Atributos Numéricos")

# Gráfico para cada atributo numérico
for i in range(0, len(numericos)):
    feature = numericos[i]
    sns.boxplot(dataset[feature], ax=ax[i], orient='vertical')
fig, ax = plt.subplots(ncols=len(numericos), nrows=1)
plt.suptitle("Histogramas dos Atributos Numéricos")

# Histograma para cada atributo numérico
for i in range(0, len(numericos)):
    feature = numericos[i]
    ax[i].set_title(feature)
    dataset[feature].plot(kind='hist', ax=ax[i])
fig, ax = plt.subplots(ncols=len(categoricos), nrows=1)
plt.suptitle("Gráficos de Barra dos Atributos Categóricos")

# Gráfico para cada atributo categórico
for i in range(0, len(categoricos)):
    feature = categoricos[i]
    ax[i].set_title(feature)
    dataset[feature].value_counts().plot(kind='bar', ax=ax[i])
# Somente atributos numéricos são considerados
plt.suptitle("Gráfico de Calor das Correlações entre os Atributos Numéricos")
sns.heatmap(dataset.corr(), annot=True, cmap='Blues')
dataset_original.shape
# removendo features que não serão utilizadas no treinamento
dataset = dataset_original.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Unnamed: 0'], axis=1)
dataset.head()
# divisão do dataset em treino e teste
train, test_split = train_test_split(dataset.copy(), test_size=0.05)
train.isnull().sum()
# Para as instâncias onde Age é nulo, imputar a média (29.298)
#train.loc[train['Age'].isnull(), 'Age'] = train.mean()['Age']
#train.mean()['Age']
train.isnull().sum()
train.loc[train['Embarked'].isnull(), 'Embarked'] = (train.mode())['Embarked'][0]
(train.mode())['Embarked'][0]
train.isnull().sum()
train = pd.get_dummies(train)
train.head()
#train.drop(['Sex_female','Embarked_Q'], axis=1, inplace=True)
#train.head()
# separando o target do treinamento
X = train.drop(['Survived'], axis=1)
y = train['Survived']
# treinamento do modelo com Random Forest
model = RandomForestClassifier(criterion='entropy')

# treinamento do modelo com Rede Neural
#model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4, 2), random_state=1)

model.fit(X, y)
#MLPClassifier(alpha=1e-05, hidden_layer_sizes=(4, 2), random_state=1,solver='lbfgs')
y_pred = model.predict(X)
# acurácia
accuracy_score(y, y_pred)
print(classification_report(y, y_pred))
# Imputar valores nulos usando padrões do conjunto de treino
#test_split.loc[test_split['Age'].isnull(), 'Age'] = 29.84
test_split.loc[test_split['Embarked'].isnull(), 'Embarked'] = 'S'

# Transfrmando dummy
test_split = pd.get_dummies(test_split)
test_split.head()
test_split.isnull().sum()
X_test = test_split.drop(['Survived'], axis=1)
y_test = test_split['Survived']
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
# gerando a matriz confusão
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.title('Matriz de Confusão')
plt.ylabel('True label')
plt.xlabel('Predicted label')
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
fp, tp, thresholds = roc_curve(y_test, y_proba[:, 1])
# Gerando a Curva ROC
plt.plot(fp, tp)

plt.plot([0, 1], [0, 1], '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.xlabel('Falso Positivo')
plt.ylabel('Verdadeiro Positivo')
plt.title('Curva ROC')
# Área sob a curva ROC
auc(fp, tp)
validation = pd.read_csv('../input/teste3/test3.csv')
validation.head()
# Descartar colunas textuais e identificador
identificador = validation[['PassengerId']]
validation.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'Unnamed: 0'], axis=1, inplace=True)

# Imputar valores nulos usando padrões do conjunto de treino
#validation.loc[validation['Age'].isnull(), 'Age'] = 29.84
validation.loc[validation['Embarked'].isnull(), 'Embarked'] = 'S'

# Transfrmando dummy
validation = pd.get_dummies(validation)
validation.head()
y_pred = model.predict_proba(validation)
y_pred = y_pred[:, 1]
resultado = pd.concat([identificador, pd.DataFrame(y_pred, columns=['Survived'])], axis=1)
resultado.head()
# gerando arquivos para submissão na competição
resultado.to_csv('submission.csv', index=False)