# importar pacotes necessários

import numpy as np

import pandas as pd

import seaborn as sns

# definir parâmetros extras

pd.set_option('precision', 3)

pd.set_option('display.max_columns', 100)



# carregar arquivos de dados de treino

data1 = pd.read_csv('../input/zoo-train.csv', index_col='animal_name')

data2 = pd.read_csv('../input/zoo-train2.csv', index_col='animal_name')

data3 = pd.read_csv('../input/zoo-test.csv', index_col='animal_name')



# quantas linhas e colunas existem?

print("data1.shape")

print(data1.shape)

print("data2.shape")

print(data2.shape)

print("data3.shape")

print(data3.shape)

# mostrar alguns exemplos de registros

data1.head(5)


# Preparando os Dados

from sklearn.preprocessing import StandardScaler



# Normalizando as bases

data1['hair'] = np.where(data1['hair'] =='n', 0, 1)

data1['feathers'] = np.where(data1['feathers'] =='n', 0, 1)

data1['eggs'] = np.where(data1['eggs'] =='n', 0, 1)

data1['milk'] = np.where(data1['milk'] =='n', 0, 1)

data1['airborne'] = np.where(data1['airborne'] =='n', 0, 1)

data1['aquatic'] = np.where(data1['aquatic'] =='n', 0, 1)

data1['predator'] = np.where(data1['predator'] =='n', 0, 1)

data1['toothed'] = np.where(data1['toothed'] =='n', 0, 1)

data1['backbone'] = np.where(data1['backbone'] =='n', 0, 1)

data1['breathes'] = np.where(data1['breathes'] =='n', 0, 1)

data1['venomous'] = np.where(data1['venomous'] =='n', 0, 1)

data1['fins'] = np.where(data1['fins'] =='n', 0, 1)

data1['tail'] = np.where(data1['tail'] =='n', 0, 1)

data1['domestic'] = np.where(data1['domestic'] =='n', 0, 1)

data1['catsize'] = np.where(data1['catsize'] =='n', 0, 1)



data2['hair'] = np.where(data2['hair'] =='n', 0, 1)

data2['feathers'] = np.where(data2['feathers'] =='n', 0, 1)

data2['eggs'] = np.where(data2['eggs'] =='n', 0, 1)

data2['milk'] = np.where(data2['milk'] =='n', 0, 1)

data2['airborne'] = np.where(data2['airborne'] =='n', 0, 1)

data2['aquatic'] = np.where(data2['aquatic'] =='n', 0, 1)

data2['predator'] = np.where(data2['predator'] =='n', 0, 1)

data2['toothed'] = np.where(data2['toothed'] =='n', 0, 1)

data2['backbone'] = np.where(data2['backbone'] =='n', 0, 1)

data2['breathes'] = np.where(data2['breathes'] =='n', 0, 1)

data2['venomous'] = np.where(data2['venomous'] =='n', 0, 1)

data2['fins'] = np.where(data2['fins'] =='n', 0, 1)

data2['tail'] = np.where(data2['tail'] =='n', 0, 1)

data2['domestic'] = np.where(data2['domestic'] =='n', 0, 1)

data2['catsize'] = np.where(data2['catsize'] =='n', 0, 1)



data3['hair'] = np.where(data3['hair'] =='n', 0, 1)

data3['feathers'] = np.where(data3['feathers'] =='n', 0, 1)

data3['eggs'] = np.where(data3['eggs'] =='n', 0, 1)

data3['milk'] = np.where(data3['milk'] =='n', 0, 1)

data3['airborne'] = np.where(data3['airborne'] =='n', 0, 1)

data3['aquatic'] = np.where(data3['aquatic'] =='n', 0, 1)

data3['predator'] = np.where(data3['predator'] =='n', 0, 1)

data3['toothed'] = np.where(data3['toothed'] =='n', 0, 1)

data3['backbone'] = np.where(data3['backbone'] =='n', 0, 1)

data3['breathes'] = np.where(data3['breathes'] =='n', 0, 1)

data3['venomous'] = np.where(data3['venomous'] =='n', 0, 1)

data3['fins'] = np.where(data3['fins'] =='n', 0, 1)

data3['tail'] = np.where(data3['tail'] =='n', 0, 1)

data3['domestic'] = np.where(data3['domestic'] =='n', 0, 1)

data3['catsize'] = np.where(data3['catsize'] =='n', 0, 1)



data1.head()
# quais são as colunas e respectivos tipos de dados?

data1.info()
# sumário estatístico das características numéricas

data1.describe()
# quantos registros existem de cada classe?

data1['class_type'].value_counts()
# gerar boxplot para cada uma das características por espécie

data1.boxplot(by="class_type", figsize=(20, 10))
# gerar mapa de calor com a correlação das características

sns.heatmap(data1.corr(), annot=True, cmap='cubehelix_r')
# importar pacotes necessários

import numpy as np





# importar pacotes usados na seleção do modelo e na medição da precisão

from sklearn.model_selection import train_test_split



# importar os pacotes necessários para os algoritmos de classificação

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier





# definir dados de entrada



X = data1.drop(['class_type'], axis=1) # tudo, exceto a coluna alvo

y = data1['class_type'] # apenas a coluna alvo



print('Forma dos dados originais:', X.shape, y.shape)
# separarar dados para fins de treino (70%) e de teste (30%)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



print('Forma dos dados separados:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# A) Support Vector Machine (SVM)



model = SVC()



model.fit(X_train, y_train)

score = model.score(X_test, y_test) * 100



print(model, '\nScore:', score, '%')

# B) Logistic Regression



model = LogisticRegression()



model.fit(X_train, y_train)

score = model.score(X_test, y_test) * 100



print(model, '\nScore:', score, '%')
# C) Decision Tree



model = DecisionTreeClassifier()



model.fit(X_train, y_train)

score = model.score(X_test, y_test) * 100



print(model, '\nScore:', score, '%')
# D) K-Nearest Neighbours



model = KNeighborsClassifier(n_neighbors=3)



model.fit(X_train, y_train)

score = model.score(X_test, y_test) * 100



print(model, '\nScore:', score, '%')
# carregar arquivo de dados de treino

train_data = data1



# carregar arquivo de dados de teste

test_data = data3



test_data.head()

# definir dados de treino

from sklearn import preprocessing



X_train = train_data.drop(['class_type'], axis=1) # tudo, exceto a resposta

y_train = train_data['class_type'] # apenas a coluna alvo

X_train_pro=preprocessing.scale(X_train)



print('Forma dos dados de treino:', X_train.shape, y_train.shape)
# definir dados de teste



X_test = test_data

X_test_pro=preprocessing.scale(X_test)



print('Forma dos dados de teste:', X_test.shape)
# Usando Decision Tree



model = DecisionTreeClassifier()

model.fit(X_train, y_train)



print(model)
# executar previsão usando o modelo escolhido

y_pred = model.predict(X_test)



print('Exemplos de previsões:\n', y_pred[:10])
# gerar dados de envio (submissão)



submission = pd.DataFrame({

  'animal_name': X_test.index,

  'class_type': y_pred

})

submission.set_index('animal_name', inplace=True)



# mostrar dados de exemplo

submission.head(10)

# gerar arquivo CSV para o envio

submission.to_csv('./zoo-submission-dectree.csv')
# importar pacotes necessários

import numpy as np





# importar pacotes usados na seleção do modelo e na medição da precisão

from sklearn.model_selection import train_test_split



# importar os pacotes necessários para os algoritmos de classificação

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier





# definir dados de entrada

T = data1.append(data2, ignore_index=True)

X = T.drop(['class_type'], axis=1) # tudo, exceto a coluna alvo

y = T['class_type'] # apenas a coluna alvo



print('Forma dos dados originais:', X.shape, y.shape)
# separarar dados para fins de treino (70%) e de teste (30%)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



print('Forma dos dados separados:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# A) Support Vector Machine (SVM)



model = SVC()



model.fit(X_train, y_train)

score = model.score(X_test, y_test) * 100



print(model, '\nScore:', score, '%')

# B) Logistic Regression



model = LogisticRegression()



model.fit(X_train, y_train)

score = model.score(X_test, y_test) * 100



print(model, '\nScore:', score, '%')
# C) Decision Tree



model = DecisionTreeClassifier()



model.fit(X_train, y_train)

score = model.score(X_test, y_test) * 100



print(model, '\nScore:', score, '%')
# D) K-Nearest Neighbours



model = KNeighborsClassifier(n_neighbors=3)



model.fit(X_train, y_train)

score = model.score(X_test, y_test) * 100



print(model, '\nScore:', score, '%')
data1.head()
data2.head()
# carregar arquivo de dados de treino

train_data = data1.append(data2, ignore_index=True)
print(train_data.shape)

train_data.head()
# carregar arquivo de dados de teste

test_data = data3



test_data.head()

# definir dados de treino

from sklearn import preprocessing



X_train = train_data.drop(['class_type'], axis=1) # tudo, exceto a resposta

y_train = train_data['class_type'] # apenas a coluna alvo





print('Forma dos dados de treino:', X_train.shape, y_train.shape)
# definir dados de teste



X_test = test_data



print('Forma dos dados de teste:', X_test.shape)
# Usando Decision Tree



model = DecisionTreeClassifier()

model.fit(X_train, y_train)



print(model)
# executar previsão usando o modelo escolhido

y_pred = model.predict(X_test)



print('Exemplos de previsões:\n', y_pred[:10])
# gerar dados de envio (submissão)



submission = pd.DataFrame({

  'animal_name': X_test.index,

  'class_type': y_pred

})

submission.set_index('animal_name', inplace=True)



# mostrar dados de exemplo

submission.head(10)
# gerar arquivo CSV para o envio

submission.to_csv('./zoo-submission-dectree-full.csv')
#### Usando Vizinhos mais Próximos...
# Usando K-Nearest Neighbours



model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)



print(model)
# executar previsão usando o modelo escolhido

y_pred = model.predict(X_test)



print('Exemplos de previsões:\n', y_pred[:10])
# gerar dados de envio (submissão)



submission = pd.DataFrame({

  'animal_name': X_test.index,

  'class_type': y_pred

})

submission.set_index('animal_name', inplace=True)



# mostrar dados de exemplo

submission.head(10)
# gerar arquivo CSV para o envio

submission.to_csv('./zoo-submission-KNN-full.csv')
# importar pacotes necessários

import numpy as np





# importar pacotes usados na seleção do modelo e na medição da precisão

from sklearn.model_selection import train_test_split



# importar os pacotes necessários para os algoritmos de classificação

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier





# definir dados de entrada

T = data2

X = T.drop(['class_type'], axis=1) # tudo, exceto a coluna alvo

y = T['class_type'] # apenas a coluna alvo



print('Forma dos dados originais:', X.shape, y.shape)
# separarar dados para fins de treino (70%) e de teste (30%)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



print('Forma dos dados separados:', X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# A) Support Vector Machine (SVM)



model = SVC()



model.fit(X_train, y_train)

score = model.score(X_test, y_test) * 100



print(model, '\nScore:', score, '%')

# B) Logistic Regression



model = LogisticRegression()



model.fit(X_train, y_train)

score = model.score(X_test, y_test) * 100



print(model, '\nScore:', score, '%')
# C) Decision Tree



model = DecisionTreeClassifier()



model.fit(X_train, y_train)

score = model.score(X_test, y_test) * 100



print(model, '\nScore:', score, '%')
# D) K-Nearest Neighbours



model = KNeighborsClassifier(n_neighbors=3)



model.fit(X_train, y_train)

score = model.score(X_test, y_test) * 100



print(model, '\nScore:', score, '%')