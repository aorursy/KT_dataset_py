import numpy as np # importa a biblioteca para fazer operações em matrizes
import pandas as pd # importa a biblioteca para facilitar a manipulação e análise dos dados
import sklearn # importa a biblioteca com ferramentas de machine learning

# importa o arquivo e guarda em um dataframe do Pandas
df_dataset = pd.read_csv( '../input/iris2csv/iris.csv', sep=',', index_col=None) 
display(df_dataset.head(5))
import seaborn as sns
import matplotlib.pyplot as plt

# matriz de gráficos scatter 
sns.pairplot(df_dataset, hue='classe', size=3.5);

# mostra o gráfico usando a função show() da matplotlib
plt.show()
## INSIRA SEU CÓDIGO AQUI

df_setosa = df_dataset[df_dataset["comprimento_petala"] <= 2]
display(df_setosa["classe"].unique())
display(df_setosa.shape)
## INSIRA SEU CÓDIGO AQUI
df_virginica = df_dataset[df_dataset["largura_petala"] >= 1.5]
display(df_virginica["classe"].unique())
display(df_virginica.shape)
import seaborn as sns
import matplotlib
# cria um gráfico de barras com a frequência de cada classe
sns.countplot(x="classe", data=df_virginica)
#scatter plot
sns.lmplot(x='comprimento_sepala', y='largura_petala', data=df_virginica, 
           fit_reg=False, # No regression line
           hue='classe')   # Color by evolution stage
from sklearn.model_selection import train_test_split

X = df_dataset.iloc[:,:-1].values # para não selecionar a última coluna, que é a classe
y = df_dataset["classe"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state = 10) # 70% treino e 30% teste

display("X_train shape:")
display(X_train.shape)

display("X_test shape:")
display(X_test.shape)

display("y_train shape:")
display(y_train.shape)

display("y_test shape:")
display(y_test.shape)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
display(y_pred)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix

classes = df_dataset["classe"].unique()
display(classes)
display("true x pred")
display(confusion_matrix(y_test, y_pred, labels=classes))
# importa o arquivo e guarda em um dataframe do Pandas
df_dataset = pd.read_csv( '../input/iris2csv/iris.csv', sep=',', index_col=None) 
from sklearn.model_selection import train_test_split

X = df_dataset.iloc[:,:-1].values # para não selecionar a última coluna, que é a classe
y = df_dataset["classe"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state = 10) # 70% treino e 30% teste

display("X_train shape:")
display(X_train.shape)

display("X_test shape:")
display(X_test.shape)

display("y_train shape:")
display(y_train.shape)

display("y_test shape:")
display(y_test.shape)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
display(y_pred)
from sklearn.model_selection import train_test_split

X = df_dataset.iloc[:,:-1].values # para não selecionar a última coluna, que é a classe
y = df_dataset["classe"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state = 10) # 70% treino e 30% teste

display("X_train shape:")
display(X_train.shape)

display("X_test shape:")
display(X_test.shape)

display("y_train shape:")
display(y_train.shape)

display("y_test shape:")
display(y_test.shape)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
display(y_pred)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.model_selection import train_test_split

X = df_dataset.iloc[:,:-1].values # para não selecionar a última coluna, que é a classe
y = df_dataset["classe"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state = 10) # 70% treino e 30% teste

display("X_train shape:")
display(X_train.shape)

display("X_test shape:")
display(X_test.shape)

display("y_train shape:")
display(y_train.shape)

display("y_test shape:")
display(y_test.shape)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
display(y_pred)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# importa o arquivo e guarda em um dataframe do Pandas
df_dataset2 = pd.read_csv( '../input/data2csv/data2.csv', sep=',', index_col=None) 
## COMECE AQUI
import seaborn as sns
import matplotlib.pyplot as plt

# matriz de gráficos scatter 
sns.pairplot(df_dataset2, hue='classe', size=3.5);

# mostra o gráfico usando a função show() da matplotlib
plt.show()
from sklearn.model_selection import train_test_split

X = df_dataset2.iloc[:,:-1].values # para não selecionar a última coluna, que é a classe
y = df_dataset2["classe"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state = 10) # 70% treino e 30% teste

display("X_train shape:")
display(X_train.shape)

display("X_test shape:")
display(X_test.shape)

display("y_train shape:")
display(y_train.shape)

display("y_test shape:")
display(y_test.shape)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
display(y_pred)

from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))