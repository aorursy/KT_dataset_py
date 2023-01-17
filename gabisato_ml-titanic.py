import pandas



train_df = pandas.read_csv("../input/train.csv")

test_df = pandas.read_csv("../input/test.csv")

train_df.head()
import seaborn as sms

import matplotlib.pyplot as plt



g = sms.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'PassengerId', bins=20)
# Cria gráficos com os valores de Survived relacionados com Age por exemplo

# Bins se refere a espessura das colunas

g = sms.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
g = sms.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Pclass', bins=20)
g = sms.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'SibSp', bins=20)
g = sms.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Parch', bins=10)
g = sms.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Fare', bins=10)
# Descreve os dados numéricos(quantidade, média...)

train_df.describe()
# Descreve os dados categóricos(quantidade, valores únicos, ...)

train_df.describe(include = ('O'))
# Dados de treinamento

# inplace salva a seleção direto na lista



# Retira as colunas PassengerId, Cabin, Ticket, Name

train_df = train_df.drop('PassengerId', axis = 1)

train_df.drop('Cabin', axis= 1, inplace = True)

train_df.drop(['Ticket', 'Name'], axis=1, inplace=True)
train_df.head()
# Dados de teste

# inplace salva a seleção direto na lista



test_df = test_df.drop('PassengerId', axis = 1)

test_df.drop('Cabin', axis= 1, inplace = True)

test_df.drop(['Ticket', 'Name'], axis=1, inplace=True)
test_df.head()
# Quantos valores nulos

train_df.isnull().sum().sort_values(ascending = False)



# dataframe do pandas

# Tira a media dos valores de Age

train_df.Age.mean()
# Preenche valores nulos em Age com a media

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())
train_df.isnull().sum()
# Número de repetição de cada item dos dados categoricos

# S é o que mais se repete

train_df['Embarked'].value_counts()
# Preenche valores nulos em Embarked com S(Valor que mais se repete)

train_df['Embarked'] = train_df['Embarked'].fillna('S')
train_df.isnull().sum()
test_df.isnull().sum().sort_values(ascending = False)
test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean())
test_df.isnull().sum()
test_df['Fare'] = test_df['Fare'].fillna(test_df['Fare'].mean())
test_df.isnull().sum()
train_df.head()
from sklearn.preprocessing import LabelEncoder 



labelencoder = LabelEncoder()

train_df['Sex'] = labelencoder.fit_transform(train_df['Sex'])

train_df['Embarked'] = labelencoder.fit_transform(train_df['Embarked'])



train_df.head()
# cria a classe pegando a coluna Survived

classe = train_df['Survived']

# seta os atributos copiando a tabela train_df tirando a coluna survived

atributos = train_df.drop('Survived', axis=1)

atributos.head()
classe.head()
# separação dos dados de treinamento em 

# dados de treinamento-atributo treinamento, classe treinamento-(75%)

# e teste/validação-atributo de teste, classe de teste-(25%)

from sklearn.model_selection import train_test_split



atributos_train, atributos_test, classe_train, classe_test = train_test_split(atributos, classe, test_size = 0.25)



atributos_train.describe()
from sklearn.tree import DecisionTreeClassifier

# criando a árvore

dtree = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=3, random_state=0)

# criando o modelo

model = dtree.fit(atributos_train, classe_train)
from sklearn.metrics import accuracy_score

# fazendo a predição dos dados de teste/validação

classe_pred = model.predict(atributos_test)

# a classificação de survived que o modelo conseguiu chegar

classe_pred
# compara a predição dos valores de survived em relação aos dados de classe de teste 

acc = accuracy_score(classe_test, classe_pred)

print("My decision tree accuracy is: ", format(acc))