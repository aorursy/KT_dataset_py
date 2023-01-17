from google.colab import drive

drive.mount('/content/drive')
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv("https://raw.githubusercontent.com/WoMakersCode/data-science-bootcamp/master/4.0%20Modelos%20de%20classifica%C3%A7%C3%A3o/data/train.csv", encoding = "ISO-8859-1")



submission = pd.read_csv("https://raw.githubusercontent.com/WoMakersCode/data-science-bootcamp/master/4.0%20Modelos%20de%20classifica%C3%A7%C3%A3o/data/test.csv", encoding = "ISO-8859-1")
train.head ()

submission.head ()
train.shape
submission.shape


train.info()
train.describe()
train.shape
def count_unique(df):

  print("Quantidade de valores únicos para cada feature no conjunto de treinamento")

  for i in df.columns:

    print(f"{i}: {df[i].nunique()}")
count_unique(train)
columns = [ 'PassengerId', 'Name' , 'Ticket' , 'Cabin']
train = train.drop(columns, axis= 1)
train.head()
train.shape
train.Survived.value_counts()
xy = sns.countplot (x = "Survived", data = train)

plt.title("Contagem de sobrevivência")

plt.xlabel('Sobrevivência')

plt.ylabel('Contagem')

plt.show()
print(f"Considerando nosso conjunto de treinamento, {train.Survived.value_counts()[0]/train.shape[0]*100:.2f}% dos passageiros não sobreviveram ao naufrágio :(")
train.Pclass.value_counts()
rr = sns.countplot (data = train , x= 'Survived', hue= 'Pclass' )

plt.title("Contagem de sobrevivência de acordo com a classe")

plt.xlabel("Sobrevivência")

plt.ylabel("Quantidade")

plt.show()



      
train[['Pclass', 'Survived']].groupby( ['Pclass']).mean() * 100



train[['Sex','Survived']].groupby(['Sex']).mean() *100

_ = sns.countplot(data=train, x = 'Survived', hue = 'Sex')

plt.title("Contagem de sobrevivência de acordo com o gênero")

plt.xlabel("Sobrevivência")

plt.ylabel("Quantidade")

plt.show()

# proporção de sobrevivência por gênero

train[["Sex", "Survived"]].groupby(['Sex']).mean()*100
sns.catplot(x="Pclass", y="Survived", col="Sex", data=train,kind="bar");
train['Age'].describe()
ag = sns.boxplot(train ['Age']).set_title("Idade")
survived_age_not_null = train.loc[(train.Survived == 1) & (train.Age.isnull()==False), 'Age']

not_survived_age_not_null = train.loc[(train.Survived == 0) & (train.Age.isnull()==False), 'Age']
agr = sns.distplot(train.loc [(train.Survived == 1) & (train.Age.isnull() ==False), "Age"], hist=True, label= 'Sobreviveu')

agrh = sns.distplot(train.loc[(train.Survived == 0) & (train.Age.isnull() ==False), "Age"], hist=False, label= 'Não sobreviveu') 

_ = plt.title("Distribuições das idades de acordo com o target")
median_age = train['Age'].median()
train.loc[train['Age'].isnull(), 'Age'] = median_age
agr = sns.distplot(train.loc [(train.Survived == 1) & (train.Age.isnull() ==False), "Age"], hist=True, label= 'Sobreviveu')

agrh = sns.distplot(train.loc[(train.Survived == 0) & (train.Age.isnull() ==False), "Age"], hist=False, label= 'Não sobreviveu') 

_ = plt.title("Distribuições das idades de acordo com o target")
from sklearn.preprocessing import LabelEncoder



encoder = LabelEncoder()

train ['Sex'] = encoder.fit_transform(train["Sex"])
train.Sex.value_counts()
# nossas features

x = train[["Age", 'Sex', 'Pclass']]

# nosso target

y = train['Survived']
from sklearn.model_selection import train_test_split



# Separando os dados em treinamento e teste

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
x.head()
y.head()
from sklearn.model_selection import train_test_split



# Separando os dados em treinamento e teste

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=42)
x_train.shape
x_test.shape
from sklearn.tree import DecisionTreeClassifier



#TODO 



# Instanciando o classificador

model = DecisionTreeClassifier(criterion='entropy', random_state=42 )



# Treinamento do modelo

model.fit(x_train, y_train)
# arquivo de texto que armazena a estrutura da nossa árvore de decisão

from sklearn.tree import export_graphviz

export_graphviz(model,out_file='titanic_tree.dot',feature_names=['Age', 'Sex', 'Pclass'],rounded=True,filled=True,class_names=['Não sobreviveu','Sobreviveu'])

!dot -Tpng titanic_tree.dot -o titanic_tree.png
from IPython.core.display import Image, display

display(Image('titanic_tree.png', width=1900, unconfined=True))
y_pred = model.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix

import itertools
# plota a matriz de confusão. Código retirado da documentação do próprio Sklearn

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Matriz de confusão',

                          cmap=plt.cm.Blues):



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    

    #plt.ylim(0.5, 0.5)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    plt.ylim(1.5, -0.5) 



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.ylabel('Classe real')

    plt.xlabel('Classe prevista')

    plt.tight_layout()
cnf_matrix = confusion_matrix(y_test ,y_pred)
cnf_matrix
plot_confusion_matrix(cnf_matrix, classes=['não sobreviveu', 'sobreviveu'])
plot_confusion_matrix( cnf_matrix, normalize=True, classes= ['não sobreviveu', 'sobreviveu'], title='Matriz de confusão normalizada' )
# substituímos os valores faltantes pela mediana da idade do conjunto de treinamento

submission.loc[submission ['Age'].isnull(), 'Age'] = median_age



median_age

# utilizamos o encoder que foi criado com base no conjunto de treinamento

submission['Sex'] = encoder.transform(submission['Sex'])
# realiza a predição para o conjunto de teste

result= model.predict(submission[['Age', 'Sex', 'Pclass']]) 
result
# transformar o array em um DataFrame para concatenarmos como ID

results = pd.DataFrame(list(result), columns= ['Survided'])

submission = pd.concat([submission["PassengerId"], results], axis=1)

submission.head()
submission.to_csv("titanic_submission.csv", index=False)