#carregando arquivo
import pandas as pd
#ignore warnings
import warnings
warnings.filterwarnings('ignore')

#prestem atenção no código abaixo, tem duas formas de carregar os csvs, esolham a que preferir e comentem a outra pra desativar
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


#train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
#train = pd.read_csv(train_url)

#test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
#test = pd.read_csv(test_url)


#Solte os recursos que não usaremos
train = train.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)
test = test.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)

#Veja as 3 primeiras linhas dos nossos dados de treinamento
train.head(3)
#Converter ['male','female'] para [1,0] para que nossa árvore de decisão possa ser construída
for df in [train,test]:
    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})
    
#Preencha os valores de idade ausentes com 0 (presumindo que sejam bebês se não tiverem uma idade listada)
train['Age'] = train['Age'].fillna(0)
test['Age'] = test['Age'].fillna(0)

#Select feature column names and target variable we are going to use for training
features = ['Pclass','Age','Sex_binary']
target = 'Survived'

#Observe as primeiras 3 linhas (temos mais de 800 linhas no total) dos nossos dados de treinamento. 
#Este é o "input" que nosso classificador usará como "input"
train[features].head(3)
train.info()
print('_'*40)
test.info()

#Exibe as primeiras 3 variáveis de destino
train[target].head(3).values
from sklearn.tree import DecisionTreeClassifier

#Criar um objeto classificador com hiperparâmetros padrão
#clf = DecisionTreeClassifier()  
clf = DecisionTreeClassifier(max_depth=3,min_samples_leaf=2)
#Ajuste nosso classificador usando os recursos de treinamento e os valores de meta de treinamento
clf.fit(train[features],train[target]) 
#Create decision tree ".dot" file

#Remove each '#' below to uncomment the two lines and export the file.
#from sklearn.tree import export_graphviz
#export_graphviz(clf,out_file='titanic_tree.dot',feature_names=features,rounded=True,filled=True,class_names=['Survived','Did not Survive'])
#Display decision tree

#Blue on a node or leaf means the tree thinks the person did not survive
#Orange on a node or leaf means that tree thinks that the person did survive

#In Chrome, to zoom in press control +. To zoom out, press control -. If you are on a Mac, use Command.

#Remove each '#' below to run the two lines below.
#from IPython.core.display import Image, display
#display(Image('titanic_tree.png', width=1900, unconfined=True))
#Faça previsões usando os recursos do conjunto de dados de teste
predictions = clf.predict(test[features])

#Exibir nossas previsões - elas são 0 ou 1 para cada instância de treinamento 
#dependendo se nosso algoritmo acredita que a pessoa sobreviveu ou não.
predictions
#Crie um DataFrame com os IDs dos passageiros e nossa previsão sobre se eles sobreviveram ou não
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})

#Visualize as primeiras 5 linhas
submission.head()
#Converter DataFrame em um arquivo csv que pode ser carregado
#Isso é salvo no mesmo diretório do seu notebook
filename = 'Titanic Predictions 2.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)