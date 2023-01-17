import warnings
warnings.filterwarnings('ignore')

# pandas é uma biblioteca de software criada para a linguagem Python para manipulação e análise de dados 
# Em particular, oferece estruturas e operações para manipular tabelas numéricas e séries temporais
import pandas as pd 

# NumPy é um pacote para a linguagem Python que suporta arrays e matrizes multidimensionais, possuindo uma larga coleção de funções matemáticas para trabalhar com estas estruturas
import numpy as np

# Matplotlib é uma biblioteca de software para criação de gráficos e visualizações de dados em geral, feita para a da linguagem de programação Python e sua extensão de matemática NumPy
import matplotlib.pyplot as plt

# Seaborn é uma biblioteca de visualização de dados Python baseada no matplotlib. Ele fornece uma interface de alto nível para desenhar gráficos estatísticos atraentes e informativos
import seaborn as sns
%matplotlib inline

# Lê os dados de treino e de teste
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.info()
# Mostra as estatísticas que resumem a tendência central, dispersão e forma da distribuição de um conjunto de dados, excluindo os valores de NaN.
train.describe()
# Mostra a taxa de sobrevivência geral (38,38)
print('Taxa geral de sobrevivência:',train['Survived'].mean())
# função get_dummies (pandas) / Converte variável categórica em variáveis dummy / indicador
def dummies(col,train,test):
    train_dum = pd.get_dummies(train[col])
    test_dum = pd.get_dummies(test[col])
    train = pd.concat([train, train_dum], axis=1)
    test = pd.concat([test,test_dum],axis=1)
    train.drop(col,axis=1,inplace=True)
    test.drop(col,axis=1,inplace=True)
    return train, test

# se livra de colunas não utilizadas
dropping = ['PassengerId', 'Name', 'Ticket']
train.drop(dropping,axis=1, inplace=True)
test.drop(dropping,axis=1, inplace=True)
# Classes da embarcação (1, 2 e 3 classe)
# dropna (Remova os valores ausentes)
print(train.Pclass.value_counts(dropna=False))
sns.factorplot('Pclass', 'Survived',data=train, order=[1,2,3])

# De acordo com o gráfico, há uma grande diferença entre cada grupo percenteces as classes mais altas.
train, test = dummies('Pclass', train, test)
# Sexo
print(train.Sex.value_counts(dropna=False))
sns.factorplot('Sex','Survived', data=train)

# Taxa de sobrevivência para mulheres é muito melhor do que para homens.
train, test = dummies('Sex', train, test)

# Por conta da taxa de sobrevivência para homens ser menor, a coluna masculina será removida.
train.drop('male',axis=1,inplace=True)
test.drop('male',axis=1,inplace=True)
# Idade
# Lidando com dados ausentes
nan_num = train['Age'].isnull().sum()

# Há 177 dados ausentes, então serão preenchidos com valores inteiros randomicos
age_mean = train['Age'].mean()
age_std = train['Age'].std()
filling = np.random.randint(age_mean-age_std, age_mean+age_std, size=nan_num)
train['Age'][train['Age'].isnull()==True] = filling
nan_num = train['Age'].isnull().sum()

# Lidando com valores ausentes no teste
nan_num = test['Age'].isnull().sum()

# 86 são null
age_mean = test['Age'].mean()
age_std = test['Age'].std()
filling = np.random.randint(age_mean-age_std,age_mean+age_std,size=nan_num)
test['Age'][test['Age'].isnull()==True]=filling
nan_num = test['Age'].isnull().sum()

# Análise da coluna idade
s = sns.FacetGrid(train,hue='Survived',aspect=3)
s.map(sns.kdeplot,'Age',shade=True)
s.set(xlim=(0,train['Age'].max()))
s.add_legend()

# A partir do gráfico, podemos ver que a taxa de sobreviência para crianças é maior do que as outras e para 15-30 a taxa é menor
def under15(row):
    result = 0.0
    if row<15:
        result = 1.0
    return result
def young(row):
    result = 0.0
    if row>=15 and row<30:
        result = 1.0
    return result

train['under15'] = train['Age'].apply(under15)
test['under15'] = test['Age'].apply(under15)
train['young'] = train['Age'].apply(young)
test['young'] = test['Age'].apply(young)

train.drop('Age',axis=1,inplace=True)
test.drop('Age',axis=1,inplace=True)
# Família
# Validando dados
print(train['SibSp'].value_counts(dropna=False))
print(train['Parch'].value_counts(dropna=False))

sns.factorplot('SibSp','Survived',data=train,size=5)
sns.factorplot('Parch','Survived',data=train,size=5)

'''through the plot, we suggest that with more family member, 
the survival rate will drop, we can create the new col
add up the parch and sibsp to check our theory''' 
train['family'] = train['SibSp'] + train['Parch']
test['family'] = test['SibSp'] + test['Parch']
sns.factorplot('family','Survived',data=train,size=5)

train.drop(['SibSp','Parch'],axis=1,inplace=True)
test.drop(['SibSp','Parch'],axis=1,inplace=True)
# fare
# Checkando se há null, encontrado um no grupo de teste. Deixa-lo para trás enquanto encontramos sua utilidade.
train.Fare.isnull().sum()
test.Fare.isnull().sum()

sns.factorplot('Survived','Fare',data=train,size=5)
#according to the plot, smaller fare has higher survival rate, keep it
# Lidando com valores null no teste
test['Fare'].fillna(test['Fare'].median(),inplace=True)
# Cabine
# Validando dados ausentes
# 687 de 891 está ausentes, não considerar essa coluna
train.Cabin.isnull().sum()
train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)
# Embarque
train.Embarked.isnull().sum()

# 2 valores ausentes
train.Embarked.value_counts()

# Preenchendo o valor principal 's', na coluna de valores ausentes
train['Embarked'].fillna('S',inplace=True)

sns.factorplot('Embarked','Survived',data=train,size=6)

# C contém maior taxa de sobrevivência, então desconsiderar as outras duas
train,test = dummies('Embarked',train,test)
train.drop(['S','Q'],axis=1,inplace=True)
test.drop(['S','Q'],axis=1,inplace=True)
# import das bibliotecas de Machine Learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold

def modeling(clf,ft,target):
    acc = cross_val_score(clf,ft,target,cv=kf)
    acc_lst.append(acc.mean())
    return 

accuracy = []
def ml(ft,target,time):
    accuracy.append(acc_lst)

    # Regressão Logística
    logreg = LogisticRegression()
    modeling(logreg,ft,target)
    
    #RandomForest
    rf = RandomForestClassifier(n_estimators=50,min_samples_split=4,min_samples_leaf=2)
    modeling(rf,ft,target)
    
    #svc
    svc = SVC()
    modeling(svc,ft,target)
    
    #knn
    knn = KNeighborsClassifier(n_neighbors = 3)
    modeling(knn,ft,target)
    
    
    # Observando o coeficiente
    logreg.fit(ft,target)
    feature = pd.DataFrame(ft.columns)
    feature.columns = ['Features']
    feature["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
    print(feature)
    return 
# Teste n.1, usando todas as Features
train_ft=train.drop('Survived',axis=1)
train_y=train['Survived']

#set kf (KFold: Fornece índices de treinamento / teste para dividir dados em conjuntos de treinamento / teste)
kf = KFold(n_splits=3,random_state=1)
acc_lst = []
ml(train_ft,train_y,'test_1')
# Teste n.2, desconsiderar Young
train_ft_2=train.drop(['Survived','young'],axis=1)
test_2 = test.drop('young',axis=1)
train_ft.head()

# ml
kf = KFold(n_splits=3,random_state=1)
acc_lst=[]
ml(train_ft_2,train_y,'test_2')
# Teste n.3, desconsiderar Young e C
train_ft_3=train.drop(['Survived','young','C'],axis=1)
test_3 = test.drop(['young','C'],axis=1)
train_ft.head()

# ml
kf = KFold(n_splits=3,random_state=1)
acc_lst = []
ml(train_ft_3,train_y,'test_3')
# Teste n.4, sem Fare?
train_ft_4=train.drop(['Survived','Fare'],axis=1)
test_4 = test.drop(['Fare'],axis=1)
train_ft.head()

# ml
kf = KFold(n_splits=3,random_state=1)
acc_lst = []
ml(train_ft_4,train_y,'test_4')
# Teste n.5, perda de C
train_ft_5=train.drop(['Survived','C'],axis=1)
test_5 = test.drop('C',axis=1)

# ml
kf = KFold(n_splits=3,random_state=1)
acc_lst = []
ml(train_ft_5,train_y,'test_5')
# Teste n.6, perda de Fare e young
train_ft_6=train.drop(['Survived','Fare','young'],axis=1)
test_6 = test.drop(['Fare','young'],axis=1)
train_ft.head()

# ml
kf = KFold(n_splits=3,random_state=1)
acc_lst = []
ml(train_ft_6,train_y,'test_6')
accuracy_df=pd.DataFrame(data=accuracy,
                         index=['test1','test2','test3','test4','test5','test6'],
                         columns=['logistic','rf','svc','knn'])
accuracy_df
'''
De acordo com o gráfico, Feactures do teste 4 com svc possuem melhor performance
'''  
# Teste n.4 svc como submission
svc = SVC()
svc.fit(train_ft_4,train_y)
svc_pred = svc.predict(test_4)
print(svc.score(train_ft_4,train_y))


test = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": svc_pred
    })
#submission.to_csv("kaggle.csv", index=False)