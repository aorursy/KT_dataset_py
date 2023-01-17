# Análise de dados
import numpy as np
import pandas as pd

# Split e validação
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
# Importando os dados
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
# Para saber se há alguma variável nula no dataset
train.isnull()
# Informações gerais
train.info()
# Dando uma olhada nas primeiras linhas do dataset
train.head()
# Saber quantos numeros tem de cada sexo (julgando pelos dados acima, o Sex é uma coluna muito importante no resultado)
train['Sex'].value_counts()
# Vamos transformar as variáveis male e female em binários (como só possuem dois tipos) - transformando de string para int
def transformar_sexo(valor):
    if valor == 'female':
        return 1
    else: 
        return 0
# Transformando cada linha do Sex em binario tanto do treino quanto do teste:

train['Sex_binario'] = train['Sex'].map(transformar_sexo)
test['Sex_binario'] = test['Sex'].map(transformar_sexo)
# Definindo as primeiras variáveis para validação
variaveis = ['Sex_binario', 'Age', 'Pclass', 'SibSp', 'Parch', 'Fare']
# Definindo o input e o target
X = train[variaveis].fillna(-1)
y = train['Survived']
# Split para o treino e validação
np.random.seed(42)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5)
# Validação cruzada usando o K-Fold
results1 = []
kf = RepeatedKFold(n_splits=2, n_repeats=10, random_state=42)

for train_lines, valid_lines in kf.split(X):
    X_train, X_valid = X.iloc[train_lines], X.iloc[valid_lines]
    y_train, y_valid = y.iloc[train_lines], y.iloc[valid_lines]   

    # Encaixando no modelo
    logistic_model = LogisticRegression(solver='liblinear', n_jobs=-1, random_state=42)
    r1 = logistic_model.fit(X_train, y_train).predict(X_valid)
    
    # Comparando a acurácia do resultado com o y_valid
    acuracy = np.mean(y_valid == r1)
    results1.append(acuracy)
    print("Acuracy:", acuracy)
    print()
# Média do resultado do modelo cruzado
np.mean(results1)
# Criando dataframe com uma cópia da validação cruzada do KFold
X_valid_check = train.iloc[valid_lines].copy()

# Substituindo a coluna Result pelo resultado do modelo
X_valid_check['Result'] = r1
X_valid_check.head()
# Criando dataframe onde o modelo não acertou na previsão (Survived  != Result)
df_eror = X_valid_check[X_valid_check['Survived'] != X_valid_check['Result']]
df_eror = df_eror[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Sex_binario', 'Result', 'Survived']]
df_eror.head()
# Separando o dataframe por sexo
women = df_eror[df_eror['Sex'] == 'female']
men = df_eror[df_eror['Sex'] == 'male']
# Enumerando mulheres por sobrevivência
women.sort_values('Survived')
# Enumenando homens por sobrevivência
men.sort_values('Survived')
# Novas colunas no dataframe de treino de acordo com a análise:

# Possíveis locais de embarque
train['Embarked_S'] = (train['Embarked'] == 'S').astype(int)
train['Embarked_C'] = (train['Embarked'] == 'C').astype(int)
#train['Embarked_Q'] = (train['Embarked'] == 'Q').astype(int)

# Cabines nulas
train['Cabine_nula'] = train['Cabin'].isnull().astype(int)  

# Mulheres solteiras e casadas
train['Nome_contem_Miss'] = train['Name'].str.contains("Miss").astype(int)
train['Nome_contem_Mrs'] = train['Name'].str.contains("Mrs").astype(int)

# Patentes dos homens
train['Nome_contem_Master'] = train['Name'].str.contains("Master").astype(int)
train['Nome_contem_Mr'] = train['Name'].str.contains("Mr").astype(int)
train['Nome_contem_Col'] = train['Name'].str.contains("Col").astype(int)
train['Nome_contem_Major'] = train['Name'].str.contains("Major").astype(int)
# Incluindo as novas colunas na variável de entrada
variaveis = ['PassengerId', 'Pclass', 'Age', 'SibSp',
       'Parch', 'Fare', 'Sex_binario',
       'Embarked_S', 'Embarked_C', 'Cabine_nula', 'Nome_contem_Miss',
       'Nome_contem_Mrs', 'Nome_contem_Master', 'Nome_contem_Mr',
       'Nome_contem_Col', 'Nome_contem_Major']

X = train[variaveis].fillna(-1)
y = train['Survived']
# Split para treino e validação (novamente)
np.random.seed(42)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.5)
# Validação cruzada com a nova variável de entrada
results2 = []
kf = RepeatedKFold(n_splits=2, n_repeats=10, random_state=42)

for train_lines, valid_lines in kf.split(X):
    X_train, X_valid = X.iloc[train_lines], X.iloc[valid_lines]
    y_train, y_valid = y.iloc[train_lines], y.iloc[valid_lines]   

    # Encaixando no modelo
    model = RandomForestClassifier(n_jobs=-1, random_state=42)
    r2 = model.fit(X_train, y_train).predict(X_valid)
    
    # Comparando a acurácia do resultado com o y_valid (tirando a prova real)
    acuracy = np.mean(y_valid == r2)
    results2.append(acuracy)
    print("Acuracy:", acuracy)
    print()
np.mean(results2)
# Novas colunas no dataframe de teste:

# Possíveis locais de embarque
test['Embarked_S'] = (test['Embarked'] == 'S').astype(int)
test['Embarked_C'] = (test['Embarked'] == 'C').astype(int)
#train['Embarked_Q'] = (train['Embarked'] == 'Q').astype(int)

# Cabines nulas
test['Cabine_nula'] = test['Cabin'].isnull().astype(int)  

# Mulheres solteiras e casadas
test['Nome_contem_Miss'] = test['Name'].str.contains("Miss").astype(int)
test['Nome_contem_Mrs'] = test['Name'].str.contains("Mrs").astype(int)

# Patentes dos homens
test['Nome_contem_Master'] = test['Name'].str.contains("Master").astype(int)
test['Nome_contem_Mr'] = test['Name'].str.contains("Mr").astype(int)
test['Nome_contem_Col'] = test['Name'].str.contains("Col").astype(int)
test['Nome_contem_Major'] = test['Name'].str.contains("Major").astype(int)
features = test[variaveis].fillna(-1)
model = RandomForestClassifier(n_jobs=-1, random_state=42)
r3 = model.fit(X, y).predict(features)
sub = pd.Series(r3, index=test['PassengerId'], name='Survived')
sub.shape
sub.to_csv('modelo_final2.csv', header=True)
!head -n10 modelo_final2.csv