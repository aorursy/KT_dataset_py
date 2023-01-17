import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
df_train = pd.read_csv("../input/adult-pmr3508/train_data.csv", 

                      na_values = '?',

                      sep=r'\s*,\s*',

                      engine='python')

print ('Tamanho do DataFrame: ', df_train.shape)

df_train.head()
df_train.isnull().sum()
print (df_train['workclass'].describe())

print ()

print(df_train['occupation'].describe())

print ()

print(df_train['native.country'].describe())
print ('Porcentagem de aparição dos dados do atributo "workclass" \n')

print ((df_train['workclass'].value_counts()/df_train['workclass'].count())*100)

print ('-------------------------------------------')

print ('Porcentagem de aparição dos dados do atributo "native.country"\n')

print ((df_train['native.country'].value_counts()/df_train['native.country'].count())*100)

print ('-------------------------------------------')

print ('Porcentagem de aparição dos dados do atributo "occupation" \n')

print ((df_train['occupation'].value_counts()/df_train['occupation'].count())*100)
# Substitui os valores NaN pelos valores da moda de cada categoria

rempl = df_train['workclass'].describe().top

df_train['workclass'] = df_train['workclass'].fillna(rempl)



rempl = df_train['native.country'].describe().top

df_train['native.country'] = df_train['native.country'].fillna(rempl)



value = df_train['occupation'].describe().top

df_train['occupation'] = df_train['occupation'].fillna(value)
df_train_Na = df_train.dropna()

df_train_Na
df_train.isnull().sum()
df_train_Na.shape
sns.pairplot(df_train_Na,diag_kws={'bw':"1.0"}, hue="income")
plt.figure(figsize=(13, 7))

plt.hist(df_train_Na['age'], color='darkmagenta',bins = 72)

plt.xlabel('Age')

plt.ylabel('Quantidade')

plt.title('Distribuição de Idades')



sns.catplot(x="income", y="age", kind="boxen", data=df_train_Na,aspect=1.5, height=5)

plt.xlabel('Income')

plt.ylabel('Quantidade')

plt.title('Relação entre Idade e Income')
plt.figure(figsize=(13, 7))

plt.hist(df_train_Na['workclass'], color='darkmagenta', bins = 15)

plt.xlabel('Workclass')

plt.ylabel('Número de ocorrências')

plt.title('Análise do atributo "workclass"')
renda_maior_50k = df_train_Na[df_train_Na['income'] == '>50K']

renda_menor_50k = df_train_Na[df_train_Na['income'] == '<=50K']
plt.figure(figsize=(13, 7))

renda_menor_50k['workclass'].value_counts().plot(kind = 'bar', color = 'green')

plt.ylabel('Quantidade')

plt.title('Contagem de indivíduos com income <=50k')



plt.figure(figsize=(13, 7))

renda_maior_50k['workclass'].value_counts().plot(kind = 'bar', color = 'crimson')

plt.ylabel('Quantidade')

plt.title('Contagem de indivíduos com income >50k')
plt.figure(figsize=(13, 7))

plt.hist(df_train_Na['hours.per.week'], color='darkmagenta', bins = 10)

plt.xlabel('Hours per week')

plt.ylabel('Número de ocorrências')

plt.title('Análise do atributo "hours.per.week"')
sns.catplot(y="hours.per.week", x="age", kind="bar", data=df_train_Na, aspect=3, height=5)
plt.figure(figsize=(13, 7))

plt.hist(df_train_Na['education.num'], color='darkmagenta', bins = 10)

plt.xlabel('Education')

plt.ylabel('Quantidade')

plt.title('Distribuição do atributo "educação"')



sns.catplot(x="income", y="education.num", kind="boxen", data=df_train_Na,aspect=1.5, height=5)

plt.xlabel('income')

plt.ylabel('Quantidade')

plt.title('Relação entre as horas trabalhadas e a renda de cada indivíduo')
plt.figure(figsize=(13, 7))

df_train_Na['capital.loss'].hist(color = 'darkmagenta')

plt.xlabel('capital loss')

plt.ylabel('Quantidade')

plt.title('Análise Capital Loss')



plt.figure(figsize=(13, 7))

df_train_Na['capital.gain'].hist(color = 'darkmagenta')

plt.xlabel('capital gain')

plt.ylabel('Quantidade')

plt.title('Análise Capital Gain')
plt.figure(figsize=(13, 7))

df_train_Na["race"].value_counts().plot(kind="bar", color="darkmagenta")

plt.xlabel('Etnias')

plt.ylabel('Quantidade')

plt.title('Análise da distribuição do atributo "etnias"')
plt.figure(figsize=(13, 7))

renda_menor_50k['race'].value_counts().plot(kind = 'bar', color = 'green')

plt.ylabel('Quantidade')

plt.xlabel('Etnias')

plt.title('Contagem de indivíduos com income <=50k')



plt.figure(figsize=(13, 7))

renda_maior_50k['race'].value_counts().plot(kind = 'bar', color = 'crimson')

plt.ylabel('Quantidade')

plt.xlabel('Etnias')

plt.title('Contagem de indivíduos com income >50k')
etnias = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']

for i in range(len(etnias)):

    num = renda_maior_50k[renda_maior_50k['race'] == etnias[i]]

    den = df_train_Na[df_train_Na['race'] == etnias[i]]

    print ('Porcentagem de indivíduos declarados "'+ etnias[i] +'" que recebem mais 50k de renda:')

    print (num['race'].count()/den['race'].count()*100)

    print ()
plt.figure(figsize=(13, 7))

df_train_Na["occupation"].value_counts().plot(kind="bar", color='darkmagenta')

plt.ylabel('Quantidade')

plt.xlabel('Occupation')

plt.title('Contagem dos diferentes tipos de ocupação')
plt.figure(figsize=(13, 7))

renda_menor_50k['occupation'].value_counts().plot(kind = 'bar', color = 'green')

plt.ylabel('Quantidade')

plt.xlabel('Occupation')

plt.title('Contagem de indivíduos com income <=50k')



plt.figure(figsize=(13, 7))

renda_maior_50k['occupation'].value_counts().plot(kind = 'bar', color = 'crimson')

plt.ylabel('Quantidade')

plt.xlabel('Occupation')

plt.title('Contagem de indivíduos com income >50k')
num = renda_maior_50k[renda_maior_50k['occupation'] == 'Exec-managerial']

den = df_train_Na[df_train_Na['occupation'] == 'Exec-managerial']

print ('Porcentagem de indivíduos que trabalham como "'+ 'Exec-managerial' +'" que recebem mais 50k de renda:')

print (num['occupation'].count()/den['occupation'].count()*100)

print ()
plt.figure(figsize=(13, 7))

df_train_Na['sex'].value_counts().plot(kind = 'pie')
gender = ['Male', 'Female']

for i in range(len(gender)):

    num = renda_maior_50k[renda_maior_50k['sex'] == gender[i]]

    den = df_train_Na[df_train_Na['sex'] == gender[i]]

    print ('Porcentagem de indivíduos do sexo "'+ gender[i] +'" que recebem mais 50k de renda:')

    print (num['sex'].count()/den['sex'].count()*100)

    print ()

print('-----------------------------------------------------')



plt.figure(figsize=(10, 7))

renda_menor_50k['sex'].value_counts().plot(kind = 'bar', color = 'green')

plt.ylabel('Quantidade')

plt.xlabel('Gender')

plt.title('Contagem de indivíduos com income <=50k')



plt.figure(figsize=(10, 7))

renda_maior_50k['sex'].value_counts().plot(kind = 'bar', color = 'crimson')

plt.ylabel('Quantidade')

plt.xlabel('Gender')

plt.title('Contagem de indivíduos com income >50k')

plt.figure(figsize=(13, 7))

df_train_Na['native.country'].value_counts().plot(kind = 'bar', color = 'darkmagenta')

plt.xlabel('Países')

plt.ylabel('Quantidade')

plt.title('Distribuição de nacionalidades no extrato da base Adult')
plt.figure(figsize=(13, 7))

df_train_Na['marital.status'].value_counts().plot(kind = 'bar', color = 'darkmagenta')

plt.xlabel('Estado Civil')

plt.ylabel('Quantidade')

plt.title('Distribuição de status civil no extrato da base Adult')
plt.figure(figsize=(13, 7))

renda_menor_50k['marital.status'].value_counts().plot(kind = 'bar', color = 'green')

plt.ylabel('Quantidade')

plt.xlabel('Estado Civil')

plt.title('Contagem de indivíduos com income <=50k')



plt.figure(figsize=(13, 7))

renda_maior_50k['marital.status'].value_counts().plot(kind = 'bar', color = 'crimson')

plt.ylabel('Quantidade')

plt.xlabel('Estado Civil')

plt.title('Contagem de indivíduos com income >50k')
plt.figure(figsize=(15, 7))

sns.boxplot(x="marital.status", y="hours.per.week", data=df_train_Na)

plt.title('Plotagem do tempo trabalhado por cada estado civil')
df_train_clean = df_train_Na.copy()

df_train_clean = df_train_clean.drop(['fnlwgt', 'native.country', 'education', 'Id'], axis=1)
df_train_clean.head()
#Variável com o classificador que queremos estimar

Y_train_adjusted = df_train_clean.pop('income')

#Variável com todos os atributos restantes

X_train = df_train_clean
X_train
Y_train_adjusted
#Colunas numéricas que não precisarão de tratamento

Col_Num = ['age', 'education.num','hours.per.week']

#Colunas numéricas que precisarão de tratamento

Col_Tra = ['capital.gain', 'capital.loss']

#Colunas categóricas

Col_Cat = ['workclass','occupation', 'race', 'sex','marital.status','relationship']
from sklearn.preprocessing import StandardScaler
Col_Num_Method = StandardScaler()
from sklearn.preprocessing import RobustScaler
Col_Tra_Method = RobustScaler()
from sklearn.preprocessing import OneHotEncoder
Col_Cat_Method = OneHotEncoder(drop='if_binary')
from sklearn.compose import ColumnTransformer
Col_Num_Trans = ColumnTransformer(transformers=[('Col_Num', Col_Tra_Method, Col_Num),('Col_tra',Col_Tra_Method, Col_Tra),

                                               ('Col_Cat', Col_Cat_Method, Col_Cat)])
X_train_adjusted = Col_Num_Trans.fit_transform(X_train)

X_train_adjusted
#Variavel auxiliar apenas para o plot do heatmap

aux = X_train

aux['income'] = Y_train_adjusted



heatmap = aux.corr()

heatmap

sns.set()

plt.figure(figsize=(10,8))

sns.heatmap(heatmap, annot=True)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(knn, X_train_adjusted, Y_train_adjusted, cv = 5, scoring="accuracy")

print("Acurácia com cross validation:", accuracy.mean())
k = 1

best_k = k

best_accuracy = 0

k_array = []

accuracy_array = []

while k <=30:

    knn = KNeighborsClassifier(n_neighbors=k)

    accuracy = cross_val_score(knn, X_train_adjusted, Y_train_adjusted, cv = 5)

    

    k_array.append(k)

    accuracy_array.append(accuracy.mean())

    if accuracy.mean() >= best_accuracy:

        best_accuracy = accuracy.mean()

        best_k = k

        print('K = ', k)

        print('Accuracy: ', accuracy.mean())

    k = k + 1

    

print("O melhor k encontrado foi {0} que levou a uma acurácia de {1}".format(best_k,best_accuracy))
plt.figure(figsize=(13, 7))

plt.plot(k_array,accuracy_array,'r--*')

plt.ylabel('Acurácia')

plt.xlabel('K')

plt.title('Testes de diversos valores de K')
k = best_k



KNN = KNeighborsClassifier(n_neighbors=k)

KNN.fit(X_train_adjusted, Y_train_adjusted)
df_test = pd.read_csv("../input/adult-pmr3508/test_data.csv", 

                      na_values = '?',

                      index_col=['Id'])

print ('Tamanho do DataFrame: ', df_test.shape)
df_test
df_test.isnull().sum()
# Substitui os valores NaN pelos valores da moda de cada categoria

rempl = df_test['workclass'].describe().top

df_test['workclass'] = df_test['workclass'].fillna(rempl)



rempl = df_test['native.country'].describe().top

df_test['native.country'] = df_test['native.country'].fillna(rempl)



rempl = df_test['occupation'].describe().top

df_test['occupation'] = df_test['occupation'].fillna(rempl)
df_test_Na = df_test.dropna()

df_test_Na
df_test_Na.isnull().sum()
df_test_clean = df_test_Na.copy()

df_test_clean = df_test_clean.drop(['fnlwgt', 'native.country', 'education'], axis=1)

df_test_clean
X_test_adjusted = Col_Num_Trans.fit_transform(df_test_clean)

X_test_adjusted
predicoes = KNN.predict(X_test_adjusted)
predicoes
submission = pd.DataFrame()
submission[0] = df_test.index

submission[1] = predicoes

submission.columns = ['Id','income']

submission.head()
submission.to_csv('submission.csv',index = False)