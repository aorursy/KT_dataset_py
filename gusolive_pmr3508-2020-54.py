import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import sklearn

from IPython.display import display, HTML



%matplotlib inline

plt.style.use('seaborn')
# Salvando o arquivo de teste csv em um DataFrame

df_train = pd.read_csv("../input/adult-pmr3508/train_data.csv", na_values = '?')
# Exibindo o formato do df

df_train.shape
# Usando o comando .info() para obter algumas informações básicas do df

df_train.info()
# Vendo quais são os tipos das colunasÇ

df_train.dtypes
# Usando a função describe para ter uma noção inicial dos valores numéricos e como estão distribuídos na base.

df_train.describe(include='int64')
# Usando a função describe para ter uma noção inicial dos categóricos.

df_train.describe(include='object')
# Podemos verificar dados faltantes

missing = df_train.isnull().sum()

count = df_train.isnull().count()



df_missing = pd.concat([missing,missing/count],axis=1,keys=['#','%'])

df_missing.style.format({'#':'{:.2f}','%':'{:.2%}'})
mode = df_train['workclass'].mode().item()

df_train['workclass'] = df_train['workclass'].fillna(mode)



mode = df_train['occupation'].mode().item()

df_train['occupation'] = df_train['occupation'].fillna(mode)





mode = df_train['native.country'].mode().item()

df_train['native.country'] = df_train['native.country'].fillna(mode)

# Podemos verificar dados faltantes

missing = df_train.isnull().sum()

count = df_train.isnull().count()



df_missing = pd.concat([missing,missing/count],axis=1,keys=['#','%'])

df_missing.style.format({'#':'{:.2f}','%':'{:.2%}'})
# Visualizando as 5 primeiras linhas do Dataframe:

df_train.head()
# Vendo quais são os valores das categoricas.

for column in df_train.select_dtypes('object').columns: 

    

    print(column+':','\n',pd.unique(df_train[column].values).tolist(),'\n')

# # Usando pairplot para uma visão geral do cruzamento entre as variáveis numéricas

# cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

# sns.set()

# sns.pairplot(df_train, vars = cols, hue='income')
less_mean = df_train[df_train['income']=='>50K'].age.mean()

more_mean = df_train[df_train['income']=='<=50K'].age.mean()

statement = 'average age of people with income >50K is {}, while for people with income <=50K it is {}'.format(less_mean,more_mean)

print(statement)
df_train[['capital.gain','capital.loss']].hist(color = 'coral',grid=False,sharey=True)
zeros_gain = df_train[df_train['capital.gain']==0]['capital.gain'].count()

zeros_loss = df_train[df_train['capital.loss']==0]['capital.loss'].count()

statement = 'there are {} zeros in capital.gain and {} zeros in capital.loss'.format(zeros_gain,zeros_loss)

print(statement)
sns.catplot(x="income", y="capital.gain", data=df_train)
np.unique(df_train['capital.gain'].sort_values())[-5:]
df_train[df_train['income']=='<=50K']['hours.per.week'].hist(color = 'coral',bins=16,grid=False)
sns.jointplot(data=df_train, x="education.num", y="hours.per.week")
df_train['education.num'].hist(color = 'coral',bins=16,grid=False)
df_train['education.num'].value_counts()
df_train['hours.per.week'].hist(color = 'coral',bins=16,grid=False)
from sklearn.preprocessing import LabelEncoder



# Iniciando o LabelEncoder

le = LabelEncoder()







# Transformando a variável de classe em 0s e 1s

df_train['income'] = le.fit_transform(df_train['income'])
sns.catplot(x='workclass', y="hours.per.week", kind="violin",hue='income', data=df_train,aspect=3)
df_train[df_train['hours.per.week']<0].shape
sns.catplot(x='income', y='sex', kind='bar', data=df_train,aspect=1)
sns.catplot(x='income', y='native.country', kind='bar', data=df_train,height=8)
sns.catplot(x='income', y='race', kind='bar', data=df_train,height=5)
sns.catplot(x='income', y='workclass', kind='bar', data=df_train,height=5)
sns.catplot(x='income', y='occupation', kind='bar', data=df_train,height=5)
sns.catplot(x='income', y='relationship', kind='bar', data=df_train,height=5)
sns.catplot(x='income', y='marital.status', kind='bar', data=df_train,height=5)
sparse = ['capital.gain', 'capital.loss']

numeric = ['age', 'education.num', 'hours.per.week']

categoric = ['workclass','marital.status', 'occupation', 'relationship', 'race', 'sex']
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')

df_enc = pd.DataFrame(enc.fit_transform(df_train[categoric]).toarray())



df_enc_train = df_train.join(df_enc)

df_enc_train
df_enc_train[numeric]
from sklearn.preprocessing import StandardScaler



# Criando nosso StandardScaler

scaler = StandardScaler()



scaled = scaler.fit_transform(df_enc_train[numeric])



df_enc_train[numeric] = scaled

df_enc_train[numeric]
df_enc_train.columns
Xadult = df_enc_train.drop(df_train.select_dtypes('object'),axis=1)

Xadult = Xadult.drop(['Id','fnlwgt'],axis=1)



Yadult = Xadult.pop('income')
Xadult
%%time



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



knn = KNeighborsClassifier(30)



score = np.mean(cross_val_score(knn, Xadult, Yadult, cv = 10))

score
%%time



best_score = 0.0



for k in range(25, 35):

    knn = KNeighborsClassifier(k)

    score = np.mean(cross_val_score(knn, Xadult, Yadult, cv = 10))

    

    if score > best_score:

        best_k = k

        best_score = score

        best_knn = knn



        

best_knn.fit(Xadult, Yadult)

        

print("the best score is {}, using k = {}".format(best_score, best_k))
df_test = pd.read_csv("../input/adult-pmr3508/test_data.csv", na_values="?")
df_test.isnull().sum()
mode = df_test['workclass'].mode().item()

df_test['workclass'] = df_test['workclass'].fillna(mode)



mode = df_test['occupation'].mode().item()

df_test['occupation'] = df_test['occupation'].fillna(mode)





mode = df_test['native.country'].mode().item()

df_test['native.country'] = df_test['native.country'].fillna(mode)

# Criando nosso OneHotScaler

enc = OneHotEncoder(handle_unknown='ignore')

df_enc = pd.DataFrame(enc.fit_transform(df_test[categoric]).toarray())



df_enc_test = df_test.join(df_enc)

df_enc_test
# Criando nosso StandardScaler

scaler = StandardScaler()



scaled = scaler.fit_transform(df_enc_test[numeric])



df_enc_test[numeric] = scaled

df_enc_test[numeric]
Xadult = df_enc_test.drop(df_test.select_dtypes('object'),axis=1)

Xadult = Xadult.drop(['Id','fnlwgt'],axis=1)
Xadult
output = best_knn.predict(Xadult)
submission = pd.DataFrame()

submission[0] = df_test['Id']

submission[1] = output

submission.columns = ['Id','income']
submission
def correctOutput(x):

    if x == 1:

        return '>50K'

    else:

        return '<=50K'

    
submission['income'] = submission.apply(lambda x: correctOutput(x['income']), axis=1)
submission
submission.to_csv('submission.csv',index = False)