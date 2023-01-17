import numpy as np 

import pandas as pd 

import seaborn as sn

import sklearn

import matplotlib.pyplot as plt
df_training = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values='?')

df_training.drop('Id',axis='columns',inplace=True)  # A coluna de identificação não é útil para futuras análises
df_training
print('Age statistics:\n')

df_training['age'].describe()
plt.figure(figsize=(15, 7))

df_training['age'].value_counts().plot(kind='bar')

plt.title('Distribution of age in the dataset')
print('Workclass statistics:\n')

df_training['workclass'].describe()
plt.figure(figsize=(15, 7))

df_training['workclass'].value_counts().plot(kind='bar')

plt.title('Distribution of workclass in the dataset')
print('Education statistics:\n')

df_training['education'].describe()
plt.figure(figsize=(15, 7))

df_training['education'].value_counts().plot(kind='bar')

plt.title('Distribution of levels of education in the dataset')
print('Education.num statistics:\n')

df_training['education.num'].describe()
plt.figure(figsize=(15, 7))

df_training['education.num'].value_counts().plot(kind='bar')

plt.title(r'Distribution of levels of education (codified in numbers) in the dataset')
print('Marital.status statistics:\n')

df_training['marital.status'].describe()
plt.figure(figsize=(15, 7))

df_training['marital.status'].value_counts().plot(kind='bar')

plt.title('Distribution of marital status in the dataset')
print('Occupation statistics:\n')

df_training['occupation'].describe()
plt.figure(figsize=(15, 7))

df_training['occupation'].value_counts().plot(kind='bar')

plt.title('Distribution of work occupation in the dataset')
print('Relationship statistics:\n')

df_training['relationship'].describe()
plt.figure(figsize=(15, 7))

df_training['relationship'].value_counts().plot(kind='bar')

plt.title('Distribution of relationship status in the dataset')
print('Race statistics:\n')

df_training['race'].describe()
plt.figure(figsize=(15, 7))

df_training['race'].value_counts().plot(kind='bar')

plt.title('Distribution of race in the dataset')
print('Sex statistics:\n')

df_training['sex'].describe()
plt.figure(figsize=(15, 7))

df_training['sex'].value_counts().plot(kind='bar')

plt.title('Distribution of sex in the dataset')
print('Capital.gain statistics:\n')

df_training['capital.gain'].describe()
print('Capital.loss statistics:\n')

df_training['capital.loss'].describe()
print('Hours.per.week statistics:\n')

df_training['hours.per.week'].describe()
plt.figure(figsize=(15, 7))

df_training['hours.per.week'].value_counts().plot(kind='bar')

plt.title('Distribution of hours of work per week in the dataset')
print('Native.country statistics:\n')

df_training['native.country'].describe()
plt.figure(figsize=(15, 7))

df_training['native.country'].value_counts().plot(kind='bar')

plt.title('Distribution of native country per week the dataset')
print('Income statistics:\n')

df_training['income'].describe()
plt.figure(figsize=(15, 7))

df_training['income'].value_counts().plot(kind='bar')

plt.title('Distribution of income in the dataset')
df_training.isnull().sum()
#df_training = df_training.dropna()   # Eliminação de todas as linhas com dados faltantes

df_training_prep = df_training[['age','workclass', 'fnlwgt','education.num','marital.status','occupation','relationship','race','sex',

                                'capital.gain','capital.loss','hours.per.week','native.country','income']]



# Codificação de todos os atributos não numéricos

workclass_num = []

for i in range(len(df_training_prep['workclass'])):

    if (df_training_prep['workclass'][i] == 'Private'):

        workclass_num.append(1)

    elif (df_training_prep['workclass'][i] == 'Self-emp-not-inc'):

        workclass_num.append(2)

    elif (df_training_prep['workclass'][i] == 'Local-gov'):

        workclass_num.append(3)

    elif (df_training_prep['workclass'][i] == 'State-gov'):

        workclass_num.append(4)

    elif (df_training_prep['workclass'][i] == 'Self-emp-inc'):

        workclass_num.append(5)

    elif (df_training_prep['workclass'][i] == 'Federal-gov'):

        workclass_num.append(6)

    elif (df_training_prep['workclass'][i] == 'Without-pay'):

        workclass_num.append(7)

    elif (df_training_prep['workclass'][i] == 'Never-worked'):

        workclass_num.append(8)

    else:

        workclass_num.append(None)



df_training_prep.drop('workclass',axis='columns',inplace=True)

df_training_prep.insert(1,'workclass.num',workclass_num,True)



marital_status_num = []

for i in range(len(df_training_prep['marital.status'])):

    if (df_training_prep['marital.status'][i] == 'Married-civ-spouse'):

        marital_status_num.append(1)

    elif (df_training_prep['marital.status'][i] == 'Never-married'):

        marital_status_num.append(2)

    elif (df_training_prep['marital.status'][i] == 'Divorced'):

        marital_status_num.append(3)

    elif (df_training_prep['marital.status'][i] == 'Separated'):

        marital_status_num.append(4)

    elif (df_training_prep['marital.status'][i] == 'Widowed'):

        marital_status_num.append(5)

    elif (df_training_prep['marital.status'][i] == 'Married-spouse-absent'):

        marital_status_num.append(6)

    elif (df_training_prep['marital.status'][i] == 'Married-AF-spouse'):

        marital_status_num.append(7)

    else:

        marital_status_num.append(None)

        

df_training_prep.drop('marital.status',axis='columns',inplace=True)

df_training_prep.insert(4,'marital.status.num',marital_status_num,True)



occupation_num = []

for i in range(len(df_training_prep['occupation'])):

    if (df_training_prep['occupation'][i] == 'Prof-specialty'):

        occupation_num.append(1)

    elif (df_training_prep['occupation'][i] == 'Craft-repair'):

        occupation_num.append(2)

    elif (df_training_prep['occupation'][i] == 'Exec-managerial'):

        occupation_num.append(3)

    elif (df_training_prep['occupation'][i] == 'Adm-clerical'):

        occupation_num.append(4)

    elif (df_training_prep['occupation'][i] == 'Sales'):

        occupation_num.append(5)

    elif (df_training_prep['occupation'][i] == 'Other-service'):

        occupation_num.append(6)

    elif (df_training_prep['occupation'][i] == 'Machine-op-inspct'):

        occupation_num.append(7)

    elif (df_training_prep['occupation'][i] == 'Transport-moving'):

        occupation_num.append(8)

    elif (df_training_prep['occupation'][i] == 'Handlers-cleaners'):

        occupation_num.append(9)

    elif (df_training_prep['occupation'][i] == 'Farming-fishing'):

        occupation_num.append(10)

    elif (df_training_prep['occupation'][i] == 'Tech-support'):

        occupation_num.append(11)

    elif (df_training_prep['occupation'][i] == 'Protective-serv'):

        occupation_num.append(12)

    elif (df_training_prep['occupation'][i] == 'Priv-house-serv'):

        occupation_num.append(13)

    elif (df_training_prep['occupation'][i] == 'Armed-Forces'):

        occupation_num.append(14)

    else:

        occupation_num.append(None)



df_training_prep.drop('occupation',axis='columns',inplace=True)

df_training_prep.insert(5,'occupation.num',occupation_num,True)        



relationship_num = []

for i in range(len(df_training_prep['relationship'])):

    if (df_training_prep['relationship'][i] == 'Husband'):

        relationship_num.append(1)

    elif (df_training_prep['relationship'][i] == 'Not-in-family'):

        relationship_num.append(2)

    elif (df_training_prep['relationship'][i] == 'Own-child'):

        relationship_num.append(3)

    elif (df_training_prep['relationship'][i] == 'Unmarried'):

        relationship_num.append(4)

    elif (df_training_prep['relationship'][i] == 'Wife'):

        relationship_num.append(5)

    elif (df_training_prep['relationship'][i] == 'Other-relative'):

        relationship_num.append(6)

    else:

        relationship_num.append(None)



df_training_prep.drop('relationship',axis='columns',inplace=True)

df_training_prep.insert(6,'relationship.num',relationship_num,True)



race_num = []

for i in range(len(df_training_prep['race'])):

    if (df_training_prep['race'][i] == 'White'):

        race_num.append(1)

    elif (df_training_prep['race'][i] == 'Black'):

        race_num.append(2)

    elif (df_training_prep['race'][i] == 'Asian-Pac-Islander'):

        race_num.append(3)

    elif (df_training_prep['race'][i] == 'Amer-Indian-Eskimo'):

        race_num.append(4)

    elif (df_training_prep['race'][i] == 'Other'):

        race_num.append(5)

    else:

        race_num.append(None)

        

df_training_prep.drop('race',axis='columns',inplace=True)

df_training_prep.insert(7,'race.num',race_num,True)



sex_num = []

for i in range(len(df_training_prep['sex'])):

    if (df_training_prep['sex'][i] == 'Male'):

        sex_num.append(1)

    elif (df_training_prep['sex'][i] == 'Female'):

        sex_num.append(2)

    else:

        race_num.append(None)

        

df_training_prep.drop('sex',axis='columns',inplace=True)

df_training_prep.insert(8,'sex.num',sex_num,True)



native_country_num = []

control = 0

for i in range(len(df_training_prep['native.country'])):

    for j in range(len(df_training_prep['native.country'].unique())):

        if (df_training_prep['native.country'][i] == df_training_prep['native.country'].unique()[j]):

            native_country_num.append(j+1)

            control = 1

            break

    if (control == 0):

        native_country_num.append(None)

    control = 0

        

df_training_prep.drop('native.country',axis='columns',inplace=True)

df_training_prep.insert(12,'native.country.num',native_country_num,True)



income_num = []

for i in range(len(df_training_prep['income'])):

    if (df_training_prep['income'][i] == '<=50K'):

        income_num.append(1)

    elif (df_training_prep['income'][i] == '>50K'):

        income_num.append(2)

    else:

        income_num.append(None)

        

df_training_prep.drop('income',axis='columns',inplace=True)

df_training_prep.insert(13,'income.num',income_num,True)



df_training_prep = df_training_prep.dropna()   # Eliminação de todas as linhas com dados faltantes

df_training_prep

corr_matrix = df_training_prep.corr()

plt.figure(figsize=(15, 7))

sn.heatmap(corr_matrix, annot=True)

plt.show()
X = df_training_prep[['age','workclass.num','education.num','marital.status.num','occupation.num','capital.gain','capital.loss','hours.per.week']] 

#X é a base de treinamento



Y = df_training.dropna()

Y = Y[['income']]        #Y contém os rótulos de treinamento
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
scores_mean = []

scores_std = []

max_score = 0

min_std = 10

k_max = 0

k_min = 0



for k in range(0,40):

    KNN = KNeighborsClassifier(n_neighbors=k+1)

    scores = cross_val_score(KNN, X, Y, cv=10)

    scores_mean.append(scores.mean())

    scores_std.append(scores.std())

    if (scores_mean[k] > max_score):

        max_score = scores_mean[k]

        k_max = k

    if (scores_std[k] < min_std):

        min_std = scores_std[k]

        k_min = k
print('Maior média de acurácia para validação cruzada: ' + str(max_score) + ' (k = '+str(k_max+1)+')' + ' com desvio padrão de ' + str(scores_std[k_max]))
print('Menor desvio padrão de acurácia para validação cruzada: ' + str(min_std) + ' (k = '+str(k_min+1)+')' + ' com média de ' + str(scores_mean[k_min]))
X_new = df_training_prep[['age','education.num','marital.status.num','occupation.num','capital.gain','capital.loss','hours.per.week']] 



scores_mean = []

scores_std = []

max_score = 0

min_std = 10

k_max = 0

k_min = 0



for k in range(0,40):

    KNN = KNeighborsClassifier(n_neighbors=k+1)

    scores = cross_val_score(KNN, X_new, Y, cv=10)

    scores_mean.append(scores.mean())

    scores_std.append(scores.std())

    if (scores_mean[k] > max_score):

        max_score = scores_mean[k]

        k_max = k

    if (scores_std[k] < min_std):

        min_std = scores_std[k]

        k_min = k
print('Maior média de acurácia para validação cruzada: ' + str(max_score) + ' (k = '+str(k_max+1)+')' + ' com desvio padrão de ' + str(scores_std[k_max]))
print('Menor desvio padrão de acurácia para validação cruzada: ' + str(min_std) + ' (k = '+str(k_min+1)+')' + ' com média de ' + str(scores_mean[k_min]))
index = []

for i in range(0,40):

    index.append(i+1)
plt.figure(figsize=(15, 7))

plt.plot(index, scores_mean)

plt.xlabel('k')

plt.ylabel('mean accuracy')

plt.title('KNN performed with cross-validation (without workclass.num)')
plt.figure(figsize=(15, 7))

plt.plot(index, scores_std)

plt.xlabel('k')

plt.ylabel('standard deviation')

plt.title('KNN performed with cross-validation (without workclass.num)')
KNN = KNeighborsClassifier(n_neighbors=k_max+1)

KNN.fit(X_new, Y)
df_test = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv",

        sep=r'\s*,\s*',

        engine='python',

        na_values='?')

#Eliminando as colunas que não serão utilizadas

df_test.drop('Id',axis='columns',inplace=True)

df_test.drop('fnlwgt',axis='columns',inplace=True)

df_test.drop('workclass',axis='columns',inplace=True)

df_test.drop('education',axis='columns',inplace=True)

df_test.drop('relationship',axis='columns',inplace=True)

df_test.drop('race',axis='columns',inplace=True)

df_test.drop('sex',axis='columns',inplace=True)

df_test.drop('native.country',axis='columns',inplace=True)

df_test.isnull().sum()
fill = df_test['occupation'].describe().top

df_test['occupation'] = df_test['occupation'].fillna(fill)
marital_status_num = []

for i in range(len(df_test['marital.status'])):

    if (df_test['marital.status'][i] == 'Married-civ-spouse'):

        marital_status_num.append(1)

    elif (df_test['marital.status'][i] == 'Never-married'):

        marital_status_num.append(2)

    elif (df_test['marital.status'][i] == 'Divorced'):

        marital_status_num.append(3)

    elif (df_test['marital.status'][i] == 'Separated'):

        marital_status_num.append(4)

    elif (df_test['marital.status'][i] == 'Widowed'):

        marital_status_num.append(5)

    elif (df_test['marital.status'][i] == 'Married-spouse-absent'):

        marital_status_num.append(6)

    elif (df_test['marital.status'][i] == 'Married-AF-spouse'):

        marital_status_num.append(7)

    else:

        marital_status_num.append(None)

        

df_test.drop('marital.status',axis='columns',inplace=True)

df_test.insert(2,'marital.status.num',marital_status_num,True)



occupation_num = []

for i in range(len(df_test['occupation'])):

    if (df_test['occupation'][i] == 'Prof-specialty'):

        occupation_num.append(1)

    elif (df_test['occupation'][i] == 'Craft-repair'):

        occupation_num.append(2)

    elif (df_test['occupation'][i] == 'Exec-managerial'):

        occupation_num.append(3)

    elif (df_test['occupation'][i] == 'Adm-clerical'):

        occupation_num.append(4)

    elif (df_test['occupation'][i] == 'Sales'):

        occupation_num.append(5)

    elif (df_test['occupation'][i] == 'Other-service'):

        occupation_num.append(6)

    elif (df_test['occupation'][i] == 'Machine-op-inspct'):

        occupation_num.append(7)

    elif (df_test['occupation'][i] == 'Transport-moving'):

        occupation_num.append(8)

    elif (df_test['occupation'][i] == 'Handlers-cleaners'):

        occupation_num.append(9)

    elif (df_test['occupation'][i] == 'Farming-fishing'):

        occupation_num.append(10)

    elif (df_test['occupation'][i] == 'Tech-support'):

        occupation_num.append(11)

    elif (df_test['occupation'][i] == 'Protective-serv'):

        occupation_num.append(12)

    elif (df_test['occupation'][i] == 'Priv-house-serv'):

        occupation_num.append(13)

    elif (df_test['occupation'][i] == 'Armed-Forces'):

        occupation_num.append(14)

    else:

        occupation_num.append(None)



df_test.drop('occupation',axis='columns',inplace=True)

df_test.insert(3,'occupation.num',occupation_num,True)



df_test
classes = KNN.predict(df_test)

output = pd.DataFrame()

output[0] = df_test.index

output[1] = classes

output.columns = ['Id', 'Income']
output.to_csv('submission.csv',index = False)