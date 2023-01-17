import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import warnings

warnings.filterwarnings('ignore')

from sklearn import metrics

%matplotlib inline
x_treino = pd.read_csv('competicao-dsa-machine-learning-sep-2019/X_treino.csv')
x_treino.head(5)
x_treino.describe()
x_treino.info()
def missing_values_table(df):

        mis_val = df.isnull().sum()

        

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        print ("Seu dataframe tem " + str(df.shape[1]) + " colunas.\n"      

            "Há " + str(mis_val_table_ren_columns.shape[0]) +

              " colunas que possuem valores ausentes.")

        

        return mis_val_table_ren_columns
missing_values_table(x_treino)
y_treino = pd.read_csv('competicao-dsa-machine-learning-sep-2019/y_treino.csv')
y_treino.head(5)
x_treino_group = x_treino.groupby(['series_id']).mean()
dados_treino = pd.merge(x_treino_group, y_treino, on='series_id', how='left')
dados_treino.head(5)
dados_treino = dados_treino.drop(['measurement_number'], axis=1)
colunas = dados_treino.columns[1:13]

plt.subplots(figsize=(18,15))

lenght = len(colunas)

for i, j in zip(colunas, range(lenght)):

    plt.subplot((lenght/2), 3, j + 1)

    plt.subplots_adjust(wspace = 0.2, hspace = 0.5)

    dados_treino[i].hist(bins=20, edgecolor='black')

    plt.title(i)

plt.show()

    
dados_treino['surface'].value_counts().plot(kind='bar', figsize=(10,10))

plt.title('Ocorrência de pisos')

plt.ylabel('Frequency')

plt.xlabel('surface')

plt.show()
dados_treino.plot(kind='box', subplots=True, layout = (4,4), figsize = (15,15))
dados_treino.boxplot(column='orientation_X', by='surface', figsize=(15,6))

plt.show()
dados_treino.boxplot(column='orientation_Y', by='surface', figsize=(15,6))

plt.show()
dados_treino.boxplot(column='orientation_Z', by='surface', figsize=(15,6))

plt.show()
dados_treino.boxplot(column='orientation_W', by='surface', figsize=(15,6))

plt.show()
dados_treino.boxplot(column='angular_velocity_X', by='surface', figsize=(15,6))

plt.show()
dados_treino.boxplot(column='angular_velocity_Y', by='surface', figsize=(15,6))

plt.show()
dados_treino.boxplot(column='angular_velocity_Z', by='surface', figsize=(15,6))

plt.show()
dados_treino.boxplot(column='linear_acceleration_X', by='surface', figsize=(15,6))

plt.show()
dados_treino.boxplot(column='linear_acceleration_Y', by='surface', figsize=(15,6))

plt.show()
dados_treino.boxplot(column='linear_acceleration_Z', by='surface', figsize=(15,6))

plt.show()
def normalize(value):

    

    if value == 'fine_concrete':

        return 0

    elif value == 'concrete':

        return 1

    elif value == 'soft_tiles':

        return 2

    elif value == 'tiled':

        return 3

    elif value == 'soft_pvc':

        return 4

    elif value == 'hard_tiles_large_space':

        return 5

    elif value == 'carpet':

        return 6

    elif value == 'hard_tiles':

        return 7

    elif value == 'wood':

        return 8
dados_treino['surface_norm'] = dados_treino['surface'].map(lambda x: normalize(x))
dados_treino
corr = dados_treino.corr()

_ , ax = plt.subplots( figsize =( 12 , 10 ) )

cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

_ = sns.heatmap(corr, cmap = cmap, square=True, cbar_kws={ 'shrink' : .9 }, ax=ax, annot = True, annot_kws = {'fontsize' : 12 })
correlation_data = dados_treino.corr()['surface_norm'].sort_values()



print(correlation_data)
dados_treino = dados_treino.drop(['surface','group_id'], axis=1)
dados = dados_treino.drop(['surface_norm'], axis=1)
import imblearn
#Balanceando as classes

np.random.seed(75)

from imblearn.over_sampling import SMOTE, ADASYN

data_o, target_o = SMOTE().fit_sample(dados, dados_treino.surface_norm)
data_o.shape
target_o.shape
import collections

collections.Counter(target_o)
from sklearn.model_selection import train_test_split
Xo_train, Xo_test, yo_train, yo_test = train_test_split(data_o, target_o, test_size=0.20, random_state=4)
Xo_train.shape
yo_train.shape
Xo_test.shape
yo_test.shape
data = dados_treino[dados_treino.columns[:12]]

target = dados_treino['surface_norm']
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

cols = data.iloc[:, 0:12].columns

data[cols] = scaler.fit_transform(data)

data.head()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(Xo_train, yo_train)
pred = knn.predict(Xo_test)
pred.shape
from sklearn.metrics import confusion_matrix

print(confusion_matrix(yo_test,pred))
from sklearn.metrics import classification_report

print(classification_report(yo_test, pred))
error_rate = []



for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(Xo_train, yo_train)

    pred_i = knn.predict(Xo_test)

    error_rate.append(np.mean(pred_i != yo_test))

    

plt.figure(figsize=(10,6))

plt.plot(range(1,40), error_rate, color='blue', linestyle='dashed', marker='o',

        markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
dados_treino4 = dados_treino.drop(['orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W',

                                 'linear_acceleration_Y', 'linear_acceleration_Z', 'series_id'], axis=1)
dados4 = dados_treino4.drop(['surface_norm'], axis=1)
np.random.seed(75)

from imblearn.over_sampling import SMOTE, ADASYN

data_o, target_o = SMOTE().fit_sample(dados4, dados_treino4.surface_norm)
knn = KNeighborsClassifier(n_neighbors=3)
data_o.shape
target_o.shape
Xo_train, Xo_test, yo_train, yo_test = train_test_split(data_o, target_o, test_size=0.2, random_state=4)
Xo_train.shape
Xo_test.shape
yo_train.shape
yo_test.shape
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(Xo_train, yo_train)
pred = knn.predict(Xo_test)
print(confusion_matrix(yo_test,pred))
print(classification_report(yo_test, pred))
print("Acurácia para o algoritmo decision tree classifier é",metrics.accuracy_score(pred,yo_test))
print(correlation_data)
dados_centro = dados_treino.drop(['angular_velocity_Y', 'angular_velocity_X', 'linear_acceleration_Y',

                                 'linear_acceleration_Z', 'series_id', 'linear_acceleration_X', 

                                  'angular_velocity_Z'], axis=1)
dados_c = dados_centro.drop(['surface_norm'], axis=1)
data_o, target_o = SMOTE().fit_sample(dados_c, dados_centro.surface_norm)
Xo_train, Xo_test, yo_train, yo_test = train_test_split(data_o, target_o, test_size=0.2, random_state=4)
knn.fit(Xo_train, yo_train)
pred = knn.predict(Xo_test)
print(confusion_matrix(yo_test,pred))
print(classification_report(yo_test, pred))
print(correlation_data)
dados_positivo = dados_treino.drop(['angular_velocity_Y','angular_velocity_X', 'linear_acceleration_Y',

                                 'linear_acceleration_Z','series_id'], axis=1)
dados_p = dados_positivo.drop(['surface_norm'], axis=1)
knn = KNeighborsClassifier(n_neighbors=2)
data_o, target_o = SMOTE().fit_sample(dados_p, dados_positivo.surface_norm)
Xo_train, Xo_test, yo_train, yo_test = train_test_split(data_o, target_o, test_size=0.20, random_state=4)
knn.fit(Xo_train, yo_train)
pred = knn.predict(Xo_test)
print(confusion_matrix(yo_test,pred))
print(classification_report(yo_test, pred))
from sklearn.tree import DecisionTreeClassifier

random_state=234

dtree = DecisionTreeClassifier(random_state=998)
dtree.fit(Xo_train, yo_train)
pred = dtree.predict(Xo_test)

print("Acurácia para o algoritmo decision tree classifier é",metrics.accuracy_score(pred,yo_test))
dados_teste = pd.read_csv('competicao-dsa-machine-learning-sep-2019/X_teste.csv')
series_id = pd.DataFrame(dados_teste['series_id'])

series_id
dados_teste = dados_teste.drop(['row_id', 'series_id', 'measurement_number','angular_velocity_Y','angular_velocity_X', 'linear_acceleration_Y',

                                 'linear_acceleration_Z', 'series_id'], axis=1)
dados_teste['surface_norm'] = dtree.predict(dados_teste)
def valores_nominais(value):

    

    if value == 0 :

        return 'fine_concrete'

    elif value == 1:

        return 'concrete'

    elif value == 2:

        return 'soft_tiles'

    elif value == 3:

        return 'tiled'

    elif value == 4:

        return 'soft_pvc'

    elif value == 5:

        return 'hard_tiles_large_space'

    elif value == 6:

        return 'carpet'

    elif value == 7:

        return 'hard_tiles'

    elif value == 8:

        return 'wood'
dados_teste['surface'] = dados_teste['surface_norm'].map(lambda x: valores_nominais(x))

dados_teste = dados_teste.drop(['surface_norm'], axis=1)

dados_teste.sample(10)
dados_teste['surface'].value_counts().plot(kind='bar', figsize=(10,10))

plt.title('Ocorrência de pisos')

plt.ylabel('Frequency')

plt.xlabel('surface')

plt.show()
result = dados_teste.drop(['orientation_X', 'orientation_Y', 'orientation_Z', 'orientation_W', 

                           'angular_velocity_Z', 'linear_acceleration_X'], axis=1)

result = pd.concat([series_id, result], axis=1)

result
result = result.drop_duplicates('series_id')
result.to_csv('submission_robo.csv')