import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
arq = pd.read_csv("../input/train.csv", encoding="latin")

df = arq
df.info()
df.head()
df.describe()
df['target']=df['target'].map({' <=50K':1,' >50K':0}).astype(int)



target = df['target']
df.columns = ['id','age','workclass', 'fnlwgt','education','enum', 'marital', 'occupation',

              'relationship', 'race', 'sex', 'gain', 'loss', 'hours', 'country', 'target']
df.head()
df['sex'] = df['sex'].map({' Male':0,' Female':1}).astype(int)
#Ordenação



race_sorted = (df[['race','target']].groupby(['race']).mean()).sort_values('target')

country_sorted = (df[['country','target']].groupby(['country']).mean()).sort_values('target')

workclass_sorted = (df[['workclass','target']].groupby(['workclass']).mean()).sort_values('target')

marital_sorted = (df[['marital','target']].groupby(['marital']).mean()).sort_values('target')

occupation_sorted = (df[['occupation','target']].groupby(['occupation']).mean()).sort_values('target')

relationship_sorted = (df[['relationship','target']].groupby(['relationship']).mean()).sort_values('target')
relationship_sorted
race_dic = {}

country_dic = {}

workclass_dic = {}

marital_dic = {}

occupation_dic = {}

relationship_dic = {}



for k in range(race_sorted.size):

    race_dic[race_sorted['target'].index[k]] = k



for k in range(country_sorted.size):

    country_dic[country_sorted['target'].index[k]] = k



for k in range(workclass_sorted.size):

    workclass_dic[workclass_sorted['target'].index[k]] = k



for k in range(marital_sorted.size):

    marital_dic[marital_sorted['target'].index[k]] = k

    

for k in range(occupation_sorted.size):

    occupation_dic[occupation_sorted['target'].index[k]] = k

    

for k in range(relationship_sorted.size):

    relationship_dic[relationship_sorted['target'].index[k]] = k
relationship_dic
df['race'] = df['race'].map(race_dic)

df['country'] = df['country'].map(country_dic)

df['workclass'] = df['workclass'].map(workclass_dic)

df['marital'] = df['marital'].map(marital_dic)

df['occupation'] = df['occupation'].map(occupation_dic)

df['relationship'] = df['relationship'].map(relationship_dic)
df.head()
df.pop('id')

df.pop('education')



target = df.pop('target')



df.head()
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression # Importando o módulo



logr = LogisticRegression(solver="liblinear", multi_class="auto") #criando o objeto do modelo



cv_result = cross_val_score(logr, df, target, cv=10, scoring="accuracy")

print("Acurácia com cross validation:", cv_result.mean()*100)
from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=25) #Alterar o número de vizinhos altera o desempenho do modelo



cv_result = cross_val_score(knn, df, target, cv=10, scoring="accuracy")

print("Acurácia com cross validation:", cv_result.mean()*100)
from sklearn.naive_bayes import GaussianNB # ou MultinomialNB, BinomialNB, ComplementNB



gnb = GaussianNB()



cv_result = cross_val_score(gnb,df,target, cv = 10,scoring = "accuracy")

print("Acurácia com cross validation:" + str(cv_result.mean()*100))
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier



dtc = DecisionTreeClassifier()

rfc = RandomForestClassifier(n_estimators=100, min_samples_split=15, min_samples_leaf=4) #RFC tunadasso fodase



cv_result = cross_val_score(dtc, df, target, cv=10, scoring="accuracy")

print("Acurácia de uma árvore com cross validation:", cv_result.mean()*100)



cv_result = cross_val_score(rfc, df, target, cv=10, scoring="accuracy")

print("Acurácia da floresta com cross validation:", cv_result.mean()*100)
test = pd.read_csv("../input/test.csv", encoding="latin")
test.head()
test.columns = ['id','age','workclass', 'fnlwgt','education','enum', 'marital', 'occupation',

              'relationship', 'race', 'sex', 'gain', 'loss', 'hours', 'country']
test['sex'] = test['sex'].map({' Male':0,' Female':1}).astype(int)





test['race'] = test['race'].map(race_dic)

test['country'] = test['country'].map(country_dic)

test['workclass'] = test['workclass'].map(workclass_dic)

test['marital'] = test['marital'].map(marital_dic)

test['occupation'] = test['occupation'].map(occupation_dic)

test['relationship'] = test['relationship'].map(relationship_dic)



test.head()
test_id = test.pop('id')

test.pop('education')



test.head()
#Predict com Random Forest



rfc.fit(df, target)

rfc.predict(test)
#Predict com Regressão Linear



#logr.fit(df, target)

#logr.predict(test)



#Predict com KNN



#knn.fit(df, target)

#knn.predict(test)
submission = pd.DataFrame()
submission[0] = test_id

submission[1] = rfc.predict(test)
submission.columns = ['Id','target']



submission['target'] = submission['target'].map({0:' >50K', 1:' <=50K'})
submission
submission.to_csv('submission.csv',index = False)
import keras

from keras.models import Sequential

from keras.layers import Dense
from sklearn.preprocessing import StandardScaler







scaler = StandardScaler().fit(df)



# Feature Scaling do dataset de treino

df = scaler.transform(df)



# Feature Scaling do dataset de testes

test = scaler.transform(test)
#from sklearn.preprocessing import RobustScaler

#from sklearn.preprocessing import Normalizer



#scaler = RobustScaler().fit(df)

#scaler = Normalizer().fit(df)
# Inicializando a ANN

classifier = Sequential()
# Input layer e o primeiro hidden layer

classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu', input_dim = 13))

               

# Adicionando o segundo hidden layer

classifier.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'tanh'))



# Um terceiro hidden layer, se for necessário

#classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'tanh'))



# E o output layer

classifier.add(Dense(units = 1,kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(df, target, batch_size = 256, epochs = 10, validation_split=.2)
y_pred = classifier.predict(test)
y_pred = (y_pred >= 0.5)



y_pred
SubKeras = pd.DataFrame()



SubKeras[0] = test_id

SubKeras[1] = y_pred



SubKeras.columns = ['Id','target']



SubKeras['target'] = SubKeras['target'].map({False:' >50K', True:' <=50K'})



SubKeras
SubKeras.to_csv('SubKeras.csv',index = False)