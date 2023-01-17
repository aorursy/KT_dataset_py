import pandas as pd

import matplotlib.pylab as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler, RobustScaler

from keras.layers import Dense

from keras.models import Sequential

from sklearn.model_selection import train_test_split

from keras.utils import np_utils

from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import ExtraTreesClassifier

from scipy.stats import norm





base = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
# -*- coding: utf-8 -*-

"""

Created on Fri Aug 30 23:22:46 2019



@author: Andre B Silva

"""







cor = base.corr()

base.plot(kind="hist", bins=10 , figsize=(10,10))



#RobustScaler para deixa os dados das colunas em escalas mais proximas e menos sensiveis a outliers

robus_scaler = RobustScaler()



#adicionando uma coluna chamada scaled_amount, ao data frame

#essa coluna seria os valores da coluna V4 e V14 , mas Escalados



base['scaled_V4'] = robus_scaler.fit_transform(base['V4'].values.reshape(-1,1))

base['scaled_V14'] = robus_scaler.fit_transform(base['V14'].values.reshape(-1,1))





base.drop(['V4','V14'],axis=1,inplace=True)



#Embaralhando as linhas do dataframe

df = base.sample(frac=1)



#selecionando todos os registros de fraude

fraud_df = df.loc[df["Class"]==1]



#selecionando a mesma quantidade de registro sem fraude 492 registros

nofraud_df = df.loc[df["Class"]==0][:492]



#juntando os dataframes 

distributed_df = pd.concat([fraud_df,nofraud_df])



#Emralhando e criando um novo Dataframe

#Se random_state != None o embaralhamento sempre sera igual quando executada 

new_df = distributed_df.sample(frac=1,random_state = 42)



#selecionando os previsores 

x_data = new_df.drop("Class",axis=1)











x_data = x_data[['scaled_V4','scaled_V14']]





#------------

#selecionando a classe

y_data = new_df['Class']


x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size = 0.3,random_state = 0)

#

model = Sequential()



model.add(Dense(units = 50,input_dim = 2,activation='sigmoid'))



model.add(Dense(units = 100))

model.add(Dense(units = 100))



model.add(Dense(units=1,activation='sigmoid'))



model.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])



model.fit(x_train,y_train,epochs = 100,validation_data=[x_test,y_test])



predicts = model.predict(x_test)



predicts2 = (predicts > 0.5)



confusion_neural = confusion_matrix(predicts2,y_test)



accuracy_neural = accuracy_score(predicts2,y_test)



print(f'Accuracy Neural Network :{accuracy_neural}')




naive_bayes = GaussianNB()



naive_bayes.fit(x_train,y_train)



predict_naive = naive_bayes.predict(x_test)



confusion_naive = confusion_matrix(predict_naive,y_test)



accuracy_naive = accuracy_score(predict_naive,y_test)



print(f'Accuracy Naive Bayes:{accuracy_naive}')

forest = ExtraTreesClassifier(  n_estimators = 100,random_state = 1)



forest.fit(x_train,y_train)



predict_forest = forest.predict(x_test)



attribute_selected = forest.feature_importances_



confusion_forest = confusion_matrix(y_test,predict_forest)

accuracy_forest = accuracy_score(y_test,predict_forest)



print(f'Acurracy Random Forest:{accuracy_forest}')



f, (ax1) = plt.subplots(1,1, figsize=(10, 6))



v14_fraud_dist = new_df['scaled_V14'].loc[new_df['Class'] == 1].values

sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861')

ax1.set_title('V14 Distribution \n (Fraud Transactions)', fontsize=14)
