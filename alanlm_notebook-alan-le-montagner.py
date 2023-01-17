import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import random

import os

from random import *

from collections import Counter

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import linear_model

from keras.layers import Dense

from keras.models import Sequential

from keras.models import load_model

from keras.layers import Dropout







%matplotlib inline
df = pd.read_csv("/kaggle/input/atpdata/ATP.csv",dtype=str)
df['tourney_month']=df.tourney_date.astype(str).str[:6]

df = df[df['tourney_month'].between('199101','201909')]



df.info()



df = df.drop(columns=['draw_size','winner_entry','winner_seed','loser_entry','loser_seed'])

df = df.dropna()

df.shape



#nb_data=df.shape[0]

#nb_param=df.shape[1]

#count=np.zeros(nb_param)

#for i in range(1,nb_data):

#    for j in range(1,nb_param):

#        if np.isnan(df[df.columns[j]].iloc[i]):

#            count[j]+=1



    



#df = df.dropna().astype(float)
df.describe().transpose()
#Visualisation des variables propres aux joueurs



df['winner_age']=pd.to_numeric(df['winner_age'])

df['winner_ht']=pd.to_numeric(df['winner_ht'])

df['winner_id']=pd.to_numeric(df['winner_id'])

df['winner_rank']=pd.to_numeric(df['winner_rank'])

df['winner_rank_points']=pd.to_numeric(df['winner_rank_points'])



df['loser_age']=pd.to_numeric(df['loser_age'])

df['loser_ht']=pd.to_numeric(df['loser_ht'])

df['loser_id']=pd.to_numeric(df['loser_id'])

df['loser_rank']=pd.to_numeric(df['loser_rank'])

df['loser_rank_points']=pd.to_numeric(df['loser_rank_points'])





plt.figure(figsize=(20,10))

plt.subplot(2,4,1)

df['winner_age'].plot(kind='hist',bins=26, xlim=(15,40), ylim=(0,10000), title='Age du gagnant')



plt.subplot(2,4,2)

df['loser_age'].plot(kind='hist',bins=26, xlim=(15,40), ylim=(0,10000), title='Age du perdant')



plt.subplot(2,4,3)

df['winner_ht'].plot(kind='hist',bins=15, xlim=(160,210), ylim=(0,25000), title='Taille du gagnant')



plt.subplot(2,4,4)

df['loser_ht'].plot(kind='hist',bins=15, xlim=(160,210), ylim=(0,25000), title='Taille du perdant')



plt.subplot(2,4,5)

df['winner_rank'].plot(kind='hist',bins=100, xlim=(0,800), ylim=(0,25000), title='Rang du gagnant')



plt.subplot(2,4,6)

df['loser_rank'].plot(kind='hist',bins=100, xlim=(0,800), ylim=(0,25000), title='Rang du perdant')



plt.subplot(2,4,7)

df['winner_rank_points'].plot(kind='hist',bins=100, xlim=(0,14000), ylim=(0,15000), title='Points de classement du gagnant')



plt.subplot(2,4,8)

df['loser_rank_points'].plot(kind='hist',bins=100, xlim=(0,14000), ylim=(0,15000), title='Points de classement du perdant')
df['w_1stWon']=pd.to_numeric(df['w_1stWon'])

df['w_2ndWon']=pd.to_numeric(df['w_2ndWon'])

df['w_SvGms']=pd.to_numeric(df['w_SvGms'])

df['w_ace']=pd.to_numeric(df['w_ace'])

df['w_bpFaced']=pd.to_numeric(df['w_bpFaced'])

df['w_bpSaved']=pd.to_numeric(df['w_bpSaved'])

df['w_df']=pd.to_numeric(df['w_df'])

df['w_svpt']=pd.to_numeric(df['w_svpt'])



df['l_1stWon']=pd.to_numeric(df['l_1stWon'])

df['l_2ndWon']=pd.to_numeric(df['l_2ndWon'])

df['l_SvGms']=pd.to_numeric(df['l_SvGms'])

df['l_ace']=pd.to_numeric(df['l_ace'])

df['l_bpFaced']=pd.to_numeric(df['l_bpFaced'])

df['l_bpSaved']=pd.to_numeric(df['l_bpSaved'])

df['l_df']=pd.to_numeric(df['l_df'])

df['l_svpt']=pd.to_numeric(df['l_svpt'])





plt.figure(figsize=(20,15))



plt.subplot(3,4,1)

df['w_1stWon'].plot(kind='hist', title='Premiers services remportés par le gagnant (%)', bins=100, xlim=(0,100), ylim=(0,10000))



plt.subplot(3,4,2)

df['l_1stWon'].plot(kind='hist', title='Premiers services remportés par le perdant (%)', bins=100, xlim=(0,100), ylim=(0,10000))



plt.subplot(3,4,3)

df['w_2ndWon'].plot(kind='hist', title='Seconds services remportés par le gagnant (%)', bins=50, xlim=(0,50), ylim=(0,10000))



plt.subplot(3,4,4)

df['l_2ndWon'].plot(kind='hist', title='Seconds services remportés par le perdant (%)', bins=50, xlim=(0,50), ylim=(0,10000))



plt.subplot(3,4,5)

df['w_bpFaced'].plot(kind='hist', title='Breakpoints endurés par le gagnant', bins=30, xlim=(0,30), ylim=(0,20000))



plt.subplot(3,4,6)

df['l_bpFaced'].plot(kind='hist', title='Breakpoints endurés par le perdant', bins=30, xlim=(0,30), ylim=(0,20000))



plt.subplot(3,4,7)

df['w_bpSaved'].plot(kind='hist', title='Breakpoints sauvés par le gagnant', bins=25, xlim=(0,25), ylim=(0,20000))



plt.subplot(3,4,8)

df['l_bpSaved'].plot(kind='hist', title='Breakpoints sauvés par le perdant', bins=25, xlim=(0,25), ylim=(0,20000))



plt.subplot(3,4,9)

df['w_ace'].plot(kind='hist', title='Aces du gagnant', bins=40, xlim=(0,40), ylim=(0,30000))



plt.subplot(3,4,10)

df['l_ace'].plot(kind='hist', title='Aces du perdant', bins=40, xlim=(0,40), ylim=(0,30000))



plt.subplot(3,4,11)

df['w_df'].plot(kind='hist', title='Double fautes du gagnant', bins=20, xlim=(0,20), ylim=(0,30000))



plt.subplot(3,4,12)

df['l_df'].plot(kind='hist', title='Double fautes du perdant', bins=20, xlim=(0,20), ylim=(0,30000))
plt.figure(1, figsize=(20,12))



country_names_loser=np.array(Counter(df['loser_ioc']).most_common())[:,0]

country_appearances_loser=list(map(int,np.array(Counter(df['loser_ioc']).most_common())[:,1]))

plt.subplot(2,4,1)

P1=plt.pie(country_appearances_loser,labels=country_names_loser)



country_names_winner=np.array(Counter(df['winner_ioc']).most_common())[:,0]

country_appearances_winner=list(map(int,np.array(Counter(df['winner_ioc']).most_common())[:,1]))

plt.subplot(2,4,2)

P2=plt.pie(country_appearances_winner,labels=country_names_winner)
#On définit les variables d'entrée (= paramètres connus avant le match)

input_param=['p1_hand','p1_ht','p1_age','p1_rank','p1_rank_points','p2_hand','p2_ht','p2_age','p2_rank','p2_rank_points','surface']



#On définit les variables de sortie (= statistiques du match utilisées sur les sites de paris sportifs)

output_param=['p1_1stWon', 'p1_2ndWon', 'p1_SvGms', 'p1_ace', 'p1_bpFaced', 'p1_bpSaved', 'p1_df', 'p1_svpt', 'p2_1stWon', 'p2_2ndWon', 'p2_SvGms', 'p2_ace', 'p2_bpFaced', 'p2_bpSaved', 'p2_df', 'p2_svpt', 'winner_number']
df_shuffle=df.copy()

winner_number=np.zeros(df.shape[0])



pd.options.mode.chained_assignment = None  # default='warn'



#for i in range(0,1000):

for i in range(0,df.shape[0]):

    n=randint(1,2)

    winner_number[i]=n-1

    if n==1:

        

        #Inputs

        

        tmp=df_shuffle.iloc[i]['loser_id']

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_id')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_id')]

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_id')]=tmp

  

        tmp=df_shuffle.iloc[i]['loser_hand']

        if (df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_hand')]=='R'):

            df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_hand')]=0

        else:

            df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_hand')]=1

            

        if (tmp=='R'):

            df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_hand')]=0

        else:

            df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_hand')]=1

        

        tmp=df_shuffle.iloc[i]['loser_ht']

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_ht')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_ht')]

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_ht')]=tmp

        

        tmp=df_shuffle.iloc[i]['loser_age']

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_age')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_age')]

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_age')]=tmp

        

        tmp=df_shuffle.iloc[i]['loser_rank']

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_rank')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_rank')]

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_rank')]=tmp

        

        tmp=df_shuffle.iloc[i]['loser_id']

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_rank_points')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_rank_points')]

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_rank_points')]=tmp

        

        #Outputs

        

        tmp=df_shuffle.iloc[i]['l_1stWon']

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('l_1stWon')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_1stWon')]

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_1stWon')]=tmp

        

        tmp=df_shuffle.iloc[i]['l_2ndWon']

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('l_2ndWon')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_2ndWon')]

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_2ndWon')]=tmp

        

        tmp=df_shuffle.iloc[i]['l_SvGms']

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('l_SvGms')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_SvGms')]

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_SvGms')]=tmp

        

        tmp=df_shuffle.iloc[i]['l_ace']

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('l_ace')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_ace')]

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_ace')]=tmp

        

        tmp=df_shuffle.iloc[i]['l_bpFaced']

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('l_bpFaced')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_bpFaced')]

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_bpFaced')]=tmp

        

        tmp=df_shuffle.iloc[i]['l_bpSaved']

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('l_bpSaved')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_bpSaved')]

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_bpSaved')]=tmp

        

        tmp=df_shuffle.iloc[i]['l_df']

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('l_df')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_df')]

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_df')]=tmp

        

        tmp=df_shuffle.iloc[i]['l_svpt']

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('l_svpt')]=df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_svpt')]

        df_shuffle.iloc[i,df_shuffle.columns.get_loc('w_svpt')]=tmp

        

    else:

        if (df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_hand')]=='R'):

            df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_hand')]=0

        else:

            df_shuffle.iloc[i,df_shuffle.columns.get_loc('winner_hand')]=1

            

        if (df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_hand')]=='R'):

            df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_hand')]=0

        else:

            df_shuffle.iloc[i,df_shuffle.columns.get_loc('loser_hand')]=1   
#input_df=pd.DataFrame(columns=['p1_id','p1_hand','p1_ht','p1_age','p1_rank','p1_rank_points','p2_id','p2_hand','p2_ht','p2_age','p2_rank','p2_rank_points'])#,'surface'])

input_df=pd.DataFrame(columns=['p1_hand','p1_ht','p1_age','p1_rank','p1_rank_points','p2_hand','p2_ht','p2_age','p2_rank','p2_rank_points'])#,'surface'])



#input_df.iloc[:,input_df.columns.get_loc('p1_id')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('loser_id')].copy()

#input_df.iloc[:,input_df.columns.get_loc('p2_id')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('winner_id')].copy()

input_df.iloc[:,input_df.columns.get_loc('p1_hand')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('loser_hand')].copy()

input_df.iloc[:,input_df.columns.get_loc('p2_hand')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('winner_hand')].copy()

input_df.iloc[:,input_df.columns.get_loc('p1_ht')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('loser_ht')].copy()

input_df.iloc[:,input_df.columns.get_loc('p2_ht')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('winner_ht')].copy()

input_df.iloc[:,input_df.columns.get_loc('p1_age')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('loser_age')].copy()

input_df.iloc[:,input_df.columns.get_loc('p2_age')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('winner_age')].copy()

input_df.iloc[:,input_df.columns.get_loc('p1_rank')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('loser_rank')].copy()

input_df.iloc[:,input_df.columns.get_loc('p2_rank')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('winner_rank')].copy()

input_df.iloc[:,input_df.columns.get_loc('p1_rank_points')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('loser_rank_points')].copy()

input_df.iloc[:,input_df.columns.get_loc('p2_rank_points')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('winner_rank_points')].copy()

#input_df.iloc[:,input_df.columns.get_loc('surface')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('surface')].copy()
output_df=pd.DataFrame(columns=['p1_1stWon', 'p1_2ndWon', 'p1_SvGms', 'p1_ace', 'p1_bpFaced', 'p1_bpSaved', 'p1_df', 'p1_svpt', 'p2_1stWon', 'p2_2ndWon', 'p2_SvGms', 'p2_ace', 'p2_bpFaced', 'p2_bpSaved', 'p2_df', 'p2_svpt', 'winner_number'])



output_df.iloc[:,output_df.columns.get_loc('p1_1stWon')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_1stWon')].copy()

output_df.iloc[:,output_df.columns.get_loc('p2_1stWon')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_1stWon')].copy()

output_df.iloc[:,output_df.columns.get_loc('p1_2ndWon')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_2ndWon')].copy()

output_df.iloc[:,output_df.columns.get_loc('p2_2ndWon')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_2ndWon')].copy()

output_df.iloc[:,output_df.columns.get_loc('p1_SvGms')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_SvGms')].copy()

output_df.iloc[:,output_df.columns.get_loc('p2_SvGms')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_SvGms')].copy()

output_df.iloc[:,output_df.columns.get_loc('p1_ace')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_ace')].copy()

output_df.iloc[:,output_df.columns.get_loc('p2_ace')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_ace')].copy()

output_df.iloc[:,output_df.columns.get_loc('p1_bpFaced')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_bpFaced')].copy()

output_df.iloc[:,output_df.columns.get_loc('p2_bpFaced')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_bpFaced')].copy()

output_df.iloc[:,output_df.columns.get_loc('p1_bpSaved')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_bpSaved')].copy()

output_df.iloc[:,output_df.columns.get_loc('p2_bpSaved')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_bpSaved')].copy()

output_df.iloc[:,output_df.columns.get_loc('p1_df')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_df')].copy()

output_df.iloc[:,output_df.columns.get_loc('p2_df')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_df')].copy()

output_df.iloc[:,output_df.columns.get_loc('p1_svpt')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_svpt')].copy()

output_df.iloc[:,output_df.columns.get_loc('p2_svpt')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_svpt')].copy()

#output_df.iloc[:,output_df.columns.get_loc('winner_number')]=df_shuffle.iloc[:,df.columns.get_loc('winner_id')].copy()

output_df.iloc[:,output_df.columns.get_loc('winner_number')]=winner_number
X_train, X_test, y_train, y_test = train_test_split(input_df[:],output_df[:],test_size=0.2) #20%

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
print(X_train)

print(y_train)
linear_model = LinearRegression()



m = linear_model.fit(X_train,y_train['winner_number'])
#calcul du RMSE

RMSE_train = np.sqrt(((y_train['winner_number'] - linear_model.predict(X_train))**2).sum()/len(y_train['winner_number']))

RMSE_test = np.sqrt(((y_test['winner_number'] - linear_model.predict(X_test))**2).sum()/len(y_test['winner_number']))



print("RMSE en apprentissage : ", RMSE_train)

print("RMSE en test : ", RMSE_test)
from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(solver='lbfgs', max_iter=5000)

logistic_model.fit(X_train, y_train['winner_number'])
RMSE_train = np.sqrt(((y_train['winner_number'] - logistic_model.predict(X_train))**2).sum()/len(y_train['winner_number']))

RMSE_test = np.sqrt(((y_test['winner_number'] - logistic_model.predict(X_test))**2).sum()/len(y_test['winner_number']))

print("RMSE en apprentissage : ", RMSE_train)

print("RMSE en test : ", RMSE_test)
model = Sequential()

X_param_nb = X_train.shape[1]

model.add(Dropout(0.1, input_shape = (X_param_nb, )))

model.add(Dense(X_param_nb, activation = 'relu'))

model.add(Dense(30, activation = 'relu'))

model.add(Dense(5, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(X_train, y_train['winner_number'], epochs = 20, validation_split = 0.2, batch_size = 256, shuffle=True)



plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training_set', 'Validation_set'])

plt.show()



#Affichage du taux de prédiction sur la base de test

print(model.evaluate(X_test, y_test['winner_number']))
input_df_2=pd.DataFrame(columns=['p1_1stWon', 'p1_2ndWon', 'p1_SvGms', 'p1_ace', 'p1_bpFaced', 'p1_bpSaved', 'p1_df', 'p1_svpt', 'p2_1stWon', 'p2_2ndWon', 'p2_SvGms', 'p2_ace', 'p2_bpFaced', 'p2_bpSaved', 'p2_df', 'p2_svpt'])



input_df_2.iloc[:,input_df_2.columns.get_loc('p1_1stWon')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_1stWon')].copy()

input_df_2.iloc[:,input_df_2.columns.get_loc('p2_1stWon')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_1stWon')].copy()

input_df_2.iloc[:,input_df_2.columns.get_loc('p1_2ndWon')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_2ndWon')].copy()

input_df_2.iloc[:,input_df_2.columns.get_loc('p2_2ndWon')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_2ndWon')].copy()

input_df_2.iloc[:,input_df_2.columns.get_loc('p1_SvGms')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_SvGms')].copy()

input_df_2.iloc[:,input_df_2.columns.get_loc('p2_SvGms')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_SvGms')].copy()

input_df_2.iloc[:,input_df_2.columns.get_loc('p1_ace')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_ace')].copy()

input_df_2.iloc[:,input_df_2.columns.get_loc('p2_ace')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_ace')].copy()

input_df_2.iloc[:,input_df_2.columns.get_loc('p1_bpFaced')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_bpFaced')].copy()

input_df_2.iloc[:,input_df_2.columns.get_loc('p2_bpFaced')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_bpFaced')].copy()

input_df_2.iloc[:,input_df_2.columns.get_loc('p1_bpSaved')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_bpSaved')].copy()

input_df_2.iloc[:,input_df_2.columns.get_loc('p2_bpSaved')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_bpSaved')].copy()

input_df_2.iloc[:,input_df_2.columns.get_loc('p1_df')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_df')].copy()

input_df_2.iloc[:,input_df_2.columns.get_loc('p2_df')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_df')].copy()

input_df_2.iloc[:,input_df_2.columns.get_loc('p1_svpt')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('l_svpt')].copy()

input_df_2.iloc[:,input_df_2.columns.get_loc('p2_svpt')]=df_shuffle.iloc[:,df_shuffle.columns.get_loc('w_svpt')].copy()



X_train, X_test, y_train, y_test = train_test_split(input_df_2[:],output_df[:],test_size=0.2) #20%
linear_model = LinearRegression()



m = linear_model.fit(X_train,y_train['winner_number'])



RMSE_train = np.sqrt(((y_train['winner_number'] - linear_model.predict(X_train))**2).sum()/len(y_train['winner_number']))

RMSE_test = np.sqrt(((y_test['winner_number'] - linear_model.predict(X_test))**2).sum()/len(y_test['winner_number']))



print("RMSE en apprentissage : ", RMSE_train)

print("RMSE en test : ", RMSE_test)
from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(solver='lbfgs', max_iter=5000)

logistic_model.fit(X_train, y_train['winner_number'])



RMSE_train = np.sqrt(((y_train['winner_number'] - logistic_model.predict(X_train))**2).sum()/len(y_train['winner_number']))

RMSE_test = np.sqrt(((y_test['winner_number'] - logistic_model.predict(X_test))**2).sum()/len(y_test['winner_number']))

print("RMSE en apprentissage : ", RMSE_train)

print("RMSE en test : ", RMSE_test)
from keras.callbacks import EarlyStopping

model = Sequential()

X_param_nb = X_train.shape[1]

es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=20)

model.add(Dropout(0.1, input_shape = (X_param_nb, )))

model.add(Dense(X_param_nb, activation = 'relu'))

model.add(Dense(30, activation = 'relu'))

model.add(Dense(1, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(X_train, y_train['winner_number'], epochs = 1000, validation_split = 0.2, batch_size = 256, shuffle=True, callbacks=[es])



plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Training_set', 'Validation_set'])

plt.show()



#Affichage du taux de prédiction sur la base de test

print("Loss value for test data : ", model.evaluate(X_test, y_test['winner_number'])[0])

print("Accuracy value for test data : ", model.evaluate(X_test, y_test['winner_number'])[1])