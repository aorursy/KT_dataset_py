# -*- coding: utf-8 -*-

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import LinearRegression  #Regressão Linear Simples

from sklearn.model_selection import train_test_split





#Função que recebe os Ys calculados pela regressão, que são os Percents, e os Ys reais que são os PL (perdas/lucros) de cada aposta

def SomaLog(Percent,PL):

    MIN_BANCA,MAX_BANCA= 0.01, 0.1

    return sum((MAX_BANCA if p>MAX_BANCA else p)*pl for p,pl in zip(Percent,PL) if p>=MIN_BANCA )







#Le o csv e filtra as colunas que serão utilizadas

df=pd.read_csv('../input/jogos_.csv')

df=df['s_g,s_c,s_da,s_s,s_r,d_g,d_c,d_da,d_s,goal,goal_diff,oddsU,probU,probU_diff,mod0,mod25,mod50,mod75,pl_u'.split(',')]



#Remove as linhas estão NaN, (preguiça de procurar, mas como é só um exemplo de uso tudo bem)

df.dropna(inplace=True)



#Mostra as 5 primeiras linhas

df.head(5)
somalogs=[]

for i in range(1000):

    df_train, df_test = train_test_split(df, test_size=50000)

    X_train, Y_train=df_train.loc[:,df_train.columns!='pl_u'], df_train.pl_u

    X_test, Y_test  =df_test.loc[:,df_test.columns!='pl_u'], df_test.pl_u

    

    somalogs+=[SomaLog(LinearRegression().fit(X_train,Y_train).predict(X_test), Y_test)]







print("SomaLog Médio:", np.array(somalogs).mean())

print("SomaLog Médio/mil jogos:", np.array(somalogs).mean()/(len(Y_test)/1000))
somalogs=[]

for i in range(1000):

    df_train, df_test = train_test_split(df, test_size=50000)

    

    #Filtra apenas os jogos empatados

    df_train=df_train[df_train.d_g==0]

    df_test=df_test[df_test.d_g==0]

    

    X_train, Y_train=df_train.loc[:,df_train.columns!='pl_u'], df_train.pl_u

    X_test, Y_test  =df_test.loc[:,df_test.columns!='pl_u'], df_test.pl_u

    

    somalogs+=[SomaLog(LinearRegression().fit(X_train,Y_train).predict(X_test), Y_test)]







print("SomaLog Médio:", np.array(somalogs).mean())

print("SomaLog Médio/mil jogos:", np.array(somalogs).mean()/(len(Y_test)/1000))
