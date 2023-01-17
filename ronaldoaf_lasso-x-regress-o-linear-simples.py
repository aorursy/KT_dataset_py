# -*- coding: utf-8 -*-

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import Lasso    #Regressão Linear com L1

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
#Divide entre treinamento e teste, foi escolhi o random_state=881, pois nesse as médias de Y_train e  Y_test estão bem próximas

X_train, X_test, Y_train, Y_test = train_test_split(df.loc[:,df.columns!='pl_u'], df.pl_u, test_size=50000, random_state=881)



print('pl_medio_train:', Y_train.mean(), '\npl_medio_test: ', Y_test.mean())
#Treinamos a regressão com alpha=1e-5  ( ou seja, 0.00001)

reg1=Lasso(alpha=1e-5, max_iter=10000).fit(X_train,Y_train)



#Exibe a equação

print(  'pl_u= '+' + '.join("{0:.6}".format(reg1.coef_[i])+'*'+col for i,col in enumerate(X_train.columns)) +" + {0:.6}".format(reg1.intercept_)  )



#Exibe a Lucratividade através do SomaLog

print( '\nSomaLog: ', SomaLog( reg1.predict(X_test), Y_test)  )
#Treinamos a regressão com alpha=1e-4  ( ou seja, 0.0001)

reg2=LinearRegression().fit(X_train,Y_train)



#Exibe a equação

print(  'pl_u= '+' + '.join("{0:.6}".format(reg2.coef_[i])+'*'+col for i,col in enumerate(X_train.columns)) +" + {0:.6}".format(reg2.intercept_)  )



#Exibe a Lucratividade através do SomaLog

print( '\nSomaLog: ', SomaLog( reg2.predict(X_test), Y_test)  )