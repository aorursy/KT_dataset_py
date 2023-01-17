# -*- coding: utf-8 -*-

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.linear_model import Lasso





#Le o csv e filtra as colunas que serão utilizadas

df=pd.read_csv('../input/jogos_.csv')

df=df['s_g,s_c,s_da,s_s,s_r,d_g,d_c,d_da,d_s,goal,goal_diff,oddsU,probU,probU_diff,mod0,mod25,mod50,mod75,pl_u'.split(',')]



#Remove as linhas estão NaN, (preguiça de procurar, mas como é só um exemplo de uso tudo bem)

df.dropna(inplace=True)



#Mostra as 5 primeiras linhas

df.head(5)
#Divide em X e Y

X=df.loc[:,df.columns!='pl_u']

Y=df.pl_u





#Treinamos a regressão com alpha=1e-4  ( ou seja, 0.0001)

reg1=Lasso(alpha=1e-4).fit(X,Y)



#Exibe a equação

print('\npl_u= '+' + '.join([ "{0:.9}".format(reg1.coef_[i])+'*'+df.columns[col_sel] for i,col_sel in enumerate(range(len(X.columns))) ])+" + {0:.9}".format(reg1.intercept_))

#Treinamos a regressão com alpha=1e-3  ( ou seja, 0.001)

reg2=Lasso(alpha=1e-3).fit(X,Y)



#Exibe a equação

print('\npl_u= '+' + '.join([ "{0:.9}".format(reg2.coef_[i])+'*'+df.columns[col_sel] for i,col_sel in enumerate(range(len(X.columns))) ])+" + {0:.9}".format(reg2.intercept_))
