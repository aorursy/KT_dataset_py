import numpy as np 

import pandas as pd 

from sklearn.linear_model import LinearRegression  



df=pd.read_csv('/kaggle/input/exemplo-regresso-apostas/under.csv')



#Quais colunas que entraram na regressão

colunas='A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P'



#Fitra o df baseado nas colunas

df=df[(colunas+',PL').split(',')]





SLs=[] 

for i in range(100):

    #Embaralha o dataframe baseado no random_state i 

    df=df.sample(frac=1, random_state=i)

    

    #Divide em 100 mil linhas para teste e o restante treinamento

    df_test,df_train=df[:100000],df[100000:]



    #Os Xs são todas as colunas exceto a PL que será o Y

    X_train,Y_train = df_train.loc[:,(df_train.columns!='PL') ], df_train.PL

    X_test, Y_test  = df_test.loc[:,(df_test.columns!='PL') ], df_test.PL



    #Treina a regressão os dados de treinamento

    reg=LinearRegression().fit(X_train,Y_train)

    

    #Veifica a lucratividade nos dados de teste

    SLs+=[sum(np.log(1+y*y_pred) for y_pred,y in zip(reg.predict(X_test),Y_test) if y_pred>0 ) ]

    



#Mostra a lucrativida média e colunas selecionadas que deram origem a essa lucratividade

print( round(np.mean(SLs),2), colunas  )
