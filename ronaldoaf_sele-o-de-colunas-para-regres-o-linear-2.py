import numpy as np 

import pandas as pd 

from sklearn.linear_model import LinearRegression  



df=pd.read_csv('/kaggle/input/exemplo-regresso-apostas/under.csv')



#Criamos colunas novas baseadas nas originais "A a P"

df['A/D']=df.A/(df.D+1)

df['D1']=np.log(df.D+1)

df['D2']=np.log(df.D1+1)

df['D3']=np.log(df.D2+1)

df['J1']=np.log(df.J+1)





#Quais colunas que entraram na regressão

colunas='A,B,C,D,E,G,H,J,K,A/D,D1,D2,D3,J1'



#Fitra o df baseado nas colunas

df=df[(colunas+',PL').split(',')]





SLs=[] 

for i in range(100):

    #Embaralha o dataframe baseado no random_state i 

    df=df.sample(frac=1, random_state=i)

    

    #Divide em 100 mil linhas para teste e o restante treinamento

    df_test,df_train=df[:100000],df[100000:]



    #Filta o dataframe por intervalo de alguns campos que podem melhorar a regressão

    df_train=df_train[(df_train.E<=3)& (df_train.J>=1.25) & (df_train.J<=4)]

    df_test=df_test[(df_test.E<=3)& (df_test.J>=1.25) & (df_test.J<=4)]

    

    

    #Os Xs são todas as colunas exceto a PL que será o Y

    X_train,Y_train = df_train.loc[:,(df_train.columns!='PL') ], df_train.PL

    X_test, Y_test  = df_test.loc[:,(df_test.columns!='PL') ], df_test.PL



    #Treina a regressão os dados de treinamento

    reg=LinearRegression().fit(X_train,Y_train)

    

    #Veifica a lucratividade nos dados de teste

    SLs+=[sum(np.log(1+y*y_pred) for y_pred,y in zip(reg.predict(X_test),Y_test) if y_pred>0 ) ]

    



#Mostra a lucrativida média e colunas selecionadas que deram origem a essa lucratividade

print( round(np.mean(SLs),2), colunas  )
