import numpy as np  

import pandas as pd 



df=pd.read_csv('/kaggle/input/exemplo-regresso-apostas/under.csv')

df
print('PL médio:',round(df.PL.mean()*100,2), '%')
from sklearn.preprocessing import MinMaxScaler



#Essa construção faz sentido se você for reprimir alguma coluna de input

colunas='A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P'.split(',')

df=df[colunas+['PL']] 



#Defina defini a escala a partir dos dados e faz a trasnformação dos inputs, mas não do PL

escala=MinMaxScaler().fit(df[colunas])

df[colunas]=escala.transform(df[colunas])



df
df_train=df[:int(len(df)*0.8)]

df_test=df[-int(len(df)*0.2):]



print('df_train:', len(df_train), 'rows')

print('df_test :', len(df_test), 'rows')
import torch



INPUTS=torch.from_numpy(df_train[colunas].values).float()

PL=torch.from_numpy(df_train[['PL']].values).float()



print('INPUTS: ',INPUTS)

print('PL: ',PL)
MODELO=torch.nn.Linear(INPUTS.shape[1], 1)



MODELO
loss_func = torch.nn.MSELoss()

otimizador = torch.optim.Adam(MODELO.parameters(), lr = 0.1) 

print('Função Custo:',loss_func)

print('Otimizador:',otimizador)
loss_list=[]

for _ in range(100):

    PL_pred=MODELO( INPUTS )        #calcula  PL_pred usando o modelo 

    loss=loss_func(PL_pred, PL)     #calcula o erro médio da estimativa do PL_pred 

    

    otimizador.zero_grad() #limpa gradientes antigos

    loss.backward()       #calcula a derivada do loss, através do retropropação

    otimizador.step()     #faz com que o otimizador dê um passo com base nos gradientes dos parâmetros. 

    

    loss_list+=[loss.item()] #acumula o valor para análise
import matplotlib.pyplot as plt



plt.plot(loss_list)

plt.show()
INPUTS_test=torch.from_numpy(df_test[colunas].values).float()  #Transforma em tensor o inputs de testes

PL_pred=MODELO(INPUTS_test).detach().numpy()  #Calcula os PL previstos para cada input de testes



#Calcula a média do PL reais das apostas cujo o PL previsto é positivo, ou seja lucrativas.

PL_medio_test=np.mean([pl for pl_pred,pl in zip(PL_pred, df_test.PL.values ) if pl_pred[0]>0])



print('PL_medio_test:',round(PL_medio_test*100,2),'%')
cortes=[i*0.01 for i in range(10)]

pl_medios=[np.mean([pl for pl_pred,pl in zip(PL_pred, df_test.PL.values ) if pl_pred[0]>i*0.01]) for i in range(10) ]



plt.plot(cortes, pl_medios)

plt.show()
#Otimiza para maximizar o crescimento da banca    

def loss_somalog(P_pred,PL):

    return -torch.log(1+PL*P_pred.relu()).sum()



loss_somalog
#Mesmo otimizador, mas com taxa de aprendizagem menor

otimizador = torch.optim.Adam(MODELO.parameters(), lr = 0.001) 



otimizador
loss_list=[]

for _ in range(2000):

    P_pred=MODELO( INPUTS )         #P_pred (percentual ótimo da banca ) usando o modelo 

    loss=loss_somalog(P_pred, PL)     #loss (quanto menor maior o lucro) dado P_pred 

    

    otimizador.zero_grad() #limpa gradientes antigos

    loss.backward()       #calcula a derivada do loss, através do retropropação

    otimizador.step()     #faz com que o otimizador dê um passo com base nos gradientes dos parâmetros. 

    

    loss_list+=[loss.item()] #acumula o valor para análise

    



#Plota o gráfico da evolução do loss

plt.plot(loss_list)

plt.show()
P_pred=MODELO(INPUTS_test).detach().numpy()



somalog=sum(np.log(1+pl*p_pred ) for p_pred,pl in zip(P_pred, df_test.PL.values) if p_pred>0)



print('Log do Crescimento da Banca do Cojunto de Teste:', somalog )
parms=[parm.data.numpy() for parm in MODELO.parameters()][0][0]

inter=[parm.data.numpy() for parm in MODELO.parameters()][1][0]



print('PERCENT_BANCA=')

for p,c in zip(parms/(escala.data_max_-escala.data_min_), colunas):

    print(round(p,5),'*',c,'+')

    

print(round(sum(-parms*escala.data_min_/(escala.data_max_-escala.data_min_))+inter,5))