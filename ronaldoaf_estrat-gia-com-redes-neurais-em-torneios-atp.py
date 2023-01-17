import pandas as pd

import numpy as np

from datetime import datetime



#Lê o arquivo CSV 

df=pd.read_csv('../input/ATP_matches_tennis-data.co.uk.csv',low_memory=False)



#Lê a data do formatp m/d/Y para o formato Y-m-d

df.Date = df.Date.apply(lambda x:datetime.strptime(x, '%m/%d/%Y')) 



#Remove as linhas que não possuam preenchidos os campos WPts, B365W, PSW

df= df[df.B365W.notnull() &  df.PSW.notnull()]   



df
df2=pd.DataFrame(columns=['Best_of_5','J1_B365','J2_B365','J1_PS','J2_PS','J1_W'])



# 1 se é melhor de 5 sets senão 0  

df2.Best_of_5 = np.where(df['Best of']==5, 1, 0)  



df2.J1_B365 = np.where(df.PSW<df.PSL, df.B365W, df.B365L) #Odds Bet365 para J1 vencer

df2.J2_B365 = np.where(df.PSW>df.PSL, df.B365W, df.B365L) #Odds Bet365 para J2 vencer



df2.J1_PS = np.where(df.PSW<df.PSL, df.PSW, df.PSL) #Odds Pinnacle para J1 vencer

df2.J2_PS = np.where(df.PSW>df.PSL, df.PSW, df.PSL) #Odds Pinnacle para J2 vencer





# 1 se J1 venceu, 0 se J2 venceu

df2.J1_W = np.where(df.PSW<df.PSL, 1, 0) 



#df2.pl1 = np.where(df.PSW<df.PSL, df2.J1_PS, 0)-1







df2
import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter

import numpy as np



medias={

    'Pinnacle_Favorito': np.where(df2.J1_W==1, df2.J1_PS-1, -1).mean(),

    'Pinnacle_Underdog': np.where(df2.J1_W==0, df2.J2_PS-1, -1).mean(),

    'Bet365_Favorito': np.where(df2.J1_W==1, df2.J1_B365-1, -1).mean(),

    'Bet365_Underdog': np.where(df2.J1_W==0, df2.J2_B365-1, -1).mean()

}



print(medias)





labels = [e for e in medias]

rois = [medias[e]*100 for e in medias]



index = np.arange(len(labels))



plt.figure(figsize=(12,6))

plt.bar(index, rois)

plt.ylabel('ROI')

plt.gca().yaxis.set_major_formatter(PercentFormatter())

plt.xticks(index, labels, fontsize=10, rotation=0)

plt.title('ROI Se Apostar em Todos os Jogos ')

for i in range(len(rois)): plt.text(i-0.1, rois[i]-0.35, "{0:.2%}".format(rois[i]/100) ) 

plt.show()

#Para cada coluna de odds tomamos o inverso

for coluna in ['J1_B365','J2_B365','J1_PS','J2_PS']: 

    df2[coluna]=1/df2[coluna]



#Exporta o DataFrame 

df2.to_csv('df2.csv')

    

df2

df_train=df2[:int(len(df2)*0.75)]

df_test=df2[int(len(df2)*0.75):]



print('Qtd de Registos')

print('train:', len(df_train) )

print('test : ', len(df_test) )
import torch

from torch.nn import Linear,Sigmoid,Sequential,CrossEntropyLoss, ReLU, Tanh

from torch import sum as sum_

from torch.nn.functional import relu as relu_



from torch import optim



#Tamanho da amostra aletória que será usada do treinamento de cada vez

batch_size=500



#D_in é a dimensão do nossos inputs X, a quantidade de atributos 

D_in=df_train.columns.size



#D_h é da camada oculta

D_h=10



#D_out é a dimensão da nossa saída Y, que será 1, apenas uma váriavel, a probabilidade de J1 vencer

D_out=1



#Definimos um modelo linear com apenas 2 camadas, entrada e saída, depois seguirá para um Sigmoid para retornar valores entre 0 e 1  

modelo=Sequential(Linear(D_in,D_out) )





#Inicia os parametros

with torch.no_grad():

    for param in modelo.parameters(): param.uniform_(0,1)





#Definimos otimizador no nosso modelo, com taxa de aprendizagem 0.001

otimizador=optim.Adam(modelo.parameters(), lr=0.001)





#Essa é função de custo, que o "Pulo do Gato", quanto menor maior será lucrativa sa estratégia

def customLoss(Y,Y_pred,O1,O2):

    return -sum_( (O1*Y-1)*relu_(O1*Y_pred-1)+ (O2*(1-Y)-1)*relu_(O2*(1-Y_pred)-1)   )

#def customLoss(PL,Y_pred):

#    return -sum_(PL*Y_pred   )    

pl=0

stake=0



loss_total=0

#Loop de treino, em cada interessaram mais treinado e preciso deverá ficar nosso modelo

for i in range(5000+1):

    

    #Seleciona um amostra aleatória do treinamento

    df_sample=df_train.sample(batch_size)

    

    #Cria o tensor X com os inputs do modelo serão todas as colunas exceto J1_W 

    #X=torch.from_numpy(df_sample.loc[:,(df_sample.columns!='J1_W') ].values).float()

    X=torch.from_numpy(df_sample.loc[:,(df_sample.columns!='pl1') ].values).float()

    

    

    #Cria o tensor Y com target do modelo a vitória ou derrota do favorito J1_W 

    Y=torch.from_numpy(df_sample['J1_W'].values).float()

    #Y=torch.from_numpy(df_sample['pl1'].values).float()

    

    #Cria os tensores O1 e O2 com as odds da Pinnacle para J1 e J2, será usado para verificar a lucratividade

    O1=1/torch.from_numpy(df_sample['J1_PS'].values).float()

    O2=1/torch.from_numpy(df_sample['J2_PS'].values).float()

    



    #Y previstos pelo modelo, serão as probabilidades previstas para J1 vencer

    Y_pred=modelo(X)



    #Função custo customizada que tenta maximizar a lucratividade

    loss=customLoss(Y,Y_pred,O1,O2)

    #loss=customLoss(PL,Y_pred)

    

    # Backward pass

    loss.backward()

    otimizador.step()        

    otimizador.zero_grad()

    

    



        

    with torch.no_grad():

        #pl+=( PL*Y_pred ).sum().item()

        #stake+=( Y_pred  ).sum().item()

        pl+=( (O1*Y-1)*relu_(O1*Y_pred-1) + (O2*(1-Y)-1)*relu_(O2*(1-Y_pred)-1)    ).sum().item()

        stake+=(relu_(O1*Y_pred-1)  + relu_(O2*(1-Y_pred)-1)  ).sum().item()

        #Cada 100 interações mostra a lucratividade da estratégia

        if i%100==0:

            if i==0: 

                print('Interação      ROI' )

            else:

                print(i,'         ',"{0:.2%}".format(pl/stake))    

            pl=0

            stake=0

        

  
