import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import PercentFormatter





#Função para criar os gráficos de barras

def graficoDeBarras(valores,titulo):

    labels = [e for e in valores]

    rois = [valores[e]*100 for e in valores]



    index = np.arange(len(labels))



    plt.figure(figsize=(12,6))

    plt.bar(index, rois)

    plt.ylabel('ROI')

    plt.gca().yaxis.set_major_formatter(PercentFormatter())

    plt.xticks(index, labels, fontsize=10, rotation=0)

    plt.title(titulo)

    

    ajuste_labels=0.0248*min(rois) - 0.0272

    for i in range(len(rois)): plt.text(i-0.1, rois[i]+ajuste_labels, "{0:.2%}".format(rois[i]/100) ) 

    plt.show()







#Lê o arquivo CSV 

df=pd.read_csv('../input/ATP_matches_tennis-data.co.uk.csv',low_memory=False)





#Remove as linhas que não possuam preenchidos os campos B365W, PSW e que o Piso é Carpete

df= df[df.B365W.notnull() &  df.PSW.notnull() & (df.Surface!='Carpet') ]   



df2=pd.DataFrame(columns=['Hard','Clay','Grass','Indoor','odds1','odds2','J1_W', 'pl1', 'pl2'])





df2.Hard  = np.where(df.Surface=='Hard', 1, 0)

df2.Clay  = np.where(df.Surface=='Clay', 1, 0)

df2.Grass  = np.where(df.Surface=='Grass', 1, 0)

df2.Indoor = np.where(df.Court=='Indoor', 1, 0) 



#Odds Pinnacle para J1 e J2

df2.odds1 = np.where(df.PSW<df.PSL, df.PSW, df.PSL)

df2.odds2 = np.where(df.PSW>df.PSL, df.PSW, df.PSL) 





# 1 se J1 venceu, 0 se J2 venceu

df2.J1_W = np.where(df.PSW<df.PSL, 1, 0) 



#Lucro se tivessem apostado no J1 e no J2

df2.pl1=np.where(df2.J1_W==1, df2.odds1, 0)-1

df2.pl2=np.where(df2.J1_W==0, df2.odds2, 0)-1



df=df2



df

medias={

    'Favorito': df.pl1.mean(),

    'Underdog': df.pl2.mean()

}



print(medias)

graficoDeBarras(medias,'ROI Se Apostar em Todos os Jogos')
medias={

    'Hard': df[df.Hard==1].pl1.mean(),

    'Clay': df[df.Clay==1].pl1.mean(),

    'Grass':df[df.Grass==1].pl1.mean()

}



print(medias)

graficoDeBarras(medias,'ROI do Favorito por Piso')
medias={

    'Indoor': df[df.Indoor==1].pl1.mean(),

    'Outdoor':df[df.Indoor==0].pl1.mean()

}



print(medias)

graficoDeBarras(medias,'ROI do Favorito Indoor e Outdoor')
print('Indoor: ',"{0:.1%}".format(len(df[df.Indoor==1])/len(df)) )

print('Outdoor:',"{0:.1%}".format(len(df[df.Indoor==0])/len(df)) )
medias={

    'Geral': df.pl1.mean(),

    'Outdoor e Não Hard':df[(df.Indoor==0) &  (df.Hard==0)].pl1.mean()

}



print(medias)

graficoDeBarras(medias,'ROI do Favorito Jogando Outdoor em piso que não seja Hard')
medias={}

for percent in [0,25,50,75]:

    odds_percent_i, odds_percent_f = np.percentile(df.odds1,percent), np.percentile(df.odds1,percent+25)

    medias[str( odds_percent_i)+' <--> '+ str( odds_percent_f)]=df[(df.odds1>=odds_percent_i) & (df.odds1<odds_percent_f)].pl1.mean()



print(medias)

graficoDeBarras(medias,'ROI do Favorito por Intervalo de Odds')