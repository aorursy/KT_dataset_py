# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import io
import requests
import matplotlib.pylab as plt
import seaborn as sns
from datetime import timedelta  


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data_atualizacao_brasil='2020-06-07'
data_atualizacao='2020-06-08'

df_brasil = pd.read_csv("../input/corona-virus-brazil/brazil_covid19.csv")

governadores_brasil = pd.read_csv("../input/governadores2020/GOVERNADORES.csv", encoding = 'latin1', sep=';')

#df_brasil[df_brasil.state=='Distrito Federal'].tail().style.background_gradient(cmap='summer')


fp= df_brasil[['date', 'state', 'cases']].pivot_table(columns='date', index='state',values='cases').fillna(0)

novo_indice=df_brasil[(df_brasil['date']==data_atualizacao_brasil)][['cases', 'state']].sort_values(['cases'], 
                                                                                                    ascending=False)['state']

#sns.heatmap(fp.reindex(novo_indice), cmap='coolwarm')
#governadores_brasil

df_inner = pd.merge(df_brasil, governadores_brasil, left_on='state', right_on='Estado', how='inner')

df_brasil=df_inner.copy()



#df_brasil[df_brasil.state=='Distrito Federal'].tail().style.background_gradient(cmap='summer')

posicao_diaria=df_brasil[df_brasil.date==data_atualizacao_brasil].sort_values(['cases'], ascending=False)

total_casos=posicao_diaria.cases.sum()
total_mortes=posicao_diaria.deaths.sum()

posicao_diaria.style.background_gradient(cmap='summer')


posicao_diaria['casos_acumulados(%)']=round(posicao_diaria.cases.cumsum()/total_casos*100,2)
posicao_diaria['mortes_acumuladas(%)']=round(posicao_diaria.deaths.cumsum()/total_mortes*100,2)
posicao_diaria['sequencial']=range(1,posicao_diaria.cases.size+1)

posicao_diaria[['sequencial','date','state','cases','casos_acumulados(%)','deaths','mortes_acumuladas(%)']]
fp= df_brasil[['date', 'state', 'cases']].pivot_table(columns='date', index='state',values='cases').fillna(0) 
#sns.heatmap(fp, cmap='coolwarm')

fp.reindex(novo_indice)

#fp= df_brasil[['date', 'state', 'suspects']].pivot_table(columns='date', index='state',values='suspects').fillna(0) 
#sns.heatmap(fp, cmap='coolwarm')
#https://plotly.com/python/line-and-scatter/
import plotly.offline as py
import plotly.graph_objs as go

py.init_notebook_mode(connected=True)


trace = go.Scatter(x = df_brasil['date'],
                   y = df_brasil['cases'],
                   mode = 'markers')
# Armazenando gráfico em uma lista
data = [trace]
# Criando Layout
layout = go.Layout(title='Evolução de casos no Brasil',
                   yaxis={'title':'Nr de Casos Acumulados'},
                   xaxis={'title': 'Data de Apuração'})
# Criando figura que será exibida
fig = go.Figure(data=data, layout=layout)
# Exibindo figura/gráfico
py.iplot(fig)
#df_brasil[['state','cases','date']]
import plotly.express as px
df = df_brasil
fig = px.scatter(df[df.date>'2020-03-15'], x="date", y="cases", color="state",
                 size='deaths', hover_data=['region'])
fig.show()
fig = px.line(df[df.date>'2020-03-15'], x='date', y='deaths', color='state')
fig.show()

url="https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-{0}.xlsx".format(data_atualizacao)
c=pd.read_excel(url)
c.info()
c['acumulado_confirmados']=0
c['acumulado_mortos']=0
c['dobrou_confirmados']=0
c['dobrou_mortos']=0
c['primeiro_caso']=0
c['primeiro_morto']=0
c['dias_apos_primeiro_caso']=-1
c['dias_apos_primeiro_morto']=-1

c['GeoId']=c['geoId']
c['DateRep']=c['dateRep']
c['Deaths']=c['deaths']
c['Cases']=c['cases']
c['Countries and territories']=c['countriesAndTerritories']
paises=c.GeoId.unique()
datas=sorted(c.DateRep.unique())
temp=[]
contador_duplicacao=1
qtde_dias_duplicacao=0
casos=0
for i in datas:
    qtde_dias_duplicacao=qtde_dias_duplicacao+1
    casos=casos+c[(c.DateRep==i)].Cases.sum()
    if casos>=(contador_duplicacao*2):
        print("dobrou o nr de casos no mundo em {0} dias, para {2} casos, no dia {1}".format(qtde_dias_duplicacao, i, casos))
        contador_duplicacao=contador_duplicacao*2
        qtde_dias_duplicacao=0

print("Ultima posicao: {0}".format(casos))
temp=[]
contador_duplicacao=1
qtde_dias_duplicacao=0
casos=0
for i in datas:
    qtde_dias_duplicacao=qtde_dias_duplicacao+1
    casos=casos+c[(c.DateRep==i)].Deaths.sum()
    if casos>=(contador_duplicacao*2):
        print("dobrou o nr de mortos no mundo em {0} dias, para {2} casos, no dia {1}".format(qtde_dias_duplicacao, i, casos))
        contador_duplicacao=contador_duplicacao*2
        qtde_dias_duplicacao=0

print("Ultima posicao: {0}".format(casos))
for j in paises:
    historico=c[c.GeoId==j]
    for pais_ in historico['Countries and territories'].unique(): pais=pais_ 
    #print(j, pais)
    casos=0
    contador_duplicacao=1
    contador_ciclos=0
    qtde_dias_duplicacao=0
    comecou=0
    marcador=0
    contador_dias=0
    for i in sorted(historico.DateRep):
        novadata=i
        qtde_dias_duplicacao=qtde_dias_duplicacao+1
        casos=casos+historico[historico.DateRep==i].Cases.sum()
        
        condicao=[(c['Countries and territories']==pais) & (c.DateRep>=novadata)]
        resultado=[casos]
        c['acumulado_confirmados']=np.select(condicao, resultado, c['acumulado_confirmados'])
        
        
        if comecou==0:
            if casos>0:
                #print("Primeiro registro em {0}, {1} casos".format(i,casos))
                condicao=[(c['Countries and territories']==pais) & (c.DateRep==novadata)]
                resultado=[1]
                c['primeiro_caso']=np.select(condicao, resultado, 0)
                c['dobrou_confirmados']=np.select(condicao, [-1], c['dobrou_confirmados'])
                comecou=1
        
        elif casos>=(contador_duplicacao*2):
            #if contador_duplicacao>1: print("dobrou o nr de casos em {0} dias, foi para {2} casos, no dia {1}".format(qtde_dias_duplicacao, i, casos))
            contador_duplicacao=contador_duplicacao*2
            qtde_dias_duplicacao=0
            condicao=[(c['Countries and territories']==pais) & (c.DateRep==novadata)]
            contador_ciclos=contador_ciclos+1
            resultado=[contador_ciclos]
            c['dobrou_confirmados']=np.select(condicao, resultado, c['dobrou_confirmados'])
        
        if comecou==1:
            condicao=[(c['Countries and territories']==pais) & (c.DateRep==novadata)]
            c['dias_apos_primeiro_caso']=np.select(condicao, [contador_dias], c['dias_apos_primeiro_caso'])
            contador_dias=contador_dias+1

c[c['Countries and territories']=='Brazil'].head(10)
for j in paises:
    
    historico=c[c.GeoId==j]
    for pais_ in historico['Countries and territories'].unique(): pais=pais_ 
   
    #print(j, pais)
    casos=0
    contador_duplicacao=1
    contador_ciclos=0
    qtde_dias_duplicacao=0
    comecou=0
    marcador=0
    contador_dias=0
    for i in sorted(historico.DateRep):
        qtde_dias_duplicacao=qtde_dias_duplicacao+1
        casos=casos+historico[historico.DateRep==i].Deaths.sum()
        novadata=i
        condicao=[(c['Countries and territories']==pais) & (c.DateRep>=novadata)]
        resultado=[casos]
        c['acumulado_mortos']=np.select(condicao, resultado, c['acumulado_mortos'])
        #print (novadata)
        if comecou==0:
            if casos>0:
                #print("Primeiro registro em {0}, {1} mortes".format(i,casos))
                condicao=[(c['Countries and territories']==pais) & (c.DateRep==novadata)]
                resultado=[1]
                c['primeiro_morto']=np.select(condicao, resultado, 0)
                c['dobrou_mortos']=np.select(condicao, [-1], c['dobrou_mortos'])
                comecou=1      
        elif casos>=(contador_duplicacao*2):
           # if contador_duplicacao>1: print("dobrou o nr de mortos em {0} dias, foi para {2} mortos, no dia {1}".format(qtde_dias_duplicacao, i, casos))
            contador_duplicacao=contador_duplicacao*2
            qtde_dias_duplicacao=0
            
            condicao=[(c['Countries and territories']==pais) & (c.DateRep==novadata)]
            contador_ciclos=contador_ciclos+1
            resultado=[contador_ciclos]
            c['dobrou_mortos']=np.select(condicao, resultado, c['dobrou_mortos'])

        if comecou==1:
            condicao=[(c['Countries and territories']==pais) & (c.DateRep==novadata)]
            c['dias_apos_primeiro_morto']=np.select(condicao, [contador_dias], c['dias_apos_primeiro_morto'])
            contador_dias=contador_dias+1
         

large = 14; med = 12; small = 10
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 32),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")


#[c.CountryExp!='China']
coluna_estudada='dobrou_confirmados'
fp= c[['Countries and territories','dias_apos_primeiro_caso',coluna_estudada]].pivot_table(columns='dias_apos_primeiro_caso', index='Countries and territories',values=coluna_estudada).fillna(-1) 
#novo_indice=c[c[coluna_estudada]==-1][['DateRep', 'CountryExp']].sort_values(['DateRep'])['CountryExp']
novo_indice=c[c['DateRep']==data_atualizacao][['acumulado_confirmados', 'Countries and territories']].sort_values(['acumulado_confirmados'], ascending=False).head(30)['Countries and territories']
sns.heatmap(fp.reindex(novo_indice), cmap='coolwarm')

c[(c['DateRep']==data_atualizacao) & (c['acumulado_confirmados']>0)][['acumulado_confirmados', 'Countries and territories', 'cases', 'DateRep']].sort_values(['acumulado_confirmados'], ascending=False)[['Countries and territories','acumulado_confirmados', 'cases', 'DateRep']].head(30)

#[c.CountryExp!='China']
coluna_estudada='dobrou_mortos'
fp= c[['Countries and territories','dias_apos_primeiro_morto',coluna_estudada]].pivot_table(columns='dias_apos_primeiro_morto', index='Countries and territories',values=coluna_estudada).fillna(-1) 
#novo_indice=c[c[coluna_estudada]==-1][['DateRep', 'CountryExp']].sort_values(['DateRep'])['CountryExp']
novo_indice=c[(c['DateRep']==data_atualizacao) & (c['acumulado_mortos']>0)][['acumulado_mortos', 'Countries and territories']].sort_values(['acumulado_mortos'], ascending=False)['Countries and territories']
sns.heatmap(fp.reindex(novo_indice).head(30), cmap='coolwarm')


c[(c['DateRep']==data_atualizacao) & (c['acumulado_mortos']>0)][['acumulado_mortos', 'Countries and territories', 'deaths','DateRep']].sort_values(['acumulado_mortos'], ascending=False)[['Countries and territories','acumulado_mortos','deaths','DateRep']].head(30)

c[(c['Countries and territories']=='Mexico') & (c['acumulado_mortos']>0)][['acumulado_mortos', 'Countries and territories', 'deaths','DateRep']].sort_values(['acumulado_mortos'], ascending=False)[['Countries and territories','acumulado_mortos','deaths','DateRep']].head(30)
