import pandas as pd
import matplotlib.pyplot as plt
df = pd.DataFrame()
df = pd.read_csv('../input/anv.csv', delimiter=',')
print(df.columns)
df
resposta = [["codigo_ocorrencia", "Quantitativa Discreta"],["aeronave_matricula","Quantitativa Discreta"],["aeronave_tipo_veiculo","Qualitativa Nominal"]
           ,["aeronave_fabricante","Qualitativa Nominal"],["aeronave_modelo","Qualitativa Nominal"],["aeronave_voo_origem","Qualitativa Nominal"],["aeronave_voo_destino","Qualitativa Nominal"]
            ,["total_fatalidades", "Quantitativa Discreta"],["aeronave_nivel_dano", "Quantitativa Discreta"]] 
#variáveis relacionadas a tempo são contínuas, mas podem ser discretas pois não há perdas -- (discretização)
resposta = pd.DataFrame(resposta, columns=["Variavel", "Classificacao"])
resposta
df = df[['codigo_ocorrencia', 'aeronave_matricula', 'aeronave_tipo_veiculo', 'aeronave_fabricante', 'aeronave_modelo', 'aeronave_voo_origem', 'aeronave_voo_destino', 'total_fatalidades', 'aeronave_nivel_dano']].copy()
for i in range(int(resposta.Classificacao.count())):
    if 'Qualitativa' in resposta.Classificacao[i]:
        print(df[resposta.Variavel[i]].value_counts().sort_index())
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True)
#Tipo Veiculo
df_graf = df.aeronave_tipo_veiculo.value_counts()
df_graf = pd.DataFrame(df_graf)
df_graf = df_graf.reset_index()
df_graf.columns = ['Tipo', 'Numero']

data = [
    go.Bar(
        x=df_graf.Tipo,
        y=df_graf.Numero
    )
]
py.iplot(data)
#Fabricante
df_graf = df.aeronave_fabricante.value_counts()
df_graf = pd.DataFrame(df_graf)
df_graf = df_graf.reset_index()
df_graf.columns = ['Fabricante', 'Numero']

data = [
    go.Bar(
        x=df_graf.Fabricante,
        y=df_graf.Numero
    )
]
py.iplot(data)
#Modelo
df_graf = df.aeronave_modelo.value_counts()
df_graf = pd.DataFrame(df_graf)
df_graf = df_graf.reset_index()
df_graf.columns = ['Modelo', 'Numero']

data = [
    go.Bar(
        x=df_graf.Modelo,
        y=df_graf.Numero
    )
]
py.iplot(data)
#Origem
df_graf = df.aeronave_voo_origem.value_counts()
df_graf = pd.DataFrame(df_graf)
df_graf = df_graf.reset_index()
df_graf.columns = ['Origem', 'Numero']

data = [
    go.Pie(
        labels=df_graf.Origem, values=df_graf.Numero, hoverinfo='label+value+percent', textinfo='none'
    )
]
py.iplot(data)
#Destino
df_graf = df.aeronave_voo_destino.value_counts()
df_graf = pd.DataFrame(df_graf)
df_graf = df_graf.reset_index()
df_graf.columns = ['Destino', 'Numero']
data = [
    go.Pie(
        labels=df_graf.Destino, values=df_graf.Numero, hoverinfo='label+value+percent', textinfo='none'
    )
]
#Colocar a informacao do text deixa o grafico bem poluido. Portanto, deixei no hover.
py.iplot(data)
#Fatalidades
df_graf = df.total_fatalidades.value_counts()
df_graf = pd.DataFrame(df_graf)
df_graf = df_graf.reset_index()
df_graf.columns = ['Fatalidades', 'Numero']
data = [
    go.Histogram(
        x=df_graf.Fatalidades,
        y=df_graf.Numero,
        
    )
]
py.iplot(data)
#Nivel de Dano
df_graf = df.aeronave_nivel_dano.value_counts()
df_graf = pd.DataFrame(df_graf)
df_graf = df_graf.reset_index()
df_graf.columns = ['Dano', 'Numero']
data = [
    go.Bar(
        x=df_graf.Dano,
        y=df_graf.Numero
    )
]
py.iplot(data)
#extra - Fatalidades por Modelo
modelo = df.groupby(by='aeronave_modelo') 
tot_fat = modelo.total_fatalidades.sum().sort_index()
tot_fat_df = pd.DataFrame(tot_fat)
tot_fat_df = tot_fat_df.reset_index()
tot_fat_final = tot_fat_df[tot_fat_df['total_fatalidades'] != 0]
data = [
    go.Bar(
        x=tot_fat_final.aeronave_modelo,
        y=tot_fat_final.total_fatalidades
    )
]
py.iplot(data)
#fatalidades por tipo de aeronave
modelo = df.groupby(by='aeronave_tipo_veiculo') 
tot_fat = modelo.total_fatalidades.sum()
tot_fat_df = pd.DataFrame(tot_fat)
tot_fat_df = tot_fat_df.reset_index()
tot_fat_final = tot_fat_df[tot_fat_df['total_fatalidades'] != 0]
data = [
    go.Pie(labels=tot_fat_final.aeronave_tipo_veiculo, values=tot_fat_final.total_fatalidades, hoverinfo='percent', textinfo='label+value')
] 
py.iplot(data)