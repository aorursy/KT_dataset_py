import gc

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing



from sklearn.model_selection import train_test_split

from sklearn import metrics



import folium

from folium import plugins

from folium.plugins import HeatMap

from folium.plugins import FastMarkerCluster

from folium.plugins import MarkerCluster

from catboost import Pool, CatBoostClassifier



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



from scipy.stats import chi2_contingency

from scipy.stats import chi2





plt.style.use('ggplot')



import warnings  

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")

sns.despine()



def plot_count(df, title, fig_size=(12,8), filename='plot', fontsize=8):

    p = (

        'Set2', 'tab20')

    color = sns.color_palette(np.random.choice(p), len(df))

    #color = sns.color_palette('Set2', len(ativos['Cargo']))

    bar   = df.value_counts().plot(kind='barh',

                    title=title,

                    fontsize=fontsize,

                    figsize=fig_size,

                    stacked=False,

                    width=1,

                    color=color,

    )



    bar.figure.savefig('%s.png' % filename)



    plt.show()
df_2017 = pd.read_csv("../input/prf20171819/datatran2017.csv", sep=";", encoding='cp1252')

df_2018 = pd.read_csv("../input/prf20171819/datatran2018.csv", sep=";", encoding='cp1252')



# merge

df = df_2017.append(df_2018)
del df_2017, df_2018

gc.collect()
df.head(10)
df.info()
df.isnull().sum()
df['br'] = df['br'].fillna(0.0)

df['km'] = df['km'].fillna(0)

# UOP nós não vamos usar na nossa análise, então basta descartar a coluna

df.drop(['uop'], axis=1, inplace=True)

df.isnull().sum()
def arruma_coordenada(x):

    if isinstance(x, str):

        return(np.float64(x.replace(",", ".").strip()))

    else:

        return x



df['latitude'] = df['latitude'].apply(lambda x: arruma_coordenada(x))

df['longitude'] = df['longitude'].apply(lambda x: arruma_coordenada(x))



df['tipo_acidente'] = df['tipo_acidente'].apply(lambda x: x.strip())

df['tipo_pista'] = df['tipo_pista'].apply(lambda x: x.strip())

df['fase_dia'] = df['fase_dia'].apply(lambda x: x.strip())

df['condicao_metereologica'] = df['condicao_metereologica'].apply(lambda x: x.strip())

df['municipio'] = df['municipio'].apply(lambda x: x.strip())

df['dia_semana'] = df['dia_semana'].apply(lambda x: x.strip())

df['causa_acidente'] = df['causa_acidente'].apply(lambda x: x.strip())
def chuva(x):

    return int('Chuva' in x) or int('Garoa/Chuvisco' in x)



def final_semana(x):

    return int('domingo' in x) or int('sábado' in x)



def noite(x):

    return int('Plena Noite' in x) or int('Anoitecer' in x)



def data_ano(x):

    return int(x.split("-")[0])



def data_mes(x):

    return int(x.split("-")[1])



def data_dia(x):

    return int(x.split("-")[2])



def hora(x):    

    hora = x.split(":")[0]    

    return int(hora)



def minuto(x):

    minuto = x.split(":")[1]

    return int(minuto)



def reta(x):

    return int('Reta' in x)



def curva(x):

    return int('Curva' in x)



df['data_ano'] = df['data_inversa'].apply(lambda x: data_ano(x))

df['data_mes'] = df['data_inversa'].apply(lambda x: data_mes(x))

df['data_dia'] = df['data_inversa'].apply(lambda x: data_dia(x))



df['noite'] = df['fase_dia'].apply(lambda x: noite(x))

df['fim_de_semana'] = df['dia_semana'].apply(lambda x: final_semana(x))

df['chovendo'] = df['condicao_metereologica'].apply(lambda x: chuva(x))

df['hora'] = df['horario'].apply(lambda x: hora(x))

df['minuto'] = df['horario'].apply(lambda x: minuto(x))



df['reta'] = df['tracado_via'].apply(lambda x: reta(x))

df['curva'] = df['tracado_via'].apply(lambda x: curva(x))
df.head(10)
fig = plt.figure(figsize=(20,15))

fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(2, 2, 1)

df[df.data_ano == 2017][['feridos_graves', 'feridos_leves', 'mortos']].boxplot(figsize=(12,8))

ax.tick_params(labelsize=12)

ax.set_title('2017')

ax = fig.add_subplot(2, 2, 2)

ax.tick_params(labelsize=12)

df[df.data_ano == 2018][['feridos_graves', 'feridos_leves', 'mortos']].boxplot(figsize=(12,8))

ax.set_title('2018')
df[(df.data_ano == 2017) & (df.mortos > 20)][['uf', 'br', 'km', 'municipio', 'causa_acidente', 'tipo_acidente', 'data_inversa', 'horario', 'reta', 'curva', 'mortos']]
df[['br']].mode()
plot_count(df['br'], 'Contagem de acidentes de acordo com a BR', fig_size=(12,22), filename='por_br', fontsize=12)
fig = plt.figure(figsize=(12,8))

fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(2, 2, 1)

sns.lineplot(x="data_mes", y="mortos", data=df[df.data_ano==2017], ax=ax)

ax.set_title('2017')



ax = fig.add_subplot(2, 2, 2)

sns.lineplot(x="data_mes", y="mortos", data=df[df.data_ano==2018], ax=ax)

ax.set_title('2018')
ct=pd.crosstab(df['chovendo'],df.tipo_acidente.str.contains("Colisão traseira").astype(int))

print(ct)
print(chi2_contingency(ct))
qui, p_valor, graus_liberdade, frequencias_esperadas = chi2_contingency(ct)



print("P-VALOR: ", p_valor)



if( p_valor < 0.05 ):

    print("Não aceita a hipótese nula por que {0} é menor do que 0.05".format(p_valor))

else:

    print("Aceita a hipótese nula")
plot_count(df['uf'], 'Por UF', fig_size=(20,18), filename='por_uf', fontsize=12)
plot_count(df[(df.uf == 'MG') & (df.br > 0)]['br'], 'Contagem de acidentes em MG de acordo com a BR', fig_size=(20,18), filename='por_br_MG', fontsize=12)
plot_count(df['tipo_acidente'], 'Por tipo de acidente', fig_size=(20,18), filename='tipo_acidente', fontsize=12)
plot_count(df['tipo_pista'], \

           'Acidentes gravíssimos por tipo de pista', fig_size=(10,8), filename='acidentes_gravissimos_tipo_pista', fontsize=12)
fig, ((axis1,axis2)) = plt.subplots(nrows=1, ncols=2)

fig.set_size_inches(22,7)



sns.countplot(data=df, x='data_mes', ax=axis1)

sns.countplot(data=df, x='data_dia', ax=axis2)

plt.show()
fig = plt.figure(figsize=(26,12))

fig.subplots_adjust(hspace=0.4, wspace=0.4)

ax = fig.add_subplot(2, 2, 1)

sns.countplot(data=df[(df.data_ano == 2017) & (df.causa_acidente.str.contains("Ingestão de Álcool"))], x='data_mes', palette="Set3", ax=ax)

ax.set_title('2017')



ax = fig.add_subplot(2, 2, 2)

sns.countplot(data=df[(df.data_ano == 2018) & (df.causa_acidente.str.contains("Ingestão de Álcool"))], x='data_mes', palette="Set2", ax=ax)

ax.set_title('2018')
print("Reta: ", len(df[df.reta == 1]))

print("Curva: ", len(df[df.curva == 1]))



fig = {

    'data': [{'labels': ['Reta', 'Curva'],

              'values': [len(df[df.reta == 1]), len(df[df.curva == 1])],

              'type': 'pie'}],

    'layout': {'title': 'Ocorrências em curvas X ocorrências em retas'}

     }



py.iplot(fig)

fig = {

    'data': [{'labels': ['Em retas e com chuva', 'Em retas e sem chuva'],

              'values': [len(df[(df.reta == 1) & (df.chovendo == 1)]), len(df[(df.reta == 1) & (df.chovendo == 0)])],

              'type': 'pie'}],

    'layout': {'title': 'Ocorrências em retas com chuva X ocorrências em retas sem chuva'}

     }



py.iplot(fig)

fig = {

    'data': [{'labels': ['Em retas e de noite', 'Em retas e de dia'],

              'values': [len(df[(df.reta == 1) & (df.noite == 1)]), len(df[(df.reta == 1) & (df.noite == 0)])],

              'type': 'pie'}],

    'layout': {'title': 'Ocorrências em retas e de noite X ocorrências em retas retas e durante o dia'}

     }



py.iplot(fig)
fig = {

    'data': [{'labels': ['Curva', 'Reta'],

              'values': [

                          len(df[(df.tipo_acidente.str.contains("Colisão")) & (df.curva == 1)]), 

                            len(df[(df.tipo_acidente.str.contains("Colisão")) & (df.reta == 1)])],

              'type': 'pie'}],

    'layout': {'title': 'Colisões em curvas X Colisões em retas'}

     }



py.iplot(fig)

fig = {

    'data': [{'labels': ['Curva', 'Reta'],

              'values': [

                          len(df[(df.tipo_acidente.str.contains("Capotamento")) & (df.curva == 1)]), 

                            len(df[(df.tipo_acidente.str.contains("Capotamento")) & (df.reta == 1)])],

              'type': 'pie'}],

    'layout': {'title': 'Capotamentos em curvas X Capotamentos em retas'}

     }



py.iplot(fig)

data=[]

for i in range(2017,2019+1):

    year=df[df['data_ano']==i]

    year_count=year['data_mes'].value_counts().reset_index().sort_values(by='index')

    year_count.columns=['data_mes','Count']

    

    trace = go.Scatter(

        x = year_count.data_mes,

        y = year_count.Count,

    name = i)

    data.append(trace)

    



py.iplot(data, filename='basic-line')
new=df[(df.tipo_acidente.str.contains("Colisão"))]

M= folium.Map(location=[df.latitude.median(), df.longitude.median()],tiles= "Stamen Terrain", zoom_start = 5) 



heat_data = [[[row['latitude'],row['longitude']] 

                for index, row in new.head(1000).iterrows()] 

                 for i in range(0,11)]



hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8)

hm.add_to(M)



hm.save('heatmap.html')



M
new=df[(df.tipo_acidente.str.contains("Capotamento"))]

M= folium.Map(location=[df.latitude.median(), df.longitude.median()],tiles= "Stamen Terrain", zoom_start = 5) 



heat_data = [[[row['latitude'],row['longitude']] 

                for index, row in new.head(1000).iterrows()] 

                 for i in range(0,11)]



hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8)

hm.add_to(M)



hm.save('heatmap.html')



M
fig = {

    'data': [{'labels': ['Quantidade de acidentes em finais de semana', 'Quantidade de acidentes em dias de semana'],

              'values': [

                          len(df[df.fim_de_semana == 1]), 

                            len(df[df.fim_de_semana == 0])],

              'type': 'pie'}],

    'layout': {'title': 'Fim de semana X Dias de semana'}

     }



py.iplot(fig)

plot_count(df['causa_acidente'], 'Maiores causas de acidentes', fig_size=(12,8), fontsize=12, filename='causas')
plot_count(df[df.causa_acidente.str.contains("Defeito na Via")]['uf'], 'Acidentes por defeito na via', fig_size=(15,8), filename='defeito_via', fontsize=12)
plot_count(df[df.causa_acidente.str.contains("Animais na Pista")]['uf'], \

           'Acidentes envolvendo animais na pista', fig_size=(15,8), filename='animais', fontsize=12)
plot_count(df[(df.causa_acidente.str.contains("Animais na Pista") & (df.uf == 'MG'))]['municipio'],\

           'Acidentes envolvendo animais na pista / MG', fig_size=(20,25), filename='animais', fontsize=10)
plot_count(df[df.noite == 1]['uf'], 'Acidentes durante o período da noite', fig_size=(10,8), filename='acidentes_noite', fontsize=12)
gravissimo = df[(df.mortos > 1) & (df.feridos_graves >= 1)]

grave = df[(df.mortos == 0) & (df.feridos_graves >= 1)]

leve = df[(df.mortos == 0) & (df.feridos_graves == 0) & (df.feridos_leves >= 1)]



gravissimo['categoria'] = 'gravissimo'

grave['categoria'] = 'grave'

leve['categoria'] = 'leve'

df = gravissimo.append(grave.append(leve))
del gravissimo, grave, leve

gc.collect()
df[['mortos', 'feridos_leves', 'feridos_graves', 'categoria', 'chovendo', 'curva', 'tipo_acidente']].head(10)
plot_count(df[df.categoria == 'gravissimo']['tipo_acidente'], 'Acidentes gravíssimos', fig_size=(10,5), filename='acidentes_gravissimos', fontsize=12)
plot_count(df[(df.categoria == 'gravissimo') & (df.tipo_acidente.str.contains('Colisão frontal'))]['uf'], \

           'Acidentes gravíssimos envolvendo colisão frontal por estado', fig_size=(10,8), filename='acidentes_gravissimos_colisao_frontal', fontsize=12)
plot_count(df[(df.categoria == 'gravissimo')]['tipo_pista'], \

           'Acidentes gravíssimos por tipo de pista', fig_size=(10,8), filename='acidentes_gravissimos_tipo_pista', fontsize=12)
def pista_simples(x):

    if 'Simples' in x:

        return 1

    else:

        return 0

    

def pista_dupla(x):

    if 'Dupla' in x:

        return 1

    else:

        return 0

    

df['pista_simples'] = df['tipo_pista'].apply(lambda x: pista_simples(x) )



df['pista_dupla'] = df['tipo_pista'].apply(lambda x: pista_dupla(x) )



X = df[['mortos', 'feridos_leves', 'feridos_graves', 'pista_simples', 'pista_dupla', 'noite', 'reta']].values
lbl = preprocessing.LabelEncoder()

lbl.fit(list(df[['categoria']].values))

Y = lbl.transform(list(df[['categoria']].values))
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

train_data = Pool(data=X_train,label=y_train)

val_data = Pool(data=X_test,label=y_test)





model = CatBoostClassifier(iterations=500,

                           learning_rate=0.2,

                           bagging_temperature=0.1,

                           l2_leaf_reg=30,

                           depth=12,

                           max_bin=255,

                           #max_leaves=48,

                           bootstrap_type='Bayesian',

                           random_seed=1337,

                           #task_type = 'GPU',

                           early_stopping_rounds=100,

                           loss_function='MultiClass')
model.fit(train_data)

preds_class = model.predict(val_data)

preds_proba = model.predict_proba(val_data)

preds_raw = model.predict(val_data, prediction_type='RawFormulaVal')
print ("\nprecision_score :",metrics.precision_score(y_test, preds_class, average='macro'))

print("\nf1_score: ", metrics.f1_score(y_test, preds_class, average='weighted')  )

print ("\nclassification report :\n",(metrics.classification_report(y_test, preds_class)))
df_2019 = pd.read_csv("../input/prf20171819/test_set_2019.csv")

df_2019.head(5)
df_2019.iloc[:, [0,1,2,3,4,5,6]].head(5)
X_df_2019 = df_2019.iloc[:, [0,1,2,3,4,5,6]].values

predictions = model.predict(X_df_2019)

predictions_prob = model.predict_proba(X_df_2019)



probs = []

for i in range(len(df_2019)):

    probs.append(predictions_prob[i][np.argmax(predictions_prob[i])])
df_2019['predicao_modelo'] = lbl.inverse_transform(predictions[:, 0].astype(int))

df_2019['pred_probabilidade'] = probs
df_2019.head(10)
df_2019.to_csv("predicoes.csv", index=False)