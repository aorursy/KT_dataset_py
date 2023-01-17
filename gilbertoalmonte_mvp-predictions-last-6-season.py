# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
cf.go_offline()
%matplotlib inline
nba = pd.read_excel('Seasons_stats_complete..xlsx')
nba.head(5)
desc = pd.read_excel('Descripción.xlsx')
desc
def info(campo):
    print(pd.DataFrame(desc[desc['Campo']==campo]['Descripción']))
info('ORB')
nba.describe().transpose().head(10)

nba[nba['Age']==0].sort_values(by='Year', ascending=False).head(5)
nba = nba.drop(nba[nba['Age']==0].index,axis=0)

nba.describe().transpose().head(5)
nba['ppg'] = nba['PTS']/nba['G']
nba['astg'] = nba['AST']/nba['G']
nba['trbg'] = nba['TRB']/nba['G']
nba['stlg'] = nba['STL']/nba['G']
nba['tovg'] = nba['TOV']/nba['G']
nba['blkg'] = nba['BLK']/nba['G']
nba_ = nba[nba['G']>=58]

nba_1980 = nba_[nba_['Year']>=1980]

nba_1980['Year'] = nba_1980['Year'].astype('str')

nba_1980['Nombre y Temporada'] = nba_1980['Name']+', '+nba_1980['Year']

Puntos = nba_1980.groupby('Year')['PTS'].sum()
Juegos = nba_1980.groupby('Year')['G'].mean()
MP = nba_1980.groupby('Year')['MP'].mean()

fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)

Puntos.plot(kind='line',ax=ax1,color='r',title='Evolución de los Puntos por Año')
Juegos.plot(kind='line',ax=ax2,color='r',title='Evolución de los Juegos Jugados por Año')
MP.plot(kind='line',ax=ax3,color='r',title='Evolución de los Minutos Jugados por Año')

plt.subplots_adjust(wspace=0.3,hspace=0.3)
plt.show()
_p2 = nba[nba['Year']>=1980].groupby('Year')['2PA'].sum()
_p3 = nba[nba['Year']>=1980].groupby('Year')['3PA'].sum()
_p2Porc = nba[nba['Year']>=1980].groupby('Year')['2P%'].mean()*100
_p3Porc = nba[nba['Year']>=1980].groupby('Year')['3P%'].mean()*100


fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)


_p2.plot(kind='bar',ax=ax1,color=sb.color_palette('Reds_d'),title='Evolución de la Cantidad de Intentos de 2P')
_p2Porc.plot(kind='line',ax=ax2,color=sb.color_palette('rocket'),title='Evolución de % encestados de tiros de 2P')
_p3.plot(kind='bar',ax=ax3,color=sb.color_palette('Blues_d'),title='Evolución de la Cantidad de Intentos de 3P')
_p3Porc.plot(kind='line',ax=ax4,color=sb.color_palette('Blues_d'),title='Evolución de % encestados de tiros de 3P')

plt.subplots_adjust(wspace=0.3,hspace=0.3)
plt.show()
Evol_2p = pd.DataFrame(nba[nba['Year']>=1980].groupby(['Year','Pos'])['2PA'].mean().reset_index()).sort_values(by=['Year','Pos'],ascending=True)
Evol_3p = pd.DataFrame(nba[nba['Year']>=1980].groupby(['Year','Pos'])['3PA'].mean().reset_index()).sort_values(by=['Year','Pos'],ascending=True)

fig = plt.figure(figsize=(20,15))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)

sb.lineplot(x='Year',y='2PA',data=Evol_2p,ax=ax1,hue='Pos',markers='o')
sb.lineplot(x='Year',y='3PA',data=Evol_3p,ax=ax2,hue='Pos',markers='o')

plt.suptitle('Evolución de los Tiros de 3 (derecha) y de 2 (izquierda) por posicion', fontsize=35)
plt.subplots_adjust(wspace=0.3,hspace=0.3)
plt.show()


nba_ = nba[nba['G']>=58]

puntos = nba_[nba_['Year']>=1980].groupby('Year')['ppg'].mean()
asistencias = nba_[nba_['Year']>=1980].groupby('Year')['astg'].mean()
rebotes = nba_[nba_['Year']>=1980].groupby('Year')['trbg'].mean()
fieldgoals = nba_[nba_['Year']>=1980].groupby('Year')['FG%'].mean()


fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

puntos.plot(kind='line',ax=ax1,color=sb.color_palette('cool'),title='Evolución de la Puntos por Juego por jugador')
asistencias.plot(kind='line',ax=ax2,color=sb.color_palette('cool'),title='Evolución de Asistencias por Juego por jugador')
rebotes.plot(kind='line',ax=ax3,color=sb.color_palette('cool'),title='Evolución de Rebotes por Juego por jugador')
fieldgoals.plot(kind='line',ax=ax4,color=sb.color_palette('cool'),title='Evolución de Field Goals % por Juego por jugador')

plt.subplots_adjust(wspace=0.3,hspace=0.3)
plt.show()


robos = nba_[nba_['Year']>=1980].groupby('Year')['stlg'].mean()
perdidas = nba_[nba_['Year']>=1980].groupby('Year')['tovg'].mean()
tapones = nba_[nba_['Year']>=1980].groupby('Year')['blkg'].mean()
PER = nba_[nba_['Year']>=1980].groupby('Year')['PER'].mean()


fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)


robos.plot(kind='line',ax=ax1,color=sb.color_palette('winter'),title='Evolución de robos por Juego')
perdidas.plot(kind='line',ax=ax2,color=sb.color_palette('winter'),title='Evolución de perdidas por Juego')
tapones.plot(kind='line',ax=ax3,color=sb.color_palette('winter'),title='Evolución de tapones por Juego')
PER.plot(kind='line',ax=ax4,color=sb.color_palette('winter'),title='Evolución de PER por jugadores')


plt.subplots_adjust(wspace=0.3,hspace=0.3)
plt.show()
dws = nba_[nba_['Year']>=1980].groupby('Year')['DWS'].mean()
dbpm = nba_[nba_['Year']>=1980].groupby('Year')['DBPM'].mean()


fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)


dws.plot(kind='line',ax=ax1,color=sb.color_palette('winter'),title='Evolución de Win Share Defensivos por Año por Jugador')
dbpm.plot(kind='line',ax=ax2,color=sb.color_palette('winter'),title='Evolución de Más Menos Defensivos por Año por Jugador')

plt.subplots_adjust(wspace=0.3,hspace=0.3)
plt.show()
nba_ = nba[nba['G']>=58]

nba_1980 = nba_[nba_['Year']>=1980]

nba_1980['Year'] = nba_1980['Year'].astype('str')

nba_1980['Nombre y Temporada'] = nba_1980['Name']+', '+nba_1980['Year']

TopPuntos = nba_1980.groupby(['Nombre y Temporada'])['ppg'].mean().reset_index().sort_values(by=['ppg'],ascending=False).head(10).reset_index().sort_values(by=['ppg'],ascending=True)
Topws = nba_1980.groupby(['Nombre y Temporada'])['WS'].mean().reset_index().sort_values(by=['WS'],ascending=False).head(10).reset_index().sort_values(by=['WS'],ascending=True)
Topvorp = nba_1980.groupby(['Nombre y Temporada'])['VORP'].mean().reset_index().sort_values(by=['VORP'],ascending=False).head(10).reset_index().sort_values(by=['VORP'],ascending=True)
TopBPM = nba_1980.groupby(['Nombre y Temporada'])['BPM'].mean().reset_index().sort_values(by=['BPM'],ascending=False).head(10).reset_index().sort_values(by=['BPM'],ascending=True)


fig = plt.figure(figsize=(20,10))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

TopPuntos.plot(x='Nombre y Temporada',y='ppg',kind='barh',ax=ax1,color='r',title='Top Mejores Temporadas por Promedio de Puntos en los últimos 40 años')
Topws.plot(x='Nombre y Temporada',y='WS',kind='barh',ax=ax2,color='b',title='Top Mejores Temporadas por Win Shares en los últimos 40 años')
Topvorp.plot(x='Nombre y Temporada',y='VORP',kind='barh',ax=ax3,color='y',title='Top Mejores Temporadas Valor Sobre Reemplazo en los últimos 40 años')
TopBPM.plot(x='Nombre y Temporada',y='BPM',kind='barh',ax=ax4,color='g',title='Top Mejores Mas y Menos en los últimos 40 años')

plt.subplots_adjust(wspace=0.4,hspace=0.3)
plt.show()
todos = nba_.groupby(['Name','MVP'])['ppg','astg','stlg','trbg','WS','FG%','VORP','BPM'].mean().reset_index().sort_values(by='MVP',ascending=False)

sb.pairplot(todos,hue='MVP')

plt.show()
import plotly.express as px

todo = nba_.groupby(['Name','MVP'])['ppg','astg','trbg'].mean().reset_index()

fig = px.scatter_3d(todo, x='ppg', y='astg', z='trbg',color='MVP')
fig.show()
todo2 = nba_.groupby(['Name','MVP'])['WS','VORP','PER'].mean().reset_index()

fig = px.scatter_3d(todo2, x='WS', y='VORP', z='PER',color='MVP')
fig.show()
todo = nba_[nba_['Year']>1980].groupby(['Name'])['WS','PER','VORP','BPM','DWS','OWS'].mean().reset_index().sort_values(by='WS',ascending=False).head(15)

fig = px.scatter(todo, x='OWS', y='DWS', text='Name',title='Top 15 mejores Win Shares')

fig.update_layout(
    autosize=False,
    width=800,
    height=800,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ))

fig.update_traces(textposition='top center')

fig.show()
todo = nba_[nba_['Year']>1980].groupby(['Name'])['WS','PER','VORP','BPM','DBPM','OBPM'].sum().reset_index().sort_values(by='BPM',ascending=False).head(15)

fig = px.scatter(todo, x='OBPM', y='DBPM', text='Name',title='Top 15 mejores Mas - Menos')

fig.update_layout(
    autosize=False,
    width=800,
    height=800,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ))

fig.update_traces(textposition='top center')

fig.show()
todo = nba_[nba_['Year']>1980].groupby(['Name'])['WS','PER','VORP','BPM','DBPM','OBPM','ppg'].mean().reset_index().sort_values(by='ppg',ascending=False).head(15)

fig = px.scatter(todo, x='ppg', y='VORP', text='Name',title='Top 15 mejores Puntos por juego con Valor sobre reemplazo')

fig.update_layout(
    autosize=False,
    width=800,
    height=800,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ))

fig.update_traces(textposition='middle right')

fig.show()
import plotly.express as px

todo = nba_.groupby(['Name'])['WS','PER','VORP','BPM'].mean().reset_index().sort_values(by='WS',ascending=False).head(10)

fig = px.scatter(todo, x='WS', y='PER', text='Name',title='Top 15 mejores Win Shares por juego con Ratio de Eficiencia por Jugador (PER)')


fig.update_traces(textposition='top center')

fig.show()
todo = nba_.groupby(['Name'])['WS','PER','VORP','BPM'].mean().reset_index().sort_values(by='VORP',ascending=False).head(10)

fig = px.scatter(todo, x='VORP', y='BPM', text='Name',title='Top 15 mejores Valor Sobre Reemplazo con Mas - Menos')

fig.update_layout(
    autosize=False,
    width=800,
    height=800,
    margin=dict(
        l=50,
        r=50,
        b=100,
        t=100,
        pad=4
    ))

fig.update_traces(textposition='top center')

fig.show()
from sklearn.naive_bayes import GaussianNB

years = range(2014, 2020)
mvp_years = dict()
resultado_mvp = pd.DataFrame(columns = ['Name','Year','MVP'])
nba_ = nba[nba['G']>=58]



for y in years:
    Xtrain = nba_[nba_.Year < y][['ppg','stlg','astg','trbg','WS','PER','VORP','BPM','WS/48']]
    Ytrain = nba_[nba_.Year < y]['MVP']
    Xtest = nba_[nba_.Year == y][['ppg','stlg','astg','trbg','WS','PER','VORP','BPM','WS/48']]
    test = nba_[nba_.Year == y][['Name','MVP']]
    
    gb = GaussianNB()
    gb.fit(Xtrain,Ytrain)
    pred_proba = gb.predict_proba(Xtest)
    
    y_pred_proba = []
    for i in enumerate(pred_proba):
        y_pred_proba.append(i[1][1])
    y_pred_proba = np.asarray(y_pred_proba)
    mvp_years = pd.DataFrame({'Name': test['Name'],
                             'Year': y,
                             'MVP': y_pred_proba,
                             'MVP_Real': test['MVP']})
    resultado_mvp = pd.concat([resultado_mvp,mvp_years], sort=True)

resultado_mvp = resultado_mvp.reset_index()
    
resultado_mvp = resultado_mvp.sort_values(by=['Year','MVP'], ascending=False).groupby('Year').head(1)

resultado_mvp = resultado_mvp.sort_values('Year', ascending=False).drop_duplicates()

resultado_mvp
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10)

rf.fit(Xtrain,Ytrain)

importancia = pd.DataFrame()
importancia['Variables'] = Xtrain.columns
importancia['Score'] = rf.feature_importances_

importancia.sort_values(by='Score',ascending=False).plot(x='Variables',y='Score',kind='bar')
Harden2017 = nba_[nba_['Year']==2017][nba_['Name']=='James Harden'][['Name','Year','ppg','stlg','astg','trbg','blkg','WS','PER','VORP','BPM','WS/48']]
Westbrook2017 = nba_[nba_['Year']==2017][nba_['Name']=='Russell Westbrook'][['Name','Year','ppg','stlg','astg','trbg','blkg','WS','PER','VORP','BPM','WS/48']]

Comparacion = Harden2017.append([Westbrook2017])

Comparacion.reset_index()
Comparacion2 = Comparacion[Comparacion['Name']=='James Harden'].drop(['Name','Year'],axis=1).reset_index() - Comparacion[Comparacion['Name']=='Russell Westbrook'].drop(['Name','Year'],axis=1).reset_index()
Comparacion2['Name'] = 'Ventaja de James Harden sobre Russell Westbrook'
Comparacion2.drop('index',axis=1).iplot(x='Name',kind='bar')
from sklearn.neighbors import KNeighborsClassifier

years = range(2014, 2020)
mvp_years = dict()
resultado_mvp = pd.DataFrame(columns = ['Name','Year','MVP'])
nba_ = nba[nba['G']>=58]


for y in years:
    Xtrain = nba_[nba_.Year < y][['ppg','stlg','astg','trbg','WS','PER','VORP','BPM','WS/48']]
    Ytrain = nba_[nba_.Year < y]['MVP']
    Xtest = nba_[nba_.Year == y][['ppg','stlg','astg','trbg','WS','PER','VORP','BPM','WS/48']]
    test = nba_[nba_.Year == y][['Name','MVP']]
    
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(Xtrain,Ytrain)
    pred_proba = knn.predict_proba(Xtest)
    
    y_pred_proba = []
    for i in enumerate(pred_proba):
        y_pred_proba.append(i[1][1])
    y_pred_proba = np.asarray(y_pred_proba)
    mvp_years = pd.DataFrame({'Name': test['Name'],
                             'Year': y,
                             'MVP': y_pred_proba,
                             'MVP_Real': test['MVP']})
    resultado_mvp = pd.concat([resultado_mvp,mvp_years], sort=True)

resultado_mvp = resultado_mvp.reset_index()
    
resultado_mvp = resultado_mvp.sort_values(by=['Year','MVP'], ascending=False).groupby('Year').head(1)

resultado_mvp = resultado_mvp.sort_values('Year', ascending=False).drop_duplicates()

resultado_mvp = resultado_mvp.drop(['index','MVP'],axis=1)
resultado_mvp['MVP_Real'] = resultado_mvp['MVP_Real'].astype('str')
resultado_mvp['MVP_Real'] = resultado_mvp['MVP_Real'].replace('1.0','Si')
resultado_mvp['MVP_Real'] = resultado_mvp['MVP_Real'].replace('0.0','No')
resultado_mvp
Curry2016 = nba_[nba_['Year']==2016][nba_['Name']=='Stephen Curry'][['Name','Year','ppg','stlg','astg','trbg','blkg','WS','PER','VORP','BPM','WS/48']]
Westbrook2016 = nba_[nba_['Year']==2016][nba_['Name']=='Russell Westbrook'][['Name','Year','ppg','stlg','astg','trbg','blkg','WS','PER','VORP','BPM','WS/48']]

Comparacion = Westbrook2016.append([Curry2016])

Comparacion.reset_index()
Comparacion2 = Comparacion[Comparacion['Name']=='Stephen Curry'].drop('Year',axis=1).reset_index() - Comparacion[Comparacion['Name']=='Russell Westbrook'].drop(['Name','Year'],axis=1).reset_index()
Comparacion2['Titulo'] = 'Diferencias'
Comparacion2.drop('index',axis=1).iplot(x=['Titulo'], kind='barh', title='Ventaja de Westbrook sobre Curry en la temporada 2016' )
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

years = range(2014, 2020)
mvp_years = dict()
resultado_mvp = pd.DataFrame(columns = ['Name','Year','MVP'])
nba_ = nba[nba['G']>=58]



for y in years:
    Xtrain = nba_[nba_.Year < y][['ppg','stlg','astg','trbg','WS','PER','VORP','BPM','WS/48']]
    Xtrain = sc.fit_transform(Xtrain)
    Ytrain = nba_[nba_.Year < y]['MVP']
    Xtest = nba_[nba_.Year == y][['ppg','stlg','astg','trbg','WS','PER','VORP','BPM','WS/48']]
    Xtest = sc.transform(Xtest)
    test = nba_[nba_.Year == y][['Name','MVP']]
    
    gb = GaussianNB()
    gb.fit(Xtrain,Ytrain)
    pred_proba = gb.predict_proba(Xtest)
    
    y_pred_proba = []
    for i in enumerate(pred_proba):
        y_pred_proba.append(i[1][1])
    y_pred_proba = np.asarray(y_pred_proba)
    mvp_years = pd.DataFrame({'Name': test['Name'],
                             'Year': y,
                             'MVP': y_pred_proba,
                             'MVP_Real': test['MVP']})
    resultado_mvp = pd.concat([resultado_mvp,mvp_years], sort=True)

resultado_mvp = resultado_mvp.reset_index()
    
resultado_mvp = resultado_mvp.sort_values(by=['Year','MVP'], ascending=False).groupby('Year').head(1)

resultado_mvp = resultado_mvp.sort_values('Year', ascending=False).drop_duplicates()

resultado_mvp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

years = range(2014, 2020)
mvp_years = dict()
resultado_mvp = pd.DataFrame(columns = ['Name','Year','MVP'])
nba_ = nba[nba['G']>=58]


for y in years:
    Xtrain = nba_[nba_.Year < y][['ppg','stlg','astg','trbg','WS','PER','VORP','BPM','WS/48']]
    Xtrain = sc.fit_transform(Xtrain)
    Ytrain = nba_[nba_.Year < y]['MVP']
    Xtest = nba_[nba_.Year == y][['ppg','stlg','astg','trbg','WS','PER','VORP','BPM','WS/48']]
    Xtest = sc.transform(Xtest)
    test = nba_[nba_.Year == y][['Name','MVP']]
    
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(Xtrain,Ytrain)
    pred_proba = knn.predict_proba(Xtest)
    
    y_pred_proba = []
    for i in enumerate(pred_proba):
        y_pred_proba.append(i[1][1])
    y_pred_proba = np.asarray(y_pred_proba)
    mvp_years = pd.DataFrame({'Name': test['Name'],
                             'Year': y,
                             'MVP': y_pred_proba,
                             'MVP_Real': test['MVP']})
    resultado_mvp = pd.concat([resultado_mvp,mvp_years], sort=True)

resultado_mvp = resultado_mvp.reset_index()
    
resultado_mvp = resultado_mvp.sort_values(by=['Year','MVP'], ascending=False).groupby('Year').head(1)

resultado_mvp = resultado_mvp.sort_values('Year', ascending=False).drop_duplicates()

resultado_mvp = resultado_mvp.drop(['index','MVP'],axis=1)
resultado_mvp['MVP_Real'] = resultado_mvp['MVP_Real'].astype('str')
resultado_mvp['MVP_Real'] = resultado_mvp['MVP_Real'].replace('1.0','Si')
resultado_mvp['MVP_Real'] = resultado_mvp['MVP_Real'].replace('0.0','No')
resultado_mvp

