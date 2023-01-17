import pandas as pd
import numpy as np
import unicodedata as ud
import matplotlib.pyplot as plt
import matplotlib
import math
from difflib import SequenceMatcher as sm
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
df = pd.read_csv('../input/events.csv', low_memory = False)
#df.head()
#df.info(memory_usage = 'deep')
#df.describe()
df.sort_values(by=['timestamp']).sort_values(by=['person'], inplace = True)
df['search_term'] = df['search_term'].str.lower()
df['model'] = df['model'].str.lower()
df['search_term'] = df['search_term'].str.replace('!','').str.replace('?','').str.replace('¡','').str.replace('¿','').str.replace('+','').str.replace('-','').str.replace('*','').str.replace('%','').str.replace(',','').str.replace(';','').str.replace('.','').str.replace(':','').str.replace('_','').str.replace('\'','').str.replace('\"','').str.replace('\\','').str.replace('/','').str.replace('(','').str.replace(')','').str.replace('{','').str.replace('}','').str.replace('[','').str.replace(']','').str.replace('@','').str.replace('º','').str.replace('&','')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['timediff'] = df['timestamp'].diff()
df['timediff'] = df['timediff'].dt.total_seconds()

df['newperson']= df['person'].apply(lambda x: int(x, 16))
df['newperson'] = df['newperson'].diff()
df['newperson'] = ((df['newperson'] > 0) | (df.index == 0))

maxtime = 16368
df['newentry'] = ((df['timediff'] > maxtime) | (df['newperson']))
#df['entry'] = df['newentry'].cumsum()
df['newsearch'] = ((df['event'] == 'searched products') | (df['newentry']))
df['validsearch'] = ((df['event'] == 'searched products') & (df['search_term']))
df['search'] = df['newsearch'].cumsum()
df = df.groupby('search').filter(lambda x: x.iloc[0]['validsearch'])

df2 = df

def contenido(s,x):
    return x.find(s) >= 0

def obtener_marca(x):
    if (type(x) == float):
        return np.nan
    elif (contenido('asus',x) or contenido('live',x) or contenido('zenfone',x)):
        return 'asus'
    elif (contenido('blackberry',x)):
        return 'blackberry'
    elif (contenido('lg',x)):
        return 'lg'
    elif (contenido('lenovo',x) or contenido('vibe',x)):
        return 'lenovo'
    elif (contenido('moto',x)):
        return 'motorola'
    elif (contenido('quantum',x)):
        return 'quantum'
    elif (contenido('samsung',x) or contenido('galaxy',x)):
        return 'samsung'
    elif (contenido('sony',x) or contenido('xperia',x)):
        return 'sony'
    elif (contenido('apple',x) or contenido('iphone',x) or contenido('ipad',x)):
        return 'apple'
    else:
        return np.nan

def obtener_marca_de_busqueda(x):
    return obtener_marca(x.iloc[0])

dfg = df.groupby('search', as_index=False)
df2['busquedas'] = dfg['search_term'].transform(obtener_marca_de_busqueda)
df2['nmodel'] = df2.apply(lambda x: obtener_marca(x['model']), axis=1)

df2['bcm'] = ((df2['event'] == 'searched products') & (df2['busquedas'])).astype(int)
df2['vam'] = ((df2['event'] == 'viewed product') & (df2['busquedas']) & (df2['busquedas'] == df2['nmodel'])).astype(int)
df2['vao'] = ((df2['event'] == 'viewed product') & (df2['busquedas']) & (df2['busquedas'] != df2['nmodel'])).astype(int)
df2['coam'] = ((df2['event'] == 'checkout') & (df2['busquedas']) & (df2['busquedas'] == df2['nmodel'])).astype(int)
df2['coao'] = ((df2['event'] == 'checkout') & (df2['busquedas']) & (df2['busquedas'] != df2['nmodel'])).astype(int)
df2['cam'] = ((df2['event'] == 'conversion') & (df2['busquedas']) & (df2['busquedas'] == df2['nmodel'])).astype(int)
df2['cao'] = ((df2['event'] == 'conversion') & (df2['busquedas']) & (df2['busquedas'] != df2['nmodel'])).astype(int)
df2['lam'] = ((df2['event'] == 'lead') & (df2['busquedas']) & (df2['busquedas'] == df2['nmodel'])).astype(int)
df2['lao'] = ((df2['event'] == 'lead') & (df2['busquedas']) & (df2['busquedas'] != df2['nmodel'])).astype(int)

print(df2.loc[:1000, ['search', 'event', 'search_term', 'model', 'nmodel', 'busquedas', 'bcm', 'vam', 'vao', 'coam', 'coao', 'cam', 'cao', 'lam', 'lao']])
df3 = df2.loc[:, ['busquedas', 'bcm', 'vam', 'vao', 'coam', 'coao', 'cam', 'cao', 'lam', 'lao']]
df3 = df3.set_index('busquedas')
df3.index.names = ['marcas']
df3 = df3.astype(float)
df3 = df3.groupby(level=0).sum()
df3
colores_visitas = ['#cf8ccc','#af6cac']
colores_cheackouts = ['#ffdc8c','#dfbc6c']
colores_conversion = ['#96deb4','#66ae84']
colores_lead = ['#de7674','#ae4644']
bcm = 'Busquedas para esta marca'
vam = 'Visitas a esta marca'
vao = 'Visitas a otra marca'
coam = 'Checkouts a esta marca'
coao = 'Checkouts a otra marca'
cam = 'Conversiones a esta marca'
cao = 'Conversiones a otra marca'
lam = 'Leads a esta marca'
lao = 'Leads a otra marca'
df3.columns = [bcm, vam, vao, coam, coao, cam, cao, lam, lao]
fig, ax = plt.subplots()
df3.loc[:,[bcm]].plot.bar(stacked=True, color='#6cacff', width=0.1, position=0, ax=ax)
df3.loc[:,[vam, vao]].plot.bar(stacked=True, color=colores_visitas, width=0.1, position=-1, ax=ax)
df3.loc[:,[coam, coao]].plot.bar(stacked=True, color=colores_cheackouts, width=0.1, position=-2, ax=ax)
df3.loc[:,[cam, cao]].plot.bar(stacked=True, color=colores_conversion, width=0.1, position=-3, ax=ax)
df3.loc[:,[lam, lao]].plot.bar(stacked=True, color=colores_lead, width=0.1, position=-4, ax=ax)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
plt.title("Relacion entre busquedas por marca y visitas/checkouts/conversiones/leads posteriores", fontsize=18)
plt.xlabel("Marcas", fontsize=18)
plt.ylabel("Cantidad", fontsize=18)
plt.yscale('log')
plt.xticks(fontsize=14, rotation=25)
plt.yticks(fontsize=14, rotation=25)
fig = plt.gcf()
fig.set_size_inches(16.0, 8.0)
plt.legend(loc=9, bbox_to_anchor=(0.95, 1.0))
plt.show()
fig, ax = plt.subplots()
df3.loc[:,[bcm]].plot.bar(stacked=True, color='#6cacff', width=0.1, position=0, ax=ax)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
plt.xlabel("Marcas", fontsize=18)
plt.ylabel("Cantidad", fontsize=18)
plt.yscale('linear')
plt.xticks(fontsize=14, rotation=25)
plt.yticks(fontsize=14, rotation=25)
fig = plt.gcf()
fig.set_size_inches(12.0, 5.0)
plt.legend(loc=9, bbox_to_anchor=(0.85, 1.0))
plt.title("Comparacion de busquedas de marcas", fontsize=18)
plt.show()
fig, ax = plt.subplots()
df3.loc[:,[vam, vao]].plot.bar(stacked=True, color=colores_visitas, width=0.1, position=0, ax=ax)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
plt.xlabel("Marcas", fontsize=18)
plt.ylabel("Cantidad", fontsize=18)
plt.yscale('linear')
plt.xticks(fontsize=14, rotation=25)
plt.yticks(fontsize=14, rotation=25)
fig = plt.gcf()
fig.set_size_inches(12.0, 5.0)
plt.legend(loc=9, bbox_to_anchor=(0.85, 1.0))
plt.title("Comparacion de visitas tras buscar una marca", fontsize=18)
plt.show()
fig, ax = plt.subplots()
df3.loc[:,[coam, coao]].plot.bar(stacked=True, color=colores_cheackouts, width=0.1, position=0, ax=ax)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
plt.xlabel("Marcas", fontsize=18)
plt.ylabel("Cantidad", fontsize=18)
plt.yscale('linear')
plt.xticks(fontsize=14, rotation=25)
plt.yticks(fontsize=14, rotation=25)
fig = plt.gcf()
fig.set_size_inches(12.0, 5.0)
plt.legend(loc=9, bbox_to_anchor=(0.85, 1.0))
plt.title("Comparacion de checkouts tras buscar una marca", fontsize=18)
plt.show()
fig, ax = plt.subplots()
df3.loc[:,[cam, cao]].plot.bar(stacked=True, color=colores_conversion, width=0.1, position=0, ax=ax)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
plt.xlabel("Marcas", fontsize=18)
plt.ylabel("Cantidad", fontsize=18)
plt.yscale('linear')
plt.xticks(fontsize=14, rotation=25)
plt.yticks(fontsize=14, rotation=25)
fig = plt.gcf()
fig.set_size_inches(12.0, 5.0)
plt.legend(loc=9, bbox_to_anchor=(0.85, 1.0))
plt.title("Comparacion de conversiones tras buscar una marca", fontsize=18)
plt.show()
fig, ax = plt.subplots()
df3.loc[:,[lam, lao]].plot.bar(stacked=True, color=colores_lead, width=0.1, position=0, ax=ax)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
plt.xlabel("Marcas", fontsize=18)
plt.ylabel("Cantidad", fontsize=18)
plt.yscale('linear')
plt.xticks(fontsize=14, rotation=25)
plt.yticks(fontsize=14, rotation=25)
fig = plt.gcf()
fig.set_size_inches(12.0, 5.0)
plt.legend(loc=9, bbox_to_anchor=(0.85, 1.0))
plt.title("Comparacion de leads tras buscar una marca", fontsize=18)
plt.show()
def son_similares(palabra, marca, mejor_ratio, r):
    r[0] = sm(None, palabra, marca).ratio() 
    return r[0] > mejor_ratio and r[0] < 1.0

def obtener_marca_de_similar(x):
    if (type(x) == float):
        return np.nan
    mejor_ratio = 0.7
    mejor_palabra = np.nan
    r = [0.0]
    for palabra in x.split():
        palabra = unicode(palabra, 'utf-8')
        palabra = ud.normalize('NFKD', palabra).encode('ASCII', 'ignore')
        palabra = palabra.decode("utf-8")
        if (son_similares(palabra, 'asus', mejor_ratio, r)):
            mejor_ratio = r[0]
            mejor_palabra = 'asus'
        if (son_similares(palabra, 'live', mejor_ratio, r)):
            mejor_ratio = r[0]
            mejor_palabra = 'asus'
        if (son_similares(palabra, 'zenfone', mejor_ratio, r)):
            mejor_ratio = r[0]
            mejor_palabra = 'asus'
        if (son_similares(palabra, 'blackberry', mejor_ratio, r)):
            mejor_ratio = r[0]
            mejor_palabra = 'blackberry'
        if (son_similares(palabra, 'lg', mejor_ratio, r)):
            mejor_ratio = r[0]
            mejor_palabra = 'lg'
        if (palabra != 'novo' and palabra != 'novos' and son_similares(palabra, 'lenovo', mejor_ratio, r)):
            mejor_ratio = r[0]
            mejor_palabra = 'lenovo'
        if (son_similares(palabra, 'vibe', mejor_ratio, r)):
            mejor_ratio = r[0]
            mejor_palabra = 'lenovo'
        if (son_similares(palabra, 'motorola', mejor_ratio, r)):
            mejor_ratio = r[0]
            mejor_palabra = 'motorola'
        if (son_similares(palabra, 'moto', mejor_ratio, r)):
            mejor_ratio = r[0]
            mejor_palabra = 'motorola'
        if (son_similares(palabra, 'quantum', mejor_ratio, r)):
            mejor_ratio = r[0]
            mejor_palabra = 'quantum'
        if (son_similares(palabra, 'samsung', mejor_ratio, r)):
            mejor_ratio = r[0]
            mejor_palabra = 'samsung'
        if (son_similares(palabra, 'galaxy', mejor_ratio, r)):
            mejor_ratio = r[0]
            mejor_palabra = 'samsung'
        if (son_similares(palabra, 'sony', mejor_ratio, r)):
            mejor_ratio = r[0]
            mejor_palabra = 'sony'
        if (son_similares(palabra, 'xperia', mejor_ratio, r)):
            mejor_ratio = r[0]
            mejor_palabra = 'sony'
        if (son_similares(palabra, 'apple', mejor_ratio, r)):
            mejor_ratio = r[0]
            mejor_palabra = 'apple'
        if (son_similares(palabra, 'iphone', mejor_ratio, r)):
            mejor_ratio = r[0]
            mejor_palabra = 'apple'
        if (son_similares(palabra, 'ipad', mejor_ratio, r)):
            mejor_ratio = r[0]
            mejor_palabra = 'apple'
    return mejor_palabra

def obtener_marca_de_busqueda(x):
    return obtener_marca_de_similar(x.iloc[0])

dfg = df.groupby('search', as_index=False)
df2['busquedas'] = dfg['search_term'].transform(obtener_marca_de_busqueda)
df2['nmodel'] = df2.apply(lambda x: obtener_marca(x['model']), axis=1)

df2['bcm'] = ((df2['event'] == 'searched products') & (df2['busquedas'])).astype(int)
df2['vam'] = ((df2['event'] == 'viewed product') & (df2['busquedas']) & (df2['busquedas'] == df2['nmodel'])).astype(int)
df2['vao'] = ((df2['event'] == 'viewed product') & (df2['busquedas']) & (df2['busquedas'] != df2['nmodel'])).astype(int)
df2['coam'] = ((df2['event'] == 'checkout') & (df2['busquedas']) & (df2['busquedas'] == df2['nmodel'])).astype(int)
df2['coao'] = ((df2['event'] == 'checkout') & (df2['busquedas']) & (df2['busquedas'] != df2['nmodel'])).astype(int)
df2['cam'] = ((df2['event'] == 'conversion') & (df2['busquedas']) & (df2['busquedas'] == df2['nmodel'])).astype(int)
df2['cao'] = ((df2['event'] == 'conversion') & (df2['busquedas']) & (df2['busquedas'] != df2['nmodel'])).astype(int)
df2['lam'] = ((df2['event'] == 'lead') & (df2['busquedas']) & (df2['busquedas'] == df2['nmodel'])).astype(int)
df2['lao'] = ((df2['event'] == 'lead') & (df2['busquedas']) & (df2['busquedas'] != df2['nmodel'])).astype(int)

print(df2.loc[1000:2000, ['search', 'event', 'search_term', 'model', 'nmodel', 'busquedas', 'bcm', 'vam', 'vao', 'coam', 'coao', 'cam', 'cao', 'lam', 'lao']])
df3 = df2.loc[:, ['busquedas', 'bcm', 'vam', 'vao', 'coam', 'coao', 'cam', 'cao', 'lam', 'lao']]
df3 = df3.set_index('busquedas')
df3.index.names = ['marcas']
df3 = df3.astype(float)
df3 = df3.groupby(level=0).sum()
df3
df3.columns = [bcm, vam, vao, coam, coao, cam, cao, lam, lao]
fig, ax = plt.subplots()
df3.loc[:,[bcm]].plot.bar(stacked=True, color='#6cacff', width=0.1, position=0, ax=ax)
df3.loc[:,[vam, vao]].plot.bar(stacked=True, color=colores_visitas, width=0.1, position=-1, ax=ax)
df3.loc[:,[coam, coao]].plot.bar(stacked=True, color=colores_cheackouts, width=0.1, position=-2, ax=ax)
df3.loc[:,[cam, cao]].plot.bar(stacked=True, color=colores_conversion, width=0.1, position=-3, ax=ax)
df3.loc[:,[lam, lao]].plot.bar(stacked=True, color=colores_lead, width=0.1, position=-4, ax=ax)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
plt.xlabel("Marcas", fontsize=18)
plt.ylabel("Cantidad", fontsize=18)
plt.yscale('log')
plt.xticks(fontsize=14, rotation=25)
plt.yticks(fontsize=14, rotation=25)
fig = plt.gcf()
fig.set_size_inches(16.0, 8.0)
plt.legend(loc=9, bbox_to_anchor=(0.95, 1.0))
plt.title("Relacion entre busquedas por marca y visitas/checkouts/conversiones/leads posteriores", fontsize=18)
plt.show()
fig, ax = plt.subplots()
df3.loc[:,[bcm]].plot.bar(stacked=True, color='#6cacff', width=0.1, position=0, ax=ax)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
plt.xlabel("Marcas", fontsize=18)
plt.ylabel("Cantidad", fontsize=18)
plt.yscale('linear')
plt.xticks(fontsize=14, rotation=25)
plt.yticks(fontsize=14, rotation=25)
fig = plt.gcf()
fig.set_size_inches(12.0, 5.0)
plt.legend(loc=9, bbox_to_anchor=(0.65, 1.0))
plt.title("Comparacion de busquedas de marcas", fontsize=18)
plt.show()
fig, ax = plt.subplots()
df3.loc[:,[vam, vao]].plot.bar(stacked=True, color=colores_visitas, width=0.1, position=0, ax=ax)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
plt.xlabel("Marcas", fontsize=18)
plt.ylabel("Cantidad", fontsize=18)
plt.yscale('linear')
plt.xticks(fontsize=14, rotation=25)
plt.yticks(fontsize=14, rotation=25)
fig = plt.gcf()
fig.set_size_inches(12.0, 5.0)
plt.legend(loc=9, bbox_to_anchor=(0.65, 1.0))
plt.title("Comparacion de visitas tras buscar una marca", fontsize=18)
plt.show()
fig, ax = plt.subplots()
df3.loc[:,[coam, coao]].plot.bar(stacked=True, color=colores_cheackouts, width=0.1, position=0, ax=ax)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
plt.xlabel("Marcas", fontsize=18)
plt.ylabel("Cantidad", fontsize=18)
plt.yscale('linear')
plt.xticks(fontsize=14, rotation=25)
plt.yticks(fontsize=14, rotation=25)
fig = plt.gcf()
fig.set_size_inches(12.0, 5.0)
plt.legend(loc=9, bbox_to_anchor=(0.65, 1.0))
plt.title("Comparacion de checkouts tras buscar una marca", fontsize=18)
plt.show()
fig, ax = plt.subplots()
df3.loc[:,[cam, cao]].plot.bar(stacked=True, color=colores_conversion, width=0.1, position=0, ax=ax)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
plt.xlabel("Marcas", fontsize=18)
plt.ylabel("Cantidad", fontsize=18)
plt.yscale('linear')
plt.xticks(fontsize=14, rotation=25)
plt.yticks(fontsize=14, rotation=25)
fig = plt.gcf()
fig.set_size_inches(12.0, 5.0)
plt.legend(loc=9, bbox_to_anchor=(0.65, 1.0))
plt.title("Comparacion de conversiones tras buscar una marca", fontsize=18)
plt.show()
fig, ax = plt.subplots()
df3.loc[:,[lam, lao]].plot.bar(stacked=True, color=colores_lead, width=0.1, position=0, ax=ax)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
plt.xlabel("Marcas", fontsize=18)
plt.ylabel("Cantidad", fontsize=18)
plt.yscale('linear')
plt.xticks(fontsize=14, rotation=25)
plt.yticks(fontsize=14, rotation=25)
fig = plt.gcf()
fig.set_size_inches(12.0, 5.0)
plt.legend(loc=9, bbox_to_anchor=(0.85, 1.0))
plt.title("Comparacion de leads tras buscar una marca", fontsize=18)
plt.show()

#for cat in df['model'].dropna().cat.categories:
#    print('model: %s'%(cat))
