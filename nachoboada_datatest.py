import pandas as pd

import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
%matplotlib inline
data = pd.read_csv("/kaggle/input/trainparadatos/train.csv")
df = pd.DataFrame(data)
df.columns.values
df.head()
data0 = {'tipodepropiedad':data['tipodepropiedad'],'ciudad':data['ciudad']}
df0 = pd.DataFrame(data0)
df0.head()
df0['quantity'] = 1
df0 = df0.groupby(['tipodepropiedad','ciudad']).agg({'quantity':sum})
df0.head()
df0=df0.reset_index()
df0.set_index('tipodepropiedad',inplace=True)
df0 = df0[df0.index.str.contains('Apartamento|Casa|Casa en condominio|Casa uso de suelo|Departamento compartido|Duplex')]
df0
df0 = df0.groupby(['ciudad']).agg({'quantity':sum})
df0
df0 = df0.sort_values(['quantity'], ascending=[False]) #Ordeno a los operadores de más a menos relevante
df0.head(50)
df0 = df0.reset_index()
d = {}

for a, x in df0.values:

    d[a] = x
wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate_from_frequencies(frequencies=d)

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
data00 = {'provincia':data['provincia']}
df00 = pd.DataFrame(data00)
df00.head()
df00['quantity'] = 1
df00 = df00.groupby(['provincia']).agg({'quantity':sum})
df00.head()
df00 = df00.sort_values(['quantity'], ascending=[False]) #Ordeno a los operadores de más a menos relevante
df00.head()
df00 = df00.reset_index()
d = {}

for a,x in df00.values:

    d[a] = x
wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate_from_frequencies(frequencies=d)

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
data01 = {'titulo':data['titulo']}
df01 = pd.DataFrame(data01)
df01.head()
df01 = df01.applymap(lambda x: str(x).replace(',',' '))
df01 = df01.applymap(lambda x: str(x).replace('.',' '))
df01 = df01.applymap(lambda x: str(x).split())
df01.head()
d = {}

for l in df01.values:

    for i in l:

        for x in i: 

            if not x in d:

                d[x] = 1

            else:

                d[x]+=1
d
wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate_from_frequencies(frequencies=d)

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
data02 = {'descripcion':data['descripcion']}
df02 = pd.DataFrame(data02)
df02.head()
df02 = df02.applymap(lambda x: str(x).replace(',',' '))
df02 = df02.applymap(lambda x: str(x).replace('.',' '))
df02 = df02.applymap(lambda x: str(x).split())
df02.head()
d = {}

for l in df01.values:

    for i in l:

        for x in i: 

            if not x in d:

                d[x] = 1

            else:

                d[x]+=1
d
wordcloud = WordCloud(

    width = 3000,

    height = 2000,

    background_color = 'black',

    stopwords = STOPWORDS).generate_from_frequencies(frequencies=d)

fig = plt.figure(

    figsize = (40, 30),

    facecolor = 'k',

    edgecolor = 'k')

plt.imshow(wordcloud, interpolation = 'bilinear')

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
data1 = {'antiguedad':data['antiguedad']}
df1 = pd.DataFrame(data1)
df1.head()
df1['quantity']=1
df1 = df1.groupby(['antiguedad']).agg({'quantity':sum})
df1.head()
df1 = df1.sort_values(['quantity'], ascending=[False]) #Ordeno a los operadores de más a menos relevante
df1 = df1.reset_index()
df1.head()
def filtrarMenosSignificativos(df,col):

    dfResto = pd.DataFrame({col:["Otros"], "quantity":[0]}) 

    for fila in df.itertuples():

        if(fila[2]<df["quantity"][4]):

            dfResto["quantity"][0]+=fila[2]

    return dfResto
filtrarMenosSignificativos(df1,"antiguedad").head()
df1=df1.head().append(filtrarMenosSignificativos(df1,"antiguedad"))
df1
df1.columns = ['Antigüedad','Cantidad']
df1.set_index('Antigüedad',inplace=True)
df1.plot.pie(subplots=True,title="Proporción de propiedades por Antiguedad",figsize=(15,10),fontsize=15)
df2 = data.pivot(index = 'id',columns = 'tipodepropiedad',values = 'precio')
df2.columns
df2.head()
df2.boxplot(column=['Apartamento','Casa','Casa en condominio','Casa uso de suelo','Departamento Compartido','Duplex'],figsize=(15,10),fontsize=10)
data20 = {'tipodepropiedad':data['tipodepropiedad'],'precio':data['precio']}
df20 = pd.DataFrame(data20)
df20.head()
df20 = df20[df20["tipodepropiedad"].isin(['Apartamento','Casa','Casa en condominio','Casa uso de suelo','Departamento Compartido','Duplex'])]
df20
plt.figure(figsize=(40,20))

sns.set(font_scale = 3)

sns.violinplot(x="tipodepropiedad", y="precio",data=df20,dodge=False)

plt.title('Variación del precio de las propiedades residenciales')

plt.xlabel('Propiedad')

plt.ylabel('Precio')



plt.ticklabel_format(style='plain', axis='y')
data3 = {'provincia':data['provincia'],'antiguedad':data['antiguedad']}
df3 = pd.DataFrame(data3)
df3.head()
df3['quantity']=1
df3 = df3.groupby(['provincia','antiguedad']).sum()
df3.head()
for_heatmap = df3.pivot_table(index='antiguedad', columns='provincia', values='quantity')





fig, ax = plt.subplots(figsize=(40,20))         # Sample figsize in inches

b = sns.heatmap(for_heatmap, linewidths = 0.1, ax = ax, linecolor = 'black', center=0, cmap = 'Blues') #con annot = True puedo ver el 

                                                                                        #valor de la posicion

b.set_title("Antiguedad de las propiedades por provincia",fontsize=40,fontdict=dict(weight='bold'))

b.set_xlabel("Provincias",fontsize=40,fontdict=dict(weight='bold'))

b.set_ylabel("Antiguedad",fontsize=40,fontdict=dict(weight='bold'))
data30 = {'ciudad':data['ciudad'],'antiguedad':data['antiguedad']}
df30 = pd.DataFrame(data30)
df30.head()
df30['quantity']=1
df30 = df30.groupby(['ciudad','antiguedad']).sum()
df30.head()
for_heatmap0 = df30.pivot_table(index='antiguedad', columns='ciudad', values='quantity')





fig, ax = plt.subplots(figsize=(40,20))         # Sample figsize in inches

b = sns.heatmap(for_heatmap, linewidths = 0.1, ax = ax, linecolor = 'black', center=0, cmap = 'Reds') #con annot = True puedo ver el 

                                                                                        #valor de la posicion

b.set_title("Antiguedad de las propiedades por ciudad",fontsize=40,fontdict=dict(weight='bold'))

b.set_xlabel("Ciudades",fontsize=40,fontdict=dict(weight='bold'))

b.set_ylabel("Antiguedad",fontsize=40,fontdict=dict(weight='bold'))
data4 = {'tipodepropiedad':data['tipodepropiedad'],'precio':data['precio']}
df4 = pd.DataFrame(data4)
df4.head()
df4 = df4.groupby(['tipodepropiedad']).mean()

df4
df4 = df4.sort_values(['precio'], ascending=[False])
df4
graph4 = df4.plot.barh(figsize=(15,10),fontsize=15, legend=False)

graph4.set_title('Tipo de propiedad vs Precio promedio',fontsize=20,fontdict=dict(weight='bold'))

graph4.set_xlabel("Precio promedio", fontsize=15,fontdict=dict(weight='bold'))

graph4.set_ylabel("Tipo de propiedad", fontsize=15,fontdict=dict(weight='bold'))

graph4
df5 = pd.DataFrame(data4)
df5 = df5.groupby(['tipodepropiedad']).std()
df5 = df5.sort_values(['precio'], ascending=[False])
df5['precio'] = df5['precio'].map(lambda x: x/df5['precio'].max())
graph5 = df5.plot.bar(figsize=(15,10),fontsize=15, legend=False)

graph5.set_title('Tipo de propiedad vs Proporción precio',fontsize=20,fontdict=dict(weight='bold'))

graph5.set_ylabel("Proporción precio", fontsize=15,fontdict=dict(weight='bold'))

graph5.set_xlabel("Tipo de propiedad", fontsize=15,fontdict=dict(weight='bold'))

graph5
data7 = {'ciudad':data['ciudad']}

df7 = pd.DataFrame(data7)
df7.head()
df7['cantidad'] = 1
df7 = df7.groupby('ciudad').sum()
df7
df7 = df7.sort_values(['cantidad'], ascending=[False])

df7
df7 = df7.head(20)
graph7 = df7.plot.bar(figsize=(15,10),fontsize=15, legend=False)

graph7.set_title('Cantidad de propiedades por ciudad',fontsize=20,fontdict=dict(weight='bold'))

graph7.set_ylabel("Cantidad de propiedades", fontsize=15,fontdict=dict(weight='bold'))

graph7.set_xlabel("Ciudad", fontsize=15,fontdict=dict(weight='bold'))

graph7