from IPython.core.display import display, HTML

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from scipy.stats import boxcox, probplot, norm, shapiro

from sklearn.preprocessing import PowerTransformer, MinMaxScaler

from sklearn.cluster import KMeans

import os

import warnings

warnings.filterwarnings('ignore')
#Main function for plot. Usefull for other case of clustering.



def comprueba_normalidad(df, return_type='axes', title='Comprobación de normalidad'):

    '''

    '''

    fig_tot = (len(df.columns))

    fig_por_fila = 3.

    tamanio_fig = 4.

    num_filas = int( np.ceil(fig_tot/fig_por_fila) )    

    plt.figure( figsize=( fig_por_fila*tamanio_fig+5, num_filas*tamanio_fig+2 ) )

    c = 0 

    shapiro_test = {}

    lambdas = {}

    for i, col in enumerate(df.columns):

        ax = plt.subplot(num_filas, fig_por_fila, i+1)

        probplot(x = df[df.columns[i]], dist=norm, plot=ax)

        plt.title(df.columns[i])

        shapiro_test[df.columns[i]] = shapiro(df[df.columns[i]])

    plt.suptitle(title)

    plt.show()

    shapiro_test = pd.DataFrame(shapiro_test, index=['Test Statistic', 'p-value']).transpose()

    return shapiro_test
os.listdir()
XY = pd.read_csv('../input/uci-wholesale-customers-data/Wholesale customers data.csv')
XY.head(2)
XY.describe()
XY.info()
XY.isnull().sum()
# Mapeo los datos

XY['Channel'] = XY['Channel'].map({1:'Horeca', 2:'Retail'})

XY['Region'] = XY['Region'].map({3:'Other Region', 2:'Oporto', 1: 'Lisboa'})
XY_cuants = XY[['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicassen']].copy()
XY_normalizado = (XY_cuants-XY_cuants.mean())/XY.std()

# This function, let as see a more ordered graph. 

# try not to use it yourself and see how the graph changes 
plt.figure(figsize=(14,6))

ax = sns.boxplot(data=XY_normalizado)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.title(u'Representación de cajas de las variables independientes X')

plt.ylabel('Valor de la variable normalizada')

_ = plt.xlabel('Nombre de la variable')
plt.figure(figsize=(14,6))

ax = sns.boxplot(data=XY)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.title(u'Representación de cajas de las variables independientes X')

plt.ylabel('Valor de la variable normalizada')

_ = plt.xlabel('Nombre de la variable')

#For this case there are not so much difference. But always its a good idea tried it. 
## Representation of the distributions of the variables using histograms.
plt.figure(figsize=(18,20))

n = 0

for i, column in enumerate(XY_cuants.columns):

    n+=1

    plt.subplot(5, 5, n)

    sns.distplot(XY_cuants[column], bins=30)

    plt.title('Distribución var {}'.format(column))

plt.show()
matriz_correlaciones = XY.corr(method='pearson')

n_ticks = len(XY.columns)

plt.figure( figsize=(9, 9) )

plt.xticks(range(n_ticks), XY.columns, rotation='vertical')

plt.yticks(range(n_ticks), XY.columns)

plt.colorbar(plt.imshow(matriz_correlaciones, interpolation='nearest', 

                            vmin=-1., vmax=1., 

                            cmap=plt.get_cmap('Blues')))

_ = plt.title('Matriz de correlaciones de Pearson')
shapiro_test = comprueba_normalidad(XY_cuants, title='Normalidad variables originales')
shapiro_test
bc = PowerTransformer(method='box-cox')

X_cuants_boxcox = bc.fit_transform(XY_cuants)

X_cuants_boxcox = pd.DataFrame(X_cuants_boxcox, columns=XY_cuants.columns)
shapiro_test = comprueba_normalidad(X_cuants_boxcox, title='Normalidad variables transformadas')
# looks perfect ¡
shapiro_test
plt.figure(figsize=(18,20))

n = 0

for i, column in enumerate(X_cuants_boxcox.columns):

    n+=1

    plt.subplot(4, 4, n)

    sns.distplot(X_cuants_boxcox[column], bins=30)

    plt.title('Distribución var {}'.format(column))

plt.show()
plt.figure(figsize=(15,7))

ax = sns.boxplot(data=X_cuants_boxcox)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.title(u'Representación de cajas de las variables independientes X')

plt.ylabel('Valor de la variable normalizada')

_ = plt.xlabel('Nombre de la variable')
for k in list(X_cuants_boxcox.columns):

    IQR = np.percentile(X_cuants_boxcox[k],75) - np.percentile(X_cuants_boxcox[k],25)

    

    limite_superior = np.percentile(X_cuants_boxcox[k],75) + 1.5*IQR

    limite_inferior = np.percentile(X_cuants_boxcox[k],25) - 1.5*IQR

    

    X_cuants_boxcox[k] = np.where(X_cuants_boxcox[k] > limite_superior,limite_superior,X_cuants_boxcox[k])

    X_cuants_boxcox[k] = np.where(X_cuants_boxcox[k] < limite_inferior,limite_inferior,X_cuants_boxcox[k])
plt.figure(figsize=(15,7))

ax = sns.boxplot(data=X_cuants_boxcox)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90)

plt.title(u'Representación de cajas de las variables independientes X')

plt.ylabel('Valor de la variable normalizada')

_ = plt.xlabel('Nombre de la variable')
#No Outliers now. Ok, next step. 
#In df one the two initial categorical variables and the transformed numeric variables

df  =  pd.concat([XY[['Channel','Region']],X_cuants_boxcox],axis=1)

df[:3]
df = pd.get_dummies(df,columns=['Channel','Region'],drop_first=True)

df[:3]
scaler = MinMaxScaler(feature_range=(0, 1))

X_escalado = scaler.fit_transform(df)

X_escalado = pd.DataFrame(X_escalado,columns=df.columns)

X_escalado.head()
# Now, with the next code, we are looking for the best number of cluster for our dataset.
cluster_range = range(1,20)

cluster_wss=[] 

for cluster in cluster_range:

    model = KMeans(cluster)

    model.fit(X_escalado)

    cluster_wss.append(model.inertia_)
plt.figure(figsize=[10,6])

plt.title('Curva WSS para encontrar el valor óptimo de clústers o grupos')

plt.xlabel('# grupos')

plt.ylabel('WSS')

plt.plot(list(cluster_range),cluster_wss,marker='o')

plt.show()
model = KMeans(n_clusters=6,random_state=0)

model.fit(X_escalado)
#Original Dataset with the predictions

df_total = XY.copy()

df_total['cluster']=model.predict(X_escalado)

df_total[:2]
df_total.cluster.value_counts().plot(kind='bar', figsize=(10,4))

plt.title('Conteo de clientes por grupo')

plt.xlabel('Grupo')

_ = plt.ylabel('Conteo')
#Here, we coud see our clients inside of a cluster
descriptivos_grupos = df_total.groupby(['cluster'],as_index=False).mean()

descriptivos_grupos
df_total.groupby('cluster').mean().plot(kind='bar', figsize=(15,7))

plt.title('Gasto medio por producto en cada clúster')

plt.xlabel(u'Número de clúster')

_ = plt.ylabel('Valor medio de gasto')
df_total[:2]