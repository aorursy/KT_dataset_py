from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import seaborn as sns; sns.set() # heatmap

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import geopandas as gpd

from shapely.geometry import Point

import datetime as dt
print(os.listdir('../input'))
nRowsRead = 30000 # specify 'None' if want to read whole file

df1 = pd.read_csv('../input/crime-data-in-brazil/BO_2007_1.csv', delimiter=',', nrows = nRowsRead)

df2 = pd.read_csv('../input/crime-data-in-brazil/BO_2007_2.csv', delimiter=',', nrows = nRowsRead)

df3 = pd.read_csv('../input/crime-data-in-brazil/BO_2008_1.csv', delimiter=',', nrows = nRowsRead)

df4 = pd.read_csv('../input/crime-data-in-brazil/BO_2008_2.csv', delimiter=',', nrows = nRowsRead)

df5 = pd.read_csv('../input/crime-data-in-brazil/BO_2009_1.csv', delimiter=',', nrows = nRowsRead)

df6 = pd.read_csv('../input/crime-data-in-brazil/BO_2009_2.csv', delimiter=',', nrows = nRowsRead)

df7 = pd.read_csv('../input/crime-data-in-brazil/BO_2010_1.csv', delimiter=',', nrows = nRowsRead)

df8 = pd.read_csv('../input/crime-data-in-brazil/BO_2010_2.csv', delimiter=',', nrows = nRowsRead)

df9 = pd.read_csv('../input/crime-data-in-brazil/BO_2011_1.csv', delimiter=',', nrows = nRowsRead)

df10 = pd.read_csv('../input/crime-data-in-brazil/BO_2011_2.csv', delimiter=',', nrows = nRowsRead)

df11 = pd.read_csv('../input/crime-data-in-brazil/BO_2012_1.csv', delimiter=',', nrows = nRowsRead)

df12 = pd.read_csv('../input/crime-data-in-brazil/BO_2012_2.csv', delimiter=',', nrows = nRowsRead)

df = df1.append(df2).append(df3).append(df4).append(df5[df1.columns.values]).append(df6[df1.columns.values]).append(df7[df1.columns.values]).append(df8[df1.columns.values]).append(df9[df1.columns.values]).append(df10[df1.columns.values])

df.dataframeName = 'BO_2007_2008_2012.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.info()
df.head()
df_preview = df.dropna(subset = ['HORA_OCORRENCIA_BO', 'LATITUDE', 'LONGITUDE'])

df_preview = df[~df.LATITUDE.isin(['Informação restrita (art. 31 da LAI)', 'Informação restrita (Art. 31 da LAI)', 'Informação restrita (art.31 da LAI)'])]

df_preview = df[~df.LONGITUDE.isin(['Informação restrita (art. 31 da LAI)', 'Informação restrita (Art. 31 da LAI)', 'Informação restrita (art.31 da LAI)'])]



df_preview = df_preview.copy()



df_preview.LATITUDE = df_preview.LATITUDE.astype(float)

df_preview.LONGITUDE = df_preview.LONGITUDE.astype(float)



df_preview.info()

df_preview.head()
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()
# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()
def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    plt.title('Pearson Correlation of Features', y=1.05, size=15)
correlation_heatmap(df_preview)
plotPerColumnDistribution(df_preview.iloc[np.random.choice(df_preview.shape[0], 2000)], 10, 2)
plotScatterMatrix(df_preview.iloc[np.random.choice(df_preview.shape[0], 2000)], 18, 10)
ten_most_common = df_preview[df_preview['RUBRICA'].isin(df_preview['RUBRICA'].value_counts().head(10).index)]



ten_most_crime_by_city = pd.crosstab(ten_most_common['CIDADE'], ten_most_common['RUBRICA'])

ten_most_crime_by_city.plot(kind='barh', figsize=(16,12), stacked=True, colormap='Greens', title='Disbribution of the Ten Most Common Crimes in Each City')
date = pd.to_datetime(df_preview[['DATA_OCORRENCIA_BO', 'HORA_OCORRENCIA_BO']].fillna(method='ffill').apply(lambda x: '{} {}'.format(x[0], x[1]), axis=1), format='%d/%m/%Y %H:%M')

df_preview.set_index(pd.DatetimeIndex(date), inplace=True)

#df_interval1 = df_preview[df_preview.DATA_OCORRENCIA_BO == '01/01/2007'].between_time('00:15', '02:45')

df_interval1 = df_preview.between_time('00:15', '00:45')



df_interval1['LOGRADOURO'].value_counts().head(3)
data_month_hour = pd.crosstab(df_preview['HORA_OCORRENCIA_BO'],df_preview['MES'])

data_month_hour.plot(figsize=(12, 12))
df_preview.ANO.value_counts().plot(kind='bar', title='Overall number of occurences (crimes) per year', figsize=(14, 8))
df_sexo = df_preview[df_preview['SEXO_PESSOA'].isin(['F', 'M']) & df_preview['CIDADE'].isin(df_preview['CIDADE'].value_counts().head(5).index)]

data_cidade_sexo = pd.crosstab(df_sexo['CIDADE'], df_sexo['SEXO_PESSOA'])

data_cidade_sexo.plot.bar(stacked=True, figsize=(16, 8))
df_cor = df_preview[df_preview['CIDADE'].isin(df_preview['CIDADE'].value_counts().head(3).index) & df_preview['RUBRICA'].isin(df_preview['RUBRICA'].value_counts().head(3).index)]

df_cor = df_cor[df_cor.DESCR_TIPO_PESSOA == 'Vítima              ']

data_cidade_cor = pd.crosstab(df_cor['COR'], df_cor['RUBRICA'])

data_cidade_cor.plot.bar(stacked=True, figsize=(16, 8))
df_preview.RUBRICA.value_counts().plot(kind='pie', figsize=(16, 16), title='Proportion of crimes')
df_preview.COR.value_counts().plot(kind='pie', figsize=(12, 12), title='Proportion of colors')

plt.legend(prop={'size': 12})
geometry = [Point(xy) for xy in zip(df_preview['LONGITUDE'], df_preview['LATITUDE'])]

crs = {'init': 'epsg:4326'}

geo_df = gpd.GeoDataFrame(df_preview.copy(), crs = crs, geometry=geometry)
fig, ax = plt.subplots(figsize=(12, 12))



brasil_map = gpd.read_file('../input/distrito-sp/DISTRITO_MUNICIPAL_SP_SMDUPolygon.shp')



brasil_map.plot(ax=ax)



geo_df[np.logical_and(geo_df.LATITUDE < -23.2, geo_df.RUBRICA == 'Homicídio simples (art. 121)')].geometry.plot(ax=ax, markersize=5, color='black', marker='o', label='Homicídio simples (art. 121)')

geo_df[np.logical_and(geo_df.LATITUDE < -23.2, geo_df.RUBRICA == 'Homicídio qualificado (art. 121, §2o.)')].geometry.plot(ax=ax, markersize=5, color='red', marker='o', label='Homicídio qualificado (art. 121, §2o.)')

geo_df[np.logical_and(geo_df.LATITUDE < -23.2, geo_df.RUBRICA == 'Drogas sem autorização ou em desacordo (Art.33, caput)')].geometry.plot(ax=ax, markersize=5, color='green', marker='^', label='Drogas sem autorização ou em desacordo (Art.33, caput)')



plt.legend(prop={'size': 10})
df_pred = df.copy()

date = pd.to_datetime(df[['DATA_OCORRENCIA_BO', 'HORA_OCORRENCIA_BO']].fillna(method='ffill').apply(lambda x: '{} {}'.format(x[0], x[1]), axis=1), format='%d/%m/%Y %H:%M')

#df_pred.set_index(pd.DatetimeIndex(date), inplace=True)

df_pred = df_pred[['NUM_BO', 'ID_DELEGACIA', 'ANO', 'MES', 'RUBRICA', 'SEXO_PESSOA', 'CIDADE', 'IDADE_PESSOA', 'COR']]

#df_pred = df_pred[['NUM_BO', 'ID_DELEGACIA']]



df_pred.IDADE_PESSOA = df_pred.IDADE_PESSOA.replace({'. 25 ANOS                         ': 25, 'IDADE APROX. 22 ANOS           ': 22})

df_pred.IDADE_PESSOA = df_pred.IDADE_PESSOA.astype(float)

df_pred.IDADE_PESSOA = df_pred.IDADE_PESSOA.fillna(df_pred.IDADE_PESSOA.mean()).astype(int)

df_pred = df_pred[df_pred['IDADE_PESSOA'].between(1, 101, inclusive=True)]



#df_pred.fillna(method='ffill', inplace=True)



df_pred.CIDADE = df_pred.CIDADE.astype('category').cat.codes

df_pred.RUBRICA = df_pred.RUBRICA.astype('category').cat.codes

df_pred.COR = df_pred[df_pred['COR'].isin([])].COR.astype('category').cat.codes

df_pred.SEXO_PESSOA = df_pred.SEXO_PESSOA.astype('category').cat.codes



#df_pred = pd.get_dummies(df_pred)