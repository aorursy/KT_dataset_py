# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
%matplotlib inline
import folium
import mpld3
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
!ls -lh ../input/
df_national = pd.read_csv('../input/MPI_national.csv', ',')
df_subNational = pd.read_csv('../input/MPI_subnational.csv', ',')
print('Shape:', df_national.shape, '\n')
print(df_national.count(), '\n')
df_national.head()
print('Shape:', df_subNational.shape, '\n')
print(df_subNational.count(), '\n')
df_subNational.head()
df_national[df_national['ISO'].isin(['BRA'])]
#df_national.head()
df_national.describe().transpose()
df_subNational.describe().transpose()
print('IsnaAnyAny:', df_national.isna().any().any(), '\n')
# Some different ways to check if we have missing data on dataframes 
print('IsnaAnyAny:', df_subNational.isna().any().any(), '\n')
print('IsnullAnyAny:', df_subNational.isnull().any().any(),'\n')
print('IsnullAny:\n',df_subNational.isnull().any(),'\n')
# Show what are the records with missing data
df_subNational[df_subNational['Intensity of deprivation Regional'].isna()]
# Just dataframe "df_subNational" has null values, but it will change after the next line do it's job
df_subNational = df_subNational.dropna()

# Cheking the outcome
print(df_subNational.isna().any(), '\n')
df_subNational.count()
# Finally column "Country" can be deleted from both dataframes
del df_national['Country']
del df_subNational['Country']

print('national:', df_national.columns, '\n')
print('subNational:', df_subNational.columns, '\n')
df_countryMpi = df_subNational[['ISO country code','MPI National']].drop_duplicates(
    subset=None,
    keep='first',
    inplace=False)

df_countryMpi.rename(columns={'ISO country code': 'ISO'}, inplace=True)

df_mpiNational = df_national.merge(df_countryMpi, on='ISO', how='inner' )

df_mpiNational.describe().transpose()
print(df_mpiNational.isna().any(), '\n')
df_mpiNational.count()
df_subNational[df_subNational['ISO country code'].isin(['BRA'])]

# Filter data by CountryName BRA
df_subBRA = df_subNational[df_subNational['ISO country code'] == 'BRA']
ls_braStates = ['Acre',
                 'Alagoas',
                 'Amapá',
                 'Amazonas',
                 'Bahia',
                 'Ceará',
                 'Distrito Federal',
                 'Espírito Santo',
                 'Goiás',
                 'Maranhão',
                 'Mato Grosso',
                 'Mato Grosso do Sul',
                 'Minas Gerais',
                 'Paraná',
                 'Paraíba',
                 'Pará',
                 'Pernambuco',
                 'Piauí',
                 'Rio Grande do Norte',
                 'Rio Grande do Sul',
                 'Rio de Janeiro',
                 'Rondônia',
                 'Roraima',
                 'Santa Catarina',
                 'Sergipe',
                 'São Paulo',
                 'Tocantins']

df_subBRA.loc[:,('Sub-national region')] = ls_braStates
df_subBRA.set_index('Sub-national region', inplace=True)
df_subBRA
norte = ['Rondônia',
         'Acre',
         'Amazonas',
         'Roraima',
         'Pará',
         'Amapá',
         'Tocantins']
         

nordeste = ['Maranhão',
            'Piauí',
            'Ceará',
            'Rio Grande do Norte',
            'Paraíba',
            'Pernambuco',
            'Alagoas',
            'Sergipe',
            'Bahia']
           
sudeste = ['Minas Gerais',
           'Espírito Santo',
           'Rio de Janeiro',
           'São Paulo']
           
sul = ['Paraná',
       'Santa Catarina',
       'Rio Grande do Sul']

centro_oeste = ['Mato Grosso do Sul',
                'Mato Grosso',
                'Goiás',
                'Distrito Federal']


for state in nordeste:
    df_subBRA.loc[state, 'National region'] = 'nordeste'
    df_subBRA.loc[state, 'color'] = '#00BFFF'
    
for state in norte:
    df_subBRA.loc[state, 'National region'] = 'norte'
    df_subBRA.loc[state, 'color'] = 'yellow'

for state in sudeste:
    df_subBRA.loc[state, 'National region'] = 'sudeste'
    df_subBRA.loc[state, 'color'] = 'green'

for state in sul:
    df_subBRA.loc[state, 'National region'] = 'sul'
    df_subBRA.loc[state, 'color'] = 'blue'
    
for state in centro_oeste:
    df_subBRA.loc[state, 'National region'] = 'centro oeste'
    df_subBRA.loc[state, 'color'] = 'orange'
df_subBRA
# Use 
# 1- 'MPI National' as a horizontal line (multiplyed by 1000)
# 2-  'Intensity of deprivation Regional' as Y axis data (multiplyed by 1000)
# 3- 'Headcount Ratio Regional' as X axis data
# 4- 'MPI Regional' as boble size
# 5- 'National region' as boble color

plt.figure(figsize=(40, 15), facecolor='gray')

# national plot
national_x = df_subBRA['Headcount Ratio Regional']
national_y = df_subBRA['Intensity of deprivation Regional']
nsizes = df_subBRA['MPI National'] * 11000
plt.scatter(national_x, national_y, s=nsizes, marker='o', c='white', alpha=1.0, edgecolors='black', linewidths=2.0, label='National')

# region "sul" plot
sul_x = df_subBRA[df_subBRA['National region'] == 'sul']['Headcount Ratio Regional']
sul_y = df_subBRA[df_subBRA['National region'] == 'sul']['Intensity of deprivation Regional']
sul_sizes = df_subBRA[df_subBRA['National region'] == 'sul']['MPI Regional'] * 11000
plt.scatter(sul_x, sul_y, s=sul_sizes, marker='o', c='blue', alpha=0.3, edgecolors='black', linewidths=0.8, label='Region Sul')

# region "norte" plot
norte_x = df_subBRA[df_subBRA['National region'] == 'norte']['Headcount Ratio Regional']
norte_y = df_subBRA[df_subBRA['National region'] == 'norte']['Intensity of deprivation Regional']
norte_sizes = df_subBRA[df_subBRA['National region'] == 'norte']['MPI Regional'] * 11000
plt.scatter(norte_x, norte_y, s=norte_sizes, marker='o', c='yellow', alpha=0.3, edgecolors='black', linewidths=0.8, label='Region Norte')

# region "nordeste" plot
nordeste_x = df_subBRA[df_subBRA['National region'] == 'nordeste']['Headcount Ratio Regional']
nordeste_y = df_subBRA[df_subBRA['National region'] == 'nordeste']['Intensity of deprivation Regional']
nordeste_sizes = df_subBRA[df_subBRA['National region'] == 'nordeste']['MPI Regional'] * 11000
plt.scatter(nordeste_x, nordeste_y, s=nordeste_sizes, marker='o', c='red', alpha=0.3, edgecolors='black', linewidths=0.8, label='Region Nordeste')

# region "sudeste" plot
sudeste_x = df_subBRA[df_subBRA['National region'] == 'sudeste']['Headcount Ratio Regional']
sudeste_y = df_subBRA[df_subBRA['National region'] == 'sudeste']['Intensity of deprivation Regional']
sudeste_sizes = df_subBRA[df_subBRA['National region'] == 'sudeste']['MPI Regional'] * 11000
plt.scatter(sudeste_x, sudeste_y, s=sudeste_sizes, marker='o', c='green', alpha=0.3, edgecolors='black', linewidths=0.8, label='Region Sudeste')

# region "centro oeste" plot
coeste_x = df_subBRA[df_subBRA['National region'] == 'centro oeste']['Headcount Ratio Regional']
coeste_y = df_subBRA[df_subBRA['National region'] == 'centro oeste']['Intensity of deprivation Regional']
coeste_sizes = df_subBRA[df_subBRA['National region'] == 'centro oeste']['MPI Regional'] * 11000
plt.scatter(coeste_x, coeste_y, s=coeste_sizes, marker='o', c='orange', alpha=0.3, edgecolors='black', linewidths=0.8, label='Region Centro Oeste')

# adding state names to plot
for record in df_subBRA.itertuples():
    name  = record[0]
    rec_x = record[5]
    rec_y = record[6] + 0.4
    rec_color = record[8]
    plt.text(x=rec_x, y=rec_y , s=name, horizontalalignment='center', verticalalignment='top', color='black')

plt.title('Brazilian MPI Distribution\n Markers size repersents the MPI', fontsize=20, color='white')
plt.xlabel('Headcount Ratio Regional', fontsize=15, color='white')
plt.ylabel('Intensity of deprivation Regional', fontsize=15, color='white')
plt.legend(bbox_to_anchor=(0.65, -0.05), markerscale=0.5, facecolor='white', framealpha=0.9, ncol=6)
plt.grid(axis='y')



#plt.draw()
plt.show()
ax = df_subBRA[['MPI National','MPI Regional']].plot(kind='bar',
                                                     rot=True, legend=True, table=False,
                                                     figsize=(50,4),
                                                     title='Brazil - National and Sub-Regional MPI Index')
ax.set_xlabel('Sub-national region')
ax.set_ylabel('MPI Index')
ax.set_ylim([0,0.1])
ax.grid(axis='y')
ax2 = df_subBRA['Headcount Ratio Regional'].plot(kind='bar',
                                                rot=True,
                                                figsize=(40,4),
                                                title='Brazil - % of Pours by Sub-Region')
ax2.set_ylabel('%')
ax2.set_ylim([0,20])
ax2.grid(axis='y')
ax3 = df_subBRA['Intensity of deprivation Regional'].plot(kind='bar',
                                                rot=True,
                                                figsize=(40,4),
                                                title='Brazil - Intensity of deprivation Regional')
ax3.set_ylim([0,50])
ax3.grid(axis='y')