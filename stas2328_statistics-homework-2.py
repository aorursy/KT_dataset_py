# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import stats
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# Importar datos de archivo train.csv
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
figsize_rect=(14,7) # dimensiones para graficos en formato rectangular

mpl.rc('figure', max_open_warning = 0)
mpl.rcParams['font.size'] = 15
mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['lines.linestyle'] = '-'
mpl.rcParams['lines.color'] = '#1372B2'
mpl.rcParams["legend.title_fontsize"] = 16
colors = ['#1372B2', "#F19917",'#F76413','#2B6B85','#359CAE']
def get_outliers(attr):
    list_data = sorted(train[attr].tolist())
    Q1 = train[attr].quantile(0.25)
    Q3 = train[attr].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return [num for num in list_data if num < lower_bound or num > upper_bound]
variablesNumericas = [
  'LotFrontage',  'LotArea',      'YearBuilt',
  'YearRemodAdd', 'MasVnrArea',   'BsmtFinSF1',
  'BsmtFinSF2',   'BsmtUnfSF',    'TotalBsmtSF',
  '1stFlrSF',     '2ndFlrSF',     'LowQualFinSF',
  'GrLivArea',    'BsmtFullBath', 'BsmtHalfBath',
  'FullBath',     'HalfBath',     'BedroomAbvGr',
  'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
  'GarageYrBlt',  'GarageCars',   'GarageArea',
  'WoodDeckSF',   'OpenPorchSF',  'EnclosedPorch',
  '3SsnPorch',    'ScreenPorch',  'PoolArea',
  'MiscVal',      'MoSold',       'YrSold',
  'SalePrice'
]
flierprops = dict(markerfacecolor='#00ADEE', markeredgecolor='#00ADEE', marker='.', markersize=8, linestyle='none')
for variable in variablesNumericas:
    plt.figure(figsize=figsize_rect)
    train.boxplot(column=[variable], grid=False, vert=False, color='#00ADEE',
                  flierprops=flierprops, patch_artist=True,
                  boxprops=dict(facecolor='#00ADEE', color='#000'),
                  capprops=dict(color='#000'),
                  whiskerprops=dict(color="#000"),
                  medianprops=dict(color="#7ae9ff", linewidth=1.5))
    plt.suptitle(variable)
    plt.yticks([])
    plt.draw()
    print("Variable:", variable)
    print("Media:", train[variable].mean())
    print("Mediana:", train[variable].median())
    print("Moda:", train[variable].mode()[0])
    print("Varianza:", train[variable].var())
    print("Desviacion estandar:", train[variable].std())
    print("Valor minimo:", train[variable].min())
    print("Valor maximo:", train[variable].max())
    print("Rango:", train[variable].max() - train[variable].min())
    print("Q1:", train[variable].quantile(0.25))
    print("Q3:", train[variable].quantile(0.75))
    print("Kurtosis:", train[variable].kurt())
    print("Sesgo:", train[variable].skew())
    print("Valores atípicos:", get_outliers(variable)) 
    print()

plt.show()