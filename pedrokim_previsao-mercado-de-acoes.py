from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import keras
import keras.backend as K
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplolib inline
import os
os.listdir()

base=pd.read_excel('../input/cotacao.xls',header=1,decimal=',',parse_dates=True)
base.Data = pd.to_datetime(base.Data, format='%d/%m/%Y')

base=base.loc[::-1]
base.head()
base.tail()
pd.set_option('display.float_format','{:.2f}'.format)
base.describe().transpose()
base.isnull().sum()
base.shape
plt.figure(figsize=(10,7))
sns.set_context('notebook', font_scale=1.5, rc={'font.size':20,
                                               'axes.titlesize':20,
                                               'axes.labelsize':18})
sns.kdeplot(base['Abertura'],color='green')
sns.rugplot(base['Abertura'],color='red')
sns.distplot(base['Abertura'],color='green')
sns.set_style('darkgrid')

plt.xlabel('Distribuicao do Preco de Abertura');
plt.figure(figsize=(13,7))
#base.iloc[:,0].plot(label='Abertura',color='red')
#base.iloc[:,3].plot(label='Fechamento',color='blue')
base.plot(x='Data', y=['Abertura', 'Fechado'],color = ['red', 'blue'])
plt.ylabel('Preco de Abertura')
plt.xlabel('Período')
plt.title('Histórico de Preço da BBAS3 entre 2015 e 2020')
plt.legend();
plt.figure(figsize=(13,7))
base.plot(x='Data', y='Volume')
np.sort(np.where(base['Quantidade de Ações'] < 1, base['Quantidade de Ações'], base['Quantidade de Ações']/base['Negócios']))
#np.where(base['Quantidade de Ações'] < 1, base['Quantidade de Ações'], base['Quantidade de Ações']/base['Negócios'])
#pd.DataFrame(np.where(base['Quantidade de Ações'] < 1, base['Quantidade de Ações'], base['Quantidade de Ações']/base['Negócios'])).plot()

base.plot(x='Data', y=['Quantidade de Ações'])
