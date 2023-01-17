import numpy as np 

import seaborn as sns

import pandas as pd

	

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker
data = pd.read_hdf('../input/train.h5')
pd.set_option('display.max_columns', 150)

pd.set_option('display.max_rows', 100)
def myticks(x,pos):



    exponent = abs(int(np.log10(np.abs(x))))  

    return exponent
def plot_exp(data, title):

    fig, ax =plt.subplots(figsize = (12, 8))

    ax.plot(data.t16_exp, data.timestamp)

    

    ax.set_title(title)

    ax.set_xlabel('Negative Power of Technical_16')

    ax.set_ylabel('Timestamp')

    plt.show()
data.technical_16.fillna(0, inplace = True)
data['t16_exp'] = data.technical_16.map(lambda z: int(np.log10(np.abs(z))) - 1 if z!=0  else 0)
plot_exp(data.loc[(data.id == 288) & (data.t16_exp != 0)  & (~data.t16_exp.isnull()) ,['timestamp', 't16_exp']], 'id = 288')
plot_exp(data.loc[(data.id == 1201) & (data.t16_exp != 0.0)  & (~data.t16_exp.isnull()) ,['timestamp', 't16_exp']], 'id = 1201')
data['t16_first_number'] = (data.technical_16 * (10**(data.t16_exp.abs())).astype('int'))
def plot_first_numb(data, title):

    fig, ax =plt.subplots(figsize = (12, 8))

    ax.plot(data.timestamp, data.t16_first_number,marker = 'v',  mfc = 'g')

    

    ax.set_title(title)

    ax.set_yticks(range(-10,10))

    ax.set_xlabel('Timestamp')

    ax.set_ylabel('First_number')

    plt.show()
plot_first_numb(data.loc[(data.id == 300) & (data.t16_exp != 0) ,['timestamp', 't16_first_number']], 'id = 300')
plot_first_numb(data.loc[(data.id == 1201) & (data.t16_exp != 0) ,['timestamp', 't16_first_number']], 'id = 1201')