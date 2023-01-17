# Generic Libraries

import numpy as np

import pandas as pd



# Visualisation Libraries

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns

import warnings

from matplotlib import cm



pd.plotting.register_matplotlib_converters()

%matplotlib inline

plt.style.use('seaborn-whitegrid')

pd.set_option('display.max_columns', 500)

warnings.filterwarnings("ignore")

pd.options.display.float_format = '{:.2f}'.format
url = '../input/advertising/advertising.csv'

data = pd.read_csv(url, header='infer')
#Record Count

data.shape
#Check for missing values

data.isna().sum()
#Dropping unwanted column

data = data.drop(columns=['Unnamed: 0'], axis=1)
#Inspect

data.head()
#Stats Summary

data.describe().transpose()
# Function to summarize the advertising data per channel



def summarize(x):

    

   

    x_min = data[x].min()

    x_max = data[x].max()

    

    Q1 = data[x].quantile(0.25)

    Q2 = data[x].quantile(0.50)

    Q3 = data[x].quantile(0.75)

    x_mean = data[x].mean()

    

    print(f'6 Point Summary of {x.capitalize()} Attribute:\n'

          f'{x.capitalize()}(min)   : {x_min}\n'

          f'Q1                      : {Q1}\n'

          f'Q2(Median)              : {Q2}\n'

          f'Q3                      : {Q3}\n'

          f'{x.capitalize()}(max)   : {x_max}\n'

          f'{x.capitalize()}(mean)  : {round(x_mean)}')



    fig = plt.figure(figsize=(15, 10))

    plt.subplots_adjust(hspace = 0.6)

    sns.set_palette('Pastel2')

    

    plt.subplot(221)

    ax1 = sns.distplot(data[x], color = 'cornflowerblue')

    plt.title(f'{x.capitalize()} Density Distribution')

      

    plt.subplot(222)

    ax2 = sns.violinplot(x = data[x], palette = 'Pastel2', split = True, color='cornflowerblue')

    plt.title(f'{x.capitalize()} Violinplot')

    

    plt.subplot(223)

    ax2 = sns.boxplot(x=data[x], palette = 'Pastel2', width=0.7, linewidth=0.6, color='cornflowerblue')

    plt.title(f'{x.capitalize()} Boxplot')

    

    plt.subplot(224)

    ax3 = sns.kdeplot(data[x], cumulative=True)

    plt.title(f'{x.capitalize()} Cumulative Density Distribution')

    

    plt.show()
# Summarize TV

summarize('TV')
# Summarize Radio

summarize('radio')
# Summarize Newspaper

summarize('newspaper')
# Summarize Sales

summarize('sales')