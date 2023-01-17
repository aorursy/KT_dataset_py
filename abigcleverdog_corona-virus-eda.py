import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

import warnings

warnings.filterwarnings('ignore')



# Allow several prints in one cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
file = '/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv'

df = pd.read_csv(file)

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

df.head()
stat_df = df.groupby('Country').Confirmed.sum()#.sort_values(ascending=False)

stat_df = pd.concat([stat_df, df.groupby('Country').Deaths.sum(), df.groupby('Country').Recovered.sum()], axis=1)

stat_df.sort_values(by='Confirmed', ascending=False).head()
stat_df.sort_values(by='Deaths', ascending=False).head()
mask = (df['Country'] == 'Mainland China') | (df['Country'] == 'China')

china_df = df[mask]

print("The death rate of Coronavirus infection within China is {:.2f}".format(china_df.Deaths.sum()/china_df.Confirmed.sum() * 100))



outof_china = df[~mask]

print("The death rate of Coronanirus infection outside China is {:.2f}".format(outof_china.Deaths.sum()/outof_china.Confirmed.sum() * 100))



print("The death rate of flu in the U.S. in 2016-2017 is about {:.2f}".format(38000/5000000 * 100))