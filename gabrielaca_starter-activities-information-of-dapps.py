import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

sns.set()

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('../input/activities-information-of-dapps/Activity Information of DApps/state_of_the_dapp.csv', encoding = 'gb18030',parse_dates=[17])

data.info()
data.head(3)
sns.countplot (data['platform'])
data.category.value_counts()
mask=(data.platform=='Ethereum')

data_ethereum=data[mask]

data_ethereum.info()
data_e=data_ethereum.drop(columns=['user_total_loss','user_loss_ratio','user_loss_average','user_count_unique_remove_contract_creator','contract_count','transaction_count','transaction_count_volume_greater_0','transaction_count_ratio','link', 'rank','short_intro','long_intro', 'submitted_updated','contract','submitted_updated','mainnet'])

data_e
data_e.dev_activity_30d.dtype
data_e.head()
data_e[data_ethereum['dev_activity_30d']=='-']
data_e['dev_activity_30d'].replace(regex=True, inplace=True, to_replace='-', value=0)

data_e['users_24h'].replace(regex=True, inplace=True, to_replace='-', value=0)
data_e['dev_activity_30d'] = data_e['dev_activity_30d'].astype(np.float64)

data_e['users_24h'] = data_e['users_24h'].astype(np.float64)
data_e.shape
sns.pairplot(data=data_e)
ax = sns.distplot(data_e['users_24h'], rug=True, hist=False, label='UW', kde_kws={'bw':0.1})
plt.xticks(rotation = 90)

sns.barplot(x="category", y="dev_activity_30d", data=data_e);
plt.xticks(rotation = 90)

sns.barplot(x="category", y="users_24h", data=data_e);
def plot_corre_heatmap(corr):

    '''

    Definimos una función para ayudarnos a graficar un heatmap de correlación

    '''

    plt.figure(figsize=(12,10))

    sns.heatmap(corr, cbar = True,  square = False, annot=True, fmt= '.2f'

                ,annot_kws={'size': 15},cmap= 'coolwarm')

    plt.xticks(rotation = 45)

    plt.yticks(rotation = 45)

    # Arreglamos un pequeño problema de visualización

    b, t = plt.ylim() # discover the values for bottom and top

    b += 0.5 # Add 0.5 to the bottom

    t -= 0.5 # Subtract 0.5 from the top

    plt.ylim(b, t) # update the ylim(bottom, top) values

    plt.show()
corr = data_e.corr()

plot_corre_heatmap(corr)
sl=data_e.software_license.value_counts()

sl.describe()
data_e['tag'].value_counts().idxmax()
data_e.groupby(['category']).count()
e_wallet = data_e[data_e.category == 'Wallet']

e_wallet.reset_index(inplace = True, drop = True)

e_wallet.head()
e_wallet['tag'].value_counts().idxmax()
import datetime as dt

e_wallet['last_updated']=e_wallet['last_updated'].map(dt.datetime.toordinal)

e_wallet.head()
volume=e_wallet.volume_7d.str.extract(r'([-]?[\d.]*)([\w\D]*)').rename(columns={0:'Volume_7d', 1:'divisa'})
vol_num=volume.drop(['divisa'], axis=1)
vol_num['Volume_7d'].replace(regex=True, inplace=True, to_replace='-', value=0)

vol_num
vol_num.insert(1, "divisa", "USD") 

vol_num
e_wallet = pd.concat([e_wallet, vol_num], axis=1, join='inner')

e_wallet.head()
e_wallet.status = e_wallet.status.map({'Beta': 0, 'Live': 1,'Prototype':0,'WIP':0})

e_wallet.head()
sns.countplot (e_wallet['status'])
import statsmodels.api as sm

sns.lmplot(x="last_updated", y="status", data=e_wallet,

           logistic=True, y_jitter=.03);
corr = e_wallet.corr()

plot_corre_heatmap(corr)