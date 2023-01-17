import numpy as np

import pandas as pd

import seaborn as sns

import random

import scipy.stats as stt

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

%pylab inline
data = pd.read_csv('../input/googleplaystore.csv')
data.shape
data.head()
data.Category.unique()
def f(x):

    if x[-1]=='M':

        res = float(x[:-1])*1024

    elif x[-1]=='k':

        res = float(x[:-1])

    else:

        res = np.nan

    return res



new_size = data.Size.apply(f)
data[data.Size=='1,000+']
data_1 = data[data.Rating!=19]

data_1['Size'] = new_size

data_1.drop('Current Ver',axis=1, inplace=True)
def inst(x):

    if x[-1]=='+':

        res = float(x[:-1].replace(',',''))

    else:

        res = float(x)

    return res



def rev(x):

        res = int(x)

        return res

    

def prc(x):

    if x[0]=='$':

        res = float(x[1:])

    else:

        res = float(x)

    return res



new_inst = data_1.Installs.apply(inst)

new_rev = data_1.Reviews.apply(rev)

new_prc = data_1.Price.apply(prc)



data_1['Installs'] = new_inst

data_1['Reviews'] = new_rev

data_1['Price']  = new_prc
pop_data = pd.merge(data_1[['App','Reviews']],\

         data_1[['App','Category','Rating','Installs']][data_1.Rating>4].drop_duplicates(),\

         left_index=True, right_index=True)

pop_data.drop('App_x',axis=1, inplace=True)

pop_data.shape
pop_data.sort_index(by=['Installs','Reviews'], ascending=False, inplace=True)

pop_data.head()
fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(2,3,1)

ax1.set_title('Most popular apps')

pd.value_counts(pop_data.Category)[:15].plot(kind='barh');

ax2 = fig.add_subplot(2,3,2)

ax2.set_title('All apps')

pd.value_counts(data_1.Category).head(15).plot.barh();

plt.tight_layout()
pop_data[pop_data.Category=='FAMILY'].head(10)
game_data = data_1[data_1['Category']=='GAME']

fig = plt.figure(figsize=(10,10))

sns.set(style="white",font_scale=2)

sns.heatmap(game_data.dropna()[['Rating','Reviews','Size','Installs','Price']].corr(), fmt='.2f',annot=True,linewidth=2);
fig = plt.figure(figsize=(5,5))

sns.set(font_scale=2)

sns.countplot(game_data.Type, facecolor=(0, 0, 0, 0),linewidth=5, \

              edgecolor=sns.color_palette("dark", 3));
sns.set(style="white",font_scale=1.5)

fig = plt.figure(figsize=(7,7))

sns.countplot(y = game_data[game_data.Type=='Paid'].Genres, palette="Set3");
fig = plt.figure(figsize=(15,15))

game_data['Installs'].groupby(game_data['Genres']).sum().plot(kind='barh');
fig = plt.figure(figsize=(15,15))

sns.boxplot(x='Reviews',y='Genres', data=game_data);
fig = plt.figure(figsize=(5,5))

sm_game = game_data['Rating'].groupby(game_data['Genres']).sum()

sm_game[sm_game>150].plot(kind='barh');
sns.set(style="white",font_scale=1)

sns.countplot(game_data['Content Rating'], facecolor=(0, 0, 0, 0),linewidth=5, \

              edgecolor=sns.color_palette("dark", 3));
pd.value_counts(game_data['Genres'])[:10]
gen_dat = game_data[game_data['Genres'].isin(['Action','Arcade','Racing','Adventure','Card','Casual'])]

g = plt.figure(figsize=(15,5))

sns.set(style="whitegrid",font_scale=2,palette="muted")

sns.countplot(hue=gen_dat['Content Rating'],x=gen_dat['Genres']);
game_data['Rev/Inst'] = game_data.Reviews/game_data.Installs

plt.figure(figsize=(10,10))

game_data[game_data['Rev/Inst']>0.05]['Rev/Inst'].groupby(game_data['Genres']).mean().plot(kind='barh');
from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

tsn = TSNE()

pca = PCA(n_components=2)

scaler =  StandardScaler()
num_data = scaler.fit_transform(game_data.dropna()[['Rating','Reviews','Size','Installs','Price']])

res_tsne = pca.fit_transform(num_data)

pp_data = np.hstack((res_tsne,game_data.dropna().Genres.values.reshape(974,1)))
ss_dat = pd.DataFrame(pp_data)

ss_dat = ss_dat[ss_dat[2].isin(['Action','Arcade','Racing','Adventure'])]

plt.figure(figsize=(15,10))

sns.swarmplot(x=ss_dat[0],y=ss_dat[1],size=10,hue=ss_dat[2]);