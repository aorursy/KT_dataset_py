%reset -f



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture

from sklearn.manifold import TSNE

import re  #regular expression

from sklearn.preprocessing import StandardScaler

from pandas.plotting import andrews_curves

from mpl_toolkits.mplot3d import Axes3D
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
pd.options.display.max_rows = 1000

pd.options.display.max_columns = 1000
%matplotlib inline
cc = pd.read_csv("/kaggle/input/ccdata/CC GENERAL.csv")

cc.shape

cc.head()
cc.columns = [i.lower() for i in cc.columns]

cc.columns
cc.drop(columns=["cust_id"], inplace=True)
cc.head()
cc.describe()
cc.info()
cc.isnull().sum()
fig = plt.figure(figsize=(15,5))

ax=plt.subplot(1,2,1)

sns.distplot(cc.credit_limit)

plt.xlim([0,20000])

ax=plt.subplot(1,2,2)

sns.distplot(cc.minimum_payments)

plt.xlim([0,10000])
cc.fillna(value = {

                 'minimum_payments' :   cc['minimum_payments'].median(),

                 'credit_limit'               :     cc['credit_limit'].median()

               }, inplace=True)
cc.isnull().sum()
cc.describe()
from sklearn.preprocessing import normalize
ss =  StandardScaler()

out = ss.fit_transform(cc)

out = normalize(out)

out.shape
df_out = pd.DataFrame(out, columns=cc.columns)

df_out.head()
fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(15,15))

ax = axes.flatten()

fig.tight_layout()

# Do not display 18th, 19th and 20th axes

axes[3,3].set_axis_off()

axes[3,2].set_axis_off()

axes[3,4].set_axis_off()

# Below 'j' is not used.

for i,j in enumerate(df_out.columns):

    sns.distplot(df_out.iloc[:,i], ax = ax[i])

fig = plt.figure(figsize=(15,5))

ax=plt.subplot(2,2,1)

sns.distplot(df_out.credit_limit)

ax=plt.subplot(2,2,2)

sns.distplot(df_out.purchases)

ax=plt.subplot(2,2,3)

sns.distplot(df_out.payments)

ax=plt.subplot(2,2,4)

sns.distplot(df_out.balance)
fig = plt.figure(figsize=(15,5))

ax=plt.subplot(2,2,1)

sns.violinplot(df_out.credit_limit)

ax=plt.subplot(2,2,2)

sns.violinplot(df_out.purchases)

ax=plt.subplot(2,2,3)

sns.violinplot(df_out.payments)

ax=plt.subplot(2,2,4)

sns.violinplot(df_out.balance)

fig = plt.figure(figsize=(15,5))

ax=plt.subplot(1,4,1)

sns.boxplot(y=df_out.credit_limit)

ax=plt.subplot(1,4,2)

sns.boxplot(y=df_out.purchases)

ax=plt.subplot(1,4,3)

sns.boxplot(y=df_out.payments)

ax=plt.subplot(1,4,4)

sns.boxplot(y=df_out.balance)
sns.jointplot(df_out.credit_limit,df_out.purchases)
sns.jointplot(df_out.purchases,df_out.payments)
sns.pairplot(df_out, vars=["credit_limit","purchases","payments",'balance'])
bic = []

aic = []

for i in range(8):

    gm = GaussianMixture(

                     n_components = i+1,

                     n_init = 10,

                     max_iter = 100)

    gm.fit(df_out)

    bic.append(gm.bic(df_out))

    aic.append(gm.aic(df_out))



fig = plt.figure()

plt.plot([1,2,3,4,5,6,7,8], aic)

plt.plot([1,2,3,4,5,6,7,8], bic)

plt.show()
df_out.columns
gm = GaussianMixture(n_components = 3,

                     n_init = 10,

                     max_iter = 100)

gm.fit(df_out)
fig = plt.figure()



plt.scatter(df_out.iloc[:, 0], df_out.iloc[:, 2],

            c=gm.predict(df_out),

            s=5)

plt.scatter(gm.means_[:, 0], gm.means_[:, 2],

            marker='v',

            s=10,               # marker size

            linewidths=5,      # linewidth of marker edges

            color='red'

            )

plt.show()

fig = plt.figure()



plt.scatter(df_out.iloc[:, 2], df_out.iloc[:, 13],

            c=gm.predict(df_out),

            s=5)

plt.scatter(gm.means_[:, 2], gm.means_[:, 13],

            marker='v',

            s=10,               # marker size

            linewidths=5,      # linewidth of marker edges

            color='red'

            )

plt.show()

gm = GaussianMixture(

                     n_components = 3,

                     n_init = 10,

                     max_iter = 100)

gm.fit(df_out)



tsne = TSNE(n_components = 2)

tsne_out = tsne.fit_transform(df_out)

plt.scatter(tsne_out[:, 0], tsne_out[:, 1],

            marker='x',

            s=50,              # marker size

            linewidths=5,      # linewidth of marker edges

            c=gm.predict(df_out)   # Colour as per gmm

            )
densities = gm.score_samples(df_out)

densities



density_threshold = np.percentile(densities,4)

density_threshold



anomalies = df_out[densities < density_threshold]

anomalies.shape



unanomalies = df_out[densities >= density_threshold]

unanomalies.shape   
df_anomalies = pd.DataFrame(anomalies)

df_anomalies['type'] = 'anomalous'  

df_normal = pd.DataFrame(unanomalies)

df_normal['type'] = 'unanomalous'



df_anomalies.shape
df_anomalies.head()
df_normal.head()


fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(15,8))

ax = axes.flatten()

fig.tight_layout()

for i in range(15):

    sns.distplot(df_anomalies.iloc[:,i], ax = ax[i],color='b')

    sns.distplot(df_normal.iloc[:,i], ax = ax[i],color='g')

df = pd.concat([df_anomalies,df_normal])

df.head()
fig = plt.figure(figsize=(15,5))

ax=plt.subplot(2,2,1)

sns.boxplot(x = df['type'], y = df['balance'])

ax=plt.subplot(2,2,2)

sns.boxplot(x = df['type'], y = df['purchases'])

ax=plt.subplot(2,2,3)

sns.boxplot(x = df['type'], y = df['credit_limit'])

ax=plt.subplot(2,2,4)

sns.boxplot(x = df['type'], y = df['payments'])