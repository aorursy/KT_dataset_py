# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')

df.head()
df = df.drop('Serial No.', axis=1)

list(df)
gdf = df[['GRE Score', 'TOEFL Score','CGPA']]

mlabels = df['Chance of Admit ']

gdf.head(),mlabels.head()



labels = mlabels.map(lambda x : 'Accepted' if x>=0.72 else 'Rejected')

numlabels = mlabels.map(lambda x :  1 if x>=0.72 else 0)



gdf['label'] = labels

gdf['numlabels'] = numlabels

gdf.head(10)
def show(df, x, y, label):

    facet = sns.lmplot(data=df, x=x, y=y, hue=label,fit_reg=False, legend=True, legend_out=True )

    return facet

data = gdf

x, y, z = 'GRE Score', 'TOEFL Score', 'CGPA'

dx, dy, dz = gdf[x], gdf[y],  gdf[z]

lbl = 'label'

# y, x = x, y 

show(gdf,x,y,lbl)

show(gdf, y, z, lbl)

show(gdf, x,z,lbl)

sns.heatmap(data.corr(),annot=True,cmap='RdYlGn')

plt.figure(figsize=(10,10))

plt.show()
# def get_axes(ax, dx, dy, labels, colors=['#2300A8', '#00A658']):

#     for i in range(len(df)):

#         ax.scatter(dx.iloc[i], dy.iloc[i],alpha=1, color = colors[i%len(colors)], label=labels.iloc[i])

#         ax.legend(labels.unique())

#     return ax

# # gdf.plot.hist()

# fig = plt.figure(figsize=(10,10))



# ax1 = fig.add_subplot(221)

# ax2 = fig.add_subplot(222)



# ax1 = get_axes(ax1, dx, dy, labels)

# ax2 = get_axes(ax2, dx, dy, labels)

# # ax1.scatter(dx, dy, alpha=0.5, color = colors[i%len(colors)], label=labels.iloc[i])

# # ax2.scatter(dx,dy, c='r')



# plt.show()
gdf.head()
def kde(gdf, x,y):

    ax = sns.kdeplot(gdf[gdf['numlabels']==0][x], # <== 😀 Look here!

                     gdf[gdf['numlabels']==0][y],   # <== 😀 Look here!

                     cmap="Reds", 

                     shade=True, shade_lowest=False)



    ax = sns.kdeplot(gdf[gdf['numlabels']==1][x], # <== 😀 Look here!

                     gdf[gdf['numlabels']==1][y],   # <== 😀 Look here!

                     cmap="YlOrBr", 

                     shade=True, shade_lowest=False)

    plt.show()

kde(gdf, x, y)

kde(gdf, y, z)

kde(gdf, z, x)
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt







fig = plt.figure(figsize=(18,18))

ax = fig.add_subplot(111, projection='3d')

trs = gdf[gdf['numlabels']==1]

fls = gdf[gdf['numlabels']==0]

tdx, tdy, tdz = trs[x], trs[y], trs[z]

tlbls = trs['numlabels']

flbls = fls['numlabels']

fdx, fdy, fdz = fls[x], fls[y], fls[z]

 

colors = ['#2300A8', '#00A658']

ax.scatter(tdx, tdy, tdz, color='red', alpha=0.5,s=fdy, marker='v')

ax.scatter(fdx, fdy, fdz, color='#38FF59', s=fdy, alpha=0.4, marker='o')



# for i in range(len(df)):

#      ax.scatter(dx.iloc[i], dz.iloc[i], dy.loc[i], alpha=0.50, color = colors[i%len(colors)], label=labels.iloc[i])



ax.set_xlabel(x)

ax.set_ylabel(y)

ax.set_zlabel(z)

ax.legend(labels.unique())

plt.show()
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score

from sklearn.preprocessing import MinMaxScaler

from sklearn.cluster import KMeans



from keras.layers import Dense, Dropout

from keras.models import Sequential

from keras import regularizers

from keras.optimizers import Adam
X = gdf[['GRE Score', 'TOEFL Score', 'CGPA']]

y = gdf['numlabels'] 

normalizer = MinMaxScaler().fit(X)

trans_X  = normalizer.transform(X)

trans_X.shape

trans_X
model = Sequential()

model.add(Dense(len(list(X)), activation='relu', input_shape=(3,)))

model.add(Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.005)))

model.add(Dropout(.2))

model.add(Dense(10, activation='relu', kernel_regularizer=regularizers.l2(0.005)))

model.add(Dropout(.1))

model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.005)))

model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(trans_X,y, epochs=130)
preds = model.predict(trans_X)

preds = list(map(lambda x: 1 if x>=.5 else 0, preds) )

accuracy_score(y, preds)