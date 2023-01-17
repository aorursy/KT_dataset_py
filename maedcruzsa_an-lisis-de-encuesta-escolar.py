# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/firstdataframe/firstdataframe.csv')

df
dic_sex = {'Hombre':0, 'Mujer':1}

df['Sexo'] =  df['Sexo'].map(dic_sex)
dic_trab = {'No':0, 'Sí':1}

df['Trabajo'] =  df['Trabajo'].map(dic_trab)

df
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(df)

scaled_features = scaler.transform(df)
scaled_data = pd.DataFrame(scaled_features, columns = df.columns)

scaled_data
fig, axs = plt.subplots(3, 2)



axs[0][0].scatter(scaled_data['Ut_est'],scaled_data['Ag_est'], marker = "x")

axs[0][0].set_xlabel('Utilidad/Agrado', labelpad = 5)



axs[0][1].scatter(scaled_data['Ag_est'],scaled_data['Promedio'], marker = "x")

axs[0][1].set_xlabel('Agrado/Promedio', labelpad = 5)



axs[1][0].scatter(scaled_data['Ut_est'],scaled_data['Promedio'], marker = "x")

axs[1][0].set_xlabel('Utilidad/Promedio', labelpad = 5)



axs[1][1].scatter(scaled_data['Hrs_est'],scaled_data['Promedio'], marker = "x")

axs[1][1].set_xlabel('Horas de estudio/Promedio', labelpad = 5)



axs[2][0].scatter(scaled_data['Hrs_est'],scaled_data['Ut_est'], marker = "x")

axs[2][0].set_xlabel('Horas de estudio/Promedio', labelpad = 5)



axs[2][1].scatter(scaled_data['Hrs_est'],scaled_data['Ag_est'], marker = "x")

axs[2][1].set_xlabel('Horas de estudio/Promedio', labelpad = 5)
df = pd.DataFrame (scaled_data, columns = ['Promedio','Hrs_est','Ing_men'])

df
from sklearn.neighbors import NearestNeighbors

neigh = NearestNeighbors()

nbrs = neigh.fit(df)

distances, indices = nbrs.kneighbors(df)

distances = np.sort(distances, axis=0)

distances = distances[:,1]

plt.plot(distances)
from sklearn.cluster import DBSCAN

model = DBSCAN(eps = 0.85, min_samples = 2)
clusters = model.fit_predict(df)

df['k'] = clusters

df
import plotly.express as px

fig = px.scatter_3d(df, x="Hrs_est", y="Promedio", z="Ing_men",color="k")

fig.update_layout(scene_zaxis_type="log")

fig.show()
n_clusters_ = len(set(df['k'])) - (1 if -1 in df['k'] else 0)

n_noise_ = list(df['k']).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)

print('Estimated number of noise points: %d' % n_noise_)