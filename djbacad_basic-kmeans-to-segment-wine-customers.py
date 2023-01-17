import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns 

import plotly as py

import plotly.graph_objs as go

from sklearn.cluster import KMeans

import warnings

import os

warnings.filterwarnings("ignore")

py.offline.init_notebook_mode(connected = True)
df = pd.read_csv('../input/wine-customer-segmentation/Wine.csv',sep = ",", header = 0)
df.head()
df.tail()
df.info()
df.drop('Customer_Segment', axis = 1, inplace = True)
df.describe()
df.info()

sns.set_palette("GnBu_d")

plt.title("Missingess Map")

plt.rcParams['figure.figsize'] = (8.0, 5.0)

sns.heatmap(df.isnull(), cbar=False)
sns.set_palette("GnBu_d")

sns.pairplot(df)
plt.rcParams['figure.figsize'] = (15.0, 15.0)

plt.title("Correlation Plot")

sns.heatmap(df.corr())
from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant

df_numeric = add_constant(df)

VIF_frame = pd.Series([variance_inflation_factor(df_numeric.values, i) 

               for i in range(df_numeric.shape[1])], 

              index=df_numeric.columns).to_frame()



VIF_frame.drop('const', axis = 0, inplace = True) 

VIF_frame.rename(columns={VIF_frame.columns[0]: 'VIF'},inplace = True)

VIF_frame[~VIF_frame.isin([np.nan, np.inf, -np.inf]).any(1)]
df.drop('Flavanoids', axis = 1, inplace = True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df_copy = df.copy()

df_copy = pd.DataFrame(scaler.fit_transform(df), 

                                      index=df.index,

                                      columns=df.columns)

df_copy
X1 = df_copy.iloc[: , :].values

inertia = []

for n in range(1 , 11):

    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state = 823, algorithm='elkan') )

    algorithm.fit(X1)

    inertia.append(algorithm.inertia_)

plt.figure(1 , figsize = (15 ,6))

plt.plot(np.arange(1 , 11) , inertia , 'o')

plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)

plt.xlabel('Number of Clusters') , plt.ylabel('WCSS')

plt.title('Elbow Method Diagram')

plt.show()
algorithm = (KMeans(n_clusters = 3 ,init='k-means++', n_init = 10 ,max_iter=300, 

                        tol=0.0001,  random_state= 823, algorithm='elkan') )

algorithm.fit(X1)

labels = algorithm.labels_

centroids = algorithm.cluster_centers_

centroids
df['Cluster'] = labels

df_copy['Cluster'] = labels

df.head()
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state = 823)

df_dv = df.copy()

df_dv.drop('Cluster', axis = 1, inplace = True)

rfc.fit(df_dv,df['Cluster'])

features = df_dv.columns.tolist()

feature_value = rfc.feature_importances_

d = {'Features' : features, 'Values' : feature_value}

fi = pd.DataFrame(d).sort_values('Values', ascending = False).reset_index()

fi

plt.rcParams['figure.figsize'] = (20.0, 5.0)

ax = sns.barplot(x=fi['Features'], y = fi['Values'], data = fi, palette="Blues_d")
sns.pairplot(df[['Proline','OD280','Color_Intensity','Alcohol','Hue','Total_Phenols','Cluster']],palette = 'colorblind',hue='Cluster');
fig, axs = plt.subplots(ncols=3,nrows=3, figsize = (15,15))

sns.scatterplot(x="Alcohol", y="Proline", hue="Cluster",

                     palette = 'colorblind', data = df, legend = False, s = 200, ax=axs[0][0])

sns.scatterplot(x="Ash", y="Proline", hue="Cluster",

                     palette = 'colorblind', data = df, legend = False, s = 200, ax=axs[0][1])

sns.scatterplot(x="Magnesium", y="Proanthocyanins", hue="Cluster",

                     palette = 'colorblind', data = df, legend = False, s = 200,  ax=axs[0][2])

sns.scatterplot(x="Proanthocyanins", y="Color_Intensity", hue="Cluster",

                     palette = 'colorblind', data = df, legend = False, s = 200,  ax=axs[1][0])

sns.scatterplot(x="Proline", y="OD280", hue="Cluster",

                     palette = 'colorblind', data = df, legend = False, s = 200,  ax=axs[1][1])

sns.scatterplot(x="Nonflavanoid_Phenols", y="Proline", hue="Cluster",

                     palette = 'colorblind', data = df, legend = False, s = 200, ax=axs[1][2])

sns.scatterplot(x="Color_Intensity", y="Hue", hue="Cluster",

                     palette = 'colorblind', data = df, legend = False, s = 200,  ax=axs[2][0])

sns.scatterplot(x="Color_Intensity", y="Proline", hue="Cluster",

                     palette = 'colorblind', data = df, legend = False, s = 200,  ax=axs[2][1])

sns.scatterplot(x="Ash_Alcanity", y="Proline", hue="Cluster",

                     palette = 'colorblind', data = df, legend = False, s = 200, ax=axs[2][2])
fig, axs = plt.subplots(ncols=4,nrows=2, figsize = (20,10))

sns.boxplot(x="Cluster", y="Proline", palette = 'colorblind', data = df, ax=axs[0][0])

sns.stripplot(x='Cluster',y='Proline', palette = 'colorblind', data=df, jitter=True, ax=axs[0][0])



sns.boxplot(x="Cluster", y="Color_Intensity", palette = 'colorblind', data = df, ax=axs[0][1])

sns.stripplot(x='Cluster',y='Color_Intensity', palette = 'colorblind', data=df, jitter=True, ax=axs[0][1])



sns.boxplot(x="Cluster", y="OD280", palette = 'colorblind', data = df, ax=axs[0][2])

sns.stripplot(x='Cluster',y='OD280', palette = 'colorblind', data=df, jitter=True, ax=axs[0][2])



sns.boxplot(x="Cluster", y="Alcohol", palette = 'colorblind', data = df, ax=axs[0][3])

sns.stripplot(x='Cluster',y='Alcohol', palette = 'colorblind', data=df, jitter=True, ax=axs[0][3])



sns.violinplot(x="Cluster", y="Hue", palette = 'colorblind', data = df, ax=axs[1][0])

sns.violinplot(x="Cluster", y="Total_Phenols", palette = 'colorblind', data = df, ax=axs[1][1])

sns.violinplot(x="Cluster", y="Malic_Acid", palette = 'colorblind', data = df, ax=axs[1][2])

sns.violinplot(x="Cluster", y="Ash_Alcanity", palette = 'colorblind', data = df, ax=axs[1][3])
from pandas.plotting import parallel_coordinates

parallel_coordinates(df_copy[['Proline','OD280','Alcohol','Color_Intensity','Total_Phenols','Cluster']], "Cluster",  colormap = 'Accent')

plt.ioff()
df_grouped = df_copy.groupby('Cluster',as_index=False).mean()

df_grouped = df_grouped[['Proline','OD280','Alcohol','Color_Intensity','Total_Phenols','Hue','Cluster']]

df_grouped
# Libraries

import matplotlib.pyplot as plt

import pandas as pd

from math import pi

 

categories = ['Proline','OD280','Alcohol','Color_Intensity','Hue','Total_Phenols']

N = len(categories)

 

angles = [n / float(N) * 2 * pi for n in range(N)]

angles += angles[:1]

 

ax = plt.subplot(111, polar=True)



ax.set_theta_offset(pi / 2)

ax.set_theta_direction(-1)

 

plt.xticks(angles[:-1], categories)

 

ax.set_rlabel_position(0)

plt.yticks([-2,-1,0,1,2], ["-2","0","1","2"], color="grey", size=7)

plt.ylim(-1.8,1.8)



values=df_grouped.loc[0].drop('Cluster').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="Cluster 0")

ax.fill(angles, values, 'b', alpha=0.2, color = "#3475B9")

 

values=df_grouped.loc[1].drop('Cluster').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="Cluster 1")

ax.fill(angles, values, 'r', alpha=0.5, color = "#BAAC43")



values=df_grouped.loc[2].drop('Cluster').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="Cluster 1")

ax.fill(angles, values, 'r', alpha=0.2, color = "#4EA976")

 

# Add legend

plt.rcParams['figure.figsize'] = (10.0, 10.0)
data = go.Scatter3d(

    x= df['Proline'],

    y= df['OD280'],

    z= df['Alcohol'],

    mode='markers',

     marker=dict(

        color= df['Cluster'],

        size= 18,

        opacity=0.8,

        colorscale = 'Geyser'

     )

)

data = [data]

layout = go.Layout(

    title= 'Clusters',

    scene = dict(

            xaxis = dict(title  = 'Proline'),

            yaxis = dict(title  = 'OD280'),

            zaxis = dict(title  = 'Alcohol')

        )

)

fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)
data = go.Scatter3d(

    x= df['OD280'],

    y= df['Color_Intensity'],

    z= df['Alcohol'],

    mode='markers',

     marker=dict(

        color= df['Cluster'],

        size= 18,

        opacity=0.8,

        colorscale = 'Geyser'

     )

)

data = [data]

layout = go.Layout(

    title= 'Clusters',

    scene = dict(

            xaxis = dict(title  = 'OD280'),

            yaxis = dict(title  = 'Color_Intensity'),

            zaxis = dict(title  = 'Alcohol')

        )

)

fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)