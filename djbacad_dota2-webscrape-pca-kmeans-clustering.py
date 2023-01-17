import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly as py

import plotly.graph_objs as go

from sklearn.cluster import KMeans

%matplotlib inline

from urllib.request import urlopen

from bs4 import BeautifulSoup

import warnings

import os

warnings.filterwarnings("ignore")

py.offline.init_notebook_mode(connected = True)
url = "https://dota2.gamepedia.com/Table_of_hero_attributes"

html = urlopen(url)
soup = BeautifulSoup(html, 'lxml')

type(soup)
title = soup.title

print(title)
rows = soup.find_all('tr')



all_rows = []

for row in rows:

    row_td = row.find_all('td')

    row_td = str(row_td)

    row_td = BeautifulSoup(row_td, "lxml").get_text()

    row_td = row_td.replace("\r","")

    row_td = row_td.replace(" ","")

    row_td = row_td.replace("\n","")

    all_rows.append(row_td)



print(all_rows[:10])
contents = pd.DataFrame(all_rows)

contents.head(10)
contents = contents[0].str.split(',', expand=True)

contents.head(10)
contents[0] = contents[0].str.strip('[]')

contents.head(10)
contents[28] = contents[28].str.strip('[]')

contents.head(10)
col_labels = soup.find_all('th')

all_header = []

col_str = str(col_labels)

col_str = col_str.replace("\r","")

col_str = col_str.replace(" ","")

col_str = col_str.replace("\n","")

col_str = col_str.replace("[","")

col_str = col_str.replace("]","")

cleantext2 = BeautifulSoup(col_str, "lxml").get_text()

all_header.append(cleantext2)

print(all_header)
headers = pd.DataFrame(all_header)

headers.head()
headers = headers[0].str.split(',',expand=True)

headers.head()
frames = [headers,contents]

final = pd.concat(frames)

final.head()
final = final.rename(columns=final.iloc[0])

final.head()
unnecessary = list(range(120,136))

print(unnecessary)
#Dropping unnecessary rows

unnecessary = list(range(120,136))

final.drop(0, axis = 0, inplace = True) #inplace = True to overwrite

final.drop(unnecessary, axis = 0, inplace = True)

#Dropping unnecesasry columns

final.drop(final.columns[29], axis = 1, inplace = True) 
final.head()
### Getting the MAINATT (because MAINATT is on images in the website) #CHEATSHEET - https://www.crummy.com/software/BeautifulSoup/bs4/doc/

MAINATT = []

main_attribute = soup.select('table tr td:nth-of-type(2) > a')

for i in main_attribute:

    MAINATT.append(i["title"])



final.drop("A", axis = 1, inplace = True)

final['MAINATT'] = MAINATT

final.head()
final.tail()
final.to_csv('dota2.csv', index = False)

df = final
df.info()
df['MAINATT'] = df['MAINATT'].map({'Intelligence':0, 'Strength':1, 'Agility': 2})

print(df)
df.drop('HERO', axis =1, inplace = True)

df[df.columns] = df[df.columns].astype('float')

df['MAINATT'] = df['MAINATT'].astype('int')

df['MAINATT'] = df['MAINATT'].astype('category')

df.info()
colors = {0: "#4CA9FF", 1: "#FF4343", 2: "#2FFE64"}

df1 = df.loc[:, 'STR':'INT+']

df1['MAINATT'] = df['MAINATT']

sns.pairplot(df1,palette = colors, hue = 'MAINATT')
df2 = df.loc[:, 'INT30':'DMG(MAX)']

df2['MAINATT'] = df['MAINATT']

sns.pairplot(df2,palette = colors, hue = 'MAINATT')
#Commenting out because notebook size limit is reached

#sns.distributions._has_statsmodels = False

#df3 = df.loc[:, 'RG':'ATKBS']

#df3['MAINATT'] = df['MAINATT']

#sns.pairplot(df3,palette = colors, hue = 'MAINATT',diag_kind = "kde")
sns.distributions._has_statsmodels = True

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
df.drop('DMG(MIN)', axis = 1, inplace = True)

df.drop('DMG(MAX)', axis = 1, inplace = True)
dv = pd.DataFrame(df['MAINATT'])

df.drop('MAINATT',axis=1,inplace=True)

df.info()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df = pd.DataFrame(scaler.fit_transform(df), 

                                      index = df.index,

                                      columns = df.columns)

df.head()
from sklearn.decomposition import PCA

pca = PCA().fit(df)

d = {'Proportion of Variance':pca.explained_variance_ratio_.tolist(),

    'Cumulative Proportion':np.cumsum(pca.explained_variance_ratio_).tolist()

    }

i = range(1,len(pca.explained_variance_ratio_.tolist())+1)

pc_no = ['PC ' + str(e) for e in i]

ImpOfComp = pd.DataFrame(d,index = pc_no)

ImpOfComp.style.format("{:.3f}")
plt.figure(figsize=(15,6)) #Adjust as necessary

sns.set_style("whitegrid")

ax = sns.lineplot(x = range(1,len(pc_no)+1), y = "Cumulative Proportion", data=ImpOfComp)

ax.set(xlabel='Number of Principal Components', ylabel='Cumulative Explained Variance')

plt.rcParams["axes.labelsize"] = 15
i = range(1,len(pca.explained_variance_ratio_.tolist())+1)

pc_no = ['PC ' + str(e) for e in i]

lc = pd.DataFrame(pca.components_,columns = df.columns,index = pc_no)

final_lc = lc[0:13]

final_lc
n_pca = PCA(n_components = 13)

x_pca = n_pca.fit_transform(df)

df_pca = pd.DataFrame(x_pca,columns=final_lc.index)

df_pca.head(12)
pcomps = abs(pd.DataFrame(n_pca.components_,columns=df.columns))

plt.figure(figsize=(15,8))

sns.heatmap(pcomps,cmap='YlGnBu')
X1 = df_pca.iloc[: , :].values

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

df_pca['Cluster'] = labels

##### Mapping the values to match the groups of original dataset (0 - Int, 1 - Str , 2 - Agi)

df_pca['Cluster'] = df_pca['Cluster'].map({0:1, 2:0, 1:2})

df_pca.head()
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state = 823)

df_dv = df_pca.copy()

df_dv.drop('Cluster', axis = 1, inplace = True)

rfc.fit(df_dv,df['Cluster'])

features = df_dv.columns.tolist()

feature_value = rfc.feature_importances_

d = {'Features' : features, 'Values' : feature_value}

fi = pd.DataFrame(d).sort_values('Values', ascending = False).reset_index()

fi

plt.rcParams['figure.figsize'] = (20.0, 5.0)

ax = sns.barplot(x=fi['Features'], y = fi['Values'], data = fi, palette="Greens_d")
sns.pairplot(df_pca[['PC 1','PC 2','PC 4','Cluster']], palette = colors ,hue='Cluster');
fig, axs = plt.subplots(ncols=4,nrows=2, figsize = (20,10))

sns.boxplot(x = "Cluster", y = "PC 1", palette = colors, data = df_pca, ax=axs[0][0])

sns.stripplot(x='Cluster',y='PC 1', palette = colors, data = df_pca, jitter=True, ax=axs[0][0])



sns.boxplot(x="Cluster", y="PC 2", palette = colors, data = df_pca, ax=axs[0][1])

sns.stripplot(x='Cluster',y='PC 2', palette = colors, data = df_pca, jitter=True, ax=axs[0][1])



sns.boxplot(x="Cluster", y="PC 4", palette = colors, data = df_pca, ax=axs[0][2])

sns.stripplot(x='Cluster',y='PC 4', palette = colors, data = df_pca, jitter=True, ax=axs[0][2])



sns.boxplot(x="Cluster", y="PC 10", palette = colors, data = df_pca, ax=axs[0][3])

sns.stripplot(x='Cluster',y='PC 10', palette = colors, data = df_pca, jitter=True, ax=axs[0][3])



sns.violinplot(x="Cluster", y="PC 1", palette = colors, data = df_pca, ax=axs[1][0])

sns.violinplot(x="Cluster", y="PC 2", palette = colors, data = df_pca, ax=axs[1][1])

sns.violinplot(x="Cluster", y="PC 4", palette = colors, data = df_pca, ax=axs[1][2])

sns.violinplot(x="Cluster", y="PC 10", palette = colors, data = df_pca, ax=axs[1][3])
df_pca_grouped = df_pca.groupby('Cluster',as_index=False).mean()

df_pca_grouped = df_pca_grouped[['PC 1','PC 2','PC 4','PC 10','Cluster']]

df_pca_grouped
# Libraries

import matplotlib.pyplot as plt

import pandas as pd

from math import pi

 

categories = ['PC 1','PC 2','PC 4','PC 10']

N = len(categories)

 

angles = [n / float(N) * 2 * pi for n in range(N)]

angles += angles[:1]

 

ax = plt.subplot(111, polar=True)



ax.set_theta_offset(pi / 2)

ax.set_theta_direction(-1)

 

plt.xticks(angles[:-1], categories)

 

ax.set_rlabel_position(0)

plt.yticks([-3,-1.5,0,1.5,3], ["-3","-1.5","0","1.5","3"], color="grey", size=7)

plt.ylim(-3,3)



values=df_pca_grouped.loc[0].drop('Cluster').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="Cluster 0")

ax.fill(angles, values, 'b', alpha=0.2, color = "#4CA9FF")

 

values=df_pca_grouped.loc[1].drop('Cluster').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="Cluster 1")

ax.fill(angles, values, 'r', alpha=0.5, color = "#FF4343")



values=df_pca_grouped.loc[2].drop('Cluster').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', label="Cluster 1")

ax.fill(angles, values, 'r', alpha=0.2, color = "#2FFE64")

 

# Add legend

plt.rcParams['figure.figsize'] = (10.0, 10.0)
data = go.Scatter3d(

    x= df_pca['PC 1'],

    y= df_pca['PC 2'],

    z= df_pca['PC 4'],

    mode='markers',

     marker=dict(

        color= df_pca['Cluster'],

        size= 18,

        opacity=0.8,

        colorscale = 'Geyser'

     )

)

data = [data]

layout = go.Layout(

    title= 'Clusters',

    scene = dict(

            xaxis = dict(title  = 'PC 1'),

            yaxis = dict(title  = 'PC 2'),

            zaxis = dict(title  = 'PC 4')

        )

)

fig = go.Figure(data=data, layout=layout)

py.offline.iplot(fig)
df_original = pd.read_csv('./dota2.csv',sep = ",", header = 0)
df_original['MAINATT']
df_pca['Cluster'] = df_pca['Cluster'].map({1:"Strength", 0:"Intelligence", 2:"Agility"})

df_pca['Cluster']
from sklearn.metrics import classification_report,confusion_matrix

data = confusion_matrix(df_original['MAINATT'], df_pca['Cluster'])

df_cm = pd.DataFrame(data, columns=np.unique(df_original['MAINATT']), index = np.unique(df_original['MAINATT']))

df_cm.index.name = 'Predicted'

df_cm.columns.name = 'Actual'

plt.figure(figsize = (11,8))

sns.set(font_scale=1.5)

ax = sns.heatmap(df_cm,cmap = 'Purples', annot=True,annot_kws={"size": 16})# font size

ax.set_title('Confusion Matrix')

#For cmaps:

#https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html

print("Classification Report: ")

print(classification_report(df_original['MAINATT'], df_pca['Cluster']))
df_pca['Original'] = df_original['MAINATT']

df_compare = df_pca.loc[:, 'Cluster':'Original']

df_compare['HERO'] = df_original['HERO']

df_compare.loc[df_compare['Cluster'] != df_compare['Original']]