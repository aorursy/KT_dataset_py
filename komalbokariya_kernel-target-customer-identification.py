# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans

import seaborn as sns

import plotly as py

import colorlover as cl

import plotly.graph_objs as go

py.offline.init_notebook_mode(connected = True)

from matplotlib import pyplot as plt

plt.style.use('ggplot')



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/Mall_Customers.csv", index_col=0)

df.head()
df.info()

df.dtypes

df.isnull().sum()

df.describe()

df=df.rename(columns={'Gender':'gender','Age':'age','Annual Income (k$)':'annual income','Spending Score (1-100)':'spending score'})

df['gender'].replace(['Female','Male'],[0,1],inplace=True)

df.head()



df.mean()

df.std()

dfsp=pd.concat([df.mean(),df.std()],axis=1).transpose()



dfsp.index = ['mean', 'std']

dfsp



df_scaled=pd.DataFrame()

for c in df.columns:

    if(c=='gender'): df_scaled[c] = df[c]

    else: df_scaled[c] = (df[c] - dfsp.loc['mean',c]) / dfsp.loc['std',c]

df_scaled.head()

  
dff=df_scaled.loc[df_scaled.gender==0].iloc[:,1:]

dff

dfm=df_scaled.loc[df_scaled.gender==1].iloc[:,1:]

dfm

def no_of_clusters(df):

    inertia = []

    for i in range(1,20):

        km=KMeans(n_clusters=i,random_state=0)

        km.fit(dff)

        inertia.append(km.inertia_)

        df_elbow=pd.DataFrame(inertia)

        df_elbow=df_elbow.reset_index()

        df_elbow

        df_elbow.columns= ['n_clusters', 'within_cluster_sum_of_square']

        df_elbow

    return df_elbow



dff_elbow=no_of_clusters(dff)

dfm_elbow=no_of_clusters(dfm)

dfm_elbow



fig,ax=plt.subplots(1,2,figsize=(17,5))



sns.lineplot(data=dff_elbow,x='n_clusters',y='within_cluster_sum_of_square',ax=ax[0])

sns.scatterplot(data=dff_elbow[5:6],x='n_clusters',y='within_cluster_sum_of_square',color='black',ax=ax[0])

ax[0].set(xticks=dff_elbow.index)

ax[0].set_title('Female')



sns.lineplot(data=dfm_elbow,x='n_clusters',y='within_cluster_sum_of_square',ax=ax[1])

sns.scatterplot(data=dfm_elbow[5:6],x='n_clusters',y='within_cluster_sum_of_square',color='black',ax=ax[1])

ax[1].set(xticks=dfm_elbow.index)

ax[1].set_title('Male')



























def k_means(n_clusters, df, gender):



    kmf = KMeans(n_clusters=n_clusters, random_state=0) #defining the algorithm

    kmf.fit_predict(df) #fitting and predicting

    centroids = kmf.cluster_centers_ #extracting the clusters' centroids

    cdf = pd.DataFrame(centroids, columns=df.columns) #stocking in dataframe

    cdf['gender'] = gender

    return cdf



df1 = k_means(5, dff, 'female')

df2 = k_means(5, dfm, 'male')

dfc_scaled = pd.concat([df1, df2])

dfc_scaled

#UNSCALING

dfc = pd.DataFrame()

for c in dfc_scaled.columns:

    if(c=='gender'): dfc[c] = dfc_scaled[c]

    else: 

        dfc[c] = (dfc_scaled[c] * dfsp.loc['std', c] + dfsp.loc['mean', c])

        dfc[c] = dfc[c].astype(int)

        

dfc
dfcf = dfc[dfc.gender=='female']

dfcf

dfcm = dfc[dfc.gender=='male']

dfcm

dfcf.dtypes
def plot(dfs, names,colors, title):



    data_to_plot = []

    

    for i, df in enumerate(dfs):



        x = df['spending score']

        y = df['annual income']

        z = df['age']

        data = go.Scatter3d(x=x , y=y , z=z , mode='markers', name=names[i], marker = colors[i])

        data_to_plot.append(data)

        layout = go.Layout(margin=dict(l=0,r=0,b=0,t=40),

        title= title, scene = dict(xaxis = dict(title  = x.name,), 

        yaxis = dict(title  = y.name), zaxis = dict(title = z.name)))



    fig = go.Figure(data=data_to_plot, layout=layout)

    py.offline.iplot(fig)

    purple = dict(color=cl.scales['9']['seq']['RdPu'][3:8])

    blue   = dict(color=cl.scales['9']['seq']['Blues'][3:8])

    

plot([dfcf, dfcm], names=['male', 'female'], colors=[purple, blue], title = 'Clusters - All Targets')

  

        
dfc=dfc[(dfc['annual income'] > 40) & (dfc['spending score'] > 40)]

dfc=dfc.sort_values('age').reset_index(drop=True)

dfc





dfcf = dfc[dfc.gender=='female']

dfcm = dfc[dfc.gender=='male']

purple = dict(color=cl.scales['9']['seq']['RdPu'][3:8])

blue = dict(color=cl.scales['9']['seq']['Blues'][3:8])

plot([dfcf, dfcm], names=['male', 'female'], colors=[purple, blue], title = 'Clusters - Primary Targets')
df1 = dfc.iloc[[0], :]

df1

df2 = dfc.iloc[[1,2], :]

df2

df3 = dfc.iloc[[3,4], :]

df3



names = ['younger women - moderated spenders', 'rich & independant young adults', 'parents - moderated spenders']



colors = []

for i in [1, 3, 5]: 

    colors.append(dict(color = cl.scales['11']['qual']['Paired'][i]))



plot([df1, df2, df3], names=names, colors=colors, title = 'Marketing Clusters - Primary Targets')
