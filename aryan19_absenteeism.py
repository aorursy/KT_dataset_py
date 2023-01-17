import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 

init_notebook_mode(connected=True)  

import cufflinks as cf  

cf.go_offline() 

df = pd.read_csv('../input/Absenteeism_at_work.csv')

df.head()
df.isna().sum()
df.info()
df.describe()
df.iplot(kind='box')
cols = df.columns.tolist()

cols.pop(0)
for i in cols:

    print(i)

    df[i].iplot()
AvgR = df[cols[1:]].mean()

AvgR = AvgR.sort_values()

plt.figure(figsize=(10,7))

plt.barh(np.arange(len(cols[1:])), AvgR.values, align='center')

plt.yticks(np.arange(len(cols[1:])), AvgR.index)

plt.ylabel('Categories')

plt.xlabel('Average')

plt.title('Average')
sns.pairplot(df)
fig, ax = plt.subplots(figsize=(20, 20)) 

sns.heatmap(df.corr(), annot = True, ax = ax)
DF = df.copy()

DF.head()
vals = DF.iloc[ :, 1:].values



from sklearn.cluster import KMeans

wcss = []

for ii in range( 1, 30 ):

    kmeans = KMeans(n_clusters=ii, init="k-means++", n_init=10, max_iter=300) 

    kmeans.fit_predict( vals )

    wcss.append( kmeans.inertia_ )

    

plt.plot( wcss, 'ro-', label="WCSS")

plt.title("Computing WCSS for KMeans++")

plt.xlabel("Number of clusters")

plt.ylabel("WCSS")

plt.show()
X = df.drop(['ID'],axis=1).values

Y = df['ID'].values
km = KMeans(n_clusters=5, init="k-means++", n_init=10, max_iter=500) 

y_pred = kmeans.fit_predict(X)
DF["Cluster"] = y_pred

cols = list(DF.columns)

cols.remove("ID")



sns.pairplot( DF[cols], hue="Cluster")
import scipy.cluster.hierarchy as sch

from sklearn.preprocessing import scale as s

from scipy.cluster.hierarchy import dendrogram, linkage
def fd(*args, **kwargs):

    max_d = kwargs.pop('max_d', None)

    if max_d and 'color_threshold' not in kwargs:

        kwargs['color_threshold'] = max_d

    annotate_above = kwargs.pop('annotate_above', 0)



    ddata = dendrogram(*args, **kwargs)



    if not kwargs.get('no_plot', False):

        plt.title('Hierarchical Clustering Dendrogram (truncated)')

        plt.xlabel('sample index or (cluster size)')

        plt.ylabel('distance')

        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):

            x = 0.5 * sum(i[1:3])

            y = d[1]

            if y > annotate_above:

                plt.plot(x, y, 'o', c=c)

                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),

                             textcoords='offset points',

                             va='top', ha='center')

        if max_d:

            plt.axhline(y=max_d, c='k')

    return ddata
Z = sch.linkage(df,method='ward')  

den = sch.dendrogram(Z)

plt.tick_params(

    axis='x',          

    which='both',      

    bottom=False,     

    top=False,         

    labelbottom=False) 

plt.title('Hierarchical Clustering')
Z = linkage(df,method='ward')

fd(Z,leaf_rotation=90.,show_contracted=True,annotate_above=750,max_d=1250)

plt.tick_params(

    axis='x',          

    which='both',      

    bottom=False,     

    top=False,         

    labelbottom=False) 
Z = sch.linkage(df,method='complete')  

den = sch.dendrogram(Z)

plt.tick_params(

    axis='x',          

    which='both',      

    bottom=False,     

    top=False,         

    labelbottom=False) 
Z = linkage(df,method='complete')

fd(Z,leaf_rotation=90.,show_contracted=True,annotate_above=160,max_d=280)

plt.tick_params(

    axis='x',          

    which='both',      

    bottom=False,     

    top=False,         

    labelbottom=False) 
from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.preprocessing import scale as s

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import train_test_split as t

import sklearn.metrics as mt
x = df.drop(['ID','Absenteeism time in hours'],axis=1).values

y = df['Absenteeism time in hours'].values
x = s(x)

y = s(y)
train_x,test_x,train_y,test_y = t(x,y,test_size=0.2)
rfr = RandomForestRegressor(n_estimators=100,max_depth=4)

rfr.fit(train_x,train_y)

print(f'Score = {rfr.score(test_x,test_y)}')

print(f'MSE = {mt.mean_squared_error(test_y,rfr.predict(test_x))}')
lr = LinearRegression()

lr.fit(train_x,train_y)

print(f'Score = {mt.r2_score(test_y,lr.predict(test_x))}')

print(f'MSE = {mt.mean_squared_error(test_y,lr.predict(test_x))}')
knr = KNeighborsRegressor(n_neighbors=10)

knr.fit(train_x,train_y)

print(f'Score = {mt.r2_score(test_y,knr.predict(test_x))}')

print(f'MSE = {mt.mean_squared_error(test_y,knr.predict(test_x))}')
xgbr = XGBRegressor()

xgbr.fit(train_x,train_y)

print(f'Score = {xgbr.score(test_x,test_y)}')

print(f'MSE = {mt.mean_squared_error(test_y,xgbr.predict(test_x))}')
mlpr = MLPRegressor(hidden_layer_sizes=(100,50,1), max_iter=500)

mlpr.fit(train_x,train_y)

print(f'Score = {mlpr.score(test_x,test_y)}')

print(f'MSE = {mt.mean_squared_error(test_y,mlpr.predict(test_x))}')