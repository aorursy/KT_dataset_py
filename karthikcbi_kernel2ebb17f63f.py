import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last"
df=pd.read_csv("/kaggle/input/fundamentals.csv")
pd.options.display.max_columns=100
#set maximum columns to display in the output

pd.options.display.max_rows=100
#set maximum rows to display in the output
df.head()
df.columns
df.columns = df.columns.str.replace(" ", "_")
df.columns = df.columns.str.replace("&", "_")
df.columns = df.columns.str.replace("/", "_")
df.columns = df.columns.str.replace(".", "")
df.columns = df.columns.str.replace("-", "_")
df.columns = df.columns.str.replace("'", "")
df.columns = df.columns.str.replace(",", "")
df.columns = df.columns.str.replace(":", "")
df.columns = df.columns.str.replace("/", "_")

df.columns =  re.sub("_+","_",",".join(df.columns)).split(",")

df.columns
grp_tickers=df.groupby("Ticker_Symbol")
df_tickers=grp_tickers.mean()
df_tickers.head()
df_Gross = df_tickers.sort_values('Gross_Profit')
plt.figure(figsize=(50,20))
plot=sns.barplot(x=df_Gross.head(30).index,y='Gross_Profit',data=df_Gross.head(30))
plot.set_xticklabels(plot.get_xticklabels(),rotation=90,fontsize=75)
plot.set_yticklabels(plot.get_yticks(),fontsize=55)
plot.set_title("Bar Plot for least 30 Gross_Profit across tickers",fontsize=75)
plot.set_xlabel('Ticker Symbol',fontsize=75)
plot.set_ylabel('Gross_Profit',fontsize=75)
plt.figure(figsize=(50,20))
plot=sns.barplot(x=df_Gross.tail(30).index,y='Gross_Profit',data=df_Gross.tail(30))
plot.set_xticklabels(plot.get_xticklabels(),rotation=90,fontsize=75)
plot.set_yticklabels(plot.get_yticks(),fontsize=55)
plot.set_title("Bar Plot for top 30 Gross_Profit across tickers",fontsize=75)
plot.set_xlabel('Ticker Symbol',fontsize=75)
plot.set_ylabel('Gross_Profit',fontsize=75)
df_tickers['Gross_Cat']=pd.qcut(
                       df_tickers['Gross_Profit'],
                        q = 3,
                       labels = ["s", "m", "l"]
                      )
df_tickers['Gross_Cat'].value_counts()
df_tickers[df_tickers['Gross_Cat']=='l'].Gross_Profit.describe()
df_tickers[df_tickers['Gross_Cat']=='m'].Gross_Profit.describe()
df_tickers[df_tickers['Gross_Cat']=='s'].Gross_Profit.describe()
fig = plt.figure(figsize = (10,10))
sns.boxplot(x="Gross_Cat",y="Gross_Profit",data=df_tickers)
plt.figure(figsize=(50,20))
plot=sns.barplot(x=df_tickers.Gross_Cat,y='Gross_Profit',data=df_tickers)
plot.set_xticklabels(plot.get_xticklabels(),fontsize=75)
plot.set_yticklabels(plot.get_yticks(),fontsize=55)
plot.set_title("Bar Plot for mean of Gross_Profit across Gross_Cat",fontsize=75)
plot.set_xlabel('Gross_Cat',fontsize=75)
plot.set_ylabel('Gross_Profit',fontsize=75)
df_tickers.shape
df_tickers1=df_tickers.copy()
df_tickers=df_tickers.dropna(axis='columns')
df_tickers.drop(['Unnamed_0'], axis = 'columns', inplace = True)
df_tickers.shape
scaler = StandardScaler()
#df1 temporarily holds numeric only columns from df_tickers
df1 = df_tickers.select_dtypes(include = ['float64', 'int64']).copy()
#df2 is transformed numpy array of the numeric data
dfarray=scaler.fit_transform(df1)
dfnum=pd.DataFrame(dfarray,columns=df1.columns)
dfnum.head()
inertia=[]

for i in range(1,11):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(dfnum)
    inertia.append(kmeans.inertia_)
    
plt.plot(range(1,11),inertia,marker='*')
for i in range(2,11):
    kmeans = KMeans(n_clusters = i)
    clust = kmeans.fit(dfnum)
    print("Number of Clusters:", i, "Silhouette Score:", silhouette_score(dfnum, clust.labels_))
kmeans = KMeans(n_clusters=2)
pred_y = kmeans.fit(dfarray)
plt.scatter(dfarray[:,0], dfarray[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red')
plt.show()
kmeans = KMeans(n_clusters=3)
y_means = kmeans.fit(dfarray)
plt.scatter(dfarray[:,0], dfarray[:,1])
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red')
plt.show()
from yellowbrick.cluster import SilhouetteVisualizer
visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
visualizer.fit(dfnum)        
visualizer.show()  
from sklearn.manifold import TSNE
tsne_array = TSNE(n_components=2).fit_transform(dfnum)
df_tsne = pd.DataFrame(tsne_array, columns=['X','Y'])
colorlist = ["#FF1111", "#11FF11", "#1111FF"]
sns.relplot(x = "X",
            y = "Y",
            hue = y_means.labels_,
            data = df_tsne,
            palette=sns.color_palette(colorlist)
            )

from yellowbrick.cluster import InterclusterDistance
visualizer = InterclusterDistance(kmeans)
visualizer.fit(dfnum)        # Fit the data to the visualizer
visualizer.show()  
