#importing dataset from kaggle:
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing necessary libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram
# for interactive visualizations
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
init_notebook_mode(connected = True)
import plotly.figure_factory as ff
import plotly.graph_objects as go
df=pd.read_csv('/kaggle/input/young-people-survey/responses.csv')
df.head()
mapping=pd.read_csv('/kaggle/input/young-people-survey/columns.csv')
final=df[['Movies','Horror','Comedy','Romantic','Sci-fi','War','Documentary','Action','History','Countryside, outdoors','Celebrities','Science and technology','Art exhibitions','Alcohol','Healthy eating','Prioritising workload','Workaholism','Number of friends','Interests or hobbies','Internet usage','Energy levels','Finances','Socializing','Entertainment spending','Spending on gadgets','Spending on healthy eating','Age']]
final.info()
final.isna().sum()
final['Alcohol'].replace(['never', 'social drinker','drink a lot'], [0,3,5],inplace=True)
final['Internet usage'].replace(['few hours a day', 'most of the day', 'less than an hour a day','no time at all'], [5,3,1,0],inplace=True)
final.fillna(df.median(),inplace=True)
final['Alcohol']=final['Alcohol'].fillna(final['Alcohol'].median())
final=final.astype(int)
final.describe()
list=['Movies','Horror','Comedy','Romantic','Sci-fi','War','Documentary','Action','History','Countryside, outdoors','Celebrities','Science and technology','Art exhibitions','Alcohol','Healthy eating','Prioritising workload','Workaholism','Number of friends','Interests or hobbies','Internet usage','Energy levels','Finances','Socializing','Entertainment spending','Spending on gadgets','Spending on healthy eating','Age']
mapping.loc[mapping['short'].isin(list)].reset_index(drop=True)
final.head()
sns.set(rc={'figure.figsize':(15,10)})
ax=sns.heatmap(final.corr(),center=0.00,cmap=sns.diverging_palette(255, 133,as_cmap=True))
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);
sns.barplot(final['Romantic'],final['Science and technology'],palette="GnBu_d");
sns.barplot(final['Energy levels'],final['Number of friends'],palette=sns.dark_palette("purple"));
sns.lineplot(final['Age'],final['Prioritising workload'],palette="GnBu_d");
sns.kdeplot(final['Art exhibitions'],final['Science and technology']);
from sklearn.cluster import KMeans
x=final
wcss = []
for i in range(2, 9):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 3000, n_init = 10,random_state=0)
    km.fit(x)
    wcss.append(km.inertia_)
    
plt.plot(range(2, 9), wcss)
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()
km = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 1000, n_init = 30)
km.fit(x)
y_means=km.predict(x)

silhouette_score(x,y_means)
x=final[['Science and technology','Countryside, outdoors','Art exhibitions']]

scaler=StandardScaler()
scaler.fit(x)
wcss = []
#x=scaler.transform(x)
for i in range(2, 6):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter = 1000, n_init = 10,random_state=0)
    km.fit(x)
    y_means=km.predict(x)
    s=silhouette_score(x,y_means)
    wcss.append(s)
plt.plot(range(2, 6), wcss)
plt.title('No. of clusters vs Silhouette Score', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('Silhouette_Score')
plt.show()
km = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 1000, n_init = 30,random_state=2)
km.fit(x)
y_means=km.predict(x)
final['y_means']=y_means
final.groupby('y_means').mean()
def configure_plotly_browser_state():
  import IPython
  display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
            },
          });
        </script>
        '''))
  
trace1 = go.Scatter3d(
    x= final['Science and technology'],
    y= final['Countryside, outdoors'],
    z= final['Art exhibitions'],
    mode='markers',
     marker=dict(
        color = final['y_means'], 
        size= 10,
        line=dict(
            color= final['y_means'],
            width= 12
        ),
        opacity=0.9
     )
)
dg = [trace1]

layout = go.Layout(
    title = '3D visualization of clusters',
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0  
    ),
 
)
configure_plotly_browser_state()

fig = go.Figure(data = dg, layout = layout)

fig.show()
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogam', fontsize = 20)
plt.xlabel('Individuals')
plt.ylabel('Ecuclidean Distance')
plt.show()
from sklearn.cluster import AgglomerativeClustering
x=final[['Science and technology','Countryside, outdoors','Art exhibitions']]
hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean')
y_hc = hc.fit_predict(x)
silhouette_score(x,y_hc)
Art = input(" Enter your score here: ")
Outdoors = input(" Enter your score here: ")
Science = input(" Enter your score here: ")
test=[[Science,Outdoors,Art]]
y_means=km.predict(test)
y_means[0]
if y_means[0]==2:
  print("You are identified to be an all rounder!")
elif y_means[0]==3:
  print("You are identified to be a geek!")
elif y_means[0]==1:
  print("You are identified to be a hopeless romantic!")
else:
  print("You are identified to be an outcast!")