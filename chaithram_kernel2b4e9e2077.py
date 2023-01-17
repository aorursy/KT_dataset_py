
import numpy as np 
import pandas as pd 


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from pandas.plotting import scatter_matrix
from matplotlib import pyplot
data = pd.read_csv("../input/online-shoppers-intention/online_shoppers_intention.csv")
data.head(12)
print(data.shape)
print(data.describe())
# for basic operations
import numpy as np
import pandas as pd



# for data visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go

# for data analysis
import pandas_profiling as profile
#import dabl


missing_percentage = data.isnull().sum()/data.shape[0]
print(missing_percentage)
data['Revenue'].replace({True:1,False:0})
data['VisitorType'].replace({"Returning_Visitor":1,"New_Visitor":0},inplace=True)
data['Month'].replace({"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12},inplace=True)
data['Weekend'].replace({True:1,False:0},inplace=True)
rev=data['Revenue']
data.drop(columns={'Revenue'})
data.info()
# import warnings
# warnings.filterwarnings('ignore')
# plt.rcParams['figure.figsize'] = (15, 10)
# plt.style.use('fivethirtyeight')
# dabl.plot(data, target_col = 'Revenue')
# lets get the profile report for the data
#data.profile_report()

# preparing the dataset
x = data.iloc[:, [1, 6]].values

# checking the shape of the dataset
x.shape


from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i,
              init = 'k-means++',
              max_iter = 300,
              n_init = 10,
              random_state = 0,
              algorithm = 'elkan',
              tol = 0.001)
    km.fit(x)
    labels = km.labels_
    wcss.append(km.inertia_)
    
plt.rcParams['figure.figsize'] = (15, 7)
plt.plot(range(1, 11), wcss)
plt.grid()
plt.tight_layout()
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()
# informational duration vs Bounce Rates
x = data.iloc[:, [3, 6]].values

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i,
              init = 'k-means++',
              max_iter = 300,
              n_init = 10,
              random_state = 0,
              algorithm = 'elkan',
              tol = 0.001)
    km.fit(x)
    labels = km.labels_
    wcss.append(km.inertia_)
    
plt.rcParams['figure.figsize'] = (15, 7)
plt.plot(range(1, 11), wcss)
plt.grid()
plt.tight_layout()
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()

# Administrative duration vs Bounce Rates
x = data.iloc[:, [1, 7]].values

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i,
              init = 'k-means++',
              max_iter = 300,
              n_init = 10,
              random_state = 0,
              algorithm = 'elkan',
              tol = 0.001)
    km.fit(x)
    labels = km.labels_
    wcss.append(km.inertia_)
    
plt.rcParams['figure.figsize'] = (15, 7)
plt.plot(range(1, 11), wcss)
plt.grid()
plt.tight_layout()
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()

x = data.iloc[:, [13, 14]].values
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i,
              init = 'k-means++',
              max_iter = 300,
              n_init = 10,
              random_state = 0,
              algorithm = 'elkan',
              tol = 0.001)
    km.fit(x)
    labels = km.labels_
    wcss.append(km.inertia_)
    
plt.rcParams['figure.figsize'] = (15, 7)
plt.plot(range(1, 11), wcss)
plt.grid()
plt.tight_layout()
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()

#silhouette method

# n_clusters=[2,3,4,5,6,7,8,9,10]
# for n_cl in n_clusters:
#     cls=KMeans(n_cl)
#     cl_label=cluster.fit_predict(ps)
#     sil_avg=silhouette_score(ps,cl_lables)
#     print("For ",n_cl," clusters The average silhouette score is :",sil_avg)

