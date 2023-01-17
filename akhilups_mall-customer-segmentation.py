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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import scipy.stats as stats

import pandas_profiling   #need to install using anaconda prompt (pip install pandas_profiling)



%matplotlib inline

plt.rcParams['figure.figsize'] = 10, 7.5

plt.rcParams['axes.grid'] = True



import warnings

warnings.filterwarnings('ignore')





import missingno as msno

plt.style.use( 'ggplot' )



from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans



# center and scale the data

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
malldata = pd.read_csv("/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv",index_col='CustomerID')
malldata
malldata.rename(columns = {'Annual Income (k$)':'Annual_Income'}, inplace = True) 

malldata.rename(columns = {'Spending Score (1-100)':'Spending_Score'}, inplace = True) 
pandas_profiling.ProfileReport(malldata)
malldata.hist(figsize=(18,18));
malldata.info()
numeric_var_names=[key for key in dict(malldata.dtypes) if dict(malldata.dtypes)[key] in ['float64', 'int64', 'float32', 'int32']]

cat_var_names=[key for key in dict(malldata.dtypes) if dict(malldata.dtypes)[key] in ['object']]

print(numeric_var_names)

print(cat_var_names)
mall_num=malldata[numeric_var_names]

mall_num.head(5)
mall_cat = malldata[cat_var_names]

mall_cat.head(5)
#Handling Outliers - Method2

def outlier_capping(x):

    x = x.clip(upper=x.quantile(0.99))

    x = x.clip(lower=x.quantile(0.01))

    return x



mall_num=mall_num.apply(lambda x: outlier_capping(x))
#Handling missings - Method2

def Missing_imputation(x):

    x = x.fillna(x.median())

    return x



mall_num=mall_num.apply(lambda x: Missing_imputation(x))
def miss_treat_cat(x):

    x = x.fillna(x.mode())

    return x
mall_cat_new = mall_cat.apply(miss_treat_cat)
cat_dummies = pd.get_dummies(mall_cat_new, drop_first=True)
data_new = pd.concat([mall_num, cat_dummies], axis=1)
data_new.corr()
# visualize correlation matrix in Seaborn using a heatmap

sns.heatmap(data_new.corr())
data_new.columns
sc=StandardScaler()

#sc.fit()

#sc.transform()



mall_scaled=sc.fit_transform(data_new)
pd.DataFrame(mall_scaled).head()
pc = PCA(n_components=4)
pc.fit(mall_scaled)
pc.explained_variance_
sum(pc.explained_variance_)
var= pc.explained_variance_ratio_

var
var1=np.cumsum(np.round(pc.explained_variance_ratio_, decimals=4)*100)

var1
from sklearn import metrics

k_range = range(3, 20)

scores = []

for k in k_range:

    km = KMeans(n_clusters=k, random_state=123)

    km.fit(mall_scaled)

    scores.append(metrics.silhouette_score(mall_scaled, km.labels_))
scores
# plot the results

plt.plot(k_range, scores)

plt.xlabel('Number of clusters')

plt.ylabel('Silhouette Coefficient')

plt.grid(True)
cluster_range = range( 1, 20 )

cluster_errors = []



for num_clusters in cluster_range:

    clusters = KMeans( num_clusters )

    clusters.fit( mall_scaled )

    cluster_errors.append( clusters.inertia_ )
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )



clusters_df[0:20]
# allow plots to appear in the notebook

%matplotlib inline

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))

plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
kmeans = KMeans(n_clusters=10)
best_cols = ["Age", "Annual_Income", "Spending_Score","Gender_Male" ]



kmeans = KMeans(n_clusters=10, init="k-means++", n_init=10, max_iter=300) 

best_vals = data_new[best_cols].iloc[ :, :].values

y_pred = kmeans.fit_predict( best_vals )



data_new["cluster"] = y_pred

best_cols.append("cluster")

sns.pairplot( data_new[ best_cols ], hue="cluster");
data_new.head()