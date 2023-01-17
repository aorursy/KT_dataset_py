import pandas as pd

data= pd.read_csv("../input/wine-quality-clustering-unsupervised/winequality-red.csv")
data.head()
import seaborn as sns

sns.pairplot(data,diag_kind='kde',hue='quality')
from sklearn.cluster import KMeans

from scipy.stats import zscore
from sklearn.model_selection import train_test_split

X=data.drop('quality',axis=1)

y=data['quality']
kmeans=KMeans(n_clusters=5,n_init=15,random_state=2)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
X_std=sc.fit_transform(X)

y.shape
y.value_counts()
kmeans.fit(X_std)
centroid=kmeans.cluster_centers_
c_df=pd.DataFrame(centroid,columns=list(X))

c_df
# acidity is very low in 0 cluster ,very high in 1 cluster

# average in 1 cluster

# alcohol presentage is different 

# sugar also good feature for different

# citric acid also good feature we also observe in pairplot

#lable 0 has least alcohol quantity 

#lable 1 has higher quality content 

# lable 2 has less than 1 quality
data.boxplot(column=['total_sulfur_dioxide'],by='quality')
ypred=pd.DataFrame(kmeans.labels_)

ypred[0].value_counts()