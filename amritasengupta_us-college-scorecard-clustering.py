import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("../input/us-college-scorecard/CollegeScorecard.csv")
df.head()
df.shape
df= df.drop_duplicates()
df= df.replace(to_replace="PrivacySuppressed", value=np.nan)
x= df.isnull().sum()[df.isnull().sum()>=2341]
x.index
df.drop(columns=x.index,inplace= True)
df.shape
cat1= df.nunique()[df.nunique()==2]
df.drop(columns= cat1.index,inplace= True)
df.nunique()[df.nunique()<=20]
cat2= df.nunique()[df.nunique()==3]
cat2
cat2.head(50)
cat2.tail(50)
cat2[51:96]
cat2[97:141]
df.drop(columns= cat2.index, inplace= True)
df.shape
df.nunique()[df.nunique()<=20]
df.region.unique()
df.region.isnull().sum()
df.LOCALE.unique()
df.LOCALE.isnull().sum()
df.drop(columns= ['PREDDEG','HIGHDEG'], inplace=True)
df.shape
df.columns
df.drop(columns=['UNITID', 'OPEID', 'opeid6', 'INSTNM', 'CITY','STABBR', 'ZIP', 'AccredAgency', 'INSTURL', 'NPCURL'],inplace=True)
df.columns
df.drop(columns=['st_fips','LATITUDE', 'LONGITUDE'],inplace=True)
df.shape
null_val= df.isnull().sum()[df.isnull().sum()>0]
null_val.count()
col= null_val.index
v=0
for i in col:
    medians= df[i].median()
    df[i]= df[i].fillna(value=medians)
df.isnull().sum()[df.isnull().sum()>0]
df.LOCALE.unique()
X= df.values
from sklearn.decomposition import PCA

pca = PCA(n_components= 130)
pca_fit=pca.fit(X)
reduced_X = pca_fit.transform(X)

# 127 Columns present in X are now represented by 4 Principal components present in reduced_X
print(np.round(reduced_X[0:130],2))
var_explained= pca.explained_variance_ratio_
print(np.round(var_explained,2))
var_explained_cumulative=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print(var_explained_cumulative)
plt.plot( range(1,131), var_explained_cumulative )
plt.xlabel('Number of components')
plt.ylabel('% Variance explained')
plt.xlim(-1,15)
pca = PCA(n_components=4)

# fitting the data
pca_fit=pca.fit(X)

# calculating the principal components
reduced_X = pca_fit.transform(X)
#130 Columns present in X are now represented by 4-Principal components present in reduced_X
df2= pd.DataFrame(reduced_X, columns=['PC1','PC2','PC3','PC4'])
df2.head()
from sklearn.preprocessing import StandardScaler
PredictorScaler=StandardScaler()
x= df2.values
PredictorScalerFit=PredictorScaler.fit(x)

# Generating the standardized values 
x_scaled= PredictorScalerFit.transform(x)
df3= pd.DataFrame(x_scaled)
df3.columns=['PC1','PC2','PC3','PC4']
df3.head()
v= df3.values
# Finding the best number of clusters based on the inertia value
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
inertiaValue = []
for i in range(1, 15):
    km = KMeans(n_clusters=i, 
                init='random', 
                n_init=10, 
                max_iter=300,
                tol=1e-04,
                random_state=0)
    
    km.fit(v)
    inertiaValue.append(km.inertia_)
    
plt.plot(range(1, 15), inertiaValue, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.tight_layout()
plt.show()
# Defining the K-Means object for best number of clusters. n=9 in this case
km = KMeans(n_clusters= 5, 
            init='k-means++', 
            n_init=10, 
            max_iter=300,
            tol=1e-04,
            random_state=0)
predictedCluster = km.fit_predict(v)
print(predictedCluster)
print('Inertia:', km.inertia_)
df3['PredictedClusterID']=predictedCluster
df3.head()
%matplotlib inline
plt.scatter(x=df3['PC2'], y=df3['PC4'], c=df3['PredictedClusterID'])
df3['PredictedClusterID'].value_counts()
# DBSCAN automatically choosed the number of clusters based on eps and min_samples
from sklearn.cluster import DBSCAN
db = DBSCAN(eps= 1.5, min_samples=4)

df3['PredictedClusterID_2']=db.fit_predict(v)

print(df3.head())
plt.scatter(x=df3['PC2'], y=df3['PC4'], c=df3['PredictedClusterID_2'])
df3['PredictedClusterID_2'].value_counts()
