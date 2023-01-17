from sklearn.datasets import load_boston
bostan=load_boston()
X,Y=bostan.data,bostan.target
print(X.mean(axis=0))
print(X.std(axis=0))
X.shape
import seaborn as sns
import pandas as pd
sns.heatmap(pd.DataFrame(X).corr())
#Import and instantiate
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(X)
#Apply to the data
X_reduced=pca.transform(X)
print("Reduction dataset shape:",X_reduced.shape)
print("Explained Variance Ratio:",pca.explained_variance_ratio_.sum())
sns.heatmap(pd.DataFrame(X_reduced).corr())
#Standardization
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X)
X_scaled=sc.transform(X)
#Perform PCA
pca_bostan=PCA(n_components=7)
pca_bostan.fit(X_scaled)
pca_bostan.explained_variance_ratio_.sum().round(2)

pca_bostan.explained_variance_ratio_
df=pd.DataFrame(data=pca_bostan.transform(X_scaled),columns=['PC_'+str(i) for i in range(0,pca_bostan.transform(X_scaled).shape[1])])
df.plot.scatter(x='PC_1',y='PC_2')
evr={}
def runPCA(n_comp=5,X_scaled=X_scaled):
    pca_n=PCA(n_comp)
    pca_n.fit(X_scaled)
    evr[n_comp]=pca_n.explained_variance_ratio_.sum()
for i in range(5,11):
    runPCA(i)
    
evr
pd.Series(evr).plot()
{x:runPCA(x) for x in range(5,13)}