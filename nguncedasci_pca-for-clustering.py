#import the data
#standardization
#apply PCA
from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.cluster import KMeans
df=pd.read_csv("../input/usarrests/USArrests.csv").copy()
df.index = df.iloc[:,0]
df = df.iloc[:,1:5]
df.index.name = None
df.head()
#standardization
from sklearn.preprocessing import StandardScaler
df=StandardScaler().fit_transform(df)
df[0:5,0:5]
from sklearn.decomposition import PCA
pca_fit=PCA(n_components=2).fit_transform(df)
pca_fit[0:5]
#apply PCA (ex:1)
component_df=pd.DataFrame(data= pca_fit, columns=["1st Component","2nd Component"])
component_df[0:5]
#Here,these 2 components represents 4 variables(Murder, Assault, UrbanPop,Rape)
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit_transform(df)
pca.explained_variance_ratio_
#It means, first component explains 62% of data variation, second component explains 24% of data variation. 
#Totally, 86% of the data variation can be explained by these two components
#If we accept the risk of some error, we can reduce the number of variables(4) to 2(components) as it i seen. 
#We can apply this method to more complex datasets.
#apply PCA (ex:2)
pca = PCA(n_components = 3)
pca_fit = pca.fit_transform(df)
component_df=pd.DataFrame(data= pca_fit, columns=["1st Component","2nd Component","3rd Component"])
component_df[0:5]
pca.explained_variance_ratio_
#It means, first component explains 62% of data variation.
#Second component explains 24% of data variation.
#Third component explains 8% of data variation
#Totally, 94% of the data variation can be explained by these three components
#How can we specify the number of components?
#We can specify the number of components according to these explained variance ratio...
pca=PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
#From this graph,
#We can understand that less than 65% of the data can be explained by one component,
#                       less than 90% of the data can be explained by two components,
#                       more than 95% of the data can be explained by three components,
#                                100% of the data can be explained by four components.
