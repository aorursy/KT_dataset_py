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
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

df = pd.read_csv('../input/internship-dataset/assignment1.csv', engine ='python')
df #having a look on dataset
df.info()
df.describe()
nullarray=df.isnull()
nullarray

df['campaign_platform']=pd.factorize(df.campaign_platform)[0] # convwerasiongoogle ad =0 and facbook ad =1
df.sort_values('campaign_platform')
plt.figure(figsize=(15,6))
sns.countplot(x='audience_type',data=df,order=df['audience_type'].value_counts().sort_values().index);
 
plt.figure(figsize=(15,6))
sns.countplot(x='campaign_platform',data=df,order=df['campaign_platform'].value_counts().sort_values().index);
plt.figure(figsize=(15,6))
sns.countplot(x='campaign_type',data=df,order=df['campaign_type'].value_counts().sort_values().index);
plt.figure(figsize=(15,6))
sns.countplot(x='subchannel',data=df,order=df['subchannel'].value_counts().sort_values().index);
plt.figure(figsize=(15,6))
sns.countplot(x='creative_name',data=df,order=df['creative_name'].value_counts().sort_values().index);
plt.figure(figsize=(15,6))
sns.countplot(x='creative_type',data=df,order=df['creative_type'].value_counts().sort_values().index);
plt.figure(figsize=(6,6))
sns.countplot(x='device',data=df,order=df['device'].value_counts().sort_values().index);
plt.figure(figsize=(10,6))
sns.countplot(x ='age',data=df,order=df['age'].value_counts().sort_values().index);
plt.figure(figsize=(10,6))
sns.countplot(x ='communication_medium',data=df,order=df['communication_medium'].value_counts().sort_values().index);
df['campaign_platform']=pd.factorize(df.campaign_platform)[0] # convwerasiongoogle ad =0 and facbook ad =1
df['campaign_type']=pd.factorize(df.campaign_type)[0] #conversion of conversions =0. and search =1
df['subchannel'] = pd.factorize(df.subchannel)[0]
df['device']=pd.factorize(df.device)[0]
df['communication_medium'] = pd.factorize(df.communication_medium)[0]
df['audience_type'] = pd.factorize(df.audience_type)[0]
df['creative_name'] = pd.factorize(df. creative_name)[0]
df['creative_type'] = pd.factorize(df.creative_type)[0]
df['creative_name'] = pd.factorize(df. creative_name)[0]
df.head()
df.drop(["Date","product","phase"], axis = 1, inplace = True) 
df['age'].replace(
    to_replace=['Undetermined'],
    value='0',
    inplace=True
)
df['audience_type'].replace(
    to_replace=["'-"],
    value='0',
    inplace=True
)
df['creative_type'].replace(
    to_replace=["'-"],
    value='0',
    inplace=True
)
df['creative_name'].replace(
    to_replace=["'-"],
    value='0',
    inplace=True
)
df['age'].replace(
    to_replace=['65 or more', '55-64','18-24', '45-54','35-44','25-34'],
    value=['65','58.5','21','49.5','39.5','29.5'],
    inplace=True
)
df.dropna()
df.dropna()
df.shape
df
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.preprocessing import StandardScaler, normalize 
from sklearn.metrics import silhouette_score 
import scipy.cluster.hierarchy as shc 

df.fillna(method ='ffill', inplace = True) 
df.head(2)
from sklearn.model_selection import train_test_split
# Standardize data
scaler = StandardScaler() 
scaled_df = scaler.fit_transform(df) 
  
# Normalizing the Data 
normalized_df = normalize(scaled_df) 
# Converting the numpy array into a pandas DataFrame 
normalized_df = pd.DataFrame(normalized_df)
# Reducing the dimensions of the data 
pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(normalized_df) 
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2'] 
  
X_principal.head(2)
plt.figure(figsize =(6, 6)) 
plt.title('Visualising the data') 
Dendrogram = shc.dendrogram((shc.linkage(X_principal, method ='ward')))


ac2 = AgglomerativeClustering(n_clusters = 2) 

  
# Visualizing the clustering 

plt.figure(figsize =(6, 6)) 

plt.scatter(X_principal['P1'], X_principal['P2'],  

           c = ac2.fit_predict(X_principal), cmap ='rainbow') 
plt.show() 


ac3 = AgglomerativeClustering(n_clusters = 3) 

  

plt.figure(figsize =(6, 6)) 

plt.scatter(X_principal['P1'], X_principal['P2'], 

           c = ac3.fit_predict(X_principal), cmap ='rainbow') 
plt.show() 
agg = AgglomerativeClustering(n_clusters=4)
agg.fit(X_principal)
# Visualizing the clustering 
plt.scatter(X_principal['P1'], X_principal['P2'], 
           c = AgglomerativeClustering(n_clusters = 4).fit_predict(X_principal), cmap =plt.cm.winter) 
plt.show() 
ac5 = AgglomerativeClustering(n_clusters = 5) 

  

plt.figure(figsize =(6, 6)) 

plt.scatter(X_principal['P1'], X_principal['P2'], 

            c = ac5.fit_predict(X_principal), cmap ='rainbow') 
plt.show() 


ac6 = AgglomerativeClustering(n_clusters = 6) 

  

plt.figure(figsize =(6, 6)) 

plt.scatter(X_principal['P1'], X_principal['P2'], 

            c = ac6.fit_predict(X_principal), cmap ='rainbow') 
plt.show() 
silhouette_scores = [] 

for n_cluster in range(2, 8):
    silhouette_scores.append( 
        silhouette_score(X_principal, AgglomerativeClustering(n_clusters = n_cluster).fit_predict(X_principal))) 
    
# Plotting a bar graph to compare results 
k = [2, 3, 4, 5, 6,7] 
plt.bar(k, silhouette_scores) 
plt.xlabel('Number of clusters', fontsize = 10) 
plt.ylabel('Silhouette Score', fontsize = 10) 
plt.show() 