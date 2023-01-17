import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#%matplotlib inline

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import seaborn as sns

#read csv file
df = pd.read_csv("../input/iso-type/iso_type.csv")
data = df[["Iso-freq%","iso-ppp"]]

plt.hist(data["Iso-freq%"], color="orange")
plt.title("Iso-Freq%",fontsize=15)
plt.hist(data["iso-ppp"], color="red")
plt.title("Iso-Pts/P",fontsize=15)
data_corr= data.corr() #Correlation Matrix
sns.heatmap(data_corr)
#scale data
scaler = StandardScaler()
data_std = scaler.fit_transform(data)

#Classify into 7 groups
k_means = KMeans(n_clusters=7) 
k_means.fit(data_std[:,[0,1]]) #learning
    
#add cluster ID to data
data = data.assign(ID=k_means.labels_ )

#plot
plt.scatter(data["Iso-freq%"], data["iso-ppp"], c=data["ID"])
plt.title('Clustering Isoration Skill', fontsize=15)
plt.xlabel('Iso-Freq%', fontsize=10)
plt.ylabel('Iso-Pts/poss', fontsize=10)