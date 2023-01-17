# To enable plotting graphs in Jupyter notebook

%matplotlib inline 
# Numerical libraries

import numpy as np   



from sklearn.model_selection import train_test_split



from sklearn.cluster import KMeans



# to handle data in form of rows and columns 

import pandas as pd    



# importing ploting libraries

import matplotlib.pyplot as plt   



#importing seaborn for statistical plots

import seaborn as sns



from sklearn import metrics



import pandas as pd



from sklearn.metrics import silhouette_samples, silhouette_score

from mpl_toolkits.mplot3d import Axes3D
import os

os.listdir('../input')
# reading the CSV file into pandas dataframe

mydata = pd.read_csv("../input/depression-dataset/Depression.csv")
mydata.head()
##Remove id since it is redundant

mydata.drop('id', axis=1, inplace=True)
mydata.info()
mydata.describe()
import seaborn as sns

sns.pairplot(mydata, diag_kind='kde') 
##Based on the kde plots, we can work with 2 or 3 clusters
##Scale the data

from scipy.stats import zscore



mydata_z = mydata.apply(zscore)

#Finding optimal no. of clusters

from scipy.spatial.distance import cdist

clusters=range(1,10)

meanDistortions=[]



for k in clusters:

    model=KMeans(n_clusters=k)

    model.fit(mydata)

    prediction=model.predict(mydata)

    meanDistortions.append(sum(np.min(cdist(mydata, model.cluster_centers_, 'euclidean'), axis=1)) / mydata

                           .shape[0])





plt.plot(clusters, meanDistortions, 'bx-')

plt.xlabel('k')

plt.ylabel('Average distortion')

plt.title('Selecting k with the Elbow Method')
#Set the value of k=3

kmeans = KMeans(n_clusters=3, n_init = 15, random_state=2345)
kmeans.fit(mydata_z)
centroids = kmeans.cluster_centers_
centroids
#Clculate the centroids for the columns to profile

centroid_df = pd.DataFrame(centroids, columns = list(mydata_z) )
print(centroid_df)
## creating a new dataframe only for labels and converting it into categorical variable

df_labels = pd.DataFrame(kmeans.labels_ , columns = list(['labels']))



df_labels['labels'] = df_labels['labels'].astype('category')

df_labels.labels.head(3)
# Joining the label dataframe with the data frame.

df_labeled = mydata.join(df_labels)
kmeans.labels_
df_analysis = (df_labeled.groupby(['labels'] , axis=0)).head(4177) 

 # the groupby creates a groupeddataframe that needs to be converted back to dataframe. 

df_analysis.head()
df_labeled['labels'].value_counts()  
from mpl_toolkits.mplot3d import Axes3D
## 3D plots of clusters

fig = plt.figure(figsize=(10, 8))

ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=120)



kmeans.fit(mydata_z)

labels = kmeans.labels_



#3 columns and label column

ax.scatter(mydata_z.iloc[:, 0], mydata_z.iloc[:,1], mydata_z.iloc[:, 2],c=labels.astype(np.float), edgecolor='k')



ax.w_xaxis.set_ticklabels([])

ax.w_yaxis.set_ticklabels([])

ax.w_zaxis.set_ticklabels([])





ax.set_xlabel('Simplicity')

ax.set_ylabel('Fatalism')

ax.set_zlabel('Depression')

ax.set_title('3D plot of KMeans Clustering')
# Let us try with K = 2

final_model=KMeans(2)

final_model.fit(mydata_z)

prediction=final_model.predict(mydata_z)



#Append the prediction 

mydata["GROUP"] = prediction

print("Groups Assigned : \n")

mydata[["depression", "GROUP"]].head(10)
#plt.cla()



import matplotlib.cm as cm







colors = cm.rainbow(np.linspace(0, 1, 2))





#plt.scatter(mydata["simplicity"][mydata.GROUP==0],

         #mydata["simplicity"][mydata.GROUP==1],c = colors, alpha=0.5)



plt.scatter(mydata["simplicity"],mydata["depression"], c = prediction, alpha=0.5)



#plt.scatter(mydata["simplicity"][mydata.GROUP==0],

             #mydata["simplicity"][mydata.GROUP==1],c = colors, alpha=0.5)

    

mydata.boxplot(by = 'GROUP',  layout=(2,4), figsize=(20, 15))
silhouette_avg = silhouette_score(mydata_z, kmeans.labels)

print("For n_clusters =", 3, "The average silhouette_score is :", silhouette_avg)
silhouette_avg = silhouette_score(mydata_z, final_model.labels_)

print("For n_clusters =", 2, "The average silhouette_score is :", silhouette_avg)
#To determine if a relationship exists between black and white thinking(simplicity) and depression. 

mydata['simplicity'].corr(mydata['depression'])
%matplotlib inline

import pandas as pd

plt.plot(mydata['simplicity'], mydata['depression'], 'bo')

z = np.polyfit(mydata['simplicity'], mydata['depression'],1)

p = np.poly1d(z)

plt.plot(mydata['simplicity'], p(mydata['simplicity']), "r--")



#geom_point()
# As you can see from the above graphic, although the data does not form a perfectly straight line, it does fall in a way that 

#indicates a positive relationship. Therefore, we can once again conclude that there is a relationship between black and white 

#thinking and depression. It is important to note, however, that correlation does not in any way indicate causality and is merely

#indicative of a relationship between the two.