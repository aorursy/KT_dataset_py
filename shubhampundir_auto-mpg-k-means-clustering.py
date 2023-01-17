import os
os.listdir('../input/')
import pandas as pd 

df=pd.read_csv("../input/../input/autompg-dataset/auto-mpg.csv")

df
df["horsepower"].unique()
df=df[df.horsepower != "?"]  # since we find the "?" ,thus removing the value 
df["horsepower"].unique()  # "?" removed successfully
df.info()
import seaborn as sns

#horsepower distribution

sns.distplot(df['horsepower'], hist=True, kde=True, 

              color = 'Red',

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 3})
#accelration disctribution

sns.distplot(df['acceleration'], hist=True, kde=True, 

              color = 'blue',

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 3})
#weight distribution

sns.distplot(df['weight'], hist=True, kde=True, 

              color = 'blue',

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 3})
#displacement distribution

sns.distplot(df['displacement'], hist=True, kde=True, 

              color = 'blue',

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 3})
#mpg distribution

sns.distplot(df['mpg'], hist=True, kde=True, 

              color = 'blue',

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 3})
df["horsepower"]=df.horsepower.astype("int64") #changing it from object to int 
#orignal data 

horepw=df["horsepower"]

accel=df["acceleration"]
#scaled data 

scaled_horpw=whiten(horepw)

scaled_accel=whiten(accel)
from scipy.cluster.vq import whiten  #whiten hepls to scale the data 

#pusing the sacled data into data frame

df["scaled_accel"]=scaled_accel

df["scaled_horpw"]=scaled_horpw
#plotting teh scaled data and orignal data 
import  matplotlib.pyplot as plt

#orignal data 

plt.plot(accel,label="orignal")

#scaled data 

plt.plot(scaled_accel,label="scaled")

#legend 

plt.legend()

plt.show()
#orignal data 

plt.plot(horepw,label="orignal")

#scaled data 

plt.plot(scaled_horpw,label="scaled")

#legend 

plt.legend()

plt.show()
#first will check how many clusters we will take by the help of elbow plot

distortions = []

num_clusters = range(1, 7)



# Create a list of distortions from the kmeans function

for i in num_clusters:

    cluster_centers, distortion = kmeans(df[["horsepower","acceleration"]],i)

    distortions.append(distortion)



# Create a data frame with two lists - num_clusters, distortions

elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})



# Creat a line plot of num_clusters and distortions

sns.lineplot(x="num_clusters", y="distortions", data = elbow_plot)

plt.xticks(num_clusters)

plt.show()
#clustering with two variables "horsepower" and "acceleration".

# Import the kmeans and vq functions

from scipy.cluster.vq import kmeans, vq

import seaborn as sns



# Generating cluster centers

cluster_centers, distortion = kmeans(df[["horsepower","acceleration"]],2)



# Assigning cluster labels

df['cluster_labels'], distortion_list = vq(df[["horsepower","acceleration"]],cluster_centers)



# Plotting clusters

sns.scatterplot(x='horsepower', y='acceleration', 

                hue='cluster_labels', data = df)

plt.show()
#now taking the "weight" and "displacement"



#first will check how many clusters will be adiquate

distortions = []

num_clusters = range(1, 7)



# Create a list of distortions from the kmeans function

for i in num_clusters:

    cluster_centers, distortion = kmeans(df[["weight","displacement"]],i)

    distortions.append(distortion)



# Create a data frame with two lists - num_clusters, distortions

elbow_plot = pd.DataFrame({'num_clusters': num_clusters, 'distortions': distortions})



# Creat a line plot of num_clusters and distortions

sns.lineplot(x="num_clusters", y="distortions", data = elbow_plot)

plt.xticks(num_clusters)

plt.show()
#clustering with two variables "weight" and "displacement".

# Import the kmeans and vq functions

from scipy.cluster.vq import kmeans, vq

import seaborn as sns



# Generating cluster centers

cluster_centers, distortion = kmeans(df[["weight","displacement"]],2)



# Assigning cluster labels

df['cluster_labels'], distortion_list = vq(df[["weight","displacement"]],cluster_centers)



# Plotting clusters

sns.scatterplot(x='weight', y='displacement', 

                hue='cluster_labels', data = df)

plt.show()