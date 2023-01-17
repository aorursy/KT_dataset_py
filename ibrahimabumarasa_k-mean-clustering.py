import numpy as np

import pandas as pd 



#load existing dataset stored as csv file 

#dataset = pd.read_excel("../input/TestDataSet/newDataSet5.xlsx")



dataset = pd.read_csv("../input/newDataSetFinal.csv", encoding = 'utf-8', 

                              index_col = ["Category"])

## get columns' titles

print('Dataset contains on the following columns:')

titles = dataset.columns 

print(titles)
from sklearn.cluster import KMeans

clusters = 7

  

kmeans = KMeans(n_clusters = clusters) 

kmeans.fit(dataset) 

  

print(kmeans.labels_)
from sklearn.decomposition import PCA 



pca = PCA(3) 

pca.fit(dataset) 

  

pca_data = pd.DataFrame(pca.transform(dataset)) 

  

print(pca_data.head())
from matplotlib import colors as mcolors 



import math 

   

''' Generating different colors in ascending order  

                                of their hsv values '''

colors = list(zip(*sorted(( 

                    tuple(mcolors.rgb_to_hsv( 

                          mcolors.to_rgba(color)[:3])), name) 

                     for name, color in dict( 

                            mcolors.BASE_COLORS, **mcolors.CSS4_COLORS 

                                                      ).items())))[1] 

   

   

# number of steps to taken generate n(clusters) colors  

skips = math.floor(len(colors[5 : -5])/clusters) 

cluster_colors = colors[5 : -5 : skips] 
from mpl_toolkits.mplot3d import Axes3D 

import matplotlib.pyplot as plt

   

fig = plt.figure() 

ax = fig.add_subplot(111, projection = '3d') 

ax.scatter(pca_data[0], pca_data[1], pca_data[2],  

           c = list(map(lambda label : cluster_colors[label], 

                                            kmeans.labels_))) 

   

str_labels = list(map(lambda label:'% s' % label, kmeans.labels_)) 

   

list(map(lambda data1, data2, data3, str_label: 

        ax.text(data1, data2, data3, s = str_label, size = 16.5, 

        zorder = 20, color = 'k'), pca_data[0], pca_data[1], 

        pca_data[2], str_labels)) 

   

plt.show() 
import seaborn as sns 

  

# generating correlation heatmap 

sns.heatmap(dataset.corr(), annot = True) 

  

# posting correlation heatmap to output console  

plt.show() 