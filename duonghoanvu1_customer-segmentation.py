import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly as py
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import warnings
import os
warnings.filterwarnings("ignore")
py.offline.init_notebook_mode(connected = True)
df = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
df
df.describe()
df.info()
df.drop('CustomerID', axis=1, inplace=True)
plt.style.use('fivethirtyeight')
plt.figure(1 , figsize = (15 , 6))
n = 0 
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace =0.5 , wspace = 0.5)
    sns.distplot(df[x] , bins = 20)
    plt.title('Distplot of {}'.format(x))
plt.show()
labels = ['Female', 'Male']
size = df['Gender'].value_counts()
colors = ['lightgreen', 'orange']
explode = [0, 0.1]

plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')
plt.title('Gender', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()
plt.figure(figsize = (20 , 6))
sns.pairplot(df, hue='Gender', size=3)
sns.boxenplot(df['Gender'], df['Spending Score (1-100)'])
plt.title('Gender vs Spending Score', fontsize = 20)
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes = axes.flatten()


sns.scatterplot(x=df['Age'], y=df['Annual Income (k$)'], hue=df['Gender'], ax=axes[0])

sns.scatterplot(x=df['Age'], y=df['Spending Score (1-100)'], hue=df['Gender'], ax=axes[1])

sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Gender'], ax=axes[2])
plt.subplots_adjust(wspace = 0.3)
df['Gender'] = df['Gender'].map({'Male':0, 'Female':1})
from sklearn.cluster import KMeans
SSE_to_nearest_centroid = []

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    SSE_to_nearest_centroid.append(kmeans.inertia_)

#sns.set(style='whitegrid')
plt.figure(constrained_layout=True, figsize=(12, 5))
sns.lineplot(x=list(range(1,15)), y=SSE_to_nearest_centroid)
plt.xlabel("Amount of Clusters",fontsize=14)
plt.ylabel("Inertia",fontsize=14)
plt.grid(True)
kmeans = KMeans(n_clusters=6)
df["labels_6"] = kmeans.fit_predict(df)

kmeans = KMeans(n_clusters=8)
df["labels_8"] = kmeans.fit_predict(df)
fig, axes = plt.subplots(3, 2, figsize=(15, 10))
axes = axes.flatten()
plt.subplots_adjust(hspace = 0.7, wspace = 0.3)
sns.scatterplot(x=df['Age'], y=df['Annual Income (k$)'], hue=df['labels_6'], ax=axes[0], palette='deep')
# Remove Legend title & move legend outside to the right
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(0.9, 1.4), loc='upper left', borderaxespad=0.)

sns.scatterplot(x=df['Age'], y=df['Annual Income (k$)'], hue=df['labels_8'], ax=axes[1], palette='deep')
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(0.9, 1.4), loc='upper left', borderaxespad=0.)

sns.scatterplot(x=df['Age'], y=df['Spending Score (1-100)'], hue=df['labels_6'], ax=axes[2], palette='deep')
handles, labels = axes[2].get_legend_handles_labels()
axes[2].legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(0.9, 1.4), loc='upper left', borderaxespad=0.)

sns.scatterplot(x=df['Age'], y=df['Spending Score (1-100)'], hue=df['labels_8'], ax=axes[3], palette='deep')
handles, labels = axes[3].get_legend_handles_labels()
axes[3].legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(0.9, 1.4), loc='upper left', borderaxespad=0.)

sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], ax=axes[4], hue=df['labels_6'], palette='deep')
handles, labels = axes[4].get_legend_handles_labels()
axes[4].legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(0.9, 1.4), loc='upper left', borderaxespad=0.)

sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], ax=axes[5], hue=df['labels_8'], palette='deep')
handles, labels = axes[5].get_legend_handles_labels()
axes[5].legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(0.9, 1.4), loc='upper left', borderaxespad=0.)

plt.show()
#plt.legend('',frameon=False) # for 1 graph
df['labels_6_rename_AnuualIncome-SpendingScore'] = df['labels_6'].map({0: 'miser', 1:'General', 2:'General', 3:'Target', 4:'SpendThrift', 5: 'Careful'})
fig, axes = plt.subplots(figsize=(10, 5))
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['labels_6_rename_AnuualIncome-SpendingScore'], palette='deep')
handles, labels = axes.get_legend_handles_labels()
axes.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(1, 1.1), loc='upper left', borderaxespad=0.)
df['labels_6_Age-SpendingScore'] = df['labels_6'].map({0: 'Usualy Customer', 
                                                       1:'Target Customer (old)', 
                                                       2:'Target Customer (young)', 
                                                       3:'Priority Customer', 
                                                       4:'Priority Customer', 
                                                       5: 'Usualy Customer'})
fig, axes = plt.subplots(figsize=(10, 5))
sns.scatterplot(x=df['Age'], y=df['Spending Score (1-100)'], hue=df['labels_6_Age-SpendingScore'], palette='deep')
handles, labels = axes.get_legend_handles_labels()
axes.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(1, 1.1), loc='upper left', borderaxespad=0.)
df
trace1 = go.Scatter3d(x= df['Age'],
                      y= df['Spending Score (1-100)'],
                      z= df['Annual Income (k$)'],
                      mode='markers',
                      marker=dict(color = df['labels_6'],
                                  size= 10,
                                  line=dict(color= df['labels_6'],
                                            width= 12),
                                  opacity=0.8))
data = [trace1]
layout = go.Layout(title= 'Clusters',
                   scene = dict(xaxis = dict(title  = 'Age'),
                                yaxis = dict(title  = 'Spending Score'),
                                zaxis = dict(title  = 'Annual Income')))
                    #     margin=dict(
                    #         l=0,
                    #         r=0,
                    #         b=0,
                    #         t=0
                    #     )
        
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)
