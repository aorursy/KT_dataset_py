import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
#import sklearn.cluster.hierarchical as hclust
from sklearn import preprocessing
import seaborn as sns
df = pd.read_csv('../input/College.csv')
print(df.shape)
df.head()
#exclude the categorical column and the college names
features = df.drop(['Private', 'Unnamed: 0'],axis=1)
features['Acceptperc'] = features['Accept'] / features['Apps']
features['Enrollperc'] = features['Enroll'] / features['Accept']
features.describe()
scaler = preprocessing.MinMaxScaler()
features_normal = scaler.fit_transform(features)
pd.DataFrame(features_normal).describe()
inertia = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(features_normal)
    kmeanModel.fit(features_normal)
    inertia.append(kmeanModel.inertia_)
# Plot the elbow
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.show()
kmeans = KMeans(n_clusters=4).fit(features_normal)
labels = pd.DataFrame(kmeans.labels_) #This is where the label output of the KMeans we just ran lives. Make it a dataframe so we can concatenate back to the original data
labeledColleges = pd.concat((features,labels),axis=1)
labeledColleges = labeledColleges.rename({0:'labels'},axis=1)
labeledColleges.head()
sns.lmplot(x='Top10perc',y='S.F.Ratio',data=labeledColleges,hue='labels',fit_reg=False)
sns.pairplot(labeledColleges,hue='labels')
labeledColleges['Constant'] = "Data" #This is just to add something constant for the strip/swarm plots' X axis. Can be anything you want it to be.
sns.stripplot(x=labeledColleges['Constant'],y=labeledColleges['Top10perc'],hue=labeledColleges['labels'],jitter=True)
sns.swarmplot(x=labeledColleges['Constant'],y=labeledColleges['Top10perc'],hue=labeledColleges['labels'])
f, axes = plt.subplots(4, 5, figsize=(20, 25), sharex=False) #create a 4x5 grid of empty figures where we will plot our feature plots. We will have a couple empty ones.
f.subplots_adjust(hspace=0.2, wspace=0.7) #Scooch em apart, give em some room
#In this for loop, I step through every column that I want to plot. This is a 4x5 grid, so I split this up by rows of 5 in the else if statements
for i in range(0,len(list(labeledColleges))-2): #minus two because I don't want to plot labels or constant
    col = labeledColleges.columns[i]
    if i < 5:
        ax = sns.stripplot(x=labeledColleges['Constant'],y=labeledColleges[col].values,hue=labeledColleges['labels'],jitter=True,ax=axes[0,(i)])
        ax.set_title(col)
    elif i >= 5 and i<10:
        ax = sns.stripplot(x=labeledColleges['Constant'],y=labeledColleges[col].values,hue=labeledColleges['labels'],jitter=True,ax=axes[1,(i-5)]) #so if i=6 it is row 1 column 1
        ax.set_title(col)
    elif i >= 10 and i<15:
        ax = sns.stripplot(x=labeledColleges['Constant'],y=labeledColleges[col].values,hue=labeledColleges['labels'],jitter=True,ax=axes[2,(i-10)])
        ax.set_title(col)
    elif i >= 15:
        ax = sns.stripplot(x=labeledColleges['Constant'],y=labeledColleges[col].values,hue=labeledColleges['labels'],jitter=True,ax=axes[3,(i-15)])
        ax.set_title(col)
f, axes = plt.subplots(4, 5, figsize=(20, 25), sharex=False) 
f.subplots_adjust(hspace=0.2, wspace=0.7)
for i in range(0,len(list(labeledColleges))-2):
    col = labeledColleges.columns[i]
    if i < 5:
        ax = sns.swarmplot(x=labeledColleges['Constant'],y=labeledColleges[col].values,hue=labeledColleges['labels'],ax=axes[0,(i)])
        ax.set_title(col)
    elif i >= 5 and i<10:
        ax = sns.swarmplot(x=labeledColleges['Constant'],y=labeledColleges[col].values,hue=labeledColleges['labels'],ax=axes[1,(i-5)])
        ax.set_title(col)
    elif i >= 10 and i<15:
        ax = sns.swarmplot(x=labeledColleges['Constant'],y=labeledColleges[col].values,hue=labeledColleges['labels'],ax=axes[2,(i-10)])
        ax.set_title(col)
    elif i >= 15:
        ax = sns.swarmplot(x=labeledColleges['Constant'],y=labeledColleges[col].values,hue=labeledColleges['labels'],ax=axes[3,(i-15)])
        ax.set_title(col)
colleges = df['Unnamed: 0']
colleges = pd.concat((colleges,labels),axis=1)
colleges = colleges.rename({'Unnamed: 0':'College',0:'Cluster'},axis=1)
sortcolleges = colleges.sort_values(['Cluster'])
pd.set_option('display.max_rows', 1000)
sortcolleges