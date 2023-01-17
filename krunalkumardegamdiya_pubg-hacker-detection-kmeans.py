import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

from sklearn.cluster import KMeans

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
df = pd.read_csv('../input/pubg-statisctic/PUBG.csv')



# Remove the '#'to view the data from '#df.head()'



df.head()
df.dropna(inplace = True) #Dropping null values



cols = np.arange(52, 152, 1)  #It is just a range of 52 to 152 (excluded) to keep only solo players data.



df.drop(df.columns[cols], axis = 1, inplace = True) # Dropping columns from 52 to 152 



df.drop(df.columns[[0,1]], axis = 1, inplace = True) # Dropping player_name and tracker_id



df.drop(columns= ['solo_Revives'], inplace = True) #Because in solo game you don't have team mates to revive so it is always zero



df.drop(columns= ['solo_DBNOs'], inplace = True) #DBNOs = knock outs. it will always zero in solo match



df.head()
train, test = train_test_split(df, test_size=0.2, random_state = 10)

dev, test = train_test_split(test, test_size = 0.2, random_state = 10)



print("The number of training samples is", len(train))

print("The number of development samples is", len(dev))

print("The number of testing samples is", len(test))
# Selected five important features

train_data = train.loc[:,['solo_KillDeathRatio', "solo_HeadshotKillRatio", 'solo_WinRatio' , "solo_Top10Ratio",'solo_DamageDealt','solo_RoundsPlayed']]

dev_data = dev.loc[:,['solo_KillDeathRatio', "solo_HeadshotKillRatio", 'solo_WinRatio' , "solo_Top10Ratio",'solo_DamageDealt','solo_RoundsPlayed']]

test_data = test.loc[:,['solo_KillDeathRatio', "solo_HeadshotKillRatio", 'solo_WinRatio' , "solo_Top10Ratio",'solo_DamageDealt','solo_RoundsPlayed']]
#scaler = StandardScaler()

#X_train_std = scaler.fit_transform(train_data)

#X_dev_std = scaler.transform(dev_data)

#X_test_std = scaler.transform(test_data)
# The number of clusters from 1 to 10

k = range(1, 25)



inertias = []

for i in k:

    # Create a KMeans instance with k clusters: model

    model = KMeans(n_clusters=i, init='k-means++', random_state = 10)

    

    # Fit model to samples

    model.fit(train_data)

    

    # Append the inertia to the list of inertias

    inertias.append(model.inertia_)

        

    print(f'Inertia for {i} Clusters: {model.inertia_:.0f}')
# The difference between consecutive inertias



for i in range(1,24):    

    print( f'Difference between inertia is {inertias[i-1] - inertias[i]:.0f} from point {i} to {i+1} ')
# Plot number of clusters vs inertia

plt.figure(figsize = (15, 5))                                       #Figure size length 15 and height 5 

plt.plot(k, inertias, '-o', color = 'black')                        # X axis = k , Y axis = Inertia

plt.plot(4, inertias[3], '-o', color = 'red', markersize = 12)      # Red dot specifying optimal cluster

plt.xlabel('Number of clusters', fontsize = 24)                    

plt.ylabel('Inertia', fontsize = 24)

plt.title('Optimal number of clusters (Inertia)', fontsize = 24)

plt.xticks(k, fontsize = 18)

plt.yticks(fontsize = 10);
# Number of clusters

ks = range(2, 24)

score = []



# Silhouette Method

for k in ks:

    kmeans = KMeans(n_clusters = k, init='k-means++', random_state = 555).fit(train_data)

    ss = metrics.silhouette_score(train_data, kmeans.labels_, sample_size = 10000)

    score.append(ss)

    print('Silhouette Score for %i Clusters: %0.4f' % (k, ss))
# Graph

plt.figure(figsize = (15, 7))

plt.plot(ks, score, '-o', color = 'blue')



# Marking Points

s = ['D', 'D', 'D' ]

col = ['red','green','orange' ]

x = np.array([2, 3, 4])       ### Basically we are creating points for ex; (x,y) = (2.4 , 6)

y = score[0:3]                   ### value of score from first element to 3rd element

plt.xticks(ks, fontsize = 18)

plt.yticks(fontsize = 18)



## Different Markers for first three points

for _s, c, _x, _y in zip(s, col, x, y):

    plt.scatter(_x, _y, marker=_s, c=c, s = 100)

plt.xlabel("Number of clusters", fontsize = 24)

plt.ylabel("Silhouette score", fontsize = 24)

plt.title('Optimal number of clusters (Silhouette)', fontsize = 24)



plt.text(1.90, score[0] + 0.005, str(round(score[0], 3)), size = 14, color = 'red', weight = 'semibold')



plt.text(2.97, score[1] + 0.005, str(round(score[1], 3)), size = 14, color = 'green', weight = 'semibold')



plt.text(3.90, score[2] + 0.005, str(round(score[2], 3)), size = 14, color = 'orange', weight = 'semibold')







plt.show()
# Lets say for number of cluster = 2, it means that it can be "Human" or it can be "Hacker"



kmeans = KMeans(n_clusters = 4,init= 'k-means++',max_iter = 300, random_state=1).fit(train_data)

labels = kmeans.labels_   # it will be series containing 0 and 1 (0 for human and 1 for hacker)





df_x_train = pd.DataFrame(train_data)

df_x_train['Clusters'] = pd.Series(labels, index=df_x_train.index)  # Adding Column named Clusters containing 0 and 1



cluster_names = {0:'Beginner',1:'Professionals',2:'Hacker',3:'Experienced'}



df_x_train['Name_of_Cluster'] = df_x_train['Clusters'].map(cluster_names)  # adding one more Column which tells 0 as human and 1 as hacker

df_x_train.columns = ['KillDeathRatio', "HeadshotKillRatio", 'WinRatio' , "Top10Ratio",'DamageDealt','RoundsPlayed','Cluster','Name_of_Cluster']



df_x_train.head()
df_x_train.groupby('Name_of_Cluster').count()
def scat3d(df,x,y,z,code,title):

    scatter = px.scatter_3d(df,x=x,y=y,z=z,color=code,title = title)

    return scatter.show()



# Plot of Win Ratio, Kill Death Ratio, Headshott KIll Ratio

scat3d(df = df_x_train , x = 'RoundsPlayed' , y = 'HeadshotKillRatio' , z = 'WinRatio',code = 'Name_of_Cluster' , title = 'roundsplayed vs Headshot vs Win ratio')
predict_labels = kmeans.predict(dev_data)
dev_df = pd.DataFrame(dev_data)

dev_df['Cluster'] = pd.Series(predict_labels,index = dev_df.index)

dev_df['Name_of_Cluster'] = dev_df['Cluster'].map(cluster_names)

dev_df.groupby('Name_of_Cluster').count()
predict_test_labels = kmeans.predict(test_data)
df_test = pd.DataFrame(test_data)

df_test['Cluster'] = pd.Series(predict_test_labels, index = df_test.index)

df_test['Name_of_Cluster'] = df_test['Cluster'].map(cluster_names)

df_test.columns = ['KillDeathRatio', "HeadshotKillRatio", 'WinRatio' , "Top10Ratio",'DamageDealt','RoundsPlayed','Cluster','Name_of_Cluster']

df_test.groupby('Name_of_Cluster').count()
# 2D Scatter plot



px.scatter(df_test,y='DamageDealt',x = 'WinRatio',color = 'Name_of_Cluster', title = 'DamageDealt vs WinRatio')



# There can be many combinations but I am not plottting each and every combination