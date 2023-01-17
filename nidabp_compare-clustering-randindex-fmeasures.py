from copy import deepcopy
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
df = pd.read_csv("../input/Iris.csv")
# drop the first column as we need to require it 
df.drop('Id',axis=1,inplace=True)
df.head()
df["Species"] = pd.Categorical(df["Species"])
#Replace the species values with codes 0,1 and 2 so that it can be compared with the ouput clusters
df["Species"] = df["Species"].cat.codes
data = df.values[:,0:4]
category = df.values[:,4]
SampleSize = data.shape[0]
colors=['orange', 'blue', 'green']
for i in range(data.shape[0]):
    plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(category[i])])
def KMeans(X,cluster_size):
    
    k = cluster_size
    n = X.shape[0]
    c = X.shape[1]
    
    status = 'success'
        
    mean = np.mean(X, axis=0)
    std = np.std(X, axis = 0)

    #initialize the centroids with k random points from the input sample
    centroids = X[np.random.choice(n, k, replace=False), :]
    
    centroids_old = np.zeros(centroids.shape)
    centroids_new = deepcopy(centroids)
    
    clusters = np.zeros(n)
    distances = np.zeros((n,k))
  
    
    movingdistance = np.linalg.norm(centroids_new - centroids_old)
    
    while movingdistance != 0:
        
        #compute the distance from centroid to all the other points
        for i in range(k):
            distances[:,i] = np.linalg.norm(X - centroids[i], axis = 1)
            
        #select the minimum distance cluster
        clusters = np.argmin(distances, axis = 1)
    
        centroids_old = deepcopy(centroids_new)
        
        for i in range(k):
            #just incase the cluster has zero points
            if len(X[clusters==i]) == 0 :
                return clusters,centroids,'failure'
            else :
                centroids_new[i] = np.mean(X[clusters==i], axis=0)
                
        #to know if the centroids are static or not to exit the loop  
        movingdistance = np.linalg.norm(centroids_new - centroids_old)
        
    return clusters,centroids_new,status
def DBScan(X,eps,minPts) :
    
    n = X.shape[0]
    c = X.shape[1]
    
    X_visited = np.zeros(n);
    clusters = np.zeros(n);
    
    k = 0
    
    for i in range(0,n) :
        if X_visited[k] == 0 :
            X_visited[k] = 1
            
            #to get all the points within the range of eps from the core point
            Neighbours = RangeQuery(X, X[i,:], eps) 
            
            if len(Neighbours) < minPts :
                clusters[i] = -1
            else :
                k = k + 1
                clusters[i] = k
                while len(Neighbours) > 0 :
                    j = Neighbours.pop()
                    if X_visited[j] == 0 :
                        X_visited[j] = 1
                        #to get all the border points
                        bp = RangeQuery(X,X[j,:],eps)
                        if len(bp) >= minPts :
                            Neighbours.extend(bp)
                    
                    if clusters[j] <= 0 :
                        clusters[j] = k
    return clusters
        
def RangeQuery(X,Q,eps)  :
    
    n = X.shape[0]
    Neighbours = []
    for i in range(0,n):
        if np.linalg.norm(X[i] - Q) <= eps :
            Neighbours.extend([i])
    
    return Neighbours
        
def Hierarchical(X,cluster_size) :
    
    n = X.shape[0]
    
    d = np.zeros((n,n))
    
    #calculate the distance matrix
    for i in range(n):
        for j in range(n):
            d[i,j] = np.linalg.norm(X[i] - X[j])
    
    #position matrix to remember the position of each element in the distance matrix
    pos = np.array(list(range(0,n)))
    
    clusters = np.zeros(n)
    buckets = []
    
    #initialize buckets
    for i in range(0,n):
        buckets.append([])
    
     
    for i in range(n,cluster_size,-1):
        
        #initialize the diagonal with a number more than the max element
        maxelement = np.max(d) + 1
        np.fill_diagonal(d,maxelement)
    
        #get the indexes of the max element
        index_min = np.unravel_index(np.argmin(d, axis=None), d.shape)
        pt1 = index_min[0]
        pt2 = index_min[1]
        
        #replace the digonal elements with zeros
        np.fill_diagonal(d,0)
        
        #get the difference vector between the index columns of the max element.
        diff = np.minimum(d[:,pt1],d[:,pt2])
        
        minpt = min(pt1,pt2)
        maxpt = max(pt1,pt2) - 1
        
        #delete the columns and rows with the indexes of the max element. perform the same operations on position matrix as well
        d = np.delete(d,minpt,axis=0)
        d = np.delete(d,minpt,axis=1)
        pos1 = pos[minpt]
        pos = np.delete(pos,minpt)
        d = np.delete(d,maxpt,axis=0)
        d = np.delete(d,maxpt,axis=1)
        pos2 = pos[maxpt]
        pos = np.delete(pos,maxpt)
        diff = np.delete(diff,minpt,axis=0)
        diff = np.delete(diff,maxpt,axis=0)
        
        #replace the deleted rows and columns with the difference vector at the end of the distance matrix
        d = np.append(d, diff[:, None], axis=1)
        diff = np.append(diff,0)
        d = np.append(d, diff[None,:], axis=0)
        minpos = min(pos1,pos2)
        maxpos = max(pos1,pos2)
        
        #perform the same operation on the position vector as well.
        pos = np.append(pos,minpos)
        
        #append the deleted row and column indexes in the same bucket.
        if len(buckets[minpos]) == 0 :
            buckets[minpos].append(minpos)
        if len(buckets[maxpos]) == 0 :
            buckets[minpos].append(maxpos)
        else :
            for i in range(0,len(buckets[maxpos])):
                buckets[minpos].append(buckets[maxpos][i])
            
      
    clusternumber = 0
    
    #assign same cluster numbers to all the elements in the same bucket.
    for i in range(cluster_size,0,-1):
        if len(buckets[pos[i - 1]]) == 0:
            clusters[pos[i - 1]] = clusternumber
        else :
            for j in range(0,len(buckets[pos[i - 1]])) :
                clusters[buckets[pos[i-1]][j]] = clusternumber
        clusternumber = clusternumber + 1
            
    return clusters
def CalculateMetrics(clusters,Y,cluster_size) :
    
    k = cluster_size
    n = Y.shape[0]
        
    clustersizes = np.zeros(k)
    clusterelementsizes = np.zeros((k,k))

    totalpairs = (n * (n-1)) / 2

    totalpositives = 0
    totalnegatives = 0

    truepositives = 0
    falsepositives = 0
    truenegatives = 0
    falsenegatives = 0
    randindex = 0
    precision = 0
    recall = 0
    f1score = 0

    #calculate true positives, true negatives , false positives and false negatives.
    for i in range(k):
        clustersizes[i] = Y[clusters == i].shape[0]
        if clustersizes[i] > 2 :
            p = clustersizes[i]
            totalpositives += (p * (p-1))/2;
        
    totalnegatives = totalpairs - totalpositives

    for i in range(k):
        for j in range(k):
            clusterelementsizes[i,j] = np.sum(np.equal(j,Y[clusters == i]))
            if clusterelementsizes[i,j] > 2 :
                p = clusterelementsizes[i,j]
                truepositives += (p * (p-1))/2
            
    falsepositives = totalpositives - truepositives

    falsenegatives =  0   

    for column in range(k):
        for row in range(k-1):
            falsenegatives += clusterelementsizes[row,column] * (clusterelementsizes[row+1:k,column].sum(axis=0))
    
    truenegatives = totalnegatives - falsenegatives

    randindex = (truepositives + truenegatives)/(truepositives + falsepositives + truenegatives + falsenegatives)

    precision = truepositives / (truepositives + falsepositives)

    recall = truepositives / (truepositives + falsenegatives)

    f1score = (2 * precision * recall) /(precision + recall)

    return randindex, f1score
#run kmeans to get the randindex, f1scores and duration of execution of the clustering algorithm.
def RunKMeans(X,Y,cluster_size) :
    
    clusters = np.zeros(X.shape[0])
    num_iters = 100
    centroids = np.zeros((cluster_size,X.shape[1]))
    randindex = 0
    f1score = 0

    finalclusters = np.zeros(X.shape[0])
    finalcentroids = np.zeros((cluster_size,X.shape[1]))
    finalrandindex = 0
    finalf1score = 0
    status = 'success'
    duration = 0
    avgduration = 0
    
    i = 0
    while i < num_iters :
        start = time.time()
        clusters,centroids,status = KMeans(X,cluster_size)
        stop = time.time()
        if status == 'failure':
            continue
        duration = duration + (stop - start)
        i = i + 1
        randindex,f1score = CalculateMetrics(clusters,Y,cluster_size)
        if randindex > finalrandindex:
            finalrandindex = randindex
            finalf1score = f1score
            finalclusters = clusters
            finalcentroids = centroids
            
    avgduration = duration / num_iters
            
    return finalrandindex,finalf1score,finalclusters,finalcentroids,avgduration      
    
#This algorithm is used to get the eps value for which we get the max randindex.
#The value at the elbow in the below elbow curve gives the eps value.
def RunDMDBScan(X,minPts)  :
    n = X.shape[0]
    d = np.zeros((n,n))
    kdist = np.zeros((n,minPts))
    count = minPts + 1
    for i in range(n):
        for j in range(n):
            d[i,j] = np.linalg.norm(X[i] - X[j])
        kdist[i] = np.sort(d[i,:],axis=0)[1:count]
    y = np.sort(kdist.flatten())
    col = len(y)
    xaxis = []
    for i in range(col):
        xaxis.append(i)
    plt.plot(xaxis,y)
    plt.show()
    
RunDMDBScan(data,3)
#run DBScan to get the randindex, f1scores and duration of execution of the clustering algorithm.
def RunDBScan(X,Y,minPts) :
    
    clusters = np.zeros(X.shape[0])
    num_iters = 10
    randindex = 0
    f1score = 0
    
    cluster_size = 0
    
    duration = 0
    avgduration = 0

    finalclusters = np.zeros(X.shape[0])
    finalrandindex = 0
    finalf1score = 0
    finaleps = 0
    
    eps = 0.6
    
    for i in range(num_iters) :
        start = time.time()
        clusters = DBScan(X,eps,minPts)
        stop = time.time()
        duration = duration + (stop - start)
        cluster_size = int((np.max(clusters,axis=0)) + 1)
        randindex,f1score = CalculateMetrics(clusters,Y,cluster_size)
        if randindex > finalrandindex:
            finalrandindex = randindex
            finalf1score = f1score
            finalclusters = clusters
            finaleps = eps
        eps = eps + 0.001
        
    avgduration = duration / num_iters
    
    return finalrandindex,finalf1score,finalclusters,finaleps,avgduration
#run Hierarchical to get the randindex, f1scores and duration of execution of the clustering algorithm.
def RunHierarchical(X,Y,cluster_size) :
    
    clusters = np.zeros(X.shape[0])
    num_iters = 10
    randindex = 0
    f1score = 0
    
    duration = 0
    avgduration = 0
   

    for i in range(num_iters) :
        start = time.time()
        clusters = Hierarchical(X,cluster_size)
        stop = time.time()
        duration = duration + (stop - start)
        randindex,f1score = CalculateMetrics(clusters,Y,cluster_size)
                    
    avgduration = duration / num_iters
    
    return randindex,f1score,clusters,avgduration
KMcluster_size = 3
KMRandIndex = 0
KMF1Score = 0
KMClusters = np.zeros(data.shape[0])
KMCentroids = np.zeros((KMcluster_size,data.shape[1]))
KMDuration = 0

KMRandIndex,KMF1Score,KMClusters,KMCentroids,KMDuration = RunKMeans(data,category,KMcluster_size)

print("KMRandIndex = ",KMRandIndex)
print("KMF1Score = ",KMF1Score)
print("KMDuration = ",KMDuration)

colors=['orange', 'blue', 'green']
for i in range(data.shape[0]):
    plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(KMClusters[i])])
plt.scatter(KMCentroids[:,0], KMCentroids[:,1], marker='*', c='r', s=150)
DBRandIndex = 0
DBF1Score = 0
DBClusters = np.zeros(data.shape[0])
DBEps = 0
minPts = 3
DBDuration = 0

DBRandIndex,DBF1Score,DBClusters,DBEps,DBDuration = RunDBScan(data,category,minPts)

print("DBEps = ",DBEps)
print("DBRandIndex = ",DBRandIndex)
print("DBF1Score = ",DBF1Score)
print("DBDuration = ",DBDuration)

colors=['orange', 'blue', 'green']
for i in range(data.shape[0]):
    plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(DBClusters[i])])
HRandIndex = 0
HF1Score = 0
HClusters = np.zeros(data.shape[0])
HDuration = 0

Hcluster_size = 3

HRandIndex,HF1Score,HClusters,HDuration = RunHierarchical(data,category,Hcluster_size)

print("HRandIndex = ",HRandIndex)
print("HF1Score = ",HF1Score)
print("HDuration = ",HDuration)

colors=['orange', 'blue', 'green']
for i in range(data.shape[0]):
    plt.scatter(data[i, 0], data[i,1], s=7, color = colors[int(HClusters[i])])
fig = plt.figure(figsize=(10,2))

title_font = {'fontname':'Arial', 'size':'14', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom','horizontalalignment':'right'}

ax = fig.add_subplot(111, frame_on=False) 

ax.axis('tight')
ax.axis('off')

columns=('KMeans Clustering', 'DBScan Clustering','Hierarchical Clustering')
                
rows=('HyperParameters','RandIndex','F1Score','Duration(in secs)')
        
cell_text = []
        
cell_text.append(['Cluster Size=' + str(KMcluster_size),'Eps=' + str(DBEps)+', MinPts=' + str(minPts),'Cluster_Size=' + str(Hcluster_size)])
cell_text.append([str(KMRandIndex),str(DBRandIndex),str(HRandIndex)])
cell_text.append([str(KMF1Score),str(DBF1Score),str(HF1Score)])        
cell_text.append([str(KMDuration),str(DBDuration),str(HDuration)])

col = (0.5, 0.5, 0.8)

table = ax.table(cellText=cell_text,
                      rowLabels=rows,
                      colLabels=columns,
                    rowColours=[col]*16,
                    colColours=[col]*16,
                      loc='center',
                      cellLoc='center',
                        bbox=[-0.6,-0.6,1.5,1.5]
                      )

table.set_fontsize(12)
table.scale(1.25, 1.25)

ax.set_title('Comparing Clustering Methods (Sample Size = ' + str(SampleSize) + ')',**title_font)

plt.show()
