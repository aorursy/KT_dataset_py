import pandas as pd

mall = pd.read_csv("../input/mall-customers/Mall_Customers.csv")

newmall=mall.copy()

mall.head()
# in spending is not amount it is score 

# there is no information of score like 6 will be good or bad

#annual income in thousand 

# its raw problem 



mall.Genre.value_counts()



mall=mall.drop(['Genre','CustomerID','Age'],axis=1)



mall.head()
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()



scalemall=sc.fit_transform(mall)



scmall=pd.DataFrame(scalemall,columns=list(mall))

scmall.head()



# applying clustering and then visualise the cluster

# finding the optimal cluster size 

#we calculating the wcss and hence create elbow plot
from sklearn.cluster import KMeans



inertia=[] # create list of inertia or variance 

cluster=range(1,10) # fitting diffrent cluster

for i in cluster:

    kmeans=KMeans(n_clusters=i)# fitting the model 

    kmeans.fit(scmall)

    inertia.append(kmeans.inertia_)
import matplotlib.pyplot as plt



plt.plot(cluster,inertia,'o-') # plotting cluster on x axis

plt.xlabel('no of cluster')
# from elbow the method we found after 5 cluster not much change

# in inertia within inertia is very less 



kmeans=KMeans(n_clusters=5)# fitting the model 

kmeans.fit(scmall)

kmeans.cluster_centers_
# kmeans_labels giving the cluster names

# kmeans cluster centre give centroid



pd.DataFrame(kmeans.cluster_centers_,columns=list(scmall))
clusters=kmeans.predict(scmall)# it tells which column belong to that 

                               # cluster



# it is predicting the record which record belong to that record



name_cluster=kmeans.labels_ # labels of cluster you can change the name of cluster 




plt.scatter(mall.iloc[clusters==0,0],mall.iloc[clusters==0,1],color='red',label='cluster 1')



plt.scatter(mall.iloc[clusters==1,0],mall.iloc[clusters==1,1],color='purple',label='cluster 2')

plt.scatter(mall.iloc[clusters==2,0],mall.iloc[clusters==2,1],color='green',label='cluster 3')

plt.scatter(mall.iloc[clusters==3,0],mall.iloc[clusters==3,1],color='blue',label='cluster 4')

plt.scatter(mall.iloc[clusters==4,0],mall.iloc[clusters==4,1],color='pink',label='cluster 5')

plt.legend()

plt.show()



# from has much salary but not spent more

# they are much saving money

# cluster 4 has low salary and low spend 

# cluster 2 is average 

# cluster 5 is so risky because it has low salary and spend much

# cluster 3 has high salary and high spending 

# cluster 3 give the loan he has much spending and give him sale , offer

#cluster 1 has those customer who not spending money 

# our target is play with cluster with 1 he is regular customer







newmall[(newmall['Annual Income (k$)']>70) & (newmall['Spending Score (1-100)']<40)]['Age'].mean()



# cluster 1 who has less spending money has average age is 41
newmall[(newmall['Annual Income (k$)']<40) & (newmall['Spending Score (1-100)']>60)]['Age'].mean()



# cluster 5 are those who are young people average age is 25 

# young has salary less but high spending money
newmall[((newmall['Annual Income (k$)']>40)&(newmall['Annual Income (k$)']<80)) & ((newmall['Spending Score (1-100)']<60)&(newmall['Spending Score (1-100)']>40))]['Age'].mean()

# we found that age average age of 44 people and > 50 person are regular ideal customers who have salary is average and average spending