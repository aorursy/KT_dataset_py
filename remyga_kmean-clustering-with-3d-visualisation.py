import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans 

from sklearn.metrics import silhouette_score

import seaborn as sns
cust = pd.read_csv('/kaggle/input/mall-customers/Mall_Customers.csv')

cust.head()
#let's look at the men and women via histogram



sns.countplot(x='Genre', data=cust)

plt.title('Customer Gender')
#to make a piechart:

gender=cust.Genre.value_counts()

gender_label=['Female','Male']

plt.pie(gender, labels=gender_label, autopct='%0.2f%%',startangle=90)

plt.title('Distribution of men and women within the customers of the Mall')

plt.show()
#let's see the max and min of ages

cust.describe()
bin_list=[10,20,30,40,50,60,70]

plt.hist(cust['Age'], bins=bin_list, rwidth=0.9)

plt.xlabel('Age')

plt.ylabel('frequency')

plt.title('Age distribution of customers')
plt.hist(cust['Annual Income (k$)'], bins=12, rwidth=0.9)

plt.xlabel("Income in 1000's of $")

plt.ylabel("frequency")

plt.title('Annual income of customers')
plt.hist(cust['Spending Score (1-100)'], bins=[0,10,20,30,40,50,60,70,80,90,100], rwidth=0.9)

plt.xlabel("Spending score")

plt.ylabel("frequency")

plt.title('Spending Score of customers')
#let's also drop the customer ID because it's not important

cust.drop("CustomerID", axis = 1, inplace=True)

#cust.drop("Genre", axis = 1, inplace=True)





cust["Genre"].replace("Male", 0, inplace=True)

cust["Genre"].replace("Female", 1, inplace=True)

cust
plt.scatter(cust['Genre'], cust['Spending Score (1-100)'])
plt.scatter(cust['Age'], cust['Spending Score (1-100)'])
plt.scatter(cust['Annual Income (k$)'], cust['Spending Score (1-100)'])
#Let's have a new dataframe first with only the Age and the spending score



cust_age=cust.drop(["Annual Income (k$)", "Genre"], axis = 1)
#we can test a cluster number 2 to verify what we saw in the 'Age' vs 'Spending score' graph. 

#However, we will use 4 clusters here as we saw in the elbow plot that 4 is the optimal number. See below



k_means_age=KMeans(n_clusters=4)



#We can also use this code below in case we want to determine the n_init number

#k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 20)



k_means_age.fit(cust_age)

labels = k_means_age.labels_

print(labels)
centers_age=k_means_age.cluster_centers_

centers_age
plt.figure(figsize=(10, 8))



plt.scatter(cust_age['Age'], 

            cust_age['Spending Score (1-100)'], 

            c=k_means_age.labels_, s=100)



plt.scatter(centers_age[:,0], centers_age[:,1], color='blue', marker='s', s=200) 



plt.xlabel('Age')

plt.ylabel('Spending Score')

plt.title('K-Means with 2 clusters')



plt.show()
score = silhouette_score (cust_age, k_means_age.labels_)



print("The score is = ", score)
elbowlist1 = []

for i in range(1,15): 

    k_means_age = KMeans(n_clusters=i, init="k-means++",random_state=0)

    k_means_age.fit(cust_age)

    elbowlist1.append(k_means_age.inertia_)  



plt.plot(range(1,15),elbowlist1,marker="*",c="black")

plt.title("Elbow plot for optimal number of clusters: age and spending")
#we drop the annual income column

cust_income=cust.drop(["Age", "Genre"], axis = 1)
#let's test cluster number 2 to verify what we saw in the 'Age' vs 'Spending score' graph.



k_means_income=KMeans(n_clusters=5)



#We can also use this code below in case we want to determine the n_init number

#k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 20)



k_means_income.fit(cust_income)

labels = k_means_income.labels_

print(labels)
centers_income=k_means_income.cluster_centers_

centers_income
plt.figure(figsize=(10, 8))



plt.scatter(cust_income['Annual Income (k$)'], 

            cust_income['Spending Score (1-100)'], 

            c=k_means_income.labels_, s=100)



plt.scatter(centers_income[:,0], centers_income[:,1], color='blue', marker='s', s=200) 



plt.xlabel('Annual Income in K$')

plt.ylabel('Spending Score')

plt.title('K-Means with 5 clusters')



plt.show()
score_2 = silhouette_score (cust_income, k_means_income.labels_)



print("The score is = ", score_2)
elbowlist2 = []

for i in range(1,15): 

    k_means_income = KMeans(n_clusters=i, init="k-means++",random_state=0)

    k_means_income.fit(cust_income)

    elbowlist2.append(k_means_income.inertia_)  



plt.plot(range(1,15),elbowlist2,marker="*",c="black")

plt.title("Elbow plot for optimal number of clusters: income and spending")
ax=plt.figure(figsize=(10, 8))



scatter=plt.scatter(cust_income['Annual Income (k$)'], 

            cust_income['Spending Score (1-100)'], 

            c=cust['Genre'], s=100)



plt.scatter(centers_income[:,0], centers_income[:,1], color='blue', marker='s', s=200) 



legend1 = ax.legend(*scatter.legend_elements(), loc="right", title="Gender")

ax.add_artist(legend1)



plt.xlabel('Annual Income in K$')

plt.ylabel('Spending Score')

plt.title('K-Means with 5 clusters')



plt.show()
#we drop the gender column



cust_3D=cust.drop("Genre", axis = 1)
k_means_3D=KMeans(n_clusters=5)



#We can also use this code below in case we want to determine the n_init number

#k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 20)



k_means_3D.fit(cust_3D)

labels = k_means_3D.labels_

print(labels)
centers_3D=k_means_3D.cluster_centers_

centers_3D
from matplotlib import interactive

interactive(True)



%matplotlib qt



from mpl_toolkits.mplot3d import Axes3D 

fig = plt.figure(1, figsize=(8, 6))

plt.clf()

ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)



plt.cla()

#plt.ylabel('Age', fontsize=18)

#plt.xlabel('Income', fontsize=16)

#plt.zlabel('Education', fontsize=16)

ax.set_xlabel('Age')

ax.set_ylabel('Income in 1000s $')

ax.set_zlabel('Spending')



ax.scatter(cust_3D['Age'], cust_3D['Annual Income (k$)'], cust_3D['Spending Score (1-100)'], c= labels.astype(np.float), s=200)



#note that it will open in a separate window. Drag the graph and rotate with your mouse to see the results interactively.



#better to download it to your own computer to visualise it properly



#%matplotlib inline    #to add after finishing to go back to charts inside the notebook.
%matplotlib inline   



elbowlist3 = []

for i in range(1,15): 

    k_means_3D = KMeans(n_clusters=i, init="k-means++",random_state=0)

    k_means_3D.fit(cust_income)

    elbowlist3.append(k_means_3D.inertia_)  



plt.plot(range(1,15),elbowlist3,marker="*",c="black")

plt.title("Elbow plot for optimal number of clusters: age, income and spending")
score_3 = silhouette_score (cust_3D, k_means_3D.labels_)



print("The score is = ", score_3)