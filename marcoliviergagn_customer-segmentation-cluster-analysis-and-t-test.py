import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from scipy import stats
#Importe dataset

df = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

df.head()
df.info()
df.describe()
#Plot relation between all variables

plt.style.use('ggplot')

sns.pairplot(data=df,hue= 'Gender')
#Number of Male and Female

sns.countplot(x='Gender',data=df)
#BoxPlot 

for col in df.select_dtypes('int64'): 

    plt.figure(figsize=(8,8))

    sns.boxplot(x="Gender" ,y=col, data=df)
#SCALING THE DATA



#Drop 'Gender from X beacause Categorical variables do not fit into a K-Means algorithm'

X = df.drop(['CustomerID','Gender','Age'],axis=1)



#Scale the data

scale = StandardScaler()

X_scaled = scale.fit_transform(X)



X_scaled
#Create the cluster with KMeans



plt.figure(figsize=(10,8))



#Get the number of cluster to minimize the inertia. 

wcss =[]

for k in range(1,15):

    kmean=KMeans(n_clusters=k,init='k-means++',max_iter=300,n_init=20,random_state=45)

    kmean.fit(X_scaled)

    wcss.append(kmean.inertia_)

plt.plot(range(1,15),wcss)

plt.ylabel('WCSS')

plt.xlabel('Number of cluster')

plt.title('ELBOW METHOD')
#Build the model

kmean = KMeans(n_clusters=5,init='k-means++',max_iter=300,random_state=345)

model=kmean.fit(X_scaled)

cluster = model.predict(X_scaled)

cluster
#Look at the inertia

model.inertia_
#Plot the graphics with cluster in different color



plt.figure(figsize=(10,8))

sns.scatterplot(X_scaled[:,0],X_scaled[:,1],hue=model.labels_,palette='muted')



#Add the centroids to the graphics

sns.scatterplot(x=model.cluster_centers_[:,0],y=model.cluster_centers_[:,1],color='black',marker='d',sizes=20)

plt.legend(loc='upper right')

plt.title('Cluster on Annual Income and Spending Score')
#Bring back the cluster to the original dataset



df['cluster']=cluster

df
#Print back the scatter plot on the original dataset for interpretation

plt.figure(figsize=(10,8))

sns.scatterplot(x=df['Annual Income (k$)'],y=df['Spending Score (1-100)'],hue=model.labels_,palette='muted')

plt.legend(loc='upper right')

plt.title('Cluster on Annual Income and Spending Score')
cluster_0 = df[df['cluster']==0]

cluster_0.describe()
#Create a definition that will print multiple graph for each cluster and for reproducibility



def cluster_plot(cluster_df):

    plt.figure(figsize=(10,10))

    plt.subplots_adjust(bottom=0.5,top=2.5)

    plt.subplot(4,1,1)

    

    sns.countplot(cluster_df['Gender'])

    plt.title('Count of Men and Women')

    

    plt.subplot(4,1,2)

    sns.scatterplot(x='Age',y='Annual Income (k$)',size='Spending Score (1-100)',hue='Gender',data=cluster_df)

    plt.title ('Age vs Annual Income with spending score and sex distribution')

    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")



    plt.subplot(4,1,3)

    sns.boxplot(x='Gender',y='Spending Score (1-100)',data=cluster_df)

    plt.title('Spending score vs Gender')



    plt.subplot(4,1,4)

    sns.boxplot(x='Gender',y='Annual Income (k$)',data=cluster_df)

    plt.title('Annual Income vs Gender')



cluster_plot(cluster_0)
#Create the cluster

cluster_2 = df[df['cluster']==2]

cluster_2.head()
#Get some descriptive statistics

cluster_2.describe()
#General Visualisation

cluster_plot(cluster_2)
#Create a definition for the T-Test



def t_test (test_cluster):

    

    #Binarize the Male and Female observation

    test_cluster['Gender'].replace({'Female':1,'Male':0},inplace=True)



    #Create a sample for the male observation

    sample_a = test_cluster[test_cluster['Gender']==0]

    sample_a= sample_a.loc[:,('Gender','Spending Score (1-100)')]



    #Create a sample for the female observation

    sample_b = test_cluster[test_cluster['Gender']== 1]

    sample_b= sample_b.loc[:,('Gender','Spending Score (1-100)')]



    # Execute the t-test

    test=stats.ttest_ind(sample_b,sample_a)

    print(test.pvalue[1])

    

    # Print the response for the result

    if test.pvalue[1] < 0.05:

        print('Reject the Null hypothesis, meaning there is a statistical difference in spending score between Men and Women')

    else:

        print ('Accept the Null hypothesis, meaning there is no statistical difference in spending score between Men ans Women')



t_test(cluster_2)
#Create cluster #3

cluster_3 = df[df['cluster']==3]

cluster_3.head()
# Descriptive Analysis

cluster_3.describe()
#Generla Visualisation

cluster_plot(cluster_3)
# Statistical T-Test

t_test(cluster_3)
#Create the cluster dataframe

cluster_4 = df[df['cluster']==4]

cluster_4.head()
cluster_4.describe()
#Plot some visualisation

cluster_plot(cluster_4)
#Execute the statistical test

t_test(cluster_4)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session