import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()



from sklearn.preprocessing import LabelEncoder # This line of codes below can encoding the data set individually.



import os
# Function to importing Dataset 

def import_data(): 

    data = pd.read_csv('../input/hr_data.csv') 

    

    # Printing the dataswet shape 

    print ("\nDATASET LENGHT : ", len(data)) 

    print ("\nDATASET SHAPE : ", data.shape) 

    

    print("\n\n")

    

    # Printing the dataset obseravtions 

    print ("\nDATASET HEAD : \n", data.head()) 

    

    print("\n\n")

    

    print ("\nDATASET VARIABLE TYPES : \n", data.dtypes) 

    

    return data





# Funnction drop unnecessary columns.

def drop_columns(in_data):

    out_data = in_data.copy().loc[:, (in_data.columns != 'Work_accident') & (in_data.columns != 'left') & (in_data.columns != 'promotion_last_5years')]

    

    return out_data

hr_data = import_data()
# Ten (10) maximum satifaction per number of project. 

max_satifaction_project = hr_data.nlargest(10, "satisfaction_level")

max_satifaction_project.boxplot('number_project','satisfaction_level',rot = 30,figsize=(5,6))
project_count = hr_data['number_project'].value_counts()

sns.set(style="darkgrid")

sns.barplot(project_count.index, project_count.values, alpha=0.9)

plt.title('Frequency Distribution of Projects')

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('No. Project', fontsize=12)

plt.show()

# No data is missing from hr dataset.

hr_data.isnull().sum()


explode = (0.1, 0, 0.1, 0.1, 0.1,0.1)  # explode 1st slice



# Plot chart in descending order.

# Number of projects for each department for sales "#Number my assumption."

hr_data['number_project'].value_counts().plot(kind="pie", autopct='%2.1f%%', explode=explode, title='Number Project Distribution', )

# Transformation categorical data in hr dataset columns.

label_data = LabelEncoder()

hr_data["sales"] = label_data.fit_transform(hr_data["sales"])

hr_data["salary"] = label_data.fit_transform(hr_data["salary"])



hr_data.head()
# Let's view the distribution of the data, where is it possible to find groups?

# We are using boxplots of all the columns except the first (cust_id which is a string)



for col in hr_data.columns[:]:

    hr_data[col].plot(kind='box')

    plt.title('Box Plot for '+col)

    plt.show()
cluster_dataset = hr_data[['last_evaluation', 'satisfaction_level']]

cluster_dataset.head()
#sns.scatterplot(x='time_spend_company', y='number_project', hue='class', data = cluster_dataset)

cluster_ax = sns.scatterplot(x="last_evaluation", y="satisfaction_level", data=cluster_dataset)
np.array([4,3,2,1])*4
from sklearn.cluster import KMeans

hr_sample = drop_columns(hr_data)

hr_sample.head()
# Most of the time, Elbow method is used with either squared error(sse) or within cluster sum of errors(wcss).

# Which in this case is the cluster sum of errors(wcss)



wcss = []

for i in range( 1, 20 ):

    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300) 

    kmeans.fit_predict(hr_sample)

    wcss.append( kmeans.inertia_ )

    

plt.plot( wcss, 'bo-', label="WCSS")

plt.title("Computing WCSS for KMeans++")

plt.xlabel("Number of clusters")

plt.ylabel("WCSS")

plt.show()
num_of_clusters = 4

kmeans = KMeans(num_of_clusters, init="k-means++", n_init=10, max_iter=300) 

kmeans.fit(hr_sample)

hr_sample["cluster"] = kmeans.labels_

hr_sample

# Observation 4 clusters were created ranging from 0 - 3 where data columns are grouped which shared similar features.
hr_sample['cluster'].value_counts().plot(kind='bar', title='Distribution of person per group')

group_counts = hr_sample['cluster'].value_counts()

group_counts.name = 'Amount of person in each group'

pd.DataFrame(group_counts)
from sklearn.decomposition import PCA
k_sample = hr_sample[50:]

x_axis = hr_sample[['last_evaluation']]

y_axis = hr_sample[['satisfaction_level']]

# Specifically, I am devising a range from 1 to 20 (which represents our number of clusters), 

# and my score variable denotes the percentage of variance explained by the number of clusters.



Nc = range(1, 20)

kmeans = [KMeans(n_clusters=i) for i in Nc]

kmeans

score = [kmeans[i].fit(y_axis).score(y_axis) for i in range(len(kmeans))]

score



plt.plot(Nc,score)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve')

plt.show()
# Once the appropriate number of clusters have been identified (k=3), then the pca (Principal Component Analysis) 

# and kmeans algorithms can be devised.



pca = PCA(n_components=1).fit(y_axis)

pca_d = pca.transform(y_axis)

pca_c = pca.transform(y_axis)
pca = PCA(n_components=1).fit(y_axis)

pca_d = pca.transform(y_axis)

pca_c = pca.transform(x_axis)
# n_clusters equal to 3, and upon generating the k-means output use the data originally transformed using pca in order to plot the clusters:

kmeans=KMeans(n_clusters=3)



kmeansoutput=kmeans.fit(y_axis)



kmeansoutput



plt.figure('3 Cluster K-Means')

plt.scatter(pca_c[:, 0], pca_d[:, 0], c=kmeansoutput.labels_)

plt.xlabel('Last Evaluation')

plt.ylabel('Satisfaction Level')

plt.title('3 Cluster K-Means\n')

plt.show()



# The tree group may represent:

# 1. The minimum Satisfaction Level since Last Evaluation.

# 2. The average Satisfaction Level since Last Evaluation.

# 3. The maximum Satisfaction Level since Last Evaluation.