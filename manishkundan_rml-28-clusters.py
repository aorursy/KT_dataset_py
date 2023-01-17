# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
os.path.isfile('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv') # Defining the path
x= pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv') # Read the file

x.head(10) # Check if the file is in read mode
x.dtypes
missing_values = x.isnull().sum() # To check for any missing values

missing_values.head(21)
import seaborn as sns
missing_values = x.isnull() # Removing Error: Inconsistent shape b/w condition and the input (got (21, 1) and (21,))

sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')



x.shape # To check Row & col
import matplotlib.pyplot as plt #To check distribution



plt.rcParams ['figure.figsize']= (21,10)

plt.subplot(1, 2, 1)

sns.set(style = 'whitegrid')

sns.distplot(x['tenure'])

plt.title('Distribution', fontsize = 20)

plt.xlabel('tenure')

plt.ylabel('MonthlyCharges')

labels = ['Female', 'Male'] #To check, genderwise split

size = x['gender'].value_counts()

colors = ['blue', 'orange']

explode = [0, 0.1]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')

plt.title('gender', fontsize = 20)

plt.axis('off')

plt.legend()

plt.show()
Data=x.iloc[:,[5,18]] # Conversion of datatype float(18) to int
from sklearn.cluster import KMeans



wcss = []

for i in range(1, 11):

    km = KMeans(n_clusters = i, max_iter = 300, n_init = 10, random_state = 0)

    km.fit(Data)

    wcss.append(km.inertia_)

    

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.show()
km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(Data)

Data = np.array(Data) 

# solution, convert the dataframe to a np.array

#Visualizing the clusters for k=4

plt.scatter(Data[y_means == 0, 0], Data[y_means == 0, 1], s = 100, c = 'pink', label = 'miser')

plt.scatter(Data[y_means == 1, 0], Data[y_means == 1, 1], s = 100, c = 'yellow', label = 'general')

plt.scatter(Data[y_means == 2, 0], Data[y_means == 2, 1], s = 100, c = 'cyan', label = 'target')

plt.scatter(Data[y_means == 3, 0], Data[y_means == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')

plt.scatter(Data[y_means == 4, 0], Data[y_means == 4, 1], s = 100, c = 'orange', label = 'careful')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 200 , c =  'blue' , label = 'centeroid')



plt.style.use('fivethirtyeight')

plt.title('K Means Clustering', fontsize = 20)

plt.xlabel('SeniorCitizen')

plt.ylabel('tenure')

plt.legend()

plt.grid()

plt.show()
import scipy.cluster.hierarchy as sch 



dendrogram = sch.dendrogram(sch.linkage(Data, method = 'ward'))

plt.title('Dendrogam', fontsize = 20)

plt.xlabel('SeniorCitizen')

plt.ylabel('tenure')

plt.show()

from sklearn.cluster import AgglomerativeClustering



hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(Data)

Data= np.array(Data) # Else ValueError: could not convert string to float: '7590-VHVEG'

plt.scatter(Data[y_hc == 0, 0], Data[y_hc == 0, 1], s = 100, c = 'pink', label = 'miser')

plt.scatter(Data[y_hc == 1, 0], Data[y_hc == 1, 1], s = 100, c = 'yellow', label = 'general')

plt.scatter(Data[y_hc == 2, 0], Data[y_hc == 2, 1], s = 100, c = 'cyan', label = 'target')

plt.scatter(Data[y_hc == 3, 0], Data[y_hc == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')

plt.scatter(Data[y_hc == 4, 0], Data[y_hc == 4, 1], s = 100, c = 'orange', label = 'careful')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')



plt.style.use('fivethirtyeight')

plt.title('Hierarchial Clustering', fontsize = 20)

plt.xlabel('MonthlyCharges')

plt.ylabel('tenure')

plt.legend()

plt.grid()

plt.show()
from sklearn.cluster import AgglomerativeClustering



hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(Data)

Data= np.array(Data) # Else ValueError: could not convert string to float: '7590-VHVEG'

plt.scatter(Data[y_hc == 0, 0], Data[y_hc == 0, 1], s = 100, c = 'pink', label = 'miser')

plt.scatter(Data[y_hc == 1, 0], Data[y_hc == 1, 1], s = 100, c = 'yellow', label = 'general')

plt.scatter(Data[y_hc == 2, 0], Data[y_hc == 2, 1], s = 100, c = 'cyan', label = 'target')

plt.scatter(Data[y_hc == 3, 0], Data[y_hc == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')

plt.scatter(Data[y_hc == 4, 0], Data[y_hc == 4, 1], s = 100, c = 'orange', label = 'careful')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')



plt.style.use('fivethirtyeight')

plt.title('Hierarchial Clustering', fontsize = 20)

plt.xlabel('MonthlyCharges')

plt.ylabel('SeniorCitizen')

plt.legend()

plt.grid()

plt.show()