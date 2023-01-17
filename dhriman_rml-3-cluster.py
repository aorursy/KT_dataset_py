# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Step 1: Import data analysis modules



# for basic mathematics operation 

import numpy as np

import pandas as pd

from pandas import plotting



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

%matplotlib inline



# for interactive visualizations

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected = True)

import plotly.figure_factory as ff



# for path

import os
# importing the dataset



#Step 2 : Data import



input_file = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

input_file.head(5)
# checking if there is any NULL data



input_file.isnull().any().any()
input_file.dtypes
import warnings

warnings.filterwarnings('ignore')



plt.rcParams['figure.figsize'] = (18, 8)



plt.subplot(1, 2, 1)

sns.set(style = 'whitegrid')

sns.distplot(input_file['MonthlyCharges'])

plt.title('Distribution of Monthly Charges', fontsize = 20)

plt.xlabel('Range of Monthly Charges')

plt.ylabel('Count')
labels = ['Female', 'Male']

size = input_file['gender'].value_counts()

colors = ['green', 'orange']

explode = [0, 0.1]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')

plt.title('Gender', fontsize = 20)

plt.axis('off')

plt.legend()

plt.show()
x = input_file.iloc[:, [3, 4]].values



# let's check the shape of x

print(x.shape)
import scipy.cluster.hierarchy as sch 



dendrogram = sch.dendrogram(sch.linkage(Data, method = 'ward'))

plt.title('Dendrogam', fontsize = 20)

plt.xlabel('MonthlyCharges')

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

plt.ylabel('tenure')

plt.legend()

plt.grid()

plt.show()
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
Data=input_file.iloc[:,[5,18]] # Conversion of datatype float(18) to int
kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

ymeans = kmeans.fit_predict(x)



plt.rcParams['figure.figsize'] = (10, 10)

plt.title('Cluster of Ages', fontsize = 30)



plt.scatter(x[ymeans == 0, 0], x[ymeans == 0, 1], s = 100, c = 'pink', label = 'Usual Customers' )

plt.scatter(x[ymeans == 1, 0], x[ymeans == 1, 1], s = 100, c = 'orange', label = 'Priority Customers')

plt.scatter(x[ymeans == 2, 0], x[ymeans == 2, 1], s = 100, c = 'lightgreen', label = 'Target Customers(Young)')

plt.scatter(x[ymeans == 3, 0], x[ymeans == 3, 1], s = 100, c = 'red', label = 'Target Customers(Old)')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 50, c = 'black')



plt.style.use('fivethirtyeight')

plt.xlabel('MonthlyCharges')

plt.ylabel('tenure')

plt.legend()

plt.grid()

plt.show()
