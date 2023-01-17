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
#Import all required library

import pandas as pd

import numpy as np

import os 

# to save model

import pickle

# Import visualization modules

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



from pandas import plotting

plt.style.use('fivethirtyeight')



# for interactive visualizations

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

from plotly import tools

init_notebook_mode(connected = True)

import plotly.figure_factory as ff



# for path

import os

print(os.listdir('../input/'))

data = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')



data.head (5)
data.describe()
# checking if there is any NULL data

missing_values = data.isnull()

missing_values.head(5)

# checking via Heat Map

sns.heatmap(data = missing_values, yticklabels=False, cbar=False, cmap='viridis')
# See in Pi Chart

labels = ['Female', 'Male']

size = data['gender'].value_counts()

colors = ['lightpink', 'orange']

explode = [0, 0.1]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(size, colors = colors, explode = explode, labels = labels, shadow = True, autopct = '%.2f%%')

plt.title('gender', fontsize = 20)

plt.axis('off')

plt.legend()

plt.show()
#Distribution of age

plt.rcParams['figure.figsize'] = (15, 8)

sns.countplot(data['tenure'], palette = 'hsv')

plt.title('Distribution of Age', fontsize = 20)

plt.show()
# Correlation B/w Senior Citizen, Tenure , monthlycharges

plt.rcParams['figure.figsize'] = (15, 8)

sns.heatmap(data.corr(),  annot = True)

plt.title('Heatmap for the Data', fontsize = 20)

plt.show()

#Checking Tenure Vs Totalcharges

plt.rcParams['figure.figsize'] = (18, 7)

sns.stripplot(data['tenure'], data['MonthlyCharges'], palette = 'Purples', size = 10)

plt.title('Gender vs Monthly charges', fontsize = 20)

plt.show()
x = data.iloc[:, [5, 18]].values



# let's check the shape of x

print(x.shape)
#Apply K mean Algo



from sklearn.cluster import KMeans



wcss = []

for i in range(1, 11):

    km = KMeans(n_clusters = i, max_iter = 300, n_init = 10, random_state = 10)

    km.fit(x)

    wcss.append(km.inertia_)

    

plt.plot(range(1, 11), wcss)

plt.title('The Elbow Method', fontsize = 20)

plt.xlabel('No. of Clusters')

plt.ylabel('wcss')

plt.show()
km = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(x)

x=np.array(x)

plt.scatter(x[y_means == 0,1], x[y_means == 0, 1], s = 100, c = 'pink', label = 'miser')

plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'red', label = 'general')

plt.scatter(x[y_means == 2, 0], x[y_means == 2, 1], s = 100, c = 'lightblue', label = 'target')

plt.scatter(x[y_means == 3, 0], x[y_means == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')

plt.scatter(x[y_means == 4, 0], x[y_means == 4, 1], s = 100, c = 'orange', label = 'careful')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 200 , c =  'blue' , label = 'centeroid')



plt.style.use('fivethirtyeight')

plt.title('K Means Clustering', fontsize = 20)

plt.xlabel('MonthlyCharges')

plt.ylabel('tenure')

plt.legend()

plt.grid()

plt.show()
from sklearn.cluster import AgglomerativeClustering



hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(x)



plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'pink', label = 'miser')

plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'yellow', label = 'general')

plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'cyan', label = 'target')

plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'magenta', label = 'spendthrift')

plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'orange', label = 'careful')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')



plt.style.use('fivethirtyeight')

plt.title('Hierarchial Clustering', fontsize = 20)

plt.xlabel('MonthlyCharges')

plt.ylabel('tenure')

plt.legend()

plt.grid()

plt.show()