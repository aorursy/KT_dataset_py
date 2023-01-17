import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

df.head()
df.shape
df.dtypes
df.isnull().sum()
df.describe()
df.rename(columns = {'CustomerID': 'customer_id', 'Age': 'age',

                     'Gender': 'gender','Annual Income (k$)': 'income', 

                     'Spending Score (1-100)': 'spending_score'}, inplace = True)



# Create new features for scaled data

df['s_age'] = df['age']

df['s_income'] = df['income']

df['s_spending_score'] = df['spending_score']
# Scale age, income, and spending score



from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()



df[['s_age', 's_income', 's_spending_score',]] = scaler.fit_transform(df[['age', 'income', 'spending_score']])
from sklearn.cluster import KMeans



X = df[['s_age', 's_income', 's_spending_score']].values



wcss = []



for i in range(1, 11):

    kmeans = KMeans(n_clusters = i)

    kmeans.fit_predict(X)

    wcss.append(kmeans.inertia_)
# Visualize an Elbow Plot to Identify the Optimal Number of Clusters



plt.plot(range(1,11), wcss)

plt.title('Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('wcss')
kmeans = KMeans(n_clusters = 4)

y_predicted = kmeans.fit_predict(X)

df['cluster'] = y_predicted
df.head()
# Convert cluster number from integer to string and order them for visualization purposes



df['cluster'] = df['cluster'].apply(str)

df = df.sort_values(by=['cluster'])
import plotly.express as px



fig = px.scatter_3d(df, x = 's_age', y = 's_income', z = 's_spending_score', color = 'cluster', opacity = 0.7)

fig.update_layout(scene = dict(

                    xaxis = dict(

                         backgroundcolor = "rgb(200, 200, 230)",

                         gridcolor = "white",

                         showbackground = True,

                         zerolinecolor = "white",

                         title = "Age"),

                    yaxis = dict(

                        backgroundcolor = "rgb(230, 200,230)",

                        gridcolor = "white",

                        showbackground = True,

                        zerolinecolor = "white",

                        title = "Annual Income"),

                    zaxis = dict(

                        backgroundcolor = "rgb(230, 230,200)",

                        gridcolor = "white",

                        showbackground = True,

                        zerolinecolor = "white",

                        title = "Spending Score")

                  ))

fig.show()
plt.figure(1 , figsize = (15 , 6))

n = 0 



arr = list(map(str,range(0, 4)))



for cluster in arr:

    for feature in ['age' , 'income' , 'spending_score']:

        n += 1

        plt.subplot(4, 3 , n) 

        plt.tight_layout()

        plt.xlim(0, df[feature].max() + 15)

        sns.distplot(df[df['cluster'] == cluster][feature])

        plt.title('{} - Cluster {}'.format(feature, cluster))



plt.show()
df['cluster'].value_counts(normalize=True) * 100
arr = list(map(str,range(0, 4)))



print('Entire df:')

print(df[['age', 'income', 'spending_score']].describe())



for i in arr:

    print('\nCluster Number '  + i + ':' + '\n')

    print(df[df['cluster'] == i][['age', 'income', 'spending_score']].describe())