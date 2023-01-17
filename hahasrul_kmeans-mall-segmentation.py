import pandas as pd 

import matplotlib.pyplot as plt 

%matplotlib inline 

import seaborn as sns


import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# uba file cvs menjadi dataframe 





df = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv') 



# show first 3 row 

df.head(3)


df.keys()



df = df.rename(columns={'Gender': 'gender',

                        'Age' : 'age',

                        'Annual Income (k$)' : 'annual_income',

                        'Spending Score (1-100)': 'spending_score'})



# rename categorical to numeric 

df['gender'].replace(['Female', 'Male'], [0,1] , inplace=True)



# show preprocessed data 

df.head(3)
from sklearn.cluster import KMeans 

# remove kolom customer id dan gender 

X = df.drop(['CustomerID', 'gender'] , axis = 1) 



# membuat list yang berisi inertia 

clusters = [] 

for i in range(1,11):

  km = KMeans(n_clusters = i ).fit(X) 

  clusters.append(km.inertia_) 



# create inertia plot 



fig, ax = plt.subplots(figsize=(8,4)) 

sns.lineplot(x=list(range(1,11)), y = clusters, ax = ax ) 

ax.set_title('Elbow plot') 

ax.set_xlabel('Clusters') 

ax.set_ylabel('Intertia')



km5 = KMeans(n_clusters = 5).fit(X) 

# menambahkankolom label pad adataset 

X['Labels'] = km5.labels_ 



# create KMeans Visualization with 5 cluster 

plt.figure(figsize = (8,4)) 

sns.scatterplot(X['annual_income'] , 

                X['spending_score'], 

                hue = X['Labels'],

                palette = sns.color_palette('hls', 5 ))



plt.title('KMeans With 5 Cluster') 

plt.show()