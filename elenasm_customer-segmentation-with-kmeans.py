



import numpy as np 

import pandas as pd 







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from sklearn import preprocessing



import matplotlib.pyplot as plt

db = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')

db.head()

#encode gender

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()

gender_cat = db[['Gender']] 

gender_cat_encoded = ordinal_encoder.fit_transform(gender_cat)



#make it a dataframe

gender_new = pd.DataFrame(gender_cat_encoded)

gender_new.columns = ['gender_new']



#add it to the original database

db = db.merge(gender_new,  how = 'inner', left_index = True, right_index = True)



#remove the categorical Gender variable because we don't need it anymore

db = db.drop('Gender', axis = 1)

db.head()
db.info()
db2 = db.iloc[:,[2,3]]

db2.head()
from sklearn.cluster import KMeans





cl = []

list_k = list(range(1, 10))



for k in list_k:

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(db2)

    cl.append(kmeans.inertia_)



# Plot sse against k

plt.figure(figsize=(6, 6))

plt.plot(list_k, cl, '-o')

plt.xlabel(r'Clusters number')

plt.ylabel('Sum of squared distance')
kmeans = KMeans(n_clusters=5) 

kmeans.fit(db2)



y_km = kmeans.fit_predict(db2)



db2 = np.array(db2)



plt.title("Clusters", fontsize=20)

plt.xlabel("Annual Income")

plt.ylabel("Spending Score")



plt.scatter(db2[y_km ==0,0], db2[y_km == 0,1], c='red')

plt.scatter(db2[y_km ==1,0], db2[y_km == 1,1], c='black')

plt.scatter(db2[y_km ==2,0], db2[y_km == 2,1], s=100, c='blue')

plt.scatter(db2[y_km ==3,0], db2[y_km == 3,1], s=100, c='orange')

plt.scatter(db2[y_km ==4,0], db2[y_km == 4,1], s=100, c='yellow')
db3 = db.iloc[:,[1,3]]

db3.head()
cl = []

list_k = list(range(1, 10))



for k in list_k:

    kmeans = KMeans(n_clusters=k)

    kmeans.fit(db3)

    cl.append(kmeans.inertia_)



# Plot sse against k

plt.figure(figsize=(6, 6))

plt.plot(list_k, cl, '-o')

plt.xlabel(r'Clusters number')

plt.ylabel('Sum of squared distance')
kmeans = KMeans(n_clusters=4) 

kmeans.fit(db3)



y_km = kmeans.fit_predict(db3)



db3 = np.array(db3)



plt.title("Clusters", fontsize=20)

plt.xlabel("Age")

plt.ylabel("Spending Score")



plt.scatter(db3[y_km ==0,0], db3[y_km == 0,1], c='red')

plt.scatter(db3[y_km ==1,0], db3[y_km == 1,1], c='black')

plt.scatter(db3[y_km ==2,0], db3[y_km == 2,1], s=100, c='blue')

plt.scatter(db3[y_km ==3,0], db3[y_km == 3,1], s=100, c='orange')
