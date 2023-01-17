# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Airplane_Crashes_and_Fatalities_Since_1908.csv')

df.info()
df = df.drop('Flight #',axis=1)

df = df.drop('Time',axis=1)

df = df.drop('Route',axis=1)

df = df.drop('cn/In',axis=1)

df = df.drop('Summary',axis=1)

df.notnull()





df.info()
df.head()
X = df.drop('Location',axis=1)

X = X.drop('Type',axis=1)

X = X.drop('Date',axis=1)

X = X.drop('Operator',axis=1)

X = X.drop('Ground',axis=1)

X = X.drop('Registration',axis=1)

X = X.dropna()

xc=X

X = X.values

y = (df['Type'].values)





#Xamples=[]

#Xamples[:,0] = X

#Xamples[:,1] = Y



print(X)

    
samples = X.astype(int)



print(samples)



from sklearn.cluster import KMeans

from sklearn.preprocessing import normalize

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline 



scaler = StandardScaler() 

kmeans = KMeans(n_clusters=3)



model = make_pipeline(scaler,kmeans ) 

#nm = normalize(1)



#samples = nm(samples)





model.fit(samples)

labels = model.predict(samples)



import matplotlib.pyplot as plt 



xs = samples[:,0]

ys = samples[:,1]



plt.scatter(xs, ys, c=labels)



centroids = kmeans.cluster_centers_



centroids_x = centroids[:,0]

centroids_y = centroids[:,1]



plt.scatter(centroids_x,centroids_y,color='Red',marker='D',s=50)



print(kmeans.inertia_)





plt.show()

        
centroids = kmeans.cluster_centers_



centroids_x = centroids[:,0]

centroids_y = centroids[:,1]



plt.scatter(centroids_x,centroids_y,marker='D',s=50)

plt.show()
print(kmeans.inertia_)

fig,ax = plt.subplots()

ax.bar( xs,ys, width=6, color='r')



plt.show()