from sklearn import datasets

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import AgglomerativeClustering
iris = datasets.load_iris()

features = iris.data
scaler = StandardScaler()

features_std = scaler.fit_transform(features)
cluster = AgglomerativeClustering(n_clusters = 3)
model = cluster.fit(features_std)
model.labels_
pred = 0



for i in range(150):

        if i<50:

            if model.labels_[i]==1:

                pred +=1

        elif i<100:

            if model.labels_[i]==2:

                pred +=1

        elif i<150:

            if model.labels_[i]==0:

                pred +=1

        

accuracy = (pred/150)*100

print(accuracy)