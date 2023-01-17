from sklearn import datasets, manifold, cluster, metrics

from sklearn.model_selection import train_test_split

%pylab inline



mnist = datasets.load_digits(); # I use the sklearn's dataset



print(mnist.data.shape);

print(mnist.target.shape);
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target);
tsne = manifold.TSNE(n_components=2, init='pca');

X_tsne = tsne.fit_transform(X_train);
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y_train); # Visualisation du résultat attendu
cls = cluster.KMeans(n_clusters=10);

cls.fit(X_tsne);
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=cls.labels_); # Visualization of the obtained result
print("Silhouette of train set : ", metrics.silhouette_score(X_train, cls.labels_));

print("Score on train set : ", metrics.adjusted_mutual_info_score(y_train, cls.labels_));
X_tsne_pred = tsne.fit_transform(X_test);

y_pred = cls.predict(X_tsne_pred);
plt.scatter(X_tsne_pred[:,0], X_tsne_pred[:,1], c=y_test); # Visualization of the expected result
plt.scatter(X_tsne_pred[:,0], X_tsne_pred[:,1], c=y_pred); # Visualization of the obtained result
print("Silhouette on test set : ", metrics.silhouette_score(X_test, y_pred));

print("Score on test set : ", metrics.adjusted_mutual_info_score(y_test, y_pred));