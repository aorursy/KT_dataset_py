from sklearn.datasets import load_iris

iris = load_iris()
#Designating variables



X = iris.data

y = iris.target



feature_names = iris.feature_names

target_names = iris.target_names

#Splitting data into training and testing set



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



print(X_train.shape)

print(X_test.shape)
#Selecting and building ML Model



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)



y_pred = knn.predict(X_test)
from sklearn import metrics



print(metrics.accuracy_score(y_test, y_pred))
#Making predictions with the model



sample = [[3,5,4,2], [2,3,5,4]]

predictions = knn.predict(sample)

pred_species = [iris.target_names[p] for p in predictions]

print("predictions", pred_species)
