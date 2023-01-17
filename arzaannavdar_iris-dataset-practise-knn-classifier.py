from sklearn.cross_validation import train_test_split

from sklearn.datasets import load_iris

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
iris=load_iris()
#Split the data into independent variables(X) and dependent variable (Y)

X=iris.data

Y=iris.target
# Split Data into training and testing data

x_train,x_test,y_train,y_test = train_test_split(X,Y)
# Call K nearest neighbours function (for now)

knn = KNeighborsClassifier(n_neighbors=5)
#Fit the model with training data

knn.fit(x_train,y_train)
# Predict the dependent variable

y_pred=knn.predict(x_test)
# Check the accuracy score with the y_test data

AccuScore= metrics.accuracy_score(y_test, y_pred)

print(AccuScore)