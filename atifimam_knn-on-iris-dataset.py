import  pandas as pd

import matplotlib.pyplot as plt

import  seaborn as sns
iris=pd.read_csv("../input/iris.csv")
iris.columns
print(iris[:5])
iris.shape
print(iris.describe())
print(iris.groupby("species").size())
feature_columns=['sepal_length','sepal_width','petal_length','petal_width']

X=iris[feature_columns].values

y=iris['species'].values
print(X[:5])

print(y[:5])
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)
from sklearn.model_selection import train_test_split  

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
from sklearn.preprocessing import  StandardScaler

scaler=StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test=scaler.transform(X_test)
print(X_train[:5])
from sklearn.neighbors import  KNeighborsClassifier

classifier=KNeighborsClassifier(n_neighbors=3)

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
#For evaluating an algorithm, confusion matrix, precision, recall and f1 score are the most commonly used metrics. The confusion_matrix and classification_report methods of the sklearn.metrics can be used to calculate these metrics.
from sklearn.metrics import  classification_report ,confusion_matrix,accuracy_score

print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
accuracy = accuracy_score(y_test, y_pred)*100

print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')

# creating list of K for KNN

from sklearn.model_selection import  cross_val_score

k_list = list(range(1,50,2))

# creating list of cv scores

cv_scores = []



# perform 10-fold cross validation

for k in k_list:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')

    cv_scores.append(scores.mean())
# changing to misclassification error

MSE = [1 - x for x in cv_scores]



plt.figure()

plt.figure(figsize=(15,10))

plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')

plt.xlabel('Number of Neighbors K', fontsize=15)

plt.ylabel('Misclassification Error', fontsize=15)

sns.set_style("whitegrid")

plt.plot(k_list, MSE)



plt.show()
# finding best k

best_k = k_list[MSE.index(min(MSE))]

print("The optimal number of neighbors is %d." % best_k)