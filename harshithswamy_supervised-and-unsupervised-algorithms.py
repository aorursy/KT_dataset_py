import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.metrics import accuracy_score, classification_report



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.svm import SVC

from sklearn.cluster import KMeans

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
data_set = pd.read_csv("../input/iris/Iris.csv")
# Rename column names

data_set.rename(columns={"SepalLengthCm":"SepalLength", 

                         "SepalWidthCm": "SepalWidth", 

                         "PetalLengthCm": "PetalLength", 

                         "PetalWidthCm": "PetalWidth"}, inplace=True)
data_set.head()
data_set.shape
data_set.describe()
data_set.info()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.scatterplot(x='PetalLength', y='PetalWidth', hue="Species", data=data_set)

plt.title("PetalLength vs PetalWidth")

plt.subplot(2,2,2)

sns.scatterplot(x='SepalLength', y='SepalWidth', hue="Species", data=data_set)

plt.title("SepalLength vs SepalWidth")

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.boxplot(x='Species', y='PetalLength', data=data_set)

plt.subplot(2,2,2)

sns.boxplot(x='Species', y='PetalWidth', data=data_set)

plt.subplot(2,2,3)

sns.boxplot(x='Species', y='SepalLength', data=data_set)

plt.subplot(2,2,4)

sns.boxplot(x='Species', y='SepalWidth', data=data_set)

plt.show()
sns.countplot(x="Species", data=data_set)

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.distplot(data_set["SepalLength"],  kde=True, bins=10, label="SepalLength")

plt.subplot(2,2,2)

sns.distplot(data_set["SepalWidth"],  kde=True, bins=10, label="SepalWidth")

plt.subplot(2,2,3)

sns.distplot(data_set["PetalLength"],  kde=True, bins=10, label="PetalLength")

plt.subplot(2,2,4)

sns.distplot(data_set["PetalLength"],  kde=True, bins=10, label="PetalLength")

plt.show()
plt.figure(figsize=(10,6)) 

sns.heatmap(data_set.corr(), annot=True, cmap='cubehelix_r')

plt.show()
# Remove unwanted column "Id"

data_set.drop("Id", axis=1, inplace=True)

# Get the unique Species

data_set["Species"].unique()
# Convert Categorical into Numeric

data_set["Species"] = data_set["Species"].map({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
# Consider all attributes for modelling

X = data_set.drop(["Species"], axis=1)

Y = data_set.loc[:, "Species"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

print("x_train shape: ", x_train.shape)

print("x_test shape: ", x_test.shape)

print("y_train shape: ", y_train.shape)

print("y_test shape: ", y_test.shape)
lr_model = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=300)

lr_model.fit(x_train, y_train)
lr_pred = lr_model.predict(x_test)

lr_accuracy = accuracy_score(y_test, lr_pred)
print(classification_report(y_test, lr_pred))
knn_model = KNeighborsClassifier()

knn_model.fit(x_train, y_train)
knn_pred = knn_model.predict(x_test)

knn_accuracy = accuracy_score(y_test, knn_pred)
print(classification_report(y_test, knn_pred))
rf_model = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42)

rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
print(classification_report(y_test, rf_pred))
dt_model = DecisionTreeClassifier(criterion="entropy", random_state=42)

dt_model.fit(x_train, y_train)
dt_pred = dt_model.predict(x_test)

dt_accuracy = accuracy_score(y_test, dt_pred)
print(classification_report(y_test, dt_pred))
svm_model = SVC()

svm_model.fit(x_train, y_train)
svm_pred = svm_model.predict(x_test)

svm_accuracy = accuracy_score(y_test, svm_pred)
print(classification_report(y_test, svm_pred))
print("SVM Accuracy: ",svm_accuracy)

print("KNN Accuracy: ", knn_accuracy)

print("Decision Tree Accuracy: ", dt_accuracy)

print("Random Forest Accuracy: ", rf_accuracy)

print("Logistic Regression Accuracy: ", lr_accuracy)
X = data_set.iloc[:, [1, 2, 3, 4]].values



wcss = []



for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

    

plt.figure(figsize=(10,5))

sns.lineplot(range(1, 11), wcss, marker='o',color='red')

plt.title('The Elbow Method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS')

plt.show()
kmeans_model = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_kmeans = kmeans_model.fit_predict(X)
#Visualising the clusters



plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Iris-Setosa')

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Iris-Versicolour')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'yellow', label = 'Iris-Virginica')



plt.legend()
x = ["Random Forest", "Logistic Regression", "Decision Tree", "SVM", "KNN"]

y = [rf_accuracy, lr_accuracy, dt_accuracy, svm_accuracy, knn_accuracy]

plt.bar(x=x, height=y)

plt.title("Algorithm Accuracy Comparison")

plt.xticks(rotation=15)

plt.xlabel("Algorithms")

plt.ylabel("Accuracy")

plt.show()