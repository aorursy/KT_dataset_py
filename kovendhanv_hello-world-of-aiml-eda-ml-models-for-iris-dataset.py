#Kaggle's way of welcoming us 
import numpy as np # linear algebra
import pandas as pd # data processing
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Loading the iris dataset in dataframe
irisMaster = pd.read_csv("/kaggle/input/iris/Iris.csv")

#Visualizing the top 5 records
irisMaster.head()
#Getting the data summary
irisMaster.info()
print("Total Rows : ",irisMaster.shape[0])
print("Total Columns : ",irisMaster.shape[1])
irisMaster["Species"].value_counts()
#Statistical summary of the dataset
irisMaster.describe().T
#Dropping the nominal feature "Id"
iris = irisMaster.drop("Id", axis = 1)
iris.head()
sns.set_style("whitegrid");
sns.pairplot(data = iris, hue = "Species", height = 4)
plt.show()
#Visualizing the Distribution & Box plots of all the four features
fig, ax = plt.subplots(4, 2, figsize = (14,14))
sns.distplot(iris["SepalLengthCm"] , color="skyblue", ax=ax[0, 0])
sns.boxplot(iris["SepalLengthCm"] , color="skyblue", ax=ax[1, 0])
sns.distplot(iris["SepalWidthCm"] , color="olive", ax=ax[0, 1])
sns.boxplot(iris["SepalWidthCm"] , color="skyblue", ax=ax[1, 1])
sns.distplot(iris["PetalLengthCm"] , color="gold", ax=ax[2, 0])
sns.boxplot(iris["PetalLengthCm"] , color="skyblue", ax=ax[3, 0])
sns.distplot(iris["PetalWidthCm"] , color="teal", ax=ax[2, 1])
sns.boxplot(iris["PetalWidthCm"] , color="skyblue", ax=ax[3, 1])
plt.suptitle("Distribution + Box and Whiskers Plots of all the features")
plt.legend()
plt.show()
iris_Setosa = iris[iris["Species"] == "Iris-setosa"];
iris_Virginica = iris[iris["Species"] == "Iris-virginica"];
iris_Versicolor = iris[iris["Species"] == "Iris-versicolor"];

print("Setosa: Mean and SD Deviation of Sepal Length : ", np.mean(iris_Setosa["SepalLengthCm"]), np.std(iris_Setosa["SepalLengthCm"]))
print("Setosa: Mean and SD Deviation of Petal Width : ", np.mean(iris_Setosa["SepalWidthCm"]), np.std(iris_Setosa["SepalWidthCm"]))
print("Setosa: Mean and SD Deviation of Sepal Length : ", np.mean(iris_Setosa["PetalLengthCm"]), np.std(iris_Setosa["PetalLengthCm"]))
print("Setosa: Mean and SD Deviation of Petal Width : ", np.mean(iris_Setosa["PetalWidthCm"]), np.std(iris_Setosa["PetalWidthCm"]))
print("\n")
print("Virginica: Mean and SD Deviation of Sepal Length : ", np.mean(iris_Virginica["SepalLengthCm"]), np.std(iris_Virginica["SepalLengthCm"]))
print("Virginica: Mean and SD Deviation of Petal Width : ", np.mean(iris_Virginica["SepalWidthCm"]), np.std(iris_Virginica["SepalWidthCm"]))
print("Virginica: Mean and SD Deviation of Sepal Length : ", np.mean(iris_Virginica["PetalLengthCm"]), np.std(iris_Virginica["PetalLengthCm"]))
print("Virginica: Mean and SD Deviation of Petal Width : ", np.mean(iris_Virginica["PetalWidthCm"]), np.std(iris_Virginica["PetalWidthCm"]))
print("\n")
print("Versicolor: Mean and SD Deviation of Sepal Length : ", np.mean(iris_Versicolor["SepalLengthCm"]), np.std(iris_Versicolor["SepalLengthCm"]))
print("Versicolor: Mean and SD Deviation of Petal Width : ", np.mean(iris_Versicolor["SepalWidthCm"]), np.std(iris_Versicolor["SepalWidthCm"]))
print("Versicolor: Mean and SD Deviation of Sepal Length : ", np.mean(iris_Versicolor["PetalLengthCm"]), np.std(iris_Versicolor["PetalLengthCm"]))
print("Versicolor: Mean and SD Deviation of Petal Width : ", np.mean(iris_Versicolor["PetalWidthCm"]), np.std(iris_Versicolor["PetalWidthCm"]))
fig, ax = plt.subplots(2,2, figsize = (12,14))
sns.boxplot(x='Species',y = "SepalLengthCm", data=iris, ax=ax[0, 0])
sns.boxplot(x='Species',y = "SepalWidthCm", data=iris, ax=ax[0,1])
sns.boxplot(x='Species',y = "PetalLengthCm", data=iris, ax=ax[1, 0])
sns.boxplot(x='Species',y = "PetalWidthCm", data=iris, ax=ax[1, 1])
plt.suptitle("Box and Whiskers Plots - Specieswise Distribution")
plt.legend()
plt.show()
irisCorr = iris.corr()
irisCorr
irisCovar = iris.cov()
irisCovar
sns.heatmap(irisCorr, annot = True, cmap = "YlGnBu", linewidth = 0.1)
plt.show()
sns.heatmap(irisCovar, annot = True, cmap = "YlGnBu", linewidth = 0.1)
plt.show()
#Let us explore how Petal Width and Petal Length features are distributed
sns.FacetGrid(iris, hue = "Species", height = 5).map(plt.scatter, "PetalWidthCm", "PetalLengthCm").add_legend();
plt.show()
from sklearn.decomposition import PCA  

#Input Features in 4-Dimensions in X variable, Preparing the target in Y variable
X_iris = iris.drop('Species', axis=1)
y_iris = iris['Species']

model = PCA(n_components=2) # hyperparameters setting
model.fit(X_iris)                      
X_iris_2D = model.transform(X_iris)  # Transform the data to two dimensions
iris['PCA1'] = X_iris_2D[:, 0]
iris['PCA2'] = X_iris_2D[:, 1]
sns.lmplot("PCA1", "PCA2", hue='Species', data=iris, fit_reg=False);
from sklearn.mixture import GaussianMixture      
model = GaussianMixture(n_components=3, covariance_type='full')  # hyperparameters
model.fit(X_iris)                    
y_gmm = model.predict(X_iris)        # determine the cluster labels
iris['cluster'] = y_gmm
sns.lmplot("PCA1", "PCA2", data=iris, hue='Species', col='cluster', fit_reg=False);
#Cross Validation & Train, Test set preparation
from sklearn.model_selection import train_test_split #to split the dataset for training and testing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

#Performance Measurements
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Importing various classifiers
#Linear Classifiers
from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Non-linear classifiers
from sklearn.naive_bayes import GaussianNB # Naive Bayes Classifier
from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours
from sklearn.svm import SVC  #for Support Vector Machine (SVM) Algorithm
from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algorithm

#Ensemble Classifier
from sklearn.ensemble import RandomForestClassifier

#Splitting the dataset into train and test sets in the ration 70:30
X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, test_size = 0.30, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('RFC', RandomForestClassifier()))
print(models)

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: Mean: %f SD: (%f)' % (name, cv_results.mean(), cv_results.std()))
#Compare Algorithms
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()
# Make predictions on validation dataset

predict_results = []
names = []
for name, model in models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predict_results.append(predictions)
    names.append(name)
    print('%s: ' % (name))
    # Evaluate predictions
    print(accuracy_score(y_test, predictions))
    confMat = confusion_matrix(y_test, predictions)
    print(confMat)
    plt.matshow(confMat, cmap = "Greys");\
    plt.title(name)
    print(classification_report(y_test, predictions))
