import numpy as np # linear algebra

import pandas as pd # data processing

from math import pi



#Data Visualization libraries 

import matplotlib.pyplot as plt 

import seaborn as sns 



# Import necessary modules

from scipy.stats import randint



#Selection the Model 

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.cluster import KMeans



#Selection the Regulation 

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNet



#Model Selection 

from sklearn.model_selection import train_test_split,cross_val_score

from sklearn.model_selection import GridSearchCV,RandomizedSearchCV



#Pipeline

from sklearn.pipeline import Pipeline,make_pipeline



#Data Preprocessing 

from sklearn.preprocessing import StandardScaler,scale,Imputer



#Metrics “ Measure Model Performance” 

from sklearn.metrics import mean_squared_error,accuracy_score

from sklearn.metrics import classification_report , confusion_matrix

from sklearn.metrics import roc_curve ,roc_auc_score



# Input data files are available in the "../input/" directory.

import os

print(os.listdir("../input"))



iris = pd.read_csv("../input/Iris.csv") #load the dataset
iris.info()
iris.shape
iris.describe() #to know statistical data and the features scale 
iris.head(5)
plt.figure(figsize=(15,8))

plt.subplot(2,2,1)

sns.swarmplot(x='Species',y='PetalLengthCm',data=iris)

plt.subplot(2,2,2)

sns.swarmplot(x='Species',y='PetalWidthCm',data=iris)

plt.subplot(2,2,3)

sns.swarmplot(x='Species',y='SepalLengthCm',data=iris)

plt.subplot(2,2,4)

sns.swarmplot(x='Species',y='SepalWidthCm',data=iris)

sns.set(style="whitegrid", palette="muted")

# The diagonal elements in a pairplot show the histogram by default,but we update it to "kde"

# kde: which creates and visualizes a kernel density estimate of the underlying feature

iris = iris.drop("Id", axis=1)

sns.pairplot(iris, hue="Species", size=3, diag_kind="kde")

plt.show()
plt.figure(figsize=(15,8)) 

sns.heatmap(iris.corr(),annot=True,cmap='Greens',cbar=False) #The correlation matrix

plt.rc('xtick', labelsize=15)    # fontsize of the tick labels

plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
X=iris.loc[:,['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] #attributes

y = iris.loc[:,'Species'] #Target Variables

param_grid = {'n_neighbors': np.arange(1, 50)}

knn = KNeighborsClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

knn_cv = GridSearchCV(knn, param_grid, cv=5) #Hyperparameter tuning using GridSearchCV 

knn_cv.fit(X_train, y_train)

y_pred = knn_cv.predict(X_test)

print(knn_cv.best_params_)

print("KNN_CV Score:",knn_cv.score(X_test, y_test))

#or you can use Accuracy score 

print('The accuracy of the K-Nearest Neighbours is',accuracy_score(y_test, y_pred))

#KNN For Spal 

X_spal = iris.loc[:,['SepalLengthCm','SepalWidthCm']]

param_grid = {'n_neighbors': np.arange(1, 50)}

knn = KNeighborsClassifier()

X_train, X_test, y_train, y_test = train_test_split(X_spal, y, test_size=0.2, random_state=21)

knn_cv_spal = GridSearchCV(knn, param_grid, cv=5) #Hyperparameter tuning using GridSearchCV 

knn_cv_spal.fit(X_train, y_train)

y_pred = knn_cv_spal.predict(X_test)

print("Spal")

print(knn_cv_spal.best_params_)

print("KNN_CV_spal Score:",knn_cv_spal.score(X_test, y_test))

#or you can use Accuracy score 

print('The accuracy of the KNN For Spal is',accuracy_score(y_test, y_pred))





#KNN For Patal 

X_patal = iris.loc[:,['PetalLengthCm','PetalWidthCm']]

param_grid = {'n_neighbors': np.arange(1, 50)}

knn = KNeighborsClassifier()

X_train, X_test, y_train, y_test = train_test_split(X_patal, y, test_size=0.2, random_state=21)

knn_cv_patal = GridSearchCV(knn, param_grid, cv=5) #Hyperparameter tuning using GridSearchCV 

knn_cv_patal.fit(X_train, y_train)

y_pred_patal= knn_cv_patal.predict(X_test)

print("____________________________")

print("patal")

print(knn_cv_patal.best_params_)

print("KNN_CV_patal Score:",knn_cv_patal.score(X_test, y_test))

#or you can use Accuracy score 

print('The accuracy of the KNN for Patal is',accuracy_score(y_test, y_pred))



from sklearn.metrics import accuracy_score,mean_squared_error

from sklearn.metrics import classification_report,confusion_matrix,roc_curve,roc_auc_score

logreg=LogisticRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)



# Confusion matrix and Classification Report in scikit-learn

print('Confusuin_Martix:\n',confusion_matrix(y_test, y_pred))

print('Classification Report: \n ',classification_report(y_test,y_pred))

print('The accuracy of the Logistic Regression is',accuracy_score(y_test, y_pred))

samples = iris.loc[:,['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] #attributes



#First : Get the Best KMeans 

ks = range(1,6)

inertias=[]

for k in ks :

    # Create a KMeans clusters

    kc = KMeans(n_clusters=k)

    kc.fit(samples)

    inertias.append(kc.inertia_)



# Plot ks vs inertias

f, ax = plt.subplots(figsize=(15, 8))

plt.plot(ks, inertias, '-o')

plt.xlabel('Number of clusters, k')

plt.ylabel('Inertia')

plt.xticks(ks)

plt.style.use('ggplot')

plt.title('What is the Best Number for KMeans ?')

plt.show()



#We choose an "elbow" in the inertia plot Where inertia begins to decrease more slowly

#From the curve the best number of KMeans =  3 Clusters

#Second: The best number of KMeans =  3 Clusters

kc = KMeans(n_clusters=3)

kc.fit(samples)

labels = kc.predict(samples)

print(kc.inertia_)



#Thrid: Evaluating a clustering by Cross Tabulation 

df = pd.DataFrame({'labels': labels, 'species': iris.Species})

ct = pd.crosstab(df['labels'],df['species'])



print(ct)