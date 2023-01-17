import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



%matplotlib inline



import seaborn as sns

plt.style.use("fivethirtyeight")



import pandas_profiling as pp

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import warnings

warnings.filterwarnings("ignore")
#Allow for full tables to be shown

pd.options.display.max_columns = None

pd.options.display.max_rows = None
iris=pd.read_csv("/kaggle/input/iris/Iris.csv")

#iris.head(5)

iris.sample(5)
# drop the "Id" column from the iris data

iris=iris.drop("Id",axis=1)
##Split the Species names by the '-', and use the second one. i.e remove iris from the Species names.

iris['Species'] = iris['Species'].str.split('-', expand=True)[1]

iris.head()
# iris data shape

iris.shape
# data type of columns of iris dataframe.

iris.dtypes
iris.columns  ## iris.keys()
iris.info()
import pandas_profiling as pp

df=iris

report=pp.ProfileReport(df,title = "Pandas Profile Report")



#report.to_widgets()

report.to_notebook_iframe()
#report.to_file("your_report.html")
iris.isnull().values.any()
iris.duplicated().value_counts()
iris[iris.duplicated(keep="first")]
iris.drop_duplicates(keep="first",inplace=True)
iris.describe(include="all")
iris.corr()
fig=plt.figure()

sns.heatmap(iris.corr(),annot= True)

plt.show()
iris["Species"].value_counts()


fig=plt.figure()

iris["Species"].value_counts().plot.bar()

plt.show()
fig=plt.figure()

sns.countplot("Species",data=iris)

plt.show()
#!pip install squarify

import squarify

plt.figure(figsize=(8,8))

squarify.plot(sizes=iris.Species.value_counts(), label=iris['Species'], alpha=.5 ,color=['r','g','b'])

plt.axis('off')

plt.show()
iris.groupby("Species").size().plot.bar()

plt.show()
iris.Species.value_counts().plot.pie(explode=(0.1,0.1,0.1),autopct='%1.1f%%',shadow=True,figsize=(8,8))

plt.tight_layout()
fig,ax=plt.subplots(2,2,figsize=(10,10))

iris.SepalLengthCm.plot.line(ax=ax[0][0])

ax[0][0].set_title("Sepal Length")

iris.SepalWidthCm.plot.line(ax=ax[0][1])

ax[0][1].set_title("Sepal Width")

iris.PetalLengthCm.plot.line(ax=ax[1][0])

ax[1][0].set_title("Petal Length")

iris.PetalWidthCm.plot.line(ax=ax[1][1])

ax[1][1].set_title("Petal Width")

plt.show()
iris.hist(edgecolor="black",figsize=(10,10))

plt.show()
iris.boxplot(figsize=(10,10))

plt.show()
iris.plot(kind='box', subplots=True, layout=(2,2),sharex=False,  sharey=False,figsize=(8,8))

plt.show()
iris.plot(figsize=(12,12))

plt.show()
iris.boxplot(by="Species",figsize=(12,12))

plt.show()
sns.pairplot(iris,hue="Species")

plt.show()
sns.scatterplot(x="SepalLengthCm",y="SepalWidthCm",data=iris,hue="Species")

plt.show()
from IPython.display import Image

Image("http://scikit-learn.org/dev/_static/ml_map.png", width=800)


from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score,confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.cluster import KMeans

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
X=iris.iloc[:,:-1]

Y=iris.iloc[:,-1:]

#scaling of the feature

ohe=OneHotEncoder()

LE=LabelEncoder()

Y=LE.fit_transform(Y)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=2)

print("Train data size",x_train.shape,y_train.shape)

print("Test data size",x_test.shape,y_test.shape)
model=LogisticRegression()

model.fit(x_train,y_train)

pred=model.predict(x_test)







print("Confusion Matrix:\n",confusion_matrix(pred,y_test))

print("=================================================================")

print("Classification Report:\n",classification_report(pred,y_test))

print("=================================================================")

# Accuracy Score

print("Accuracy Score:\n",accuracy_score(pred,y_test))
model1=SVC()

model1.fit(x_train,y_train)

pred1=model1.predict(x_test)





print("Confusion Matrix:\n",confusion_matrix(pred,y_test))

print("=================================================================")

print("Classification Report:\n",classification_report(pred,y_test))

print("=================================================================")

#Accuracy Score

print("Accuracy Score:\n",accuracy_score(pred,y_test))

# Decision Tree Classifier

model_dt=DecisionTreeClassifier()

model_dt.fit(x_train,y_train)

pred_dt=model_dt.predict(x_test)







print("Confusion Matrix:\n",confusion_matrix(pred_dt,y_test))

print("=================================================================")

print("Classification Report:\n",classification_report(pred_dt,y_test))

print("=================================================================")



# Accuracy Score

print("Accuracy Score:\n",accuracy_score(pred_dt,y_test))
# Random Forest Classifier

model_rf=RandomForestClassifier(n_jobs=3,max_depth=3)

model_rf.fit(x_train,y_train)

pred_rf=model_rf.predict(x_test)





# Summary of the predictions made by the classifier

print("Confusion Matrix:\n",confusion_matrix(pred_rf,y_test))

print("=================================================================")

print("Classification Report:\n",classification_report(pred_rf,y_test))

print("=================================================================")

#Accuracy Score

print("accuracy Score:\n",accuracy_score(pred_rf,y_test))
# K Nearest Neighbors 

model_knn=KNeighborsClassifier(n_neighbors=3)

model_knn.fit(x_train,y_train)

pred_knn=model_knn.predict(x_test)



# Summary of the predictions made by the classifier

print("Confusion Matrix:\n",confusion_matrix(pred_knn,y_test))

print("=================================================================")

print("Classification Report:\n",classification_report(pred,y_test))

print("=================================================================")

# Accuracy Score

print("Accuracy Score:\n",accuracy_score(pred_knn,y_test))

param={'C':[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5],

      'kernel': ["linear","rbf"],

      "gamma":[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]}

               

grid_svc=GridSearchCV(model1,param_grid=param,scoring="accuracy",cv=10)

grid_svc.fit(x_train,y_train)
grid_svc.best_params_
gridsearch_svc=SVC(C=0.8,gamma=0.1,kernel='linear')

gridsearch_svc.fit(x_train,y_train)

pred_grid=gridsearch_svc.predict(x_test)



print("Confusion Matrix:\n",confusion_matrix(pred_grid,y_test))

print("=================================================================")

print("Classification Report:\n",classification_report(pred_grid,y_test))

print("=================================================================")



# Accuracy Score

print("Accuracy Score:\n",accuracy_score(pred_grid,y_test))
models=[model,model1,gridsearch_svc,model_dt,model_rf,model_knn]

accuracy_scores=[]

for i in models:

    pred=i.predict(x_test)

    accuracy=accuracy_score(pred,y_test)

    accuracy_scores.append(accuracy)

print(accuracy_scores)    

plt.bar(['LogReg','SVM','GridSVC','DT','RF','KNN'],accuracy_scores)

plt.ylim(0.90,1.01)

plt.title("Accuracy comparision for various models",fontsize=15,color='r')

plt.xlabel("Models",fontsize=18,color='g')

plt.ylabel("Accuracy Score",fontsize=18,color='g')

plt.show()

    
#converting categorical data  into int data type using labelEncoder for Linear reagration.



x = iris.iloc[:,:-1].values    #   X -> Feature Variables

y = iris.iloc[:,-1].values #   y ->  Target



from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

y = le.fit_transform(y)



print(y)  # this is y categotical to numerical
from sklearn.linear_model import LinearRegression

from sklearn import metrics

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)



model_LR = LinearRegression()

model_LR.fit(X_train, y_train)

y_pred = model_LR.predict(X_test)





print('y-intercept             :' , model_LR.intercept_)

print('beta coefficients       :' , model_LR.coef_)

print('Mean Abs Error MAE      :' ,metrics.mean_absolute_error(y_test,y_pred))

print('Mean Sqrt Error MSE     :' ,metrics.mean_squared_error(y_test,y_pred))

print('Root Mean Sqrt Error RMSE:' ,np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

print('r2 value                :' ,metrics.r2_score(y_test,y_pred))

# Naive Bayes

from sklearn.naive_bayes import GaussianNB

Model = GaussianNB()

Model.fit(X_train, y_train)



y_pred = Model.predict(X_test)



# Summary of the predictions made by the classifier

print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))

print("=================================================================")

print("Classification Report:\n",classification_report(y_test, y_pred))

print("=================================================================")

# Accuracy score

print("Accuracy Score:\n",accuracy_score(y_pred,y_test))
from sklearn.svm import NuSVC



Model = NuSVC()

Model.fit(X_train, y_train)



y_pred = Model.predict(X_test)



# Summary of the predictions made by the classifier

print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))

print("=================================================================")

print("Classification Report:\n",classification_report(y_test, y_pred))

print("=================================================================")



# Accuracy score

print("Accuracy Score:\n",accuracy_score(y_pred,y_test))
# Linear Support Vector Classification

from sklearn.svm import LinearSVC



Model = LinearSVC()

Model.fit(X_train, y_train)

y_pred = Model.predict(X_test)



# Summary of the predictions made by the classifier

print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))

print("=================================================================")

print("Classification Report:\n",classification_report(y_test, y_pred))

print("=================================================================")



# Accuracy score

print("Accuracy Score:\n",accuracy_score(y_pred,y_test))
from sklearn.neighbors import  RadiusNeighborsClassifier

Model=RadiusNeighborsClassifier(radius=2.0)

Model.fit(X_train,y_train)

y_pred=Model.predict(X_test)



# Summary of the predictions made by the classifier

print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))

print("=================================================================")

print("Classification Report:\n",classification_report(y_test, y_pred))

print("=================================================================")



# Accuracy score

print("Accuracy Score:\n",accuracy_score(y_pred,y_test))
# BernoulliNB

from sklearn.naive_bayes import BernoulliNB

Model = BernoulliNB()

Model.fit(X_train, y_train)



y_pred = Model.predict(X_test)



# Summary of the predictions made by the classifier

print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))

print("=================================================================")

print("Classification Report:\n",classification_report(y_test, y_pred))

print("=================================================================")



# Accuracy score

print("Accuracy Score:\n",accuracy_score(y_pred,y_test))
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

Model=LinearDiscriminantAnalysis()

Model.fit(X_train,y_train)

y_pred=Model.predict(X_test)



# Summary of the predictions made by the classifier

print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))

print("=================================================================")

print("Classification Report:\n",classification_report(y_test, y_pred))

print("=================================================================")



# Accuracy score

print("Accuracy Score:\n",accuracy_score(y_pred,y_test))
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

Model=QuadraticDiscriminantAnalysis()

Model.fit(X_train,y_train)

y_pred=Model.predict(X_test)



# Summary of the predictions made by the classifier

print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))

print("=================================================================")

print("Classification Report:\n",classification_report(y_test, y_pred))

print("=================================================================")



# Accuracy score

print("Accuracy Score:\n",accuracy_score(y_pred,y_test))
from sklearn.linear_model import PassiveAggressiveClassifier

Model = PassiveAggressiveClassifier()

Model.fit(X_train, y_train)



y_pred = Model.predict(X_test)



# Summary of the predictions made by the classifier

print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))

print("=================================================================")

print("Classification Report:\n",classification_report(y_test, y_pred))

print("=================================================================")



# Accuracy score

print("Accuracy Score:\n",accuracy_score(y_pred,y_test))
# BernoulliNB

from sklearn.naive_bayes import BernoulliNB

Model = BernoulliNB()

Model.fit(X_train, y_train)



y_pred = Model.predict(X_test)



# Summary of the predictions made by the classifier

print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))

print("=================================================================")

print("Classification Report:\n",classification_report(y_test, y_pred))

print("=================================================================")



# Accuracy score

print("Accuracy Score:\n",accuracy_score(y_pred,y_test))
# ExtraTreeClassifier

from sklearn.tree import ExtraTreeClassifier

Model = ExtraTreeClassifier()

Model.fit(X_train, y_train)

y_pred = Model.predict(X_test)



# Summary of the predictions made by the classifier

print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))

print("=================================================================")

print("Classification Report:\n",classification_report(y_test, y_pred))

print("=================================================================")



# Accuracy score

print("Accuracy Score:\n",accuracy_score(y_pred,y_test))
from sklearn.ensemble import BaggingClassifier

Model=BaggingClassifier()

Model.fit(X_train,y_train)

y_pred=Model.predict(X_test)



# Summary of the predictions made by the classifier

print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))

print("=================================================================")

print("Classification Report:\n",classification_report(y_test, y_pred))

print("=================================================================")



# Accuracy score

print("Accuracy Score:\n",accuracy_score(y_pred,y_test))
from sklearn.ensemble import AdaBoostClassifier

Model=AdaBoostClassifier()

Model.fit(X_train,y_train)

y_pred=Model.predict(X_test)



# Summary of the predictions made by the classifier

print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))

print("=================================================================")

print("Classification Report:\n",classification_report(y_test, y_pred))

print("=================================================================")



# Accuracy score

print("Accuracy Score:\n",accuracy_score(y_pred,y_test))
from sklearn.ensemble import GradientBoostingClassifier

Model=GradientBoostingClassifier()

Model.fit(X_train,y_train)

y_pred=Model.predict(X_test)



# Summary of the predictions made by the classifier

print("Confusion Matrix:\n",confusion_matrix(y_test, y_pred))

print("=================================================================")

print("Classification Report:\n",classification_report(y_test, y_pred))

print("=================================================================")



# Accuracy score

print("Accuracy Score:\n",accuracy_score(y_pred,y_test))


#Finding the optimum number of clusters for k-means classification

from sklearn.cluster import KMeans

wcss = []



for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 100, n_init = 5, random_state = 42)

    kmeans.fit(X_train,y_train)

    wcss.append(kmeans.inertia_)

    

#Plotting the results onto a line graph, allowing us to observe 'The elbow'

plt.plot(range(1, 11), wcss)

plt.title('The elbow method')

plt.xlabel('Number of clusters')

plt.ylabel('WCSS') # within cluster sum of squares

plt.show()
#Applying kmeans to the dataset / Creating the kmeans classifier

kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 100, n_init = 5, random_state = 42)

y_preds = kmeans.fit_predict(X)







# Summary of the predictions made by the classifier

print("Confusion Matrix:\n",confusion_matrix(y, y_preds))

print("=================================================================")

print("Classification Report:\n",classification_report(y, y_preds))

print("=================================================================")



# Accuracy score

print("Accuracy Score:\n",accuracy_score(y_preds,y))



#Visualising the clusters

plt.scatter(x[y_preds == 0, 0], x[y_preds == 0, 1], c = 'g', label = 'Setosa')

plt.scatter(x[y_preds == 1, 0], x[y_preds == 1, 1], c = 'b', label = 'Versicolour')

plt.scatter(x[y_preds == 2, 0], x[y_preds == 2, 1], c = 'y', label = 'Virginica')



#Plotting the centroids of the clusters

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'r', label = 'Centroids',marker='*')



plt.legend()
from scipy.cluster.hierarchy import linkage,dendrogram



plt.figure(figsize=(18,8))

merg=linkage(X,method="ward")

dendrogram(merg,leaf_rotation=90)

plt.xlabel("data points")

plt.ylabel("euclidian distance")

plt.show() 
from sklearn.cluster import AgglomerativeClustering



ac=AgglomerativeClustering(n_clusters=3,affinity="euclidean",linkage="ward")

y_prediction=ac.fit_predict(X) 



plt.scatter(X.loc[:,"PetalLengthCm"],X.loc[:,"PetalWidthCm"],c=y_prediction,cmap="rainbow") # Görselleştirme

plt.xlabel("PetalLength")

plt.ylabel("PetalWidth")

plt.show()