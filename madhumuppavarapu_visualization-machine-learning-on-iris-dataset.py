from IPython.display import Image
Image('../input/iris-measurement/iris_measurements.png')
# importing required modules
import numpy as np
import pandas as pd
# importing visualization maodules
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from mpl_toolkits import mplot3d
from sklearn.tree import plot_tree
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,KFold,StratifiedKFold
# importing MachineLearning Algorithms
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.linear_model import LogisticRegression,PassiveAggressiveClassifier
from sklearn.svm import SVC,SVR
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
# importing ensembling algorithms
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier,GradientBoostingClassifier,BaggingClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor,RadiusNeighborsClassifier
from sklearn.decomposition import PCA,TruncatedSVD
import warnings
warnings.filterwarnings("ignore")
# overview of the data
### Reading the iris dataset
data=pd.read_csv('../input/iris/Iris.csv')
data.head(3)
# checking whether our dataset has any null values
print(data.isnull().sum())
sns.pairplot(data.drop(['Id'],axis="columns"),hue='Species')
plt.show()
fig=plt.figure(figsize=(20,20))
data.drop(['Id'],axis="columns").plot(kind="kde",subplots=True,ax=plt.gca())
plt.show()
fig=plt.figure(figsize=(15,15))
data.drop(['Id'],axis="columns").plot(kind="hist",subplots=True,ax=plt.gca())
plt.title(' histogram plotting of all the data')
plt.show()
fig=plt.figure(figsize=(10,10))
ax=plt.gca()
data.drop(['Id'],axis="columns").hist(edgecolor='black', linewidth=1.2,ax=ax)
plt.title('histogram plotting of numerical data in iris dataset')
plt.show()
def plot_scatter(feature1,feature2,title):
    fig=px.scatter(data,x=feature1,y=feature2,color='Species',title=title,hover_data=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm',
       'Species'],template="plotly_dark")
    fig.show()
plot_scatter(feature1='SepalLengthCm',feature2='SepalWidthCm',title='SepalLengthCm vs SepalWidthCm')
plot_scatter(feature1='PetalLengthCm',feature2='PetalWidthCm',title='PetalLengthCm vs PetalWidthCm')
plot_scatter(feature1='SepalLengthCm',feature2='PetalLengthCm',title='SepalLengthCm vs PetalLengthCm')
plot_scatter(feature1='SepalWidthCm',feature2='PetalWidthCm',title='SepalWidthCm vs PetalWidthCm')
def plot_bar(feature1,feature2,title):
    fig=px.bar(data,x=feature1,y=feature2,color='Species',title=title,hover_data=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm',
       'Species'],template="plotly_dark")
    fig.show()
plot_bar(feature1='SepalLengthCm',feature2='PetalWidthCm',title='SepalLengthCm vs PetalWidthCm')
plot_bar(feature1='SepalWidthCm',feature2='PetalLengthCm',title='SepalWidthCm vs PetalLengthCm')
plot_bar(feature1='PetalLengthCm',feature2='SepalWidthCm',title='PetalLengthCm vs SepalWidthCm')
plot_bar(feature1='PetalWidthCm',feature2='SepalLengthCm',title='PetalWidthCm vs SepalLengthCm')
def pie_chart_rep(values,names,title):
    fig = px.pie(data, values=values, names=names, title=title,template="plotly_dark")
    fig.update_layout()
    fig.show()
pie_chart_rep(values='SepalLengthCm',names='Species',title='pie representation of SepalLengthCm and Species')
pie_chart_rep(values='SepalWidthCm',names='Species',title='pie representation of SepalWidthCm and Species')
pie_chart_rep(values='PetalLengthCm',names='Species',title='pie representation of PetalLengthCm and Species')
pie_chart_rep(values='PetalWidthCm',names='Species',title='pie representation of PetalWidthCm and Species')
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.Species, y=data.SepalLengthCm,mode='markers', name='SepalLengthCm'))
fig.add_trace(go.Scatter(x=data.Species, y=data.SepalWidthCm,mode='markers', name='SepalWidthCm'))
fig.add_trace(go.Scatter(x=data.Species, y=data.PetalLengthCm,mode='markers', name='PetalLengthCm'))
fig.add_trace(go.Scatter(x=data.Species, y=data.PetalWidthCm,mode='markers', name='PetalWidthCm'))
fig.update_layout(template="plotly_dark",title='specifying the species based on SepalLengthCm')
fig.show()               
fig=plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.swarmplot(x='Species',y='PetalLengthCm',data=data)
plt.legend()
plt.subplot(2,2,2)
sns.swarmplot(x='Species',y='PetalWidthCm',data=data)
plt.legend()
plt.subplot(2,2,3)
sns.swarmplot(x='Species',y='SepalLengthCm',data=data)
plt.legend()
plt.subplot(2,2,4)
sns.swarmplot(x='Species',y='SepalWidthCm',data=data)
plt.legend()
plt.show()
# we convert the categorial data into numerical data using LabelEncoder()
le=LabelEncoder()
data['SpeciesCategory']=le.fit_transform(data['Species'])
label={0:'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginica'}
# we choose feature_names and target values from the data and split the data into train and test data
X=data.drop(['Species','SpeciesCategory'],axis='columns')
y=data['SpeciesCategory']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.3)
#Classification using KNeighborsClassifier and we will take nearest neighbors as 5
knn_clf=KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train,y_train)
print("the predicted values of X_test are {}".format(knn_clf.predict(X_test)))
print("the accuracy score of trained data for KNN classifier is {}".format(knn_clf.score(X_train,y_train)))
print("the accuracy score of test and predicted data for KNN classifier is {}".format(knn_clf.score(X_test,knn_clf.predict(X_test))))
# plotting KNN for iris dataset
fig=px.scatter(data,x='PetalLengthCm',y='PetalWidthCm',color='Species',title='PetalLengthCm vs PetalWidthCm in KNN Classifier',size='PetalWidthCm',hover_data=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','Species'],template="plotly_dark")
fig.show()
# plotting accuracy for a range of numbers 
accuracy=[]
n_neighbors_range=range(1,10)
for i in n_neighbors_range:
    knn_clf=KNeighborsClassifier(n_neighbors=i)
    knn_clf.fit(X_train,y_train)
    accuracy.append(knn_clf.score(X_train,y_train))
px.line(n_neighbors_range,accuracy,title='n_neighbors_range vs accuracy',template='plotly_dark')
radius_clf=RadiusNeighborsClassifier(radius=10)
radius_clf.fit(X_train,y_train)
print("the predicted values of X_test are {}".format(radius_clf.predict(X_test)))
print("the accuracy score of trained data for radius neighbors classifier is {}".format(radius_clf.score(X_train,y_train)))
print("the accuracy score of test and predicted data for radius neighbors  classifier is {}".format(radius_clf.score(X_test,radius_clf.predict(X_test))))
accuracy=[]
radius_range=range(1,20)
for i in radius_range:
    radius_clf=RadiusNeighborsClassifier(radius=i)
    radius_clf.fit(X_train,y_train)
    accuracy.append(radius_clf.score(X_train,y_train))
px.line(radius_range,accuracy,title='radius vs accuracy',template='plotly_dark')
svm_clf=SVC()
svm_clf.fit(X_train,y_train)
print("the predicted values of X_test are {}".format(svm_clf.predict(X_test)))
print("the accuracy score of trained data for SVM is {}".format(svm_clf.score(X_train,y_train)))
print("the accuracy score of test and predicted data for SVM is {}".format(svm_clf.score(X_test,svm_clf.predict(X_test))))
# plotting accuracy for a range of numbers 
accuracy=[]
c=range(1,10)
for i in c:
    svm_clf=KNeighborsClassifier(n_neighbors=i)
    svm_clf.fit(X_train,y_train)
    accuracy.append(svm_clf.score(X_train,y_train))
px.line(n_neighbors_range,accuracy,title='Regularization factor(C) vs accuracy',template='plotly_dark')
mnb=MultinomialNB()
bnb=BernoulliNB()
gnb=GaussianNB()
mnb.fit(X_train,y_train)
bnb.fit(X_train,y_train)
gnb.fit(X_train,y_train)
print("the accuracy score of trained data for MultinomialNB is {}".format(mnb.score(X_train,y_train)))
print("the accuracy score of test and predicted data for MultinomialNB is {}".format(mnb.score(X_test,mnb.predict(X_test))))
print("the accuracy score of trained data for BernoulliNB is {}".format(bnb.score(X_train,y_train)))
print("the accuracy score of test and predicted data for BernoulliNB is {}".format(bnb.score(X_test,bnb.predict(X_test))))
print("the accuracy score of trained data for GaussianNB is {}".format(gnb.score(X_train,y_train)))
print("the accuracy score of test and predicted data for GaussianNB is {}".format(gnb.score(X_test,gnb.predict(X_test))))
dec_tree=DecisionTreeClassifier(max_depth=10)
dec_tree.fit(X_train,y_train)
print("predicted data is:",svm_clf.predict(X_test))
print("the accuracy score of trained data for DecisionTreeClassifier is {}".format(svm_clf.score(X_train,y_train)))
print("the accuracy score of test and predicted data for DecisionTreeClassifier is {}".format(svm_clf.score(X_test,svm_clf.predict(X_test))))
from sklearn import tree
fig=plt.figure(figsize=(10,10))
_=tree.plot_tree(dec_tree,feature_names=data.PetalLengthCm,class_names=data.Species,filled=True)
plt.title('DecisionTreeClassifier for PetalLengthCm vs Species')
fig.show()
# plotting accuracy for a range of numbers 
accuracy=[]
max_depth=range(1,10)
for i in max_depth:
    dec_tree=KNeighborsClassifier(n_neighbors=i)
    dec_tree.fit(X_train,y_train)
    accuracy.append(dec_tree.score(X_test,svm_clf.predict(X_test)))
px.line(max_depth,accuracy,title='max_depth of the tree vs accuracy',template='plotly_dark')
forest_clf=RandomForestClassifier(n_estimators=5)
forest_clf.fit(X_train,y_train)
print("the predicted values are:",forest_clf.predict(X_test))
print("the accuracy score of trained data for DecisionTreeClassifier is",forest_clf.score(X_train,y_train))
print("the accuracy score of tested and predicted data for DecisionTreeClassifier is",forest_clf.score(X_test,forest_clf.predict(X_test)))
fig=plt.figure(figsize=(10,10))
_=tree.plot_tree(forest_clf.estimators_[3],feature_names=data.PetalLengthCm,class_names=data.Species)
plt.title("random forest tree PetalLengthCm vs PetalLengthCm")
plt.show()
# plotting accuracy for a range of numbers 
accuracy=[]
n_estimators=range(1,10)
for i in n_estimators:
    forest_clf=RandomForestClassifier(n_estimators=5)
    forest_clf.fit(X_train,y_train)
    accuracy.append(forest_clf.score(X_test,forest_clf.predict(X_test)))
px.line(n_estimators,accuracy,title='n_estimators of the random forest classifier vs accuracy',template='plotly_dark')
log_reg=LogisticRegression(C=100)
log_reg.fit(X_train,y_train)
print("the predicted values are:",log_reg.predict(X_test))
print("the accuracy score of trained data for DecisionTreeClassifier is",log_reg.score(X_train,y_train))
print("the accuracy score of tested and predicted data for DecisionTreeClassifier is",log_reg.score(X_test,log_reg.predict(X_test)))
C_range=range(1,200)
accuracy=[]
for n in C_range:
    log_reg=LogisticRegression(C=n)
    log_reg.fit(X_train,y_train)
    accuracy.append(log_reg.score(X_test,log_reg.predict(X_test)))
px.line(C_range,accuracy,title='Regularization factor(C) of the tree vs accuracy',template='plotly_dark')