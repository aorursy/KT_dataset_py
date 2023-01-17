'''Multiclass classification based on hazelnuts variety "c_avellana, c_americana,

c_corutana and comparting with SVM, kNN, Decision tree, Naive Bayes classifiers'''



import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



#Loading data

hazel_df = pd.read_csv("../input/hazelnuts.txt",sep="\t",header=None)

hazel_df = hazel_df.transpose()

hazel_df.columns = ["sample_id","length","width","thickness","surface_area","mass","compactness",

                    "hardness","shell_top_radius","water_content","carbohydrate_content","variety"]

hazel_df.head()
#Feature selection

all_features = hazel_df.drop(["variety","sample_id"],axis=1) 

target_feature = hazel_df["variety"]

all_features.head()
#Dataset preprocessing

from sklearn import preprocessing

x = all_features.values.astype(float) #returns a numpy array of type float

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

scaled_features = pd.DataFrame(x_scaled)

scaled_features.head()
#Decision tree

from sklearn.model_selection import train_test_split #for split the data

from sklearn.metrics import accuracy_score  #for accuracy_score

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from sklearn.metrics import confusion_matrix #for confusion matrix

import seaborn as sns



X_train,X_test,y_train,y_test = train_test_split(scaled_features,target_feature,test_size=0.2,random_state=40)

X_train.shape,X_test.shape,y_train.shape,y_test.shape



from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

model= DecisionTreeClassifier(criterion='gini', 

                             min_samples_split=10,min_samples_leaf=1,

                             max_features='auto')

model.fit(X_train,y_train)

dt_pred=model.predict(X_test)

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

result_tree=cross_val_score(model,scaled_features,target_feature,cv=10,scoring='accuracy')

print('The overall score for Decision Tree classifier is:',round(result_tree.mean()*100,2))

y_pred = cross_val_predict(model,scaled_features,target_feature,cv=10)

sns.heatmap(confusion_matrix(dt_pred,y_test),annot=True,cmap='summer')

plt.title('KNN Confusion_matrix')

#Visualizing decision tree

from sklearn.tree import export_graphviz

import graphviz



dot_data = export_graphviz(

    model,

    out_file=None,

    feature_names=hazel_df.columns[1:-1],

    filled=True,

    rounded=True,

    special_characters=True)



graph = graphviz.Source(dot_data) 

graph

#KNN

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from sklearn.metrics import confusion_matrix #for confusion matrix

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 25)

model.fit(X_train,y_train)

dt_knn=model.predict(X_test)

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts= 

result_knn=cross_val_score(model,scaled_features,target_feature,cv=kfold,scoring='accuracy')

print('The overall score for K Nearest Neighbors Classifier is:',round(result_knn.mean()*100,2))

y_pred = cross_val_predict(model,scaled_features,target_feature,cv=10)

sns.heatmap(confusion_matrix(dt_knn,y_test),annot=True,cmap='summer')

plt.title('KNN Confusion_matrix')
#KNN fold accuracy visualizer

_result_knn=[r*100 for r in result_knn]

plt.plot(_result_knn)

plt.xlabel('Fold')

plt.ylabel('Accuracy')
#Naive bayes

from sklearn.naive_bayes import GaussianNB

model= GaussianNB()

model.fit(X_train,y_train)

gnb_pred=model.predict(X_test)

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

result_gnb=cross_val_score(model,scaled_features,target_feature,cv=10,scoring='accuracy')

print('The overall score for Gaussian Naive Bayes classifier is:',round(result_gnb.mean()*100,2))

y_pred = cross_val_predict(model,scaled_features,target_feature,cv=10)

sns.heatmap(confusion_matrix(gnb_pred,y_test),annot=True,cmap='summer')

plt.title('Naive Bayes Confusion_matrix')
#Naive bayes fold accuracy visualizer

_result_gnb=[r*100 for r in result_gnb]

plt.plot(_result_gnb)

plt.xlabel('Fold')

plt.ylabel('Accuracy')

plt.title('Accuracy')
#Naive bayes KFold tracking example

from sklearn.model_selection import KFold 

kf = KFold(n_splits=10, random_state=None) 



for train_index, test_index in kf.split(scaled_features):

    print("Train:", train_index, "Validation:",test_index)

    X_train, X_test = all_features.iloc[train_index], all_features.iloc[test_index] 

    y_train, y_test = target_feature.iloc[train_index], target_feature.iloc[test_index]

    model = GaussianNB()

    model.fit(X_train,y_train)

    result_gnb=cross_val_score(model,scaled_features,target_feature,cv=10,scoring='accuracy')

    print(result_gnb.mean())



#Linear SVM 

from sklearn.model_selection import KFold #for K-fold cross validation

from sklearn.model_selection import cross_val_score #score evaluation

from sklearn.model_selection import cross_val_predict #prediction

from sklearn.metrics import confusion_matrix #for confusion matrix

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.svm import SVC, LinearSVC

model = SVC(gamma='auto')

model.fit(X_train,y_train)

pred_svm = model.predict(X_test)

kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts

result_svm=cross_val_score(model,scaled_features,target_feature,cv=10,scoring='accuracy')

print('The overall score for Support Vector machine classifier is:',round(result_svm.mean()*100,2))

y_pred = cross_val_predict(model,scaled_features,target_feature,cv=10)

sns.heatmap(confusion_matrix(pred_svm,y_test),annot=True,cmap='summer')

plt.title('SVM Confusion_matrix')
#SVM fold accuracy visualizer

_result_svm=[r*100 for r in result_svm]

plt.plot(_result_svm)

plt.xlabel('Fold')

plt.ylabel('Accuracy')
#Comparing all classifiers

models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 

               'Naive Bayes', 'Decision Tree'],

    'Score': [result_svm.mean(), result_knn.mean(), 

              result_gnb.mean(), result_tree.mean()]})

models.sort_values(by='Score',ascending=False)