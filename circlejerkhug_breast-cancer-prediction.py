import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, KFold, GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm

from sklearn import metrics
data=pd.read_csv("../input/data.csv",header=0)

print(data.head(2))
data.info()
data.drop("Unnamed: 32",axis=1,inplace=True)
data.drop("id",axis=1,inplace=True)
features_mean= list(data.columns[1:11])

features_se= list(data.columns[11:20])

features_worst=list(data.columns[21:31])
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})

data.describe()
sns.countplot(data['diagnosis'],label="Count")
corr=data[features_mean].corr()

plt.figure(figsize=(14,14))

sns.heatmap(corr,cbar=True,square=True,annot=True,fmt='.2f',annot_kws={'size': 15},xticklabels=features_mean,yticklabels=features_mean,cmap='coolwarm')
prediction_var=['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean']

train,test=train_test_split(data,test_size=0.3)
#Splitting the data into train and test.

train_X=train[prediction_var]

train_y=train.diagnosis

test_X=test[prediction_var]

test_y=test.diagnosis
model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)
prediction=model.predict(test_X)
metrics.accuracy_score(prediction,test_y)
model=svm.SVC()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

metrics.accuracy_score(prediction,test_y)

prediction_var=features_mean #with all features
train_X=train[prediction_var]

train_y=train.diagnosis

test_X=test[prediction_var]

test_y=test.diagnosis
model=RandomForestClassifier(n_estimators=100)
model.fit(train_X,train_y)

prediction=model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
featimp=pd.Series(model.feature_importances_,index=prediction_var).sort_values(ascending=False)

print(featimp)
model = svm.SVC()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
prediction_var=['concave points_mean','perimeter_mean' , 'concavity_mean' , 'radius_mean','area_mean']      
train_X= train[prediction_var]

train_y= train.diagnosis

test_X = test[prediction_var]

test_y = test.diagnosis
model=RandomForestClassifier(n_estimators=100)

model.fit(train_X,train_y)

prediction = model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
model = svm.SVC()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
prediction_var=features_worst
train_X= train[prediction_var]

train_y= train.diagnosis

test_X = test[prediction_var]

test_y = test.diagnosis
model = svm.SVC()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
model=RandomForestClassifier(n_estimators=100)

model.fit(train_X,train_y)

prediction = model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
featimp = pd.Series(model.feature_importances_, index=prediction_var).sort_values(ascending=False)

print(featimp)
prediction_var = ['concave points_worst','radius_worst','area_worst','perimeter_worst','concavity_worst'] 
train_X= train[prediction_var]

train_y= train.diagnosis

test_X = test[prediction_var]

test_y = test.diagnosis
model=RandomForestClassifier(n_estimators=100)

model.fit(train_X,train_y)

prediction = model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
#check for SVM

model = svm.SVC()

model.fit(train_X,train_y)

prediction=model.predict(test_X)

metrics.accuracy_score(prediction,test_y)
color_function = {0: "blue", 1: "red"} # Here Red color will be 1 which means M and blue foo 0 means B

colors = data["diagnosis"].map(lambda x: color_function.get(x))# mapping the color fuction with diagnosis column

pd.scatter_matrix(data[features_mean], c=colors, alpha = 0.5, figsize = (18, 18)); # plotting scatter plot matrix
features_mean
predict_var=['radius_mean','perimeter_mean','compactness_mean','area_mean','concave points_mean']
#Checking the accuracy of the model

def model(model,data,prediction,outcome):

    kf=KFold(n_splits=10)
#Cross validation using different model

def classification_model(model,data,prediction_input,output):

    model.fit(data[prediction_input],data[output]) 

    predictions = model.predict(data[prediction_input])

    accuracy = metrics.accuracy_score(predictions,data[output])

    print("Accuracy : %s" % "{0:.3%}".format(accuracy))

    kf = KFold(n_splits=10,random_state=42,shuffle=False)

    error = []

    for train, test in kf.split(data):

        train_X = (data[prediction_input].iloc[train,:])

        train_y = data[output].iloc[train]

        model.fit(train_X, train_y)

        

        test_X=data[prediction_input].iloc[test,:]

        test_y=data[output].iloc[test]

        error.append(model.score(test_X,test_y))

        print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
model = DecisionTreeClassifier()

predict_var=['radius_mean','perimeter_mean','compactness_mean','area_mean','concave points_mean']

outcome_var= "diagnosis"

classification_model(model,data,prediction_var,outcome_var)

model = svm.SVC()

classification_model(model,data,prediction_var,outcome_var)
model = KNeighborsClassifier()

classification_model(model,data,prediction_var,outcome_var)
model = RandomForestClassifier(n_estimators=100)

classification_model(model,data,prediction_var,outcome_var)
model=LogisticRegression()

classification_model(model,data,prediction_var,outcome_var)
data_X=data[prediction_var]

data_y=data['diagnosis']

#GridSearchCV function

def Classification_model_gridsearchCV(model,param_grid,data_X,data_y):

    clf = GridSearchCV(model,param_grid,cv=10,scoring="accuracy")

    clf.fit(train_X,train_y)

    print("The efficient parameter to be used is")

    print(clf.best_params_)

    print("the efficient estimator is ")

    print(clf.best_estimator_)

    print("The efficient score is ")

    print(clf.best_score_)
#Here, we have to use the parameters which we used in Decision Tree Classifier

param_grid = {'max_features': ['auto', 'sqrt', 'log2'],

              'min_samples_split': [2,3,4,5,6,7,8,9,10], 

              'min_samples_leaf':[2,3,4,5,6,7,8,9,10] }

model= DecisionTreeClassifier()

Classification_model_gridsearchCV(model,param_grid,data_X,data_y)
#Here, we wil use the parameters use in KNN

model = KNeighborsClassifier()

k_range = list(range(1, 30))

leaf_size = list(range(1,30))

weight_options = ['uniform', 'distance']

param_grid = {'n_neighbors': k_range, 'leaf_size': leaf_size, 'weights': weight_options}

Classification_model_gridsearchCV(model,param_grid,data_X,data_y)
#Here, we try it with SVM 

model=svm.SVC()

param_grid = [

              {'C': [1, 10, 100, 1000], 

               'kernel': ['linear']

              },

              {'C': [1, 10, 100, 1000], 

               'gamma': [0.001, 0.0001], 

               'kernel': ['rbf']

              },

 ]

Classification_model_gridsearchCV(model,param_grid,data_X,data_y)