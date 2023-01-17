import numpy as np 

import pandas as pd  

from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression 

from sklearn.ensemble import RandomForestClassifier 

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import svm 
data = pd.read_csv("../input/breastcanser/UCIMLBreastCancer2.csv")
print(data.head(2))
data.info()
data.columns
#data['Classification'] = data['Classification'].map({'1':1,'2':2})
data_list = list(data.columns[0:9])

print(data_list)
data.describe()
#Simple Feature Scaling

#data["Age"] = data["Age"]/data["Age"].max()

#data["BMI"] = data["BMI"]/data["BMI"].max()

#data["Glucose"] = data["Glucose"]/data["Glucose"].max()

#data["Insulin"] = data["Insulin"]/data["Insulin"].max()

#data["HOMA"] = data["HOMA"]/data["HOMA"].max()

#data["Leptin"] = data["Leptin"]/data["Leptin"].max()

#data["Adiponectin"] = data["Adiponectin"]/data["Adiponectin"].max()

#data["Resistin"] = data["Resistin"]/data["Resistin"].max()

#data["MCP.1"] = data["MCP.1"]/data["MCP.1"].max()
#Min-Max

data["Age"] = (data["Age"]-data["Age"].min())/(data["Age"].max()-data["Age"].min())

data["BMI"] = (data["BMI"]-data["BMI"].min())/(data["BMI"].max()-data["BMI"].min())

data["Glucose"] = (data["Glucose"]-data["Glucose"].min())/(data["Glucose"].max()-data["Glucose"].min())

data["Insulin"] = (data["Insulin"]-data["Insulin"].min())/(data["Insulin"].max()-data["Insulin"].min())

data["HOMA"] = (data["HOMA"]-data["HOMA"].min())/(data["HOMA"].max()-data["HOMA"].min())

data["Leptin"] = (data["Leptin"]-data["Leptin"].min())/(data["Leptin"].max()-data["Leptin"].min())

data["Adiponectin"] = (data["Adiponectin"]-data["Adiponectin"].min())/(data["Adiponectin"].max()-data["Adiponectin"].min())

data["Resistin"] = (data["Resistin"]-data["Resistin"].min())/(data["Resistin"].max()-data["Resistin"].min())

data["MCP.1"] = (data["MCP.1"]-data["MCP.1"].min())/(data["MCP.1"].max()-data["MCP.1"].min())
#Z-Score

#data["Age"] = (data["Age"]-data["Age"].mean())/data["Age"].std()

#data["BMI"] = (data["BMI"]-data["BMI"].mean())/data["BMI"].std()

#data["Glucose"] = (data["Glucose"]-data["Glucose"].mean())/data["Glucose"].std()

#data["Insulin"] = (data["Insulin"]-data["Insulin"].mean())/data["Insulin"].std()

#data["HOMA"] = (data["HOMA"]-data["HOMA"].mean())/data["HOMA"].std()

#data["Leptin"] = (data["Leptin"]-data["Leptin"].mean())/data["Leptin"].std()

#data["Adiponectin"] = (data["Adiponectin"]-data["Adiponectin"].mean())/data["Adiponectin"].std()

#data["Resistin"] = (data["Resistin"]-data["Resistin"].mean())/data["Resistin"].std()

#data["MCP.1"] = (data["MCP.1"]-data["MCP.1"].mean())/data["MCP.1"].std()
data.describe()
data.head()
prediction_var = ['Age','BMI','Glucose','Insulin','HOMA','Leptin','Adiponectin','Resistin','MCP.1'] 
train, test = train_test_split(data, test_size = 0.3) 

print(train.shape)

print(test.shape)
train_X = train[prediction_var]

train_y = train.Classification 



test_X  = test[prediction_var]  

test_y  = test.Classification   
#model = RandomForestClassifier(n_estimators=100) 
#model.fit(train_X,train_y)
#prediction = model.predict(test_X) 
#metrics.accuracy_score(prediction,test_y) 
#model = svm.SVC()

#model.fit(train_X,train_y)

#prediction=model.predict(test_X)

#metrics.accuracy_score(prediction,test_y)
#color_function = {1: "blue", 2: "red"} # Here Red color will be 1 which means M and blue foo 0 means B

#colors = data["Classification"].map(lambda x: color_function.get(x))# mapping the color fuction with diagnosis column

#pd.plotting.scatter_matrix(data[data_list], c=colors, alpha = 0.5, figsize = (15, 15)); # plotting scatter plot matrix
#prediction_var = ['Insulin', 'Glucose', 'HOMA', 'Leptin']
#def model(model,data,prediction,outcome):

#    kf = KFold(data.shape[0], n_folds=10)
#def classification_model(model,data,prediction_input,output):

    

 #   model.fit(data[prediction_input],data[output])

    

 #   predictions = model.predict(data[prediction_input])

    

 #   accuracy = metrics.accuracy_score(predictions,data[output])

    

 #   print("Accuracy : %s" % "{0:.3%}".format(accuracy))

 #   

  #  kf = KFold(data.shape[0], n_folds=5)

    

 #   error = []

    

 #   for train, test in kf:

        

   #     train_X = (data[prediction_input]).iloc[train,:]

 #       train_y = data[output].iloc[train]

  #      model.fit(train_X, train_y)

        

   #     test_X = data[prediction_input].iloc[test,:]

  #      test_y = data[output].iloc[test]

    #    error.append(model.score(test_X,test_y))

   #     

    #    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
#model = DecisionTreeClassifier()

#prediction_var = ['Age','BMI','Glucose','Insulin','HOMA','Leptin','Adiponectin','Resistin','MCP.1']

#outcome_var = "Classification"

#classification_model(model,data,prediction_var,outcome_var)
data
data_list
prediction_var = ['Age','BMI','Glucose','Insulin','HOMA','Leptin','Adiponectin','Resistin','MCP.1'] #Bunlar tahmin için kullanılacak veriler.
data_X = data[prediction_var]

data_y = data["Classification"]
data[prediction_var]

data["Classification"]
def Classification_model_gridsearchCV(model,param_grid,data_X,data_y):

    clf = GridSearchCV(model,param_grid,cv=10,scoring="accuracy")



    clf.fit(train_X,train_y)

    print("Geliştirme Setinde Bulunan En İyi Parametre :")

    print(clf.best_params_)

    print("En iyi tahmin edici ")

    print(clf.best_estimator_)

    print("En iyi skor ")

    print(clf.best_score_) 
# Logistic Regression

param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},

model = LogisticRegression(solver='liblinear')



Classification_model_gridsearchCV(model,param_grid,data_X,data_y)
# Random Forest



param_grid = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



model = RandomForestClassifier()





Classification_model_gridsearchCV(model,param_grid,data_X,data_y)
# Decison Tree



param_grid = {'max_features': ['auto', 'sqrt', 'log2'],

              'min_samples_split': [2,3,4,5,6,7,8,9,10], 

              'min_samples_leaf':[2,3,4,5,6,7,8,9,10] }



model = DecisionTreeClassifier()

Classification_model_gridsearchCV(model,param_grid,data_X,data_y) 
# K Neighbors



model = KNeighborsClassifier()



k_range = list(range(1, 30))

leaf_size = list(range(1,30))

weight_options = ['uniform', 'distance']

param_grid = {'n_neighbors': k_range, 'leaf_size': leaf_size, 'weights': weight_options}

Classification_model_gridsearchCV(model,param_grid,data_X,data_y)
# SVM



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