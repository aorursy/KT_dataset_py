# Import pandas and numpy

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno



#Import scikit-learn metrics module for accuracy calculation

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.model_selection import train_test_split

results = []
def death_ratio(y, predict_y):

    tn, fp, fn, tp = confusion_matrix(y, predict_y).ravel()

    return fn/y.size





def plot_confusion_matrix(y, predict_y, name):

    # Confusion Matrix

    

    confusion_matrix_ = confusion_matrix(y, predict_y)

    



    # Ploting heatmap of confusion matrix

    # https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels

    class_names = ['edible','poisonous']

    confusion_matrix_ = pd.DataFrame(confusion_matrix_,index=class_names, columns=class_names)

    heatmap = sns.heatmap(confusion_matrix_, annot=True, fmt='g')



    plt.xlabel('Predicted Class',size=14)

    plt.ylabel('Actual Class',size=14)

    plt.title(f"{name} Confusion Matrix\n",size=24)

    plt.show()
df = pd.read_csv("/kaggle/input/mushroom-classification/mushrooms.csv")

desc = df.describe()

# sns.set(rc={'figure.figsize':(20,20)})

# plot = sns.heatmap(df.isnull())

# plt.show(plot)
desc
sum(desc.loc['unique'])
msno.bar(df)
df = df.apply(lambda col: pd.factorize(col, sort=True)[0]) ### poisonous -> 1; edible -> 0;



y = df['class']

df_ = df.drop(['class', 'veil-type'], axis=1)



X_train, X_test, y_train, y_test = train_test_split(df_, y, test_size=0.2)
df.describe()

from sklearn.preprocessing import OneHotEncoder



# Create the encoder.

encoder = OneHotEncoder(handle_unknown="ignore")

encoder.fit(X_train)    # Assume for simplicity all features are categorical.



# Apply the encoder.

X_train = encoder.transform(X_train)

X_test = encoder.transform(X_test)
# requires graphviz and python-graphviz conda packages

import graphviz



xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42, eval_metric="auc")



xgb_model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)], verbose=False)



xgb.plot_importance(xgb_model)



# plot the output tree via matplotlib, specifying the ordinal number of the target tree

# xgb.plot_tree(xgb_model, num_trees=xgb_model.best_iteration)



# converts the target tree to a graphviz instance

xgb.to_graphviz(xgb_model, num_trees=xgb_model.best_iteration)
import xgboost as xgb

import time

start_time = time.time()

clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

train, test = X_train, X_test



clf.fit(train, y_train)



test_predict_y = clf.predict(test)

train_predict_y = clf.predict(train)

elapsed_time = (time.time() - start_time)

print("--- %s seconds ---" % elapsed_time)

acc_test = accuracy_score(y_test, test_predict_y)

acc_train = accuracy_score(y_train, train_predict_y)

intoxication_ratio = death_ratio(y_train, train_predict_y)

results.append({'name': 'XGBoost', 'Accuracy (test)': acc_test, 'Accuracy (train)': acc_train, 

'intoxication':intoxication_ratio, 'Elapsed Time': elapsed_time})

print("Accuracy (test):", acc_test)

print("Accuracy (train):", acc_train)

print("Death ratio:", intoxication_ratio)
import time

start_time = time.time()



from sklearn.linear_model import LogisticRegression

# Defining the LR model and performing the hyper parameter tuning using gridsearch

#weights = np.linspace(0.05, 0.95, 20)

%time

params = {'C' : [

                10**-4,10**-3,10**-2,10**-1,1,10**1,10**2,10**3],

          'penalty': ['l1', 'l2']#,'class_weight': [{0: x, 1: 1.0-x} for x in weights]

         }

clf = LogisticRegression(n_jobs= -1,random_state=42)

clf.fit(X_train,y_train)

model = GridSearchCV(estimator=clf,cv = 2,n_jobs= -1,param_grid=params,scoring='f1',verbose= 2,)

model.fit(X_train,y_train)

elapsed_time = (time.time() - start_time)

print("--- %s seconds ---" % elapsed_time)



print("Best estimator is", model.best_params_)
import time

start_time = time.time()



# model fitting using the best parameter.

%time

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(n_jobs= -1,random_state=42,C= 10,penalty= 'l1')

train, test = X_train, X_test

clf.fit(train,y_train)

test_predict_y = clf.predict(test)

train_predict_y = clf.predict(train)

elapsed_time = (time.time() - start_time)

print("--- %s seconds ---" % elapsed_time)

acc_test = accuracy_score(y_test, test_predict_y)

acc_train = accuracy_score(y_train, train_predict_y)

intoxication_ratio = death_ratio(y_train, train_predict_y)

results.append({'name': 'Logistic Regression', 'Accuracy (test)': acc_test, 'Accuracy (train)': acc_train, 

'intoxication':intoxication_ratio, 'Elapsed Time': elapsed_time})

print("Accuracy (test):", acc_test)

print("Accuracy (train):", acc_train)

print("Death ratio:", intoxication_ratio)
plot_confusion_matrix(**{'name': "Mushroom classification",

                       'y': y_train,

                       'predict_y': train_predict_y,})
import time

start_time = time.time()



from sklearn import preprocessing

from sklearn.naive_bayes import BernoulliNB



# Creating labelEncoder

le = preprocessing.LabelEncoder()

X_train

# Encoding target data

y_train = le.fit_transform(y_train)

y_test = le.fit_transform(y_test)



#Create a Gaussian Classifier

model = BernoulliNB()



# Train the model using the training sets

model.fit(X_train, y_train)



#Predict the response for test dataset

test_predict_y = model.predict(X_test)

train_predict_y = model.predict(X_train)



elapsed_time = (time.time() - start_time)

print("--- %s seconds ---" % elapsed_time)

acc_test = accuracy_score(y_test, test_predict_y)

acc_train = accuracy_score(y_train, train_predict_y)

intoxication_ratio = death_ratio(y_train, train_predict_y)

results.append({'name': 'Naive Bayes', 'Accuracy (test)': acc_test, 'Accuracy (train)': acc_train, 

'intoxication':intoxication_ratio, 'Elapsed Time': elapsed_time})

print("Accuracy (test):", acc_test)

print("Accuracy (train):", acc_train)

print("Death ratio:", intoxication_ratio)
import time

start_time = time.time()



# Import MLP Classifier



from sklearn.neural_network import MLPClassifier



# TODO: Create a MLP Classifier 

mlp = MLPClassifier(hidden_layer_sizes=(21))



#Train the model using the training sets 

mlp.fit(X_train, y_train)



#Predict the response for test dataset

test_predict_y = mlp.predict(X_test)

train_predict_y = mlp.predict(X_train)



# Model Accuracy: how often is the classifier correct?

elapsed_time = (time.time() - start_time)

print("--- %s seconds ---" % elapsed_time)

acc_test = accuracy_score(y_test, test_predict_y)

acc_train = accuracy_score(y_train, train_predict_y)

intoxication_ratio = death_ratio(y_train, train_predict_y)

results.append({'name': 'MLP (layers=1, n=21)', 'Accuracy (test)': acc_test, 'Accuracy (train)': acc_train, 

'intoxication':intoxication_ratio, 'Elapsed Time': elapsed_time})

print("Accuracy (test):", acc_test)

print("Accuracy (train):", acc_train)

print("Death ratio:", intoxication_ratio)
# X_train.todense()[0].size
import time

start_time = time.time()



# Import MLP Classifier



from sklearn.neural_network import MLPClassifier



# TODO: Create a MLP Classifier 

mlp = MLPClassifier(hidden_layer_sizes=(1,))



#Train the model using the training sets 

mlp.fit(X_train, y_train)



#Predict the response for test dataset

test_predict_y = mlp.predict(X_test)

train_predict_y = mlp.predict(X_train)



# Model Accuracy: how often is the classifier correct?

elapsed_time = (time.time() - start_time)

print("--- %s seconds ---" % elapsed_time)

acc_test = accuracy_score(y_test, test_predict_y)

acc_train = accuracy_score(y_train, train_predict_y)

intoxication_ratio = death_ratio(y_train, train_predict_y)

results.append({'name': 'MLP (layers=1, n=1)', 'Accuracy (test)': acc_test, 'Accuracy (train)': acc_train, 

'intoxication':intoxication_ratio, 'Elapsed Time': elapsed_time})

print("Accuracy (test):", acc_test)

print("Accuracy (train):", acc_train)

print("Death ratio:", intoxication_ratio)
import time

start_time = time.time()



# Import KNN Classifier

from sklearn.neighbors import KNeighborsClassifier



# Import LabelEncoder

from sklearn import preprocessing



# Creating labelEncoder

le = preprocessing.LabelEncoder()



# Encoding target data

y_train = le.fit_transform(y_train)

y_test = le.fit_transform(y_test)



# Create a KNN Classifier

model = KNeighborsClassifier(n_neighbors=3)



#Train the model using the training sets 

model.fit(X_train, y_train)



#Predict the response for test dataset

test_predict_y = model.predict(X_test)

train_predict_y = model.predict(X_train)

# Model Accuracy: how often is the classifier correct?

elapsed_time = (time.time() - start_time)

print("--- %s seconds ---" % elapsed_time)

acc_test = accuracy_score(y_test, test_predict_y)

acc_train = accuracy_score(y_train, train_predict_y)

intoxication_ratio = death_ratio(y_train, train_predict_y)

results.append({'name': 'KNN', 'Accuracy (test)': acc_test, 'Accuracy (train)': acc_train, 

'intoxication':intoxication_ratio, 'Elapsed Time': elapsed_time})

print("Accuracy (test):", acc_test)

print("Accuracy (train):", acc_train)

print("Death ratio:", intoxication_ratio)
import time

start_time = time.time()



# Import SVM model

from sklearn import svm



#Create a SVM Classifier

clf = svm.SVC(kernel='poly', degree=10, gamma='auto')



#Train the model using the training sets

clf.fit(X_train, y_train)



#Predict the response for test dataset

test_predict_y = clf.predict(X_test)

train_predict_y = clf.predict(X_train)



# Model Accuracy: how often is the classifier correct?

elapsed_time = (time.time() - start_time)

print("--- %s seconds ---" % elapsed_time)

acc_test = accuracy_score(y_test, test_predict_y)

acc_train = accuracy_score(y_train, train_predict_y)

intoxication_ratio = death_ratio(y_train, train_predict_y)

results.append({'name': 'SVM', 'Accuracy (test)': acc_test, 'Accuracy (train)': acc_train, 

'intoxication':intoxication_ratio, 'Elapsed Time': elapsed_time})

print("Accuracy (test):", acc_test)

print("Accuracy (train):", acc_train)

print("Death ratio:", intoxication_ratio)
import time

start_time = time.time()



# Import DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier



# Create Decision Tree classifer object

clf = DecisionTreeClassifier()



# Train Decision Tree Classifer

clf = clf.fit(X_train, y_train)



#Predict the response for test dataset

test_predict_y = clf.predict(X_test)

train_predict_y = clf.predict(X_train)



# Model Accuracy: how often is the classifier correct?

elapsed_time = (time.time() - start_time)

print("--- %s seconds ---" % elapsed_time)

acc_test = accuracy_score(y_test, test_predict_y)

acc_train = accuracy_score(y_train, train_predict_y)

intoxication_ratio = death_ratio(y_train, train_predict_y)

results.append({'name': 'Decision tree', 'Accuracy (test)': acc_test, 'Accuracy (train)': acc_train, 

'intoxication':intoxication_ratio, 'Elapsed Time': elapsed_time})

print("Accuracy (test):", acc_test)

print("Accuracy (train):", acc_train)

print("Death ratio:", intoxication_ratio)
import time

start_time = time.time()



from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

clf = RandomForestClassifier(n_estimators=100, max_depth=2,

                              random_state=0)

clf.fit(X_train, y_train)  



importances = clf.feature_importances_



test_predict_y = clf.predict(X_test)

train_predict_y = clf.predict(X_train)



elapsed_time = (time.time() - start_time)

print("--- %s seconds ---" % elapsed_time)

acc_test = accuracy_score(y_test, test_predict_y)

acc_train = accuracy_score(y_train, train_predict_y)

intoxication_ratio = death_ratio(y_train, train_predict_y)

results.append({'name': 'Random forest', 'Accuracy (test)': acc_test, 'Accuracy (train)': acc_train, 

'intoxication':intoxication_ratio, 'Elapsed Time': elapsed_time})

print("Accuracy (test):", acc_test)

print("Accuracy (train):", acc_train)

print("Death ratio:", intoxication_ratio)
print(pd.DataFrame(results).to_latex(index=False))