import numpy as np 
import pandas as pd 
import seaborn as sns
import pickle
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import sys
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

sys.path.append('/kaggle/input/enron-project/enron_project')
original = "/kaggle/input/enron-project/enron_project/final_project_dataset.pkl"
destination = "final_project_dataset_unix.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))
dictionary = pickle.load(open("final_project_dataset_unix.pkl", 'rb') )

print("Population of dataset:", len(dictionary))
print(list(dictionary.keys())[4] , "\n" , dictionary[list(dictionary.keys())[4]])
print("Population of dataset:", len(dictionary))
print(list(dictionary.keys())[0] , "\n" , dictionary[list(dictionary.keys())[0]])
from feature_format import featureFormat
from feature_format import targetFeatureSplit

feature_list = ["poi", "salary", "bonus"] 
nparray = featureFormat(dictionary, ['poi','salary','bonus'])

for key in range(len(nparray)):
        if nparray[key][0]==True:
            plt.scatter(nparray[key][1],nparray[key][2],color = 'r')
        else:
            plt.scatter(nparray[key][1],nparray[key][2],color = 'lime')

plt.ylabel('bonus')
plt.xlabel('salary')   
plt.show()
dictionary.pop('TOTAL')
nparray = featureFormat(dictionary, ['poi','total_payments','bonus'])

for key in range(len(nparray)):
    if nparray[key][1]<3000000:  # to remove outliers
        if nparray[key][0]==True:
            plt.scatter(nparray[key][1],nparray[key][2],color = 'r')
        else:
            plt.scatter(nparray[key][1],nparray[key][2],color = 'lime')

plt.ylabel('bonus')
plt.xlabel('total_payments')   
plt.show()
nparray = featureFormat(dictionary, ['poi','total_stock_value','deferred_income'])

for key in range(len(nparray)):
    if nparray[key][2]>-500000 and nparray[key][1]<10000000:  # to remove outliers
        if nparray[key][0]==True:
            plt.scatter(nparray[key][1],nparray[key][2],color = 'r')
        else:
            plt.scatter(nparray[key][1],nparray[key][2],color = 'lime')

plt.ylabel('deferred_income')
plt.xlabel('total_stock_value')   
plt.show()
nparray = featureFormat(dictionary, ['poi','from_this_person_to_poi','from_poi_to_this_person'])

for key in range(len(nparray)):
    if nparray[key][1]<100:  # to remove outliers
        if nparray[key][0]==True:
            plt.scatter(nparray[key][1],nparray[key][2],color = 'r')
        else:
            plt.scatter(nparray[key][1],nparray[key][2],color = 'lime')

plt.ylabel('from_poi_to_this_person')
plt.xlabel('from_this_person_to_poi')   
plt.show()
nparray = featureFormat(dictionary, ['poi','shared_receipt_with_poi','long_term_incentive'])

for key in range(len(nparray)):
    if nparray[key][2]<10000000:  # to remove outliers
        if nparray[key][0]==True:
            plt.scatter(nparray[key][1],nparray[key][2],color = 'r')
        else:
            plt.scatter(nparray[key][1],nparray[key][2],color = 'lime')

plt.ylabel('long_term_incentive')
plt.xlabel('shared_receipt_with_poi')   
plt.show()
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

nparray = featureFormat(dictionary, ['poi','restricted_stock','exercised_stock_options','total_stock_value'])

for key in range(len(nparray)):
    if nparray[key][3]<2000000 and nparray[key][1]>0:  # to remove outliers
        if nparray[key][0]==True:
            ax.scatter(nparray[key][1],nparray[key][2], nparray[key][3], zdir='z', s=20, c='r')
        else:
            ax.scatter(nparray[key][1],nparray[key][2], nparray[key][3], zdir='z', s=20, c='lime')
plt.show()
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

nparray = featureFormat(dictionary, ['poi','shared_receipt_with_poi','from_this_person_to_poi','other'])

for key in range(len(nparray)):
    if nparray[key][3]<2000 and nparray[key][1]<5000:  # to remove outliers
        if nparray[key][0]==True:
            ax.scatter(nparray[key][1],nparray[key][2], nparray[key][3], zdir='z', s=20, c='r')
        else:
            ax.scatter(nparray[key][1],nparray[key][2], nparray[key][3], zdir='z', s=20, c='lime')
plt.show()
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

nparray = featureFormat(dictionary, ['poi','shared_receipt_with_poi','from_poi_to_this_person','other'])

for key in range(len(nparray)):
    if nparray[key][3]<2000 and nparray[key][1]<3000:  # to remove outliers
        if nparray[key][0]==True:
            ax.scatter(nparray[key][1],nparray[key][2], nparray[key][3], zdir='z', s=20, c='r')
        else:
            ax.scatter(nparray[key][1],nparray[key][2], nparray[key][3], zdir='z', s=20, c='lime')
plt.show()
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

nparray = featureFormat(dictionary, ['poi','from_this_person_to_poi','from_poi_to_this_person','shared_receipt_with_poi'])

for key in range(len(nparray)):
    if nparray[key][1]<500 and nparray[key][3]<1500:  # to remove outliers
        if nparray[key][0]==True:
            ax.scatter(nparray[key][1],nparray[key][2], nparray[key][3], zdir='z', s=20, c='r')
        else:
            ax.scatter(nparray[key][1],nparray[key][2], nparray[key][3], zdir='z', s=20, c='lime')
plt.show()
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

nparray = featureFormat(dictionary, ['poi','to_messages','from_messages','shared_receipt_with_poi'])

for key in range(len(nparray)):
    if nparray[key][3]<1500 and nparray[key][2]<6000:  # to remove outliers
        if nparray[key][0]==True:
            ax.scatter(nparray[key][1],nparray[key][2], nparray[key][3], zdir='z', s=20, c='r')
        else:
            ax.scatter(nparray[key][1],nparray[key][2], nparray[key][3], zdir='z', s=20, c='lime')
plt.show()
def dict_to_list(key,normalizer):
    new_list=[]

    for i in dictionary:
        if dictionary[i][key]=="NaN" or dictionary[i][normalizer]=="NaN":
            new_list.append(0.)
        elif dictionary[i][key]>=0:
            new_list.append(float(dictionary[i][key])/float(dictionary[i][normalizer]))
    return new_list

fraction_exercised_stock_options=dict_to_list("exercised_stock_options","total_stock_value")
fraction_restricted_stock=dict_to_list("restricted_stock","total_stock_value")
j = 0
for i in dictionary:
    dictionary[i]["fraction_exercised_stock_options"]=fraction_exercised_stock_options[j]
    dictionary[i]["fraction_restricted_stock"]=fraction_restricted_stock[j]
    j+=1
    

fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")
j = 0
for i in dictionary:
    dictionary[i]["fraction_from_poi_email"]=fraction_from_poi_email[j]
    dictionary[i]["fraction_to_poi_email"]=fraction_to_poi_email[j]
    j+=1
nparray = featureFormat(dictionary, ['poi','fraction_exercised_stock_options','long_term_incentive'])

for key in range(len(nparray)):
    if nparray[key][2]<10000000 and nparray[key][1]>0:
        if nparray[key][0]==True:
            plt.scatter(nparray[key][1],nparray[key][2],color = 'r')
        else:
            plt.scatter(nparray[key][1],nparray[key][2],color = 'lime')

plt.ylabel('long_term_incentive')
plt.xlabel('fraction_exercised_stock_options')   
plt.show()
nparray = featureFormat(dictionary, ['poi','fraction_restricted_stock','long_term_incentive'])

for key in range(len(nparray)):
    if nparray[key][1]<1 and nparray[key][2]<3000000:
        if nparray[key][0]==True:
            plt.scatter(nparray[key][1],nparray[key][2],color = 'r')
        else:
            plt.scatter(nparray[key][1],nparray[key][2],color = 'lime')

plt.ylabel('long_term_incentive')
plt.xlabel('fraction_restricted_stock')   
plt.show()
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

nparray = featureFormat(dictionary, ['poi','fraction_from_poi_email','fraction_to_poi_email','long_term_incentive'])

for key in range(len(nparray)):
    if nparray[key][3]<3000000 and nparray[key][1]<1:  # to remove outliers
        if nparray[key][0]==True:
            ax.scatter(nparray[key][1],nparray[key][2], nparray[key][3], zdir='z', s=20, c='r')
        else:
            ax.scatter(nparray[key][1],nparray[key][2], nparray[key][3], zdir='z', s=20, c='lime')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler 
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
features_list = ['poi','salary','bonus','fraction_from_poi_email','fraction_to_poi_email'
                 ,'fraction_exercised_stock_options','fraction_restricted_stock','shared_receipt_with_poi','long_term_incentive']
data = featureFormat(dictionary, features_list)
value , features = targetFeatureSplit(data)
xtrain, xtest, ytrain, ytest = train_test_split(features, value, test_size = 0.2)

xtrain_scaled = preprocessing.scale(xtrain)
xtest_scaled = preprocessing.scale(xtest)
  
print(xtrain_scaled[0:5, :]) 
lr = LogisticRegression(random_state = 0) 
grid = {'C':[0.01, 0.03, 0.1, 0.3, 1, 3, 10]}
grid_lr = GridSearchCV(lr,param_grid=grid,scoring='accuracy',cv=5)
grid_lr.fit(xtrain_scaled,ytrain) 

print(grid_lr.best_params_)
pred = grid_lr.predict(xtest_scaled)
print('Test Accuracy = ', grid_lr.score(xtest_scaled,ytest))
print(metrics.classification_report(ytest, pred, zero_division=0))
rf = RandomForestClassifier(n_estimators=200)
grid = {'n_estimators':[1, 10, 50],'max_depth':[25,30,35,40,45,50]}
grid_rf = GridSearchCV(rf,param_grid=grid,scoring='accuracy',cv=5)
grid_rf.fit(xtrain_scaled,ytrain)

print(grid_rf.best_params_)
pred = grid_rf.predict(xtest_scaled)
print('Accuracy = ',grid_rf.score(xtest_scaled,ytest))
print(metrics.classification_report(ytest,pred, zero_division = 0))
svmclassifier = svm.SVC()

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
grid_svm = GridSearchCV(svmclassifier, tuned_parameters)
grid_svm.fit(xtrain_scaled, ytrain)

print(grid_svm.best_params_)
pred = grid_svm.predict(xtest_scaled)
print('Accuracy = ',grid_svm.score(xtest_scaled,ytest))
print(metrics.classification_report(ytest,pred, zero_division = 0))
ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=200)
param_grid = {"base_estimator__criterion" : ["gini", "entropy"],"base_estimator__splitter":["best", "random"], "n_estimators": [1, 2]}
grid_ada = GridSearchCV(ada, param_grid=param_grid, scoring = 'accuracy', cv=5)
grid_ada.fit(xtrain_scaled, ytrain)

print(grid_ada.best_estimator_)
pred = grid_ada.predict(xtest_scaled)
print('Accuracy = ',grid_ada.score(xtest_scaled,ytest))
print(metrics.classification_report(ytest,pred, zero_division = 0))
pickle.dump(rf, open("my_classifier.pkl", "wb") )
pickle.dump(dictionary, open("my_dataset.pkl", "wb") )
pickle.dump(features_list, open("my_feature_list.pkl", "wb") )