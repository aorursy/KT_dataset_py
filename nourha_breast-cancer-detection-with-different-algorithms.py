#import necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import io

from time import time
#import dataset into dataFrame using Pandas

data= pd.read_csv('../input/data.csv')

data.head()
sns.countplot(data['diagnosis'],label="Count")
#visualize correlations by heatmap

f,ax=plt.subplots(figsize = (18,18))

sns.heatmap(data.corr(),annot= True,linewidths=0.5,fmt = ".1f",ax=ax)

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.title('Correlation Map')

plt.show()
#drop unnecessary data columns

data.drop(['Unnamed: 32','id'], axis=1,inplace=True)
#split data into features and labels (with all varaibles)

X = np.array(data.drop(['diagnosis'],1))

y=data['diagnosis']
#encode y

y_en = y.replace({'B':0, 'M':1})

y_en= np.array(y_en)
# the features taking into consideration the correlations between varaibles



X_1 = np.array(data.drop(['diagnosis','perimeter_mean','area_mean','radius_worst','area_worst','perimeter_worst'],1))
# split the dataset into training and testing sets (feature selection mode)

from sklearn.model_selection import train_test_split

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_1,y_en, test_size = 0.25, random_state = 0)

print("Training set with feature selection has {} samples.".format(X_train1.shape[0]))

print("Testing set with feature selection has {} samples.".format(X_test1.shape[0]))
# create a function to calculate accuracy and f beta score for each model

from sklearn.metrics import fbeta_score, accuracy_score,confusion_matrix

def test_predict(clf, X_train, y_train, X_test, y_test): 

    

    results = {}

    clf = clf.fit(X_train,y_train)

    #  Get the predictions on the test set(X_test),

    predictions_test = clf.predict(X_test)

    #  Compute accuracy on test set using accuracy_score()

    accuracy=results['acc_test'] = accuracy_score(y_test, predictions_test)

    #  Compute F-score on the test set which is y_test

    fbeta=results['f_test'] = fbeta_score(y_test,predictions_test, beta = 0.5)

    # Success

    print("accuracy for {} model is {}".format(clf.__class__.__name__,results['acc_test']))

    print("F beta for {} model is {}".format(clf.__class__.__name__,results['f_test']))

    print('------------------------------------------------------------------')

    # Return the results

    return results
# Import the models from sklearn

from xgboost import XGBClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

# Initialize the models

clf1 = XGBClassifier(random_state =15)

clf2= SVC(random_state =15)

clf3 = RandomForestClassifier(random_state =15)

# Collect results on the clf

results = {}

for clf in [clf1, clf2, clf3]:

    clf_name = clf.__class__.__name__

    results[clf_name] = {}       

    test_predict(clf, X_train1, y_train1, X_test1, y_test1)
def enable_plotly_in_cell():

  import IPython

  from plotly.offline import init_notebook_mode

  display(IPython.core.display.HTML('''<script src="/static/components/requirejs/require.js"></script>'''))

  init_notebook_mode(connected=False)
# Compare Algorithms

import matplotlib.pyplot as plt

fig = plt.figure()

fig.suptitle( 'Algorithm Comparison' )

ax = fig.add_subplot(111)

plt.boxplot([(0.9622641509433962,0.0),(0.8703071672354951, 0.0),(0.9195402298850575, 0.0)])

ax.set_xticklabels(['XGBClassifier','SVM','Random Forest'])

plt.show()
#split the data into traning and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y_en, test_size = 0.25, random_state = 0)
# train the models and calculate scores with all varaibles

for clf in [clf1, clf2, clf3]:

    clf_name = clf.__class__.__name__

    results[clf_name] = {}       

    test_predict(clf, X_train, y_train, X_test, y_test)
#plot scores in table 

from plotly.offline import iplot

import plotly.graph_objs as go

enable_plotly_in_cell()

trac = go.Table(

    header=dict(values=['Classifier', 'Accuracy Score','F-beta Score']),

    cells=dict(values=[['XGBClassifier','SVM','Random Forest'],

                       [0.9790209790209791,0.6293706293706294,0.965034965034965],

                      [0.9770114942528735,0.0,0.968379446640316]]))



tabl = [trac] 



iplot(tabl)
fig = plt.figure()

fig.suptitle( 'Algorithm Comparison' )

ax = fig.add_subplot(111)

plt.boxplot([(0.9770114942528735,0.0),(0.0, 0.0),(0.968379446640316, 0.0)])

ax.set_xticklabels(['XGBClassifier','SVM','Random Forest'])

plt.show()
#optimize the final selected model using grid search

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, r2_score, fbeta_score

#Initialize the classifier

clf = XGBClassifier(random_state =42)

parameters = {'max_depth':[6],'n_estimators':[100], 'learning_rate' : [0.1], 'min_samples_split':[7],'gamma':[0.01],'min_child_weight':[1],'Max_delta_step':[1],'colsample_bytree':[1],'reg_lambda':[1]}



scorer = make_scorer(fbeta_score, beta=0.5)



grid_obj = GridSearchCV(clf,parameters,scoring=scorer)



grid_fit = grid_obj.fit(X_train,y_train)



best_clf = grid_fit.best_estimator_



predictions = (clf.fit(X_train, y_train)).predict(X_test)

best_predictions = best_clf.predict(X_test)



# Report the before-and-afterscores

print("Unoptimized model\n------")

print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))

print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))

print("\nOptimized Model\n------")

print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))

print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
#plot scores in table 

from plotly.offline import iplot

import plotly.graph_objs as go

enable_plotly_in_cell()

tr = go.Table(

    header=dict(values=['Model', 'Accuracy Score','F-beta Score']),

    cells=dict(values=[['Unoptimized','Optimized'],

                       [0.9790,0.9860],

                      [0.9770,0.9811]]))



opt = [tr] 



iplot(opt)
fig = plt.figure()

fig.suptitle( 'Algorithm Comparison' )

ax = fig.add_subplot(111)

plt.boxplot([(0.9770,0.0),(0.9811, 0.0)])

ax.set_xticklabels(['Unoptimized','Optimized'])

plt.show()
cm = confusion_matrix(y_test,best_clf.predict(X_test))

sns.heatmap(cm,annot=True,fmt="d")

plt.title('Optimized model results')