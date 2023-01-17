# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import os, sys

from sklearn.preprocessing import MinMaxScaler

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
df=pd.read_csv('../input/detection-of-parkinson-disease/parkinsons.csv')
df
a= list(df.columns)

a
df.info()
df.shape
c_d= df.corr(method='pearson')
c_d
import seaborn as sb

sb.heatmap(c_d, 

            xticklabels=c_d.columns,

            yticklabels=c_d.columns,

            cmap='RdBu_r',

            annot=True,

            linewidth=0.5)
# Correlation matrix

import matplotlib.pyplot as plt # plotting

df.dataframeName = 'parkinsons.csv'

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()
from mpl_toolkits.mplot3d import Axes3D

plotCorrelationMatrix(df, 8)
#features and labels 

features =df.loc[:,df.columns!='status'].values[:,1:]

labels=df.loc[:,'status'].values
labels
print(labels[labels==1].shape[0], labels[labels==0].shape[0])
df.status.value_counts()
park_df = pd.DataFrame(df.status.value_counts())

park_df
park_df['Count'] = park_df.index

park_df
import seaborn as sns

sns.set_style("darkgrid")

sns.set_context({"figure.figsize": (8, 8)})

sns.barplot(x = 'Count', y = 'status', data =park_df )

#sns.countplot(df['status'],label='Count',palette="Set3")
# scaling and normalization

scaler=MinMaxScaler((-1,1))

x=scaler.fit_transform(features)

y=labels

# split the data set 

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=7)
print('Training Features Shape:', x_train.shape)

print('Training Labels Shape:', y_train.shape)

print('Testing Features Shape:', x_test.shape)

print('Testing Labels Shape:', y_test.shape)
# The baseline predictions are the historical averages

baseline_preds = x_test[:, a.index('status')]

# Baseline errors, and display average baseline error

baseline_errors = abs(baseline_preds - y_test)

print('Average baseline error: ', round(np.mean(baseline_errors), 2))

# Import the model we are using

from sklearn.ensemble import  RandomForestClassifier

# Instantiate model with 1000 decision trees

#rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

rf = RandomForestClassifier(max_depth=2, random_state=0)

# Train the model on training data

rf.fit(x_train,y_train);
# Use the forest's predict method on the test data

predictions = rf.predict(x_test)

# Calculate the absolute errors

errors = abs(predictions - y_test)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2))
from sklearn import metrics



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

from sklearn.metrics import average_precision_score,confusion_matrix,f1_score,recall_score,roc_auc_score,precision_score

confusion_matrix(y_test,predictions)
from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))
#recall_score(y_test,predictions)

#f1_score(y_test,predictions)

roc_auc_score(y_test,predictions)
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve



y_pred_proba=rf.predict_proba(x_test)[::,1]

fpr,tpr,threshold=roc_curve(y_test,y_pred_proba)

auc=roc_auc_score(y_test,y_pred_proba)



plt.plot(fpr,tpr,label="auc="+str(auc))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('AUC-ROC.Curve')



plt.legend (loc=8)

plt.show()

tpr
rf.score(x_test, y_test)
from sklearn.tree import DecisionTreeClassifier, export_graphviz

import pydot

# Limit depth of tree to 3 levels

rf_small = RandomForestClassifier(n_estimators=10, max_depth = 3)

rf_small.fit(x_train,y_train)

# Extract the small tree

tree_small = rf_small.estimators_[5]

# Save the tree as a png image

export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = b, rounded = True, precision = 1)

(graph, ) = pydot.graph_from_dot_file('small_tree.dot')

graph.write_png('small_tree.png');
from IPython.display import Image

Image(filename = 'small_tree.png')
b = a.remove('name')
b = a.remove('status')
# feature importance 

# Get numerical feature importances

importances = list(rf.feature_importances_)

# List of tuples with variable and importance

feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(a, importances)]

# Sort the feature importances by most important first

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 

[print('Variable: {:70} Importance: {}'.format(*pair)) for pair in feature_importances];


import matplotlib.pyplot as plt

%matplotlib inline

# Set the style

plt.style.use('fivethirtyeight')

# list of x locations for plotting

x_values = list(range(len(importances)))

# Make a bar chart

plt.bar(x_values, importances, orientation = 'vertical')

# Tick labels for x axis

plt.xticks(x_values,a, rotation='vertical')

# Axis labels and title

plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
#https://chrisalbon.com/machine_learning/trees_and_forests/adaboost_classifier/

#https://www.datacamp.com/community/tutorials/adaboost-classifier-python

from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn import metrics

svc=SVC(probability=True, kernel='linear')



abc =AdaBoostClassifier(n_estimators=50,

                         learning_rate=1)



model = abc.fit(x_train, y_train)



y_pred = model.predict(x_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

from sklearn.metrics import average_precision_score,confusion_matrix,f1_score,recall_score,roc_auc_score,precision_score

confusion_matrix(y_test,y_pred)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve



y_pred_proba=abc.predict_proba(x_test)[::,1]

fpr,tpr,threshold=roc_curve(y_test,y_pred_proba)

auc=roc_auc_score(y_test,y_pred_proba)



plt.plot(fpr,tpr,label="auc="+str(auc))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('AUC-ROC.Curve')



plt.legend (loc=8)

plt.show()

tpr
model=XGBClassifier()

model.fit(x_train,y_train)
y_pred2=model.predict(x_test)

print(accuracy_score(y_test, y_pred2)*100)
from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred2))
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve



y_pred_proba=model.predict_proba(x_test)[::,1]

fpr,tpr,threshold=roc_curve(y_test,y_pred_proba)

auc=roc_auc_score(y_test,y_pred_proba)



plt.plot(fpr,tpr,label="auc="+str(auc))

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('AUC-ROC.Curve')



plt.legend (loc=8)

plt.show()

tpr
import pandas

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

seed = 7

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

models.append(('abc', AdaBoostClassifier()))

models.append(('XGB', XGBClassifier()))

models.append(('rf', RandomForestClassifier()))

models.append(('log', LogisticRegression())) 

# evaluate each model in turn

results = []

a = []

scoring = 'accuracy'

for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)

    results.append(cv_results)

    a.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)

# boxplot algorithm comparison

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(a)

plt.show()





# import warnings filter

from warnings import simplefilter

# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)