import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import re

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline



from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler



from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_auc_score



# Load in the train and test datasets

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
cptrain = train.copy()

cptrain.head()
print (cptrain.isnull().sum())
cptrain.drop('Cabin', axis=1, inplace=True)
print(cptrain.loc[cptrain['Embarked'].isnull()])

cptrain.drop([61,829],axis=0, inplace=True)
cptrain['Name_title'] = cptrain['Name'].map(lambda x: re.findall('([A-Za-z]+)\.', x))

cptrain['Name_title'] = cptrain['Name_title'].map(lambda x: x[0]) 

#"Here we are using the Regex: [A-Za-z]+).. It looks for strings which lie between A-Z 

#or a-z and followed by a .(dot). Thus, we successfully extract the Initials from the Name" (by ashwin on kaggle)



cptrain['Name_title'].value_counts()
cptrain['Name_title'].replace(\

                    ['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],\

                    ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

#(written by ashwin by kaggle. Thank you, ashwin)
cptrain.groupby('Name_title')['Age'].mean()
cptrain.loc[(cptrain['Age'].isnull()) & (cptrain['Name_title']=='Master'), 'Age'] = cptrain.loc[\

                                                            cptrain['Name_title']=='Master']['Age'].mean()

cptrain.loc[(cptrain['Age'].isnull()) & (cptrain['Name_title']=='Miss'), 'Age'] = cptrain.loc[\

                                                            cptrain['Name_title']=='Miss']['Age'].mean()

cptrain.loc[(cptrain['Age'].isnull()) & (cptrain['Name_title']=='Mr'), 'Age'] = cptrain.loc[\

                                                            cptrain['Name_title']=='Mr']['Age'].mean()

cptrain.loc[(cptrain['Age'].isnull()) & (cptrain['Name_title']=='Mrs'), 'Age'] = cptrain.loc[\

                                                            cptrain['Name_title']=='Mrs']['Age'].mean()

cptrain.loc[(cptrain['Age'].isnull()) & (cptrain['Name_title']=='Other'), 'Age'] = cptrain.loc[\

                                                            cptrain['Name_title']=='Other']['Age'].mean()

# round up the age to get rid of decimal points and convert to int64

cptrain['Age'] = cptrain['Age'].round().astype(np.int64)

cptrain.info()
cptrain.drop('Name_title', axis=1, inplace=True)
cptrain.drop(['Ticket', 'Fare'], axis=1, inplace=True)
cptrain.drop(['Name','PassengerId'], axis=1, inplace=True)
cptrain.head()
cptrain['fam_size'] = 0

cptrain['fam_size'] = cptrain['SibSp']+cptrain['Parch']
cptrain.drop(['SibSp', 'Parch'], axis=1, inplace=True)
cptrain.head()
cptrain['Sex'] = cptrain['Sex'].replace({'male':0, 'female':1})

cptrain['Embarked'] = cptrain['Embarked'].replace({'Q':0, 'S':1, 'C':2})
cptrain.head()
sns.heatmap(cptrain.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #correlation matrix of our data

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
X = cptrain.drop('Survived', axis=1)

y = cptrain['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)





scaler = RobustScaler()

robust_scaled_df = scaler.fit_transform(X)

robust_scaled_df = pd.DataFrame(robust_scaled_df)



scaler = MinMaxScaler()

minmax_scaled_df = scaler.fit_transform(X)

minmax_scaled_df = pd.DataFrame(minmax_scaled_df)



scaler = StandardScaler()

scaled_df = scaler.fit_transform(X)

scaled_df = pd.DataFrame(scaled_df)





# Plot the scaled data

fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(9, 5))

ax1.set_title('Before Scaling')

sns.kdeplot(cptrain['Pclass'], ax=ax1)

sns.kdeplot(cptrain['Sex'], ax=ax1)

sns.kdeplot(cptrain['Age'], ax=ax1)

sns.kdeplot(cptrain['fam_size'], ax=ax1)





ax2.set_title('After Robust Scaling')

sns.kdeplot(robust_scaled_df[0], ax=ax2)

sns.kdeplot(robust_scaled_df[1], ax=ax2)

sns.kdeplot(robust_scaled_df[2], ax=ax2)

sns.kdeplot(robust_scaled_df[4], ax=ax2)





ax3.set_title('After Min-Max Scaling')

sns.kdeplot(minmax_scaled_df[0], ax=ax3)

sns.kdeplot(minmax_scaled_df[1], ax=ax3)

sns.kdeplot(minmax_scaled_df[2], ax=ax3)

sns.kdeplot(minmax_scaled_df[4], ax=ax3)



ax4.set_title('After Standard Scaling')

sns.kdeplot(scaled_df[0], ax=ax4)

sns.kdeplot(scaled_df[1], ax=ax4)

sns.kdeplot(scaled_df[2], ax=ax4)

sns.kdeplot(scaled_df[4], ax=ax4)



plt.show()
# shortcut to calculate null accuracy (for binary classification problems coded as 0/1)

max(y_test.mean(), 1 - y_test.mean()) # this function takes the max value from the two
X = cptrain.drop('Survived', axis=1)

y = cptrain['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)



steps = [('scaling', StandardScaler()), ('knn', KNeighborsClassifier())]

pipe = Pipeline(steps)



# map the paramater names to the values that should be searched

parameters = {'knn__n_neighbors':range(1,33), 'knn__weights':['distance','uniform']}



# instantiate the grid

grid = GridSearchCV(estimator=pipe, param_grid=parameters, cv=10, scoring='accuracy')



# fit the grid with data

grid.fit(X_train, y_train)



print ("1.Mean test score results of the cross validation: \n{}".format(grid.cv_results_["mean_test_score"]))

print ("1A.Mean result of the cross validation scores above: {}".format(grid.cv_results_["mean_test_score"].mean()))

print ("\n2.Best score for current grid: {}".format(grid.best_score_))

print ("\n3.Mean score of the hold-out (test data): {}".format(grid.score(X_test, y_test)))

print ("\n4.Best parameter for grid: \n{}".format(grid.best_params_))



plt.plot(range(1,65), grid.cv_results_["mean_test_score"])

plt.xlabel("Value of K+Weights")

plt.ylabel("Prediction Accuracy")

plt.show()

cptrain.head()
cpt_unscld = cptrain.copy()

cpt_unscld['Age_group']=0

cpt_unscld.loc[cpt_unscld['Age']<=16,'Age_group']=0

cpt_unscld.loc[(cpt_unscld['Age']>16) & (cpt_unscld['Age']<=32),'Age_group']=1

cpt_unscld.loc[(cpt_unscld['Age']>32) & (cpt_unscld['Age']<=48),'Age_group']=2

cpt_unscld.loc[(cpt_unscld['Age']>48) & (cpt_unscld['Age']<=64),'Age_group']=3

cpt_unscld.loc[(cpt_unscld['Age']>64) & (cpt_unscld['Age']<=80),'Age_group']=4

cpt_unscld.drop('Age', axis=1, inplace=True)

cpt_unscld.head()
cpt_unscld = pd.get_dummies(data=cpt_unscld,  columns=['Pclass', 'Embarked'], drop_first=True)

cpt_unscld.head()
# split data into train_test

X = cpt_unscld.drop('Survived', axis=1)

y = cpt_unscld['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)



# create pipeline

steps = [('knn', KNeighborsClassifier())]

pipe = Pipeline(steps)



# map the paramater names to the values that should be searched

parameters = {'knn__n_neighbors':range(1,33), 'knn__weights':['distance','uniform']}



# instantiate the grid

grid_unscld = GridSearchCV(estimator=pipe, param_grid=parameters, cv=10, scoring='accuracy')



# fit the grid with data

grid_unscld.fit(X_train, y_train)



print ("\nMean score of the hold-out (test data): {}".format(grid_unscld.score(X_test, y_test)))
X = cptrain.drop('Survived', axis=1)

y = cptrain['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)



steps = [('scaling', StandardScaler()), ('logreg', LogisticRegression())]

pipe = Pipeline(steps)



# map the paramater names to the values that should be searched

c_space = np.logspace(-5, 8, 15)

parameters = {'logreg__C': c_space, 'logreg__penalty': ['l1', 'l2']}



# instantiate the grid

grid = GridSearchCV(estimator=pipe, param_grid=parameters, cv=10, scoring='accuracy')



# fit the grid with data

grid.fit(X_train, y_train)



print ("\n1.Mean score of the hold-out (test data): {}".format(grid.score(X_test, y_test)))

print ("\n2.Best parameter for grid: \n{}".format(grid.best_params_))

"""For more on regularizations and penalties viist: https://goo.gl/o28sMQ"""
# split data into train_test

X_unscld = cpt_unscld.drop('Survived', axis=1)

y_unscld = cpt_unscld['Survived']

X_train, X_test, y_train, y_test = train_test_split(X_unscld, y_unscld, random_state=42, stratify=y)



# create pipeline

steps = [('logreg', LogisticRegression())]

pipe = Pipeline(steps)



# map the paramater names to the values that should be searched

c_space = np.logspace(-5, 8, 15)

parameters = {'logreg__C': c_space, 'logreg__penalty': ['l1', 'l2']} # L1 is usually better when there are ouliers



# instantiate the grid

grid_unscld = GridSearchCV(estimator=pipe, param_grid=parameters, cv=10, scoring='accuracy')



# fit the grid with data

grid_unscld.fit(X_train, y_train)



print ("\nMean score of the hold-out (test data): {}".format(grid_unscld.score(X_test, y_test)))

print ("\n2.Best parameter for grid: \n{}".format(grid_unscld.best_params_))
"""Calculate the confusion matrix of the unscaled dataset, using Logistic Regression and standardizing the features"""

X = cptrain.drop('Survived', axis=1)

y = cptrain['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)



steps = [('scaling', StandardScaler()), ('logreg', LogisticRegression())]

pipe = Pipeline(steps)



# map the paramater names to the values that should be searched

c_space = np.logspace(-5, 8, 15)

parameters = {'logreg__C': c_space, 'logreg__penalty': ['l1', 'l2']}



# instantiate the grid

grid = GridSearchCV(estimator=pipe, param_grid=parameters, cv=10, scoring='accuracy')



# fit the grid with data

grid.fit(X_train, y_train)



y_pred_class = grid.predict(X_test)

print(metrics.confusion_matrix(y_test, y_pred_class))

print("Actual number of passengers that survived: {}".format(y_test.sum()))

print("Actual number of passengers that died: {}".format(y_test.count() - y_test.sum()))
# save confusion matrix and slice into four pieces

cfsion_mtx = metrics.confusion_matrix(y_test, y_pred_class)

TP = cfsion_mtx[1, 1]

TN = cfsion_mtx[0, 0]

FP = cfsion_mtx[0, 1]

FN = cfsion_mtx[1, 0]
print((TP + TN) / float(TP + TN + FP + FN))

print(metrics.accuracy_score(y_test, y_pred_class))
print((FP + FN) / float(TP + TN + FP + FN))

print(1 - metrics.accuracy_score(y_test, y_pred_class))
print(TP / float(TP + FN))

print(metrics.recall_score(y_test, y_pred_class))
print(TN / float(TN + FP))
print(FP / float(TN + FP))
print(TP / float(TP + FP))

print(metrics.precision_score(y_test, y_pred_class))
print("First ten predictions of the passengers that survived:")

print (grid.predict(X_test)[0:10])

grid.predict_proba(X_test)[0:10, 1]
from sklearn.preprocessing import binarize

y_pred_prob = grid.predict_proba(X_test)[:, 1]

y_pred_class03 = binarize([y_pred_prob], 0.3)[0]
metrics.recall_score(y_test, y_pred_class03)
print(metrics.confusion_matrix(y_test, y_pred_class03))

print("Specificity (prediction accuracy of passengers that died): {}" .format(91/(91+47)))
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)



def evaluate_threshold(threshold):

    print('Sensitivity:', tpr[thresholds > threshold][-1])

    print('Specificity:', 1 - fpr[thresholds > threshold][-1])

    print('Specificity+Sensitivity:' ,(tpr[thresholds > threshold][-1])+(1 - fpr[thresholds > threshold][-1]))



for i in list([0.3, 0.4, 0.5, 0.6, 0.7]):

    evaluate_threshold(i)

    print ("Threshold:" ,i,"\n")
"""Let's use the Logistic Regression model with the scaler"""

X = cptrain.drop('Survived', axis=1)

y = cptrain['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

steps = [('scaling', StandardScaler()), ('logreg', LogisticRegression())]

pipe = Pipeline(steps)



c_space = np.logspace(-5, 8, 15)

parameters = {'logreg__C': c_space, 'logreg__penalty': ['l1', 'l2']}

grid = GridSearchCV(estimator=pipe, param_grid=parameters, cv=10, scoring='accuracy')



# fit the grid to the data

grid.fit(X_train, y_train)

y_pred_prob = grid.predict_proba(X_test)[:, 1]



fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)

print("Area Under the Curve (threshold=0.5): {}" .format(metrics.roc_auc_score(y_test, y_pred_prob)))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1])

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC curve for survival classifier')

plt.xlabel('False Positive Rate (1 - FPR =  Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_class03)

print("Area Under the Curve (threshold=0.3): {}" .format(metrics.roc_auc_score(y_test, y_pred_class03)))

plt.plot(fpr, tpr)

plt.plot([0,1], [0,1])

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.title('ROC curve for survival classifier')

plt.xlabel('False Positive Rate (1 - FPR =  Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)