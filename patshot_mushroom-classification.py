import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline
#Read the data using pandas

data=pd.read_csv("../input/mushrooms.csv")
data.head()
data.describe().transpose()
total_null_values = sum(data.isnull().sum())

print(total_null_values)
from sklearn.preprocessing import LabelEncoder

Enc = LabelEncoder()

data_tf = data.copy()

for i in data.columns:

    data_tf[i]=Enc.fit_transform(data[i])
X = data_tf.drop(['class'], axis=1)

Y = data_tf['class']
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 42)
X_train.head(5)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

log_clf = LogisticRegression()

log_clf.fit(X_train, Y_train)

LR_pred=log_clf.predict(X_test)



accuracy_score(Y_test, LR_pred)
from sklearn.metrics import confusion_matrix

confusion_matrix(LR_pred, Y_test)
from sklearn.model_selection import cross_val_predict



y_pred_cv = cross_val_predict(log_clf, X_test, Y_test, cv=5)
confusion_matrix(y_pred_cv, Y_test)
from sklearn.model_selection import cross_val_score

cv_score = cross_val_score(log_clf, X_train, Y_train, cv = 30)

print(cv_score)

print("Accuracy: %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2))
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier()

rnd_clf.fit(X_train, Y_train)

Y_pred = rnd_clf.predict(X_test)
accuracy_score(Y_test, Y_pred)
confusion_matrix(Y_pred, Y_test)
scores = cross_val_score(rnd_clf, X, Y, cv=10)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
from sklearn.model_selection import GridSearchCV
param_grid = [

{'n_estimators': [3, 10, 30, 50, 100], 'max_features': [2, 4, 6, 8]},

]



grid_search = GridSearchCV(rnd_clf, param_grid, cv=10, scoring='f1')
grid_search.fit(X, Y)
grid_search.best_params_
feat_score=[]

for name, score in zip(X_train.columns, grid_search.best_estimator_.feature_importances_):

    feat_score.append([name, score])
feat_score.sort(reverse=True, key= lambda x:x[1])

for char in feat_score:

    print(char)
grid_search.best_estimator_.fit(X_train, Y_train)

y_pred = grid_search.best_estimator_.predict(X_test)

confusion_matrix(y_pred, Y_test)
param_grid = [

{'n_estimators': [3, 10, 30, 50, 100], 'max_features': [2, 4, 6, 8]},

]



grid_search = GridSearchCV(rnd_clf, param_grid, cv=10, scoring='roc_auc')

#using ROC Area Under Curve to evaluate this time round
grid_search.fit(X, Y)

grid_search.best_params_
feat_score=[]

for name, score in zip(X_train.columns, grid_search.best_estimator_.feature_importances_):

    feat_score.append([name, score])
feat_score.sort(reverse=True, key= lambda x:x[1])

for char in feat_score:

    print(char)
odor_labels = data['odor'].value_counts().axes[0]

edible_o =[]

poi_o = []

N =0

for odor in odor_labels:

    size = len(data[data['odor'] == odor].index)

    edibles = len(data[(data['odor'] == odor) & (data['class'] == 'e')].index)

    edible_o.append(edibles)

    poi_o.append(size-edibles)

    N=N+1



#Plotting

ind = np.arange(N)

width = 0.35



fig, ax = plt.subplots(figsize=(12,8))



rects1 = ax.bar(ind, poi_o, width, color='r')

rects2 = ax.bar(ind + width, edible_o, width, color='y')



# Labels and Ticks along the axes.

ax.set_ylabel('Instances')

ax.set_title('Poisonous and Edible Mushrooms by their Odors')

ax.set_xticks(ind + width / 2)

ax.set_xticklabels(('None', 'Foul', 'Fishy', 'Spicy', 'Almond', 'Anise', 'Pungent', 'Creosote', 'Musty'))



ax.legend((rects1[0], rects2[0]), ('Poisonous', 'Edible'))





def autolabel(rects):

    

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,

                '%d' % int(height),

                ha='center', va='bottom')



autolabel(rects1)

autolabel(rects2)



plt.show()
gillsize_labels = data['gill-size'].value_counts().axes[0]



edible_o =[]

poi_o = []

N =0

for gs in gillsize_labels:

    size = len(data[data['gill-size'] == gs].index)

    edibles = len(data[(data['gill-size'] == gs) & (data['class'] == 'e')].index)

    edible_o.append(edibles)

    poi_o.append(size-edibles)

    N=N+1

    

#Plotting

ind = np.arange(N)

width = 0.35



fig, ax = plt.subplots(figsize=(12,8))



rects1 = ax.bar(ind, poi_o, width, color='r')

rects2 = ax.bar(ind + width, edible_o, width, color='y')



# Labels and Ticks along the axes.

plt.ylim(0,4500)

ax.set_ylabel('Instances')

ax.set_title('Poisonous and Edible Mushrooms by the size of their Gills')

ax.set_xticks(ind + width / 2)

ax.set_xticklabels(('Broad','Narrow'))



ax.legend((rects1[0], rects2[0]), ('Poisonous', 'Edible'))





def autolabel(rects):

     #To plot the labels on top of the bars.

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,

                '%d' % int(height),

                ha='center', va='bottom')



autolabel(rects1)

autolabel(rects2)



plt.show()
gillcolor_labels = data['gill-color'].value_counts().axes[0]

edible_o =[]

poi_o = []

N =0

for gc in gillcolor_labels:

    size = len(data[data['gill-color'] == gc].index)

    edibles = len(data[(data['gill-color'] == gc) & (data['class'] == 'e')].index)

    edible_o.append(edibles)

    poi_o.append(size-edibles)

    N=N+1

    

#Plotting

ind = np.arange(N)

width = 0.35



fig, ax = plt.subplots(figsize=(12,8))



rects1 = ax.bar(ind, poi_o, width, color='r')

rects2 = ax.bar(ind + width, edible_o, width, color='y')



# Labels and Ticks along the axes.

plt.ylim(0,2000)

ax.set_ylabel('Instances')

ax.set_title('Poisonous and Edible Mushrooms by the color of their Gills')

ax.set_xticks(ind + width / 2)

ax.set_xticklabels(('Buff', 'Pink', 'White', 'Brown', 'Gray', 'Chocolate', 'Purple', 'Black', 'Red', 'Yellow', 'Orange','Green'))



ax.legend((rects1[0], rects2[0]), ('Poisonous', 'Edible'))





def autolabel(rects):

    #To plot the labels on top of the bars.

    for rect in rects:

        height = rect.get_height()

        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,

                '%d' % int(height),

                ha='center', va='bottom')



autolabel(rects1)

autolabel(rects2)



plt.show()