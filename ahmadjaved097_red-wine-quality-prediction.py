import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



%matplotlib inline



# setting plot style for all the plots

plt.style.use('fivethirtyeight')
df = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')

df.head()
print('Number of rows in the dataset: ',df.shape[0])

print('Number of columns in the dataset: ',df.shape[1])
df.info()
df.describe().round(decimals=3)
plt.figure(figsize=(10, 6))

sns.countplot(x='quality', data=df)

plt.title('Number of wines present in the dataset of a given quality')

plt.show()
# Function to plot barplot and boxplot of a given feature

def plot(x_val, y_val, palette='pastel'):

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    sns.barplot(x=x_val, y=y_val, data=df, ax=ax[0], palette=palette)

    sns.boxplot(x= x_val, y= y_val, data=df, ax=ax[1],palette=palette, linewidth=3)

    plt.tight_layout(w_pad=2)

    plt.show()
plot('quality','fixed acidity')
plot('quality', 'volatile acidity')
plot('quality', 'citric acid')
plot('quality', 'residual sugar')
plot('quality', 'chlorides')
plt.figure(figsize=(12,8))

corr = df.corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



sns.heatmap(corr,mask=mask, annot=True, linewidths=1, cmap='YlGnBu')

plt.show()
bins = (2, 6.5, 8)

group_names = ['bad', 'good']

df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)
df.head()
plt.figure(figsize=(7,6))

sns.countplot(x='quality', data=df, palette='pastel')

plt.title('Number of good and bad quality wines')

plt.show()
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

df['quality'] = label_encoder.fit_transform(df['quality'])

df.head(3)
X = df.drop('quality', axis=1)

y = df['quality']
from sklearn.preprocessing import scale
X_scaled = scale(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=41)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
knn = KNeighborsClassifier()
params = {

    'n_neighbors':list(range(1,15)),

    'p':[1, 2, 3, 4],

    'leaf_size':list(range(1,50)),

    'weights':['uniform', 'distance']

}
# Doing Gridsearch to find optimal parameters

knn_grid = GridSearchCV(estimator=knn, param_grid=params, scoring='accuracy',cv=5,n_jobs=-1)

knn_grid.fit(X_train, y_train)
knn_grid.best_params_
knn_grid.best_score_
knn_predict = knn_grid.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix

print('Accuracy Score: ',accuracy_score(y_test,knn_predict))

print('Using k-NN we get an accuracy score of: ',

      round(accuracy_score(y_test,knn_predict),5)*100,'%')
from sklearn.metrics import confusion_matrix





# Fucntion to create confusion Matrix

def conf_matrix(actual, predicted, model_name):

    cnf_matrix = confusion_matrix(actual, predicted)

#     cnf_matrix

    class_names = [0,1]

    fig,ax = plt.subplots()

    tick_marks = np.arange(len(class_names))

    plt.xticks(tick_marks,class_names)

    plt.yticks(tick_marks,class_names)



    #create a heat map

    sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'YlGnBu',

               fmt = 'g')

    ax.xaxis.set_label_position('top')

    plt.tight_layout()

    plt.title('Confusion matrix for ' + model_name + ' Model', y = 1.1)

    plt.ylabel('Actual label')

    plt.xlabel('Predicted label')

    plt.show()
conf_matrix(y_test, knn_predict, 'k-Nearest Neighbors')
from sklearn.metrics import classification_report
print(classification_report(y_test, knn_predict))
from sklearn.metrics import roc_auc_score,roc_curve
y_probabilities = knn_grid.predict_proba(X_test)[:,1]



#Create true and false positive rates

false_positive_rate_knn,true_positive_rate_knn,threshold_knn = roc_curve(y_test,y_probabilities)



#Plot ROC Curve

plt.figure(figsize=(10,6))

plt.title('Revceiver Operating Characterstic')

plt.plot(false_positive_rate_knn,true_positive_rate_knn, linewidth=2)

plt.plot([0,1],ls='--', linewidth=2)

plt.plot([0,0],[1,0],c='.5', linewidth=2)

plt.plot([1,1],c='.5',linewidth=2)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
#Calculate area under the curve

roc_auc_score(y_test,y_probabilities)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
params = {'C':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],

             'class_weight': [{1:0.5, 0:0.5}, {1:0.4, 0:0.6},{1:0.6, 0:0.4}, {1:0.7, 0:0.3},{1:0.3, 0:0.7}],

             'penalty': ['l1', 'l2'],

             'solver': ['liblinear', 'saga'],

             'max_iter':[50,100,150,200]

             }
# Doing Gridsearch to find optimal parameters

log_grid = GridSearchCV(estimator=logreg, param_grid=params, scoring='accuracy', cv=5, n_jobs=-1)

log_grid.fit(X_train, y_train)
log_grid.best_params_
log_grid.best_score_
log_predict = log_grid.predict(X_test)
print('Accuracy Score: ',accuracy_score(y_test,log_predict))

print('Using k-NN we get an accuracy score of: ',

      round(accuracy_score(y_test,log_predict),5)*100,'%')
conf_matrix(y_test, log_predict, 'Logistic Regression')
print(classification_report(y_test, knn_predict))
y_probabilities = log_grid.predict_proba(X_test)[:,1]



#Create true and false positive rates

false_positive_rate_log,true_positive_rate_log,threshold_log = roc_curve(y_test,y_probabilities)



#Plot ROC Curve

plt.figure(figsize=(10,6))

plt.title('Revceiver Operating Characterstic')

plt.plot(false_positive_rate_log,true_positive_rate_log, linewidth=2)

plt.plot([0,1],ls='--', linewidth=2)

plt.plot([0,0],[1,0],c='.5', linewidth=2)

plt.plot([1,1],c='.5', linewidth=2)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
#Calculate area under the curve

roc_auc_score(y_test,y_probabilities)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
param_grid = {

    'criterion': ['gini','entropy'],

    'max_depth': [None, 1, 2, 3, 4, 5, 6],

    'max_features': ['auto', 'sqrt','log2'],

    'max_leaf_nodes': [None, 1, 2, 3, 4, 5, 6],

    'min_samples_leaf': [1,2,3,4,5,6,7],

    'min_samples_split': [2,3,4,5,6,7,8,9,10]

}
# Doing Gridsearch to find optimal parameters

dt_grid = GridSearchCV(estimator=dt, param_grid=param_grid, scoring='accuracy',cv=5, n_jobs=-1)

dt_grid.fit(X_train, y_train)
dt_grid.best_params_
dt_grid.best_score_
dt_predict = dt_grid.predict(X_test)
print('Accuracy Score: ',accuracy_score(y_test,dt_predict))

print('Using Decision Tree Classifier we get an accuracy score of: ',

      round(accuracy_score(y_test,dt_predict),5)*100,'%')
conf_matrix(y_test, log_predict, 'Decision Tree')
print(classification_report(y_test, dt_predict))
y_probabilities = dt_grid.predict_proba(X_test)[:,1]



#Create true and false positive rates

false_positive_rate_dt,true_positive_rate_dt,threshold_dt = roc_curve(y_test,y_probabilities)



#Plot ROC Curve

plt.figure(figsize=(10,6))

plt.title('Revceiver Operating Characterstic')

plt.plot(false_positive_rate_dt,true_positive_rate_dt, linewidth=2)

plt.plot([0,1],ls='--', linewidth=2)

plt.plot([0,0],[1,0],c='.5', linewidth=2)

plt.plot([1,1],c='.5', linewidth=2)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.show()
#Calculate area under the curve

roc_auc_score(y_test,y_probabilities)
#Plot ROC Curve

plt.figure(figsize=(10,6))

plt.title('Reciver Operating Characterstic Curve')

plt.plot(false_positive_rate_knn,true_positive_rate_knn,linewidth=2, label='k-Nearest Neighbor')

plt.plot(false_positive_rate_log,true_positive_rate_log, linewidth=2, label='Logistic Regression')

plt.plot(false_positive_rate_dt,true_positive_rate_dt, linewidth=2, label='Decision Tree')

plt.plot([0,1],ls='--', linewidth=2)

plt.plot([0,0],[1,0],c='.5', linewidth=2)

plt.plot([1,1],c='.5', linewidth=2)

plt.ylabel('True positive rate')

plt.xlabel('False positive rate')

plt.legend()

plt.show()