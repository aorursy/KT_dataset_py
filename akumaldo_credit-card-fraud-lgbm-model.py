# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)



# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



# Common imports

import numpy as np

import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline 

#allow you to plot directly without having to call .show()





import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.





import warnings

warnings.filterwarnings(action="ignore") #ignore warnings
data = pd.read_csv("../input/creditcard.csv") #loading our data
data.head()
data['Class'].value_counts() #492 frauds in our dataset.
data['Amount'].value_counts().sum() #just checking the sum of all the transactions in our dataset.
corr = data.corr()

corr['Class'].sort_values(ascending=False).head(12) #only the first 12 positively correlated values.
corr = data.corr()

corr['Class'].sort_values(ascending=True).head(12) #only the first 12 negatively correlated values.
from sklearn.model_selection import train_test_split #importing the necessary module



train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
print(train_set.shape, test_set.shape) #printing the shape of the training and testing set
train_labels = train_set["Class"].copy()

train = train_set.drop("Class", axis=1).copy()

test_labels = test_set["Class"].copy()

test = test_set.drop("Class", axis=1).copy()
print(train.shape, test.shape) #checking the shape again
from sklearn.metrics import mean_squared_error

from sklearn.metrics import precision_score, recall_score

from sklearn.metrics import f1_score

import time #implementing in this function the time spent on training the model

from sklearn.model_selection import cross_val_score



#Generic function for making a classification model and accessing performance:

def classification_model(model, X_train, y_train):

    #Fit the model:

    time_start = time.perf_counter() #start counting the time

    model.fit(X_train,y_train)

    n_cache = []

    

    train_predictions = model.predict(X_train)

    precision = precision_score(y_train, train_predictions)

    recall = recall_score(y_train, train_predictions)

    f1 = f1_score(y_train, train_predictions)

    

    print("Precision ", precision)

    print("Recall ", recall)

    print("F1 score ", f1)

    

    cr_val = cross_val_score(model, X_train, y_train, cv=5, scoring='recall')

    

    time_end = time.perf_counter()

    

    total_time = time_end-time_start

    print("Cross Validation Score: %f" %np.mean(cr_val))

    print("Amount of time spent during training the model and cross validation: %4.3f seconds" % (total_time))
from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression(solver="liblinear")

classification_model(log_reg, train,train_labels)
from sklearn.ensemble import RandomForestClassifier



rand_forest = RandomForestClassifier(n_estimators=10)

classification_model(rand_forest,train, train_labels)
# Plot feature importance

feature_importance = rand_forest.feature_importances_

# make importances relative to max importance

plt.figure(figsize=(20, 10)) #figure size

feature_importance = 100.0 * (feature_importance / feature_importance.max()) #making it a percentage relative to the max value

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5

plt.barh(pos,feature_importance[sorted_idx], align='center')

plt.yticks(pos,train.iloc[sorted_idx],fontsize=15)

plt.xlabel('Relative Importance', fontsize=20)

plt.ylabel('Features', fontsize=20)

plt.title('Variable Importance', fontsize=30)
from sklearn.model_selection import cross_val_predict #it is gonna be used to use the cross validation prediction

#fitting the testing dataset

test_predictions = rand_forest.predict(test)

precision = precision_score(test_labels, test_predictions)

recall = recall_score(test_labels, test_predictions)

f1 = f1_score(test_labels, test_predictions)

y_scores = rand_forest.predict_proba



cross_value = cross_val_score(rand_forest,test,test_labels ,cv=5, scoring='recall')



y_probas_forest = cross_val_predict(rand_forest, test, test_labels, cv=3,

                                    method="predict_proba")



y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class

from sklearn.metrics import roc_auc_score #gonna use the roc_auc score, importing the module necessary to use it.





#printing the results

print("Precision ", precision)

print("Recall ", recall)

print("F1 score ", f1)

print("Cross Validation Score: %f" %np.mean(cross_value))

print("ROC AUC score: ", roc_auc_score(test_labels, y_scores_forest))
from sklearn.neighbors import KNeighborsClassifier



knn_clf = KNeighborsClassifier(weights='distance', n_neighbors=4)

classification_model(knn_clf, train, train_labels)
##SHOULD'VE IMPLEMENTED A FUNCTION IN ORDER TO AVOID REPEATING CODE HERE



test_predictions_knn = knn_clf.predict(test)

precision_knn = precision_score(test_labels, test_predictions_knn)

recall_knn = recall_score(test_labels, test_predictions_knn)

f1_knn = f1_score(test_labels, test_predictions_knn)

y_scores_knn = knn_clf.predict_proba



cross_value = cross_val_score(knn_clf,test,test_labels ,cv=5, scoring='recall')



y_probas_knn = cross_val_predict(knn_clf, test, test_labels, cv=3,

                                    method="predict_proba")



y_scores_knn = y_probas_knn[:,1] # score = proba of positive class
#printing the results

print("Precision ", precision_knn)

print("Recall ", recall_knn)

print("F1 score ", f1_knn)

print("Cross Validation Score: %f" %np.mean(cross_value))

print("ROC AUC score: ", roc_auc_score(test_labels, y_scores_knn))
import lightgbm as lgb



lgb_model = lgb.LGBMClassifier(num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



classification_model(lgb_model, train, train_labels)
##SHOULD'VE IMPLEMENTED A FUNCTION IN ORDER TO AVOID REPEATING CODE HERE



test_predictions_lgb = lgb_model.predict(test)

precision_lgb = precision_score(test_labels, test_predictions_lgb)

recall_lgb = recall_score(test_labels, test_predictions_lgb)

f1_lgb = f1_score(test_labels, test_predictions_lgb)

y_scores_lgb = lgb_model.predict_proba



cross_value_lgb = cross_val_score(lgb_model,test,test_labels ,cv=5, scoring='recall')



y_probas_lgb = cross_val_predict(lgb_model, test, test_labels, cv=3,

                                    method="predict_proba")



y_scores_lgb = y_probas_lgb[:,1] # score = proba of positive class
#printing the results

print("Precision ", precision_lgb)

print("Recall ", recall_lgb)

print("F1 score ", f1_lgb)

print("Cross Validation Score: %f" %np.mean(cross_value_lgb))

print("ROC AUC score: ", roc_auc_score(test_labels, y_scores_lgb))