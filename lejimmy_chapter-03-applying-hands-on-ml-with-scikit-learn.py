import pandas as pd

import numpy as np



# read csv into DataFrame

df = pd.read_csv('../input/train.csv')



# preview the first 5 entries

df.head()
# summary of columns

df.info()
# statistical summary

df.describe()
import matplotlib.pyplot as plt



# generate a historgram for all numerical features

df.hist(bins = 50, figsize = [30, 20])

plt.show()
titanic = df.copy()
titanic = df.drop(['Survived'], axis = 1)

titanic_labels = df['Survived'].copy()
# list of columns to drop

drop_columns = ['PassengerId', 'Ticket', 'Name', 'Cabin']



titanic.drop(drop_columns, axis = 1, inplace = True)
titanic.info()
# split both categorical and numerical columns

titanic_num = titanic.drop(['Sex', 'Embarked'], axis = 1)

titanic_cat = titanic[['Sex', 'Embarked']]
titanic_cat.Sex.unique()
titanic_cat.Embarked.unique()
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder



# fill in the missing values, then scale the numerical columns

num_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy = 'median')),

        ('std_scaler', StandardScaler()),

])



# fill in the missing values, then encode the categorical columns

cat_pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy = 'most_frequent')),

        ('one_hot', OneHotEncoder(categories = [['female', 'male'], ['S', 'C', 'Q']]))

])
from sklearn.compose import ColumnTransformer



# list of both numerical and categorical features

num_attribs = list(titanic_num)

cat_attribs = list(titanic_cat)



full_pipeline = ColumnTransformer([

        # apply numerical pipeline transformation to our numerical features

        ('num', num_pipeline, num_attribs),

        # apply categorical pipeline transformation to our categorical features

        ('cat', cat_pipeline, cat_attribs)

])



# apply transformations to our training features

titanic_prepared = full_pipeline.fit_transform(titanic)
from sklearn.linear_model import SGDClassifier



# model constructor

sgd_clf = SGDClassifier()



# train model given passenger details and labels

sgd_clf.fit(titanic_prepared, titanic_labels)
# preview features

some_passenger = titanic_prepared[1]

some_passenger
# preview label

titanic_labels[1]
# use model to predict based on features

sgd_clf.predict([some_passenger])
from sklearn.model_selection import cross_val_score



cross_val_score(sgd_clf, titanic_prepared, titanic_labels, cv = 3, scoring = 'accuracy')
from sklearn.model_selection import cross_val_predict



titanic_labels_pred = cross_val_predict(sgd_clf, titanic_prepared, titanic_labels, cv = 3)
from sklearn.metrics import confusion_matrix



confusion_matrix(titanic_labels, titanic_labels_pred)
# pretend we reached perfection

titanic_labels_perfect_pred = titanic_labels

confusion_matrix(titanic_labels, titanic_labels_perfect_pred)
from sklearn.metrics import precision_score, recall_score



precision_score(titanic_labels, titanic_labels_pred)
recall_score(titanic_labels, titanic_labels_pred)
from sklearn.metrics import f1_score

f1_score(titanic_labels, titanic_labels_pred)
titanic_labels_scores = sgd_clf.decision_function([some_passenger])

titanic_labels_scores
titanic_labels_scores = cross_val_predict(sgd_clf, titanic_prepared, titanic_labels, cv = 3, method = 'decision_function')
from sklearn.metrics import precision_recall_curve



precisions, recalls, thresholds = precision_recall_curve(titanic_labels, titanic_labels_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], 'b--', label = 'Precision')

    plt.plot(thresholds, recalls[:-1], 'g-', label = 'Recall')

    

    plt.legend()

    plt.xlabel('Threshold')

    plt.grid(True)



plt.figure(figsize = [10,5])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.show()
def plot_precision_vs_recall(precisions, recalls):

    plt.plot(recalls, precisions, "b-", linewidth=2)

    plt.xlabel("Recall", fontsize=16)

    plt.ylabel("Precision", fontsize=16)

    plt.axis([0, 1, 0, 1])

    plt.grid(True)



plt.figure(figsize=(8, 6))

plot_precision_vs_recall(precisions, recalls)

plt.show()
threshold_90_precision = thresholds[np.argmax(precisions >= .90)]

threshold_90_precision
titanic_labels_pred_90 = (titanic_labels_scores >= threshold_90_precision)



precision_score(titanic_labels, titanic_labels_pred_90)
recall_score(titanic_labels, titanic_labels_pred_90)
from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(titanic_labels, titanic_labels_scores)



def plot_roc_curve(fpr, tpr, label = 'None'):

  plt.plot(fpr, tpr, linewidth = 2, label = label)

  plt.plot([0, 1], [0, 1], 'k--')

  plt.axis([0, 1, 0, 1])                                    # Not shown in the book

  plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16) # Not shown

  plt.ylabel('True Positive Rate (Recall)', fontsize=16)    # Not shown

  plt.grid(True)                                            # Not shown



plt.figure(figsize=(8, 6))                         # Not shown

plot_roc_curve(fpr, tpr)

  

plt.show()
from sklearn.metrics import roc_auc_score



# auc score based on our training set and our scores

roc_auc_score(titanic_labels, titanic_labels_scores)
from sklearn.ensemble import RandomForestClassifier



# Random Forest Classifier constructor

forest_clf = RandomForestClassifier(n_estimators = 100)



# returns array of probability the random forest believes the passenger survived or not

titanic_probas_forest = cross_val_predict(forest_clf, titanic_prepared, titanic_labels, cv = 3, method = 'predict_proba')
# scores of only the positive class

titanic_scores_forest = titanic_probas_forest[:, 1]



# compute fpr, tpr, at various thresholds

fpr_forest, tpr_forest, thresholds_forest = roc_curve(titanic_labels, titanic_scores_forest)
# plot both classifiers

plt.figure(figsize = [8, 6])

plt.plot(fpr, tpr, 'b:', label = 'SGD')

plot_roc_curve(fpr_forest, tpr_forest, label = 'Random Forest')

plt.legend(loc = 'lower right')

plt.show()
# roc score

roc_auc_score(titanic_labels, titanic_scores_forest)
# predict labels based on RFC

titanic_labels_pred_forest = cross_val_predict(forest_clf, titanic_prepared, titanic_labels, cv = 3)
precision_score(titanic_labels, titanic_labels_pred_forest)
recall_score(titanic_labels, titanic_labels_pred_forest)
from sklearn.model_selection import RandomizedSearchCV



# number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 20, stop = 2000, num = 10)]

# number of features to consider at every split

max_features = ['auto', 'sqrt']

# maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# method of selecting samples for training each tree

bootstrap = [True, False]



# create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



# iterate to find the best hyperparameters

forest_clf = RandomForestClassifier()

rnd_search = RandomizedSearchCV(forest_clf, param_distributions = random_grid, n_iter = 100, cv = 10, verbose = 2, scoring = 'accuracy')



# fit the data

rnd_search.fit(titanic_prepared, titanic_labels)
# view best estimator

rnd_search.best_estimator_
feature_importances = rnd_search.best_estimator_.feature_importances_

feature_importances
# from our transformation pipeline, extract the categorical transformation

cat_encoder = full_pipeline.named_transformers_['cat']



# extract the one hot encoder step

name, encoder = cat_encoder.steps[1]



# empty list for one hot categories

cat_one_hot_attribs = []



# iterate through categories to append the one hot categories

for i in range(len(encoder.categories_)):

    cat_one_hot_attribs += list(encoder.categories_[i])



# combine numerical features with one hot categories

attributes = num_attribs + cat_one_hot_attribs



# zip and view the feature importances next to the attribute name

sorted(zip(feature_importances, attributes), reverse = True)
# get our best model

final_model = rnd_search.best_estimator_



# load in the test set

X_test = pd.read_csv('../input/test.csv')



# drop columns

X_test_prepared = X_test.drop(drop_columns, axis = 1)



# call our transformation pipeline

X_test_prepared = full_pipeline.transform(X_test_prepared)



# make our final predictions

final_predictions = final_model.predict(X_test_prepared)
from IPython.display import HTML

import base64



# define download link function

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index = False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload = payload, title = title, filename = filename)

    return HTML(html)



# create submission DataFrame with the corresponding Id

submission = pd.DataFrame()

submission['PassengerId'] = X_test['PassengerId']

submission['Survived'] = final_predictions



# create download link

create_download_link(submission)