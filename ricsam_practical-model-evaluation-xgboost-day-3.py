# import libraries

import random

import os

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import confusion_matrix, accuracy_score

import category_encoders as ce

from xgboost import XGBClassifier
# set seed for reproducibility

random.seed(42)



# read in our data

df_2018 = pd.read_csv("../input/practical-model-evaluation/data_jobs_info_2018.csv")

df_2019 = pd.read_csv("../input/practical-model-evaluation/data_jobs_info_2019.csv")
# using df_2018

# split into predictor & target variables

X = df_2018.drop("job_title", axis=1)

y = df_2018["job_title"]



# Splitting data into training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20)



# save out the split training data to use with Cloud AutoML

with open("train_data_2018.csv", "+w") as file:

    pd.concat([X_train, y_train], axis=1).to_csv(file, index=False)
# encode all features using ordinal encoding

encoder_x = ce.OrdinalEncoder()

X_encoded = encoder_x.fit_transform(X)



# you'll need to use a different encoder for each dataframe

encoder_y = ce.OrdinalEncoder()

y_encoded = encoder_y.fit_transform(y)



# split encoded dataset

X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(X_encoded, y_encoded,

                                                    train_size=0.80, test_size=0.20)
from xgboost import XGBClassifier



# train XGBoost model with default parameters

my_model = XGBClassifier()

my_model.fit(X_train_encoded, y_train_encoded, verbose=False)



# and save our model

my_model.save_model("xgboost_baseline.model.2018")
# get predictions

xgb_predictions = my_model.predict(X_test_encoded)
# XGBoost accuracy

print("XGBoost: " + str(accuracy_score(y_test_encoded, xgb_predictions)))
import numpy as np

import matplotlib.pyplot as plt

from sklearn.utils.multiclass import unique_labels



# function based one from SciKitLearn documention (https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)

# and is modified and redistributed here under a BSD liscense, https://opensource.org/licenses/BSD-2-Clause

def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    

    # Only use the labels that appear in the data

    #classes = classes[unique_labels(y_true, y_pred)]

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    fig.set_figheight(15)

    fig.set_figwidth(15)

    return ax
# plot confusion matrix

plot_confusion_matrix(xgb_predictions, y_test_encoded, 

                      classes=unique_labels(y_test),

                      normalize=True,

                      title='XGBoost Confusion Matrix')