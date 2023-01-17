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



# Import the libraries we'll use below.

import numpy as np

from matplotlib import pyplot as plt

import pandas as pd

import seaborn as sns  # for nicer plots

sns.set(style="darkgrid")  # default style



import tensorflow as tf

from tensorflow import keras

from keras import metrics

from keras import regularizers
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
def generateBaselineOutputNobodySurvives(df):

    #  The baseline is that Nobody survived the tragedy Titanic.

    #  1502 out of 2224 people died in the tragic event.

    #  We choose 68% of the people died

    output = pd.DataFrame({'PassengerId': df.PassengerId, 'Survived': np.full((len(df)), 0)})

    return output



output_train = generateBaselineOutputNobodySurvives(train_data)

output_test = generateBaselineOutputNobodySurvives(test_data)
from sklearn.metrics import mean_squared_error

from sklearn.metrics import log_loss

print("MSE baseline: ", mean_squared_error(train_data["Survived"], output_train["Survived"]))

print("LogLoss baseline", log_loss(train_data["Survived"], output_train["Survived"]))
output_test.to_csv('baseline_submission.csv', index=False)

print("Your baseline submission was successfully saved!")
print("Correlation factors:")

print(train_data.corr(method ='pearson')['Survived'])
###  Data Preprocessing

train_data_copy = train_data.copy()

test_data_copy = test_data.copy()



###  Data Preprocessing



##  Remove all the NaN values from all columns

# Change NaN in Ages column to the median values

median_age = train_data["Age"].median()

train_data["Age"] = train_data["Age"].replace(np.nan, median_age)

#  Change NaN in the Embarked column `Southampton` or `S` which is the heighest in frequency.

train_data["Embarked"] = train_data["Embarked"].replace(np.nan, 'S')



##  Drop Unnecessary tables with bad co-relation

#  Drop Name table as it is not co-related with one's survival.

#  Drop the Cabin column beacause it is very sparse.

#  Drop the ticket number table, since its numeric value does not really mean anything.

for feature in ['PassengerId', 'Name','Cabin', 'Ticket']:

    train_data.drop(feature, axis=1, inplace=True)





##  Apply zscore to the Fare, Pclass, SibSp, Parch and Age column column

from scipy import stats

for feature in ['Fare', 'Pclass', 'SibSp', 'Parch', 'Age']:

    train_data[feature] = stats.zscore(train_data[feature])



##  Apply one-hot encoding for `Sex` and `Embarked` feature

for feature in ['Sex', 'Embarked']:

    one_hot = pd.get_dummies(train_data[feature])

    train_data = train_data.drop(feature,axis = 1)

    train_data = train_data.join(one_hot)
print("Correlations after transformation and one-hot:")

print(train_data.corr(method ='pearson')['Survived'])
#  Separiting X_train and Y_train

Y_train = pd.DataFrame(train_data["Survived"]).to_numpy().flatten()

train_data.drop("Survived", axis=1, inplace=True)

X_train = train_data.to_numpy()
##  Helper Plot function for plots

def plot_history(history):

  plt.ylabel('Loss')

  plt.xlabel('Epoch')

  plt.xticks(range(0, len(history['loss'] + 1)))

  plt.plot(history['loss'], label="training", marker='o')

  plt.plot(history['val_loss'], label="validation", marker='o')

  plt.legend()

  plt.show()
##  Function to Build the model

def build_model(input_shape, learning_rate=0.01):

    """Build a TF logistic regression model using Keras.



    Args:

    input_shape: The shape of the model's input. 

    learning_rate: The desired learning rate for SGD.



    Returns:

    model: A tf.keras model (graph).

    """

    # This is not strictly necessary, but each time you build a model, TF adds

    # new nodes (rather than overwriting), so the colab session can end up

    # storing lots of copies of the graph when you only care about the most

    # recent. Also, as there is some randomness built into training with SGD,

    # setting a random seed ensures that results are the same on each identical

    # training run.

    tf.keras.backend.clear_session()

    np.random.seed(0)

    tf.compat.v1.set_random_seed(0)



    # Build a model using keras.Sequential. While this is intended for neural

    # networks (which may have multiple layers), we want just a single layer for

    # logistic regression.

    model = keras.Sequential()



    # Keras layers can do pre-processing.

    model.add(keras.layers.Flatten(input_shape=input_shape))



    model.add(keras.layers.Dense(

      units=512,                   # number of units/neurons

      use_bias=True,               # use a bias param

      activation="relu",          # apply the relu function 

    ))

    

    model.add(keras.layers.Dense(

      units=256,                   # number of units/neurons

      use_bias=True,               # use a bias param

      activation="relu",            # apply the relu function

    ))

    

    model.add(keras.layers.Dense(

      units=128,                   # number of units/neurons

      use_bias=True,               # use a bias param

      activation="relu",            # apply the relu function

    ))

    # This layer constructs the linear set of parameters for each input feature

    # (as well as a bias), and applies a sigmoid to the result. The result is

    # binary logistic regression.

    model.add(keras.layers.Dense(

      units=1,                     # output dim (for binary classification)

      use_bias=True,               # use a bias param

      activation="sigmoid"         # apply the sigmoid function!

    ))



    # Finally, we compile the model. This finalizes the graph for training.

    # We specify the binary_crossentropy loss (equivalent to log loss).

    # Notice that we are including 'binary accuracy' as one of the metrics that we

    # ask Tensorflow to report when evaluating the model.

    model.compile(loss='binary_crossentropy', 

                optimizer='adam', 

                metrics=[metrics.binary_accuracy])



    return model
##  Train the model

model = build_model(input_shape=X_train[0].shape, learning_rate=0.01)



# Fit the model.

history = model.fit(

  x = X_train,   # our binary training examples

  y = Y_train,   # corresponding binary labels

  epochs=5,             # number of passes through the training data

  batch_size=64,        # mini-batch size for SGD

  validation_split=0.1, # use a fraction of the examples for validation

  verbose=1             # display some progress output during training

  )



history = pd.DataFrame(history.history)

display(history)

plot_history(history)
from sklearn.metrics import confusion_matrix

train_predictions = model.predict(X_train).flatten()



thresholds = [0.3,0.49, 0.5, 0.51, 0.52, 0.7]



group_names = ["True Neg", "False Pos", "False Neg" , "True Pos"]



for threshold in thresholds:

    train_predictions_copy = np.copy(train_predictions)

    train_predictions_copy[train_predictions < threshold] = 0.0

    train_predictions_copy[train_predictions >= threshold] = 1.0

    

    cf_matrix = confusion_matrix(Y_train, train_predictions_copy)

    tn, fp, fn, tp = cf_matrix.ravel()

    

    group_counts = ["{0:0.0f}".format(value) for value in

                cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in

                     cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in

          zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)



    plt.figure()

    plt.title(

      "Threshold : " + str(threshold) + "\n"

      + "Accuracy : " + str((tp+tn)/(tp+tn+fp+fn)) + "\n"

      + "Precision : " + str((tp)/(tp+fp)) + "\n"

      + "Recall : " + str((tp)/(tp+fn)))

    sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')
#  Trying to find outliers

pd.set_option("display.max_rows", None, "display.max_columns", None)

train_data_copy['Prediction'] = train_predictions

train_data_copy['Difference'] = abs(train_data_copy['Survived'] - train_data_copy['Prediction'])



train_data_copy = train_data_copy[['Difference', 'Survived','Prediction', 'Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked']]

train_data_copy = train_data_copy.sort_values('Difference')

train_data_copy.head()
##  Evaluate the model



###  Data Preprocessing

test_data_copy = test_data.copy()

##  Drop Unnecessary tables with bad co-relation

#  Drop Name table as it is not co-related with one's survival.

#  Drop the Cabin column beacause it is very sparse.

#  Drop the ticket number table, since its numeric value does not really mean anything.

for feature in ['PassengerId', 'Name','Cabin', 'Ticket']:

    test_data.drop(feature, axis=1, inplace=True)



##  Remove all the NaN values from all columns

# Change NaN in Ages column to the median values

test_data["Age"] = test_data["Age"].replace(np.nan, median_age)

#  Change NaN in the Embarked column `Southampton` or `S` which is the heighest in frequency.

test_data["Embarked"] = test_data["Embarked"].replace(np.nan, 'S')

test_data["Fare"] = test_data["Fare"].replace(np.nan, train_data_copy["Fare"].median())



##  Apply zscore to the Fare, Pclass, SibSp, Parch and Age column column

from scipy import stats

for feature in ['Fare', 'Pclass', 'SibSp', 'Parch', 'Age']:

    test_data[feature] = stats.zscore(test_data[feature])



##  Apply one-hot encoding for `Sex` and `Embarked` feature

for feature in ['Sex', 'Embarked']:

    one_hot = pd.get_dummies(test_data[feature])

    test_data = test_data.drop(feature,axis = 1)

    test_data = test_data.join(one_hot)

    

X_test = test_data.to_numpy()

test_data.head()
test_data_copy.head()
test_predictions = model.predict(X_test).flatten()

test_predictions_actual_values = np.array([1 if val >= 0.51 else 0 for val in test_predictions])



output_test_logit = pd.DataFrame({'PassengerId': test_data_copy.PassengerId, 'Survived': test_predictions_actual_values})



output_test_logit.to_csv('logit_submission.csv', index=False)

print("Your logit submission was successfully saved!")
