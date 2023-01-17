# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#load packages

import sys #access to system parameters https://docs.python.org/3/library/sys.html

print("Python version: {}". format(sys.version))



import pandas as pd #collection of functions for data processing and analysis modeled after R dataframes with SQL like features

print("pandas version: {}". format(pd.__version__))



import matplotlib #collection of functions for scientific and publication-ready visualization

from matplotlib import cm

from matplotlib import gridspec

from matplotlib import pyplot as plt

print("matplotlib version: {}". format(matplotlib.__version__))



import numpy as np #foundational package for scientific computing

print("NumPy version: {}". format(np.__version__))



import scipy as sp #collection of functions for scientific computing and advance mathematics

print("SciPy version: {}". format(sp.__version__)) 



import IPython

from IPython import display #pretty printing of dataframes in Jupyter notebook

print("IPython version: {}". format(IPython.__version__)) 



import sklearn #collection of machine learning algorithms

from sklearn import metrics

print("scikit-learn version: {}". format(sklearn.__version__))



import math

#print("math version: {}". format(math.__version__))



import seaborn as sns

print("seaborn version: {}". format(sns.__version__))



import tensorflow as tf

from tensorflow.python.data import Dataset

print("tensorflow version: {}". format(tf.__version__))



tf.logging.set_verbosity(tf.logging.ERROR)

#pd.options.display.max_columns = 15

pd.options.display.max_rows = 15

pd.options.display.float_format = '{:.1f}'.format

train_dataframe = pd.read_csv("../input/train.csv",sep=",")

train_dataframe.head()

#train_dataframe = train_dataframe.reindex(np.random.permutation(train_dataframe.index))

train_dataframe.info()
# know the dataset

train_dataframe.describe()
# Function for nullanalysis

def nullAnalysis(df):

    tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})



    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))

    tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100)

                         .T.rename(index={0:'null values (%)'}))

    return tab_info



# Show the null values

nullAnalysis(train_dataframe)



# Ageの欠損が多い・・。
def data_convert(dataset):

    # カテゴリカル変数を数値に変換

    # Pclass

    pclass_encoded = pd.get_dummies(dataset.Pclass, prefix=dataset.Pclass.name,prefix_sep="_")



    # Sex

    sex_encoded = pd.get_dummies(dataset.Sex, prefix=dataset.Sex.name,prefix_sep="_")



    # Embarked

    embarked_encoded = pd.get_dummies(dataset.Embarked, prefix=dataset.Embarked.name,prefix_sep="_")



    # エンコードしたカテゴリカル変数をdataframeに追加

    dataset = dataset.join(pclass_encoded)

    dataset = dataset.join(sex_encoded)

    dataset = dataset.join(embarked_encoded)

    

    # エンコード前のカテゴリカル変数はdataframeから削除

    dataset = dataset.drop(columns="Pclass")

    dataset = dataset.drop(columns="Sex")

    dataset = dataset.drop(columns="Embarked")

    

    return dataset



# training dataset全体をコンバート実行

train_dataframe = data_convert(train_dataframe)

train_dataframe
train_dataframe.info()
# training setの相関行列

train_dataframe.corr()



# 相関を見ると、survivedと相関が高いのはPclassとFare.

# とりあえず、この２つはまずsurvive識別には有効と思われる。使用する。

# some groups of people were more likely to survive than others, such as women, children, and the upper-class.

# 説明文にと上記とあるように、女性、子供、アッパークラスは生存しやすかった。

# 性別と子供も特徴として加えよう。(sex, age)
# describe about surviver

train_dataframe_survivers = train_dataframe[train_dataframe["Survived"] == 1]

train_dataframe_survivers.describe()
# Ageを欠損値を平均値で埋める。（PClassごとの平均で分けるという手もある。）

train_dataframe['Age'] = train_dataframe['Age'].fillna(train_dataframe['Age'].mean()) 

display.display(train_dataframe.isnull().sum())

train_dataframe.describe()
def parse_labels_and_features(dataset, train=True):

    if train==True:

        # ラベルを取り出し

        labels = dataset[["Survived"]]

    

    numerical_features = dataset[["Age",

                               "Fare"]]

    

    one_hot_features = dataset[["Pclass_1",

                               "Pclass_2",

                               "Pclass_3",

                               "Sex_female",

                               "Sex_male",

                               "Embarked_C",

                               "Embarked_Q",

                               "Embarked_S"]]



    # Scale the data to [0, 1] by dividing out the max value, 255.

    # selected_features = selected_features / 255

    numerical_features = (numerical_features - numerical_features.mean()) / numerical_features.std()

    

    # join features

    selected_features = numerical_features.join(one_hot_features)

    

    if train==True:

        return labels, selected_features

    else:

        return selected_features
# re-index dataset

train_dataframe = train_dataframe.reindex(np.random.permutation(train_dataframe.index))



# parse target and features

train_targets, train_examples = parse_labels_and_features(train_dataframe, train=True)



# spilit training set and validation set

training_targets = train_targets[:700]

training_examples = train_examples[:700]

validation_targets = train_targets[700:891]

validation_examples = train_examples[700:891]

#training_targets, training_examples = train_dataframe[:700], train=True)

#validation_targets, validation_examples = parse_labels_and_features(train_dataframe[700:891], train=True)



#Double-check that we've done the right thing.

print("Training examples summary:")

display.display(training_examples.describe())

print("Validation examples summary:")

display.display(validation_examples.describe())



print("Training targets summary:")

display.display(training_targets.describe())

print("Validation targets summary:")

display.display(validation_targets.describe())
def construct_feature_columns(input_features):

    """Construct the TensorFlow Feature Columns.



    Args:

        input_features: The names of the numerical input features to use.

    Returns:

        A set of feature columns

    """ 

        

    return set([tf.feature_column.numeric_column(my_feature)

                for my_feature in input_features])
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):

    """Trains a neural net regression model.

  

    Args:

      features: pandas DataFrame of features

      targets: pandas DataFrame of targets

      batch_size: Size of batches to be passed to the model

      shuffle: True or False. Whether to shuffle the data.

      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely

    Returns:

      Tuple of (features, labels) for next data batch

    """

    

    # Convert pandas data into a dict of np arrays.

    features = {key:np.array(value) for key,value in dict(features).items()}                                             

 

    # Construct a dataset, and configure batching/repeating.

    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit

    ds = ds.batch(batch_size).repeat(num_epochs)

    

    # Shuffle the data, if specified.

    if shuffle:

      ds = ds.shuffle(10000)

    

    # Return the next batch of data.

    features, labels = ds.make_one_shot_iterator().get_next()

    return features, labels
def train_nn_classification_model(

    learning_rate,

    steps,

    batch_size,

    hidden_units,

    training_examples,

    training_targets,

    validation_examples,

    validation_targets):

    """Trains a neural network regression model.



    In addition to training, this function also prints training progress information,

    as well as a plot of the training and validation loss over time.



    Args:

    my_optimizer: An instance of `tf.train.Optimizer`, the optimizer to use.

    steps: A non-zero `int`, the total number of training steps. A training step

      consists of a forward and backward pass using a single batch.

    batch_size: A non-zero `int`, the batch size.

    hidden_units: A `list` of int values, specifying the number of neurons in each layer.

    training_examples: A `DataFrame` containing one or more columns from

      `california_housing_dataframe` to use as input features for training.

    training_targets: A `DataFrame` containing exactly one column from

      `california_housing_dataframe` to use as target for training.

    validation_examples: A `DataFrame` containing one or more columns from

      `california_housing_dataframe` to use as input features for validation.

    validation_targets: A `DataFrame` containing exactly one column from

      `california_housing_dataframe` to use as target for validation.



    Returns:

    A tuple `(estimator, training_losses, validation_losses)`:

      estimator: the trained `DNNRegressor` object.

      training_losses: a `list` containing the training loss values taken during training.

      validation_losses: a `list` containing the validation loss values taken during training.

    """



    periods = 10

    steps_per_period = steps / periods



    # Create a DNNRegressor object.

    my_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    #my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    classifier = tf.estimator.DNNClassifier(

      feature_columns=construct_feature_columns(training_examples),

      hidden_units=hidden_units,

      optimizer=my_optimizer

    )



    # Create input functions.

    training_input_fn = lambda: my_input_fn(training_examples, 

                                          training_targets,

                                          num_epochs=50, 

                                          batch_size=batch_size)

    predict_training_input_fn = lambda: my_input_fn(training_examples, 

                                                  training_targets,

                                                  num_epochs=1, 

                                                  shuffle=False)

    predict_validation_input_fn = lambda: my_input_fn(validation_examples, 

                                                    validation_targets,

                                                    num_epochs=1, 

                                                    shuffle=False)



    # Train the model, but do so inside a loop so that we can periodically assess

    # loss metrics.

    print("Training model...")

    print("LogLoss error (on validation data):")

    training_errors = []

    validation_errors = []

    for period in range (0, periods):

        # Train the model, starting from the prior state.

        classifier.train(

            input_fn=training_input_fn,

            steps=steps_per_period

        )

        

        # Take a break and compute probabilities.

        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))

        #print(training_predictions)

        raining_probabilities = np.array([item['probabilities'] for item in training_predictions])

        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])

        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id,2)

        

        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))

        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])    

        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])

        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id,2)    



        # Compute training and validation errors.

        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)

        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)

        # Occasionally print the current loss.

        print("  period %02d : %0.2f" % (period, validation_log_loss))

        # Add the loss metrics from this period to our list.

        training_errors.append(training_log_loss)

        validation_errors.append(validation_log_loss)

        

    print("Model training finished.")

    # Remove event files to save disk space.

    #_ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))



    # Calculate final predictions (not probabilities, as above).

    final_predictions = classifier.predict(input_fn=predict_validation_input_fn)

    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])





    accuracy = metrics.accuracy_score(validation_targets, final_predictions)

    print("Final accuracy (on validation data): %0.2f" % accuracy)



    # Output a graph of loss metrics over periods.

    plt.ylabel("LogLoss")

    plt.xlabel("Periods")

    plt.title("LogLoss vs. Periods")

    plt.plot(training_errors, label="training")

    plt.plot(validation_errors, label="validation")

    plt.legend()

    plt.show()



    # Output a plot of the confusion matrix.

    cm = metrics.confusion_matrix(validation_targets, final_predictions)

    # Normalize the confusion matrix by row (i.e by the number of samples

    # in each class).

    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    ax = sns.heatmap(cm_normalized, cmap="bone_r")

    ax.set_aspect(1)

    plt.title("Confusion matrix")

    plt.ylabel("True label")

    plt.xlabel("Predicted label")

    plt.show()



    return classifier
classifier = train_nn_classification_model(

    learning_rate=0.005,

    steps=50000,

    #steps=3,

    batch_size=50,

    hidden_units=[20, 5],

    training_examples=training_examples,

    training_targets=training_targets,

    validation_examples=validation_examples,

    validation_targets=validation_targets)
# import test dataset

test_dataframe = pd.read_csv("../input/test.csv",sep=",")

test_dataframe.head()



# Data Convert

test_dataframe = data_convert(test_dataframe)



# Data Cleaning

test_dataframe['Age'] = test_dataframe['Age'].fillna(test_dataframe['Age'].mean()) 

test_dataframe['Fare'] = test_dataframe['Fare'].fillna(test_dataframe['Fare'].mean()) 



test_examples = parse_labels_and_features(test_dataframe, train=False)

#test_examples = test_dataframe["Pclass"]

#Double-check that we've done the right thing.

print("Test examples summary:")

display.display(test_examples.describe())
def my_test_input_fn(features, batch_size=1, shuffle=False, num_epochs=None):

    """Trains a neural net regression model.

  

    Args:

      features: pandas DataFrame of features

      targets: pandas DataFrame of targets

      batch_size: Size of batches to be passed to the model

      shuffle: True or False. Whether to shuffle the data.

      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely

    Returns:

      Tuple of (features, labels) for next data batch

    """

    

    # Convert pandas data into a dict of np arrays.

    features = {key:np.array(value) for key,value in dict(features).items()}                                             

 

    # Construct a dataset, and configure batching/repeating.

    ds = Dataset.from_tensor_slices(features) # warning: 2GB limit

    ds = ds.batch(batch_size).repeat(num_epochs)



    # Shuffle the data, if specified.

    if shuffle:

      ds = ds.shuffle(10000)

    

    # Return the next batch of data.

    #features, labels = ds.make_one_shot_iterator().get_next()

    features = ds.make_one_shot_iterator().get_next()

    

    return features

predict_test_input_fn = lambda: my_test_input_fn(test_examples, num_epochs=1)



print("Start prediction")

final_predictions = list(classifier.predict(input_fn=predict_test_input_fn))



final_probabilities = np.array([item['probabilities'] for item in final_predictions])    

final_pred_class_id = np.array([item['class_ids'][0] for item in final_predictions])



#書き出しの前にチェック

#test_dataframe["PassengerId"].shape

#test_dataframe["PassengerId"].head()



#回答をcsvに書き出し

print("create submission file")

my_submission = pd.DataFrame({'PassengerId': test_dataframe["PassengerId"], 'Survived': final_pred_class_id})

print(my_submission)



my_submission.to_csv('submission.csv', index=False)