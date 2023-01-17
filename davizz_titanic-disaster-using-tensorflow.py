# I have followed this great kernel has a guide: https://www.kaggle.com/carlbeckerling/kaggle-titanic-tutorial/notebook

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import tensorflow as tf

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# let's load the data into some variables,for that we use the pandas library
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print(train.keys())

# train_x will have train except PassengerId column
# train_y will have the survived 
train_x, train_y = train, train.pop("Survived")

# same as above
# test is used to evaluate the model, for the time being I am not doing that because I just want to do a submission file to see the whole process
# TODO: do the evaluation of the model to check the accuracy
# test_x, test_y = test, test.pop("Survived")
print(train_x.keys())
# let's define the feature columns, for sure there is a best practice here, but this is my first Notebook so I will add a TODO here.
# TODO: Look for a good strategy to determine which columns are good candidates to be Feature Columns, for the time being I am using common sense :)

# I will set as feature column the Pclass, Sex and Age columns as I think those features where important in the time of the sinking.
# Let's explore the data of these columns to look for nulls.

train_x['Pclass'].describe()
train_x['Sex'].describe()
train_x['Age'].describe()
# We see there are 714 over 891 rows with that field available so we need to fill up those. 
# I am gonna use the method of replacing with Mean but I set another TODO here
# TODO: Use an alternate method and check which one gives better predictions: https://www.analyticsindiamag.com/5-ways-handle-missing-values-machine-learning-datasets/

age_mean = train_x['Age'].mean()
train_x['Age'].fillna(age_mean, inplace=True)
# Pclass is an ordinal?? column, so we need to do some work here.
# Let's create three boolean columns

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

train_x = create_dummies(train_x,"Pclass")
train_x.head()

# Same for Sex column
train_x = create_dummies(train_x,"Sex")
train_x.head()
# We need to do the same for the test data
test = create_dummies(test,"Pclass")
test = create_dummies(test,"Sex")

# Now is time to remove all the columns we don't need, so we are keeping only the feature branches
train_final = train_x[['Age','Pclass_1','Pclass_2','Pclass_3', 'Sex_female', 'Sex_male']]
train_final.head()
my_feature_columns = []
for key in train_final.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

print(my_feature_columns)
# Build 2 hidden layer DNN with 10, 10 units respectively.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=2)
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    # shuffle the dataset is very important to avoid overfitting.
    # one epoch = one forward pass and one backward pass of all the training examples
    # batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
    # number of iterations = number of passes, each pass using [batch size] number of examples. To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).
    # Example: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset


# Train the Model.
batch_size = 100
train_steps = 1000
classifier.train(
    input_fn=lambda:train_input_fn(train_final, train_y, batch_size),
    steps=train_steps)
train_final.head()
predictions = test[['Age','Pclass_1','Pclass_2','Pclass_3','Sex_female','Sex_male']]

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


prediction_final = classifier.predict(
        input_fn=lambda:eval_input_fn(predictions,
                                      labels=None,
                                      batch_size=batch_size))
result = []
for pred_dict in prediction_final:
    template = ('\nPrediction is "{}" ({:.1f}%)')

    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    result.append(probability)
    print(probability)
holdout_ids = test["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": result}
submission = pd.DataFrame(submission_df)

submission.to_csv('titanic_submission.csv',index=False)