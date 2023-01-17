import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

from sklearn.linear_model import LogisticRegression

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

train_df = pd.read_csv("../input/train.csv")
# train_df = titanic_df.reindex(np.random.permutation(titanic_df.index))
test_df = pd.read_csv("../input/test.csv")

train_df.drop(["PassengerId", "Ticket", "Cabin"], axis=1, inplace=True )
test_df.drop(["Ticket", "Cabin"], axis=1, inplace=True )
combined_df = [train_df, test_df]
"""SET TITLES"""

def process_titles(input_df):

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    for df in input_df:
        df["Title"] = df["Name"].str.extract(" ([A-Za-z]+)\.", expand=False)
        df["Title"] = df["Title"].replace([
            "Lady",
            "Countess",
            "Capt",
            "Col",
            "Don",
            "Dr",
            "Major",
            "Rev",
            "Sir",
            "Jonkheer",
            "Dona"
        ], "Rare")
        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')
        
        # Fill null
        df['Title'] = df['Title'].fillna(0)
        
        # Buckets
#         passenger["Title"] = passenger["Title"].map(title_mapping)
"""IS MAN OR WOMAN"""

def process_sex(input_df):
    
    sex_mapping = {"female": 1, "male": 0}
    
    for df in input_df:
        df["Sex"] = df["Sex"].map({"female": 1, "male": 0}).astype(int)
"""GUESS AGES"""

def process_age(input_df):
    age_sets = np.zeros((2,3))
    
#     sex_map = {"male": 0, "female": 1}

    for df in input_df:
        
        # Build age_sets guesser
        for i in range(2):
            for j in range(1,4):
#                 sex = 0 if passenger["Sex"] == "male" else 1
                guess = df[(df["Sex"] == i) & (df["Pclass"] == j)]["Age"].dropna()
                
                age_guess = guess.median()
                age_sets[i,j-1] = int(age_guess/0.5 + 0.5) * 0.5

        # Assign guessed age
        for i in range(2):
            for j in range(1,4):
                df.loc[
                    (df["Age"].isnull()) &
                    (df["Sex"] == i) &
                    (df["Pclass"] == j),
                    "Age"] = age_sets[i,j-1]

        df["Age"] = df["Age"].astype(int)

        # Buckets
        df.loc[ df["Age"] <= 16, "Age" ] = 1
        df.loc[ (df["Age"] > 16) & (df["Age"] <= 32), "Age" ] = 2
        df.loc[ (df["Age"] > 32) & (df["Age"] <= 48), "Age" ] = 3
        df.loc[ (df["Age"] > 48) & (df["Age"] <= 64), "Age" ] = 4
        df.loc[ df["Age"] > 64, "Age" ] = 5
"""DETERMINE SOLO STATUS"""

def process_solo_status(input_df):

    for df in input_df:
        df["FamilySize"] = 0
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    for df in input_df:
        df["Solo"] = 0
        df.loc[df["FamilySize"] == 1, "Solo"] = 1

    train_df.drop(["FamilySize"], axis=1, inplace=True)
    test_df.drop(["FamilySize"], axis=1, inplace=True)
    combined_df = [train_df, test_df]
"""FILL MISSING EMBARKED PORTS"""

def process_embarked(input_df):

    port_mapping = {'S': 0, 'C': 1, 'Q': 2}

    for df in input_df:
        df["Embarked"] = df["Embarked"].fillna("S")
        # Buckets
#         passenger["Embarked"] = passenger["Embarked"].map(port_mapping).astype(int)
"""AGE X PCLASS"""

def cross_age_and_pclass(input_df):

    for df in input_df:
        df["AgeXClass"] = df["Age"] * df["Pclass"]
"""HANDLE FARES"""

def process_fares():

    #FILL THAT MISSING FARE IN TESTING DATAFRAME
    test_df["Fare"].fillna(test_df["Fare"].dropna().median(), inplace=True)

#     # Buckets
#     for passenger in combined_df:
#         passenger.loc[passenger["Fare"] <= 7.91, "Fare"] = 0
#         passenger.loc[(passenger["Fare"] > 7.91) & (passenger["Fare"] <= 14.454), "Fare"] = 1
#         passenger.loc[(passenger["Fare"] > 14.454) & (passenger["Fare"] <= 31), "Fare"] = 2
#         passenger.loc[passenger["Fare"] > 31, "Fare"] = 3
#         passenger['Fare'] = passenger['Fare'].astype(int)
"""FEATURE COLUMNS"""

def construct_feature_columns():
    
    title = tf.feature_column.categorical_column_with_vocabulary_list(
        key="Title",
        vocabulary_list=["Mr", "Miss", "Mrs", "Master", "Rare"],
    )
    
#     sex = tf.feature_column.categorical_column_with_vocabulary_list(
#         key="Sex",
#         vocabulary_list=["male", "female"],
#     )    
    
    sex = tf.feature_column.categorical_column_with_identity(
        key="Sex",
        num_buckets=2,
        default_value=0,
    )
    
#     age = tf.feature_column.bucketized_column(
#         tf.feature_column.numeric_column(key="Age"),
#         boundaries=[16, 32, 48, 64],
#     )    

    age = tf.feature_column.categorical_column_with_identity(
        key="Age",
        num_buckets=5,
        default_value=0,
    )
    
    pclass = tf.feature_column.categorical_column_with_identity(
        key="Pclass",
        num_buckets=3,
        default_value=0,
    )
    
    solo = tf.feature_column.categorical_column_with_identity(
        key="Solo",
        num_buckets=2,
        default_value=0,
    )
    
    embarked = tf.feature_column.categorical_column_with_vocabulary_list(
        key="Embarked",
        vocabulary_list=["C", "Q", "S"],
    )
    
    ageXpclass = tf.feature_column.categorical_column_with_identity(
        key="AgeXClass",
#         num_buckets=6,
        num_buckets=10,
        default_value=0,
    )
    
    feature_columns = set([
        title,
        sex,
        age,
        pclass,
        solo,
        embarked,
        ageXpclass,        
    ])
    
    return feature_columns
"""PREPROCESS AND DROP"""

def preprocess_everything(input_df):
    process_titles(input_df)
    process_sex(input_df)
    process_age(input_df)
    process_solo_status(input_df)
    process_embarked(input_df)
    process_fares()
    cross_age_and_pclass(input_df)
    
def drop_features():    
    train_df.drop(["Name", "Parch", "SibSp"], axis=1, inplace=True)
    test_df.drop(["Name", "Parch", "SibSp"], axis=1, inplace=True)    
    combined_df = [train_df, test_df]
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    
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
def train_linear_classifier_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
    
    periods = 10
    steps_per_period = steps / periods
    
    # Create a linear classifier object.
    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(
        feature_columns=construct_feature_columns(),
        optimizer=my_optimizer
    )
    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets,
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
    print("LogLoss (on training data):")
    training_log_losses = []
    validation_log_losses = []
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        linear_classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.    
        training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

        validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_log_loss))
        # Add the loss metrics from this period to our list.
        training_log_losses.append(training_log_loss)
        validation_log_losses.append(validation_log_loss)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    plt.plot(validation_log_losses, label="validation")
    plt.legend()

    return linear_classifier
def train_linear_classifier_model_no_validation(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,):
    
    periods = 10
    steps_per_period = steps / periods
    
    # Create a linear classifier object.
#     my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#     my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(
        feature_columns=construct_feature_columns(),
        optimizer=my_optimizer
    )
    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets,
                                          batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets,
                                                  num_epochs=1,
                                                  shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss (on training data):")
    training_log_losses = []
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        linear_classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.    
        training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_log_loss))
        # Add the loss metrics from this period to our list.
        training_log_losses.append(training_log_loss)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.tight_layout()
    plt.plot(training_log_losses, label="training")
    plt.legend()

    return linear_classifier
"""LET'S GET THIS PARTY STARTED"""

preprocess_everything(combined_df)
drop_features()
train_df.head(20)
# test_df.describe()
train_df = train_df.reindex(np.random.permutation(train_df.index))

# training_examples = train_df.head(670).drop(["Survived"], axis=1)
# training_targets = train_df.head(670)["Survived"]

# validation_examples = train_df.tail(220).drop(["Survived"], axis=1)
# validation_targets = train_df.tail(220)["Survived"]

# linear_classifier = train_linear_classifier_model(
#     learning_rate=0.0075,
#     steps=20000,
#     batch_size=30,
#     training_examples=training_examples,
#     training_targets=training_targets,
#     validation_examples=validation_examples,
#     validation_targets=validation_targets
# )

# predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
#                                                   validation_targets, 
#                                                   num_epochs=1, 
#                                                   shuffle=False)

# evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)
# print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
# print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])


training_examples = train_df.drop(["Survived"], axis=1)
training_targets = train_df["Survived"]

linear_classifier = train_linear_classifier_model_no_validation(
    learning_rate=0.0095,
#     learning_rate=0.0075,
    steps=7500,
    batch_size=60,
    training_examples=training_examples,
    training_targets=training_targets,
)

predict_validation_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets, 
                                                  num_epochs=1, 
                                                  shuffle=False)

evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)
print("AUC on the TRAINING set: %0.2f" % evaluation_metrics['auc'])
print("Accuracy on the TRAINING set: %0.2f" % evaluation_metrics['accuracy'])
test_df["Survived"] = pd.Series([])
test_examples = test_df
test_targets = test_df["Survived"]

predict_testing_input_fn = lambda: my_input_fn(test_examples, 
                                               test_targets, 
                                               num_epochs=1, 
                                               shuffle=False
                                              )

test_predictions = linear_classifier.predict(input_fn=predict_testing_input_fn)
test_df["Survived"] = np.array([int(item['classes']) for item in test_predictions])
print(test_df["Survived"])

test_df.to_csv(
   path_or_buf="solution.csv",
   index=False,
   columns=["PassengerId", "Survived"]
)
# X_train = train_df.drop("Survived", axis=1)
# Y_train = train_df["Survived"]
# X_test  = test_df.copy()
# X_train.shape, Y_train.shape, X_test.shape
# logreg = LogisticRegression()
# logreg.fit(X_train, Y_train)
# Y_pred = logreg.predict(X_test)
# acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
# acc_log