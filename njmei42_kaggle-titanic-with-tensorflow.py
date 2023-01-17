# General Python Utils
from collections import Counter
import gc
gc.enable()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scikit-learn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# Tensorflow
import tensorflow as tf
from tensorflow.python.data import Dataset
tf.logging.set_verbosity(tf.logging.ERROR)
# Okay, let's load in our datasets!
raw_train_df = pd.read_csv("../input/train.csv")
raw_test_df = pd.read_csv("../input/test.csv")
example_submission_df = pd.read_csv("../input/gender_submission.csv")

train_df = raw_train_df.copy(deep=True)
test_df = raw_test_df.copy(deep=True)
train_test_lst = [train_df, test_df]
# Taking a look at the first few values in the dataframe
display(train_df.head())
# Taking a look at the summary statistics for each feature
display(train_df.describe())
train_df[train_df['Fare'] == 0]
display(test_df.head())
display(test_df.describe())
display(train_df.isnull().sum())
print("Total individuals in train set is: {}".format(len(train_df)))
display(test_df.isnull().sum())
print("Total individuals in test set is: {}".format(len(test_df)))
# Let's only consider data that has non-NaN Cabin values (Age or Embarked can still be NaN!)
cabin_df = train_df[train_df['Cabin'].notnull()]

# Let's create a new feature 'deck_level' that groups passengers by deck levels
cabin_df = cabin_df.assign(deck_level=pd.Series([entry[:1] for entry in cabin_df['Cabin']]).values)
display(cabin_df.head())

print("Survival chances based on deck level:")
cabin_df.groupby(['deck_level'])['Survived'].mean()

def process_deck_level(train_test_lst):
    new = []
    for dataset in train_test_lst:
        dataset = dataset.copy(deep=True)
        # Take the first letter of the Cabin entry if it's not nan. Otherwise, it should be labelled as 'U'.
        dataset = dataset.assign(deck_level=pd.Series([entry[:1] if not pd.isnull(entry) else 'U' for entry in dataset['Cabin']]))
        # Okay, now let's drop the Cabin column from our dataset
        dataset = dataset.drop(['Cabin'], axis = 1)
        new.append(dataset)
    return (new)

train_df, test_df = process_deck_level(train_test_lst)

# Let's check that we did the right thing...
display(train_df.head())
display(test_df.head())
# Let's also recheck what's still missing
display(train_df.isnull().sum())
display(test_df.isnull().sum())
train_df.groupby(['Pclass', 'deck_level']).size().unstack(0).plot.bar(stacked=True)
plt.title("Histogram of deck_level grouped by Pclass")
_ = plt.ylabel("Frequency")
display(set(train_df['Embarked']))
print("Survival chances based on embarcation:")
train_df.groupby(['Embarked'])['Survived'].mean()
# Replace NaN values in the 'Embarked' column with 'N'
train_df[['Embarked']] = train_df[['Embarked']].fillna('N')
# Let's check that we filled things correctly!
display(set(train_df['Embarked']))
display(train_df.isnull().sum())
test_df[test_df['Fare'].isnull()]
Pclass_Fare_grouping = test_df.groupby(["Pclass"])['Fare']
train_df.groupby(['Pclass', pd.cut(train_df['Fare'], np.arange(0, 701, 5))]).size().unstack(0).plot.bar(stacked=True, title = 'Fare histogram grouped by Pclass')
plt.xlabel('Fare')
plt.ylabel('Frequency')
print("Mean Fare for each Pclass:")
display(Pclass_Fare_grouping.mean())
print("Median Fare for each Pclass:")
display(Pclass_Fare_grouping.median())
test_df[['Fare']] = test_df[['Fare']].fillna(Pclass_Fare_grouping.median()[3])
# Let's check that our one fill worked!
display(test_df[test_df['PassengerId'] == 1044])
display(test_df.isnull().sum())
ax = train_df[['Age']].plot(kind='hist', bins=20)
plt.xlabel("Age")
_ = plt.title("Age histogram")
train_df.groupby(['Survived', pd.cut(train_df['Age'], np.arange(0, 100, 5))]).size().unstack(0).plot.bar(stacked=True, alpha=0.75)
_ = plt.title("Age histogram grouped by survival")
train_df.groupby(['Survived', 'Sex', pd.cut(train_df['Age'], np.arange(0, 100, 5))]).size().unstack(0).plot.bar(stacked=True, alpha=0.75)
plt.title("Age histogram grouped by survival and gender")
plt.tight_layout()
# All name formats seem to be something like:
# "last_name, title. first_name "nickname" (full_name)"
# To get title, we split the string by comma and select the second half. Then we split that second half by '.' and take the first half
# i.e.
# 1) ["last_name", "title. first_name "nickname" (full_name)"] (select element 1!)
# 2) ["title", "first_name "nickname" (full_name)"] (select element 0!)
train_titles = [name.split(',')[1].lstrip(' ').split('.')[0] for name in train_df['Name']]
# Let's see if the above strategy works
print("Train set titles (and counts):")
print(Counter(train_titles))

print("\nTest set titles (and counts):")
test_titles = [name.split(',')[1].lstrip(' ').split('.')[0] for name in test_df['Name']]
print(Counter(test_titles))

print("\n===============================")

age_missing_train_titles = [name.split(',')[1].lstrip(' ').split('.')[0] for name in train_df[train_df['Age'].isnull()]['Name']]
print("\nTrain set titles (and counts) with missing ages:")
print(Counter(age_missing_train_titles))

age_missing_test_titles = [name.split(',')[1].lstrip(' ').split('.')[0] for name in test_df[test_df['Age'].isnull()]['Name']]
print("\nTest set titles (and counts) with missing ages:")
print(Counter(age_missing_test_titles))
# Let's add the titles as a new feature for our dataset
def naive_process_title(train_test_lst):
    new = []
    for dataset in train_test_lst:
        dataset = dataset.copy(deep=True)
        titles = [name.split(',')[1].lstrip(' ').split('.')[0] for name in dataset['Name']]
        dataset = dataset.assign(title=pd.Series(titles).values)
        new.append(dataset)
    return (new)

train_df, test_df = naive_process_title([train_df, test_df])

# Taking a look at our dataframes to make sure we did the right thing...
display(train_df.head())
display(test_df.head())
def plot_title_age_hist(title, train_df, bins=20):
    title_ages = train_df[train_df['title'] == title]['Age']
    title_ages.plot(kind='hist', bins=bins, legend=True)
    title_ages.describe()
    plt.xlabel("Age")
    plt.title("Age histogram for '{}' title".format(title))
title_groups = train_df.groupby(['title'])
display(title_groups['Age'].describe())
plot_title_age_hist("Master", train_df, bins=10)
plot_title_age_hist('Miss', train_df)
def title_feature_age_analysis(title, feature, train_df):
    # Let's loop through all values of our feature of interest (in this case "Parch")
    title_df =train_df[(train_df['title'] == title)]
    title_df.groupby([feature, pd.cut(title_df['Age'], np.arange(0, 100, 5))]).size().unstack(0).plot.bar(stacked=True, alpha=0.75)
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    _ = plt.title("Age histogram for '{}' title grouped by {}".format(title, feature))
    for i in range(max(train_df[train_df['title'] == title][feature]) + 1):
        # Print common descriptive stats for our title and the given level of our feature
        print("Statistics for '{}' title with {} of: {}".format(title, feature, i))
        display(train_df[(train_df['title'] == title) & (train_df[feature] == i)]['Age'].describe())
        print("Median\t{}\n".format(train_df[(train_df['title'] == title) & (train_df[feature] == i)]['Age'].median()))
        print("=========================\n")

title_feature_age_analysis('Miss', 'Parch', train_df)
title_feature_age_analysis('Miss', 'Parch', test_df)
title_feature_age_analysis('Mrs', "Parch", train_df)
title_feature_age_analysis('Mr', "Parch", train_df)
# Code to fill in missing NaN datapoints based on title analysis. No longer used but it's here for those interested in using it.
# def age_imputer(train_test_lst):
#     new = []
#     for dataset in train_test_lst:
#         dataset = dataset.copy(deep=True)
#         # This is the list of unique titles for individuals with a NaN age
#         missing_age_titles = list(set([name.split(',')[1].lstrip(' ').split('.')[0] for name in dataset[dataset['Age'].isnull()]['Name']]))
#         print("Titles for individuals with missing age are: {}".format(missing_age_titles))
#         for title in missing_age_titles:
#             # Fill in missing ages for 'Mr'/'Mrs'/'Master'/'Ms'/'Dr' titles
#             if (title in ['Mr', 'Mrs', 'Master', 'Ms', 'Dr']):
#                 median = dataset[(dataset['title'] == title)]['Age'].median()
#                 # Treat 'Ms' as 'Mrs'
#                 if (title == 'Ms'):
#                     median = dataset[(dataset['title'] == 'Mrs')]['Age'].median()
#                 dataset[(dataset['title'] == title) & (dataset['Age'].isnull())] = dataset[(dataset['title'] == title) & (dataset['Age'].isnull())].fillna(median)
#             # Fill in missing ages for "Miss" titles
#             elif (title == 'Miss'):
#                 for level in range(max(dataset[dataset['title'] == title]['Parch']) + 1):
#                     df = dataset[(dataset['title'] == 'Miss') & (dataset['Age'].isnull()) & (dataset['Parch'] == level)]
#                     if (not df.empty):
#                         median = dataset[(dataset['title'] == title) & (dataset['Parch'] == level)]['Age'].median()
#                         dataset[(dataset['title'] == 'Miss') & (dataset['Age'].isnull()) & (dataset['Parch'] == level)] = dataset[(dataset['title'] == 'Miss') & (dataset['Age'].isnull()) & (dataset['Parch'] == level)].fillna(median)
#         new.append(dataset)
#     return (new)

# train_df, test_df = age_imputer([train_df, test_df])
# display(train_df.isnull().sum())
# display(test_df.isnull().sum())
# display(raw_train_df[raw_train_df['Age'].isnull()])
# # Select passengers that have NaN ages in our raw_train_df
# train_df.loc[train_df['PassengerId'].isin(raw_train_df[raw_train_df['Age'].isnull()]['PassengerId'])]
# fig, (ax1, ax2) = plt.subplots(1, 2)
# # First column plot
# train_df[['Age']].plot(kind='hist', bins=20, ax=ax1, legend=False)
# ax1.set_xlabel("Age")
# ax1.set_title("NaN filled")
# ymin, ymax = ax1.get_ylim()
# # Second column plot
# raw_train_df[['Age']].plot(kind='hist', bins=20, ax=ax2, sharey=True, legend=False)
# ax2.set_ylim(ymin, ymax)
# ax2.set_xlabel("Age")
# _ = ax2.set_title("Original distribution")
raw_train_df[raw_train_df['Name'].str.startswith('Sage,')]
# Let's add family_size which is just the sum of 'SibSp' and 'Parch'
train_df['family_size'] = train_df['SibSp'] + train_df['Parch']
test_df['family_size'] = test_df['SibSp'] + test_df['Parch']
# Check that things were added properly
display(train_df.head())
# Plot family size grouped by survival
train_df.groupby(['Survived', pd.cut(train_df['family_size'], np.arange(0, 11))]).size().unstack(0).plot.bar(stacked=True)
plt.xlabel("Family Size")
_ = plt.title("Histogram of family size grouped by survival")
train_df.groupby(['Survived', 'Pclass', pd.cut(train_df['family_size'], np.arange(0, 11))]).size().unstack(0).plot.bar(stacked=True)
plt.ylabel("Frequency")
_ = plt.title("Histogram of Pclass x family size grouped by survival")
train_df.groupby(['Survived', pd.cut(train_df['Pclass'], np.arange(0, 4))]).size().unstack(0).plot.bar(stacked=True)
_ = plt.title("Histogram of Pclass grouped by survival")
train_df.groupby(['Survived', 'Sex', pd.cut(train_df['Pclass'], np.arange(0, 4))]).size().unstack(0).plot.bar(stacked=True)
_ = plt.title("Histogram of gender x Pclass grouped by survival")
plt.tight_layout()
train_df.groupby(['Survived', 'Sex', pd.cut(train_df['Age'], np.arange(0, 80, 10))]).size().unstack(0).plot.bar(stacked=True)
_ = plt.title("Histogram of gender x Age grouped by survival")
plt.tight_layout()
train_df.groupby(['Survived', 'Sex', 'Pclass', pd.cut(train_df['Age'], np.arange(0, 80, 10))]).size().unstack(0).plot.bar(stacked=True)
_ = plt.title("Histogram of gender x Pclass x Age grouped by survival")
plt.tight_layout()
# To get things to work nicely with tensorflow we'll need to subtract one from 'Pclass' so our classes start at 0
train_df['Pclass'] = train_df['Pclass'] - 1
test_df['Pclass'] = test_df['Pclass'] - 1
# One last check over all our data
train_df
# Let's remind ourselves of the data columns we have
train_df.columns
def build_feature_columns(use_age=False):
    """
    Build our tensorflow feature columns!
    
    For a great overview of the different feature columns in tensorflow and when to use them, see:
    https://www.tensorflow.org/versions/master/get_started/feature_columns
    
    We'll build a set of wide features (when we need to learn feature interactions) as well as deep features (when we need generalization)
    """
    # ======== Basic features =========
    # Categorical features
    Pclass = tf.feature_column.categorical_column_with_identity("Pclass", num_buckets = 3)
    Sex = tf.feature_column.categorical_column_with_vocabulary_list("Sex", ["female", "male"])
    Embarked = tf.feature_column.categorical_column_with_vocabulary_list("Embarked", ["C", "N", "Q", "S"])
    #Name = tf.feature_column.categorical_column_with_hash_bucket("Name", hash_bucket_size = 100)
    Ticket = tf.feature_column.categorical_column_with_hash_bucket("Ticket", hash_bucket_size = 400)
    
    # Continuous features
    SibSp = tf.feature_column.numeric_column("SibSp")
    Parch = tf.feature_column.numeric_column("Parch")
    Fare = tf.feature_column.numeric_column("Fare")
    if (use_age):
        Age = tf.feature_column.numeric_column("Age")
    
    # ======== Engineered features =======
    # Basic engineered features
    #deck_level = tf.feature_column.categorical_column_with_vocabulary_list("deck_level", ["A", "B", "C", "D", "E", "F", "G", "T", "U"])
    family_size = tf.feature_column.numeric_column("family_size")
    title = tf.feature_column.categorical_column_with_hash_bucket("title", hash_bucket_size = 10)
    
    # Bucketed features
    fare_buckets = tf.feature_column.bucketized_column(Fare, boundaries=list(range(0, 200, 5)))
    family_size_buckets = tf.feature_column.bucketized_column(family_size, boundaries=[1, 2, 3, 4, 5, 6, 7])
    Parch_buckets = tf.feature_column.bucketized_column(Parch, boundaries=[1,2,3,4,5,6])
    SibSp_buckets = tf.feature_column.bucketized_column(SibSp, boundaries=[1,2,3,4,5,6])
    
    # Crossed features
    Pclass_x_Sex = tf.feature_column.crossed_column(keys = [Pclass, Sex], hash_bucket_size = 10)
    Pclass_x_family_size = tf.feature_column.crossed_column(keys = [Pclass, family_size_buckets], hash_bucket_size = 30)
    Pclass_x_Sex_x_Embarked = tf.feature_column.crossed_column(keys = [Pclass, Sex, Embarked], hash_bucket_size = 30)
    
    # Age features
    if (use_age):
        age_buckets = tf.feature_column.bucketized_column(Age, boundaries=[5, 15, 25, 35, 45, 55, 65])
        Pclass_x_Sex_x_age_buckets = tf.feature_column.crossed_column(keys = [Pclass, Sex, age_buckets], hash_bucket_size = 100)
    
    # =========== Putting together wide features =================
    wide_features = set([Pclass, Sex, Ticket, Embarked, title, fare_buckets, Parch_buckets, SibSp_buckets, family_size_buckets,
                         Pclass_x_Sex, Pclass_x_family_size, Pclass_x_Sex_x_Embarked])
    if (use_age):
        wide_features |= set([age_buckets, Pclass_x_Sex_x_age_buckets])
    
    # =========== Putting together deep features =================
    deep_features = set([SibSp, Parch, Fare, family_size,
                         tf.feature_column.indicator_column(Pclass),
                         tf.feature_column.indicator_column(Sex),
                         tf.feature_column.indicator_column(title),
                         tf.feature_column.indicator_column(Embarked),
                         tf.feature_column.embedding_column(Ticket, dimension = 10)])
    if (use_age):
        deep_features |= set([Age])

    return((wide_features, deep_features))
def input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """
    This is our input function that will pass data into the tensorflow DNN class we'll create.
    It takes in a pandas dataframe.
    It outputs a tensorflow dataset one_shot_iterator
    """
    # Convert pandas df to dict of numpy arrays
    features = {key:np.array(value) for key, value in dict(features).items()}
    # Put together the tensorflow dataset. Configures batching/repeating.
    dataset = Dataset.from_tensor_slices((features, targets))
    dataset = dataset.batch(batch_size).repeat(num_epochs)
    # Shuffle data
    if (shuffle):
        dataset = dataset.shuffle(buffer_size = 50000)
    features, labels = dataset.make_one_shot_iterator().get_next()
    return (features, labels)
## ============= Previous train_test_split that was used ===========
# train_ex_df = train_df.sample(frac=0.60)
# train_targ_series = train_ex_df['Survived']

# xval_ex_df = train_df.drop(train_ex_df.index)
# xval_targ_series = xval_ex_df['Survived']

# # Double check that we don't have any train_ex_df data in our xval_ex_df data
# assert(not any(train_ex_df["PassengerId"].isin(xval_ex_df["PassengerId"])))

# # Select our variables of interest
# train_ex_df = train_ex_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "family_size"]]
# xval_ex_df = xval_ex_df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "family_size"]]

# print("Total training samples: {}".format(len(train_df)))
# print("New training split: {}".format(len(train_ex_df)))
# print("New xval split: {}".format(len(xval_ex_df)))
def plot_acc(train_accs, val_accs):
    fig, ax = plt.subplots(1, 1)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Period")
    ax.set_title("DNN model accuracy vs. Period")
    ax.plot(train_accs, label = "train")
    ax.plot(val_accs, label = "validation")
    ax.legend()
    fig.tight_layout()
    
    print("Final accuracy (train):\t\t{:.3f}".format(train_accs[-1]))
    print("Final accuracy (validation):\t{:.3f}\n".format(val_accs[-1]))

def train_wnd_classifier(periods, dnn_learning_rate, lin_learning_rate, steps, batch_size, hidden_units, train_ex, train_targ, val_ex, val_targ, use_age=False):
    #steps per period (spp)
    spp = steps / periods
    # We'll use the FTRL optimizer for our linear portion
    lin_optim = tf.train.FtrlOptimizer(learning_rate = lin_learning_rate)
    # We'll use the ProximalAdagradOptimizer with L1 regularization to punish overly complex deep models
    # We'll use L2 regularization to punish over-reliance on any one feature
    dnn_optim = tf.train.ProximalAdagradOptimizer(learning_rate = dnn_learning_rate,
                                                  l1_regularization_strength = 0.05,
                                                  l2_regularization_strength = 0.05)

    wide_features, deep_features = build_feature_columns(use_age)
    # We'll use a wide-n-deep classifier to get model that can memorize and generalize
    wnd_classifier = tf.estimator.DNNLinearCombinedClassifier(
        #Wide settings
        linear_feature_columns = wide_features,
        linear_optimizer = lin_optim,
        #Deep settings
        dnn_feature_columns = deep_features,
        dnn_hidden_units = hidden_units,
        dnn_optimizer = dnn_optim,
        dnn_dropout = 0.5,
        dnn_activation_fn = tf.nn.leaky_relu)
    
    # Input functions
    train_input_fn = lambda: input_fn(train_ex, train_targ, batch_size = batch_size)
    pred_train_input_fn = lambda: input_fn(train_ex, train_targ, num_epochs = 1, shuffle = False)
    pred_val_input_fn = lambda: input_fn(val_ex, val_targ, num_epochs = 1, shuffle = False)
    #train and validation accuracy per period
    train_app = []
    val_app = []
    for period in range(periods):
        # Train our classifier
        wnd_classifier.train(input_fn = train_input_fn, steps = spp)
        # Check how our classifier does on training set after one period
        train_pred = wnd_classifier.predict(input_fn = pred_train_input_fn)
        train_pred = np.array([pred['class_ids'][0] for pred in train_pred])
        # Check how our classifier does on the validation set after one period
        val_pred = wnd_classifier.predict(input_fn = pred_val_input_fn)
        val_pred = np.array([pred['class_ids'][0] for pred in val_pred])
        # Calculate accuracy metrics
        train_acc = accuracy_score(train_targ, train_pred)
        val_acc = accuracy_score(val_targ, val_pred)
        print("period {} train acc: {:.3f}".format(str(period).zfill(3), train_acc))
        # Add our accuracies to running list
        train_app.append(train_acc)
        val_app.append(val_acc)
    print("\nTraining done!\n")
    plot_acc(train_app, val_app)
    return (wnd_classifier, train_app, val_app)
BASIC_MODEL = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked", "Ticket", "family_size", "title"]
AGE_MODEL = BASIC_MODEL + ["Age"]
# Unused features:
# deck_level
# Cabin
def run_kfold_analysis(train_df, use_age=False):
    kf = KFold(n_splits = 10, shuffle = True)
    if (use_age):
        # If we're training our age model then we need to drop all rows where 'Age' == NaN
        train_df = train_df.dropna(axis = 0)
        features_to_use = AGE_MODEL
    else:
        features_to_use = BASIC_MODEL
    split_indices = kf.split(train_df)

    fold_classifiers = []
    for curr_fold, (train_indices, xval_indices) in enumerate(split_indices):
        # Set up our train examples and targets
        train_ex_df = train_df.iloc[train_indices]
        train_targ_series = train_ex_df['Survived']
        # Set up our xval examples and targets
        xval_ex_df = train_df.iloc[xval_indices]
        xval_targ_series = xval_ex_df['Survived']
        # Select our variables of interest
        train_ex_df = train_ex_df[features_to_use]
        xval_ex_df = xval_ex_df[features_to_use]

        print("========= Current fold: {} =============".format(curr_fold))
        print("\nTotal training samples: {}".format(len(train_df)))
        print("New training split: {}".format(len(train_ex_df)))
        print("New xval split: {}\n".format(len(xval_ex_df)))

        classifier, train_perf, val_perf = train_wnd_classifier(periods = 10,
                                                                lin_learning_rate = 0.01,
                                                                dnn_learning_rate = 0.01,
                                                                steps = 6000,
                                                                batch_size = 15,
                                                                hidden_units = [200, 150, 100],
                                                                train_ex = train_ex_df,
                                                                train_targ = train_targ_series,
                                                                val_ex = xval_ex_df,
                                                                val_targ = xval_targ_series,
                                                                use_age = use_age)
        
        fold_classifiers.append((classifier, train_perf, val_perf))
    return(fold_classifiers)
print("\n================= Results of BASIC_MODEL ==================\n")
basic_model_results = run_kfold_analysis(train_df, use_age = False)
print("\n================== Results of AGE_MODEL ===================\n")
age_model_results = run_kfold_analysis(train_df, use_age = True)
# Set up our train examples and targets
final_train_ex_df = train_df
final_train_targ_series = final_train_ex_df['Survived']
# Dummy dataset that we won't actually consider or care about
dummy_df = final_train_ex_df
dummy_targ_series = final_train_targ_series
# Select features in our basic model
final_train_ex_df = final_train_ex_df[BASIC_MODEL]
dummy_df = dummy_df[BASIC_MODEL]

basic_classifier, basic_train_perf, _ = train_wnd_classifier(periods = 10,
                                                            lin_learning_rate = 0.01,
                                                            dnn_learning_rate = 0.01,
                                                            steps = 6000,
                                                            batch_size = 15,
                                                            hidden_units = [200, 150, 100],
                                                            train_ex = final_train_ex_df,
                                                            train_targ = final_train_targ_series,
                                                            val_ex = dummy_df,
                                                            val_targ = dummy_targ_series,
                                                            use_age = False)
# Set up our train examples and targets
final_train_ex_df = train_df.dropna(axis = 0)
final_train_targ_series = final_train_ex_df['Survived']
# Dummy dataset that we won't actually consider or care about
dummy_df = final_train_ex_df
dummy_targ_series = final_train_targ_series
# Select features in our age model
final_train_ex_df = final_train_ex_df[AGE_MODEL]
dummy_df = dummy_df[AGE_MODEL]

age_classifier, age_train_perf, _ = train_wnd_classifier(periods = 10,
                                                         lin_learning_rate = 0.01,
                                                         dnn_learning_rate = 0.01,
                                                         steps = 6000,
                                                         batch_size = 15,
                                                         hidden_units = [200, 150, 100],
                                                         train_ex = final_train_ex_df,
                                                         train_targ = final_train_targ_series,
                                                         val_ex = dummy_df,
                                                         val_targ = dummy_targ_series,
                                                         use_age = True)
# Split our test set into 2 dataframes (one where we can use our age model and the other where we use our basic model)
age_test_ex_df = test_df.dropna(axis = 0)
basic_test_ex_df = test_df.drop(age_test_ex_df.index)

# Create a dummy series that will be compatible with our input_fn
age_test_targ_series = pd.Series(np.zeros(len(age_test_ex_df), dtype=int))
basic_test_targ_series = pd.Series(np.zeros(len(basic_test_ex_df), dtype=int))

# Setup our input functions
pred_age_test_input_fn = lambda: input_fn(age_test_ex_df[AGE_MODEL], age_test_targ_series, num_epochs = 1, shuffle = False)
pred_basic_test_input_fn = lambda: input_fn(basic_test_ex_df[BASIC_MODEL], basic_test_targ_series, num_epochs = 1, shuffle = False)

# Make predictions for rows that have ages
age_test_preds = age_classifier.predict(input_fn = pred_age_test_input_fn)
age_test_preds = np.array([pred['class_ids'][0] for pred in age_test_preds])

# Make predictions for rows that lack ages
basic_test_preds = basic_classifier.predict(input_fn = pred_basic_test_input_fn)
basic_test_preds = np.array([pred['class_ids'][0] for pred in basic_test_preds])

# Let's put together our submission dataframe by merging our age and basic prediction sets
age_predictions = age_test_ex_df[["PassengerId"]]
age_predictions = age_predictions.assign(Survived=pd.Series(age_test_preds).values)
basic_predictions = basic_test_ex_df[["PassengerId"]]
basic_predictions = basic_predictions.assign(Survived=pd.Series(basic_test_preds).values)
submission_df = pd.concat([age_predictions, basic_predictions])
submission_df = submission_df.sort_values(by=['PassengerId'])
display(submission_df)
submission_df.to_csv('submission.csv', index = False)