# Import libraries

import numpy as np

import pandas as pd



# Load the competition data (train features and train labels)

x_train = pd.read_csv('/kaggle/input/lish-moa/train_features.csv')

y_train = pd.read_csv('/kaggle/input/lish-moa/train_targets_scored.csv')



print(f'Train features shape: {x_train.shape}')

print(f'Train target shape: {y_train.shape}')



# The competition authors mentioned that only treatment typed samples have

# labels. We can quickly verify that



trt_ids = x_train[x_train.cp_type != 'trt_cp'].sig_id

non_trt_label_count = y_train[y_train.sig_id.isin(trt_ids)].values[:, 1:].sum()

print(f'Control type samples label count: {non_trt_label_count}')



# Keep only treatment typed samples from both train features

# and train labels

filtered_idxs = x_train[x_train.cp_type == 'trt_cp'].index

x_train = x_train.loc[filtered_idxs, :]

y_train = y_train.loc[filtered_idxs, :]



# One-encode the two categorical features: dose and time

dose_cat = pd.get_dummies(x_train.cp_dose)

time_cat = pd.get_dummies(x_train.cp_time)

x_train = pd.concat([x_train, dose_cat, time_cat], axis=1)



# Select the columns of the train dataset.

# The train dataset will consist of the numeric features plus the one-hot encoded

# representations of dose and time.

feature_columns = [i for i in x_train.columns if i not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]

target_classes = [i for i in y_train.columns if i not in ['sig_id']]



# Filter the data based on the selected columns

X = x_train[feature_columns].values

y = y_train[target_classes].values

print(f'Train dataset shape: {X.shape}')

print(f'Train labels shape: {y.shape}')



# Calculate the sparcity of labels

total_positive_labels = np.sum(y)

total_labels = y.flatten().shape[0]

print(f'Positive labels: {total_positive_labels}')

print(f'Total labels: {total_labels}')

print(f'Sparsity ratio: {total_positive_labels / total_labels:.4f}')



# Perform analysis on the representation of classes between samples

# We select the unique combination of feature classes in a 2D array

# and the frequency if each unique combination in a 1D array

unique_rows, unique_counts = np.unique(y, return_counts=True, axis=0)



# Construct a dataframe from the above extraction for easier manipulation

# Dataframe inndex will be the integer label for each unique combination,

# Row is each unique representation and class count its frequency count

class_df = pd.DataFrame({

    'row': [i for i in unique_rows],

    'class_count': list(unique_counts)

})



# Sort the dataframe in descending popularity. The impact of the sorting

# to end performance should be investigated, but this way when we filter out

# unpopular classes we do not have gaps in class numbers, can be problematic

# for certain sklearn methods

class_df.sort_values(by='class_count', ascending=False, inplace=True)

class_df.reset_index(drop=True, inplace=True)



# Construct dictionaries to map from a unique representation to a class number

# and vice versa

row_to_class = {}

class_to_row = {}

for i, df_row in class_df.iterrows():

    row_to_class[tuple(df_row.row)] = i

    class_to_row[i] = df_row.row



# Map the train labels to their respective class number. This way we can filter out

# train samples that belong to unpopular classes but also perform StratifedKFold.

# We can also use this new representation of classes to transform our multi-label

# problem to a multi-class problem which are generally easier to slove.



# But there are issues with that:

# If there are too many unique representations we end up with too many labels.

# Usually heavy class imbalance

# We cannot predict class combinations whose representation does not exist during

# training

y_classes = np.array([row_to_class[tuple(i)] for i in y])

print(f'Target class shape: {y_classes.shape}')
# Find which classes are the least popular.

outlier_classes = class_df[class_df.class_count < 6].index



# Find the index of the train labels where there are no unpopular classes

filtered_idx = [i for i,x in enumerate(y_classes) if x not in outlier_classes]



# Filter the train set without having unpopular classes

X = X[filtered_idx]

y = y[filtered_idx]

y_classes = y_classes[filtered_idx]

print(f'Outlier classes: {outlier_classes}')

print(f'Shape of filtered features: {X.shape}')

print(f'Shape of filtered classes: {y_classes.shape}')
# create label pairs

label_idxs = np.arange(y.shape[1])

label_pairs = [(i, j) for i,_ in enumerate(label_idxs) for j,_ in enumerate(label_idxs) if i>j]
from tqdm.notebook import tqdm

from sklearn.linear_model import LogisticRegressionCV, SGDClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import make_scorer, log_loss, roc_auc_score

import warnings



scorer = make_scorer(log_loss,

                     greater_is_better=False,

                     needs_proba=True)





def assign_class(class_names, pred):

    return class_names[0] if pred == 0 else class_names[1]





with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    pairwise_models = []

    pairwise_scores = []



    for pair in tqdm(label_pairs):

        partial_ds = y[:, pair]

        # filter samples that have at least one label but not both

        filtered_idxs = np.logical_xor(partial_ds[:, 0], partial_ds[:, 1])

        partial_X = X[filtered_idxs]

        partial_y = partial_ds[filtered_idxs, 1]

        ds_size = partial_y.shape[0]

        

        v, c = np.unique(partial_y, return_counts=True)

        if (v.shape[0] < 2) or any(i < 6 for i in c):

            continue

        

        model = LogisticRegressionCV(cv=5, n_jobs=5)

        model.fit(partial_X, partial_y)

        

        # calculate score

#         partial_preds = model.predict_proba(partial_X)

#         partial_score = roc_auc_score(partial_y, partial_preds[:, 1])

#         pairwise_scores.append(partial_score)

        

        # gather models

        pairwise_models.append((model, pair, ds_size))
test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')



test_time = pd.get_dummies(test_features.cp_time)

test_dose = pd.get_dummies(test_features.cp_dose)



test_features = pd.concat([test_features, test_dose, test_time], axis=1)

X_test = test_features[feature_columns].values



# Cast votes for each classifier

votes = np.zeros((X_test.shape[0], y.shape[1]))

normalization = 206

vote_normalization = max([i[2] for i in pairwise_models])



for i, (model, class_i, w) in tqdm(enumerate(pairwise_models), total=len(pairwise_models)):

#     vote = model.predict_proba(X)

#     current_class_votes = votes[:, class_i[1]]

#     current_class_votes += vote[:, 1] * (w/vote_normalization)

#     votes[:, class_i[1]] = current_class_votes



#     for j, class_idx in enumerate(class_i):

#         current_class_votes = votes[:, class_idx]

#         current_class_votes += vote[:, j] * (w/vote_normalization)

#         votes[:, class_idx] = current_class_votes



    vote = model.predict(X_test)

    vote = np.array(list(map(lambda x: assign_class(class_i, x), vote)))

    for class_name in class_i:

        vote = np.where(vote == class_name, 1, 0) * (w/vote_normalization)

        current_class_votes = votes[:, class_name]

        current_class_votes += vote

        votes[:, class_name] = current_class_votes



votes /= normalization



print(f'Votes shape: {votes.shape}')   



submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

submission.iloc[:, 1:] = votes

submission.to_csv('submission.csv', index=False)