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



ctrl_ids = x_train[x_train.cp_type == 'ctl_vehicle'].sig_id

ctrl_label_count = y_train[y_train.sig_id.isin(ctrl_ids)].values[:, 1:].sum()

print(f'Control type samples label count: {ctrl_label_count}')

x_train = x_train[~x_train.sig_id.isin(ctrl_ids)]

y_train = y_train[~y_train.sig_id.isin(ctrl_ids)]



# Onehot-encode the categorical features

# type_cat = pd.get_dummies(x_train.cp_type)

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
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Plot the distribution of unique label combinations. We can see that

# The popularity reduces rapidly over different combinations with the majority

# Of samples belonging to just a handful of combinations.

class_df.class_count.plot(figsize=(13,6), logy=True, title='Histogram of label combinations')

plt.xlabel('Label combination no')

plt.ylabel('Log Count')

plt.show()
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
cell_features = [i for i in x_train.columns if str(i)[:2] == 'c-']

gene_features = [i for i in x_train.columns if str(i)[:2] == 'g-']
import warnings

import tensorflow as tf

import tensorflow_addons as tfa

from tensorflow.keras import Model, layers, optimizers, callbacks

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.preprocessing import RobustScaler, StandardScaler, KBinsDiscretizer, MaxAbsScaler

from sklearn.decomposition import PCA, FastICA

from sklearn.feature_selection import VarianceThreshold

tf.config.optimizer.set_jit(True)





# Load the test set and preprocess it to the same format with

# the training set

test_features = pd.read_csv('/kaggle/input/lish-moa/test_features.csv')

ctrl_idxs = test_features[test_features.cp_type != 'trt_cp' ].index



# test_type = pd.get_dummies(test_features.cp_type)

test_time = pd.get_dummies(test_features.cp_time)

test_dose = pd.get_dummies(test_features.cp_dose)



test_features = pd.concat([test_features, test_dose, test_time], axis=1)

test_features = test_features[feature_columns].values
import sys

sys.path.append('../input/multilabel-skf')

from ml_stratifiers import MultilabelStratifiedKFold

from imblearn.over_sampling import SMOTENC





def make_nn(layer_size, num_features):

    # Simple neural network with one hidden layer. Nothing special

    # Last layer is shape 1xnum_classes with sigmoid activation

    # and the loss is binary crossentropy because we have to predict

    # multiple labels for each sample.

    # This means our labels are not mutually exclusive and we cannot use

    # softmax activation and categorical crossentropy loss

    optimizer = optimizers.Adam()

    loss = tf.keras.losses.BinaryCrossentropy()



    model = tf.keras.Sequential([

        layers.Input(shape=(num_features,)),

        layers.BatchNormalization(),

        layers.Dropout(.3),

        tfa.layers.WeightNormalization(layers.Dense(layer_size // 2,

                                                    activation='elu')),

        layers.Dropout(.3),

        layers.BatchNormalization(),

        tfa.layers.WeightNormalization(layers.Dense(layer_size,

                                                    activation='elu')),

        layers.Dropout(.5),

        layers.BatchNormalization(),

        tfa.layers.WeightNormalization(layers.Dense(layer_size,

                                                    activation='elu')),

        layers.Dropout(.5),

        layers.BatchNormalization(),

        tfa.layers.WeightNormalization(layers.Dense(y.shape[1], activation='sigmoid'))

    ])

    

    model.compile(optimizer=optimizer, loss=loss)

    

    return model







early_stopping = callbacks.EarlyStopping(patience=10, restore_best_weights=True)

reduce_lr = callbacks.ReduceLROnPlateau(patience=3, mode='min', monitor='val_loss')



# Assert predictions shape matches the number of models

param_list = [42, 99, 4]



oof_score = []

validation_preds = np.zeros((X.shape[0], y.shape[1]))

submission_preds = np.zeros((test_features.shape[0], y.shape[1]))



# Loop over combinations of models using fold cross validation. For each

# fold make predictions of the test set.



for j, d in enumerate(param_list):

    kf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=d)

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):

        

        # Define oversampling strategy

        # Because there is huge class imbalance

        # Creating too many synthetic samples

        # Could be problematic

        X_smote = X[train_index]

        y_smote = y_classes[train_index]

        

        # Find the distribution of label power-sets within the fold

        fold_unique_y, fold_unique_counts = np.unique(y_smote, return_counts=True)

        

        # Define oversampling by touching only minority classes

        # with few enough samples but not too few

        resample_dict = {}

        for k, c_name in enumerate(fold_unique_y):

            if fold_unique_counts[k] > 150:

                oversample_count = fold_unique_counts[k]

            elif fold_unique_counts[k] < 100:

                oversample_count = fold_unique_counts[k]

            else:

                oversample_count = 150

            

            resample_dict[c_name] = oversample_count

            

        cat_idxs = np.array([872, 873, 874, 875, 876])

        smote = SMOTENC(cat_idxs, sampling_strategy=resample_dict)

        

        X_train, y_cat_train = smote.fit_resample(X_smote, y_smote)

        

        y_train = np.array([class_to_row[i] for i in y_cat_train])

        

        X_test = X[test_index]

        y_test = y[test_index]



        ckp_filepath = f'weights.{j:02d}-{i:02d}.hdf5'

        model_checkpoint = callbacks.ModelCheckpoint(filepath=ckp_filepath,

                                                     save_weights_only=True,

                                                     monitor='val_loss',

                                                     mode='min',

                                                     save_best_only=True)

        tf.keras.backend.clear_session()

        model_instance = make_nn(1024, X_train.shape[1])

        oof_h = model_instance.fit(X_train, y_train,

                                   epochs=50, batch_size=64,

                                   validation_data=(X_test, y_test),

                                   callbacks=[early_stopping,

                                              reduce_lr,

                                              model_checkpoint],

                                   verbose=0)

        model_instance.load_weights(ckp_filepath)

               

        # Predict the hold out set

        fold_score = min(oof_h.history['val_loss'])

        

        # Keep track of the validation score

        print(f'Finished training seed {j} fold {i}. OOF log loss: {fold_score:.5f}')

        oof_score.append(fold_score)

        

        # Predict the validation set

        valid_preds = model_instance.predict(X_test)

        validation_preds[test_index] += valid_preds

        

        # Predict the test set

        test_preds = model_instance.predict(test_features)

        submission_preds += test_preds

        

        

# Average the predictions

validation_preds /= (j + 1)

submission_preds /= (i + 1) * (j + 1)
# Print the average validation score across all folds

print(f'Average fold log loss: {np.mean(oof_score):.5} +- {np.std(oof_score):.5}')



bce = tf.keras.losses.BinaryCrossentropy()

valid_loss = bce(y, validation_preds).numpy()

print(f'Validation log loss: {valid_loss:.5}')
# Fill in the predictions to the submission csv and save it

submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')

submission.iloc[:, 1:] = submission_preds

submission.loc[ctrl_idxs, submission.columns[1:]] = 0

submission.to_csv('submission.csv', index=False)