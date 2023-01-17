# For the random forest model

import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

import joblib



# Continued for keras model

import tensorflow as tf

from keras import layers, optimizers, callbacks, metrics, utils, regularizers

from keras.models import Model

from sklearn import metrics

from sklearn import model_selection

from keras import backend as K

import gc

import warnings

warnings.filterwarnings("ignore")
train = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")

test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")

train.columns
# Adding a column in Test data, to concatenate it with Training data [more on this later]

test["target"] = -1

test.columns
# Full dataset

full_data = pd.concat([train, test]).reset_index(drop=True)

full_data.shape 
# Categorical Features don't include the ID and the Target (obviously)

CATEGORICAL_FEATURES = [c for c in full_data.columns if c not in ["id", "target"]]
# I combined the datasets, just so that I could handle the 'previously unseen values' while inferencing

labels = dict()

for feature in CATEGORICAL_FEATURES:

    full_data.loc[:, feature] = full_data.loc[:, feature].fillna("-1").astype(str)

    label = preprocessing.LabelEncoder()

    label.fit(full_data.loc[:, feature].values.tolist())

    full_data.loc[:, feature] = label.transform(full_data.loc[:, feature].values.tolist())

    labels[feature] = label

# joblib.dump(labels, f"../input/label_dict.pkl")
train_data = full_data[full_data.target != -1].reset_index(drop=True)

test_data = full_data[full_data.target == -1].drop(["target"], axis=1).reset_index(drop=True)

print(f"Train shape: {train_data.shape} ; Test shape: {test_data.shape}")
train_data.head()
train_data.to_csv("train_categorical.csv", index=False)

test_data.to_csv("test_categorical.csv", index=False)
# Number of folds defined defined for StratifiedKFold, for the Random Forest model

NUMBER_OF_FOLDS = 5



# Just for sanity check

train_data = pd.read_csv("train_categorical.csv")

test_data = pd.read_csv("test_categorical.csv")
# Creating a column to incorporate the folds

train_data.loc[:, "kfold"] = -1

kfold = model_selection.StratifiedKFold(n_splits=NUMBER_OF_FOLDS, shuffle=True, random_state=7) # 7 is my lucky number! (I know about 42 don't worry :p)



for fold, (train_idx, valid_idx) in enumerate(kfold.split(X = train_data, y = train_data.target.values)):

    train_data.loc[valid_idx, "kfold"] = fold

    print(f"Shape of {fold} fold: ({len(train_data.loc[train_idx])},{len(train_data.loc[valid_idx])})")

# train_data.to_csv("../input/train_folds.csv", index = False)
# Creating and saving a classifier for each fold

clf_dict = dict()

for fold in range(NUMBER_OF_FOLDS):

    train_df = train_data[train_data.kfold != fold].reset_index(drop=True)

    valid_df = train_data[train_data.kfold == fold].reset_index(drop=True)

    training_y = train_df.target.values

    training_x = train_df.drop(["id", "target", "kfold"], axis=1)

    validation_y = valid_df.target.values

    validation_x = valid_df.drop(["id", "target", "kfold"], axis=1)

    

    clf = RandomForestClassifier(n_estimators=200, n_jobs = 12, verbose=0)

    clf.fit(training_x, training_y)

    pred_y = clf.predict_proba(validation_x)[:, 1]

    print(metrics.roc_auc_score(validation_y, pred_y))

    clf_dict[fold] = clf

#     joblib.dump(clf, f"./{MODEL}_{fold}")
# test_data = pd.read_csv("../input/test_categorical.csv")

sample = pd.read_csv("../input/cat-in-the-dat-ii/sample_submission.csv")

sample.head()
# Averaging out the 5 classifiers' output

predictions = None

test_idx = test_data["id"]

test_data = test_data.drop(["id"], axis=1)



for fold in range(NUMBER_OF_FOLDS):

#     clf = joblib.load(f"./{MODEL}_{fold}")

    clf = clf_dict[fold]

    predict = clf.predict_proba(test_data)[:,1]

    if fold == 0:

        predictions = predict

    else:

        predictions += predict

rf_prediction = predictions / float(NUMBER_OF_FOLDS)
submission = pd.DataFrame(np.column_stack((test_idx, rf_prediction)), columns=["id", "target"])

submission.id = submission.id.astype(int)

submission.to_csv("randomforest_submission.csv", index=False)
%reset -f
# For the random forest model

import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.ensemble import RandomForestClassifier

import joblib



# Continued for keras model

import tensorflow as tf

from keras import layers, optimizers, callbacks, metrics, utils, regularizers

from keras.models import Model

from sklearn import metrics

from sklearn import model_selection

from keras import backend as K

import gc

import warnings

warnings.filterwarnings("ignore")
# There's a keras metric AUC with which I was initially training, however I realised it wasn't a reliable metric

# Therefore after searching for reliable metrics, I came across this function which will act as our metric.

def auc(y_true, y_pred):

    def fallback_auc(y_true, y_pred):

        try:

            return metrics.roc_auc_score(y_true, y_pred)

        except:

            return 0.5

    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)
# Number of folds defined defined for StratifiedKFold, for the keras model

NUMBER_OF_FOLDS = 3



# Just for sanity check

train_data = pd.read_csv("train_categorical.csv")

test_data = pd.read_csv("test_categorical.csv")



# Since we cleared the memory

CATEGORICAL_FEATURES = [c for c in train_data.columns if c not in ["id", "target"]]
# This function will return the created keras model

def make_model(data, features):

    input_cols = []

    output_emb = []

    for column in features:

        embedding_dim = min(int(len(data[column].unique())/2)+1, 65)

        input_layer = layers.Input(shape=(1,)) #input will be batches of 1 dimension

        mid_layer = layers.Embedding(len(data[column].values.tolist())+1, embedding_dim)(input_layer) 

        out = layers.SpatialDropout1D(0.1)(mid_layer)

        emb = layers.Reshape(target_shape=(embedding_dim,))(out)

        input_cols.append(input_layer)

        output_emb.append(emb)

    x = layers.Concatenate()(output_emb)

    x = layers.BatchNormalization()(x)

    x = layers.Dense(4096, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)

    x = layers.Dropout(0.1)(x)

    x = layers.BatchNormalization()(x)

    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)

    x = layers.Dropout(0.1)(x)

    x = layers.BatchNormalization()(x)

    y = layers.Dense(2, activation="softmax", kernel_regularizer=regularizers.l2(0.001))(x)

    model = Model(inputs = input_cols, outputs = y)

    return model
final_valid_preds = np.zeros((len(train_data)))

final_test_preds = np.zeros((len(test_data)))



kfold = model_selection.StratifiedKFold(n_splits=NUMBER_OF_FOLDS, shuffle=True)



# Defining callbacks

earlystop = callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001, patience=5, verbose=1, mode='max', baseline=None, restore_best_weights=True)

reducelr = callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, min_lr=1e-6, mode='max', verbose=1)



test_idx = test_data["id"]

test_df = test_data.drop(["id"], axis=1)

test = [test_df.values[:, k] for k in range(test_df.values.shape[1])]



for (train_idx, valid_idx) in kfold.split(X = train_data, y = train_data.target.values):

    print(f"Shape of fold: ({len(train_data.loc[train_idx])},{len(train_data.loc[valid_idx])})")

    train_df = train_data.loc[train_idx]

    valid_df = train_data.loc[valid_idx]

    ytrain = train_df.target.values

    valid_y = valid_df.target.values

    

    # The input to the model will be list of lists such that each list will represent encoded data of a single feature

    X = [train_df.loc[:, CATEGORICAL_FEATURES].values[:, k] for k in range(train_df.loc[:, CATEGORICAL_FEATURES].values.shape[1])]

    Xvalid = [valid_df.loc[:, CATEGORICAL_FEATURES].values[:, k] for k in range(valid_df.loc[:, CATEGORICAL_FEATURES].values.shape[1])]



    # Defining the model

    model = make_model(train_df, CATEGORICAL_FEATURES)

    

    # Metric is the metric function defined above

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=[auc])

 

    history = model.fit(X, utils.to_categorical(ytrain), validation_data=(Xvalid, utils.to_categorical(valid_y)),

                        batch_size = 1024, callbacks=[earlystop, reducelr], epochs=100, verbose=1)

    # Predict on test data

    test_preds = model.predict(test)[:,1]

    # Predict on validation data per fold

    valid_preds = model.predict(Xvalid)[:, 1]

    

    final_valid_preds[valid_idx] = valid_preds.ravel()

    final_test_preds += test_preds.ravel()

    

    print(metrics.roc_auc_score(valid_y, valid_preds))

    

    # To clear out the GPU memory held by the model

    K.clear_session()    
# The final prediction is taken as the average of the predictions per fold

keras_prediction = final_test_preds / float(NUMBER_OF_FOLDS)
submission = pd.DataFrame(np.column_stack((test_idx, keras_prediction)), columns=["id", "target"])

submission.id = submission.id.astype(int)

submission.to_csv("keras_submission.csv", index=False)