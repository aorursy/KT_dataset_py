import os

import time

import datetime

import numpy as np

import pandas as pd



# Keras

import tensorflow.keras as keras

from tensorflow.keras import layers

from tensorflow.keras import models

from tensorflow.keras import callbacks

from tensorflow.keras import backend as K



# Standard ML stuff

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD, FastICA

from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection



# Oversampling of minority class 'Churn customers'

from imblearn.over_sampling import SMOTE



# Plotting

import matplotlib.pyplot as plt
def get_keras_dataset(df):

    X = {str(col) : np.array(df[col]) for col in df.columns}

    return X
# Plot the results of the training

def plot_history(history):

    fig = plt.figure(figsize=(15,8))

    ax = plt.subplot(211)

    

    plt.xlabel('Epoch')

    plt.ylabel('loss, acc')

    

    # Losses

    ax.plot(history.epoch, history.history['loss'], label='Train LOSS')

    ax.plot(history.epoch, history.history['val_loss'], label='Val LOSS')

    ax.plot(history.epoch, history.history['acc'], label ='Train Accuracy')

    ax.plot(history.epoch, history.history['val_acc'], label='Val Accuracy')

    plt.legend()

    

    # Plot the learning_rate

    if 'lr' in history.history:

        ax = plt.subplot(212)

        plt.ylabel('Learning rate')

        ax.plot(history.epoch, history.history['lr'], label='learning_rate')

        plt.legend()

    plt.show()

    plt.close(fig)
# Load the dataset

telcom = pd.read_csv("../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")

telcom.head()
telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ",np.nan)

print("Missing values in TotalCharges: ", telcom["TotalCharges"].isnull().sum())



telcom = telcom[telcom["TotalCharges"].notnull()]

telcom = telcom.reset_index()[telcom.columns]

print("Missing values in TotalCharges: ", telcom["TotalCharges"].isnull().sum())



telcom["TotalCharges"] = telcom["TotalCharges"].astype(float)

print("dType TotalCharges: ", telcom['TotalCharges'].dtype)
telcom["SeniorCitizen"] = telcom["SeniorCitizen"].replace({1:"Yes", 0:"No"})
def tenure_lab(telcom) :

    if telcom["tenure"] <= 12 :

        return "Tenure_0-12"

    elif (telcom["tenure"] > 12) & (telcom["tenure"] <= 24 ):

        return "Tenure_12-24"

    elif (telcom["tenure"] > 24) & (telcom["tenure"] <= 48) :

        return "Tenure_24-48"

    elif (telcom["tenure"] > 48) & (telcom["tenure"] <= 60) :

        return "Tenure_48-60"

    elif telcom["tenure"] > 60 :

        return "Tenure_gt_60"

    

telcom["tenure_group"] = telcom.apply(lambda telcom:tenure_lab(telcom), axis=1)
numeric_cols = ['MonthlyCharges', 'TotalCharges', 'tenure']

target_col = ['Churn']

ignored_cols = ['customerID']

categorical_cols = telcom.select_dtypes(include='object').columns

categorical_cols = [col for col in categorical_cols if col not in target_col + ignored_cols]
for col in categorical_cols:

    telcom[col] = LabelEncoder().fit_transform(telcom[col])



telcom['Churn'] = telcom['Churn'].map({'Yes' : 1, 'No' : 0})
telcom[numeric_cols] = StandardScaler().fit_transform(telcom[numeric_cols])
pca = PCA(n_components=3)

_X = pca.fit_transform(telcom[numeric_cols + categorical_cols])

pca_data = pd.DataFrame(_X, columns=["PCA1", "PCA2", "PCA3"])

telcom[["PCA1", "PCA2", "PCA3"]] = pca_data



fica = FastICA(n_components=3)

_X = fica.fit_transform(telcom[numeric_cols + categorical_cols])

fica_data = pd.DataFrame(_X, columns=["FICA1", "FICA2", "FICA3"])

telcom[["FICA1", "FICA2", "FICA3"]] = fica_data



tsvd = TruncatedSVD(n_components=3)

_X = tsvd.fit_transform(telcom[numeric_cols + categorical_cols])

tsvd_data = pd.DataFrame(_X, columns=["TSVD1", "TSVD2", "TSVD3"])

telcom[["TSVD1", "TSVD2", "TSVD3"]] = tsvd_data



grp = GaussianRandomProjection(n_components=3)

_X = grp.fit_transform(telcom[numeric_cols + categorical_cols])

grp_data = pd.DataFrame(_X, columns=["GRP1", "GRP2", "GRP3"])

telcom[["GRP1", "GRP2", "GRP3"]] = grp_data



srp = SparseRandomProjection(n_components=3)

_X = srp.fit_transform(telcom[numeric_cols + categorical_cols])

srp_data = pd.DataFrame(_X, columns=["SRP1", "SRP2", "SRP3"])

telcom[["SRP1", "SRP2", "SRP3"]] = srp_data



#tsne = TSNE(n_components=3)

#_X = tsne.fit_transform(telcom[numeric_cols + categorical_cols])

#tsne_data = pd.DataFrame(_X, columns=["TSNE1", "TSNE2", "TSNE3"])

#telcom[["TSNE1", "TSNE2", "TSNE3"]] = tsne_data



numeric_cols.extend(pca_data.columns.values)

numeric_cols.extend(fica_data.columns.values)

numeric_cols.extend(tsvd_data.columns.values)

numeric_cols.extend(grp_data.columns.values)

numeric_cols.extend(srp_data.columns.values)

#numeric_cols.extend(tsne_data.columns.values)
train_df, test_df = train_test_split(telcom, test_size=0.15, random_state=42)

print(train_df.shape)
train_df.head()
smote = SMOTE(sampling_strategy='minority', random_state=42)

os_smote_X, os_smote_Y = smote.fit_sample(train_df[numeric_cols + categorical_cols], train_df[target_col].values.ravel())



train_df = pd.DataFrame(os_smote_X, columns=numeric_cols + categorical_cols)

train_df['Churn'] = os_smote_Y

print(train_df.shape)
os_smote_X
customer_id = telcom['customerID']

telcom = telcom.drop('customerID', axis=1)
K.clear_session()
from keras.callbacks import EarlyStopping



es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1)



FEATURE_COLS = numeric_cols + categorical_cols

TARGET_COL = 'Churn'

EPOCHS = 500

BATCH_SIZE = 100000

CLASS_WEIGHTS = {0 : 1., 1 : 2.5}
cat_inputs = []

num_inputs = []

embeddings = []

embedding_layer_names = []

emb_n = 10
# Embedding for categorical features

for col in categorical_cols:

    _input = layers.Input(shape=[1], name=col)

    _embed = layers.Embedding(telcom[col].max() + 1, emb_n, name=col+'_emb')(_input)

    cat_inputs.append(_input)

    embeddings.append(_embed)

    embedding_layer_names.append(col+'_emb')

    

# Simple inputs for the numeric features

for col in numeric_cols:

    numeric_input = layers.Input(shape=(1,), name=col)

    num_inputs.append(numeric_input)

    

# Merge the numeric inputs

merged_num_inputs = layers.concatenate(num_inputs)

#numeric_dense = layers.Dense(20, activation='relu')(merged_num_inputs)



# Merge embedding and use a Droput to prevent overfittting

merged_inputs = layers.concatenate(embeddings)

spatial_dropout = layers.SpatialDropout1D(0.2)(merged_inputs)

flat_embed = layers.Flatten()(spatial_dropout)



# Merge embedding and numeric features

all_features = layers.concatenate([flat_embed, merged_num_inputs])



# MLP for classification

x = layers.Dropout(0.2)(layers.Dense(100, activation='relu')(all_features))

x = layers.Dropout(0.2)(layers.Dense(50, activation='relu')(x))

x = layers.Dropout(0.2)(layers.Dense(25, activation='relu')(x))

x = layers.Dropout(0.2)(layers.Dense(15, activation='relu')(x))



# Final model

output = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs=cat_inputs + num_inputs, outputs=output)
def dice_coef(y_true, y_pred, smooth=1):

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)

    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)



def dice_coef_loss(y_true, y_pred):

    return 1-dice_coef(y_true, y_pred)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
# TB Callback

log_folder = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')

tb_callback = callbacks.TensorBoard(

    log_dir=os.path.join('tb-logs', log_folder),

)



# Best model callback

bm_callback = callbacks.ModelCheckpoint(

    filepath=os.path.join('tb-logs', log_folder, 'bm.h5'),

    save_best_only=True,

    save_weights_only=False

)
_hist = model.fit(

    x=get_keras_dataset(train_df[FEATURE_COLS]),

    y=train_df[TARGET_COL],

    validation_data=(get_keras_dataset(test_df[FEATURE_COLS]), test_df[TARGET_COL]),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    class_weight=CLASS_WEIGHTS,

    callbacks=[tb_callback, bm_callback],

    verbose=2

)
plot_history(_hist)
model = keras.models.load_model(os.path.join('tb-logs', log_folder, 'bm.h5'), compile=False)
pred = np.around(model.predict(get_keras_dataset(test_df[FEATURE_COLS])))



print(accuracy_score(test_df[TARGET_COL], pred))

print(classification_report(test_df[TARGET_COL], pred))