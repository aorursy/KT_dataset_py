import pandas as pd 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_auc_score
from IPython.display import display
from sklearn.metrics import roc_auc_score

%matplotlib inline 
data_path="../input"
data = pd.concat([pd.read_csv(os.path.join(data_path, "application_train.csv"), index_col="SK_ID_CURR"), pd.read_csv(os.path.join(data_path, "application_test.csv"), index_col="SK_ID_CURR")], axis=0)
data_embeddings = data.select_dtypes("object").columns
data_embeddings
for c in data.drop(data_embeddings.tolist() + ["TARGET"], axis=1):
    mean = data[c].dropna().mean()
    std = data[c].dropna().std()
    data[c] = (data[c].fillna(mean) - mean) / std
for c in data_embeddings:
    data[c].fillna("N/A", inplace=True)
    data[c] = pd.factorize(data[c])[0]
data_embeddings
from keras.layers import Input, Dense, Embedding, TimeDistributed, concatenate, Flatten, Lambda, LSTM
from keras.models import Model
non_embeddings_ínput = Input(shape=(data.drop(data_embeddings.tolist() + ["TARGET"], axis=1).shape[1],), name="non_embeddings")
inputs = [non_embeddings_ínput]
first_layer = [non_embeddings_ínput]
for c in data_embeddings:
    current_input = Input(shape=(1,), name=c)
    inputs.append(current_input)
    cardinality = data[c].unique().shape[0]
    first_layer.append(Flatten()(Embedding(cardinality, int(cardinality ** 0.25))(current_input)))
    
fully_connected = concatenate(first_layer)
fully_connected = Dense(256, activation="relu")(fully_connected)
fully_connected = Dense(128, activation="relu")(fully_connected)
fully_connected = Dense(64, activation="relu")(fully_connected)
fully_connected = Dense(1, activation="sigmoid")(fully_connected)
from keras.utils import plot_model
model = Model(inputs=inputs, outputs=fully_connected)
plot_model(model, to_file='model.png', show_shapes=True, rankdir='LR')
from IPython.display import Image
Image("model.png")
train = data[data.TARGET.notnull()]
valid = train.sample(frac=0.1)
train = train.drop(valid.index)
X_train = {c: train[c].values for c in data_embeddings}
X_train["non_embeddings"] = train.drop(data_embeddings.tolist() + ["TARGET"], axis=1).values
y_train = train.TARGET.values
X_valid = {c: valid[c].values for c in data_embeddings}
X_valid["non_embeddings"] = valid.drop(data_embeddings.tolist() + ["TARGET"], axis=1).values
y_valid = valid.TARGET.values
non_embeddings_ínput = Input(shape=(data.drop(data_embeddings.tolist() + ["TARGET"], axis=1).shape[1],), name="non_embeddings")
inputs = [non_embeddings_ínput]
first_layer = [non_embeddings_ínput]

for c in data_embeddings:
    current_input = Input(shape=(1,), name=c)
    inputs.append(current_input)
    cardinality = data[c].unique().shape[0]
    first_layer.append(Flatten()(Embedding(cardinality, int(cardinality ** 0.25))(current_input)))

fully_connected = concatenate(first_layer)
fully_connected = Dense(256, activation="relu")(fully_connected)
fully_connected = Dense(128, activation="relu")(fully_connected)
fully_connected = Dense(64, activation="relu")(fully_connected)
fully_connected = Dense(1, activation="sigmoid")(fully_connected)

model = Model(inputs=inputs, outputs=fully_connected)
model.compile(optimizer='RMSprop', loss='binary_crossentropy')
for i in range(10):
    model.fit(X_train, y_train, batch_size=1024, validation_data=(X_valid, y_valid))
    print("ROC_AUC para iteracion", i, ":", roc_auc_score(y_train, model.predict(X_train)[:, -1]), " - ", roc_auc_score(y_valid, model.predict(X_valid)[:, -1]))
non_embeddings_ínput = Input(shape=(data.drop(data_embeddings.tolist() + ["TARGET"], axis=1).shape[1],), name="non_embeddings")
inputs = [non_embeddings_ínput]
first_layer = [non_embeddings_ínput]

for c in data_embeddings:
    current_input = Input(shape=(1,), name=c)
    inputs.append(current_input)
    cardinality = data[c].unique().shape[0]
    first_layer.append(Flatten()(Embedding(cardinality, int(cardinality ** 0.25))(current_input)))

fully_connected = concatenate(first_layer)
fully_connected = Dense(256, activation="relu")(fully_connected)
fully_connected = Dense(128, activation="relu")(fully_connected)
fully_connected = Dense(64, activation="relu")(fully_connected)
fully_connected = Dense(1, activation="sigmoid")(fully_connected)

model = Model(inputs=inputs, outputs=fully_connected)
model.compile(optimizer='RMSprop', loss='binary_crossentropy')
for _ in range(10):
    model.fit(X_train, y_train, batch_size=1024, validation_data=(X_valid, y_valid), class_weight={0: 1 / (1 - data.TARGET.mean()), 1: 1 / data.TARGET.mean()})
    print("ROC_AUC para iteracion", i, ":", roc_auc_score(y_train, model.predict(X_train)[:, -1]), " - ", roc_auc_score(y_valid, model.predict(X_valid)[:, -1]))
bureau = pd.read_csv(os.path.join(data_path, "bureau.csv"))
bureau_embeddings = bureau.select_dtypes("object").columns
for c in bureau.drop(bureau_embeddings, axis=1):
    if c == "SK_ID_CURR": continue
    mean = bureau[c].dropna().mean()
    std = bureau[c].dropna().std()
    bureau[c] = (bureau[c].fillna(mean) - mean) / std
for c in bureau_embeddings:
    bureau[c].fillna("N/A", inplace=True)
    bureau[c] = pd.factorize(bureau[c])[0]
    
bureau_non_embeddings_ínput = Input(shape=(bureau.drop(bureau_embeddings, axis=1).shape[1],), name="bureau_non_embeddings")
bureau_inputs = [bureau_non_embeddings_ínput]
bureau_first_layer = [bureau_non_embeddings_ínput]

for c in bureau_embeddings:
    current_input = Input(shape=(1,), name=c)
    bureau_inputs.append(current_input)
    cardinality = bureau[c].unique().shape[0]
    bureau_first_layer.append(Flatten()(Embedding(cardinality, int(cardinality ** 0.25))(current_input)))

bureau_fully_connected = concatenate(bureau_first_layer)
bureau_fully_connected = Dense(32, activation="relu")(bureau_fully_connected)
bureau_fully_connected = Dense(8, activation="relu")(bureau_fully_connected)
model = Model(inputs=bureau_inputs, outputs=bureau_fully_connected)
plot_model(model, to_file='model.png', show_shapes=True, rankdir='LR')
from IPython.display import Image
Image("model.png")
import keras.backend as K
bureau_non_embeddings_ínput = Input(shape=(None, bureau.drop(bureau_embeddings, axis=1).shape[1]), name="bureau_non_embeddings")
bureau_inputs = [bureau_non_embeddings_ínput]
bureau_first_layer = [bureau_non_embeddings_ínput]

for c in bureau_embeddings:
    current_input = Input(shape=(None, 1), name=c)
    bureau_inputs.append(current_input)
    cardinality = bureau[c].unique().shape[0]
    target_cardinality = int(cardinality ** 0.25)
    bureau_first_layer.append(Lambda(lambda x: x[:, :, 0, :])(Embedding(cardinality, target_cardinality)(current_input)))

bureau_fully_connected = concatenate(bureau_first_layer)
bureau_fully_connected = TimeDistributed(Dense(32, activation="relu"))(bureau_fully_connected)
bureau_fully_connected = TimeDistributed(Dense(8, activation="relu"))(bureau_fully_connected)
bureau_fully_connected = LSTM(2)(bureau_fully_connected)
model = Model(inputs=bureau_inputs, outputs=bureau_fully_connected)
plot_model(model, to_file='model.png', show_shapes=True, rankdir='LR')
from IPython.display import Image
Image("model.png")

non_embeddings_ínput = Input(shape=(data.drop(data_embeddings.tolist() + ["TARGET"], axis=1).shape[1],), name="non_embeddings")
inputs = [non_embeddings_ínput]
first_layer = [non_embeddings_ínput]

for c in data_embeddings:
    current_input = Input(shape=(1,), name=c)
    inputs.append(current_input)
    cardinality = data[c].unique().shape[0]
    first_layer.append(Flatten()(Embedding(cardinality, int(cardinality ** 0.25))(current_input)))

fully_connected = concatenate(first_layer)
fully_connected = Dense(128, activation="relu")(fully_connected)
fully_connected = Dense(32, activation="relu")(fully_connected)

bureau_non_embeddings_ínput = Input(shape=(None, bureau.drop(bureau_embeddings, axis=1).shape[1]), name="bureau_non_embeddings")
inputs.append(bureau_non_embeddings_ínput)
bureau_first_layer = [bureau_non_embeddings_ínput]

for c in bureau_embeddings:
    current_input = Input(shape=(None, 1), name=c)
    inputs.append(current_input)
    cardinality = bureau[c].unique().shape[0]
    target_cardinality = int(cardinality ** 0.25)
    bureau_first_layer.append(Lambda(lambda x: x[:, :, 0, :])(Embedding(cardinality, target_cardinality)(current_input)))

bureau_fully_connected = concatenate(bureau_first_layer)
bureau_fully_connected = TimeDistributed(Dense(32, activation="relu"))(bureau_fully_connected)
bureau_fully_connected = TimeDistributed(Dense(16, activation="relu"))(bureau_fully_connected)
bureau_fully_connected = LSTM(8)(bureau_fully_connected)

final_fully_connected = concatenate([fully_connected, bureau_fully_connected])

final_fully_connected = Dense(64, activation="relu")(final_fully_connected)
final_fully_connected = Dense(32, activation="relu")(final_fully_connected)
final_fully_connected = Dense(1, activation="sigmoid")(final_fully_connected)

model = Model(inputs=inputs, outputs=final_fully_connected)
plot_model(model, to_file='model.png', show_shapes=True, rankdir='LR')
from IPython.display import Image
Image("model.png")
