import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
# No, really, that's it
# Could've been done with keras' pad_sequences, but why do it the simple way? Life's too short
def normalise_and_pad(sequence):
    ret = np.pad(np.array(sequence) / 255.0, ((0, 10-len(sequence)),(0,0)), 'wrap')
    return ret
df = pd.read_json('../input/train.json')
x_train = np.asarray([normalise_and_pad(x) for x in df['audio_embedding']])
y_train = df['is_turkey'].values

# Split the data into 60% training, 20% cross-validation, 20% test
x_train, x_crossvalidation, y_train, y_crossvalidation = train_test_split(x_train, y_train, test_size=0.2)
x_crossvalidation, x_test, y_crossvalidation, y_test = train_test_split(x_crossvalidation, y_crossvalidation, test_size=0.5)
def cnn_model():
    dropout = 0.2

    activation = keras.activations.elu

    input_layer = keras.layers.Input(shape=(10,128))
    conv_1 = keras.layers.Conv1D(256, 1, activation=activation, padding='causal')(input_layer)
    maxpool_1 = keras.layers.MaxPool1D()(conv_1)
    
    conv_2 = keras.layers.Conv1D(512, 3, activation=activation, padding='causal')(maxpool_1)
    maxpool_2 = keras.layers.MaxPool1D()(conv_2)

    conv_3 = keras.layers.Conv1D(1024, 1, activation=activation, padding='causal')(maxpool_2)
    maxpool_3 = keras.layers.MaxPool1D()(conv_3)

    flattened = keras.layers.Flatten()(maxpool_3) 
    fc1 = keras.layers.Dense(256, activation=activation)(flattened)
    fc1 = keras.layers.Dropout(rate=dropout)(fc1)

    fc2 = keras.layers.Dense(128, activation=activation)(fc1)
    fc2 = keras.layers.Dropout(rate=dropout)(fc2)

    dense_out = keras.layers.Dense(1, activation='sigmoid')(fc2)
    model = keras.models.Model(input_layer, dense_out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
def try_model(model, batch_size=64, epochs=40):
    checkpointer = keras.callbacks.ModelCheckpoint(filepath='./weights.hdf5', monitor='val_loss', save_best_only=True, save_weights_only=True)

    model.fit(x_train, y_train, epochs=epochs, callbacks=[checkpointer], batch_size=batch_size, validation_data=(x_crossvalidation, y_crossvalidation))
    model.load_weights('./weights.hdf5')
    _, train_acc = model.evaluate(x_train, y_train)
    _, cv_acc = model.evaluate(x_crossvalidation, y_crossvalidation)
    _, test_acc = model.evaluate(x_test, y_test)
    print ("Training         accuracy is {}%".format(train_acc * 100))
    print ("Cross-validation accuracy is {}%".format(cv_acc * 100))
    print ("Hold-out test    accuracy is {}%".format(test_acc * 100))
def gru_model():
    dropout = 0.2
    input_layer = keras.layers.Input(shape=(10,128))
    gru_out = keras.layers.Bidirectional(keras.layers.GRU(128, dropout=dropout, recurrent_dropout=dropout))(input_layer)
    dense_out = keras.layers.Dense(1, activation='sigmoid')(gru_out)
    model = keras.models.Model(input_layer, dense_out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
def lstm_model():
    dropout = 0.2
    input_layer = keras.layers.Input(shape=(10,128))
    lstm_out = keras.layers.LSTM(128, dropout=dropout, recurrent_dropout=dropout)(input_layer)
    dense_out = keras.layers.Dense(1, activation='sigmoid')(lstm_out)
    model = keras.models.Model(input_layer, dense_out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
def blend_models(models, x):
    ys = np.zeros((x.shape[0],1))
    for model in models:
        ys += model.predict(x)
    ys /= len(models)
    return ys
# let's try out some stuff
cranberry_neural_network = cnn_model() # forced pun, I know
try_model(cranberry_neural_network, batch_size=512, epochs=100) # how good is it anyway?
gobbling_recurrent_unit = gru_model() # less of a pun, more of a WTF
try_model(gobbling_recurrent_unit, batch_size=512, epochs=100)
long_short_turkey_memory = lstm_model()
try_model(long_short_turkey_memory, batch_size=512, epochs=100)
def test_blend(models):
    y1 = blend_models(models, x_train)
    y2 = blend_models(models, x_crossvalidation)
    y3 = blend_models(models, x_test)
    print ("Blended train accuracy            {}%".format(roc_auc_score(y_train, y1) * 100))
    print ("Blended cross-validation accuracy {}%".format(roc_auc_score(y_crossvalidation, y2) * 100))
    print ("Blended test accuracy             {}%".format(roc_auc_score(y_test, y3) * 100))
all_models = [cranberry_neural_network, gobbling_recurrent_unit, long_short_turkey_memory]
test_blend(all_models)
df_test = pd.read_json('../input/test.json')
x_submission = np.asarray([normalise_and_pad(x) for x in df_test['audio_embedding']])
y_pred = blend_models(all_models, x_submission)
df_out = pd.DataFrame({'vid_id':df_test['vid_id'],'is_turkey':[x[0] for x in y_pred]})
df_out.to_csv('submission.csv', index=False)