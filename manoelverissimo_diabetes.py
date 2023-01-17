import os

import string

import numpy as np

import pandas as pd



from keras.utils.np_utils import to_categorical

from keras.models import Model, Sequential, model_from_json

from keras.optimizers import SGD, Adam, RMSprop

from keras.layers import Input, Dense, Dropout, Flatten, Lambda, Embedding

from keras.layers.convolutional import Convolution1D, MaxPooling1D

from keras.initializers import RandomNormal, Constant

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras import regularizers



import sklearn

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score

from sklearn.utils import class_weight

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import ExtraTreesClassifier



import matplotlib.pyplot as plt



import itertools

import pickle



from imblearn.under_sampling import RandomUnderSampler

from imblearn.over_sampling import RandomOverSampler



import seaborn as sns



import json



import warnings



np.random.seed(123)  # for reproducibility

%matplotlib inline
def load_data():

    warnings.filterwarnings('ignore') 

    data = pd.read_csv('../input/dataset_treino.csv')

    data.drop(['id'], axis=1, inplace=True)

    data.drop(['indice_historico'], axis=1, inplace=True)



    data = data[(data.pressao_sanguinea != 0) & (data.grossura_pele != 0) & (data.insulina != 0) & (data.glicose != 0)]



    X_train = data.iloc[:,0:7]

    y_train = data['classe']

    

    X_train, y_train = balanceDataset('oversample', X_train, y_train)

    

    X_train = StandardScaler().fit_transform(X_train)



    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20, random_state=42, stratify=y_train)



    map_characters = {0: 'Negative', 1: 'Positive'}



    fig, ax = plt.subplots(1,2, figsize=(15,5))

    ax[0].set_title("Training data")

    df = pd.DataFrame()

    df["labels"]=y_train

    df["labels"].replace(0, map_characters[0], inplace=True)

    df["labels"].replace(1, map_characters[1], inplace=True)



    lab = df['labels']

    ax1 = sns.countplot(lab, ax=ax[0], order = df['labels'].value_counts().index)



    for p, label in zip(ax1.patches, df["labels"].value_counts().index):

        ax1.annotate(p.get_height(), (p.get_x()+0.275, p.get_height()+0.25))



    ax[1].set_title("Test data")

    df2 = pd.DataFrame()

    df2["labels"]=y_test

    df2["labels"].replace(0, map_characters[0], inplace=True)

    df2["labels"].replace(1, map_characters[1], inplace=True)



    lab2 = df2['labels']

    ax2 = sns.countplot(lab2, ax=ax[1], order = df2['labels'].value_counts().index)



    for p, label in zip(ax2.patches, df["labels"].value_counts().index):

        ax2.annotate(p.get_height(), (p.get_x()+0.275, p.get_height()+0.25))



    fig.show()



    return (X_train, y_train), (X_test, y_test)
def balanceDataset(sampling_strategy:str, X, y):

    sampler = {

        "oversample": RandomOverSampler(random_state=42),

        "undersample": RandomUnderSampler(random_state=42),

    }



    random_sampler = sampler.get(sampling_strategy)



    X_res, y_res = random_sampler.fit_sample(X, y)

       

    return X_res, y_res
def load_test_data():

    data = pd.read_csv('../input/dataset_teste.csv')

    data.drop(['id'], axis=1, inplace=True)

    data.drop(['indice_historico'], axis=1, inplace=True)



    X_test = StandardScaler().fit_transform(data)



    return X_test
# Show info of model

def show_info(model, X, y, log, weights = None):

    warnings.filterwarnings('ignore')

    if (log != None):

        # summarize history for accuracy

        plt.figure(figsize=(15,10))

        plt.plot(log.history['acc'])

        plt.plot(log.history['val_acc'])

        plt.title('Model Accuracy')

        plt.ylabel('accuracy')

        plt.xlabel('epoch')

        plt.legend(['train', 'test'], loc='upper left')

        plt.show()



        # summarize history for loss

        plt.figure(figsize=(15,10))

        plt.plot(log.history['loss'])

        plt.plot(log.history['val_loss'])

        plt.title('Model Loss')

        plt.ylabel('loss')

        plt.xlabel('epoch')

        plt.legend(['train', 'test'], loc='upper left')

        plt.show()



    if (weights != None):

        model.load_weights(weights)



    score = model.evaluate(X, y, verbose=0)

    print('Accuracy {:2.2f}%'.format(score[1]*100))



    # Evaluate model

    y_pred = model.predict(X)

    y_pred = np.where(y_pred > 0.5, 1, 0)

    class_labels = ['Negative', 'Positive']



    # Report

    report = sklearn.metrics.classification_report(y_test, y_pred, target_names=class_labels)

    print(report)



    return None
def create_model():

    initializer = RandomNormal(mean=0.0, stddev=0.05, seed=None)



    model = Sequential()

    model.add(Dense(32, input_dim=X_train.shape[1], kernel_initializer=initializer, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

    model.add(Dropout(0.3))

    model.add(Dense(32, input_dim=X_train.shape[1], kernel_initializer=initializer, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

    model.add(Dropout(0.3))



    model.add(Dense(1, kernel_initializer=initializer, activation='sigmoid'))

    

    sgd = SGD(lr=0.01, momentum=0.9)

    adam = Adam(lr=0.001, decay=0.01)  # Feel free to use SGD above. I found Adam with lr=0.001 is faster than SGD with lr=0.01

    rmsprop = RMSprop(lr=0.001, decay=0.01)



    # Compile model

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])



    return model
batch_size = 8

nb_epoch = 200



print('Loading data...')

(X_train, y_train), (X_test, y_test) = load_data()



print('Build model...')

model = create_model()
model.summary()

print('Fit model...')

filepath="weights_diabetes.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

early_stopping = EarlyStopping(monitor='val_acc', patience=5, verbose=1, mode='max')

callbacks_list = [checkpoint]



class_weight1 = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)



log = model.fit(X_train, y_train,

          validation_split=0.2, batch_size=batch_size, epochs=nb_epoch, class_weight=class_weight1, shuffle=True, callbacks=callbacks_list)
predictions = show_info(model, X_test, y_test, log, weights='weights_diabetes.best.hdf5')
test_data = load_test_data()



df_teste = pd.read_csv('../input/dataset_teste.csv')
predict = model.predict_classes(test_data)



submission = pd.DataFrame()

submission['id'] = df_teste.index + 1

submission['classe'] = predict



submission.to_csv('submission.csv', index=False)