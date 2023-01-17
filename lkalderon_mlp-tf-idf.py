import numpy as np

import pandas as pd

import os

import gc



from deepreplay.callbacks import ReplayData

from deepreplay.datasets.parabola import load_data



from keras import backend as K # Importing Keras backend (by default it is Tensorflow)

from keras.layers import Input, Dense, Dropout # Layers to be used for building our model

from keras.models import Model # The class used to create a model

from keras.optimizers import Adam

from keras.layers.normalization import BatchNormalization

from keras.regularizers import l2

from keras.utils import np_utils # Utilities to manipulate numpy arrays

from keras.callbacks import Callback, ReduceLROnPlateau

from keras.wrappers.scikit_learn import KerasClassifier #Wrapper for scikit

from tensorflow import set_random_seed # Used for reproducible experiments

import keras 



import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split, learning_curve, cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import roc_auc_score, confusion_matrix

from sklearn.decomposition import TruncatedSVD



from scipy.sparse import coo_matrix

from scipy.sparse import hstack

from scipy import sparse

%matplotlib inline
#Import clean data

classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']



datasets_path = '../input/'

train = pd.read_csv(datasets_path + 'cleaned_train.csv').fillna(' ')

test = pd.read_csv(datasets_path + 'cleaned_test.csv').fillna(' ')



train_text = train['comment_text']

test_text = test['comment_text']



all_text = pd.concat([train_text, test_text])



y_train = train[classes].values

y_test = pd.read_csv(datasets_path + 'test_labels.csv')

y_test = y_test[classes].values
all_text = pd.concat([train_text, test_text])



word_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='word',

    token_pattern=r'\w{1,}',

    stop_words='english',

    ngram_range=(1, 2),

    max_features=5000)

word_vectorizer.fit(all_text)



char_vectorizer = TfidfVectorizer(

    sublinear_tf=True,

    strip_accents='unicode',

    analyzer='char',

    stop_words='english',

    ngram_range=(3, 4),

    max_features=2000)

char_vectorizer.fit(all_text)



train_word_features = word_vectorizer.transform(train_text)

test_word_features = word_vectorizer.transform(test_text)



train_char_features = char_vectorizer.transform(train_text)

test_char_features = char_vectorizer.transform(test_text)



train_features = hstack([train_char_features, train_word_features])

test_features = hstack([test_char_features, test_word_features])
svd = TruncatedSVD(n_components=1000, random_state=4321)

train_features_svd = svd.fit_transform(train_features)

test_features_svd = svd.transform(test_features)
# https://www.kaggle.com/yekenot/pooled-gru-fasttext



#Define a class for model evaluation

class RocAucEvaluation(Callback):

    def __init__(self, training_data=(),validation_data=()):

        super(Callback, self).__init__()



        self.interval = interval

        self.X_tra, self.y_tra = training_data

        self.X_val, self.y_val = validation_data

        self.aucs_val = []

        self.aucs_tra = []

        

    def on_epoch_end(self, epoch, logs={}):                   

        y_pred_val = self.model.predict(self.X_val, verbose=0)

        score_val = roc_auc_score(self.y_val, y_pred_val)



        y_pred_tra = self.model.predict(self.X_tra, verbose=0)

        score_tra = roc_auc_score(self.y_tra, y_pred_tra)



        self.aucs_tra.append(score_val)

        self.aucs_val.append(score_tra)

        print("\n ROC-AUC - epoch: %d - score_tra: %.6f - score_val: %.6f \n" % (epoch+1, score_tra, score_val))



def recall(y_true, y_pred):    

    """

    Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of

    how many relevant items are selected.

    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (possible_positives + K.epsilon())

    return recall





def precision(y_true, y_pred):    

    """

    Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of

    how many selected items are relevant.

    Source

    ------

    https://github.com/fchollet/keras/issues/5400#issuecomment-314747992

    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    precision = true_positives / (predicted_positives + K.epsilon())

    return precision





def f1(y_true, y_pred):

    

    """Calculate the F1 score."""

    p = precision(y_true, y_pred)

    r = recall(y_true, y_pred)

    return 2 * ((p * r) / (p + r))





def accuracy(y_true, y_pred):

    return K.mean(K.equal(y_true, K.round(y_pred)), axis=1)
class Plots:

    def plot_history(history):

        loss = history.history['loss']

        val_loss = history.history['val_loss']

        x = range(1, len(val_loss) + 1)



        plt.plot(x, loss, 'b', label='Training loss')

        plt.plot(x, val_loss, 'r', label='Validation loss')

        plt.title('Training and validation loss')

        plt.legend()



    def plot_roc_auc(train_roc, val_roc):

        x = range(1, len(val_roc) + 1)



        plt.plot(x, train_roc, 'b', label='Training RocAuc')

        plt.plot(x, val_roc, 'r', label='Validation RocAuc')

        plt.title('Training and validation RocAuc')

        plt.legend()
def MLP_model(

    input_size,

    optimizer,    

    classes=6,  

    epochs=100,

    batch_size=128,

    hidden_layers=1,

    units=600,

    dropout_rate=0.8,

    l2_lambda=0.0,

    batch_norm=False,

    funnel=False,

    hidden_activation='relu',

    output_activation='sigmoid'

):

  

    # Define the seed for numpy and Tensorflow to have reproducible experiments.

    np.random.seed(1402) 

    set_random_seed(1981)

       

    input = Input(

        shape=(input_size,),

        name='Input'

    )

    x = input

    print(x.shape)

    # Define the hidden layers.

    for i in range(hidden_layers):

        if funnel:

            layer_units=units // (i+1)

        else: 

            layer_units=units

        x = Dense(

           units=layer_units,

           kernel_initializer='glorot_uniform',

           kernel_regularizer=l2(l2_lambda),

           activation=hidden_activation,

           name='Hidden-{0:d}'.format(i + 1)

        )(x)

        #Dropout

        if dropout_rate != 0:

            x = Dropout(dropout_rate)(x)

        if batch_norm:

            x = BatchNormalization()(x)

            

    # Define the output layer.    

    output = Dense(

        units=classes,

        kernel_initializer='uniform',

        activation=output_activation,

        name='Output'

    )(x)

    # Define the model and train it.

    model = Model(inputs=input, outputs=output)

      

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_crossentropy'])

    

    return model
X_tra, X_val, y_tra, y_val = train_test_split(train_features, y_train, train_size=0.90, random_state=3)

RocAuc = RocAucEvaluation(training_data=(X_tra, y_tra) ,validation_data=(X_val, y_val))



batch_size = 256

num_classes = 6

epochs = 1

optimizer = Adam(lr=0.001)



model = MLP_model(

    input_size = X_tra.shape[1],

    optimizer = optimizer,    

    classes=num_classes,  

    epochs=100,

    batch_size=128,

    hidden_layers=2,

    units=1000,

    dropout_rate=0.5,

    l2_lambda=0.0,

    batch_norm=False,

    funnel=True,

    hidden_activation='relu',

    output_activation='sigmoid'

)



# Keras Callbacks

reducer_lr = ReduceLROnPlateau(factor = 0.00002, patience = 1, min_lr = 1e-6, verbose = 1)

early_stopper = keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', mode='min', patience = 10) # Change 4 to 8 in the final run

model_file_name = 'weights_base.best.hdf5'

check_pointer = keras.callbacks.ModelCheckpoint(model_file_name, monitor='val_binary_crossentropy', mode='min', verbose = 1, save_best_only = True)  

log_file_name = 'model.log'

csv_logger = keras.callbacks.CSVLogger(log_file_name)

# replaydata = ReplayData(X_tra, y_tra, filename='hyperparms_in_action.h5', group_name='part1')

callbacks_list = [early_stopper, check_pointer, csv_logger, RocAuc, reducer_lr]



model.fit(x=X_tra,

          y=y_tra,          

          validation_data=(X_val, y_val),

          epochs=epochs,

          shuffle=True,

          verbose=1,

          batch_size=batch_size,

          callbacks = callbacks_list)



print('Finished training.')

print('------------------')
# model.summary() # Print a description of the model.

# plot_roc_auc(RocAuc.aucs_tra, RocAuc.aucs_val)

plot_history(model.history)
K.clear_session()

del model

gc.collect()

# y_test_pred = model.predict(test_features.tocsr(), batch_size=batch_size) #tocsr()