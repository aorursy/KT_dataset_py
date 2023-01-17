# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# imports dans l'ordre d'utilisation 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence

from keras.models import Model

from keras.layers import Dense, Input, Dropout, LSTM, Activation

from keras.layers.embeddings import Embedding

from tensorflow import keras



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
nRowsRead = None # specify 'None' if want to read whole file

df = pd.read_csv('../input/granddebat/LA_TRANSITION_ECOLOGIQUE.csv', delimiter=',', nrows = nRowsRead)

df.dataframeName = 'LA_TRANSITION_ECOLOGIQUE.csv'

nRow, nCol = df.shape

print(f'There are {nRow} rows and {nCol} columns')
df.head(1)
df.columns.values.tolist()
q1q2_tmp = df.loc[:,['id', "QUXVlc3Rpb246MTYw - Quel est aujourd'hui pour vous le problème concret le plus important dans le domaine de l'environnement ?", 'QUXVlc3Rpb246MTYx - Que faudrait-il faire selon vous pour apporter des réponses à ce problème ?']]

q1q2_tmp.head(1)
q1q2 = q1q2_tmp.rename(columns={"QUXVlc3Rpb246MTYw - Quel est aujourd'hui pour vous le problème concret le plus important dans le domaine de l'environnement ?": "Q1", 'QUXVlc3Rpb246MTYx - Que faudrait-il faire selon vous pour apporter des réponses à ce problème ?': "Q2"})

q1q2.head(1)
q1q2['Q1'].value_counts().head(4)
q1q2[q1q2['Q1']=='Les dérèglements climatiques (crue, sécheresse)']
q1q2_lignes_1 = q1q2[q1q2['Q1'].isin(['Les dérèglements climatiques (crue, sécheresse)', 'La biodiversité et la disparition de certaines espèces', "La pollution de l'air", "L'érosion du littoral"])]

q1q2_lignes_1.head(5)
q1q2_lignes_2 = q1q2_lignes_1[pd.notna(q1q2_lignes_1['Q1']) & pd.notna(q1q2_lignes_1['Q2'])]

q1q2_lignes_2.head(5)
le = preprocessing.LabelEncoder()

le.fit(q1q2_lignes_2['Q1'])

le.classes_
q1q2_lignes_2.loc[:,'Q1E'] = le.transform(q1q2_lignes_2['Q1']).tolist()

q1q2_lignes_2.head(3)
# Note: reshape(-1, 1) permet ca, -1 indiquant de conserver le nombre de ligne, cf doc de reshape

# y = q1q2_lignes_2['Q1E'].values.reshape(-1,1)

# ici on fait un one hot encoding

y = to_categorical(q1q2_lignes_2['Q1E'].values)

y, y.shape
X = q1q2_lignes_2['Q2']

X, X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 333)

X_train, X_test, y_train, y_test
max_words = 1001

max_len = 1000

# http://faroit.com/keras-docs/1.2.2/preprocessing/text/

tok = Tokenizer(num_words=max_words)

tok.fit_on_texts(X_train)

sequences = tok.texts_to_sequences(X_train)

# https://keras.io/preprocessing/sequence/

sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
sequences_matrix
def modelgraph_transition_eco():

    """

    Function creating the Emojify-v2 (nope the transition eco) model's graph.

    

    Arguments:

    input_shape -- shape of the input, usually (max_len,)



    Returns:

    model -- a model instance in Keras

    """

    

    ### START CODE HERE ###

    # Define sequences_matrix as the input of the graph, it should be of shape [max_len]

    inputs = Input(name='inputs',shape=[max_len])

    

    # Embedding layer to be learned

    embedding_layer = Embedding(max_words,50,input_length=max_len)(inputs)

    

    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state

    # Be careful, the returned output should be a batch of sequences.

    X = LSTM(units = 128, return_sequences = True)(embedding_layer)

    # Add dropout with a probability of 0.5

    X = Dropout(rate=0.5)(X)

    # Propagate X trough another LSTM layer with 128-dimensional hidden state

    # Be careful, the returned output should be a single hidden state, not a batch of sequences.

    X = LSTM(units = 128)(X)

    # Add dropout with a probability of 0.5

    X = Dropout(rate=0.5)(X)

    # Propagate X through a Dense layer with softmax activation to get back a batch of 5-dimensional vectors.

    X = Dense(4)(X)

    # Add a softmax activation

    X = Activation(activation="softmax")(X)

    

    # Create Model instance which converts sentence_indices into X.

    model = Model(inputs=inputs, outputs=X)

    

    ### END CODE HERE ###

    

    return model
model = modelgraph_transition_eco()

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(sequences_matrix, y_train, epochs = 1, batch_size = 1024, shuffle=True)
model.save('/kaggle/working/modele_lstm_classif_gdte.h5')
modele_lstm_classif_gdte = keras.models.load_model('/kaggle/working/modele_lstm_classif_gdte.h5')

modele_lstm_classif_gdte
sequences_test = tok.texts_to_sequences(X_test)

sequences_matrix_test = sequence.pad_sequences(sequences_test,maxlen=max_len)

sequences_matrix_test
loss, acc = modele_lstm_classif_gdte.evaluate(sequences_matrix_test, y_test)

print()

print("Test accuracy = ", acc)
pred_test = modele_lstm_classif_gdte.predict(sequences_matrix_test)

pred_test
pd.crosstab(np.argmax(y_test, axis=1), np.argmax(pred_test, axis=1), rownames=['Actual'], colnames=['Predicted'], margins=True)