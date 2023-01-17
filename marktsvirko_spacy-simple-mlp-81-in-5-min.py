# Import necessary libraries

import numpy as np

import pandas as pd

np.random.seed(2020)

import spacy



from sklearn.model_selection import train_test_split



# Import elements for multilayer percepton 

from keras import Sequential

from keras.layers import Dense, Dropout

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
df = pd.read_csv('../input/nlp-getting-started/train.csv')

df.head()
df.info()
df['target'].value_counts()
# Fill empties (anyway now I will use only text column)

df[['keyword', 'location']] = df[['keyword', 'location']].astype('str')



df.info()
# Load core model

nlp = spacy.load('en_core_web_lg')
# Convert tweets into vectors 

with nlp.disable_pipes():

    text_vectors = np.array([nlp(text).vector for text in df['text']])  
labels = df['target']
# Split data and labels on train and valid set

X_train, X_valid, y_train, y_valid = train_test_split(text_vectors, labels, test_size=0.2, random_state=3)
# Input shape for model

input_shape = text_vectors.shape[1]



# Set callbacks to avoid overfitting

callbacks = [EarlyStopping(patience=3), ReduceLROnPlateau(patience=2)]
# Construct multilayer net



mlp_model = Sequential([

    Dense(64, input_dim=input_shape, activation='relu'),

    Dropout(0.2),

    Dense(32, activation='relu'),

    Dense(1, activation='sigmoid'),   

])



mlp_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train model

mlp_model.fit(X_train, y_train,

             batch_size=32,

             epochs=30, 

             verbose=2,

             callbacks=callbacks,

             validation_data=(X_valid, y_valid))
# Check model on test validation_set

test_score = mlp_model.evaluate(X_valid, y_valid)

print(test_score)
# Factorize tweets from test part

test = pd.read_csv('../input/nlp-getting-started/test.csv')



with nlp.disable_pipes():

    test_text_vectors = np.array([nlp(text).vector for text in test['text']])
# Predict targets for test data 

test_labels = mlp_model.predict_classes(test_text_vectors)



# ...and write result to csv

submit_data = pd.DataFrame(test['id'])

submit_data['target'] = test_labels

submit_data.to_csv('nlp_submit.csv', index=False)