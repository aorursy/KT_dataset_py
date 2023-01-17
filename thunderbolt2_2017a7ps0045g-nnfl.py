import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
dtrain = pd.read_csv('/kaggle/input/bitsf312-lab1/train.csv')

dtest = pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv')

dtrain
dtrain.Size.replace(to_replace = '?', value = 'Medium', inplace = True) # mode

dtest.Size.replace(to_replace = '?', value = 'Medium', inplace = True)



dtrain['Number of Quantities'].replace(to_replace = '?', value = 0, inplace = True)

dtrain['Number of Insignificant Quantities'].replace(to_replace = '?', value = 0, inplace = True)

dtrain['Total Number of Characters'].replace(to_replace = '?', value = 0, inplace = True)

dtrain['Total Number of Words'].replace(to_replace = '?', value = 0, inplace = True)

dtrain['Number of Special Characters'].replace(to_replace = '?', value = 0, inplace = True)

dtrain['Total Number of Words'].replace(to_replace = '?', value = 0, inplace = True)

dtrain['Difficulty'].replace(to_replace = '?', value = 0, inplace = True)



dtest['Number of Quantities'].replace(to_replace = '?', value = 0, inplace = True)

dtest['Number of Insignificant Quantities'].replace(to_replace = '?', value = 0, inplace = True)

dtrain['Total Number of Characters'].replace(to_replace = '?', value = 0, inplace = True)

dtest['Total Number of Words'].replace(to_replace = '?', value = 0, inplace = True)

dtest['Number of Special Characters'].replace(to_replace = '?', value = 0, inplace = True)

dtest['Total Number of Words'].replace(to_replace = '?', value = 0, inplace = True)

dtest['Difficulty'].replace(to_replace = '?', value = 0, inplace = True)



# dtrain = dtrain.astype({

#     'ID':int,

#     'Number of Quantities':int,

#     'Number of Insignificant Quantities':int,

#     'Total Number of Words': int,

#     'Number of Special Characters': int,

#     'Difficulty': float,

#     'Score': int

# })

# dtest = dtest.astype({

#     'ID':int,

#     'Number of Quantities':int,

#     'Number of Insignificant Quantities':int,

#     'Total Number of Words': int,

#     'Number of Special Characters': int,

#     'Difficulty': float,

#     'Score': int

# })



dtrain = dtrain.astype({

    'ID':int,

    'Number of Quantities':int,

    'Number of Insignificant Quantities':int,

    'Size':str,

    'Total Number of Characters': int,

    'Total Number of Words': int,

    'Number of Special Characters': int,

    'Number of Sentences': int,

    'First Index': int,

    'Second Index':int,

    'Difficulty': float,

    'Score' : float,

})



dtest = dtest.astype({

    'ID':int,

    'Number of Quantities':int,

    'Number of Insignificant Quantities':int,

    'Size':str,

    'Total Number of Characters': int,

    'Total Number of Words': int,

    'Number of Special Characters': int,

    'Number of Sentences': int,

    'First Index': int,

    'Second Index':int,

    'Difficulty': float,

    'Score' : float,

})





dtest['Number of Quantities']=dtest['Number of Quantities'];

dtrain['Number of Quantities']=dtrain['Number of Quantities'];



dtrain['Number of Quantities'].replace(to_replace = 0, value = dtrain['Number of Quantities'].mean(), inplace = True)

dtrain['Total Number of Words'].replace(to_replace = 0, value = dtrain['Total Number of Words'].mean(), inplace = True)

dtrain['Number of Special Characters'].replace(to_replace = 0, value = dtrain['Number of Special Characters'].mean(), inplace = True)

dtrain['Total Number of Words'].replace(to_replace = 0, value = dtrain['Total Number of Words'].mean(), inplace = True)

dtrain['Difficulty'].replace(to_replace = 0, value = dtrain['Difficulty'].mean(), inplace = True)



dtest['Number of Quantities'].replace(to_replace = 0, value = dtest['Number of Quantities'].mean(), inplace = True)

dtest['Total Number of Words'].replace(to_replace = 0, value = dtest['Total Number of Words'].mean(), inplace = True)

dtest['Number of Special Characters'].replace(to_replace = 0, value = dtest['Number of Special Characters'].mean(), inplace = True)

dtest['Total Number of Words'].replace(to_replace = 0, value = dtest['Total Number of Words'].mean(), inplace = True)

dtest['Difficulty'].replace(to_replace = 0, value = dtest['Difficulty'].mean(), inplace = True)







# dft['CharacterperWords']=dft['Total Number of Characters'].divide(dft['Total Number of Words'])

# dft['CharacterperSentences']=dft['Total Number of Characters'].divide(dft['Number of Sentences'])

# dft['WordsperSentences']=dft['Total Number of Words'].divide(dft['Number of Sentences'])

# dft['Difficulty_score']=1/((dft['Difficulty'])**10)



# dfx['CharacterperWords']=dfx['Total Number of Characters'].divide(dfx['Total Number of Words'])

# dfx['CharacterperSentences']=dfx['Total Number of Characters'].divide(dfx['Number of Sentences'])

# dfx['WordsperSentences']=dfx['Total Number of Words'].divide(dfx['Number of Sentences'])

# dfx['Difficulty_score']=1/((dfx['Difficulty'])**10)





dtrain.drop(['ID', 'Number of Insignificant Quantities'], axis = 1, inplace=True)

dtrain = pd.get_dummies(dtrain, columns=['Size'], prefix = ['size'])

dtest.drop(['ID', 'Number of Insignificant Quantities'], axis = 1, inplace=True)

dtest = pd.get_dummies(dtest, columns=['Size'], prefix = ['size'])

print(dtrain.isin(['?']).sum(axis = 0))



y = dtrain.Class

X = dtrain.drop(['Class'], axis = 1)
corr = X.corr()



fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

square=True, ax=ax, annot = True)
X.drop(['Total Number of Words','Number of Special Characters'], axis = 1, inplace = True)

dtest.drop(['Total Number of Words','Number of Special Characters'], axis = 1, inplace = True)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 11)



X_train = X_train.reset_index().drop(['index'], axis = 1)

X_test = X_test.reset_index().drop(['index'], axis = 1)



numerical_columns = ["Number of Quantities","Total Number of Characters","Number of Sentences","First Index","Second Index",

                     "Difficulty","Score"]

numerical_df_train = pd.DataFrame(X_train[numerical_columns])

rest_df_train = X_train.drop(numerical_columns, axis = 1)

numerical_df_test = pd.DataFrame(X_test[numerical_columns])

rest_df_test = X_test.drop(numerical_columns, axis = 1)



numerical_dtest = pd.DataFrame(dtest[numerical_columns])

rest_dtest = dtest.drop(numerical_columns, axis = 1)



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler



# scaler = MinMaxScaler()

# numerical_df_train_scaled = pd.DataFrame(scaler.fit_transform(numerical_df_train),columns=numerical_columns)

# numerical_df_test_scaled = pd.DataFrame(scaler.fit_transform(numerical_df_test),columns=numerical_columns)



# numerical_dtest_scaled = pd.DataFrame(scaler.fit_transform(numerical_dtest), columns=numerical_columns)



X_train = pd.concat([numerical_df_train, rest_df_train], axis=1)

X_test = pd.concat([numerical_df_test, rest_df_test], axis=1)

dtest = pd.concat([numerical_dtest, rest_dtest], axis=1)



X_train
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from tensorflow import keras

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.models import Sequential

from sklearn.metrics import mean_absolute_error

from keras.layers import Dense, Dropout

from keras.models import Sequential

from sklearn import preprocessing

from tensorflow import keras

import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from keras import layers

from keras.layers import Activation, Dense

from keras.regularizers import l2


model = Sequential()

model.add(Dense(16,input_dim=10, activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(32, activation='relu'))

model.add(Dense(32, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

model.add(Dropout(rate=0.4))

model.add(Dense(16, activation='sigmoid'))

#model.add(layers.Dense(16, use_bias = False))

#model.add(layers.BatchNormalization())

model.add(Dropout(rate=0.2))

#model.add(Activation("relu"))

model.add(Dense(8, activation='relu'))

model.add(Dense(8, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

model.add(Dropout(rate=0.2))

model.add(Dense(6,activation='softmax'))





from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x=X_train.values, y=y_train.values, validation_split=0.2, epochs=1000, batch_size=40,

          shuffle=True,

          callbacks = [

#                 EarlyStopping(monitor='val_loss', patience=2)

          ])
model.summary()
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
scores = model.evaluate(X, y, verbose=0)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predictions = model.predict_classes(dtest.values)
predictions
sub = pd.read_csv("/kaggle/input/bitsf312-lab1/sample_submission.csv")

sub['Class']=predictions

print(sub)

sub.to_csv("submission.csv", index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(sub, title = "Download CSV file", filename = "data.csv"):

    csv = sub.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(sub)