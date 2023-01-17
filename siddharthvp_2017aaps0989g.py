import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
# dfx = pd.read_csv('train.csv')

# dft = pd.read_csv('test.csv')



dfx = pd.read_csv('/kaggle/input/bitsf312-lab1/train.csv')

dft = pd.read_csv('/kaggle/input/bitsf312-lab1/test.csv')



pd.set_option('display.max_rows', 500)



dfx
print(dfx.isin(['?']).sum(axis = 0))



dfx.Size.replace(to_replace = '?', value = 'Medium', inplace = True) # mode

dft.Size.replace(to_replace = '?', value = 'Medium', inplace = True)



dfx['Number of Quantities'].replace(to_replace = '?', value = 0, inplace = True)

dfx['Number of Insignificant Quantities'].replace(to_replace = '?', value = 0, inplace = True)

dfx['Total Number of Characters'].replace(to_replace = '?', value = 0, inplace = True)

dfx['Total Number of Words'].replace(to_replace = '?', value = 0, inplace = True)

dfx['Number of Special Characters'].replace(to_replace = '?', value = 0, inplace = True)

dfx['Total Number of Words'].replace(to_replace = '?', value = 0, inplace = True)

dfx['Difficulty'].replace(to_replace = '?', value = 0, inplace = True)



dft['Number of Quantities'].replace(to_replace = '?', value = 0, inplace = True)

dft['Number of Insignificant Quantities'].replace(to_replace = '?', value = 0, inplace = True)

dfx['Total Number of Characters'].replace(to_replace = '?', value = 0, inplace = True)

dft['Total Number of Words'].replace(to_replace = '?', value = 0, inplace = True)

dft['Number of Special Characters'].replace(to_replace = '?', value = 0, inplace = True)

dft['Total Number of Words'].replace(to_replace = '?', value = 0, inplace = True)

dft['Difficulty'].replace(to_replace = '?', value = 0, inplace = True)



type_transform_map = {

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

}



dfx = dfx.astype(type_transform_map)

dft = dft.astype(type_transform_map)



# dfx['Total Number of Words'].mean()



dfx['Number of Quantities'].replace(to_replace = 0, value = dfx['Number of Quantities'].mean(), inplace = True)

dfx['Total Number of Words'].replace(to_replace = 0, value = dfx['Total Number of Words'].mean(), inplace = True)

dfx['Number of Special Characters'].replace(to_replace = 0, value = dfx['Number of Special Characters'].mean(), inplace = True)

dfx['Total Number of Words'].replace(to_replace = 0, value = dfx['Total Number of Words'].mean(), inplace = True)

dfx['Difficulty'].replace(to_replace = 0, value = dfx['Difficulty'].mean(), inplace = True)



dft['Number of Quantities'].replace(to_replace = 0, value = dft['Number of Quantities'].mean(), inplace = True)

dft['Total Number of Words'].replace(to_replace = 0, value = dft['Total Number of Words'].mean(), inplace = True)

dft['Number of Special Characters'].replace(to_replace = 0, value = dft['Number of Special Characters'].mean(), inplace = True)

dft['Total Number of Words'].replace(to_replace = 0, value = dft['Total Number of Words'].mean(), inplace = True)

dft['Difficulty'].replace(to_replace = 0, value = dft['Difficulty'].mean(), inplace = True)





dfx.drop(['ID', 'Number of Insignificant Quantities'], axis = 1, inplace=True)

dfx = pd.get_dummies(dfx, columns=['Size'], prefix = ['size'])

dft.drop(['ID', 'Number of Insignificant Quantities'], axis = 1, inplace=True)

dft = pd.get_dummies(dft, columns=['Size'], prefix = ['size'])



print(dfx.isin(['?']).sum(axis = 0))



y = dfx.Class

X = dfx.drop(['Class'], axis = 1)
corr = X.corr()



fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

square=True, ax=ax, annot = True)
X.drop(['Total Number of Words','Number of Special Characters'], axis = 1, inplace = True)

dft.drop(['Total Number of Words','Number of Special Characters'], axis = 1, inplace = True)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 11)



X_train = X_train.reset_index().drop(['index'], axis = 1)

X_test = X_test.reset_index().drop(['index'], axis = 1)





X_train
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from tensorflow import keras

from tensorflow.keras.layers import Activation, Dense, Dropout

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout

from tensorflow.keras.models import Sequential



from sklearn import preprocessing

from sklearn.metrics import mean_absolute_error



from tensorflow.keras import layers

from tensorflow.keras.layers import Activation, Dense

from tensorflow.keras.regularizers import l2


model = Sequential()

model.add(Dense(16,input_dim=10, activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(32, activation='relu'))

model.add(Dense(32, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

model.add(Dropout(rate=0.4))

model.add(Dense(16, activation='sigmoid'))

model.add(Dropout(rate=0.2))

model.add(Dense(8, activation='relu'))

model.add(Dense(8, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

model.add(Dropout(rate=0.2))

model.add(Dense(6,activation='softmax'))





from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
callbacks = [ #                 EarlyStopping(monitor='val_loss', patience=2)

];



history = model.fit(x=X_train.values, y=y_train.values, validation_split=0.2, epochs=1000, batch_size=40,

          shuffle=True, callbacks = callbacks)
model.summary()
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
model.save_weights("model.h5")
scores = model.evaluate(X, y, verbose=0)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.get_weights()
predictions = model.predict(dft.values)
predictions
# submission_1 = pd.read_csv('sample_submission.csv')

submission_1 = pd.read_csv('/kaggle/input/bitsf312-lab1/sample_submission.csv')

for i in range(len(submission_1)):

    max_p = 0

    p = -1

    for j in range(6):

        if predictions[i][j] > max_p:

            max_p = predictions[i][j]

            p = j

    submission_1['Class'][i] = p

print(submission_1)

submission_1.to_csv("submission.csv", index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link( df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(submission_1)