import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import torch

import torchvision

from keras.layers import Dense, Dropout

from keras.models import Sequential

from sklearn import preprocessing

from tensorflow import keras

from keras import layers

from keras.layers import Activation, Dense

from keras.regularizers import l2
df2 = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv",  encoding="utf-8",header=0)

df2.head()
df2['Size'].unique()
df2['Size'] = df2['Size'].map({

    'Small': 0, 

    'Medium': 1,

    'Big': 2,

    '?': np.NaN,

    })



for col in df2.columns:

    df2[col] = pd.to_numeric(df2[col], errors='coerce')



df2 = df2.dropna(axis=0)

df2.head(50)
df2.isnull().any()
X = df2.drop("Class", axis= 1)

X = X.drop("ID", axis= 1)

X.columns = [''] * len(X.columns)

y=df2['Class']
df = df2.drop("Class", axis= 1)

df = df.drop("ID", axis= 1)

x = df.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df3 = pd.DataFrame(x_scaled)
df = df2.drop("Class", axis= 1)

df3 = df.drop("ID", axis= 1)

x=df3.values

df3 = pd.DataFrame(x)
df3.head()
import seaborn as sns



corr = df3.corr()



fig, ax = plt.subplots(figsize=(20,10))

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),

square=True, ax=ax, annot = True)
df3 = df3.drop([1,3,5], axis= 1)

df3.head()
X=np.array(df3)

Y=np.array(y)
#from keras.utils.np_utils import to_categorical

#from keras.utils import np_utils

#from sklearn.preprocessing import LabelEncoder

#y_train = to_categorical(y_train, num_classes=6)

#encoder = LabelEncoder()

#encoder.fit(Y)

#Y = encoder.transform(Y)

#Y = np_utils.to_categorical(Y)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test=train_test_split(X,Y,test_size = 0.1, random_state = 1)
x_train.shape, y_train.shape, x_test.shape, y_test.shape

#x_train=X

#y_train=Y

#x_train.shape, y_train.shape
input_dim = len(df2.columns) - 2 - 3



model = Sequential()

model.add(Dense(64, input_dim = input_dim , activation = 'relu'))

model.add(Dense(32, activation = 'sigmoid'))

#model.add(Dropout(0.3))

model.add(Dense(16, activation = 'sigmoid'))

model.add(layers.Dense(16, use_bias = False))

model.add(layers.BatchNormalization())

model.add(Activation("relu"))

model.add(Dense(128, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

#model.add(Dense(128, activation = 'relu'))

#model.add(Dropout(0.3))

#model.add(Dense(64, activation = 'relu'))

#model.add(layers.Dense(64, use_bias = False))

#model.add(layers.BatchNormalization())

#model.add(Activation("relu"))

#model.add(Dropout(0.2))

#model.add(Dense(32, activation = 'relu'))

model.add(layers.Dense(32, use_bias = False))

model.add(layers.BatchNormalization())

model.add(Activation("relu"))

#model.add(Dense(32, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

model.add(Dense(16, activation = 'relu'))

model.add(Dense(6, activation = 'softmax'))
input_dim = len(df2.columns) - 2 -3 

model = Sequential()

model.add(Dense(16, input_dim = input_dim , activation = 'relu'))

model.add(Dense(16, kernel_regularizer=l2(0.02), bias_regularizer=l2(0.02)))

model.add(Dense(32, activation = 'relu'))

model.add(Dense(32, activation = 'relu'))

model.add(Dropout(0.3))

model.add(layers.Dense(32, use_bias = False))

model.add(layers.BatchNormalization())

model.add(Activation("relu"))

model.add(Dense(16, activation = 'relu'))

model.add(Dense(16, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

model.add(Dropout(0.4))

model.add(Dense(8, activation = 'relu'))

model.add(Dense(6, activation = 'softmax'))
input_dim = len(df2.columns) - 2 - 3



model = Sequential()

model.add(Dense(512, input_dim = input_dim , activation = 'relu'))

#model.add(Dense(1024, activation = 'sigmoid'))

model.add(Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

model.add(Dropout(0.2))

#model.add(Dense(512, activation = 'sigmoid'))

model.add(Dense(256, activation = 'sigmoid'))

model.add(Dropout(0.4))

model.add(layers.Dense(256, use_bias = False))

model.add(layers.BatchNormalization())

model.add(Activation("relu"))

model.add(Dropout(0.4))

#model.add(Dense(128, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

model.add(Dense(128, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

model.add(Dense(128, activation = 'relu'))

model.add(Dropout(0.4))

#model.add(Dense(64, activation = 'relu'))

model.add(Dense(64, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

model.add(layers.Dense(64, use_bias = False))

model.add(layers.BatchNormalization())

model.add(Activation("relu"))

model.add(Dropout(0.4))

#model.add(Dense(32, activation = 'relu'))

model.add(layers.Dense(32, use_bias = False))

model.add(layers.BatchNormalization())

model.add(Activation("relu"))

model.add(Dense(32, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

model.add(Dropout(0.4))

model.add(Dense(16, activation = 'relu'))

model.add(Dense(6, activation = 'softmax'))
from keras import optimizers

sgd=keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False)

adam=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )
from keras.callbacks import EarlyStopping, ModelCheckpoint



callbacks = [EarlyStopping(monitor='val_loss', patience=20),

            ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
#history=model.fit(x_train, y_train, epochs = 200, batch_size = 20)

#history=model.fit(x_train, y_train, epochs=250, validation_split=0.2,batch_size=10)

#history=model.fit(x_train, y_train, validation_split=0.2, epochs=150,callbacks=callbacks,verbose=1)
input_dim = len(df2.columns) - 2 - 3

model = Sequential()

model.add(Dense(16,input_dim=input_dim, activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(32, activation='relu'))

model.add(Dense(64, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

model.add(Dropout(rate=0.4))

model.add(Dense(32, activation='sigmoid'))

model.add(layers.Dense(16, use_bias = False))

model.add(layers.BatchNormalization())

model.add(Dropout(rate=0.2))

#model.add(Activation("relu"))

model.add(Dense(16, activation='relu'))

model.add(Dense(8, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

model.add(Dropout(rate=0.2))

model.add(Dense(6,activation='softmax'))
input_dim = len(df2.columns) - 2 - 3

model = Sequential()

model.add(Dense(16,input_dim=input_dim, activation='relu'))

model.add(Dropout(rate=0.2))

model.add(Dense(32, activation='relu'))

model.add(Dense(32, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

model.add(Dropout(rate=0.2))

model.add(Dense(16, activation='sigmoid'))

model.add(Dropout(rate=0.2))

model.add(Dense(8, activation='relu'))

model.add(Dense(8, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))

model.add(Dropout(rate=0.2))

model.add(Dense(6,activation='softmax'))
#model = Sequential()

#model.add(Dense(7, input_dim=8, activation='relu'))

#model.add(Dropout(rate=0.2))

#model.add(Dense(8, activation='relu'))

#model.add(Dense(6, activation='softmax'))





#from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

#from keras.callbacks import EarlyStopping, ModelCheckpoint



callbacks = [EarlyStopping(monitor='val_loss', patience=200),ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]



model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



history = model.fit(x_train, y_train, validation_split=0.2, epochs=2000, batch_size=40,shuffle=True)
#from keras.callbacks import ReduceLROnPlateau



#earlyStopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='min')

#mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss')

#reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=70, verbose=1, epsilon=1e-4, mode='min')

#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#history=model.fit(x_train, y_train, batch_size=40, epochs=1000, verbose=0, callbacks=[mcp_save], validation_split=0.2)
from matplotlib import pyplot



_, train_acc = model.evaluate(x_train, y_train, verbose=0)

_, test_acc = model.evaluate(x_test, y_test, verbose=0)

print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# plot loss during training

pyplot.subplot(211)

pyplot.title('Loss')

pyplot.plot(history.history['loss'], label='train')

pyplot.plot(history.history['val_loss'], label='test')

pyplot.legend()

# plot accuracy during training

pyplot.subplot(212)

pyplot.title('Accuracy')

pyplot.plot(history.history['accuracy'], label='train')

pyplot.plot(history.history['val_accuracy'], label='test')

pyplot.legend()

pyplot.show()
# demonstration of calculating metrics for a neural network model using sklearn

from sklearn.datasets import make_circles

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import f1_score

from sklearn.metrics import cohen_kappa_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix

from keras.models import Sequential

from keras.layers import Dense

 



# predict probabilities for test set

yhat_probs = model.predict(x_test, verbose=0)

# predict crisp classes for test set

yhat_classes = model.predict_classes(x_test, verbose=0)

# reduce to 1d array

yhat_probs = yhat_probs[:, 0]

#y_test=[np.where(r==1)[0][0] for r in y_test]

y_test=np.asarray(y_test)

print(y_test)

print(yhat_classes)

#yhat_classes = yhat_classes[:, 0]



# accuracy: (tp + tn) / (p + n)

accuracy = accuracy_score(y_test, yhat_classes)

print('Accuracy: %f' % accuracy)

# precision tp / (tp + fp)

precision = precision_score(y_test, yhat_classes, pos_label='positive',

                                           average='micro')

print('Precision: %f' % precision)

# recall: tp / (tp + fn)

recall = recall_score(y_test, yhat_classes, pos_label='positive',

                                           average='micro')

print('Recall: %f' % recall)

# f1: 2 tp / (2 tp + fp + fn)

f1 = f1_score(y_test, yhat_classes, pos_label='positive',

                                           average='micro')

print('F1 score: %f' % f1)	

# kappa

kappa = cohen_kappa_score(y_test, yhat_classes)

print('Cohens kappa: %f' % kappa)

# ROC AUC

#auc = roc_auc_score(y_test, yhat_probs, multi_class='ovo')

#print('ROC AUC: %f' % auc)

# confusion matrix

#matrix = confusion_matrix(y_test, yhat_classes)

#print(matrix)
scores = model.evaluate(x_test,y_test)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
model.summary()
df1 = pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv",  encoding="utf-8",header=0)

df1.head()

df1.isnull().any()
df1['Size'] = df1['Size'].map({

    'Small': 0, 

    'Medium': 1,

    'Big': 2,

    })



df1.isnull().any()
x = df1.drop("ID", axis= 1)

x.columns = [''] * len(x.columns)

print(x)
X = x.values #returns a numpy array

#min_max_scaler = preprocessing.MinMaxScaler()

#x_scaled = min_max_scaler.fit_transform(X)

X_test = pd.DataFrame(X)

X_test = X_test.drop([1,3,5], axis= 1)
X_test.head()
x_test=np.array(X_test)
print(x_test.shape)
prediction = model.predict(x_test)

b = np.zeros_like(prediction)

b[np.arange(len(prediction)), prediction.argmax(1)] = 1

Prediction=[np.where(r==1)[0][0] for r in b]

df = pd.read_csv('/kaggle/input/bitsf312-lab1/sample_submission.csv')
df["Class"]=Prediction
print(df)
df.to_csv('mycsvfile.csv',index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df)