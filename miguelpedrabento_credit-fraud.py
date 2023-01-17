# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import pandas as pd 

import numpy as np

import seaborn as sns



np.random.seed(203)



df = pd.read_csv("../input/creditcard.csv")
df.head()
from sklearn.preprocessing import StandardScaler
True in df.isnull().values
sns.set_style("ticks")

sns.countplot(x= 'Class', data= df)

plt.title("Fraud class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")

plt.show()
print('Frauds: {0} \nNon-frauds: {1} '.format(df['Class'].value_counts()[1],df['Class'].value_counts()[0]))

print('Fraud percentage: {0:.3f}%'.format(100*df['Class'].value_counts()[1]/(df['Class'].value_counts()[1]+df['Class'].value_counts()[0])))
df[['Time','Class']].corr()
df = df.drop('Time', axis= 1)
sc = StandardScaler()

sc.fit(df['Amount'].values.reshape(-1,1))

df['scAmount'] = sc.transform(df['Amount'].values.reshape(-1,1))

df = df.drop('Amount', axis= 1)

df.head(3)
df_sam0 = df.loc[df['Class']==0].sample(492)

df_sam1 = df.loc[df['Class']==1].sample(492)
df1 = pd.concat([df_sam0,df_sam1])
cols = df1.columns.tolist()

cols = cols[-1:] + cols[:-1]

df1 = df1[cols]

df1 = df1.sort_index()

df1.head()
x = df1.loc[:,'scAmount':'V28'].values

y = df1[['Class']].values.reshape(-1,1)

x.shape[1]
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras.optimizers import SGD, Adam

from keras import regularizers
model = Sequential()

model.add(Dense(64, input_shape=(x.shape[1],),activation = 'tanh',use_bias = True))

Dropout(.2)

model.add(Dense(32,activation = 'tanh',use_bias = True))

Dropout(.2)

model.add(Dense(16,activation = 'tanh',use_bias = True))

Dropout(.2)

model.add(Dense(1,activation = 'sigmoid',use_bias = True))



model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00001), metrics=['accuracy'])
history = model.fit(x,y,validation_split=0.2, epochs=30, batch_size=1, verbose=2)
plt.figure(figsize=(15,4))

plt.subplot(121)

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Fraud')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(122)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Fraud')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

conf = confusion_matrix(y, np.round(model.predict(x)), labels=None, sample_weight=None)
sns.heatmap(conf, annot=True, fmt="d")

plt.show()
fpr, tpr, thresholds = roc_curve(y, model.predict(x))
plt.plot(fpr,tpr)

plt.show()
roc_auc_score(y, model.predict(x))
from sklearn.model_selection  import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=42)
model = Sequential()

model.add(Dense(64, input_shape=(x.shape[1],),activation = 'tanh',use_bias = True))

Dropout(.2)

model.add(Dense(32,activation = 'tanh',use_bias = True))

Dropout(.2)

model.add(Dense(16,activation = 'tanh',use_bias = True))

Dropout(.2)

model.add(Dense(1,activation = 'sigmoid',use_bias = True))



model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.00001), metrics=['accuracy'])
history1 = model.fit(x_train,y_train,validation_split=0.2, epochs=40, batch_size=1, verbose=2)
plt.figure(figsize=(15,4))

plt.subplot(121)

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Fraud')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(122)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Fraud')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
fpr, tpr, thresholds = roc_curve(y_test, model.predict(x_test))

plt.plot(fpr,tpr)

plt.show()
thresholds, fpr, tpr
roc_auc_score(y_test, model.predict(x_test))
sns.heatmap(confusion_matrix(y_test, np.round(model.predict(x_test)), labels=None, sample_weight=None), annot=True, fmt="d")

plt.show()
from keras.layers import Input, LeakyReLU

from keras.models import Model

from keras.optimizers import Adam



## input layer 

input_layer = Input(shape=(x.shape[1],))



## encoding part

encoded = Dense(25)(input_layer)

LeakyReLU(alpha=0.3)

encoded = Dense(15)(encoded)

LeakyReLU(alpha=0.3)



## decoding part

decoded = Dense(15)(encoded)

LeakyReLU(alpha=0.3)

decoded = Dense(25)(decoded)

LeakyReLU(alpha=0.3)



## output layer

output_layer = Dense(x.shape[1], )(decoded)

LeakyReLU(alpha=0.3)



autoencoder = Model(input_layer, output_layer)

autoencoder.compile(metrics=['accuracy'], optimizer=Adam(lr=0.001, decay= 0.0001), loss="mse")
history = autoencoder.fit(x,x, batch_size = 1, epochs = 50, shuffle = True, validation_split = 0.20, verbose=1);
plt.figure(figsize=(15,4))

plt.subplot(121)

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Fraud')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(122)

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Fraud')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
x_auto = autoencoder.predict(x)
df2 = pd.DataFrame(data=np.concatenate((x_auto,y), axis=1))

df2.columns = df1.columns
plt.figure(figsize=(20,8))

plt.subplot(121)

plt.title('Autoencoder')

sns.heatmap(df2.corr(), cmap="RdBu_r")

plt.subplot(122)

plt.title('Real')

sns.heatmap(df1.corr(), cmap="RdBu_r")

plt.show()
df2.head()
df1.head()
df1.columns
df2.columns = df1.columns
df2.head()
df1.head()
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor(n_estimators= 100)
rfr.fit(x_train,y_train.ravel())

rfr.score(x_train,y_train.ravel())
imp = rfr.feature_importances_
feature_imp = pd.DataFrame(data=imp,  index=df1.columns[:-1], columns= ['Importance'])

feature_imp = feature_imp.sort_values(by=['Importance'], ascending=False)

feature_imp
f_imp = feature_imp.iloc[:,0]

plt.figure(figsize=(15,8))

plt.bar(np.arange(len(f_imp)),f_imp)

plt.xticks(np.arange(len(f_imp)),feature_imp.index,rotation=90)

plt.show()
y_rfr = np.round(rfr.predict(x_test)).reshape(-1,1)
sns.heatmap(confusion_matrix(y_test, y_rfr, labels=None, sample_weight=None), annot=True, fmt="d")

plt.show()