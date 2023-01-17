import numpy as np

import pandas as pd

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("/kaggle/input/bitsf312-lab1/train.csv", sep=',')
df.head()
mask = df['Size'] != '?'

for col in df:

    mask = mask & (df[col] != '?')

    

df3 = df[~mask]

#mask.value_counts()

df = df[mask]
cols = df.columns.drop(['ID', 'Difficulty', 'Score', 'Size'])

df[cols] = df[cols].apply(pd.to_numeric)

df['Difficulty'] = df['Difficulty'].astype(float) 
df3['Size'] = df3['Size'].replace(to_replace='?', value = 'Medium')

for col in df3.columns.drop(['Size']):

    df3[col] = df3[col].replace(to_replace='?', value = df.loc[:, col].median())
df = df.append(df3).sort_values('ID')

#Got a nice clean dataset with all values
df_train = df.drop(['Class'], axis =1)
df_test = pd.read_csv("/kaggle/input/bitsf312-lab1/test.csv", sep=',')
df_proc = df_train.append(df_test).reset_index(drop = True)
df_proc
one_hot = pd.get_dummies(df_proc['Size'],columns = 'list_like')

df_proc = df_proc.join(one_hot).drop(['Size'], axis = 1)

df_proc
df_fin = df_proc.drop(['ID'], axis = 1)

df_fin = df_fin.reset_index().drop('index', axis = 1)



#Drop all with a correlation with 1

dropCols = ['Number of Insignificant Quantities','Number of Sentences', 'Total Number of Words']

df_fin = df_fin.drop(dropCols, axis = 1)
inps = df_fin.columns#Just to train the model, number of inputs I am using

cols = df_fin.columns.drop(['Difficulty', 'Score'])

df_fin[cols] = df_fin[cols].apply(pd.to_numeric)

df_fin['Difficulty'] = df_fin['Difficulty'].astype(float) 

df_fin.info()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

to_scale_m = df_fin.columns.drop(['Big', 'Small', 'Medium'])

#to_scale_m = ['Score']

df_fin[to_scale_m] = 10*scaler.fit_transform(df_fin[to_scale_m].to_numpy())

df_fin.describe()
#Preprocessed and now split data again

df_train = df_fin[:371]

df_test1 = df_fin[371:]
df_classes = pd.get_dummies(df['Class'],columns = 'list_like')

df_classes
#Train test split

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test,train_val, test_val=train_test_split(df_train,df_classes, df['Class'])



x_train = x_train.reset_index().drop('index', axis = 1)

y_train = y_train.reset_index().drop('index', axis = 1)

train_val = train_val.reset_index().drop('index', axis = 1)



x_test = x_test.reset_index().drop('index', axis = 1)

y_test = y_test.reset_index().drop('index', axis = 1)

test_val = test_val.reset_index().drop('index', axis = 1)



x_train.shape, x_test.shape, y_train.shape, y_test.shape,train_val.shape, test_val.shape
train_val['Class'].value_counts()
#Gonna try over-sampling here

from sklearn.utils import resample

from sklearn.utils import shuffle



train_data = x_train.join([train_val, y_train])



samples_0 = train_data[train_data.Class==0]

samples_1 = train_data[train_data.Class==1]

samples_2 = train_data[train_data.Class==2]

samples_3 = train_data[train_data.Class==3]

samples_4 = train_data[train_data.Class==4]

samples_5 = train_data[train_data.Class==5]



#samples_0 = resample(samples_0,replace=True,n_samples=80)

samples_1 = resample(samples_1,replace=True,n_samples=40)

#samples_2 = resample(samples_2,replace=True,n_samples=60)

samples_3 = resample(samples_3,replace=True,n_samples=45)

samples_4 = resample(samples_4,replace=True,n_samples=45)

#samples_5 = resample(samples_5,replace=True,n_samples=70)



upsampled = pd.concat([samples_0, samples_1, samples_2,samples_3,samples_4,samples_5])

upsampled = shuffle(upsampled).reset_index(drop=True)



upsampled
x_train = upsampled[x_test.columns]

train_val = upsampled['Class']

y_train = upsampled[y_test.columns]

train_val.value_counts()
from keras.layers import Dense, Dropout, PReLU, LeakyReLU, BatchNormalization

from keras.models import Sequential

from keras.regularizers import l2
# model = Sequential()

# #model.add(Dense(32,input_dim=len(inps), activation='relu'))

# #model.add(Dropout(rate = 0.2))

# #model.add(Dense(20, activation='relu'))

# ##model.add(Dropout(rate = 0.5))

# #model.add(Dense(80, activation='relu',kernel_regularizer=l2(0.01)))

# #model.add(Dropout(rate = 0.2))

# #model.add(Dense(20, activation='relu', kernel_regularizer=l1(0.01)))

# #model.add(Dense(6,  activation='softmax', kernel_regularizer=l1(0.05)))



model = Sequential()

model.add(Dense(64,input_dim=len(inps), activation='tanh'))#, use_bias = False))

# model.add(BatchNormalization())

model.add(Dropout(rate = 0.2))

# model.add(PReLU())



model.add(Dense(32,  activation='relu',kernel_regularizer=l2(0.01)))

model.add(Dropout(rate = 0.1))



model.add(Dense(16, activation='relu',kernel_regularizer=l2(0.01)))

# model.add(BatchNormalization())

model.add(Dropout(rate = 0.2))

# model.add(LeakyReLU(alpha = 0.5))



model.add(Dense(8, activation='relu',kernel_regularizer=l2(0.001)))

model.add(Dropout(rate = 0.2))



#OUTPUT LAYER

model.add(Dense(6,  activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
# from keras.callbacks import EarlyStopping, ModelCheckpoint

# callbacks = [EarlyStopping(monitor='val_loss', patience=50)]
# Fit the training set on the model

history = model.fit(x_train, y_train, validation_split=0.1, epochs=200, batch_size = 32,class_weight = {0:1, 1:2, 2:1, 3:1.5, 4:1.2, 5:1})#, callbacks=callbacks)
plt.subplot(121)

plt.plot(range(len(history.history['accuracy'])), history.history['accuracy'],'r')

plt.plot(range(len(history.history['accuracy'])), history.history['val_accuracy'], 'b')

plt.subplot(122)

plt.plot(range(len(history.history['accuracy'])), history.history['loss'],'r')

plt.plot(range(len(history.history['accuracy'])), history.history['val_loss'], 'b')
prediction = model.predict(x_test)

prediction = np.argmax(prediction, axis=1)

np.vstack((test_val.to_numpy().transpose(), prediction))
from sklearn.metrics import accuracy_score

accuracy_score(test_val, prediction)

#PRINTING FINAL PREDICTION ON TESTING DATA
finalPre = model.predict(df_test1)

finalPre = np.argmax(finalPre, axis=1)

df_test.join(pd.Series(finalPre, name='Class'))[['ID', "Class"]].to_csv('cleanData.csv',index=False)

finalPre
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "cleanData.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

create_download_link(df)