import matplotlib.pyplot as plt
import sys,os
import numpy as np
from sklearn import datasets, preprocessing, metrics, cross_validation
from sklearn.decomposition import PCA
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint
from sklearn.pipeline import Pipeline
from keras.callbacks import EarlyStopping 
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD,Adam,Adamax,Nadam,Adadelta
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
#from keras.regularizers import l1l2
from sklearn.preprocessing import LabelEncoder

train_df = pd.read_csv('../input/train.csv')
train_df.fillna(0,axis=1,inplace=True)
train_df.shape

#df1=train_df.loc[train_df['Target'] ==1]
#df2=train_df.loc[train_df['Target'] ==2]
#df3=train_df.loc[train_df['Target'] ==3]
#df4=train_df.loc[train_df['Target'] ==4]    

#rows=int(df1.shape[0]+df2.shape[0]+df3.shape[0])/3

#df4=df4.sample(rows+1000, replace=True)


#train_df=pd.concat([df1,df3,df2,df4],axis=0)
train_df = shuffle(train_df)
print (train_df.shape)
test_df = pd.read_csv('../input/test.csv')
test_df.fillna(0,axis=1,inplace=True)
test_df.shape
plt.figure(figsize=(12, 5))
plt.hist(train_df.Target.values, bins=4)
plt.title('Histogram target counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()
columns_to_use = train_df.columns[1:-1]
y = train_df['Target'].values
train_test_df = pd.concat([train_df[columns_to_use], test_df[columns_to_use]], axis=0)
train_test_df.fillna(0,axis=1,inplace=True)
cols = [f_ for f_ in train_test_df.columns if train_test_df[f_].dtype == 'object']

for col in cols:
    le = LabelEncoder()
    le.fit(train_test_df[col].astype(str))
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
del le

train_df.drop(['Id','Target'],axis=1,inplace=True)
test_df.drop(['Id'],axis=1,inplace=True)

scaler = MinMaxScaler()

df_train_scaled = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns)
df_test_scaled = pd.DataFrame(scaler.fit_transform(test_df), columns=test_df.columns)
print (df_train_scaled.shape)
print (df_test_scaled.shape)
df_test_scaled=df_test_scaled.values
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(
       train_df, y, test_size=0.20, random_state=42)
y_train = np_utils.to_categorical(y_train)  
y_valid = np_utils.to_categorical(y_valid)  

#Build the model Here , Following is open playground to play on 

model = Sequential()
#Base Model
model.add(Dense(64, input_dim=X_train.shape[1], init='uniform', activation='relu'))
model.add(Dropout(0.20))
model.add(BatchNormalization(moving_mean_initializer='zeros',momentum=0.9))
model.add(Dense(32 ,init='uniform', activation='relu'))
model.add(BatchNormalization(moving_mean_initializer='zeros',momentum=0.9))
model.add(Dropout(0.20))
#model.add(Dense(128, init='uniform', activation='relu'))
#model.add(BatchNormalization(moving_mean_initializer='zeros',momentum=0.9))
#model.add(Dropout(0.20))
model.add(Dense(y_train.shape[1], init='uniform', activation='softmax'))
adam=Adam(lr=1e-3, decay=1e-6)

model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='max')
callbacks = [EarlyStopping(monitor='val_loss', patience=3, verbose=0),checkpoint]   

history=model.fit(X_train, y_train,batch_size=100,validation_data=(X_valid,y_valid), epochs=50,shuffle=True,callbacks=callbacks)
output=model.predict_proba(df_test_scaled)
predicted_probs = np.argmax(output,axis=1)
preds=pd.Series(predicted_probs)

#Some visualization to see model has any overfitting (Though it cant as this got handled by early stopping criteria)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('loss')
plt.ylabel('val_loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission['Target'] = preds
sample_submission.to_csv('DNN_submission.csv', index=False)
sample_submission.head()