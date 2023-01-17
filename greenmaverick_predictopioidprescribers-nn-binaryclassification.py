###################################
# Author: Abhijay
# Created: 29th Jan 18
# Last modified date: 31st Jan 18
###################################

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
prescribers = pd.read_csv('../input/prescriber-info.csv')
# prescribers.shape
prescribers.head()
prescribers.describe()
# prescribers.columns
# len(prescribers['Specialty'].unique())
specialty = pd.DataFrame(prescribers.groupby(['Specialty']).count()['NPI']).sort_values('NPI')
# specialty.loc[specialty['NPI']<40].shape
rareSpecialty = list(specialty.loc[specialty['NPI']<40].index)
prescribers.loc[prescribers['Specialty'].isin(rareSpecialty),'Specialty'] = prescribers.loc[prescribers['Specialty'].isin(rareSpecialty),'Specialty'].apply(lambda x: 'Surgery' if 'Surgery' in list(x.split( )) else 'Other')
# Credentials
Credentials = pd.DataFrame(prescribers.groupby(['Credentials']).count()['NPI']).reset_index(False)
Credentials[Credentials['NPI']<20]
prescribersData = prescribers.drop( ['NPI','Credentials'], axis=1)
prescribersData.head()
prescribersData = pd.get_dummies(prescribersData, columns=['Gender','Specialty','State'], drop_first=True)
# len(prescribersData.columns)
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load dataset
X = prescribersData.drop(['Opioid.Prescriber'],axis=1).values.astype(float)
Y = prescribersData['Opioid.Prescriber'].values
# # encode class values as integers
# encoder = LabelEncoder()
# encoder.fit(Y)
# encoded_Y = encoder.transform(Y)
# ## See and remove IMP
def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=354, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
# # evaluate model with standardized dataset
# estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=1)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
# print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
# evaluate baseline model with standardized dataset
np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=20, batch_size=5, verbose=2)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
# print("Standardized: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
print("On cross validation it can be assessed that the model gives a good accuracy of: %.2f%% with a std of (%.2f%%)" % (results.mean()*100, results.std()*100))
# create model
model = Sequential()
model.add(Dense(60, input_dim=354, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
# Compile model
earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')
callbacks_list = [earlystop]
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit( X, Y, validation_split=0.1, epochs=20, batch_size=5, verbose=2, callbacks=callbacks_list)
loss, accuracy = model.evaluate(X, Y)
accuracy
loss
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# min_max_scaler = preprocessing.MinMaxScaler()
# X_minmax = min_max_scaler.fit_transform(X)
# # X_test_minmax = min_max_scaler.transform(X_test)
# # http://scikit-learn.org/stable/modules/preprocessing.html

# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
history = model.fit( X_scaled, Y, validation_split=0.25, epochs=20, batch_size=5, verbose=2, callbacks=callbacks_list)
loss, accuracy = model.evaluate(X_scaled, Y)
print (loss)
print (accuracy)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim(0.7,1)
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim(0,0.7)
plt.legend(['train', 'test'], loc='upper left')
plt.show()