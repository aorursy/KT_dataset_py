import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from keras import optimizers

from keras.models import Sequential

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.layers import Dense, Dropout, BatchNormalization, Activation



from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, roc_auc_score



# Set seeds to make the experiment more reproducible.

from tensorflow import set_random_seed

from numpy.random import seed

set_random_seed(0)

seed(0)



%matplotlib inline

sns.set(style="whitegrid")

warnings.filterwarnings("ignore")
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
print('Train set shape:', train.shape)

print('Test set shape:', test.shape)

print('Train set overview:')

display(train.head())
f, ax = plt.subplots(figsize=(6, 6))

ax = sns.countplot(x="target", data=train, label="Label count")

sns.despine(bottom=True)
def plot_distribution():

    f, axes = plt.subplots(1, 3, figsize=(20, 8), sharex=True)

    for feature in train.columns[1:31]:

        sns.distplot(train[feature], ax=axes[0], axlabel='First 30 features').set_title("Complete set")

        sns.distplot(train[train['target']==1][feature], ax=axes[1], axlabel='First 30 features').set_title("target = 1")

        sns.distplot(train[train['target']==0][feature], ax=axes[2], axlabel='First 30 features').set_title("target = 0")

    sns.despine(left=True)

    plt.tight_layout()



plot_distribution()
train = pd.concat([train, pd.get_dummies(train['wheezy-copper-turtle-magic'], prefix='magic', drop_first=True)], axis=1).drop(['wheezy-copper-turtle-magic'], axis=1)

test = pd.concat([test, pd.get_dummies(test['wheezy-copper-turtle-magic'], prefix='magic', drop_first=True)], axis=1).drop(['wheezy-copper-turtle-magic'], axis=1)
labels = train['target']

train.drop('target', axis=1, inplace=True)

train.drop('id', axis=1, inplace=True)

test.drop('id', axis=1, inplace=True)



X_train, X_val, Y_train, Y_val = train_test_split(train, labels, test_size=0.2, random_state=1)
non_cat_features = list(train.filter(regex='^(?!magic_)'))

scaler = MinMaxScaler()

X_train[non_cat_features] = scaler.fit_transform(X_train[non_cat_features])

X_val[non_cat_features] = scaler.transform(X_val[non_cat_features])

test[non_cat_features] = scaler.transform(test[non_cat_features])
BATCH_SIZE = 128

EPOCHS = 30

LEARNING_RATE = 0.01

ES_PATIENCE = 5
model = Sequential()

model.add(Dense(1024, input_dim=X_train.shape[1]))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(256))

model.add(BatchNormalization())

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(1, activation="sigmoid"))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=ES_PATIENCE)

callback_list = [es]



optimizer = optimizers.Adam(lr=0.0001)

model.compile(optimizer=optimizer, loss="binary_crossentropy",  metrics=['binary_accuracy'])

model.summary()
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), 

                    callbacks=callback_list, 

                    epochs=EPOCHS, 

                    batch_size=BATCH_SIZE, 

                    verbose=2)
sns.set_style("whitegrid")

fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', figsize=(20, 7))



ax1.plot(history.history['loss'], label='Train loss')

ax1.plot(history.history['val_loss'], label='Validation loss')

ax1.legend(loc='best')

ax1.set_title('Loss')



ax2.plot(history.history['binary_accuracy'], label='Train Accuracy')

ax2.plot(history.history['val_binary_accuracy'], label='Validation accuracy')

ax2.legend(loc='best')

ax2.set_title('Accuracy')



plt.xlabel('Epochs')

sns.despine()

plt.show()
train_pred = model.predict_classes(X_train)

val_pred = model.predict_classes(X_val)



f, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)

train_cnf_matrix = confusion_matrix(Y_train, train_pred)

val_cnf_matrix = confusion_matrix(Y_val, val_pred)



train_cnf_matrix_norm = train_cnf_matrix / train_cnf_matrix.sum(axis=1)[:, np.newaxis]

val_cnf_matrix_norm = val_cnf_matrix / val_cnf_matrix.sum(axis=1)[:, np.newaxis]



train_df_cm = pd.DataFrame(train_cnf_matrix_norm, index=[0, 1], columns=[0, 1])

val_df_cm = pd.DataFrame(val_cnf_matrix_norm, index=[0, 1], columns=[0, 1])



sns.heatmap(train_df_cm, annot=True, fmt='.2f', cmap="Blues", ax=axes[0]).set_title("Train")

sns.heatmap(val_df_cm, annot=True, fmt='.2f', cmap="Blues", ax=axes[1]).set_title("Validation")

plt.show()
print('Train AUC %.2f' % roc_auc_score(Y_train.values, train_pred))

print('Validation AUC %.2f' % roc_auc_score(Y_val.values, val_pred))
predictions = model.predict(test)

df = pd.read_csv('../input/sample_submission.csv')

df['target'] = predictions

df.to_csv('submission.csv', index=False)

df.head(10)