import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv', index_col=0)

test = pd.read_csv('../input/titanic/test.csv', index_col=0)

train
# number of nans in signs

nans = train.isnull().sum()

nans.plot(kind='bar', grid=True, color='darkgreen', rot=30, title='Missed')



nans[nans != 0]
freq = train['Embarked'].value_counts().index[0]

train['Embarked'].fillna(freq, inplace=True)
train.drop(['Ticket', 'Cabin'], axis=1, inplace=True)

train
train = pd.get_dummies(train, columns=['Embarked', 'Sex'], prefix=['Embarked', 'Sex'], prefix_sep=' - ')

train
list_handling = train['Name'].apply(lambda x: x.split(',')[1].split('.')[0]).value_counts()

list_handling
for elem in list_handling.index:

    cond1 = train['Name'].str.contains(elem + '.', regex=False)

    cond2 = train['Age'].isnull()

    train.loc[cond1 & cond2, 'Age'] = train.loc[cond1, 'Age'].mean()
nans = train.isnull().sum()

nans[nans != 0]
val = set(train['Pclass'])

for i in val:

    train.loc[train['Pclass'] == i, 'Age'].plot(kind='kde', grid=True, label='%s Class' % str(i))

plt.legend()
test.isnull().sum()
test = pd.get_dummies(test, columns=['Embarked', 'Sex'], prefix=['Embarked', 'Sex'], prefix_sep=' - ')

test.drop(['Cabin', 'Ticket'], axis=1, inplace=True)

test
for elem in list_handling.index:

    cond1 = test['Name'].str.contains(elem + '.', regex=False)

    cond2 = test['Age'].isnull()

    test.loc[cond1 & cond2, 'Age'] = train.loc[train['Name'].str.contains(elem + '.', regex=False), 'Age'].mean()

    

    cond3 = test['Fare'].isnull()

    test.loc[cond3, 'Fare'] = train.loc[train['Name'].str.contains(elem + '.', regex=False), 'Fare'].mean()
train.drop(['Name'], axis=1, inplace=True)

test.drop(['Name'], axis=1, inplace=True)
train
test
Y_train = train['Survived'].values

X_train = train.drop(['Survived'], axis=1).values

X_test = test.values
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout
mean = X_train.mean(axis=0)

std = X_train.std(axis=0)



X_train -= mean

X_train /= std



X_test -= mean

X_test /= std
X_test
from keras import models, optimizers





model = Sequential()



model.add(Dense(40, kernel_initializer='uniform', activation='relu', input_shape=(X_train.shape[1],) ))

model.add(Dropout(0.5))

model.add(Dense(100, kernel_initializer='uniform', activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(40, kernel_initializer='uniform', activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(optimizer=optimizers.Adam(lr=1e-5), loss='binary_crossentropy', metrics=['acc'])



model.summary()
from keras.callbacks import ModelCheckpoint





checkpoint_path = 'bestmodel4.hdf5'



checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=0, save_best_only=True, mode='max')



callbacks_list = [checkpoint]

history = model.fit(X_train, Y_train, batch_size=30, epochs=3000, 

                    callbacks=callbacks_list, verbose=0, validation_split=0.2)
def graph_plot(history):

    

    for i in history.history.keys():

        print(f'{i}\nmin = {min(history.history[i])}, max = {max(history.history[i])}')

    

    epoch = len(history.history['loss'])

    for k in list(history.history.keys()):

        if 'val' not in k:

            plt.figure(figsize=(10, 7))

            plt.plot(history.history[k])

            plt.plot(history.history['val_' + k])

            plt.title(k, fontsize=10)



            plt.ylabel(k)

            plt.xlabel('epoch')

            plt.grid()



            plt.yticks(fontsize=10, rotation=30)

            plt.xticks(fontsize=10, rotation=30)

            plt.legend(['train', 'test'], loc='upper left', fontsize=10, title_fontsize=15)

            plt.show()
graph_plot(history)
model.load_weights(checkpoint_path)

# model.load_weights('bestmodel1.hdf5')





pred = model.predict(X_test)



Y_pred = (pred > 0.5).astype(int)
Y_pred = Y_pred[:, 0]

Y_pred
submission = pd.DataFrame({"Survived": Y_pred}, index=test.index)

submission.to_csv("submission.csv")
pd.concat([submission, pd.read_csv('../input/titanic/gender_submission.csv', index_col=0)], axis=1)
from sklearn.metrics import confusion_matrix, accuracy_score

print(confusion_matrix(pd.read_csv('../input/titanic/gender_submission.csv', index_col=0).values, Y_pred))

accuracy_score(pd.read_csv('../input/titanic/gender_submission.csv', index_col=0).values, Y_pred)
from sklearn.metrics import precision_recall_fscore_support

precision_recall_fscore_support(pd.read_csv('../input/titanic/gender_submission.csv', index_col=0).values, 

                Y_pred, average='binary')
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV



param_grid = [    

    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],

    'C' : np.logspace(-4, 4, 500),

    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],

    'max_iter' : np.arange(100, 150, 10)

    }

]



scoring = {'Accuracy': 'accuracy'}



gs = GridSearchCV(LogisticRegression(), return_train_score=True,

                  param_grid=param_grid, scoring=scoring, cv=10, refit='Accuracy', n_jobs=-1)
gs.fit(X_train, Y_train)



print("best params: " + str(gs.best_estimator_))

print("best params: " + str(gs.best_params_))

print('best score:', gs.best_score_)
Y_pred2 = gs.predict(X_test)
submission2 = pd.DataFrame({"Survived": Y_pred2}, index=test.index)

submission2.to_csv("submission2.csv")
pd.concat([submission2, pd.read_csv('../input/titanic/gender_submission.csv', index_col=0)], axis=1)
print(confusion_matrix(pd.read_csv('../input/titanic/gender_submission.csv', index_col=0).values, Y_pred2))

accuracy_score(pd.read_csv('../input/titanic/gender_submission.csv', index_col=0).values, Y_pred2)
precision_recall_fscore_support(pd.read_csv('../input/titanic/gender_submission.csv', index_col=0).values, 

                Y_pred2, average='binary')