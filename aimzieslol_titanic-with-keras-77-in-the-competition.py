import pandas as p

import numpy as np

import matplotlib.pyplot as plot

import keras as k

import seaborn as sns; sns.set()



from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold

from sklearn.decomposition import PCA



import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
orig_train = p.read_csv('train.csv')

orig_test = p.read_csv('test.csv')



orig_train['type'] = 1

orig_test['type'] = 0
small_fig_size=(5, 5)



def plot_loss(history):

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    

    plot.figure(figsize=small_fig_size)

    plot.plot(epochs, loss, color='red', label='Training loss')

    plot.plot(epochs, val_loss, color='green', label='Validation loss')

    plot.title('Training and validation loss')

    plot.xlabel('Epochs')

    plot.ylabel('Loss')

    plot.legend()

    plot.show()

    

def plot_acc(history):

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    epochs = range(1, len(loss) + 1)    

    

    plot.figure(figsize=small_fig_size)

    plot.plot(epochs, acc, color='red', label='Training acc')

    plot.plot(epochs, val_acc, color='green', label='Validation acc')

    plot.title('Training and validation accuracy')

    plot.xlabel('Epochs')

    plot.ylabel('Loss')

    plot.legend()

    plot.show()
combined_df = p.concat([orig_train, orig_test])



# Fill empty stuff.

combined_df['Embarked'].fillna('C', inplace=True)

combined_df['Cabin'].fillna('U', inplace=True)

combined_df['Fare'] = combined_df['Fare'].interpolate()



# Fill age w/median based on class, sex, and family size or some of the above lmao

combined_df['Age'] = combined_df.groupby(['Pclass','Sex','Parch','SibSp'])['Age'].transform(lambda x: x.fillna(x.mean()))

combined_df['Age'] = combined_df.groupby(['Pclass','Sex','Parch'])['Age'].transform(lambda x: x.fillna(x.mean()))

combined_df['Age'] = combined_df.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))



# Split out the title, save it, and change it to a value we know

combined_df['Title'] = p.Series((name.split('.')[0].split(',')[1].strip() for name in combined_df['Name']), index=combined_df.index)



combined_df['Title'] = combined_df['Title'].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

combined_df['Title'] = combined_df['Title'].replace(['Mlle', 'Ms'], 'Miss')

combined_df['Title'] = combined_df['Title'].replace('Mme', 'Mrs')



# Split out the cabin information.

combined_df['NumCabins'] = p.Series((len(cabin.split(' ')) for cabin in combined_df['Cabin']), index=combined_df.index)

combined_df['CabinPrefix'] = p.Series((cabin.split(' ')[0][0] for cabin in combined_df['Cabin']), index=combined_df.index)

combined_df['FirstCabinNum'] = p.Series((cabin.split(' ')[0][1:] for cabin in combined_df['Cabin']), index=combined_df.index)

combined_df['FirstCabinNum'] = combined_df['FirstCabinNum'].transform(lambda x: 0 if not x else int(x))



# Fill ticket and just use the #

combined_df['Ticket'].fillna('U', inplace=True)    

combined_df['TicketID'] = p.Series([blah.split(' ')[::-1][0] for blah in combined_df['Ticket']], index=combined_df.index)



# One of these came out to be called "LINE"

combined_df['TicketID'] = combined_df['TicketID'].transform(lambda x: 0 if x == 'LINE' else int(x))



combined_df['FamilySize'] = combined_df['SibSp'] + combined_df['Parch'] + 1



combined_df['FarePerPerson'] = combined_df['Fare'] / combined_df['FamilySize']

combined_df['FarePerCabin'] = combined_df['Fare'] / combined_df['NumCabins']
combined_df = p.concat([combined_df, p.get_dummies(combined_df['Sex'], 'gender')], axis=1)

combined_df = p.concat([combined_df, p.get_dummies(combined_df['Pclass'], 'pclass')], axis=1)

combined_df = p.concat([combined_df, p.get_dummies(combined_df['Title'], 'title')], axis=1)

combined_df = p.concat([combined_df, p.get_dummies(combined_df['CabinPrefix'], 'cabin_prefix')], axis=1)

combined_df = p.concat([combined_df, p.get_dummies(combined_df['Embarked'], 'embarked')], axis=1)
dump_cols = ['Cabin', 'Embarked', 'Name', 'PassengerId', 'Pclass', 'Sex', 'Ticket', 'Title', 'CabinPrefix']

combined_df.drop(dump_cols, inplace=True, axis=1)
# cols_to_transform = ['Age', 'Fare', 'FirstCabinNum', 'TicketID', 'FarePerPerson', 'FarePerCabin', 'Parch', 'SibSp', 'NumCabins', 'FamilySize' ]

# combined_scaled = p.DataFrame(StandardScaler().fit_transform(combined_df[cols_to_transform]), \

#                               columns=cols_to_transform, index=combined_df.index)



combined_df = p.DataFrame(MinMaxScaler().fit_transform(combined_df), columns=combined_df.columns, index=combined_df.index)
# combined_df = p.concat([combined_df.drop(cols_to_transform, axis=1), combined_scaled], axis=1)
scaled_train = (combined_df[combined_df['type'] == 1]).drop('type', axis=1)

scaled_test = (combined_df[combined_df['type'] == 0]).drop(['type', 'Survived'], axis=1)
scaled_train.to_csv('pca-me.csv', index=False)
def get_model(shape):

    model = k.models.Sequential()

    

    model.add(k.layers.Dense(100, input_dim=shape))

    model.add(k.layers.Activation('relu'))

    model.add(k.layers.Dropout(.25))    

    model.add(k.layers.Dense(50))

    model.add(k.layers.Activation('relu'))

    model.add(k.layers.BatchNormalization())

    model.add(k.layers.Dropout(.25))

    model.add(k.layers.Dense(20))

    model.add(k.layers.Activation('relu'))

    model.add(k.layers.Dropout(.25))    

    model.add(k.layers.Dense(1))

    model.add(k.layers.Activation('sigmoid'))

    

    model.compile(k.optimizers.Adam(lr=0.03), 'binary_crossentropy', ['acc'])



    return model



def step_decay(epoch):

    initial_lrate = 0.1

    drop = 0.6

    epochs_drop = 3.0

    lrate = initial_lrate * np.power(drop, np.floor((1 + epoch) / epochs_drop))

    

    return lrate
X = scaled_train.drop('Survived', axis=1).copy()

Y = scaled_train['Survived'].values



X_train, X_test, y_train, y_test = train_test_split(X, Y)



model = get_model(X_train.shape[1])



early_stopper = k.callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True)

reduce_lr = k.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.2)

lr_sched = k.callbacks.LearningRateScheduler(step_decay)



all_callbacks = [ ]

%%time



shuffler = StratifiedKFold()



for train_idx, test_idx in shuffler.split(X_train, y_train):

    __X_train, __y_train = X_train.values[train_idx], y_train[train_idx]

    __X_test, __y_test = X_train.values[test_idx], y_train[test_idx]

    

    model.fit(__X_train, __y_train, epochs=20, validation_data=(__X_test, __y_test), verbose=0, callbacks=all_callbacks)

    
plot_acc(model.history)
plot_loss(model.history)
model.evaluate(X_test, y_test)
prediction = model.predict(scaled_test)



pred_df = p.DataFrame(prediction, columns=['SurvivalProb'], index=orig_test.index)



pred_df['Survived'] = [1 if x > .90 else 0 for x in pred_df['SurvivalProb']]



combined_test = p.concat([orig_test, pred_df], axis=1)





combined_test.filter(['PassengerId', 'Survived']).to_csv('submission.csv', index=False)