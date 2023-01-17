import numpy as np

import pandas as pd



from keras import models

from keras import layers

from keras import optimizers

from keras import losses 

from keras import metrics



import matplotlib.pyplot as plt
training_set = pd.read_csv('../input/train.csv')

testing_set = pd.read_csv('../input/test.csv')



x_train = training_set.drop(['PassengerId','Name','Ticket','Survived'], axis=1)

y_train = training_set['Survived']



x_test = testing_set.drop(['PassengerId','Name','Ticket'], axis=1)
x_train['Age'] = x_train['Age'].fillna(x_train['Age'].mean())

x_test['Age'] = x_test['Age'].fillna(x_test['Age'].mean())
def simplify_ages(df):

    #df['Age'] = df['Age'].fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df['Age'], bins, labels=group_names)

    df['Age'] = categories.cat.codes 

    return df



def simplify_cabins(df):

    df['Cabin'] = df['Cabin'].fillna('N')

    df['Cabin'] = df['Cabin'].apply(lambda x: x[0])

    df['Cabin'] =  pd.Categorical(df['Cabin'])

    df['Cabin'] = df['Cabin'].cat.codes 

    return df



def simplify_fares(df):

    df['Fare'] = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df['Fare'], bins, labels=group_names)

    df['Fare'] = categories.cat.codes 

    return df



def simplify_sex(df):

    df['Sex'] = pd.Categorical(df['Sex'])

    df['Sex'] = df['Sex'].cat.codes 

    return df



def simplify_embarked(df):

    df['Embarked'] = pd.Categorical(df['Embarked'])

    df['Embarked'] = df['Embarked'].cat.codes + 1

    return df



def transform_features(df):

    df = simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fares(df)

    df = simplify_sex(df)

    df = simplify_embarked(df)

    return df

transform_features(x_train)

transform_features(x_test)
model = models.Sequential()

model.add(layers.Dense(32, activation='relu', 

                       input_shape=(8,)))

model.add(layers.Dense(32, activation='relu'))

model.add(layers.Dense(16, activation='relu'))

#model.add(layers.Dense(8, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(optimizer=optimizers.RMSprop(lr=0.001),

             loss=losses.binary_crossentropy,

             metrics=[metrics.binary_accuracy])
y_train = np.asarray(y_train)

x_train = np.asarray(x_train)

x_test = np.asarray(x_test)



validation_size = 200



x_val = x_train[:validation_size]

partial_x_train = x_train[validation_size:]



y_val = y_train[:validation_size]

partial_y_train = y_train[validation_size:]
history = model.fit(partial_x_train, partial_y_train, epochs=30, validation_data=(x_val, y_val))
acc = history.history['binary_accuracy']

val_acc = history.history['val_binary_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(1, len(acc) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss,'b', label='Validation loss')



plt.legend()

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.show()
plt.clf()



epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training Acc')

plt.plot(epochs, val_acc,'b', label='Validation Acc')

plt.legend()

plt.xlabel('Epochs')

plt.ylabel('Acc')

plt.show()
predictions = model.predict_classes(x_test)

ids = testing_set['PassengerId'].copy()

new_output = ids.to_frame()

new_output["Survived"]=predictions

new_output.head(10)

new_output.to_csv("my_submission.csv",index=False)