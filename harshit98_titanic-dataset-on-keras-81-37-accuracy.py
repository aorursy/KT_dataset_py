import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import re

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
# Pre-process dataset
def pre_process(data):
    # Cabin
    data.Cabin.fillna('0', inplace=True)
    data.loc[data.Cabin.str[0] == 'A', 'Cabin'] = 1
    data.loc[data.Cabin.str[0] == 'B', 'Cabin'] = 2
    data.loc[data.Cabin.str[0] == 'C', 'Cabin'] = 3
    data.loc[data.Cabin.str[0] == 'D', 'Cabin'] = 4
    data.loc[data.Cabin.str[0] == 'E', 'Cabin'] = 5
    data.loc[data.Cabin.str[0] == 'F', 'Cabin'] = 6
    data.loc[data.Cabin.str[0] == 'G', 'Cabin'] = 7
    data.loc[data.Cabin.str[0] == 'T', 'Cabin'] = 8

    # Convert sex category into integers
    data['Sex'].replace('female', 1, inplace=True)
    data['Sex'].replace('male', 2, inplace=True)

    # Convert embarked to integer
    data['Embarked'].replace('S', 1, inplace=True)
    data['Embarked'].replace('C', 2, inplace=True)
    data['Embarked'].replace('Q', 3, inplace=True)

    # Median
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].median(), inplace=True)
    
    return data
def group_titles(data):
    data['Names'] = data['Name'].map(lambda x: len(re.split(' ', x)))
    data['Title'] = data['Name'].map(lambda x: re.search(', (.+?) ', x).group(1))
    data['Title'].replace('Master.', 0, inplace=True)
    data['Title'].replace('Mr.', 1, inplace=True)
    data['Title'].replace(['Ms.','Mlle.', 'Miss.'], 2, inplace=True)
    data['Title'].replace(['Mme.', 'Mrs.'], 3, inplace=True)
    data['Title'].replace(['Dona.', 'Lady.', 'the Countess.', 'Capt.', 'Col.', 'Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'the'], 4, inplace=True)
def data_subset(data):
    features = ['Pclass', 'SibSp', 'Parch', 'Sex', 'Names', 'Title', 'Age', 'Cabin']
    length_features = len(features)
    subset = data[features]
    return subset, length_features
def model(train_set_size, input_length, num_epochs, batch_size):
    model = Sequential()
    model.add(Dense(7, input_dim=input_length, activation='softplus'))
    model.add(Dense(3, activation='softplus'))
    model.add(Dense(1, activation='softplus'))
    
    learning_rate = 0.001
    adam = Adam(lr = learning_rate)
    
    # compile model and save weights
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    filepath = 'weights.best.h5f'
    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callback_list = [checkpoint]
    
    history_model = model.fit(X_train[:train_set_size], Y_train[:train_set_size], callbacks=callback_list, epochs=num_epochs, batch_size=batch_size, verbose=0)
    return model, history_model
def plots(history):
    loss_history = history.history['loss']
    acc_history = history.history['acc']
    epochs = [(i + 1) for i in range(num_epochs)]
    
    ax = plt.subplot(211)
    ax.plot(epochs, loss_history, color='red')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Error Rate')
    ax.set_title('Error Rate per Epoch')
    
    ax2 = plt.subplot(212)
    ax2.plot(epochs, acc_history, color='blue')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy per Epoch')
    
    plt.subplots_adjust(hspace=0.8)
    # plt.savefig('Accuracy_loss.png')
    plt.show()
def test(batch_size):
    test = pd.read_csv('../input/test.csv', header=0)
    test_ids = test['PassengerId']
    test = pre_process(test)
    group_titles(test)
    testdata, _ = data_subset(test)
    
    X_test = np.array(testdata).astype(float)
    
    output = model.predict(X_test, batch_size=batch_size, verbose=0)
    output = output.reshape((418,))
    
    column_1 = np.concatenate((['PassengerId'], test_ids), axis=0)
    column_2 = np.concatenate((['Survived'], output), axis=0)
    
    f = open('output.csv', 'wb')
    writer = csv.writer(f)
    for i in range(len(column_1)):
        writer.writerow([column_1[i]] + [column_2[i]])
    f.close()
if __name__ == '__main__':
    np.random.seed(7)
    
    # Read training dataset
    train = pd.read_csv('../input/train.csv', header=0)
    
    # Preprocess train.csv
    pre_process(train)
    group_titles(train)
    
    num_epochs = 100
    batch_size = 32
    
    train_data, length_features = data_subset(train)
    
    # Target value
    Y_train = np.array(train['Survived']).astype(int)
    X_train = np.array(train_data).astype(float)
    
    train_set_size = int(0.67 * len(X_train))

    model, history_model = model(train_set_size, length_features, num_epochs, batch_size)
    
    # plot
    plots(history_model)
    
    # Validation set
    X_validation = X_train[train_set_size:]
    Y_validation = Y_train[train_set_size:]
    
    # Evaluate model
    loss_and_metrics = model.evaluate(X_validation, Y_validation, batch_size=batch_size)
    print(loss_and_metrics)
    
    # Uncomment next line if you want to test against test.csv and output a csv file
    
    # test(batch_size)
