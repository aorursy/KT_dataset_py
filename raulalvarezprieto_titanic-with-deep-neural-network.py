# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import callbacks
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer


path_Titanic = "/kaggle/input/titanic/"
path_training = path_Titanic + "train.csv"
path_test = path_Titanic + "test.csv"
training = pd.read_csv(path_training)
x_test = pd.read_csv(path_test)
x_train = training
print("Number of traning examples (data points) = %i " % training.shape[0])
unique, count= np.unique(training["Survived"], return_counts=True)
print("Count Non Survivors vs Survivors in the dataset = %s " % dict (zip(unique, count) ), "\n" )
print("Number of features = %i " % training.shape[1])
training.isna().sum()
total_data_Set = pd.concat([training, x_test])
np.random.seed(0)
# Management of data
training.drop([ "PassengerId", "Ticket"], inplace = True, axis = 1 )
id = x_test['PassengerId']
x_test.drop([ "PassengerId", "Ticket"], inplace = True, axis = 1 )
mediana = training['Age'].quantile(q=0.50)
mediana_fare = training['Fare'].quantile(q=0.50)
common_value = "S"
cabin_value = "M"
common_family_name = "Smith"
repCol3 = {  "male":0, "female" : 1}
repCol8 = {"C" : 0 ,   "Q" : 1 , 'S' : 2  }
dictionary = {'A':1,'B':2,'C':3,'D':4,'E':5,'F':6,'G':7,'H':8,'I':9,'J':10,'K':11,'L':12,'M':13,'N':14,
              'O':15,'P':16,'Q':17,'R':18,'S':19,'T':20,'U':21,'V':22,'W':23,'X':24,'Y':25,'Z':26}
i=0
family_names_dict = {}
for family_name in total_data_Set["Name"]:
    i+=1
    first = family_name.split()    
    family_names_dict.update({family_name.split()[0]: i})
    training.loc[training['Name'] == family_name, "Name"] = first[0]
    x_test.loc[x_test['Name'] == family_name, "Name"] = first[0]
family_names_dict.update({common_family_name: 0})
#Management of Train Dataset
x_train["Age"].fillna(mediana,inplace=True)
x_train["Embarked"].fillna(common_value,inplace=True)
x_train["Fare"].fillna(mediana_fare,inplace=True)
x_train["Cabin"].fillna(cabin_value,inplace=True)
x_train["Name"].fillna(common_family_name,inplace=True)
x_train['Family'] = x_train ['SibSp'] + x_train['Parch'] 
x_train['IsAlone'] = 1
x_train['IsAlone'].loc[x_train['Family'] > 0] = 0
##Replacing values 
y_train = x_train["Survived"]
x_train = x_train.drop(['Survived'], axis = 1)
x_train['Cabin'] = (x_train['Cabin'].str[0])
x_train.replace({"Sex": repCol3, "Embarked": repCol8, "Cabin": dictionary, "Name": family_names_dict}, inplace = True )
x_train.isna().sum()
print("Number of features after Engineering = %i " % x_train.shape[1])
# Management of test dataset
x_test["Age"].fillna(mediana,inplace=True)
x_test["Embarked"].fillna(common_value,inplace=True)
x_test["Fare"].fillna(mediana_fare,inplace=True)
x_test['Cabin'].fillna(cabin_value,inplace=True)
x_test["Name"].fillna(common_family_name,inplace=True)
x_test['Family'] = x_test ['SibSp'] + x_test['Parch']
x_test['IsAlone'] = 1
x_test['IsAlone'].loc[x_test['Family'] > 0] = 0
##Replacing values 
x_test['Cabin'] = (x_test['Cabin'].str[0])
x_test.replace({"Sex": repCol3, "Embarked": repCol8, "Cabin": dictionary, "Name": family_names_dict}, inplace = True )
x_test.isna().sum()
# fit scaler		
my_scaler = MinMaxScaler()
# transform training dataset
x_train = my_scaler.fit_transform(x_train)
# transform test dataset
x_test = my_scaler.fit_transform(x_test)

InputDimension = x_train.shape[1]
dropout_rate = 0.2
model = Sequential()
model.add(Dense(512, input_dim=InputDimension, activation='relu'))
# First layer - Hidden Layer
model.add(Dense(512, activation='relu'))
model.add(Dropout(dropout_rate))
# Second layer - Hidden Layer
model.add(Dense(512, activation='relu'))
model.add(Dropout(dropout_rate))
# Output layer
model.add(Dense(2, activation='softmax'))
model.summary()

batch_size = 256
learning_rate = 0.0001
epochs = 1000
patience = 50
validation_split=0.1
earlystopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=0, mode='min')
optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
history = model.fit(x_train, pd.get_dummies(y_train), epochs=epochs, batch_size=batch_size, validation_split=validation_split, verbose=0, callbacks=[earlystopping])

# Print results CNN.
def print_model(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()

print_model(history)
history2 = history.history
print('Validation accuracy: {acc}, loss: {loss}'.format(acc=history2['val_acc'][-1], loss=history2['val_loss'][-1]))

predictions = model.predict(x_test)
predictions = np.rint(predictions)
predict = pd.DataFrame(predictions, columns=['0', '1']).astype('int')
predict['Survived'] = 0
predict.loc[predict['0'] == 1, 'Survived'] = 0
predict.loc[predict['1'] == 1, 'Survived'] = 1
predict.reset_index(drop=True, inplace=True)
output = pd.concat([id,predict['Survived'] ], axis=1)
output.to_csv('titanic-predictions.csv', index = False)
output.head(100)

