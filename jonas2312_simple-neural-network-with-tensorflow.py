import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os
import math
from IPython.display import clear_output
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
training_data_raw = pd.read_csv("/kaggle/input/titanic/train.csv")
testing_data_raw = pd.read_csv("/kaggle/input/titanic/test.csv")

print(training_data_raw.head())
print(testing_data_raw.head())
training_data = training_data_raw[['Sex','Name','Age','Pclass','SibSp','Parch','Fare','Survived']]
testing_data = testing_data_raw[['Sex','Name','Age','Pclass','SibSp','Parch','Fare']]
print((training_data[['Name','Sex','Age','Pclass','SibSp','Parch','Fare','Survived']].isnull()).sum())
print((testing_data[['Name','Sex','Age','Pclass','SibSp','Parch','Fare']].isnull()).sum())
def handle_nans(dataframe):

    dataframe['HasNAN'] = 0

    for column in dataframe:
        if  dataframe[column].dtypes != 'int32' and dataframe[column].dtypes != 'float32' and \
            dataframe[column].dtypes != 'int64' and dataframe[column].dtypes != 'float64':
                continue
        

        for row, value in dataframe[column].iteritems():          
            if math.isnan(value):
                dataframe['HasNAN'][row] = 1
        dataframe[column] = dataframe[column].fillna(dataframe[column].median())
        
    return dataframe
    
training_data = handle_nans(training_data)
testing_data = handle_nans(testing_data)


#print(training_data.head())
#print(testing_data.head())
def get_title(name):
    first_part_name = name.split(",")[1]
    title = first_part_name.split(" ")[1]
    return title

def age_class_dummies(dataframe):
    dataframe['Sex'] = pd.factorize(dataframe['Sex'])[0]
    dataframe['Age'] = (dataframe['Age'] / 15).astype('int32') * 15
    dataframe['Fare'].loc[(dataframe['Fare'] < 1)] = 0
    dataframe['Fare'].loc[(dataframe['Fare'] >= 1)] = 1
    dataframe = dataframe.rename(columns={'Fare':'IsPassenger'})
    dataframe['SibSp'].loc[(dataframe['SibSp'] > 3)] = 3
    dataframe['Parch'].loc[(dataframe['Parch'] > 3)] = 3
    
    dataframe['Name'] = dataframe['Name'].apply(get_title)
    dataframe = dataframe.rename(columns={'Name':'Title'})
        
    dataframe = pd.get_dummies(dataframe, columns=['Title','Age','Pclass', 'SibSp', 'Parch'])
    
    dataframe = dataframe.astype('int32')
    return dataframe

training_data = age_class_dummies(training_data)
testing_data = age_class_dummies(testing_data)

for column in training_data:
    if (column not in testing_data) and (column != 'Survived'):        
        print("Column not in testing data: ", column)
        testing_data[column] = 0
        
for column in testing_data:
    if column not in training_data:
        print("Column not in training data: ", column)
        training_data[column] = 0

training_data = training_data.reindex(sorted(training_data.columns), axis=1)
testing_data = testing_data.reindex(sorted(testing_data.columns), axis=1)

print(testing_data)
shuffled_training_data = shuffle(training_data)
shuffled_training_data.reset_index(inplace=True, drop=True) 

train_x = shuffled_training_data.drop(columns=['Survived']).to_numpy()
train_y = shuffled_training_data['Survived'].to_numpy()

print(len(train_x))
validation_split = 400

validation_x = train_x[validation_split + 1:]
validation_y = train_y[validation_split + 1:]
train_x = train_x[0:validation_split]
train_y = train_y[0:validation_split]

test_x = testing_data.to_numpy()


input_size = len(shuffled_training_data.columns) - 1

inputs = tf.keras.layers.Input(shape=(input_size))
dense_1 = tf.keras.layers.Dense(12)(inputs)
dense_2 = tf.keras.layers.Dense(4)(dense_1)
dense_3 = tf.keras.layers.Dense(2)(dense_2)
outputs = tf.keras.layers.Dense(1)(dense_3)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.build(input_shape=(input_size))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
model.compile(optimizer=optimizer, loss='mse')
loss_log = pd.DataFrame(columns=['loss_train','Validation percentage'])

def Cross_Validation():
    validation_pred = model.predict(validation_x)
    validation_pred  = np.where(validation_pred  < 0.5,0,1)
    validation_pred  = validation_pred.flatten()
    pred_error = validation_pred - validation_y        
    pred_error = np.abs(pred_error)
    num_zeros = (pred_error == 1).sum()
    percentage_correct = num_zeros / len(pred_error)    
    return percentage_correct

class MyCustomCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None): 
    global loss_log
    if epoch % 1 is 0:
        loss_log.loc[epoch] = [logs['loss'], Cross_Validation()]
    if epoch % 50 is 49:        
        clear_output(wait=True)
        print("Current epoch %d, current loss on \n Training data is %f \n Percentage of validation data wrong is %f" % (epoch, logs['loss'], Cross_Validation()))  
        if len(loss_log) > 0:
            loss_log.plot(figsize=[20,4])
            plt.show()  
        while len(loss_log) > (100):
            loss_log = loss_log.drop(loss_log.index[:1])
model.fit(train_x, train_y, epochs=500, batch_size=1, verbose=0, callbacks=[MyCustomCallback()])
test_y = model.predict(test_x)
test_y = np.where(test_y < 0.5,0,1)
test_y = test_y.flatten()
final_prediction = pd.DataFrame(columns=['PassengerId','Survived'])
final_prediction['Survived'] = test_y
final_prediction['PassengerId'] = testing_data_raw['PassengerId']
print(final_prediction.head())

final_prediction.to_csv('submission.csv', index=False)