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
                dataframe.loc[row,'HasNAN'] = 1
        
        #print("Column %s has %d null." % (column, dataframe[column].isnull().sum()))
        if dataframe[column].isnull().sum() != 0:
            temp = testing_data[column]
            median = temp.append(training_data[column]).median()
            dataframe[column] = dataframe[column].fillna(median)
            
        
        #print("Column %s has %d null." % (column, dataframe[column].isnull().sum()))
        
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
    #dataframe['Age'] = (dataframe['Age'] / 15).astype('int32') * 15
    #dataframe['Fare'].loc[(dataframe['Fare'] < 1)] = 0
    #dataframe['Fare'].loc[(dataframe['Fare'] >= 1)] = 1
    #dataframe = dataframe.rename(columns={'Fare':'IsPassenger'})
    #dataframe['SibSp'].loc[(dataframe['SibSp'] > 3)] = 3
    #dataframe['Parch'].loc[(dataframe['Parch'] > 3)] = 3
    
    dataframe['Name'] = dataframe['Name'].apply(get_title)    
    dataframe['Name'] = pd.factorize(dataframe['Name'])[0]
    dataframe = dataframe.rename(columns={'Name':'Title'})
    
    dataframe = pd.get_dummies(dataframe, columns=['Title'])
    #dataframe = pd.get_dummies(dataframe, columns=['Title','Age','Pclass', 'SibSp', 'Parch'])
    
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

#print(testing_data)
print(len(training_data))
validation_split = int(len(training_data) / 2)

def shuffle_data():
    global train_x, train_y, validation_x, validation_y, test_x
    shuffled_training_data = shuffle(training_data)
    shuffled_training_data.reset_index(inplace=True, drop=True) 

    train_x = shuffled_training_data.drop(columns=['Survived']).to_numpy()
    train_y = shuffled_training_data['Survived'].to_numpy()


    validation_x = train_x[validation_split + 1:]
    validation_y = train_y[validation_split + 1:]
    train_x = train_x[0:validation_split]
    train_y = train_y[0:validation_split]

    test_x = testing_data.to_numpy()
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error
loss_log = pd.DataFrame(columns=['loss'])

best_model = None
best_score = 99

for i in range(0,50):
    shuffle_data()
    model = XGBClassifier(n_estimators=2000, learning_rate=0.001)
    # Add silent=True to avoid printing out updates with each cycle
    model.fit(train_x, train_y, early_stopping_rounds=150,
                 eval_set=[(train_x, train_y)], verbose=False)

    predictions = model.predict(validation_x)

    #predictions = np.where(predictions < 0.5,0,1)
    #print(predictions - validation_y)
    mae = mean_absolute_error(predictions, validation_y)
    #print("Mean Absolute Error : " + str(mae))
    
    if mae < best_score:
        best_score = mae
        
    loss_log.loc[len(loss_log)] = [mae]

#print(loss_log)
loss_log.plot(figsize=[20,4])
plt.show()
test_y = model.predict(test_x)
test_y = test_y.flatten()
final_prediction = pd.DataFrame(columns=['PassengerId','Survived'])
final_prediction['Survived'] = test_y
final_prediction['PassengerId'] = testing_data_raw['PassengerId']
print(final_prediction.head())

final_prediction.to_csv('submission.csv', index=False)