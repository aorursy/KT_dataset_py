# Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
# Read the data 
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
# OHE sex
train_data = train_data.join(pd.get_dummies(train_data['Sex']))
test_data = test_data.join(pd.get_dummies(test_data['Sex']))
# Most people are from S
train_data.loc[train_data['Embarked'].isnull(), 'Embarked'] = 'S'
# Encode Embarked data
embarked_encoder = LabelEncoder()
train_data['Embarked'] = embarked_encoder.fit_transform(train_data['Embarked'])
test_data['Embarked'] = embarked_encoder.transform(test_data['Embarked'])
# Fill missing values of age with the average age
train_data_male_age = train_data.loc[(train_data['Age'].notnull() & train_data['male'] == 1), 'Age']
test_data_male_age = test_data.loc[(test_data['Age'].notnull() & test_data['male'] == 1), 'Age']
male_avg_age = pd.concat([train_data_male_age, test_data_male_age]).mean()

train_data_female_age = train_data.loc[(train_data['Age'].notnull() & train_data['female'] == 1), 'Age']
test_data_female_age = test_data.loc[(test_data['Age'].notnull() & test_data['female'] == 1), 'Age']
female_avg_age = pd.concat([train_data_female_age, test_data_female_age]).mean()

train_data.loc[(train_data['Age'].isnull()) & (train_data['male'] == 1), 'Age'] = male_avg_age
test_data.loc[(test_data['Age'].isnull()) & (test_data['male'] == 1), 'Age'] = male_avg_age

train_data.loc[(train_data['Age'].isnull()) & (train_data['female'] == 1), 'Age'] = female_avg_age
test_data.loc[(test_data['Age'].isnull()) & (test_data['female'] == 1), 'Age'] = female_avg_age
test_data.loc[(test_data['Fare'].isnull())] = 14.5000
train_data.loc[(train_data['SibSp'] == 0) & (train_data['Parch'] == 0), 'Is_alone'] = 1
test_data.loc[(test_data['SibSp'] == 0) & (test_data['Parch'] == 0), 'Is_alone'] = 1

train_data.loc[(train_data['SibSp'] > 0) | (train_data['Parch'] > 0), 'Is_alone'] = 0
test_data.loc[(test_data['SibSp'] > 0) | (test_data['Parch'] > 0), 'Is_alone'] = 0
train_data.head()
# Remove unnecessary columns
test_data_passenger_id = test_data['PassengerId']

columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex', 'SibSp', 'Parch']
train_data.drop(columns=columns_to_drop, inplace=True)
train_data_columns = train_data.columns

test_data.drop(columns=columns_to_drop, inplace=True)
test_data_columns = test_data.columns
train_data.head()
age_scaler = MinMaxScaler()
train_data['Age'] = age_scaler.fit_transform(train_data['Age'].values.reshape(-1, 1))
test_data['Age'] = age_scaler.transform(test_data['Age'].values.reshape(-1, 1))

fare_scaler = MinMaxScaler()
train_data['Fare'] = fare_scaler.fit_transform(train_data['Fare'].values.reshape(-1, 1))
test_data['Fare'] = fare_scaler.transform(test_data['Fare'].values.reshape(-1, 1))

train_data = pd.DataFrame(train_data)
train_data.columns = train_data_columns

test_data = pd.DataFrame(test_data)
test_data.columns = test_data_columns
train_data_label = train_data['Survived']
train_data.drop(columns=['Survived'], inplace=True)
train_data.head()
train_data = train_data.values
train_data_label = train_data_label.values

test_data = test_data.values
from tensorflow import set_random_seed
np.random.seed(42)
set_random_seed(42)

model = Sequential()
model.add(Dense(45, input_dim=7, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, train_data_label, batch_size=100, epochs=150, verbose=1)

loss, accuracy = model.evaluate(train_data, train_data_label)
print(accuracy)
# Predict the data
predicted_data = pd.DataFrame(model.predict_classes(test_data))
final_output = pd.DataFrame()
final_output['PassengerId'] = test_data_passenger_id
final_output['Survived'] = predicted_data
final_output.to_csv('MLP.csv', index=False)