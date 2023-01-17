#Loading necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
#Loading the training data set
titanic = pd.read_csv('../input/titanic/train.csv')
titanic.head()
#Checking for Null values
titanic.info()
#Checkpoint 1
titanic_copy = titanic.copy()
titanic_copy.drop(columns=['Cabin','Name'], axis=1, inplace=True)
titanic_copy.head()
#Checking Dataframe info
titanic_copy.info()
titanic_copy = titanic_copy[pd.notnull(titanic_copy['Embarked'])]
titanic_copy.info()
#Calculating Mean age
mean_age = titanic_copy['Age'].mean()
mean_age
#Filling missing values with Mean Age
titanic_copy = titanic_copy.fillna(value={'Age':mean_age})
titanic_copy.info()
titanic_copy.head(5)
#Final Checkpoint
titanic_final = titanic_copy.copy()
#We drop Ticket column as well due to redundancy
titanic_final.drop(columns=['Ticket'], axis=1, inplace=True)
titanic_preprocessed_data = pd.get_dummies(titanic_final, columns=['Embarked','Pclass','Sex'], drop_first=True)
titanic_preprocessed_data.reset_index(inplace=True, drop=True)
titanic_preprocessed_data.head()
#Checking whether the dataset is balanced or not 
titanic_preprocessed_data[titanic_preprocessed_data['Survived'] == 1]['Survived'].count()*100/889
titanic_preprocessed_data[titanic_preprocessed_data['Survived'] == 0].describe()
sns.distplot(titanic_preprocessed_data[titanic_preprocessed_data['Survived'] == 0]['Fare'])
#Setting quantile limit to 75%
quantile_limit = 0.75
#Calculating the Quantile fare for deceased cases
quantile_fare = titanic_preprocessed_data[titanic_preprocessed_data['Survived'] == 0]['Fare'].quantile(quantile_limit)
quantile_fare
#Discarding values beyond Quantile fare
titanic_bal = titanic_preprocessed_data[(titanic_preprocessed_data['Survived'] == 1) | (titanic_preprocessed_data['Fare'] <=quantile_fare)]
#Checking for the current survivor percentage
titanic_bal[titanic_bal['Survived'] == 1]['Survived'].count()*100/len(titanic_bal)
#Import relevant libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
#Model creation function
def create_model(input_size=4):
    model = Sequential([
       Flatten(input_shape=(input_size,)),
       Dense(input_size,activation="relu"),    
       Dense(4,activation="relu"),
       Dense(1,activation="sigmoid")                      
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
#Instantiating our model
survival_model = create_model(9)
#List of columns
titanic_bal.columns.values
#Input Set
X_train = titanic_bal.drop(columns=['PassengerId','Survived'],axis=1)
X_train.head()
#Output Set
y_train = titanic_bal['Survived']
y_train.head()
#Fitting the model
survival_model.fit(X_train, y_train, epochs=500, batch_size=100, verbose=1)
#Loading testing data
titanic_test_raw = pd.read_csv('../input/titanic/test.csv')
titanic_test_raw.head()
#Removed redundant columns
titanic_test_2 = titanic_test_raw.drop(columns=['Cabin','Ticket','Name'])
titanic_test_2.info()
#Calculating mean Fare value
mean_fare = titanic_test_2['Fare'].mean()
mean_fare
#Calculating Mean Age
mean_age = titanic_test_2['Age'].mean()
mean_age
titanic_test_2.fillna(value={'Age': mean_age, 'Fare':mean_fare}, inplace=True)
titanic_test_2.info()
titanic_test = pd.get_dummies(titanic_test_2, columns=['Pclass','Sex','Embarked'], drop_first=True)
titanic_test.head()
#Input Set
X_test = titanic_test.drop(columns=['PassengerId'],axis=1)
X_test.head()
#Outputs Predicted
y_pred = survival_model.predict_classes(X_test)
y_pred[:5]
#Creating our Survival Data
passenger_ids = np.array(titanic_test['PassengerId'])
survived_stats = np.array(y_pred).reshape(len(y_pred),)
survival_data = {'PassengerId':passenger_ids,'Survived':survived_stats}
#(passenger_ids.shape,survived_stats.shape)
survival_df = pd.DataFrame(data=survival_data, columns=['PassengerId','Survived'] )
survival_df.head()
#Exporting to CSV file
survival_df.to_csv('test_submission.csv')