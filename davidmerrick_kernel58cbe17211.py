import numpy as np 
import pandas as pd 

train = pd.read_csv('../input/train.csv')
train.head()
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, Imputer

def preprocess_data(data):
    drop_columns = ['Cabin', 'Name', 'Ticket', 'Embarked']
    data = data.drop(drop_columns, axis=1)
    
    # Encode categorical data
    labelEncoder = LabelEncoder()
    data.Sex = labelEncoder.fit_transform(data.Sex)
    data = data.fillna({'Age': data.Age.median()})
    data = data.fillna({'Fare': data.Fare.median()})
    
    return data

train = preprocess_data(train)

X_train = train.drop(['Survived'], axis=1)
y_train = train['Survived']
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(n_estimators = 50, random_state = 0)
forest_model.fit(X_train, y_train)
test = pd.read_csv('../input/test.csv')
sanitized_test = preprocess_data(test)

X_test = sanitized_test

predicted = forest_model.predict(X_test)

# Round to nearest int
predicted = np.rint(predicted).astype(int)
my_submission = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': predicted})
my_submission.to_csv('submission.csv', index=False)