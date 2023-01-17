import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Sklearn model packages
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
train_data = pd.read_csv('../input/titanic/train.csv')
test_data = pd.read_csv('../input/titanic/test.csv')

#Setting up data to train the model
y = train_data['Survived']
features = ['SibSp','Parch','Fare','Sex','Pclass']
object_cols = ['Sex']
X = train_data[features]

#Using label encoder to classify sex data
label_encoder = LabelEncoder()
label_X = X.copy()
for col in object_cols:
    label_X[col] = label_encoder.fit_transform(label_X[col])
ship_model = RandomForestRegressor(random_state=1)
ship_model.fit(label_X, y)
X_test = test_data[features]
X_test_label = X_test.copy()

#Encoding sex for test data
for col in object_cols:
    X_test_label[col] = label_encoder.fit_transform(X_test_label[col])

#Imputing missing data
my_imputer = SimpleImputer()
imputed_X_test = pd.DataFrame(my_imputer.fit_transform(X_test_label))

imputed_X_test.columns = X_test_label.columns
predictions = ship_model.predict(imputed_X_test)
def round(number):
    if number >= 0.5:
        return 1
    else:
        return 0
    
#Converting the predictions to a Pandas series in order to round up or down based on 
#probability of survival.

pre_series = pd.Series(predictions)

rounded_results = pre_series.apply(round)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': rounded_results})
output.to_csv('../output/my_submission.csv', index=False)
print("Your submission was successfully saved!")