# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer
# Load data and visualize

train_data = pd.read_csv('../input/train.csv')



train_data.head(5)
# Remove columns we dont need

train_data = train_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])



# Replace Categorical features to One Hot Encoding

train_data = pd.get_dummies(train_data)



# Replace NaN values

columns = list(train_data)

my_imputer = SimpleImputer()

new_train_data = my_imputer.fit_transform(train_data)

train_data = pd.DataFrame(data=new_train_data, columns=columns)



train_data.head(5)
# Split train data

y_train = train_data['Survived']

x_train = train_data.drop(columns='Survived')
# Create Simple Random Forest

rf = RandomForestClassifier(n_estimators=32, 

                            random_state=100,

                            max_depth=8,

                            min_samples_split = 2)

print(rf.fit(x_train, y_train))



# Print Train accuracy

print("Train Accuracy:", rf.score(x_train, y_train))
# Load Test Input

x_test = pd.read_csv('../input/test.csv')



passengers = x_test['PassengerId']

x_test = x_test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])



x_test = pd.get_dummies(x_test)



# Replace NaN values

columns = list(x_test)

my_imputer = SimpleImputer()

new_x_test = my_imputer.fit_transform(x_test)

x_test = pd.DataFrame(data=new_x_test, columns=columns)



# Load Test Output

y_test = pd.read_csv('../input/gender_submission.csv')

y_test = y_test['Survived']



# Print Test accuracy

print("Test Accuracy:", rf.score(x_test, y_test))
# Save Predicted values

pred = rf.predict(x_test)



final_df = pd.DataFrame({'PassengerId': passengers, 'Survived': pred})



final_df = final_df.astype('int')



final_df.to_csv('output.csv', index=False)