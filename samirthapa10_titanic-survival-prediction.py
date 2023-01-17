# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_path='/kaggle/input/titanic/train.csv'
test_path='/kaggle/input/titanic/test.csv'
train_data=pd.read_csv(train_path)
test_data=pd.read_csv(test_path)
train_data.info()
test_data.info()
train = train_data.copy()
test = test_data.copy()
train.dropna(subset = ["Embarked"], inplace=True)
train.describe()
y_train=train.Survived
features=['Pclass','Sex','Age','Fare','SibSp','Parch','Embarked']
X_train=train[features]
X_test=test[features]

X_train.columns
plt.figure(figsize=(6,6))
sns.barplot(x=X_train.Sex, y=y_train)
plt.figure(figsize=(6,6))
sns.barplot(x=X_train.Embarked, y=y_train)
# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)

#Unique string present in Cols 

print("Unique values in 'Sex' column in training data:", X_train['Sex'].unique())
print("Unique values in 'Embarked' column in training data:", X_train['Embarked'].unique())

print("\nUnique values in 'Sex' column in validation data:", X_test['Sex'].unique())
print("Unique values in 'Embarked' column in validation data:", X_test['Embarked'].unique())
from sklearn.preprocessing import OneHotEncoder


OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_val = pd.DataFrame(OH_encoder.transform(X_test[object_cols]))

OH_cols_train.index = X_train.index
OH_cols_val.index = X_test.index

num_X_train =X_train.drop(object_cols, axis=1)
num_X_val = X_test.drop(object_cols, axis=1)

OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_test = pd.concat([num_X_val, OH_cols_val], axis=1)
        
 
OH_X_train.describe()
#Missing data in train data
missing_val_count_by_column = (OH_X_train.isnull().sum())
print("Missing data in train data:")
print(missing_val_count_by_column[missing_val_count_by_column > 0])

#Missing data in test data
missing_val_count_by_column = (OH_X_test.isnull().sum())
print("Missing data in test data:")
print(missing_val_count_by_column[missing_val_count_by_column > 0])
from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(OH_X_train))
imputed_X_test = pd.DataFrame(my_imputer.transform(OH_X_test))

imputed_X_train.columns = OH_X_train.columns
imputed_X_test.columns = OH_X_test.columns

imputed_X_train.head
#Missing data in train data after using SimpleImputer
notavailable_val_count_by_column = (imputed_X_train.isnull().sum())
print("Missing data after imputed in train data:")
print(missing_val_count_by_column[notavailable_val_count_by_column > 0])

#Missing data in test data after using SimpleImputer
notavailable_val_count_by_column = (imputed_X_test.isnull().sum())
print("Missing data after imputed in test data:")
print(missing_val_count_by_column[notavailable_val_count_by_column > 0])
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

train_imputed_X_train, val_imputed_X_train, train_y_train, val_y_train = train_test_split(imputed_X_train, y_train, random_state=1)

def get_mae(max_leaf_nodes, train_imputed_X_train, val_imputed_X_train, train_y_train, val_y_train):
    
    train_model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes,random_state=0)
    
    train_model.fit(train_imputed_X_train,train_y_train)
    
    val_predictions=train_model.predict(val_imputed_X_train)
    val_mae = mean_absolute_error(val_predictions, val_y_train)    
    return val_mae
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(max_leaf_nodes,train_imputed_X_train, val_imputed_X_train, train_y_train, val_y_train)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

scores = {leaf_size: get_mae(leaf_size,train_imputed_X_train, val_imputed_X_train, train_y_train, val_y_train) for leaf_size in candidate_max_leaf_nodes}
print("\n",scores)

best_tree_size = min(scores, key=scores.get)

print("\n", best_tree_size, scores[best_tree_size])
train_model = RandomForestRegressor(max_leaf_nodes=best_tree_size,random_state=0)
train_model.fit(train_imputed_X_train,train_y_train)
val_predictions=train_model.predict(val_imputed_X_train)
val_mae = mean_absolute_error(val_predictions, val_y_train)
print(val_mae)
predict=train_model.predict(imputed_X_test)
print(predict)
predict[predict > 0.5] = int(1)
predict[predict < 0.5] = 0
print(type(predict))
predict = predict.astype(int)
print(predict)
output = pd.DataFrame({'PassengerId': test_data.PassengerId,'Survived': predict})
print(output)

output.to_csv('submission.csv', index=False)
print("Your Output has been saved")
