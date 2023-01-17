# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



!pip3 install pyod



import os

import itertools

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#from sklearn.pipeline import Pipeline

from pyod.models.iforest import IForest

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import IterativeImputer

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from xgboost import DMatrix

from xgboost import cv

from sklearn.metrics import mean_absolute_error

#import matplotlib.pyplot as plt

import category_encoders as ce

from sklearn.feature_selection import SelectFromModel

from sklearn.feature_selection import SelectKBest, f_classif



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Load training data

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.columns

train_data.head(2)
# Load testing data

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.columns

test_data.head(2)
# Define a function to strip the title out of the name

def get_title(fullname):

    start_chars = ', '

    end_chars = '. '

    start_index = fullname.find(start_chars) + len(start_chars)

    end_index = fullname.find(end_chars)

    return fullname[start_index : end_index]
# Select feature columns to train the model

copy_cols = ['Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']



# Select the label column

label_col = 'Survived'



# Strip out the label

X = train_data[copy_cols].copy()

X_test = test_data[copy_cols].copy()

y = train_data[label_col].copy()
# Binary encode Sex

X['Sex'] = X['Sex'].map( {'female':0, 'male':1} )

X_test['Sex'] = X_test['Sex'].map( {'female':0, 'male':1} )



# Create Title column

X['Title'] = X['Name'].apply(lambda x: get_title(x))

X_test['Title'] = X_test['Name'].apply(lambda x: get_title(x))



# Create YoungBoy column

X['YoungBoy'] = [1 if x == 'Master' else 0 for x in X['Title']]

X_test['YoungBoy'] = [1 if x == 'Master' else 0 for x in X_test['Title']]



# Create OfficerOrRev column

X['OfficerOrRev'] = [1 if x == 'Rev' or x == 'Capt' or x == 'Col' or x == 'Major' else 0 for x in X['Title']]

X_test['OfficerOrRev'] = [1 if x == 'Rev' or x == 'Capt' or x == 'Col' or x == 'Major' else 0 for x in X_test['Title']]



# Correct some titles

X['Title'] = X['Title'].replace('Mme', 'Mrs')

X_test['Title'] = X_test['Title'].replace('Mme', 'Mrs')

X['Title'] = X['Title'].replace('Ms', 'Miss')

X_test['Title'] = X_test['Title'].replace('Ms', 'Miss')

X['Title'] = X['Title'].replace('Mlle', 'Miss')

X_test['Title'] = X_test['Title'].replace('Mlle', 'Miss')



# Drop Name column

X = X.drop(['Name'], axis=1)

X_test = X_test.drop(['Name'], axis=1)



# Label encode Title

l_encoder_title = LabelEncoder()

l_encoder_title.fit((pd.concat([X, X_test]))['Title'])

X['Title'] = l_encoder_title.transform(X['Title'])

X_test['Title'] = l_encoder_title.transform(X_test['Title'])



# Set the two missing Embarked values to Southampton (https://www.encyclopedia-titanica.org/titanic-survivor/martha-evelyn-stone.html)

X['Embarked'] = X['Embarked'].fillna('S')



# Label encode Embarked

l_encoder_embarked = LabelEncoder()

l_encoder_embarked.fit(X['Embarked'])

X['Embarked'] = l_encoder_embarked.transform(X['Embarked'])

X_test['Embarked'] = l_encoder_embarked.transform(X_test['Embarked'])





# Split the data

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=3)





# One-hot encode category columns

#X_train = pd.get_dummies(X_train)

#X_val = pd.get_dummies(X_val)

#X_test = pd.get_dummies(X_test)



# Imput missing values

column_names = X_train.columns

estimator = GradientBoostingRegressor(n_estimators=8000, max_depth=2, loss = 'ls', learning_rate = .001, random_state=0)

imp = IterativeImputer(imputation_order='ascending')

X_train = pd.DataFrame(imp.fit_transform(X_train))

X_val = pd.DataFrame(imp.transform(X_val))

X_test = pd.DataFrame(imp.transform(X_test))

X_train.columns = column_names

X_val.columns = column_names

X_test.columns = column_names



X_train.sample(5)

# Create outlier detection model

oulier_model = IForest(contamination=0.025,random_state=0)

oulier_model.fit(X_train[['Age', 'SibSp', 'Parch', 'Fare']])    #[['Age', 'SibSp', 'Parch', 'Fare']]

# predict raw anomaly score

#scores_pred = oulier_model.decision_function(X_train) * -1

# prediction of a datapoint category outlier or inlier

y_pred = oulier_model.predict(X_train[['Age', 'SibSp', 'Parch', 'Fare']])

X_train['Outlier'] = y_pred

X_train[X_train['Outlier'] == 1]
# Get indexes to keep

keep_indexes = X_train.index[X_train['Outlier'] != 1]

# Remove outliers from the training sets

X_train = X_train.loc[keep_indexes]

y_train = y_train.iloc[keep_indexes]

X_train = X_train.drop(['Outlier'], axis=1)

X_train.shape, y_train.shape
# Create age group feature

X_train['Age_Group'] = pd.cut(x=abs(X_train['Age']), bins=[0, 18, 30, 55, 100], labels=[1, 2, 3, 4]).astype('float64')

X_val['Age_Group'] = pd.cut(x=abs(X_val['Age']), bins=[0, 18, 30, 55, 100], labels=[1, 2, 3, 4]).astype('float64')

X_test['Age_Group'] = pd.cut(x=abs(X_test['Age']), bins=[0, 18, 30, 55, 100], labels=[1, 2, 3, 4]).astype('float64')



# Create family size feature

X_train['Family_Size'] = X_train['SibSp'] + X_train['Parch'] + 1

X_val['Family_Size'] = X_val['SibSp'] + X_val['Parch'] + 1

X_test['Family_Size'] = X_test['SibSp'] + X_test['Parch'] + 1



X_train.sample(10)
# Select feature columns to train the model

feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'YoungBoy', 'OfficerOrRev', 'Age_Group', 'Family_Size']



# Select category columns

#cat_cols = ['Pclass', 'Title', 'Sex', 'Embarked_C', 'Embarked_Q', 'Embarked_S']



# Count encode category columns

#ce_target = ce.CountEncoder(cols=cat_cols)

#ce_target.fit(X[cat_cols], y)

#X[cat_cols] = ce_target.transform(X[cat_cols])

#X_val[cat_cols] = ce_target.transform(X_val[cat_cols])

#X_test[cat_cols] = ce_target.transform(X_test[cat_cols])



# Select interaction pair columns

inter_cols = ['Pclass', 'Sex', 'Embarked', 'YoungBoy', 'OfficerOrRev']



# Encode interaction pairs

for col1, col2 in itertools.combinations(inter_cols, 2):

    new_col_name = '_'.join([col1, col2])

    # Convert to strings and combine

    train_values = X_train[col1].map(str) + "_" + X_train[col2].map(str)

    val_values = X_val[col1].map(str) + "_" + X_val[col2].map(str)

    test_values = X_test[col1].map(str) + "_" + X_test[col2].map(str)

    label_enc = LabelEncoder()

    X_train[new_col_name] = label_enc.fit_transform(train_values)

    X_val[new_col_name] = label_enc.transform(val_values)

    X_test[new_col_name] = label_enc.transform(test_values)

    feature_cols.append(new_col_name)

    

encoded_cols = X_train.columns

encoded_cols



#X_train = pd.concat([X_train, X_val])

#y_train = pd.concat([y_train, y_val])
X_train.shape, X_test.shape
# Select best features with univariate method

selector = SelectKBest(f_classif, k=6)

X_new = selector.fit_transform(X_train, y_train)

all_features = pd.DataFrame(selector.inverse_transform(X_new), 

                                 index=pd.DataFrame(X_train).index, 

                                 columns=encoded_cols)

selected_features = all_features.columns[all_features.var() != 0]

selected_features
# Create data matrix

data_dmatrix = DMatrix(data=X_train[selected_features], label=y_train)


params = {"objective":"multi:softmax", 'num_class': 2,'colsample_bytree': 0.7,'learning_rate': 0.01,

                'max_depth': 8, 'alpha': 10}



cv_results = cv(dtrain=data_dmatrix, params=params, nfold=5, num_boost_round=100, early_stopping_rounds=10,

                metrics="merror", as_pandas=True, seed=100)



print((cv_results))
#model = XGBClassifier(n_estimators=1000, learning_rate=0.01, colsample_bytree=0.7, objective="multi:softmax", num_class=2, max_depth=8, alpha=10)

model = GradientBoostingClassifier(n_estimators=500, max_depth=4, learning_rate=0.01)

model.fit(X_train[selected_features], y_train)  #,

             #early_stopping_rounds=10, 

             #eval_set=[(X_val[selected_features], y_val)], 

             #verbose=False)



val_prediction = model.predict(X_val[selected_features])



# Calculate MAE

mae = mean_absolute_error(val_prediction, y_val)

print("Mean Absolute Error:" , mae)
# Predict using the test data

predictions = model.predict(X_test[selected_features])

predictions.shape
# Generate output for submission

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('my_XGBoost_submission.csv', index=False)

print("Your submission was successfully saved!")