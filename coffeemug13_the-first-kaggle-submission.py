import numpy as np

import pandas as pd



import matplotlib.pyplot as plt 
# Let's load both sets

train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



# Let's load the sample submission

submission = pd.read_csv('../input/sample_submission.csv')
for col in train_data:

    values = {}

    print('column: ' + col)

    for val in train_data[col]:

        if val.isnumeric():

            break

        if val not in values:

            values[str(val)] = 1

    print('values for column <' + col + '>:',end='')

    print(";".join(values.keys()))

 

  
# Let's take a look at the first 5 entries of the Kaggle training set

train_data.head(5)
print("Number of observations: %i" % len(train_data))

print("List of columns: %s" % ", ".join(train_data.columns))
# Let's take a look at the first 5 entries of the Kaggle testing set

test_data.head(5)
print("Number of observations: %i" % len(test_data))

print("List of columns: %s" % ", ".join(test_data.columns))
submission.head(5)
print("Number of observations: %i" % len(submission))

print("List of columns: %s" % ", ".join(submission.columns))
train_data.describe()
from sklearn import preprocessing



label_encoders = {}

category_labels = {}



def transform_x(data_df, phase="train"):

    """Transforms the input dataframe to a dataframe containing

    the input variables (= features)"""

    X = data_df.drop(['id'], axis=1)

    

    if 'loss' in X.columns:

        X = X.drop(['loss'], axis=1)

    

    # List of categorical features

    cat_features = X.select_dtypes(include=['object']).columns



    # List of numerical features

    num_features = X.select_dtypes(exclude=['object']).columns

    

    # Replace each categorical feature with encoded labels

    for cat in cat_features:

        if phase == "train":

            # Let's store the used labels

            category_labels[cat] = list(set(X[cat]))     

  

            # We need to fit the Label Encoder in the training phase

            label_encoders[cat] = preprocessing.LabelEncoder()

            label_encoders[cat].fit(X[cat])

        

        # We replace unseen labels by the first label

        mask = X[cat].apply(lambda x: x not in category_labels[cat])

        X.loc[mask, cat] = category_labels[cat][0]

        

        X[cat] = label_encoders[cat].transform(X[cat])

    

    return X



def transform_y(data_df):

    """Transforms the input dataframe to a dataframe containing

    the ground truth data"""

    y = data_df['loss']

    

    # You can do some crazy stuff here

    # y = np.log(y)

    

    return y



def inverse_transform_y(data):

    """Inverse transforms the y values to match the original

    Kaggle testing set"""

    y = data

    

    # You should invert all the crazy stuff

    # y = np.exp(y)

    

    return y
X_train_df = transform_x(train_data)

y_train_df = transform_y(train_data)
import xgboost as xgb



# Create our DMatrix to make XGBoost more efficient

xgdmat_train = xgb.DMatrix(X_train_df.values, y_train_df.values)



params = {'eta': 0.01, 'seed':0, 'subsample': 0.5, 'colsample_bytree': 0.5, 

             'objective': 'reg:linear', 'max_depth':6, 'min_child_weight':3} 



num_rounds = 100

mdl = xgb.train(params, xgdmat_train, num_boost_round=num_rounds)
X_test_df = transform_x(test_data, phase="test")
xgdmat_test = xgb.DMatrix(X_test_df.values)

y_pred = mdl.predict(xgdmat_test)
submission.iloc[:, 1] = inverse_transform_y(y_pred)

submission.to_csv('vienna_kaggle_submission.csv', index=None)