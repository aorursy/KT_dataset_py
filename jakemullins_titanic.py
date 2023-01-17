import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv") #?

test = pd.read_csv("../input/titanic/test.csv", index_col = 'PassengerId')

train = pd.read_csv("../input/titanic/train.csv", index_col = 'PassengerId')
'''

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import LabelEncoder

from xgboost import XGBRegressor



#create dataframse of our selected features

features = ['Pclass', 'Sex', 'Age']

X = train[features]

y = train.Survived

#separate dtypes



X_num = pd.DataFrame(X['Age'])

X_num

X_cat = pd.DataFrame(X[['Pclass', 'Sex']])

X_cat



#impute numerical columns

imputer = SimpleImputer(missing_values=np.nan, strategy='median')

imputed_X = pd.DataFrame(imputer.fit_transform(X_num))

imputed_X.columns = X_num.columns

imputed_X.index = X_num.index

imputed_X

#OneHotEncode categorical columns

encoder = LabelEncoder()

encoded_X = pd.DataFrame(encoder.fit_transform(X_cat))

encoded_X.index = X_cat.index

encoded_X



#recombine num and cat columns





final_X = imputed_X.join(encoded_X)



my_model = XGBRegressor(n_estimators=100)

my_model.fit(X,y)

'''

scores = -1 * cross_val_score(my_model, X, y,

                              cv=5,

                              scoring='neg_mean_absolute_error')
def get_mae(my_model):

    #fit

    my_model.fit(X_train, y_train)

    # Get predictions

    predictions = my_model.predict(X_valid)

    # Calculate MAE

    mae = mean_absolute_error(predictions, y_valid)

    print('MAE: {}'.format(mae))

    return mae
features = ['Pclass', 'Sex', 'Age']

train_X = train[features]

train_y = train['Survived']



#change Sex to a boolean

train_X = train_X.replace({'male': True, 'female': False})



#Impute missing Ages

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputed_train_X = pd.DataFrame(imputer.fit_transform(train_X))

imputed_train_X.columns = train_X.columns

imputed_train_X.index = train_X.index



#create randomTree model

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBRegressor

my_model_1 = XGBRegressor(random_state = 0, n_estimators=200)

my_model_1.fit(imputed_train_X, train_y)



#change test file to same format as the training data

test.head()

test_X = test[features]

test_X = test_X.replace({'male':True, 'female':False})

imputed_test_X = pd.DataFrame(imputer.fit_transform(test_X))

imputed_test_X.columns = test_X.columns

imputed_test_X.index = test_X.index
#make predictions

predictions = my_model_1.predict(imputed_test_X)

submission = pd.DataFrame(predictions, index = test_X.index)

submission.columns = ['Survived']

submission = submission.reset_index()

submission.to_csv('submission.csv', index=False)