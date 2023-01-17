# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
titanic = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic
#Lets start with a basic SVM implementation to see whether or not the data itself is ready to be processed.

# Note: tried to do this with loc but it said that pandas doesn't support list-like indexing in loc with missing index (example:

# i didn't include name and ticket in the label indexing with loc, therefore it is treated as a missing index and the label indexing list isn't

# legit.



X = titanic.iloc[:,[0,2,4,5,6,7,8,9,10]]

X



y = titanic.iloc[:, 1]

y
from sklearn import svm



titanic_SVM_check = svm.SVC(random_state = 1)

titanic_SVM_check.fit(X,y)



check_predictions = titanic_SVM_check.predict(X)

check_predictions
one_hot_encoded_X = pd.get_dummies(X)

titanic_SVM_check.fit(one_hot_encoded_X,y)



check_predictions = titanic_SVM_check.predict(one_hot_encoded_X)

check_predictions
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

one_hot_encoded_imputed_X = my_imputer.fit_transform(one_hot_encoded_X)

one_hot_encoded_imputed_X
titanic_SVM_check_final = svm.SVC(random_state = 1)

titanic_SVM_check_final.fit(one_hot_encoded_imputed_X,y)



check_predictions = titanic_SVM_check_final.predict(one_hot_encoded_imputed_X)

print(check_predictions[0:5])

print(y[0:5])
#Lets take a look at the dataset again

X = titanic.drop(['Name','Ticket', 'Survived'], axis = 1)

X



y = titanic.iloc[:, 1]

y
# A workshop i've attended suggest me to scale/standarize the data.

# According to it, we can use one of scikit-learn's function. Handy!

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



one_hot_encoded_X = pd.get_dummies(X)



my_imputer = SimpleImputer()

one_hot_encoded_imputed_X = my_imputer.fit_transform(one_hot_encoded_X)

one_hot_encoded_imputed_X



scaler_is_ready = scaler.fit(one_hot_encoded_imputed_X)



scaled_imputed_and_encoded_X = scaler.transform(one_hot_encoded_imputed_X)

scaled_imputed_and_encoded_X
# Alright! X is scaled,imputed, and encoded! now lets split it, along with y

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(scaled_imputed_and_encoded_X, y, random_state=1)
# That specific workshop also instructed us to use RBF kernel. Most of the codes below are from the scikit learn documentation.

# Honestly i am baffled. I don't know what they're talking about in the math section of SVM but i know these codes do select the best C and

# gamma.

from sklearn.svm import SVC

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import GridSearchCV



param_grid = {

    'C' : 10.0 ** np.arange(-3,4),

    'gamma' : 10.0 ** np.arange (-3,4),

}

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)

grid.fit(train_X, train_y)

svm = grid.best_estimator_

svm_val = grid.best_score_



# We've selected... the best SVM! alright then lets use the model

svm.fit(train_X,train_y)

predictions = svm.predict(val_X)

print(predictions[0:10])

print(val_y[0:10])
#Defining a pre-processing function based on what we've done before.

def preprocessing_for_titanic_features(X):

    from sklearn.preprocessing import StandardScaler

    from sklearn.impute import SimpleImputer

    

    # First, let us encode the categorical datas

    one_hot_encoded_X = pd.get_dummies(X)

    

    # Now, let us imput the NaNs (including if there are any NaNs in the encoded data)

    my_imputer = SimpleImputer()

    one_hot_encoded_imputed_X = my_imputer.fit_transform(one_hot_encoded_X)

    one_hot_encoded_imputed_X

    

    # Now, let us prepare the data so that it's scaling is appropriate for SVM.

    scaler = StandardScaler()

    scaler_is_ready = scaler.fit(one_hot_encoded_imputed_X)

    

    encoded_imputed_and_scaled_X = scaler.transform(one_hot_encoded_imputed_X)

    

    return encoded_imputed_and_scaled_X
test_data_path = '/kaggle/input/titanic/test.csv'

test_data = pd.read_csv(test_data_path)

test_X = test_data.drop(['Ticket', 'Name'], axis = 1)

test_X
preprocessed_test_X = preprocessing_for_titanic_features(test_X)

preprocessed_test_X
# Lets use our svm

test_preds = svm.predict(preprocessed_test_X)
X = titanic.iloc[:,[0,2,4,5,6,7,8,9,10]]

y = titanic.iloc[:, 1]

test_data_path = '/kaggle/input/titanic/test.csv'

test_data = pd.read_csv(test_data_path)

test_X = test_data.drop(['Ticket', 'Name'], axis = 1)



from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



one_hot_encoded_X = pd.get_dummies(train_X)

one_hot_encoded_test_X = pd.get_dummies(test_X)

final_train, final_test = one_hot_encoded_X.align(one_hot_encoded_test_X,

                                                                    join='left', 

                                                                    axis=1)



from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

imputer_for_both_train_and_test = my_imputer.fit(final_train)

imputed_final_train = imputer_for_both_train_and_test.transform(final_train)

imputed_final_test = imputer_for_both_train_and_test.transform(final_test)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



scaler_for_BOTH_TRAIN_AND_TEST = scaler.fit(imputed_final_train)    

imputed_and_scaled_final_train = scaler_for_BOTH_TRAIN_AND_TEST.transform(imputed_final_train)

imputed_and_scaled_final_test = scaler_for_BOTH_TRAIN_AND_TEST.transform(imputed_final_test)



from sklearn.svm import SVC

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import GridSearchCV



param_grid = {

    'C' : 10.0 ** np.arange(-3,4),

    'gamma' : 10.0 ** np.arange (-3,4),

}

grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5)

grid.fit(imputed_and_scaled_final_train,train_y)

svm = grid.best_estimator_

svm_val = grid.best_score_



svm.fit(imputed_and_scaled_final_train,train_y)

test_preds = svm.predict(imputed_and_scaled_final_test)
# Making the output...

output = pd.DataFrame({'PassengerId': test_X['PassengerId'],

                       'Survived': test_preds})

output.to_csv('submission.csv', index=False)