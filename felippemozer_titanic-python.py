# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score
# Get data

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
# Drop unused columns on analysis

train.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)

test.drop(['Name','Ticket','Cabin'], axis=1, inplace=True)
# Impute Method to replace missing/NA values

imputer = SimpleImputer(strategy="median")
# Here, we drop the cathegorical columns to impute numeric columns

# These columns will be treated later

number_train = train.drop(['Sex', 'Embarked'], axis=1)

number_test = test.drop(['Sex', 'Embarked'], axis=1)
# Applying imputer on train and test files

imputer.fit(number_train)

fitted_train = imputer.transform(number_train)



imputer.fit(number_test)

fitted_test = imputer.transform(number_test)
# imputer.transform returns a NumPy array, and we need data frames

# So, we convert fitted_train and fitted_test

train_frame = pd.DataFrame(fitted_train, columns=number_train.columns)

test_frame = pd.DataFrame(fitted_test, columns=number_test.columns)
# Now, we get the cathegorical variables to manipulate and remove missing values

# Also, we transform these data using One Hot Encoding technique

cath_train = train[['Sex','Embarked']]

cath_test = test[['Sex','Embarked']]



# get_dummies function, from Pandas library, fits perfectly on that case

one_hot_cath_train = pd.get_dummies(cath_train)

one_hot_cath_test = pd.get_dummies(cath_test)
# Join imputed data with encoded data

new_train = train_frame.join(one_hot_cath_train)

new_test = test_frame.join(one_hot_cath_test)
# Time to prepare train data to apply ML

# We split 20% of the entire file to final tests

X = new_train.drop(['PassengerId','Survived'], axis=1)

y = train.Survived



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# It's time to finally apply ML!

# Using GradientBoostClassifier, we got around 85% accuracy

gradient_boost = GradientBoostingClassifier()

gradient_boost.fit(X_train, y_train)

y_gb = gradient_boost.predict(X_test)



acc_gradient_boost = round(accuracy_score(y_gb, y_test) * 100, 2)

#print(acc_gradient_boost)
# After training your model, let's predict test data

ids = test['PassengerId'] # ids' an auxiliary variable used on submission

predictions = gradient_boost.predict(new_test.drop('PassengerId', axis=1))
# Submit to Kaggle's Competition

submission = pd.DataFrame({'PassengerId': ids, 'Survived': predictions.astype('int64')})

submission.to_csv('submission.csv', index=False)