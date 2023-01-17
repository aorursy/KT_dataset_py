!pip install decision-tree-id3

import pandas  as pd



# The following is a workaround for: ImportError: cannot import name 'six' from 'sklearn.externals' 

import six

import sys

sys.modules['sklearn.externals.six'] = six
#===========================================================================

# read in the Titanic data

#===========================================================================

train_data = pd.read_csv('../input/titanic/train.csv')

test_data  = pd.read_csv('../input/titanic/test.csv')



#===========================================================================

# select some features

#===========================================================================

features = ["Pclass", "Sex", "SibSp", "Parch"]



#===========================================================================

# for the features that are categorical we use pd.get_dummies

#===========================================================================

X_train       = pd.get_dummies(train_data[features])

y_train       = train_data["Survived"]

final_X_test  = pd.get_dummies(test_data[features])



#===========================================================================

# perform the classification

#===========================================================================

from id3 import Id3Estimator

classifier = Id3Estimator()

classifier.fit(X_train, y_train)



#===========================================================================

# use the model to predict 'Survived' for the test data

#===========================================================================

predictions = classifier.predict(final_X_test)



#===========================================================================

# write out CSV submission file

#===========================================================================

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 

                       'Survived': predictions})

output.to_csv('submission.csv', index=False)
