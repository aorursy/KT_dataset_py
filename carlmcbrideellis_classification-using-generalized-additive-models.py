!pip install pygam
import pandas  as pd

import matplotlib.pyplot as plt



#===========================================================================

# read in the data

#===========================================================================

train_data = pd.read_csv('../input/titanic/train.csv')

test_data  = pd.read_csv('../input/titanic/test.csv')



#===========================================================================

# select some features of interest ("ay, there's the rub", Shakespeare)

#===========================================================================

features = ["Pclass", "Sex", "SibSp", "Parch"]



#===========================================================================

# for the features that are categorical we use pd.get_dummies:

# "Convert categorical variable into dummy/indicator variables."

#===========================================================================

X_train       = pd.get_dummies(train_data[features])

y_train       = train_data["Survived"]

final_X_test  = pd.get_dummies(test_data[features])
from pygam import LogisticGAM, s, f



classifier = LogisticGAM(s(0) + s(1) + s(2) + s(3))



classifier.fit(X_train, y_train)



#===========================================================================

# use the model to predict 'Survived' for the test data

#===========================================================================

predictions = classifier.predict(final_X_test)



# convert from True/False to 1/0

predictions = (predictions)*1
classifier.summary()
plt.figure(figsize=(15, 4))

for i, term in enumerate(classifier.terms):

    if term.isintercept:

        continue

    plt.plot(classifier.partial_dependence(term=i), label="s({})".format(i))

    plt.legend()
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

output.to_csv('submission.csv', index=False)