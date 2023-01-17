import pandas  as pd



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



#===========================================================================

# create the kernel 

#===========================================================================

from sklearn.gaussian_process.kernels import RBF

kernel = 1.0 * RBF(1.0)



#===========================================================================

# perform the classification

#===========================================================================

from sklearn.gaussian_process import GaussianProcessClassifier

classifier = GaussianProcessClassifier(kernel=kernel)



#===========================================================================

# and the fit 

#===========================================================================

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