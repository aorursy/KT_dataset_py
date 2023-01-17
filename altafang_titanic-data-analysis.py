%matplotlib inline



import numpy as np 

import pandas as pd

from matplotlib import pyplot

import seaborn

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score



# Read in data

training_set = pd.read_csv('../input/train.csv')
print(training_set.head())



# Check for nulls

print("Number of null values in training set:")

print(training_set.isnull().sum())



# Overall shape

print(training_set.shape)
# Fraction of females who survived and fraction of males who survived

print("Fraction of each sex who survived")

print(training_set[['Sex', 'Survived']].groupby('Sex').mean())

# The same result can be gotten using:

# training_set.groupby('Sex')['Survived'].mean()

# but the latter will give a Series instead of a DataFrame

# https://github.com/jakevdp/PythonDataScienceHandbook/blob/master/notebooks/03.09-Pivot-Tables.ipynb

print()

print("Fraction who survived by sex and Pclass")

print(training_set.pivot_table('Survived', index='Sex', columns='Pclass'))



print()

print("Total number who survived vs. did not")

print(training_set['Survived'].value_counts())
# Try scatterplot of Age and Fare. There's no pattern though...

pyplot.scatter(training_set['Age'], training_set['Fare'], c=training_set['Survived'], 

               cmap='rainbow')

pyplot.colorbar()

pyplot.show()
def prepare_features(data, training=True):

    """Take input DataFrame and return array of features to input into training models. 

    Modifies input in-place.

    

    Parameters

    ----------

    data : pandas.DataFrame

        input data

    training : boolean (optional)

        flag for whether this is the training or test set (to decide whether to fit 

        Scaler) object, etc.

        

    Returns

    -------

    X : numpy.array

        2D array of cleaned up features for input into model

    """

    # Encode categorical features

    if training:

        prepare_features.sex_enc = LabelEncoder() # Attribute of this function

        prepare_features.sex_enc.fit(data['Sex'])

    data['Sex'] = prepare_features.sex_enc.transform(data['Sex'])

    

    # get_dummies does one hot encoding and automatically handles nan's by making them all 0

    data = data.join(pd.get_dummies(data['Embarked']))



    # Impute missing values in Age and Fare using mean

    data['Age'] = data['Age'].fillna(value=data['Age'].mean()) 

    data['Fare'] = data['Fare'].fillna(value=data['Fare'].mean()) 



    # Add name length as a feature to replace name (idea from a public kernel on kaggle)

    data['Name Length'] = data['Name'].apply(len)



    # Choose features to consider. Last few are from embarked

    features = ['Age', 'Sex', 'Fare', 'Pclass', 'SibSp', 'Name Length', 'C', 'Q', 'S']



    X = np.array(data[features]) # Features



    # Scale features

    if training:

        prepare_features.scaler = StandardScaler()

        prepare_features.scaler.fit(X)

    X = prepare_features.scaler.transform(X)

    

    return X



X = prepare_features(training_set, training=True)

y = np.array(training_set['Survived']) # Outcomes

    

# Split to get a cross-validation set to better estimate performance

X_train, X_cv, y_train, y_cv = train_test_split(X, y, test_size=0.25, random_state=42)
clf = SVC()



# Train the model

clf.fit(X_train, y_train)
predicted_y = clf.predict(X_cv)

print("accuracy of test: ", accuracy_score(y_cv, predicted_y))

predicted_y = clf.predict(X_train)

print("accuracy of training: ", accuracy_score(y_train, predicted_y))
# Better method for cross-validation: multiple splits

# https://jakevdp.github.io/PythonDataScienceHandbook/05.03-hyperparameters-and-model-validation.html

scores = cross_val_score(clf, X, y, cv=4)

scores.mean()
# Plot the learning curve

N, train_lc, val_lc = learning_curve(clf, X, y, cv=4)



pyplot.plot(N, np.mean(train_lc, 1), color='blue', label='training score')

pyplot.plot(N, np.mean(val_lc, 1), color='red', label='validation score')

pyplot.xlabel("training size")

pyplot.ylabel("score")

pyplot.legend()

pyplot.show()
param_grid = [{'kernel':('linear', 'rbf'), 'C':[0.1, 1, 10, 100], 'gamma': [0.0001, 0.001, 0.01]}]



svc = SVC()

clf = GridSearchCV(svc, param_grid)

clf.fit(X, y)

# Check what the best parameters were

print(clf.best_params_)

model = clf.best_estimator_

model = model.fit(X, y)

scores = cross_val_score(model, X, y, cv=4)

print(scores.mean())
test_set = pd.read_csv("../input/test.csv")



X_test = prepare_features(test_set, training=False)



y_test = model.predict(X_test)

predictions = pd.DataFrame(data={'Survived': y_test}, index=test_set['PassengerId'])

predictions.to_csv("submission.csv")