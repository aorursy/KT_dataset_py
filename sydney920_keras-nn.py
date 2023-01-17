import math

import numpy as np

import pandas

from keras.layers import Dense

from keras.layers import Dropout

from keras.models import Sequential

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
root_path = '../input'



def get_data(filepath):

    df = pandas.read_csv(filepath)

    return get_data_sets(df)



features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked']

def get_data_sets(df):

    df['Sex'] = df['Sex'].apply(lambda s: 0 if s == 'male' else 1)

    df['Age'] = df['Age'].apply(lambda a: df['Age'].median() if math.isnan(a) else a)

    df['Fare'] = df['Fare'].fillna(df['Fare'].dropna().median())

    df['Embarked'] = df['Embarked'].fillna('S')

    df['Embarked'] = df['Embarked'].apply(lambda x: 1 if (x == 'Q') else (2 if (x == 'S') else 3))

    x = StandardScaler().fit_transform(df[features].values)

    y = [] 

    if 'Survived' in df:

        y = df['Survived']

        return x, y

    return x
def get_model(dropout=0.0):

    m = Sequential()

    m.add(Dense(input_dim=len(features), output_dim=50, activation='relu'))

    m.add(Dropout(dropout))

    m.add(Dense(output_dim=50, activation='relu'))

    m.add(Dropout(dropout))

    m.add(Dense(output_dim=2, activation='softmax'))

    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return m



X, y = get_data(root_path + '/train.csv')

y_dummy = pandas.get_dummies(y).values



# Use grid search to optimise hyperparameters

# param_grid= dict(batch_size=[8, 16, 32], nb_epoch=[50, 70, 100], dropout=[0.0, 0.3, 0.5])



# model = KerasClassifier(build_fn=get_model, verbose=0)

# grid = GridSearchCV(estimator=model, param_grid=param_grid)

#grid.fit(X, y_dummy)



# print(grid.best_params_)

# print(grid.best_score_)



# {'dropout': 0.3, 'nb_epoch': 70, 'batch_size': 32}

# 0.823793490862
# classifier = grid.best_estimator_

# predictions = classifier.predict(X)

# survived = [int(round(p)) for p in predictions[:]]

# print('Training success: {}'.format(accuracy_score(survived,y)))



# train a classifier with the best parameters we got earlier

classifier = get_model(0.3)

classifier.fit(X, y_dummy, nb_epoch=70, batch_size=32, verbose=0)



predictions = classifier.predict(X)

survived = [int(round(p)) for p in predictions[:,1]]

print('Training set success: {}'.format(accuracy_score(survived,y)))
# get the predictions on the test set

test_df = pandas.read_csv(root_path + '/test.csv')

X_test = get_data_sets(test_df)

passenger_ids = test_df['PassengerId']



predictions = classifier.predict(X_test)

survived = [int(round(p)) for p in predictions[:, 1]]



submission = pandas.DataFrame({'PassengerId': passenger_ids, 'Survived': survived})

submission.to_csv("titanic_keras.csv", index=False)