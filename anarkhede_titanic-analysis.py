import pandas as pd



# Step 1-3

titanic_train = pd.read_csv('../input/train.csv')

# Dropped the following variuables because the information is either redundant or captured

# by the variables we have kept. 

titanic_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

titanic_train = titanic_train.dropna(axis=0)

titanic_train = titanic_train.replace(to_replace=['male', 'female'], value=[1, 0])
# Step 4:

# Features:

X_train = titanic_train.ix[:, 1:titanic_train.shape[1]]

# Target:

y_train = titanic_train.ix[:, 0]
# Step 5:

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=3, random_state=0)

forest.fit(X_train, y_train)

print("Random Forest Train Score: ", forest.score(X_train, y_train))
# Step 6:

from sklearn.model_selection import cross_val_score

scores = cross_val_score(forest, X_train, y_train, cv=10)

print("Cross-validation scores: {}".format(scores))

print("Average cross-va;idation scores: {}".format(scores.mean()))
# Step 7:

titanic_test = pd.read_csv('../input/test.csv')

titanic_test.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

titanic_test = titanic_test.dropna(axis=0)

pas_id = titanic_test.ix[:,0]

titanic_test = titanic_test.replace(to_replace=['male', 'female'], value=[1, 0])



X_test = titanic_test.ix[:, 1:titanic_test.shape[1]]
# Step 8:

predictions = forest.predict(X_test)

pred_df = pd.DataFrame.from_records({'Predictions': predictions, 'PasID': pas_id})

pred_df.head(10)

#pred_df.to_csv('../input/titanic_predictions.csv')