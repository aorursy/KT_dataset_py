#Libraries
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import Imputer


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

X = train[["PassengerId", "Pclass", "Age", "Fare", "Embarked", "Sex"]]
y = train[["Survived"]]
X = pd.get_dummies(X)  # Treat nominal data

# Replace missing numerical attribute by their mean value
fill_NaN = Imputer()
imputed_DF = pd.DataFrame(fill_NaN.fit_transform(X))
imputed_DF.columns = X.columns
imputed_DF.index = X.index
X = imputed_DF

# Train model
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

# Format test set
X_test = test[["PassengerId", "Pclass", "Age", "Fare", "Embarked", "Sex"]]
X_test = pd.get_dummies(X_test)  # Treat nominal data
fill_NaN = Imputer()
imputed_DF = pd.DataFrame(fill_NaN.fit_transform(X_test))
imputed_DF.columns = X_test.columns
imputed_DF.index = X_test.index
X_test = imputed_DF

# Predict test data
predicted = clf.predict(X_test)

# Output predictions into csv
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predicted
    })
submission.to_csv('./submission.csv', index=False)
