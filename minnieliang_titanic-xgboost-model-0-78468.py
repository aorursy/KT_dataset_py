import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pandas as pd
from sklearn.model_selection import train_test_split
X_full = pd.read_csv('../input/titanic/train.csv')
X_test_full = pd.read_csv('../input/titanic/test.csv')
X_full
import matplotlib.pyplot as plt 
import seaborn as sns

plt.subplots(figsize=(8, 5))
sns.heatmap(X_full.corr(), annot=True)
plt.show()
X_full.isnull().sum()
X_full['Sex'] = X_full['Sex'].map({"male":1, "female":0})
X_test_full['Sex'] = X_test_full['Sex'].map({"male":1, "female":0})
X_full = X_full.drop(columns = "Cabin")
X_test = X_test_full.drop(columns = "Cabin")
X_full['TravelAlone']=np.where((X_full["SibSp"]+X_full["Parch"])>0, 0, 1)
X_full.drop('SibSp', axis=1, inplace=True)
X_full.drop('Parch', axis=1, inplace=True)
X_full
X_test['TravelAlone']=np.where((X_test["SibSp"]+X_test["Parch"])>0, 0, 1)
X_test.drop('SibSp', axis=1, inplace=True)
X_test.drop('Parch', axis=1, inplace=True)
X_test
object_cols = [col for col in X_full.columns if X_full[col].dtype == "object"]
object_nunique = list(map(lambda col: X_full[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])
object_cols = [col for col in X_test.columns if X_test[col].dtype == "object"]
object_nunique = list(map(lambda col: X_test[col].nunique(), object_cols))
d = dict(zip(object_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])
drop_X_full = X_full.drop(columns = ['Ticket', 'Name'])
drop_X_test = X_test.drop(columns = ['Ticket', 'Name'])
drop_X_full
drop_X_test
OH_X = pd.get_dummies(drop_X_full, columns=["Pclass","Embarked"])
OH_X
OH_X_test = pd.get_dummies(drop_X_test, columns = ['Pclass', 'Embarked'])
OH_X_test
null_data = OH_X_test[OH_X_test['Fare'].isnull()]
null_data
OH_X_test["Fare"].fillna(OH_X_test["Fare"].median(skipna=True), inplace=True)
OH_X_test["Age"].fillna(OH_X_test["Age"].median(skipna=True), inplace=True)
OH_X_test
OH_y = OH_X['Survived']
OH_X = OH_X.drop(columns = 'Survived')
OH_X
from sklearn.model_selection import train_test_split, cross_val_score
X_train, X_valid, y_train, y_valid = train_test_split(OH_X, OH_y,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=0)
X_train['Age'].fillna(X_train['Age'].median(skipna=True), inplace=True)
X_train.isnull().sum()
X_valid["Age"].fillna(X_valid["Age"].median(skipna=True), inplace=True)
X_valid.isnull().sum()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier

my_model = XGBClassifier(n_estimators=1000, learning_rate=0.01)
my_model.fit(X_train, y_train, 
             early_stopping_rounds=10, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
test_preds = my_model.predict(OH_X_test)
submission = pd.DataFrame({"PassengerId":OH_X_test['PassengerId'], "Survived":test_preds})
submission.to_csv('submission.csv', index=False)
submission