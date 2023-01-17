import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
X = train_data[["Pclass", "Sex", "SibSp", "Parch", "Fare"]].replace({'male': 1, 'female': 0})

y = train_data[["Survived"]]

X_target = test_data[["Pclass", "Sex", "SibSp", "Parch", "Fare"]].replace({'male': 1, 'female': 0})
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
clf = SVC()

clf.fit(X_train, y_train) 
predicted = clf.predict(X_test)
print(metrics.classification_report(y_test, predicted))
### Saving output
X_target_non_nan = np.nan_to_num(X_target)  # Simply replacing nans with 0s

y_target = clf.predict(X_target_non_nan)
test_data["Survived"] = y_target

output = test_data[["PassengerId", "Survived"]]
output.to_csv("output.csv", index=False)