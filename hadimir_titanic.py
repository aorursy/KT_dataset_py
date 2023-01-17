import pandas as pd
from sklearn.tree  import  DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import warnings


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



tranning_data = pd.read_csv("../titanic/train.csv")
tranning_data.head()


features = ["Pclass", "Sex", "SibSp", "Parch"]
X_train = pd.get_dummies(tranning_data[features])
X_train.head(5)
X_train
y_train = tranning_data["Survived"]
y_train.head(2)
test_data = pd.read_csv("../titanic/test.csv")
features = ["Pclass", "Sex", "SibSp", "Parch"] 
X_test = pd.get_dummies(test_data[features])
X_test.head(5)  

random_forest_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
random_forest_model.fit(X_train, y_train)
RF_predictions = random_forest_model.predict(X_test)
RF_predictions
# SAVING OUTPUT TO CSV
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': RF_predictions})
output.to_csv('/kaggle/working/my_submission.csv', index=False)
