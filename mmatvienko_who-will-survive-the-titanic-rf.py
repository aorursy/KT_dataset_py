# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import model_selection

from sklearn.ensemble import RandomForestClassifier

from sklearn import tree

from sklearn import pipeline as pipeline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/titanic/train.csv')
data.head()
na_count = sum(data["Cabin"].isna())

print(f'Number of NaNs in the Cabin column: {na_count}')
del data["Name"]        # definitely no predictive power

del data["Cabin"]       # too many NaN

del data["Ticket"]      # too many unique values

del data["PassengerId"] # too many unique values
data.head()
print(set(data["Embarked"]))

na_count = sum(data["Embarked"].isna())

print(f'Number of NaNs in the Embarked column: {na_count}')
sex_converter = {'male': 0, 'female': 1}

embarked_converter = {'C': 0, np.nan: -1, 'S': 1, 'Q':2}

l = data.shape[0]



for i in range(l):

    data.loc[i, "Embarked"] = embarked_converter[data["Embarked"][i]]

    data.loc[i ,"Sex"] = sex_converter[data["Sex"][i]]
data = data.dropna()



labels = data["Survived"]

del data["Survived"]
data.head()
param_grid = {

    'rf__max_depth': range(2, 10),

    'rf__min_samples_leaf': range(1, 20, 2)

}



rf_pipeline = pipeline.Pipeline(memory=None, steps=[('rf', RandomForestClassifier())])

rf_gs = model_selection.GridSearchCV(rf_pipeline, param_grid, scoring='accuracy', cv=5)

rf_cv_results = model_selection.cross_val_score(rf_gs, data, y=labels, cv=5)



rf_gs.fit(data, labels)

best_rf = rf_gs.best_estimator_
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
l = test_data.shape[0]

for i in range(l):

    test_data.loc[i, "Embarked"] = embarked_converter[test_data["Embarked"][i]]

    test_data.loc[i ,"Sex"] = sex_converter[test_data["Sex"][i]]
del test_data["Name"]

del test_data["Cabin"]

del test_data["Ticket"]

pid = test_data["PassengerId"]

del test_data["PassengerId"]
test_data = test_data.fillna(test_data.mean())
y_pred = best_rf.predict(test_data)
submission_data = pd.DataFrame({"PassengerId": pid, "Survived": y_pred})
submission_data.to_csv('csv_to_submit.csv', index = False)
res = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
from sklearn.metrics import f1_score

score = f1_score(y_pred, res["Survived"])

print(f'Random Forest has a f1 score of: {score:.3f}')

print("Best Random Forest parameters:  {}".format(rf_gs.best_params_))
