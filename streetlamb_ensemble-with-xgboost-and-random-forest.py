# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test_set_full_raw = pd.read_csv("../input/sf-crime/test.csv.zip", compression="zip", index_col='Id')

train_set_full_raw = pd.read_csv("../input/sf-crime/train.csv.zip", compression="zip")
train_set_full_raw.head()
#Check what columns are present in the test set.

train_set_full_raw.columns
train_set_full_raw.info()
train_set_full_raw.describe()
ts = train_set_full_raw.copy()

ts = ts[ts["Y"]<90]
ts.plot(x="X", y="Y", kind="scatter", alpha=0.01,figsize=(15,12))
ts["Category"].value_counts()
print("Most common resolutions for each category in percentage\n")

for i in ts.groupby(["Category"])["Resolution"]:

  print('\033[95m'+i[0]+'\033[0m')

  print(round(i[1].value_counts()[:3]/i[1].count()*100,1))

  print()
from datetime import datetime



ts["Hour"] = ts.Dates.apply(lambda date_string: date_string[11:-6])

ts.head()
ts.groupby(["DayOfWeek"])["Hour"].value_counts()
import numpy as np



train_full = ts.copy()

X_train_full, y_train_full = np.array(train_full[["DayOfWeek", "PdDistrict", "X", "Y", "Hour"]]), np.array(train_full[["Category"]])

y_train_full = y_train_full.ravel()

X_test = test_set_full_raw.copy()

X_test["Hour"] = X_test.Dates.apply(lambda date_string: date_string[11:-6])

X_test = X_test.drop(columns=["Dates", "Address"])

X_test = np.array(X_test)
import sklearn

from sklearn.model_selection import StratifiedShuffleSplit



sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)



for train_index, test_index in sss.split(X_train_full, y_train_full):

  X_train, y_train = X_train_full[train_index], y_train_full[train_index]

  X_val, y_val = X_train_full[test_index], y_train_full[test_index]
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer



num_attribs = [2,3]

cat_attribs = [0,1,4]



num_pipeline = Pipeline([

                         ('std_scaler', StandardScaler())

])



full_pipeline = ColumnTransformer([

                                 ('num', num_pipeline, num_attribs),

                                 ('cat', OneHotEncoder(), cat_attribs) 

])



X_train_prepared = full_pipeline.fit_transform(X_train)

X_val_prepared = full_pipeline.transform(X_val)

X_test_prepared = full_pipeline.transform(X_test)
from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier,VotingClassifier

import xgboost
X_train_prepared.shape
ss = StratifiedShuffleSplit(n_splits=1, train_size=100_000, random_state=42)

for train_index, _ in ss.split(X_train_prepared, y_train):

  X_train_prepared_small, y_train_small = X_train_prepared[train_index], y_train[train_index].ravel()



X_train_prepared_small.shape, y_train_small.shape
rf_clf = RandomForestClassifier(max_depth=16, random_state=42, n_jobs=-1, verbose=3)

xg_clf = xgboost.XGBClassifier()



estimators = [

            ("rf", rf_clf),

            ("xg", xg_clf)

]



voting_clf = VotingClassifier(estimators, n_jobs=-1, voting="soft")

voting_clf.fit(X_train_prepared_small, y_train_small)

voting_clf.score(X_val_prepared, y_val)
y_pred = voting_clf.predict_proba(X_test_prepared)

pred_df = pd.DataFrame(y_pred, columns=[voting_clf.classes_])

pred_df["Id"]= list(range(pred_df.shape[0]))

pred_df.to_csv("crime_pred_02.zip", compression="zip", index=False)

pred_df.head()