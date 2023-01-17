import numpy as np

import pandas as pd



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import log_loss
df = pd.read_json(open("../input/train.json", "r"))
df.head()
def preprocess_features(_df):

    _df["num_photos"] = _df["photos"].apply(len)

    _df["num_features"] = _df["features"].apply(len)

    _df["num_description_words"] = _df["description"].apply(lambda x: len(x.split(" ")))

    created = pd.to_datetime(_df["created"])

    _df["created_year"] = created.dt.year

    _df["created_month"] = created.dt.month

    return _df
df = preprocess_features(df)
num_feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",

             "num_photos", "num_features", "num_description_words",

             "created_year", "created_month"]

X = df[num_feats]

y = df["interest_level"]

X.head()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)
clf = RandomForestClassifier(n_estimators=1100)

clf.fit(X_train, y_train)

y_val_pred = clf.predict_proba(X_val)

log_loss(y_val, y_val_pred)
test_df = pd.read_json(open("../input/test.json", "r"))

test_df = preprocess_features(test_df)

X = df[num_feats]



y = clf.predict_proba(X)

print(y.shape)
labels2idx = {label: i for i, label in enumerate(clf.classes_)}

labels2idx
sub = pd.DataFrame()

sub["listing_id"] = df["listing_id"]

for label in ["high", "medium", "low"]:

    sub[label] = y[:, labels2idx[label]]

sub.to_csv("submission_rf.csv", index=False)