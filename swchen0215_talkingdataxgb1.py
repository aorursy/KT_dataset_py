
import pandas as pd

import numpy as np

import xgboost as xgb

from scipy import sparse

from sklearn.feature_extraction import FeatureHasher

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale

from sklearn.decomposition import TruncatedSVD, SparsePCA

from sklearn.cross_validation import train_test_split, cross_val_score

from sklearn.feature_selection import SelectPercentile, f_classif, chi2

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import log_loss

##################

#   App Events

##################

print("# Read App Events")

app_ev = pd.read_csv("../input/app_events.csv", dtype={'device_id': np.str})

# remove duplicates(app_id)

app_ev = app_ev.groupby("event_id")["app_id"].apply(

    lambda x: " ".join(set("app_id:" + str(s) for s in x)))
##################

#     Events

##################

print("# Read Events")

events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})

events["app_id"] = events["event_id"].map(app_ev)



events = events.dropna()



del app_ev



events = events[["device_id", "app_id"]]



# remove duplicates(app_id)

events = events.groupby("device_id")["app_id"].apply(

    lambda x: " ".join(set(str(" ".join(str(s) for s in x)).split(" "))))

events = events.reset_index(name="app_id")



# expand to multiple rows

events = pd.concat([pd.Series(row['device_id'], row['app_id'].split(' '))

                    for _, row in events.iterrows()]).reset_index()

events.columns = ['app_id', 'device_id']


























