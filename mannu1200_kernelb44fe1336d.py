import pandas as pd

from statistics import mean 

import numpy as np
df = pd.read_csv("./train.csv")
df.head()
print("Class distribution after fetching unique orders")

df[["order_id","label"]].drop_duplicates().reset_index().groupby(["label"]).size().to_frame(name="count").reset_index()
print("Average accuracy for label 0:", mean(df.groupby("label")["accuracy_in_meters"].apply(list)[0]))

print("Average accuracy for label 1:", mean(df.groupby("label")["accuracy_in_meters"].apply(list)[1]))
agg_df = df.groupby('order_id').agg(pd.Series.tolist)
agg_df.head()
for index,row in agg_df.iterrows():

    for c in ["service_type", "date", "hour","label"]:

        agg_df.loc[index, c]=list(set(row[c]))
agg_df.head()
from math import atan2, cos, sin, pi, sqrt, radians

def haversine_distance(lat1, lon1,lat2, lon2):

    # earth radius in kilometers

    R = 6373.0

    # convert decimal degrees to radians

    lon1=radians(lon1)

    lon2=radians(lon2)

    lat1=radians(lat1)

    lat2=radians(lat2)

    # distance between pairs

    dlon = lon2 - lon1

    dlat = lat2 - lat1

    a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2

    c = 2 * atan2(sqrt(a), sqrt(1-a))

    # return distance in meters

    distance = R * c * 1000

    return distance
#agg_df["speeds"] =  [[]] * agg_df.shape[0]

ss = []

for index, row in agg_df.iterrows():

    lats, lngs, secs = row["latitude"], row["longitude"], row["seconds"]

    speeds = []

    for i in range(len(lats)-1):

        distance = haversine_distance(lats[i], lngs[i], lats[i+1], lngs[i+1])

        if secs[i+1] - secs[i] != 0:

            speed = distance/(secs[i+1] - secs[i])

            speeds.append(speed)

    ss.append(speeds)



agg_df["speeds"] = ss
counter=0

for index,row in agg_df.iterrows():

    try:

        max_acc = max(row["accuracy_in_meters"])

        min_acc = min(row["accuracy_in_meters"])

        max_alt = max(row["altitude_in_meters"])

        min_alt = min(row["altitude_in_meters"])

        mean_alt = mean(row["altitude_in_meters"])

        mean_acc = mean(row["accuracy_in_meters"])



        max_speed = max(row["speeds"])

        min_speed = min(row["speeds"])

        mean_speed = mean(row["speeds"])

        

        agg_df.loc[index, "max_alt"] = max_alt

        agg_df.loc[index, "min_alt"] = min_alt

        agg_df.loc[index, "diff_max_min_alt"] = max_alt - min_alt

        agg_df.loc[index, "max_acc"] = max_acc

        agg_df.loc[index, "min_acc"] = min_acc

        agg_df.loc[index, "diff_max_min_acc"] = max_acc - min_acc

        agg_df.loc[index, "mean_alt"] = mean_alt

        agg_df.loc[index, "mean_acc"] = mean_acc

        

        agg_df.loc[index, "max_speed"] = max_speed

        agg_df.loc[index, "min_speed"] = min_speed

        agg_df.loc[index, "diff_max_min_speed"] = max_speed - min_speed

        agg_df.loc[index, "mean_speed"] = mean_speed

    except:

        pass
agg_df.head()
agg_df.columns
_X,_Y = agg_df[["service_type", "max_alt", "min_alt", "diff_max_min_alt", "max_acc", "min_acc", "diff_max_min_acc", "mean_alt", "mean_acc", 'max_speed','min_speed', 'diff_max_min_speed', 'mean_speed']], agg_df["label"]

_X = _X.fillna(-1)
X = []

for x in _X.values:

    if x[0][0] == "GO_FOOD":

        x[0] = 1

    else:

        x[0] = 2

    X.append(x)
Y = []

for y in _Y:

    Y.append(y[0])
__X = []

for x in X:

    __X.append(x.tolist())

X = __X
from xgboost import XGBClassifier

from sklearn.multiclass import OneVsRestClassifier

import xgboost as xgb

import sklearn

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression, RidgeClassifier



clf_multilabel = XGBClassifier(max_depth=30)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.33, random_state=42)
LR_classifier = LogisticRegression()

accuracy_score(Y_test, LR_classifier.fit(X_train, Y_train).predict(X_test))
clf_multilabel.fit(np.matrix(X_train), Y_train)
y_val_predicted_labels_tfidf = clf_multilabel.predict(np.matrix(X_test))

accuracy_score(Y_test, y_val_predicted_labels_tfidf)
import numpy as np

import matplotlib.pyplot as plt

import sklearn.datasets

from sklearn.model_selection import train_test_split

%matplotlib inline



from keras import Sequential

from keras.layers import Dense

import matplotlib.pyplot as plt 

from keras.utils import np_utils
np.matrix(X_train).shape
classifier = Sequential()

classifier.add(Dense(64, activation='relu', kernel_initializer='random_normal', input_dim=np.matrix(X_train).shape[1]))

classifier.add(Dense(32, activation='tanh'))

classifier.add(Dense(16, activation='relu'))

classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

classifier.fit(np.matrix(X_train),Y_train, batch_size=10, epochs=50)
eval_model=classifier.evaluate(np.matrix(X_test), Y_test)

eval_model
classifier.predict(np.matrix(X_test)) < .5
test_df = pd.read_csv("./test.csv")
test_df.head(2)
agg_df_test = test_df.groupby('order_id').agg(pd.Series.tolist)
for index,row in agg_df_test.iterrows():

    for c in ["service_type", "date", "hour"]:

        agg_df_test.loc[index, c]=list(set(row[c]))
#agg_df["speeds"] =  [[]] * agg_df.shape[0]

ss = []

for index, row in agg_df_test.iterrows():

    lats, lngs, secs = row["latitude"], row["longitude"], row["seconds"]

    speeds = []

    for i in range(len(lats)-1):

        distance = haversine_distance(lats[i], lngs[i], lats[i+1], lngs[i+1])

        if secs[i+1] - secs[i] != 0:

            speed = distance/(secs[i+1] - secs[i])

            speeds.append(speed)

    ss.append(speeds)



agg_df_test["speeds"] = ss
counter=0

for index,row in agg_df_test.iterrows():

    try:

        max_acc = max(row["accuracy_in_meters"])

        min_acc = min(row["accuracy_in_meters"])

        max_alt = max(row["altitude_in_meters"])

        min_alt = min(row["altitude_in_meters"])

        mean_alt = mean(row["altitude_in_meters"])

        mean_acc = mean(row["accuracy_in_meters"])



        max_speed = max(row["speeds"])

        min_speed = min(row["speeds"])

        mean_speed = mean(row["speeds"])

        

        agg_df_test.loc[index, "max_alt"] = max_alt

        agg_df_test.loc[index, "min_alt"] = min_alt

        agg_df_test.loc[index, "diff_max_min_alt"] = max_alt - min_alt

        agg_df_test.loc[index, "max_acc"] = max_acc

        agg_df_test.loc[index, "min_acc"] = min_acc

        agg_df_test.loc[index, "diff_max_min_acc"] = max_acc - min_acc

        agg_df_test.loc[index, "mean_alt"] = mean_alt

        agg_df_test.loc[index, "mean_acc"] = mean_acc

        

        agg_df_test.loc[index, "max_speed"] = max_speed

        agg_df_test.loc[index, "min_speed"] = min_speed

        agg_df_test.loc[index, "diff_max_min_speed"] = max_speed - min_speed

        agg_df_test.loc[index, "mean_speed"] = mean_speed

    except:

        pass
_X = agg_df_test[["service_type", "max_alt", "min_alt", "diff_max_min_alt", "max_acc", "min_acc", "diff_max_min_acc", "mean_alt", "mean_acc", 'max_speed','min_speed', 'diff_max_min_speed', 'mean_speed']]

_X = _X.fillna(-1)
X = []

for x in _X.values:

    if x[0][0] == "GO_FOOD":

        x[0] = 1

    else:

        x[0] = 2

    X.append(x)
__X = []

for x in X:

    __X.append(x.tolist())

X = __X
y_preds = clf_multilabel.predict(np.matrix(X))
y_preds.shape,agg_df_test.shape
results = []

counter=0

for index in agg_df_test.index:

    results.append([index,y_preds[counter]])

    counter=counter+1
import csv



with open("output.csv", "w") as f:

    writer = csv.writer(f)

    writer.writerows(results)
print(clf_multilabel.feature_importances_)
_X.columns
feat_imp = {}

i=0

for val in clf_multilabel.feature_importances_:

    feat_imp[_X.columns[i]] = val

    i=i+1
feat_imp