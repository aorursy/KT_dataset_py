import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

import os
print(os.listdir("../input"))

df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")

print(df_train.shape)
print(df_test.shape)
X = []
y = []
for row in tqdm(df_train.iterrows()) :
    label = row[1][0] # label (the number visible in the image)
    image = list(row[1][1:]) # image information as list, without label
    image = np.array(image) / 255
    X.append(image)
    y.append(label)

X = np.array(X)
y = np.array(y)
print(len(X))
print(len(y))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(len(X_train), len(y_train))
print(X_train[1].shape)
clf = RandomForestClassifier(max_depth=42, n_estimators=42, random_state=42, verbose=2)

clf.fit(X, y)

print(clf.feature_importances_)

y_pred = clf.predict(X_test)

print(y_pred[0:20], ".....")
print(y_test[0:20], ".....")
print(metrics.accuracy_score(y_test, y_pred))
X_new = []
for row in tqdm(df_test.iterrows()) :
    image = list(row[1])
    image = np.array(image) / 255
    X_new.append(image)
X_new = np.array(X_new)
print(len(X_new))
print(len(df_test))
y_new_pred = clf.predict(X_new)
print(y_new_pred)
df_sub = pd.DataFrame(list(range(1,len(X_new)+1)))
df_sub.columns = ["ImageID"]
df_sub["Label"] = y_new_pred
df_sub.to_csv("submission.csv", sep=",", header=True, index=False)
print(df_sub.head())