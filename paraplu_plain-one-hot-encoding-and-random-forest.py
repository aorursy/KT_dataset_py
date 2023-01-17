# simple implementation using straight forward one-hot encoding and random forest as classifier. No cleaning of indigridients done yet

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# parse the json file
import json
with open("../input/train.json", "r") as f:
    inp = json.loads(f.read())
with open("../input/test.json", "r") as f:
    test = json.loads(f.read())
# extract all labels
alphabet = []
classes = []
for row in inp:
    alphabet.append(row["ingredients"])
    classes.append(row["cuisine"])
    

# We encode the indigrents using one hot encoding
from sklearn import preprocessing

oe = preprocessing.MultiLabelBinarizer()
oencoded = oe.fit_transform(alphabet)
    
# Train random forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(oencoded, classes)
# predict test data
test_data = []
for row in test:
    test_data.append(row["ingredients"])
transformed_test_data = oe.transform(test_data)
result = clf.predict(transformed_test_data)
# write result
with open("oedt_submission.csv", "w") as f:
    f.write("id,cuisine\n")
    for i in range(len(result)):
        f.write(str(test[i]["id"]) + "," + str(result[i]) + "\n")