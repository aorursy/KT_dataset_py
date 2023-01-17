import pandas as pd 

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv('../input/sf-crime/train.csv.zip', compression='zip', header=0, sep=',', quotechar='"')
train.head()
train.info()
sns.heatmap(train.isnull(),yticklabels=0)
from sklearn.preprocessing import LabelEncoder

l = LabelEncoder()
train["DayOfWeek"] = l.fit_transform(train["DayOfWeek"])

train["PdDistrict"] = l.fit_transform(train["PdDistrict"])

train["Address"] = l.fit_transform(train["Address"])
train.info()
train["Category"].value_counts()
test = pd.read_csv("../input/sf-crime/test.csv.zip")
test.info()
sns.heatmap(test.isnull(),yticklabels=0)
test["DayOfWeek"] = l.fit_transform(test["DayOfWeek"])

test["PdDistrict"] = l.fit_transform(test["PdDistrict"])

test["Address"] = l.fit_transform(test["Address"])
test.info()
test.columns
x_train = train[['DayOfWeek', 'PdDistrict', 'Address', 'X', 'Y']]
y_train = train["Category"]
x_test = test[['DayOfWeek', 'PdDistrict', 'Address', 'X', 'Y']]
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
classes = np.unique(y_train)

"Number of unique classes: {0}".format(len(classes))
result = pd.DataFrame(index= test.Id, columns=model.classes_, data=model.predict_proba(x_test))

#result = pd.DataFrame(result)

result.to_csv("submission.csv", index_label='Id')
s = pd.read_csv("../input/sf-crime/sampleSubmission.csv.zip")
s.head()
result.head()