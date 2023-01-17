import numpy as np

import pandas as pd

import matplotlib as plt
data_train = pd.read_csv("Train_data.csv")

train = data_train

data_test = pd.read_csv("Test_data.csv")

test = data_test
train.head()
train = train.drop(['FileName'],axis=1)
train.head()
test.drop(['FileName'],axis=1)
test_sub = test.drop(['FileName'],axis=1)
y=train['Class']

X=train.drop(['Class'],axis=1)

y

from sklearn.naive_bayes import GaussianNB as NB

nb = NB()

nb.fit(X,y)
test_sub = test_sub.dropna(axis=1)
test_sub.isna()
pred = nb.predict(test_sub)
pred
res1 = pd.DataFrame(pred)

result = pd.concat([test['FileName'], res1], axis=1).reindex()

result = result.rename(columns={0: "Class"})

result
result.to_csv("Sub1.csv",index=False)
from sklearn.neighbors import KNeighborsClassifier

train_acc = []

 

knn = KNeighborsClassifier(n_neighbors=12) #k=12 is a randomly chosen number

knn.fit(X,y)
pred1 = knn.predict(test_sub)
pred1
res2 = pd.DataFrame(pred1)

result2 = pd.concat([test['FileName'], res2], axis=1).reindex()

result2 = result2.rename(columns={0: "Class"})

result2
result2.to_csv("Sub2.csv",index=False)
from sklearn.tree import DecisionTreeClassifier

train_acc = []

dTree = DecisionTreeClassifier(max_depth=8) #again, max_depth=8 is a randomly chosen number

dTree.fit(X,y)
dt_pred = dTree.predict(test_sub)
dt_pred
res3 = pd.DataFrame(dt_pred)

result3 = pd.concat([test['FileName'], res3], axis=1).reindex()

result3 = result3.rename(columns={0: "Class"})

result3
result3.to_csv("Sub3.csv",index=False)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=19, random_state = 42) #n_estimators is random

rf.fit(X, y)
rf_pred = rf.predict(test_sub)
rf_pred
res4 = pd.DataFrame(rf_pred)

result4 = pd.concat([test['FileName'], res4], axis=1).reindex()

result4 = result4.rename(columns={0: "Class"})

result4
result4.to_csv("Sub8.csv",index=False)
from sklearn.ensemble import AdaBoostClassifier

abc = AdaBoostClassifier(n_estimators=63, learning_rate=1)

abc.fit(X,y)
ad_pred = abc.predict(test_sub)
ad_pred
res5 = pd.DataFrame(ad_pred)

result5 = pd.concat([test['FileName'], res5], axis=1).reindex()

result5 = result5.rename(columns={0: "Class"})

result5
result5.to_csv("Submit6.csv",index=False)
from IPython.display import HTML

import numpy as np

import base64

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html='<a download="{filename}"href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)

    create_download_link("Submission.csv")