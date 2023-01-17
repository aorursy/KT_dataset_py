import pandas as pd
data=pd.read_csv("../input/dataset.csv")
data.head()
labels=data.activity
data_dropped=data.drop(["username","activity"],axis=1)
data_dropped.head()
data_dropped=data_dropped.set_index("date")
data_dropped.head()
features=data_dropped.values
features
data_time_dropped=data_dropped.drop(["time"],axis=1)
data_time_dropped.head()
features=data_time_dropped.values
features
LABELS=labels.values
FEATURES=features
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(FEATURES,LABELS,test_size=0.3,random_state=1)
from sklearn.ensemble import RandomForestClassifier
RanFor=RandomForestClassifier(n_estimators=100,random_state=1)
RanFor.fit(x_train,y_train)
from sklearn.metrics import *
accuracy_score(y_train,RanFor.predict(x_train))
accuracy_score(y_test,RanFor.predict(x_test))
precision_score(y_test,RanFor.predict(x_test))
recall_score(y_test,RanFor.predict(x_test))
f1_score(y_test,RanFor.predict(x_test))