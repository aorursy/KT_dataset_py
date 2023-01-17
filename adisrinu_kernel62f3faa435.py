# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier as DTC

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

path_train="/kaggle/input/data-science-bowl-2019/train.csv"

path_test="/kaggle/input/data-science-bowl-2019/test.csv"

path_lable="/kaggle/input/data-science-bowl-2019/train_labels.csv"

path_spec="/kaggle/input/data-science-bowl-2019/specs.csv"

data=pd.read_csv(path_train)

data.head(5)
data_lable=pd.read_csv(path_lable)

data_lable.info()

# column details

for col in data_lable.columns:

    print(col,data_lable[col].nunique())
data_lable.head(5)
mask2=data["event_code"].isin([4100,4110])

mask6=data["type"]=="Assessment"

mask7=data["event_data"].str.contains("correct")

train1=data[mask2&mask6&mask7]



train1.shape
n1=train1["event_code"]==4110

n2=train1["title"]=="Bird Measurer (Assessment)"

p1=train1["event_code"]==4100

p2=train1["title"]!="Bird Measurer (Assessment)"

train1=train1[n1&n2|p1&p2]
train2=train1.groupby(["installation_id","game_session","world","title"]).agg({"event_count":["min","mean","max"]}).astype("int")

train3=train1.groupby(["installation_id","game_session","world","title"]).agg({"game_time":["min","mean","max"]}).astype("int")

train2=train2[("event_count")].rename(columns={"min":"eventmin","mean":"eventmean","max":"eventmax"})

train3=train3[("game_time")].rename(columns={"min":"gamemin","mean":"gamemean","max":"gamemax"})

train4=train2.merge(train3,how="left",on=["installation_id","game_session","world","title"])

train4=train4.merge(data_lable,how="left",on=["installation_id","game_session","title"])

train5=train4.drop(axis=1,columns=["num_correct","num_incorrect","accuracy","title"])

train5
# train6=pd.get_dummies(train5,columns=["title"],drop_first=True)

train6=train5.copy()

train6
x=train6.drop(columns=["accuracy_group","installation_id","game_session"])

x=x.values

y=train6.loc[:,"accuracy_group"].values

y=y.reshape(-1,1)

x
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
classifier=DTC(criterion="entropy",random_state=0)

classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
cm=confusion_matrix(y_test,y_pred)

cm
accuracy=(cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3])/cm.sum()

accuracy
classifier=DTC(criterion="entropy",random_state=0,min_samples_leaf=15)

classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)



cm=confusion_matrix(y_test,y_pred)

accuracy=(cm[0,0]+cm[1,1]+cm[2,2]+cm[3,3])/cm.sum()

accuracy
data_test=pd.read_csv(path_test)

data_test.head(3)
# mask2=data_test["event_code"].isin([4100,4110])

# mask6=data_test["type"]=="Assessment"

# mask7=data_test["event_data"].str.contains("correct")

# data_test1=data_test[mask2&mask6&mask7]

# data_test1.shape
# n1=data_test1["event_code"]==4110

# n2=data_test1["title"]=="Bird Measurer (Assessment)"

# p1=data_test1["event_code"]==4100

# p2=data_test1["title"]!="Bird Measurer (Assessment)"

# test1=data_test1[n1&n2|p1&p2]



test1=data_test.copy()

test2=test1.groupby(["installation_id"]).agg({"event_count":["min","mean","max"]}).astype("int")

test3=test1.groupby(["installation_id"]).agg({"game_time":["min","mean","max"]}).astype("int")

test2=test2[("event_count")].rename(columns={"min":"eventmin","mean":"eventmean","max":"eventmax"})

test3=test3[("game_time")].rename(columns={"min":"gamemin","mean":"gamemean","max":"gamemax"})



test4=test2.merge(test3,how="left",on=["installation_id"])

test4=test4.merge(data_lable,how="left",on=["installation_id"])

test5=test4.drop(axis=1,columns=["num_correct","num_incorrect","accuracy","accuracy_group","installation_id","game_session","title"])

test5
# test5=pd.get_dummies(test5,columns=["title"],drop_first=True)

# test5
test6=np.array(test5)

y_prediction=classifier.predict(test6)

y_pred1=pd.DataFrame(y_prediction)

type(y_pred1)

submission=test4[["installation_id","title"]]

submission["y_predicted"]=y_pred1.iloc[:,0]
submission1=submission.copy()

del submission1["title"]

submission1.rename(columns={"y_predicted":"accuracy_group"},inplace=True)

# submission.to_csv("temp.csv",index=False)

submission1.to_csv("submission.csv",index=False)

# submission1.head(20)