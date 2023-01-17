# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.head()
test.head()
train['int_sex'] = train['Sex'].apply(lambda x:1 if x =="male" else 0)

test['int_sex'] = test['Sex'].apply(lambda x:1 if x =="male" else 0)

train['has_cabin'] = train['Cabin'].apply(lambda x:0 if type(x) == float else 1)

test['has_cabin'] = test['Cabin'].apply(lambda x:0 if type(x) == float else 1)


train['embarked_c'] = train['Embarked'].apply(lambda x:1 if x == "C" else 0)

train['embarked_q'] = train['Embarked'].apply(lambda x:1 if x == "Q" else 0)

train['embarked_s'] = train['Embarked'].apply(lambda x:1 if x == "S" else 0)
train.head()
mean_age = train["Age"].mean() 

train['Age'] = train['Age'].fillna(mean_age)

clean_train = train[["Pclass","Age","SibSp","Parch","Fare", "int_sex","has_cabin","embarked_c","embarked_q", "embarked_s"]]

clean_train.tail()
label_clean_train = train[["Survived"]]

label_clean_train.head()
len(clean_train[clean_train.isnull().values==True].index)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty='l2')

model.fit(clean_train, label_clean_train) 

model
test['embarked_c'] = test['Embarked'].apply(lambda x:1 if x == "C" else 0)

test['embarked_q'] = test['Embarked'].apply(lambda x:1 if x == "Q" else 0)

test['embarked_s'] = test['Embarked'].apply(lambda x:1 if x == "S" else 0)

#mean_age = test["Age"].mean() 

test['Age'] = test['Age'].fillna(mean_age)

test['Fare'] = test['Fare'].fillna(test["Fare"].mean())

test_x = test[["Pclass","Age","SibSp","Parch","Fare", "int_sex","has_cabin","embarked_c","embarked_q", "embarked_s"]]

test_x.tail()
train_predict = model.predict(clean_train)

test_predict = model.predict(test_x)
res = pd.DataFrame()

res["train_y"] = label_clean_train["Survived"]

res["train_predict"] = train_predict

res.tail()
# res["train_y"][886]
tp = 0

tn = 0

fp = 0

fn = 0

for i in range(len(res.index)):

#     print(res["train_y"][i] == 1)

#     break

    if res["train_y"][i] == 1 and res["train_predict"][i] == 1:

        tp += 1

    elif res["train_y"][i] == 0 and res["train_predict"][i] == 0:

        tn += 1

    elif res["train_y"][i] == 1 and res["train_predict"][i] == 0:

        fp += 1

    elif res["train_y"][i] == 0 and res["train_predict"][i] == 1:

        fn += 1

        
#准确率

peci = tp/(tn+tp)

#召回率

recall = tp/(fp+tp)

#精确率

accuracy = (tp + tn)/(len(res.index))

print("%f %f %f"%(peci,recall,accuracy))
out_df = pd.DataFrame()

out_df["PassengerId"] = test["PassengerId"]

out_df["Survived"] = test_predict

out_df.to_csv("submission_lr_v1.csv",index=False)