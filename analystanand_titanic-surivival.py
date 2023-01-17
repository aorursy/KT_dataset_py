import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve,cross_val_score
from sklearn.linear_model import Ridge
import numpy as np # linear algebra
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
train_df = pd.read_csv("../input/train.csv")
train_df.dtypes
print("Shape before dropping  rows containing null",train_df.shape)
train_df = train_df.fillna(train_df.mode().iloc[0])
print("Shape after dropping  rows containing null",train_df.shape)

train_label = train_df["Survived"]
train_df.drop(['Survived','Name','Ticket','Cabin'], axis = 1, inplace = True)
train_df.head()
train_df["Sex"] = train_df["Sex"].astype('category')
train_df["Sex"] = train_df["Sex"].cat.codes


train_df["Embarked"] = train_df["Embarked"].astype('category')
train_df["Embarked"] = train_df["Embarked"].cat.codes
train_df.head()
test_df = pd.read_csv("../input/test.csv")
test_df.dtypes
print("Shape before dropping  rows containing null",test_df.shape)
test_df = test_df.fillna(test_df.mode().iloc[0])
print("Shape after dropping  rows containing null",test_df.shape)
test_df.drop(['Name','Ticket','Cabin'], axis = 1, inplace = True)
test_df.head()
test_df["Sex"] = test_df["Sex"].astype('category')
test_df["Sex"] = test_df["Sex"].cat.codes

test_df["Embarked"] = test_df["Embarked"].astype('category')
test_df["Embarked"] = test_df["Embarked"].cat.codes
test_df.head()
logreg = LogisticRegression(C=1e5)
logreg.fit(train_df.as_matrix(),train_label)
print("mean accracy:",np.mean(cross_val_score(logreg,train_df.as_matrix(),train_label)))
predict =  pd.Series(logreg.predict(test_df.as_matrix()))
submission_df = pd.DataFrame({"PassengerId":list(test_df["PassengerId"]),"Survived":list(predict)})
submission_df.head
submission_df.to_csv("submission_3.csv",index = False)
# import os
print(os.listdir("../working/"))

