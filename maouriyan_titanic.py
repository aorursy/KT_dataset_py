import pandas as pd
test = pd.read_csv("../input/test.csv")
train = pd.read_csv("../input/train.csv")
print(test.shape)
train.head(10)

pt = train.pivot_table(index="Sex",values="Survived")

pt.plot.bar()
from matplotlib import pyplot as plt
Pclass_Pivot = train.pivot_table(index="Pclass",values="Survived")
Pclass_Pivot
Pclass_Pivot.plot.bar()
Pclass_Pivot
survived = train[train["Survived"]==1]
survived.head(10)
died = train[train["Survived"]==0]
survived["Age"].plot.hist(alpha=0.5,color="red",bins=50)
died["Age"].plot.hist(alpha=0.5,color="blue",bins=50)
plt.legend(['Survived','Died'])
plt.show()
def process_age(df,cut_points,label_names):
  df["Age"] = df["Age"].fillna(-0.5)
  df["Age_Categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
  return df
cut_points=[-1,0,5,12,18,35,60,100]
label_names=["Missing","Infant","Child","Teenager","YoungAdult","Adult","Senior"]
train = process_age(train,cut_points,label_names)
test = process_age(test,cut_points,label_names)
age_cat_pivot = train.pivot_table(index="Age_Categories",values="Survived")
age_cat_pivot.plot.bar()
plt.show()
train["Pclass"].value_counts()
def create_dummies(df,column_name):
  dummies = pd.get_dummies(df[column_name],prefix=column_name)
  df = pd.concat([df,dummies],axis=1)
  return df

column_names = ["Pclass","Age_Categories","Sex"]
for x in column_names:
  train=create_dummies(train,x)
  test=create_dummies(test,x)

test.head(10)  
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

columns = ['Pclass_1','Pclass_2','Pclass_3','Sex_female','Sex_male','Age_Categories_Missing','Age_Categories_Infant','Age_Categories_Child', 
           'Age_Categories_Teenager',
       'Age_Categories_YoungAdult', 'Age_Categories_Adult',
       'Age_Categories_Senior']
lr.fit(train[columns],train["Survived"])
holdout = test

from sklearn.model_selection import train_test_split
all_x = train[columns]
all_y = train["Survived"]

train_x, test_x,train_y,test_y = train_test_split(all_x,all_y,test_size=0.2,random_state=0)
test_x.shape

train_x.head(10)
from sklearn.metrics import accuracy_score
lr.fit(train_x,train_y)
predictions = lr.predict(test_x)
acc_sc = accuracy_score(test_y,predictions)
acc_sc
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(test_y, predictions)
pd.DataFrame(conf_matrix, columns=['Survived', 'Died'], index=[['Survived', 'Died']])
from sklearn.model_selection import cross_val_score
import numpy as np
scores = cross_val_score(lr,all_x,all_y,cv=5)
np.mean(scores)
lr.fit(all_x,all_y)
holdout_predictions=lr.predict(holdout[columns])
holdout_predictions
holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId":holdout_ids,"Survived":holdout_predictions}
submission = pd.DataFrame(submission_df)
submission
submission.to_csv("titanic_submission.csv",index=False)