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
df = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
#v= df[df['Survived']==1]
#list(v['Ticket'])
#def flagPC(string):
    #if 'PC' in string:
        #return 1
    #else:
        #return 0
#df["PC"] = df["Ticket"].apply(flagPC)
#df.groupby("Survived")["PC"].hist()
#def is_num_ticket(string):
#    try:
#        int(string)
#        return 1
#    except:
#        return 0
#df["is_num_ticket"] = df["Ticket"].apply(is_num_ticket)
#df.groupby("Survived")["is_num_ticket"].hist()
df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}) 
df_test['Sex'] = df_test['Sex'].map({'female': 1, 'male': 0}) 
df
df["Embarked"].unique()
df["S"] = df["Embarked"].map({'S':1,'C':0,'Q':0})
df["C"] = df["Embarked"].map({'S':0,'C':1,'Q':0})
df["Q"] = df["Embarked"].map({'S':0,'C':0,'Q':1})
df_test["S"] = df_test["Embarked"].map({'S':1,'C':0,'Q':0})
df_test["C"] = df_test["Embarked"].map({'S':0,'C':1,'Q':0})
df_test["Q"] = df_test["Embarked"].map({'S':0,'C':0,'Q':1})
df.pop("Embarked")
df_test.pop("Embarked")
count = 0
for item in df["Name"]:
    if "Mr." in item.split():
        print(False)
        df.loc[count, "Name"] = 0
    else: 
        if "Mrs." in item.split(): 
            print(False)
            df.loc[count, "Name"] = 0
        else: 
            if "Miss." in item.split():
                print(False)
                df.loc[count, "Name"] = 0
            else:
                print(True)
                df.loc[count, "Name"] = 1
    count=count+1

count1 = 0
for item in df_test["Name"]:
    if "Mr." in item.split():
        print(False)
        df_test.loc[count1, "Name"] = 0
    else: 
        if "Mrs." in item.split(): 
            print(False)
            df_test.loc[count1, "Name"] = 0
        else: 
            if "Miss." in item.split():
                print(False)
                df_test.loc[count1, "Name"] = 0
            else:
                print(True)
                df_test.loc[count1, "Name"] = 1
    count1=count1+1
df["Age"] = df["Age"].fillna(df["Age"].mean())
df_test["Age"] = df_test["Age"].fillna(df_test["Age"].mean())
df["Ticket"] = df["Ticket"].str.len()
df_test["Ticket"] = df_test["Ticket"].str.len()
df.pop("PassengerId")
df_test_pas = df_test["PassengerId"]
df_test.pop("PassengerId")
df["Survived1"]=df["Survived"]
#df_test["Survived1"]=df_test["Survived"]
df.pop("Survived")
#df_test.pop("Survived")
df
df["Cabin1"] = df["Cabin"].isnull()
df_test["Cabin1"] = df_test["Cabin"].isnull()
df["Cabin"] = df["Cabin1"].map({True: 0, False: 1})
df.pop("Cabin1")
df_test["Cabin"] = df_test["Cabin1"].map({True: 0, False: 1})
df_test.pop("Cabin1")
#df_test.pop("Cabin1")
df
df_test
df.pop("SibSp")
df.pop("Parch")
df_test.pop("SibSp")
df_test.pop("Parch")
df.pop("Fare")
df_test.pop("Fare")
df
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived1', 1), df['Survived1'], test_size = .2, random_state=10) 
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
best = 0 
average = 0
total_for_average = 0
model1 = xgb.XGBClassifier(max_depth=2, n_estimators=200, learning_rate=0.02)
model1.fit(df.drop('Survived1', 1), df['Survived1'])
y_pred = model1.predict(X_test)
print(accuracy_score(y_test, y_pred))
total_for_average += 1
average += accuracy_score(y_test, y_pred)
if (accuracy_score(y_test, y_pred) > best): 
    best = accuracy_score(y_test, y_pred)
print("\nThe Best is", best)
print("The Average is", average/total_for_average)
y_pred_l = model1.predict(df_test)
df_test
y_pred_l
'''y = df['Survived1']
x = df[df.columns[:-1]]
best = 0 
average = 0
total_for_average = 0

models=[]
for i in range(50):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)
    model = xgb.XGBClassifier(max_depth=6, n_estimators=77, learning_rate=0.11)
    model.fit(x_train, y_train)
    models.append(model)
    y_pred = model.predict(x_test)


print(accuracy_score(y_test, y_pred))
total_for_average += 1
average += accuracy_score(y_test, y_pred)'''
df
df_test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('Survived1', 1), df['Survived1'], test_size = .2, random_state=10) 
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score
best = 0 
average = 0
total_for_average = 0
for i in [2]:
  for j in [2]:
    model = xgb.XGBClassifier(max_depth=i*2, n_estimators=100*j, learning_rate=0.01*j)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric=["error"])
    y_pred = model.predict(X_test)
    print(str(i) + "+" + str(j) + ": " + str(accuracy_score(y_test, y_pred)))
    total_for_average += 1
    average += accuracy_score(y_test, y_pred)
    if (accuracy_score(y_test, y_pred) > best): 
      best = accuracy_score(y_test, y_pred)
    print("\nThe Best is", best)

print("The Average is", average/total_for_average)
'''db=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
y_pred = np.zeros(len(df_test))
for model in models:
    y_pred += model.predict(df_test)

for i in range(len(df_test)):    
    if y_pred[i]>23:
         y_pred[i]=1 
    else:    
         y_pred[i]=0
        
y_pred = pd.DataFrame(y_pred).astype(int)
db['Survived'] = y_pred
db.to_csv("BJladikaSumbmission.csv", index = False)'''
db=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
db['Survived'] = y_pred_l
db.to_csv("BJladikaSumbmission6.csv", index = False)
df_group = df.groupby("Survived1")
df.groupby("Survived1")["Survived1"].hist(bins=18)
df_group["Sex"].hist()
