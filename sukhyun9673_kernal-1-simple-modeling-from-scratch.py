# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
# Any results you write to the current directory are saved as output.
nulls=df_train.isnull().sum()
print(nulls)


length = len(df_train)
percentage = (nulls/length)*100
print (percentage)
#plotting
x = percentage.values
y = np.array(percentage.index)

plt.figure(figsize=(16, 5))
sns.set(font_scale=1.2)
ax = sns.barplot(y, x, palette='hls', log=False)
ax.set(xlabel='Feature', ylabel='(Percentage of Nulls)', title='Number of Nulls')
df_train.head()
for i in np.array(df_train.columns):
    print ("{0} has {1} attributes".format(i, len(df_train[i].unique())))
df = pd.concat([df_train.drop("Survived", axis = 1), df_test])
label = df_train["Survived"]
index = df_train.shape[0]
df["Ticket_number"] = df["Ticket"].apply(lambda x: x.split()[-1])
df["Ticket_code"] = df["Ticket"].apply(lambda x: x.split()[0] if len(x.split())!= 1 else "No Code")
df["Ticket_code"].unique()
df['Ticket_code'].value_counts()
df["Ticket_code"]= df["Ticket_code"].apply(lambda a: a[:-1] if a[-1]=="." else a)
df['Ticket_code'].value_counts()
#/와 . 모두 분리
import re
codes= [i for i in df["Ticket_code"].unique() if i!= "No Code"]
def split_codes(code):
    return re.split('[^a-zA-Z0-9]+', code)
    
new_codes = []
for i in codes:
    for j in  split_codes(i):
        new_codes.append(j)
        
pd.Series(new_codes).value_counts()
#/만 분리
def split_codes2(code):
    return re.split('/+', code)
    
new_codes2 = []
for i in codes:
    for j in  split_codes2(i):
        new_codes2.append(j)
        
pd.Series(new_codes2).value_counts()
#.만 분리
def split_codes3(code):
    return re.split('\.+', code)
    
new_codes3 = []
for i in codes:
    for j in  split_codes3(i):
        new_codes3.append(j)
        
pd.Series(new_codes3).value_counts()
df["Ticket_code_HEAD"] = df["Ticket_code"].apply(lambda x: re.split('[^a-zA-Z0-9]+', x)[0])
df['Ticket_code_HEAD'].value_counts()
df["Name"]
df["Initial"] = df.Name.str.extract('([A-Za-z]+)\.')
pd.crosstab(df["Initial"], df["Sex"]).T.style.background_gradient(cmap='summer_r')
df['Initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona'],
                        ['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr', 'Mr'],inplace=True)
train = df[:length]
train["Survived"] = label
train.groupby('Initial').mean()["Survived"].plot.bar()
train.groupby('Initial').mean()
print ("{} percent of Cabin data is null".format(df["Cabin"].isnull().sum()/len(df)*100))
train_cabin = train[train["Cabin"].notnull()]
train_cabin["Cabin_Initial"] = train_cabin["Cabin"].apply(lambda x: x[0])
train_cabin["Cabin_Number"] = train_cabin["Cabin"].apply(lambda x: (x[1:].split(" ")[0]))
train_cabin["Cabin_Number"].replace("", -1, inplace = True)
train_cabin["Cabin_Number"] = train_cabin["Cabin_Number"].apply(lambda x: int(x))

train_cabin["Cabin_Initial"].value_counts()
train_cabin.groupby("Cabin_Initial").mean()
train_cabin.groupby("Cabin_Initial").mean()["Survived"].plot.bar()
df_cabin = df[df["Cabin"].notnull()]
df_cabin["Cabin_Initial"] = df_cabin["Cabin"].apply(lambda x: x[0])
df_cabin["Cabin_Number"] = df_cabin["Cabin"].apply(lambda x: (x[1:].split(" ")[0]))
df_cabin["Cabin_Number"].replace("", -1, inplace = True)
df_cabin["Cabin_Number"] = df_cabin["Cabin_Number"].apply(lambda x: int(x))

df_cabin.groupby("Cabin_Initial").mean()
#Heatmap 그려보기
df_cabin_heatmap = df_cabin[["Pclass", "Age", "SibSp", "Parch", "Fare", "Cabin_Number", "Cabin_Initial"]]
df_cabin_heatmap['Cabin_Initial'] = df_cabin_heatmap['Cabin_Initial'].map({'A': 0, 'B': 1, "C":2, "D":3, "E":4, "F":5, "G":7, "H":8})

colormap = plt.cm.RdBu
plt.figure(figsize=(14, 12))
plt.title('Correlation, y=1.05, size=15')
sns.heatmap(df_cabin_heatmap.astype(float).corr(), linewidths=0.1, vmax=1.0,
           square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={"size": 16})
df["Embarked"].isnull().sum()
df_train['Embarked'].fillna('S', inplace=True)
from sklearn import preprocessing

df["Cabin_Initial"] = df["Cabin"].apply(lambda x: x[0] if pd.notnull(x) else x)
#Drop non-using columns
df = df.drop(["Name", "Ticket", "Ticket_code", "Cabin"], axis = 1)
categorical = ["Sex", "Embarked", "Ticket_code_HEAD", "Initial", "Cabin_Initial"] 

lbl = preprocessing.LabelEncoder()
for col in categorical:
    df[col].fillna('Unknown')
    df[col] = lbl.fit_transform(df[col].astype(str))

df.groupby("Initial").mean()
#Initial 을 가지고 Fillna
df["Age"] = df.groupby("Initial").transform(lambda x: x.fillna(x.mean()))["Age"]
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df= df.drop("PassengerId", axis = 1)
df= df.drop("Ticket_number", axis = 1)
train = df[:length]
test = df[length:]

y = label

import xgboost as xgb
from sklearn.model_selection import train_test_split
X = np.array(train)
y = np.array(y)

dtest = np.array(test)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=23)

def score(a, b):
    count = 0
    for i in range(len(a)):
        if a[i] == b[i]:
           count+=1
    return count

gbm =xgb.XGBClassifier(max_depth=3,n_eatimators=1000,early_stop_rounds = 100, learning_rate=0.15).fit(X_train,y_train)
prediction_test = gbm.predict(X_valid)
score(prediction_test, y_valid)



prediction = gbm.predict(dtest)

submission=pd.DataFrame({'PassengerId':pd.read_csv("../input/test.csv")['PassengerId'],'Survived':prediction})
submission.to_csv("submission.csv",index=False)