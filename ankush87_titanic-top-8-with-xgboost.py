import pandas as pd
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df.info()
train_df.head()
train_df.describe()
train_df.describe(include="O")
test_df.describe()
test_df.describe(include="O")
train_df['Cabin_Start'] = train_df['Cabin'].str[0]
train_df[['Survived','Cabin_Start']].groupby(['Cabin_Start']).mean()
pd.crosstab(train_df['Survived'],train_df['Cabin_Start'])
train_df[['Pclass','Cabin_Start']].groupby(['Cabin_Start']).mean()
pd.crosstab(train_df['Pclass'],train_df['Cabin_Start'])
train_df[['Fare','Cabin_Start']].groupby(['Cabin_Start']).mean()
combine = [train_df, test_df]
for dataset in combine:

    dataset['Cabin'].fillna('0', inplace=True)

    dataset.loc[ dataset['Cabin'].str[0] == 'A', 'Cabin'] = 1

    dataset.loc[ dataset['Cabin'].str[0] == 'B', 'Cabin'] = 2

    dataset.loc[ dataset['Cabin'].str[0] == 'C', 'Cabin'] = 3

    dataset.loc[ dataset['Cabin'].str[0] == 'D', 'Cabin'] = 4

    dataset.loc[ dataset['Cabin'].str[0] == 'E', 'Cabin'] = 5

    dataset.loc[ dataset['Cabin'].str[0] == 'F', 'Cabin'] = 6

    dataset.loc[ dataset['Cabin'].str[0] == 'G', 'Cabin'] = 7

    dataset.loc[ dataset['Cabin'].str[0] == 'T', 'Cabin'] = 8

    dataset['Cabin'] = dataset['Cabin'].astype(int)
train_df.describe()
train_df.describe(include='O')
train_df = train_df.drop(["Cabin_Start"],axis=1)
train_df = train_df.drop(["PassengerId"],axis=1)
train_df["Ticket_Length"] = train_df["Ticket"].str.len()
train_df.head()
train_df["Ticket_Contains_Alpha"] = train_df["Ticket"].str.contains('^[a-zA-Z]')
train_df.head()
train_df[["Survived","Ticket_Length"]].groupby(['Ticket_Length'],as_index=False).mean()
pd.crosstab(train_df['Survived'],train_df['Ticket_Length'])
train_df[["Survived","Ticket_Contains_Alpha"]].groupby(['Ticket_Contains_Alpha']).mean()
pd.crosstab(train_df['Survived'],train_df['Ticket_Contains_Alpha'])
train_df[["Survived","Ticket_Contains_Alpha","Ticket_Length"]].groupby(["Ticket_Contains_Alpha","Ticket_Length"]).mean()
train_df = train_df.drop(["Ticket","Ticket_Length","Ticket_Contains_Alpha"],axis=1)

test_df = test_df.drop(["Ticket"],axis=1)
train_df.head()
train_df[["Survived","Pclass"]].groupby(["Pclass"],as_index=False).mean()
train_df["Name_Title"] = train_df["Name"].str.extract('([A-Za-z]+\.)',expand=False)
train_df.head()
train_df["Name_Title"].unique()
train_df[["Survived","Name_Title"]].groupby(["Name_Title"],as_index=False).mean()
pd.crosstab(train_df["Survived"],train_df["Name_Title"])
train_df["Name_Title"] = train_df["Name_Title"].replace(['Capt.','Don.','Jonkheer.','Rev.'],'Gone.')
pd.crosstab(train_df["Survived"],train_df["Name_Title"])
train_df["Name_Title"] = train_df["Name_Title"].replace(['Countess.','Lady.','Mlle.','Mme.','Ms.','Sir.'],'Left.')
pd.crosstab(train_df["Survived"],train_df["Name_Title"])
train_df["Name_Title"] = train_df["Name_Title"].replace(['Col.','Dr.','Major.'],'Half.')
pd.crosstab(train_df["Survived"],train_df["Name_Title"])
test_df["Name_Title"] = test_df["Name"].str.extract('([A-Za-z]+\.)',expand=False)

test_df["Name_Title"].unique()
test_df["Name_Title"] = test_df["Name_Title"].replace(['Capt.','Don.','Jonkheer.','Rev.'],'Gone.')

test_df["Name_Title"] = test_df["Name_Title"].replace(['Col.','Dr.','Major.'],'Half.')
test_df["Name_Title"].unique()
pd.crosstab(test_df["Name_Title"],test_df["Sex"])
test_df["Name_Title"] = test_df["Name_Title"].replace(['Dona.','Ms.'],'Mrs.')
pd.crosstab(test_df["Sex"],test_df["Name_Title"])
title_mapping = {"Mr.": 1, "Miss.": 2, "Mrs.": 3, "Master.": 4, "Gone.": 5, "Half.": 6, "Left.": 7}

train_df['Title_Encoded'] = train_df['Name_Title'].map(title_mapping)

train_df['Title_Encoded'] = train_df['Title_Encoded'].fillna(0)
train_df.head()
test_df['Title_Encoded'] = test_df['Name_Title'].map(title_mapping)

test_df['Title_Encoded'] = test_df['Title_Encoded'].fillna(0)
train_df = train_df.drop(["Name","Name_Title"],axis=1)

test_df = test_df.drop(["Name","Name_Title"],axis=1)
train_df.head()
test_df.head()
train_df.describe()
train_df.describe(include='O')
pd.crosstab(train_df["Survived"],train_df["Sex"])
train_df[["Survived","Sex"]].groupby(["Sex"],as_index=False).mean()
train_df["Sex"] = train_df["Sex"].map({'female':1,'male':0}).astype(int)
test_df["Sex"] = test_df["Sex"].map({'female':1,'male':0}).astype(int)
train_df.head(10)
pd.crosstab(train_df["Survived"],train_df["Age"])
train_df[["Survived","Age","Pclass"]].groupby(["Pclass"],as_index=False).mean()
train_df[["Survived","Age","Pclass","Sex"]].groupby(["Pclass","Sex"],as_index=False).mean()
train_df[["Survived","Age","Pclass","Sex","Title_Encoded"]].groupby(["Pclass","Sex","Title_Encoded"],as_index=False).mean()
train_df[["Survived","Age","Sex","Title_Encoded"]].groupby(["Sex","Title_Encoded"],as_index=False).mean()
train_df[["Age","Title_Encoded"]].groupby(["Title_Encoded"],as_index=False).mean()
train_df[["Age","Title_Encoded"]].groupby(["Title_Encoded"]).mean()
age_mapping = train_df[["Age","Title_Encoded"]].groupby(["Title_Encoded"]).mean().to_dict()
age_mapping
age_mapping["Age"]
train_df["Age"] = train_df["Age"].fillna(train_df["Title_Encoded"].map(age_mapping["Age"]))
train_df.describe()
test_df["Age"] = test_df["Age"].fillna(test_df["Title_Encoded"].map(age_mapping["Age"]))
test_df.describe()
train_df.head()
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean()
pd.crosstab(train_df["Survived"],train_df["AgeBand"])
combine = [train_df, test_df]
for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 

    dataset['Age'] = dataset['Age'].astype(int)
train_df.head()
test_df.head()
train_df= train_df.drop(["AgeBand"],axis=1)
train_df.head()
train_df[["Survived","SibSp"]].groupby(["SibSp"],as_index=False).mean()
pd.crosstab(train_df["Survived"],train_df["SibSp"])
train_df[["Survived","Parch"]].groupby(["Parch"],as_index=False).mean()
pd.crosstab(train_df["Survived"],train_df["Parch"])
train_df["FareBand"] = pd.qcut(train_df["Fare"],10)

train_df[["Survived","FareBand"]].groupby(["FareBand"],as_index=False).mean()
pd.crosstab(train_df["Survived"],train_df["FareBand"])
test_df["Fare"].fillna(test_df["Fare"].dropna().median(),inplace=True)
test_df.describe()
combine = [train_df, test_df]
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.55, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.55) & (dataset['Fare'] <= 7.854), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 8.05), 'Fare']   = 2

    dataset.loc[(dataset['Fare'] > 8.05) & (dataset['Fare'] <= 10.5), 'Fare']   = 3

    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 14.454), 'Fare']   = 4

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 21.679), 'Fare']   = 5

    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 27), 'Fare']   = 6

    dataset.loc[(dataset['Fare'] > 27) & (dataset['Fare'] <= 39.688), 'Fare']   = 7

    dataset.loc[(dataset['Fare'] > 39.688) & (dataset['Fare'] <= 77.958), 'Fare']   = 8

    dataset.loc[ dataset['Fare'] > 77.958, 'Fare'] = 9

    dataset['Fare'] = dataset['Fare'].astype(int)
train_df.head()
test_df.head()
train_df = train_df.drop(["FareBand"],axis=1)
train_df.head()
train_df[["Survived","Embarked"]].groupby(["Embarked"],as_index=False).mean()
pd.crosstab(train_df["Survived"],train_df["Embarked"])
train_df.describe(include="O")
test_df.describe(include="O")
train_df["Embarked"] = train_df["Embarked"].fillna(train_df["Embarked"].dropna().mode()[0])
train_df["Embarked"] = train_df["Embarked"].map({"C":0,"Q":1,"S":2}).astype(int)

test_df["Embarked"] = test_df["Embarked"].map({"C":0,"Q":1,"S":2}).astype(int)
train_df.describe()
test_df.describe()
X_train = train_df.drop(["Survived"], axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop(["PassengerId"], axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
#from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn import model_selection,metrics

from sklearn.metrics import confusion_matrix

import xgboost

from xgboost import plot_importance

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import GridSearchCV
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(X_train, Y_train,

                                                                    test_size=0.3,stratify=Y_train,random_state=0)
train_x.head()
xgboost_model = xgboost.XGBClassifier(objective='binary:logistic',learning_rate=0.1)
eval_set = [(train_x,train_y),(valid_x,valid_y)]
xgboost_model.fit(train_x,train_y,eval_metric=['error','logloss','auc'],eval_set=eval_set,verbose=True)
xgboost_model.score(train_x,train_y)
pred_y = xgboost_model.predict(valid_x)

metrics.accuracy_score(valid_y,pred_y)
pred_test = xgboost_model.predict(X_test)
submission = pd.DataFrame({"PassengerId":test_df["PassengerId"],"Survived":pred_test})
submission.to_csv('submission2.csv', index=False)
len(submission[submission.Survived ==1 ])
plot_importance(xgboost_model)

plt.show()
results = confusion_matrix(valid_y, pred_y) 

print(results)