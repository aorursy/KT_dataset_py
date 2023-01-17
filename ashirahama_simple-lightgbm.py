import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
train.head()
test = pd.read_csv("../input/test.csv")
test.head()
# Combine train and test data
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.Survived.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['Survived'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:50]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
all_data['Cabin'] = all_data['Cabin'].fillna("Undefined")
def get_cabin_num(cabin):
    if cabin == "Undefined":
        return np.nan
    else:
        cabins = cabin.split(" ")
        count = len(cabins)
        return count
all_data['CabinCount'] = all_data['Cabin'].apply(get_cabin_num)
all_data["CabinCount"].max()
def split_cabin(cabin, num):
    if cabin == "Undefined":
        return "Undefined"
    else:
        cabins = cabin.split(" ")
        if len(cabins) >= num:
            return cabins[num - 1]
        else:
            return "Undefined"
all_data['Cabin1'] = all_data['Cabin'].apply(split_cabin, num=1)
all_data['Cabin2'] = all_data['Cabin'].apply(split_cabin, num=2)
all_data['Cabin3'] = all_data['Cabin'].apply(split_cabin, num=3)
all_data['Cabin4'] = all_data['Cabin'].apply(split_cabin, num=4)
def get_cabin_type(cabin):
    return cabin[0]
all_data['CabinType1'] = all_data['Cabin1'].apply(get_cabin_type)
all_data['CabinType2'] = all_data['Cabin2'].apply(get_cabin_type)
all_data['CabinType3'] = all_data['Cabin3'].apply(get_cabin_type)
all_data['CabinType4'] = all_data['Cabin4'].apply(get_cabin_type)
# CabinType別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('CabinType1').sum()
print(grouped_ch.sort_values("Survived", ascending=False).head(30))
# plt.bar(grouped_ch.index, grouped_ch.Survived.values)
sns.countplot('CabinType1',hue='Survived',data=train_ch)
# CabinType2別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('CabinType2').sum()
print(grouped_ch.sort_values("Survived", ascending=False).head(30))
# plt.bar(grouped_ch.index, grouped_ch.Survived.values)
sns.countplot('CabinType2',hue='Survived',data=train_ch)
def get_cabin_num(cabin):
    if cabin == "Undefined":
        return "Undefined"
    else:
        return "c_" + cabin[1:]
all_data['CabinNum1'] = all_data['Cabin1'].apply(get_cabin_num)
all_data['CabinNum2'] = all_data['Cabin2'].apply(get_cabin_num)
all_data['CabinNum3'] = all_data['Cabin3'].apply(get_cabin_num)
all_data['CabinNum4'] = all_data['Cabin4'].apply(get_cabin_num)
# CabinNum1別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('CabinNum1').sum()
print(grouped_ch.sort_values("Survived", ascending=False).head(30))
all_data = all_data.drop('Cabin', axis=1)
all_data = all_data.drop('CabinCount', axis=1)
all_data = all_data.drop('Cabin1', axis=1)
all_data = all_data.drop('Cabin2', axis=1)
all_data = all_data.drop('Cabin3', axis=1)
all_data = all_data.drop('Cabin4', axis=1)
# all_data = all_data.drop('CabinType1', axis=1)
all_data = all_data.drop('CabinType2', axis=1)
all_data = all_data.drop('CabinType3', axis=1)
all_data = all_data.drop('CabinType4', axis=1)
all_data = all_data.drop('CabinNum1', axis=1)
all_data = all_data.drop('CabinNum2', axis=1)
all_data = all_data.drop('CabinNum3', axis=1)
all_data = all_data.drop('CabinNum4', axis=1)
def get_type_from_name(name):
    array_name = name.split(",")
    family_name = array_name[0]
    name_with_type = array_name[1].split(".")
    type = name_with_type[0].replace(" ","")
    return type
all_data['Type'] = all_data['Name'].apply(get_type_from_name)
all_data['Type'].unique()
# Miss,Master,Mr,Mrs,Msくらいしかいらない。あとはOtherにする
all_data['Type'] = all_data['Type'].replace('Mlle', 'Miss')
all_data['Type'] = all_data['Type'].replace('Ms', 'Miss')
all_data['Type'] = all_data['Type'].replace('Mme', 'Mrs')
all_data['Type'] = [val if val in ['Mr', 'Mrs', 'Miss', 'Ms', 'Master'] else 'Others' for val in all_data['Type']]
# Type別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('Type').sum()
print(grouped_ch)
sns.countplot('Type',hue='Survived',data=train_ch)
all_data["Age"] = all_data.groupby(["Type", "Pclass"])["Age"].transform(
    lambda x: x.fillna(x.median()))
all_data["Embarked"] = all_data.groupby("Pclass")["Embarked"].transform(
    lambda x: x.fillna(x.mode()))
same_ticket_count_gp = all_data.groupby('Ticket', as_index=False).size().reset_index()
same_ticket_count_gp = same_ticket_count_gp.rename(columns={0: 'TicketCount'})
same_ticket_count_gp.head()
all_data = pd.merge(all_data, same_ticket_count_gp, on='Ticket', how='left')
all_data['Fare'] = all_data['Fare'] / all_data['TicketCount']
all_data.head()
all_data = all_data.drop('TicketCount', axis=1)
all_data["Fare"] = all_data.groupby("Pclass")["Fare"].transform(
    lambda x: x.fillna(x.median()))
all_data['FamilyNum'] = all_data['SibSp'] + all_data['Parch'] + 1
# FamilyNum別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('FamilyNum').sum()
print(grouped_ch)
# plt.bar(grouped_ch.index, grouped_ch.Survived.values)
sns.countplot('FamilyNum',hue='Survived',data=train_ch)
all_data['Is_Alone'] = all_data['FamilyNum'] == 1
# Is_Alone別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('Is_Alone').sum()
print(grouped_ch)
# plt.bar(grouped_ch.index, grouped_ch.Survived.values)
sns.countplot('Is_Alone',hue='Survived',data=train_ch)
def get_ticket_sign_1(ticket):
    array_name = ticket.split(" ")
    if len(array_name) > 1:
        sign = array_name[0]
        array_sign = sign.split("/")
        if len(array_sign) > 1:
            return array_sign[0]
        else:
            return array_sign[0]
    else:
        return "Undefined"
    
def get_ticket_sign_2(ticket):
    array_name = ticket.split(" ")
    if len(array_name) > 1:
        sign = array_name[0]
        array_sign = sign.split("/")
        if len(array_sign) > 1:
            return array_sign[1]
        else:
            return "Undefined"
    else:
        return "Undefined"

def get_ticket_no(ticket):
    array_name = ticket.split(" ")
    if len(array_name) > 2:
        return array_name[2]
    elif len(array_name) == 2:
        return array_name[1]
    else:
        return array_name[0]
    
all_data['Ticket_Sign1'] = all_data['Ticket'].apply(get_ticket_sign_1)
all_data['Ticket_Sign2'] = all_data['Ticket'].apply(get_ticket_sign_2)
all_data['Ticket_No'] = all_data['Ticket'].apply(get_ticket_no)
print(all_data['Ticket_Sign1'].unique())
print(all_data['Ticket_Sign2'].unique())
print(all_data['Ticket_No'].unique())
all_data['Ticket_Sign1'] = all_data['Ticket_Sign1'].replace('STON', 'SOTON')
all_data['Ticket_Sign1'] = all_data['Ticket_Sign1'].str.replace('.', '')
print(all_data['Ticket_Sign1'].unique())
# Ticket_Sign1別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('Ticket_Sign1').sum()
print(grouped_ch.sort_values("Survived", ascending=False).head(30))
# plt.bar(grouped_ch.index, grouped_ch.Survived.values)
sns.countplot('Ticket_Sign1',hue='Survived',data=train_ch)
# Ticket_Sign2別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('Ticket_Sign2').sum()
print(grouped_ch.sort_values("Survived", ascending=False).head(30))
# plt.bar(grouped_ch.index, grouped_ch.Survived.values)
sns.countplot('Ticket_Sign2',hue='Survived',data=train_ch)
# Ticket_No別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('Ticket_No').sum()
print(grouped_ch.sort_values("Survived", ascending=False).head(30))
# plt.bar(grouped_ch.index, grouped_ch.Survived.values)
# sns.countplot('Ticket_No',hue='Survived',data=train_ch)
all_data = all_data.drop('Ticket_No', axis=1)
all_data = all_data.drop('Ticket_Sign2', axis=1)
all_data = all_data.drop("Ticket", axis=1)
all_data['FareBin'] = pd.cut(all_data.Fare, 10, labels=False)
all_data['AgeBin'] = pd.cut(all_data.Age, 10, labels=False)
# FareBin別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('FareBin').sum()
print(grouped_ch)
sns.countplot('FareBin',hue='Survived',data=train_ch)
# AgeBin別の生存者数を見てみる
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
grouped_ch = train_ch.groupby('AgeBin').sum()
print(grouped_ch)
sns.countplot('AgeBin',hue='Survived',data=train_ch)
all_data = all_data.drop(['Fare', 'Age'], axis=1)
all_data.dtypes
all_data['Pclass'] = ['c_' + str(x) for x in all_data.Pclass]
# 文字列をラベル化した数値に変換する為のライブラリをインポート
from sklearn.preprocessing import LabelEncoder
# データタイプがobjectの列の値をラベル化した数値に変換
lbl = LabelEncoder()
lbl.fit(list(all_data['CabinType1'].values))
all_data['CabinType1'] = lbl.transform(list(all_data['CabinType1'].values))
lbl.fit(list(all_data['Ticket_Sign1'].values))
all_data['Ticket_Sign1'] = lbl.transform(list(all_data['Ticket_Sign1'].values))
all_data_PassengerId = all_data['PassengerId']
all_data_Name = all_data['Name']
all_data = all_data.drop('PassengerId', axis=1)
all_data = all_data.drop('Name', axis=1)
all_data = pd.get_dummies(all_data)
train_ch = all_data[:ntrain]
train_ch['Survived'] = y_train
# Find correlations with the target and sort
correlations = train_ch.corr()['Survived'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', correlations.tail(30))
all_data = all_data.drop('FamilyNum', axis=1)
all_data.head()
X_train = all_data.iloc[:train.shape[0],:]
X_test = all_data.iloc[train.shape[0]:,:]
X_test_PassengerId = all_data_PassengerId.iloc[train.shape[0]:]
import lightgbm as lgb
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
X = X_train
Y = y_train
X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.3, random_state=0
)
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)
# LightGBM parameters
params = {
    'boosting_type': 'gbdt',
    'num_boost_round': 100,
    'objective': 'binary',
    'eval_metric': 'logloss',
    'max_depth': 8,
    'learning_rate': 0.003,
    'num_iteration':5000,
    'feature_fraction': 0.50,
    'bagging_fraction': 0.80,
    'early_stopping_rounds': 1000,
    'bagging_freq': 30,
    'verbose': 0,
    'subsample': 0.8
}

# train
gbm = lgb.train(params,
                   lgb_train,
                   valid_sets=lgb_eval)
y_pred_val = gbm.predict(X_val, num_iteration=gbm.best_iteration)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

best_threadhold = 0
best_accuracy = 0
for i in range(1, 100, 1):
    i = i / 100
    print("thredhold:%f" %(i))
    
    y_pred_val_check = [1  if val >= i else 0 for val in y_pred_val]
    #Confusion matrix
    cm = confusion_matrix(y_val, y_pred_val_check)
    print(cm)

    #Accuracy
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(y_pred_val_check,y_val)
    print(accuracy)
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threadhold = i

print("best threadhold %f" %(best_threadhold))
print("best accuracy %f" %(best_accuracy))
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred
# result_mean = (LassoMd.predict(X_test.values) + ENetMd.predict(X_test.values) + KRRMd.predict(X_test.values) + GBoostMd.predict(X_test.values)  + XGBMd.predict(X_test.values)  + LGBMd.predict(X_test.values) ) / 6
# finalMd = [1  if val > 0.5 else 0 for val in result_mean]
# finalMd
y_test_pred = [1  if val >= best_threadhold else 0 for val in y_pred]
y_test_pred
submission = pd.DataFrame({
    "PassengerId": X_test_PassengerId,
    "Survived": y_test_pred
})
submission.to_csv('submission.csv', index=False)
