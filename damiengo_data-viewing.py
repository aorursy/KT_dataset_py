%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train['Original_Quote_Date_Typed'] = pd.to_datetime(train.Original_Quote_Date)
train['month'] = train.Original_Quote_Date_Typed.apply(lambda x: x.strftime('%m'))
train['day_of_week'] = train.Original_Quote_Date_Typed.apply(lambda x: x.strftime('%w'))

test['Original_Quote_Date_Typed'] = pd.to_datetime(test.Original_Quote_Date)
test['month'] = test.Original_Quote_Date_Typed.apply(lambda x: x.strftime('%m'))
test['day_of_week'] = test.Original_Quote_Date_Typed.apply(lambda x: x.strftime('%w'))
train_by_month = train[["month", "QuoteNumber"]].groupby(['month'],as_index=False).count()

fig, axis1 = plt.subplots(1,1,figsize=(15,4))
sns.barplot(x='month', y='QuoteNumber', data=train_by_month)
test_by_month = test[["month", "QuoteNumber"]].groupby(['month'],as_index=False).count()

fig, axis1 = plt.subplots(1,1,figsize=(15,4))
sns.barplot(x='month', y='QuoteNumber', data=test_by_month)
print('PersonalField10A with -1: '+str(len(train[train["PersonalField10A"] == -1])))
print('PersonalField10A with -1 and converted: '+str(len(train[(train["PersonalField10A"] == -1) & (train['QuoteConversion_Flag'] == 1)])))

train_by_prf10a = train[["PersonalField10A", "QuoteNumber"]].groupby(['PersonalField10A'],as_index=False).count()
fig, axis1 = plt.subplots(1,1,figsize=(15,4))
sns.barplot(x='PersonalField10A', y='QuoteNumber', data=train_by_prf10a)
train_prf10_m1 = train[train["PersonalField10A"] == -1]

train_prf10_m1_by_month = train_prf10_m1[["month", "QuoteNumber"]].groupby(['month'],as_index=False).count()
fig, axis1 = plt.subplots(1,1,figsize=(15,4))
sns.barplot(x='month', y='QuoteNumber', data=train_prf10_m1_by_month)
print('PersonalField10B with -1: '+str(len(train[train["PersonalField10B"] == -1])))

train_by_prf10b = train[["PersonalField10B", "QuoteNumber"]].groupby(['PersonalField10B'],as_index=False).count()
fig, axis1 = plt.subplots(1,1,figsize=(15,4))
sns.barplot(x='PersonalField10B', y='QuoteNumber', data=train_by_prf10b)
print('PropertyField20 not 0: '+str(len(train[train["PropertyField20"] != 0])))

train_by_prf20 = train[["PropertyField20", "QuoteNumber"]].groupby(['PropertyField20'],as_index=False).count()
fig, axis1 = plt.subplots(1,1,figsize=(15,4))
sns.barplot(x='PropertyField20', y='QuoteNumber', data=train_by_prf20)
print('PersonalField8 not 1: '+str(len(train[train["PersonalField8"] != 1])))

train_by_pef8 = train[["PersonalField8", "QuoteNumber"]].groupby(['PersonalField8'],as_index=False).count()
fig, axis1 = plt.subplots(1,1,figsize=(15,4))
sns.barplot(x='PersonalField8', y='QuoteNumber', data=train_by_pef8)
print('GeographicField22A not -1: '+str(len(train[train["GeographicField22A"] != -1])))
print('GeographicField22A not -1 and converted: '+str(len(train[(train["GeographicField22A"] != -1) & (train['QuoteConversion_Flag'] == 1)])))

train_by_pef8 = train[["GeographicField22A", "QuoteNumber"]].groupby(['GeographicField22A'],as_index=False).count()
fig, axis1 = plt.subplots(1,1,figsize=(15,4))
sns.barplot(x='GeographicField22A', y='QuoteNumber', data=train_by_pef8)
print('GeographicField23A not -1: '+str(len(train[train["GeographicField23A"] != -1])))
print('GeographicField23A not -1 and converted: '+str(len(train[(train["GeographicField23A"] != -1) & (train['QuoteConversion_Flag'] == 1)])))

train_by_pef8 = train[["GeographicField23A", "QuoteNumber"]].groupby(['GeographicField23A'],as_index=False).count()
fig, axis1 = plt.subplots(1,1,figsize=(15,4))
sns.barplot(x='GeographicField23A', y='QuoteNumber', data=train_by_pef8)
print('GeographicField61A not -1: '+str(len(train[train["GeographicField61A"] != -1])))
print('GeographicField61A not -1 and converted: '+str(len(train[(train["GeographicField61A"] != -1) & (train['QuoteConversion_Flag'] == 1)])))

train_by_gef61a = train[["GeographicField61A", "QuoteNumber"]].groupby(['GeographicField61A'],as_index=False).count()
fig, axis1 = plt.subplots(1,1,figsize=(15,4))
sns.barplot(x='GeographicField61A', y='QuoteNumber', data=train_by_gef61a)
print('GeographicField62A not -1: '+str(len(train[train["GeographicField62A"] != -1])))
print('GeographicField62A not -1 and converted: '+str(len(train[(train["GeographicField62A"] != -1) & (train['QuoteConversion_Flag'] == 1)])))

train_by_gef62a = train[["GeographicField62A", "QuoteNumber"]].groupby(['GeographicField62A'],as_index=False).count()
fig, axis1 = plt.subplots(1,1,figsize=(15,4))
sns.barplot(x='GeographicField62A', y='QuoteNumber', data=train_by_gef62a)
train_converted = train[train["QuoteConversion_Flag"] == 1]

train_converted_by_month = train_converted[["month", "QuoteNumber"]].groupby(['month'],as_index=False).count()
fig, axis1 = plt.subplots(1,1,figsize=(15,4))
sns.barplot(x='month', y='QuoteNumber', data=train_converted_by_month)
