import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import random



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

from catboost import CatBoostRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import  RandomForestClassifier, GradientBoostingClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

lbl = LabelEncoder()

import os

color = sns.color_palette()

sns.set_style('darkgrid')

%matplotlib inline

print(os.listdir("../input"))

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 100)
path = "../input/"

train_identity = pd.read_csv(path+'train_identity.csv')

train_transaction = pd.read_csv(path+'train_transaction.csv')

test_identity = pd.read_csv(path+'test_identity.csv')

test_transaction = pd.read_csv(path+'test_transaction.csv')
# It will be comment before final submission

train_transaction = train_transaction.sample(frac=0.05, random_state=10)
train_identity.head()
train_transaction.head()
print("Train transaction are cols {} and rows {}".format(train_transaction.shape[0], train_transaction.shape[1]))

print("Train identity are cols {} and rows {}".format(train_identity.shape[0], train_identity.shape[1]))

print("Train transaction are cols {} and rows {}".format(test_transaction.shape[0], test_transaction.shape[1]))

print("Train identity are cols {} and rows {}".format(test_identity.shape[0], test_identity.shape[1]))
train_df = pd.merge(train_transaction, train_identity, how='left', left_on=['TransactionID'], right_on=['TransactionID'], right_index=False)

test_df = pd.merge(test_transaction, test_identity, how='left', left_on=['TransactionID'], right_on=['TransactionID'], right_index=False)
print("Train dataframe are cols {} and rows {}".format(train_df.shape[0], train_df.shape[1]))

print("test dataframe are cols {} and rows {}".format(test_df.shape[0], test_df.shape[1]))
train_df.head(5)
train_df.describe()
train_df.describe(include=['O'])
train_df[["ProductCD", "isFraud"]].groupby(['ProductCD'], as_index=False).mean().sort_values(by='isFraud', ascending=False)
sns.countplot(test_df['card6'])

plt.show()
train_df[["card6", "isFraud"]].groupby(['card6'], as_index=False).mean().sort_values(by='isFraud', ascending=False)
sns.countplot(test_df['card4'])

plt.show()
train_df[["card4", "isFraud"]].groupby(['card4'], as_index=False).mean().sort_values(by='isFraud', ascending=False)
sns.countplot(train_df['ProductCD'])

plt.show()
total_fraud = train_df.loc[(train_df['isFraud'] == 1),].shape[0]
fraud_percentage = (total_fraud*100)/train_df.shape[0]

print("Total fraud percenate ",format(fraud_percentage,'.2f'),"%")
round(train_df['TransactionAmt'].sum(), 2)
round(train_df['TransactionAmt'].max(),2 )
train_df.loc[(train_df['isFraud'] == 1),'TransactionAmt'].max()
train_df.loc[(train_df['isFraud'] == 1),'TransactionAmt'].min()
format(train_df.loc[(train_df['isFraud'] == 1),'TransactionAmt'].sum(), '.2f')
format((train_df.loc[(train_df['isFraud'] == 1),'TransactionAmt'].sum()*100)/train_df['TransactionAmt'].sum(), '.2f')
train_df['TransactionAmt'].apply(np.log).plot(kind='hist', bins=50) 

plt.show()
#for columns in train_df.columns:

#    print(columns)
print(train_df['card1'].min())

print(train_df['card2'].min())

print(train_df['card3'].min())

print(train_df['card5'].min())
print(train_df['card1'].max())

print(train_df['card2'].max())

print(train_df['card3'].max())

print(train_df['card5'].max())
train_df['card_avg'] = round((train_df['card1']+train_df['card2']+train_df['card3']+train_df['card5'])/4, 2)

test_df['card_avg'] = round((test_df['card1']+test_df['card2']+test_df['card3']+test_df['card5'])/4, 2)
pd.isnull(train_df['card_avg']).sum()
card_avg = round(train_df['card_avg'].mean(), 2)

train_df['card_avg']= train_df['card_avg'].fillna(card_avg)

test_df['card_avg']= test_df['card_avg'].fillna(card_avg)
train_df.head()
test_df.shape
#pd.isnull(train_df).sum()
selected_col = ['TransactionAmt','C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','D1','V95','V96','V97','V98','V99','V100','V101','V102','V103','V104','V105','V106','V107','V108','V109','V110','V111','V112','V113','V114','V115','V116','V117','V118','V119','V120','V121','V122','V123','V124','V125','V126','V127','V128','V129','V130','V131','V132','V133','V134','V135','V136','V137','V279','V280','V281','V282','V283','V284','V285','V286','V287','V288','V289','V290','V291','V292','V293','V294','V295','V296','V297','V298','V299','V300','V301','V302','V303','V304','V305','V306','V307','V308','V309','V310','V311','V312','V313','V314','V315','V316','V317','V318','V319','V320','V321','card_avg']
len(selected_col)
train_df[selected_col].columns[pd.isnull(train_df[selected_col]).sum() != 0]
import random

#random.choice([3, 4, 99, 29, 49])
#random.choice(train_df['V106'][~train_df['V106'].isnull()])
#train_df['V106'] = train_df['V106'].fillna(random.choice(train_df['V106'][~train_df['V106'].isnull()]))
na_cols = []

na_cols = ['D1', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103',

       'V104', 'V105', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113',

       'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122',

       'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131',

       'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V281', 'V282', 'V283',

       'V288', 'V289', 'V296', 'V300', 'V301', 'V313', 'V314', 'V315']
col =''

for col in na_cols:

    train_df[col] = train_df[col].fillna(random.choice(train_df[col][~train_df[col].isnull()]))

    test_df[col] = test_df[col].fillna(random.choice(test_df[col][~test_df[col].isnull()]))
test_df[selected_col].columns[pd.isnull(test_df[selected_col]).sum() != 0]
test_na_cols = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',

       'C12', 'C13', 'C14', 'V279', 'V280', 'V284', 'V285', 'V286', 'V287',

       'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V297', 'V298', 'V299',

       'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310',

       'V311', 'V312', 'V316', 'V317', 'V318', 'V319', 'V320', 'V321']



col =''

for col in test_na_cols:

    test_df[col] = test_df[col].fillna(random.choice(test_df[col][~test_df[col].isnull()]))
train_df.loc[:,train_df.dtypes =='object'].columns
train_df['ProductCD']= lbl.fit_transform(train_df['ProductCD']) 

test_df['ProductCD']= lbl.fit_transform(test_df['ProductCD']) 
from catboost import CatBoostRegressor

from catboost import CatBoostClassifier
catboost = CatBoostClassifier(iterations=1000)

catboost.fit(train_df[selected_col],

          train_df['isFraud'],

          verbose = False)
catboost.score(train_df[selected_col],train_df['isFraud'])
#catboost.predict(test_df[selected_col])
pd.isnull(train_df[selected_col]).sum()
train_df['V106'] = train_df['V106'].fillna(random.choice(train_df['V106'][~train_df['V106'].isnull()]))

test_df['V106'] = test_df['V106'].fillna(random.choice(test_df['V106'][~test_df['V106'].isnull()]))
model_name = []

model_score = []
kneighbors = KNeighborsClassifier()

kneighbors.fit(train_df[selected_col],train_df['isFraud'])

kneighbors_score = round(kneighbors.score(train_df[selected_col],train_df['isFraud'])*100, 2)

model_name.append('KNeighborsClassifier')

model_score.append(kneighbors_score)

kneighbors_score
linsvc = LinearSVC()

linsvc.fit(train_df[selected_col],train_df['isFraud'])

linsvc_score = round(linsvc.score(train_df[selected_col],train_df['isFraud'])*100, 2)

model_name.append('LinearSVC')

model_score.append(linsvc_score)

linsvc_score
randomforest = RandomForestClassifier(n_estimators=8, max_depth=10, min_samples_split=0.8, random_state=58)

randomforest.fit(train_df[selected_col],train_df['isFraud'])

randomforest_score = round(randomforest.score(train_df[selected_col],train_df['isFraud'])*100, 2)

model_name.append('RandomForestClassifier')

model_score.append(randomforest_score)

randomforest_score
#x_train, x_test, y_train, y_test = train_test_split(train_df[selected_col], train_df['isFraud'], test_size=0.05)
gradient = GradientBoostingClassifier()

gradient.fit(train_df[selected_col],train_df['isFraud'])

gradient_score = round(gradient.score(train_df[selected_col],train_df['isFraud'])*100, 2)

model_name.append('GradientBoostingClassifier')

model_score.append(gradient_score)

gradient_score
all_score = pd.DataFrame({'model_name':model_name, 'model_score':model_score})

all_score
#pd.isnull(test_df[selected_col]).sum()
predicted_fraud_detection = kneighbors.predict(test_df[selected_col])
my_submission = pd.DataFrame({'TransactionID':test_transaction['TransactionID'], 'isFraud':predicted_fraud_detection})
my_submission.to_csv('my_submission.csv', index=False)