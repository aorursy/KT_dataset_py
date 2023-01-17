# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
base_path = "../input/"

str_trans = 'transaction'

str_ident = 'identity'

str_train = 'train'

str_test = 'test'



train_df1 = pd.read_csv(base_path+'%s_%s.csv'%(str_train, str_trans))

train_df2 = pd.read_csv(base_path+'%s_%s.csv'%(str_train, str_ident))

test_df1 = pd.read_csv(base_path+'%s_%s.csv'%(str_test, str_trans))

test_df2 = pd.read_csv(base_path+'%s_%s.csv'%(str_test, str_ident))
train_df1.head(10)
train_df2.head(10)
test_df1.head(10)
test_df2.head(10)
fraud = train_df1[train_df1['isFraud'] == 1]

print(fraud.shape[0] / train_df1.shape[0])

print(fraud.shape[0])

print(train_df1.shape[0])

train = pd.merge(train_df1, train_df2, on='TransactionID', how='left')

test = pd.merge(test_df1, test_df2, on='TransactionID', how='left')



miss_val_threshold = 0.25

col_to_del = []



for c in train.columns:

    if train[c].isnull().sum() > train.shape[0]*miss_val_threshold:

        col_to_del.append(c)



for c in test.columns:

    if train[c].isnull().sum() > test.shape[0]*miss_val_threshold:

        if c not in col_to_del:

            col_to_del.append(c)



train.drop(columns=col_to_del, inplace = True)

test.drop(columns=col_to_del, inplace = True)



target = train.pop('isFraud')
train.fillna(-1, inplace= True)

test.fillna(-1, inplace= True)



# df_train[:15].transpose()

col_int32 = ['TransactionID', 'TransactionDT']



col_int16 = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2',

           'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14',

           'D1', 'D10', 'D15', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',

           'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30',

           'V31', 'V32', 'V33', 'V34', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59',

           'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 

           'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79',

           'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89',

           'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99',

           'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109',

           'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119',

           'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129',

           'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V279',

           'V280', 'V281', 'V282', 'V283', 'V284', 'V285', 'V286', 'V287', 'V288', 'V289',

           'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298', 'V299',

           'V300', 'V301', 'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309',

           'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317', 'V318', 'V319',

           'V320', 'V321']



for c in col_int32:

    train[c] = train[c].astype(np.int32)

    test[c] = test[c].astype(np.int32)

    

for c in col_int16:

    train[c] = train[c].astype(np.int16)

    test[c] = test[c].astype(np.int16)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report

import matplotlib.pyplot as plt



def splitData(X, y):

    return train_test_split(X, y, test_size=0.1)
col_dummies = ['ProductCD', 'card4', 'card6', 'P_emaildomain']

for c in col_dummies:

    train[c] = pd.get_dummies(train[c])



train.drop(columns=col_dummies, inplace = True)

train.drop(columns=['TransactionID', 'TransactionDT'], inplace = True)
for loop in range(3):

    X_train, X_test, y_train, y_test = splitData(train, target)



    # classifier = GradientBoostingClassifier()

    classifier = RandomForestClassifier()

    classifier.fit(X_train, y_train)



    predict = classifier.predict(X_test)

    print(classification_report(y_test, predict))



    importances = classifier.feature_importances_

    indices = np.argsort(importances)

    names = list(X_train.columns)

    names = [names[idx] for idx in indices[-50:]]

    print('Importances list')

    for idx in range(len(indices[-50:])):

        print('%s: %f'%(names[idx], importances[indices[-50:][idx]]))



    plt.figure(figsize=(15, 10))

    plt.barh(range(len(names)), importances[indices[-50:]], align='center')

    plt.yticks(range(len(names)), names)

    plt.xlabel('Feature importance')

    plt.ylabel('Feature')

    plt.xlim(0, 0.2)

    plt.show()
