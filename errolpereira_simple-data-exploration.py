import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



PATH = '/kaggle/input/cat-in-the-dat/'
#Reading the dataset.

train_df = pd.read_csv(f'{PATH}train.csv', index_col='id')

test_df = pd.read_csv(f'{PATH}test.csv', index_col='id')
#exploring the train dataset.

train_df.head()
#shape of the datasets.

print(f'Training shape: {train_df.shape}')

print(f'Testing shape: {test_df.shape}')
#description

train_df.describe()
#checking for missing values.

train_df.isnull().sum()
#bin_0

train_df.bin_0.describe()
#value count.

train_df.bin_0.value_counts()
#countplot of bin_1 and bin_2

plt.figure(figsize=(19, 15))

plt.subplot(3, 2, 1)

sns.countplot(train_df.bin_1)

plt.subplot(3, 2, 2)

sns.countplot(train_df.bin_2)

plt.show()
#bin_3.

train_df.bin_3.value_counts()
#bin_4.

train_df.bin_4.value_counts(normalize=True)
#Exploring the high and low cardinality ordinal categorical features.

#1. ord_0

train_df.ord_0.value_counts(normalize=True)
#ord_1

train_df.ord_1.value_counts(normalize=True)
#ord_2

train_df.ord_2.value_counts(normalize=True)
#ord_3

train_df.ord_3.value_counts(normalize=True)
#ord_4

train_df.ord_4.value_counts(normalize=True)
#ord_5

train_df.ord_5.value_counts(normalize=True)
#Exploring high and low cardinality nominal features

#1.nom_0

train_df.nom_0.value_counts()
#1.nom_1

train_df.nom_1.value_counts()
#1.nom_2

train_df.nom_2.value_counts()
#1.nom_3

train_df.nom_3.value_counts()
#1.nom_4

train_df.nom_4.value_counts()
#1.nom_5

train_df.nom_5.value_counts()
#1.nom_6

train_df.nom_6.value_counts()
#1.nom_7

train_df.nom_7.value_counts()
#1.nom_8

train_df.nom_8.value_counts()
#1.nom_9

train_df.nom_9.value_counts()
#exploring cyclic features.

#1. day

train_df.day.value_counts()
#2. month

train_df.month.value_counts()
#exploring the target variable

sns.countplot(train_df.target)

print(train_df.target.value_counts(normalize=True))
#first checking the categrical variables containing in the train sets are also present in the test set.

def checkcat(df):

    for col in df.columns:

        length = len(set(test_df[col].values) - set(train_df[col].values))

        if length > 0:

            print(f'{col} in the test set has {length} values that are not present in the train set')

checkcat(test_df)
#One Hot Encoding binary features.

cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']



train_df = pd.get_dummies(train_df, columns=cols)

test_df = pd.get_dummies(test_df, columns=cols)
#Label encoding nominal features.

from sklearn.preprocessing import LabelEncoder



#nominal cols

cols = ['nom_'+str(i) for i in range(0, 10)]

for col in cols:

    #initializing.

    le = LabelEncoder()

    le.fit(list(train_df[col].values) + list(test_df[col].values))

    train_df[col] = le.transform(list(train_df[col].values))

    test_df[col] = le.transform(list(test_df[col].values))
train_df.head()
#Manually mapping the ordinal features.

ord_1 = {

    'Novice': 0,

    'Contributor': 1,

    'Expert': 2,

    'Master': 3,

    'Grandmaster': 4

}



ord_0 = {

    1 : 0,

    2 : 1,

    3 : 2

}



ord_2 = {

    'Freezing' : 0,

    'Cold' : 1,

    'Warm' : 2,

    'Hot' : 3,

    'Boiling Hot' : 4,

    'Lava Hot' : 5

}



ord_3 = {

    'a' : 0,

    'b' : 1,

    'c' : 2,

    'd' : 3,

    'e' : 4,

    'f': 5,

    'g' : 6,

    'h' : 7,

    'i' : 8,

    'j' : 9,

    'k' : 10,

    'l' : 11,

    'm' : 12,

    'n' : 13,

    'o' : 14

}



ord_4 = {

    'A' : 0,

    'B' : 1,

    'C' : 2,

    'D' : 3,

    'E' : 4,

    'F': 5,

    'G' : 6,

    'H' : 7,

    'I' : 8,

    'J' : 9,

    'K' : 10,

    'L' : 11,

    'M' : 12,

    'N' : 13,

    'O' : 14,

    'P' : 15,

    'Q' : 16,

    'R' : 17,

    'S' : 18,

    'T' : 19,

    'U' : 20,

    'V' : 21,

    'W' : 22,

    'X' : 23,

    'Y' : 24,

    'Z' : 25

}



#mapping.

train_df.ord_0 = train_df.ord_0.map(ord_0)

train_df.ord_1 = train_df.ord_1.map(ord_1)

train_df.ord_2 = train_df.ord_2.map(ord_2)

train_df.ord_3 = train_df.ord_3.map(ord_3)

train_df.ord_4 = train_df.ord_4.map(ord_4)



test_df.ord_0 = test_df.ord_0.map(ord_0)

test_df.ord_1 = test_df.ord_1.map(ord_1)

test_df.ord_2 = test_df.ord_2.map(ord_2)

test_df.ord_3 = test_df.ord_3.map(ord_3)

test_df.ord_4 = test_df.ord_4.map(ord_4)
#ord_5 : cannot deduce the order of importance.

##in this case will mean encode the column

mean = train_df.groupby('ord_5')['target'].mean()

train_df['ord_5'] = train_df['ord_5'].map(mean)

test_df['ord_5'] = test_df['ord_5'].map(mean)
#encoding the cyclic features.

#refrence : https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning

def encode(data, col, max_val):

    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)

    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)

    return data



#day

train_df = encode(train_df, 'day', 6)

test_df = encode(test_df, 'day', 6)



#month.

train_df = encode(train_df, 'month', 12)

test_df = encode(test_df, 'month', 12)
#dropping the day month columns.

train_df.drop(['day', 'month'], axis=1, inplace=True)

test_df.drop(['day', 'month'], axis=1, inplace=True)
#Creating X and y variables.

X = train_df.drop('target', axis=1)

y = train_df.target
print('#'*20)

print('StratifiedKFold training...')



# Same as normal kfold but we can be sure

# that our target is perfectly distribuited

# over folds

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score



folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=10)



#initializing the model

model = LogisticRegression()



#score.

score = []



for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=y)):

    print('Fold:',fold_+1)

    tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]    

    vl_x, v_y = X.iloc[val_idx,:], y[val_idx]

    

    #fitting on training data.

    model.fit(tr_x, tr_y)

    

    #predicting on test.

    y_pred = model.predict(vl_x)

    

    #storing score

    score.append(roc_auc_score(v_y, y_pred))

    print(f'AUC score : {roc_auc_score(v_y, y_pred)}')



print('Average AUC score', np.mean(score))

print('#'*20)
#fitting on the entire data.

model.fit(X, y)
#making predictions on test data.

pred_test = model.predict_proba(test_df)[:,0]
#submission file.

sub = pd.read_csv(f'{PATH}sample_submission.csv')

# #reseting index

# test_df = test_df.reset_index()

sub['target'] = pred_test

sub.to_csv('logistic_regression_base_model.csv', index=None, header=True)