import tensorflow as tf

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.metrics import accuracy_score



from sklearn.ensemble import RandomForestClassifier



import tensorflow as tf

from tensorflow import keras



warnings.filterwarnings('ignore')
# Read the test and train data sets. 

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
print("Shapes of training and test datasets:");print(df_train.shape,df_test.shape)

print("Training Data Sample");display(df_train.head())

print("Test Data Sample");display(df_test.head())
# The following will be true if the column names in the training and test data sets are identical

set(df_train.columns[df_train.columns != 'target']) == set(df_test.columns)
print("Training Data Summary");df_train.describe()
print("Test Data Summary");df_test.describe()
fig, ax = plt.subplots()

random_cols = np.random.choice(range(1,df_train.shape[1]-1),16)

for col in df_train.columns[random_cols]:

    sns.kdeplot(df_train[col], ax = ax)

# Put the legend out of the figure

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title('Distribution of features in the Training Set (Sample of 16 features)')
fig, ax = plt.subplots()

random_cols = np.random.choice(range(1,df_test.shape[1]),16)

for col in df_test.columns[random_cols]:

    sns.kdeplot(df_train[col], ax = ax)

# Put the legend out of the figure

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.title('Distribution of features in the Test Set (Sample of 16 features)')
sns.set()

ax = sns.scatterplot(

    x = df_train.loc[:,(df_train.columns != 'target') & (df_train.columns != 'id') ].median(axis = 0),

    y = df_train.loc[:,(df_train.columns != 'target') & (df_train.columns != 'id') ].std(axis = 0), alpha = 0.8)

ax.set(xlabel='Median Values', ylabel='standard deviations')

plt.title('Medians vs. Standard Deviations of all the columns in the training data')
median_vals = df_train.loc[:,(df_train.columns != 'target') & (df_train.columns != 'id') ].median(axis = 0)

median_vals[median_vals > 100]
col = 'wheezy-copper-turtle-magic'

sns.distplot(df_train[col])
X_train, X_val, y_train, y_val = train_test_split(df_train.drop(columns = ['id','target']),df_train['target'], test_size=0.15, random_state=2)
sns.countplot(df_train['target'])

plt.title('Distribution of target variable in the training data')
# Randomly choose either True or False for each row. 

y_blind_guess = np.random.randint(2, size = len(y_val))

print("Accuracy with blind guessing: %.2f" % accuracy_score(y_val, y_blind_guess, normalize=True))
logreg = LogisticRegression().fit(X_train,y_train)

#y_pred = logreg.predict(X_val)

print("Accuracy of logisic Regression on Validation Set: %.2f" % logreg.score(X_val, y_val))
fig, axs = plt.subplots(ncols = 2 ,nrows=2, figsize=(8,8))

random_cols = df_train.columns[np.random.choice(range(1,df_train.shape[1]-1),8)]

sns.scatterplot(x = random_cols[0], y = random_cols[1], hue = 'target', data = df_train, ax=axs[0,0])

sns.scatterplot(x = random_cols[2], y = random_cols[3], hue = 'target', data = df_train, ax=axs[0,1])

sns.scatterplot(x = random_cols[4], y = random_cols[5], hue = 'target', data = df_train, ax=axs[1,0])

sns.scatterplot(x = random_cols[6], y = random_cols[7], hue = 'target', data = df_train, ax=axs[1,1])
set(df_test['wheezy-copper-turtle-magic']) - set(df_train['wheezy-copper-turtle-magic'])
df_train_2 = pd.get_dummies(df_train, columns = ['wheezy-copper-turtle-magic'], prefix='wctm-')

df_test_2 = pd.get_dummies(df_test, columns = ['wheezy-copper-turtle-magic'], prefix='wctm-')

X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(df_train_2.drop(columns = ['id','target']),df_train_2['target'], test_size=0.15, random_state=2)
logreg_2 = LogisticRegression().fit(X_train_2,y_train_2)

#y_pred = logreg.predict(X_val)

print("Accuracy of logisic Regression on Validation Set: %.2f" % logreg_2.score(X_val_2, y_val_2))
# Code from: https://www.kaggle.com/cdeotte/logistic-regression-0-800

    

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



cols = [c for c in train.columns if c not in ['id', 'target']]

oof = np.zeros(len(train))

skf = StratifiedKFold(n_splits=5, random_state=42)



# INITIALIZE VARIABLES

cols.remove('wheezy-copper-turtle-magic')

interactions = np.zeros((512,255))

oof = np.zeros(len(train))

preds = np.zeros(len(test))



# BUILD 512 SEPARATE MODELS

for i in range(512):

    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I

    train2 = train[train['wheezy-copper-turtle-magic']==i]

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx1 = train2.index; idx2 = test2.index

    train2.reset_index(drop=True,inplace=True)

    test2.reset_index(drop=True,inplace=True)

    

    skf = StratifiedKFold(n_splits=25, random_state=42)

    for train_index, test_index in skf.split(train2.iloc[:,1:-1], train2['target']):

        # LOGISTIC REGRESSION MODEL

        clf = LogisticRegression(solver='liblinear',penalty='l1',C=0.05)

        clf.fit(train2.loc[train_index][cols],train2.loc[train_index]['target'])

        oof[idx1[test_index]] = clf.predict_proba(train2.loc[test_index][cols])[:,1]

        preds[idx2] += clf.predict_proba(test2[cols])[:,1] / 25.0

        # RECORD INTERACTIONS

        for j in range(255):

            if clf.coef_[0][j]>0: interactions[i,j] = 1

            elif clf.coef_[0][j]<0: interactions[i,j] = -1

    #if i%25==0: print(i)

        

# PRINT CV AUC

auc = roc_auc_score(train['target'],oof)

print('LR with interactions scores CV =',round(auc,5))
nn_model = keras.Sequential([

    keras.layers.Dense(1024, input_dim = X_train_2.shape[1]),

    keras.layers.BatchNormalization(),

    keras.layers.Activation('relu'),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(256),

    keras.layers.BatchNormalization(),

    keras.layers.Activation('relu'),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(64),

    keras.layers.BatchNormalization(),

    keras.layers.Activation('relu'),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(1, activation='sigmoid')

])



nn_model.compile(optimizer= 'adam',loss='binary_crossentropy',metrics=['accuracy'])
nn_model.summary()
nn_history = nn_model.fit(X_train_2.values, y_train_2.values,

                                  epochs=5,

                                  batch_size=128,

                                  validation_data=(X_val_2.values, y_val_2.values),

                                  verbose=2)
sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = preds

sub.to_csv('submission.csv',index=False)