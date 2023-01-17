import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn')



import warnings

warnings.filterwarnings('ignore')



from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,MinMaxScaler

from sklearn.model_selection import train_test_split as train_valid_split

from sklearn.metrics import classification_report



from keras.models import Sequential

from keras.layers import Dropout, Dense

from keras.callbacks import EarlyStopping

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

Xs = train.drop(['target'],axis=1)

y = train['target']

del train

test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')

test_ids = test['id']

df = Xs.append(test)

del test, Xs

df = df.drop('id',axis=1)

df.info()
bin_cols = [col for col in df.columns if 'bin_' in col]

nom_cols = [col for col in df.columns if 'nom_' in col]

ord_cols = [col for col in df.columns if 'ord_' in col] + ['day','month']
df[bin_cols].head()
bin_map = {'bin_3':{'T':1,'F':0},'bin_4':{'Y':1,'N':0}}

df = df.replace(bin_map)
df[bin_cols].head()
nrow=2

ncol=3

fig, axes = plt.subplots(nrow, ncol,

                         figsize=(20,10))

count=0

for r in range(nrow):

    for c in range(ncol):

        if(count==len(bin_cols)):

            axes[r,c].set_visible(False)

            break

        col = bin_cols[count]

        df[col].value_counts().plot(kind='bar',ax=axes[r,c])

        axes[r,c].set_title(col,fontsize=20)

        count = count+1
df[nom_cols].head()
df[nom_cols].describe()
nom_mapping = {}

for col in nom_cols:

    LE = LabelEncoder().fit(df[col])

    nom_mapping[col] = dict(zip(LE.classes_, LE.transform(LE.classes_)))

    df[col] = LE.transform(df[col])
df[nom_cols].head()
nrow=3

ncol=3

fig, axes = plt.subplots(nrow, ncol,

                         figsize=(30,15))

count=0

for r in range(nrow):

    for c in range(ncol):

        if(count==len(nom_cols)):

            axes[r,c].set_visible(False)

            break

        col = nom_cols[count]

        df[col].hist(bins=100,ax=axes[r,c])

        axes[r,c].set_title(col,fontsize=20)

        count = count+1
df[ord_cols].head()
df[['ord_1','ord_2','ord_3','ord_4','ord_5']].describe()
ord_mapping = {}

ord_mapping['ord_1'] = {'Novice':1,'Contributor':2,'Expert':3,'Master':4,'Grandmaster':5}

ord_mapping['ord_2'] = {'Freezing':1,'Cold':2,'Warm':3,'Hot':4,'Boiling Hot':5,'Lava Hot':6}

df = df.replace(ord_mapping)
OE = OrdinalEncoder().fit(df[['ord_3','ord_4','ord_5']])

df[['ord_3','ord_4','ord_5']] = OE.transform(df[['ord_3','ord_4','ord_5']])
df[ord_cols].head()
nrow=3

ncol=3

fig, axes = plt.subplots(nrow, ncol,

                         figsize=(30,15))

count=0

for r in range(nrow):

    for c in range(ncol):

        if(count==len(ord_cols)):

            axes[r,c].set_visible(False)

            break

        col = ord_cols[count]

        df[col].hist(bins=100,ax=axes[r,c])

        axes[r,c].set_title(col,fontsize=20)

        count = count+1
df.head()
MMS = MinMaxScaler()

MMS.fit(df)

df = pd.DataFrame(MMS.transform(df),columns=df.columns)
df.head()
train = df.loc[:299999]

train.tail()
test = df.loc[300000:]

test.head()
# y was already removed from train in the first code chunk

Xs = train

del train, df

X_train,X_valid,y_train,y_valid = train_valid_split(Xs, y, test_size = .1,

                                                    random_state=0)

X_train.shape,X_valid.shape
%%time

model = Sequential()

model.add(Dense(1042, input_dim=23, activation='relu'))

model.add(Dense(512, activation='relu'))

model.add(Dense(256, activation='relu'))

model.add(Dense(128, activation='relu'))

model.add(Dense(64, activation='relu'))

model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['binary_accuracy'])



es = EarlyStopping(monitor = 'val_accuracy',

                          min_delta = 0.007,

                          patience = 1)

logs = model.fit(X_train, y_train,

                 callbacks=[es],epochs=100)
def get_evaluations(model):

    preds = model.predict_proba(X_train)

    plt.hist(preds,bins=100)

    plt.title('Training Predictions')

    plt.show();

    print('train_report',classification_report(y_train,np.round(preds)))

    preds = model.predict_proba(X_valid)

    plt.hist(preds,bins=100)

    plt.title('Validation Predictions')

    plt.show();

    print('valid_report',classification_report(y_valid,np.round(preds)))

    

get_evaluations(model)
#free up some space for memory

del X_train, y_train, X_valid, y_valid
# test is already prep with train

# we also got the test ids in the first code chunk



sub_df = pd.DataFrame()

sub_df['id'] = test_ids

sub_df['target'] = model.predict_proba(test)

display(sub_df.head())



sub_df.to_csv('the-sub-mission.csv',index=False)

sub_df['target'].hist(bins=100)

plt.title('Test Preditions')

plt.show();