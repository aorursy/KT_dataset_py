# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

print(os.listdir('/kaggle/input'))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from sklearn.preprocessing import Imputer

from sklearn_pandas import CategoricalImputer

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Activation

from sklearn.metrics import confusion_matrix

from sklearn.utils import class_weight

from sklearn.tree import DecisionTreeClassifier
sample_submission_csv=pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')

test_identity_csv=pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

train_identity_csv=pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

train_transaction_csv=pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

test_transaction_csv=pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
print(sample_submission_csv.head())

print('************************************************************************************************')

print(test_identity_csv.head())

print('************************************************************************************************')

print(train_identity_csv.head())

print('************************************************************************************************')

print(train_transaction_csv.head())

print('************************************************************************************************')

print(test_transaction_csv.head())
train_df = train_transaction_csv.merge(train_identity_csv, how='left')

test_df = test_transaction_csv.merge(test_identity_csv, how='left')

test_id=test_df['TransactionID']

print("Train shape : "+str(train_df.shape))

print("Test shape  : "+str(test_df.shape))
del train_transaction_csv

del test_transaction_csv

del train_identity_csv

del test_identity_csv
pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)
train_df.drop('TransactionID', axis=1, inplace=True)

test_df.drop('TransactionID', axis=1, inplace=True)
y_label=train_df['isFraud']
y_label.head(10)
y_label.hist()
train_df.drop('isFraud', axis=1, inplace=True)
train_df.head(10)
test_df.head()
train_amnt=train_df.shape[0]

print(train_df.shape)

print(test_df.shape)
train_amnt
train_df=train_df.append(test_df,ignore_index=True)
train_df.shape
cat_cols=['ProductCD','card1','card2','card3','card4','card5','card6','addr1','addr2','P_emaildomain','R_emaildomain','M1','M2','M3','M4','M5','M6','M7','M8','M9','DeviceType','DeviceInfo','id_12','id_13','id_14','id_15','id_16','id_17','id_18','id_19','id_20','id_21','id_22','id_23','id_24','id_25','id_26','id_27','id_28','id_29','id_30','id_31','id_32','id_33','id_34','id_35','id_36','id_37','id_38']

imp_cat=CategoricalImputer(missing_values="NaN", strategy="most_frequent" )

for col in cat_cols:

    if(train_df[col].isnull().sum()>0):

        train_df[col] = imp_cat.fit_transform(train_df[col])
cols=train_df.columns

imp=Imputer(missing_values="NaN", strategy="mean" )

for col in cols:

    if(train_df[col].isnull().sum()>0):

        train_df[[col]] = imp.fit_transform(train_df[[col]])
train_df.isnull().sum()
for col in cols:

    if(train_df[col].dtype=='object'):

        encoder = LabelEncoder()

        train_df[col]=encoder.fit_transform(train_df[col])
train_df.head(10)
for col in cols:

    print(train_df[col].dtype)
x_train=train_df[0:train_amnt]

x_test=train_df[train_amnt:train_df.shape[0]]
x_train.shape
y_label.shape
x_test.shape
del train_df
x_train=np.array(x_train)

x_test=np.array(x_test)

y_label=np.array(y_label)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_label, test_size=0.2,random_state=42)
x_train.shape
x_val.shape
y_train.shape
y_val.shape
model = Sequential()

model.add(Dense(32, activation='relu', input_dim=x_train.shape[1]))

model.add(Dense(100, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',

              loss='binary_crossentropy',

              metrics=['accuracy'])

model.summary()

class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(y_train),

                                                 y_train)
class_weights
#model.fit(x_train,y_train,validation_data=(x_val,y_val),batch_size=10,verbose=1,epochs=50,class_weight=class_weights)
#model.save('my_model.h5')
tree_clf = DecisionTreeClassifier(max_depth=200)
tree_clf.fit(x_train, y_train)
sample_submission_csv.head()
sample_submission_csv.shape
y_pred_val=tree_clf.predict(x_val)

cm = confusion_matrix(y_val, y_pred_val)

print(cm)
y_test=tree_clf.predict(x_test)
y_test
y_test.shape
sample_submission_csv['isFraud'][0]
test_id=np.array(test_id)

sample_submission_csv['TransactionID']=test_id

sample_submission_csv['isFraud']=y_test

sample_submission_csv.head()
sample_submission_csv.to_csv('submission.csv',index=False)

submission_data=pd.read_csv('submission.csv')

submission_data.head()