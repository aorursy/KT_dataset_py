# Like & Upvote if you like it.



# Credit Card Fraud Detection

# General Neural Network

# UnderSample Dataset

# OverSample Dataset/SMOTE

# Confusion Metrics

# Data Visualiation
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_cc = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
df_cc.head()
df_cc.tail()
df_cc.columns
df_cc.isna().any()
df_cc.shape
df_cc.Amount
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

df_cc["normalized_amount"] = sc.fit_transform(df_cc["Amount"].values.reshape(-1, 1))

df_cc.drop(columns=["Amount"], inplace=True)
df_cc.normalized_amount.head(10)
# Drop time column

df_cc.drop(columns=["Time"], inplace=True)
df_cc.head()
X = df_cc.iloc[:, df_cc.columns != "Class"]

y = df_cc.iloc[:, df_cc.columns == "Class"]
X.head()
y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
print(X_train.shape)

print(X_test.shape)
X_train = np.array(X_train)

X_test = np.array(X_test)

y_train = np.array(y_train)

y_test = np.array(y_test)
X_train
import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

X.shape
clf = Sequential([

    Dense(activation="relu", units=16, input_dim=29)

    ,Dense(activation="relu", units=24)

    ,Dropout(0.5)

    ,Dense(activation="relu", units=20)

    ,Dense(activation="relu", units=24)

    ,Dense(activation="sigmoid", units=1)

])
clf.summary()
clf.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

clf.fit(X_train, y_train, batch_size=15, nb_epoch=5)
score = clf.evaluate(X_test, y_test)
print(score)
y_predict = clf.predict(X_test)

y_test = pd.DataFrame(y_test)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predict.round())

cm
sns.heatmap(cm, annot=True)
# When we are dealing with critical cases like fraud, medical and other sensitive information. We have to 

# minimise our false Negative. It mean in our case it can be seen that 39 frauds are there which we said

# its not but in actual they are. So 39 times people can fraud with us, which is a very bad thing.

# So our main target would be reducing it. 
# Plot confusion matrix for entire database

y_predict_all = clf.predict(X)

y_expected = pd.DataFrame(y)

cm_all = confusion_matrix(y_expected, y_predict_all.round())

sns.heatmap(cm_all, annot=True)
# So for complete dataset, according to our model 94 times user can make fraud. So we have to improve our model

# to reduce the possibilty of fraud
# before modell tuning, lets compare or model with random forst
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train_rf, y_train_rf.values.ravel())

y_pred_rf = rf.predict(X_test_rf)

y_test_rf = pd.DataFrame(y_test_rf)

cm_rf = confusion_matrix(y_test_rf, y_pred_rf.round())

sns.heatmap(cm_rf, annot=True)
df_cc.Class.value_counts(normalize=True)
fraud_indices = np.array(df_cc[df_cc.Class == 1].index) # All fraud indices
len(fraud_indices) # total fraudlent records
#normal indices

normal_indices = np.array(df_cc[df_cc.Class == 0].index)

len(normal_indices)
# Choose random normal rows equal to fraud_indices. So that we can have equal data to train our model.

random_normal_indices = np.random.choice(normal_indices, len(fraud_indices), replace=False)

random_normal_indices = np.array(random_normal_indices)
len(random_normal_indices) # So we selected random normal indices equals to the number of total fraud rows.
undersample_indices = np.concatenate([fraud_indices, random_normal_indices])

len(undersample_indices)
# So now we have total 984 records with 492 - Fraud case and 492 real transaction case

# lets create our dataset with these indices

undersample_df = df_cc.iloc[undersample_indices, :]
undersample_df.Class.value_counts().plot(kind="bar") # So now our new undersampled dataset is balanced

# Which is a good thing. So lets check whether it predict data in proper way or not?
X_under = undersample_df.iloc[:, undersample_df.columns != 'Class']

y_under = undersample_df.iloc[:, undersample_df.columns == 'Class']
y_under.head()
X_train_us, X_test_us, y_train_us, y_test_us = train_test_split(X_under, y_under, test_size=0.3, random_state=0)
X_train_us = np.array(X_train_us)

X_test_us = np.array(X_test_us)

y_train_us = np.array(y_train_us)

y_test_us = np.array(y_test_us)
clf.summary()
clf.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

clf.fit(X_train_us, y_train_us, batch_size=15, nb_epoch=5)
y_pred_us = clf.predict(X_test_us)
cm_us = confusion_matrix(y_test_us, y_pred_us.round() )

sns.heatmap(cm_us, annot=True)

# Wow we literally reduced the number of false negative. But lets check for whole dataset.
y.shape
y_pred = clf.predict(X)

y_expected = pd.DataFrame(y)



cm_us = confusion_matrix(y_expected, y_pred.round() )

sns.heatmap(cm_us, annot=True)

# So we have better result than previous. Lets try to improve it more.
# uncomment the below line to install library to use SMOTE.

# !pip install -U imbalanced-learn
X.shape
from imblearn.over_sampling import SMOTE
# Generating OVERSAMPLE data

X_oversample, y_oversample = SMOTE().fit_sample(X, y.values.ravel())
# Converting array into dataframe

X_oversample = pd.DataFrame(X_oversample)

y_oversample = pd.DataFrame(y_oversample)
X_oversample.head()
print(X_oversample.shape)

print(y_oversample.shape)

# Now we can see we have much more data to train because of oversampling
# Get train test data

X_train_smote, X_test_smote, y_train_smote, y_test_smote = train_test_split(X_oversample, y_oversample, 

                                                                            test_size=0.3)

X_test_smote = np.array(X_test_smote)

X_train_smote = np.array(X_train_smote)

y_train_smote = np.array(y_train_smote)

y_test_smote = np.array(y_test_smote)
# Fit model for SMOTE data

clf.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

clf.fit(X_train_smote, y_train_smote, batch_size=15, nb_epoch=5)
y_predict_smote = clf.predict(X_test_smote)
cm_smote = confusion_matrix(pd.DataFrame(y_test_smote), y_predict_smote.round())

sns.heatmap(cm_smote, annot=True)
# So finally we reduced it. We will stop here We can do other changes like we can change the epoch,

# or change the batch size. Change test_size. Try them all and tell us the best model you got..

# Thank You... :) SHARE AND UPVOT
