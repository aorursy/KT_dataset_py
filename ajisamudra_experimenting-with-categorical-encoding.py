# Library

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import scipy



from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OrdinalEncoder

from sklearn.impute import SimpleImputer

import category_encoders as cat_encoder



import warnings

warnings.filterwarnings('ignore')
# Read file

train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')

submission = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')



# Drop id on train and test dataset

train = train.drop('id', axis = 1)

test = test.drop('id', axis = 1)



# Select only baseline features

baseline_features = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'day', 'month', 'nom_0',

                     'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_7', 'nom_8',

                     'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']



train_target = train['target']

train = train[baseline_features]

test = test[baseline_features]





train.shape, train_target.shape, test.shape

# We have 600k samples for training and 400k for test
# See number of unique values of each features in train dataset

train.nunique()



# We could see there are features with high number of unique value such as nom_5 - nom_9

# There are also features with low number of unique value, or we commonly call result of one-hot encoding, such as bin_0 - bin_4

# Then let see the missing value in the dataset
# See number of missing values of each features in train dataset

train.isnull().sum()



# All of features have missing values with quite high of number
# Impute with constant

columns = train.columns



for i in columns:

    imputer = SimpleImputer(missing_values=np.nan, strategy= 'constant', add_indicator= True)

    imputer.fit(train[i].to_numpy().reshape(-1,1))

    

    train[i] = imputer.transform(train[i].to_numpy().reshape(-1,1))

    test[i] = imputer.transform(test[i].to_numpy().reshape(-1,1))



train.shape, test.shape
# Split train validation dataset

X_train, X_val, y_train, y_val = train_test_split(train,

                                                  train_target,

                                                  test_size = 0.2,

                                                  stratify = train_target,

                                                  random_state = 41)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
# Label encoding to all columns

columns = X_train.columns



for i in columns:

    label_encoder = LabelEncoder()

    label_encoder.fit(X_train[i])

    

    X_train[i] = label_encoder.transform(X_train[i])

    X_val[i] = label_encoder.transform(X_val[i])



X_train.shape, X_val.shape
# Standardize the values

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)
# Measure performance on Validation dataset

logit = LogisticRegression()

logit.fit(X_train, y_train)

y_pred =  logit.predict_proba(X_val)



score = roc_auc_score(y_val, y_pred[:,1])

print("Baseline model AUC Score: {}".format(score))
# Train on all dataset and make submission



# Label encoding on Training and Test dataset

columns = train.columns



for i in columns:

    label_encoder = LabelEncoder()

    label_encoder.fit(train[i])

    

    train[i] = label_encoder.transform(train[i])

    test[i] = label_encoder.transform(test[i])

    

# Standardize the values

scaler = StandardScaler()

X_train = scaler.fit_transform(train)

X_test = scaler.transform(test)



# Training model

logit = LogisticRegression()

logit.fit(X_train, train_target)



# Predict

y_pred = logit.predict_proba(X_test)
# Make file for submission

baseline_submission = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')

baseline_submission['target'] = y_pred[:,1]

baseline_submission.to_csv('baseline_model.csv', index=False)
# Pearson correlation of each features to target

cor_mtx = pd.concat( [train[baseline_features], train_target], axis = 1).corr()

plt.subplots()

sns.heatmap(cor_mtx)



print(cor_mtx['target'].sort_values(ascending=False))



# Knowing that we only have a few features that have high linearly correlated to target

# We need to look another way to do features engineering
# Read file

train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')



# Drop id on train and test dataset

train = train.drop('id', axis = 1)

test = test.drop('id', axis = 1)



# Select only baseline features

baseline_features = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'day', 'month', 'nom_0',

                     'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_7', 'nom_8',

                     'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']



train_target = train['target']

train = train[baseline_features]

test = test[baseline_features]



train.shape, train_target.shape, test.shape

# We have 600k samples for training and 400k for test
# Impute with constant



columns = train.columns



for i in columns:

    

    if train[i].dtype == object:

        imputer = SimpleImputer(missing_values=np.nan, strategy= 'constant', add_indicator= True)

    else:

        imputer = SimpleImputer(missing_values=np.nan, strategy= 'constant', fill_value= -1, add_indicator= True)



    imputer.fit(train[i].to_numpy().reshape(-1,1))

    

    train[i] = imputer.transform(train[i].to_numpy().reshape(-1,1))

    test[i] = imputer.transform(test[i].to_numpy().reshape(-1,1))



train.shape, test.shape
# Split train validation dataset

X_train, X_val, y_train, y_val = train_test_split(train,

                                                  train_target,

                                                  test_size = 0.2,

                                                  stratify = train_target,

                                                  random_state = 41)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
# Ordinal Encoding for features ord_1, ord_2, ord_3, ord_4



def encode_ord_1(x):

    if x == "Novice":

        return 0

    elif x == "Contributor":

        return 1

    elif x == "Expert":

        return 2

    elif x == "Master":

        return 3

    elif x == "Grandmaster":

        return 4

    else:

        return -1

    



def encode_ord_2(x):

    if x == "Freezing":

        return 0

    elif x == "Cold":

        return 1

    elif x == "Warm":

        return 2

    elif x == "Hot":

        return 3

    elif x == "Boiling Hot":

        return 4

    elif x == "Lava Hot":

        return 5

    else:

        return -1



def encode_ord_3(x):

    if x == "a":

        return 0

    elif x == "b":

        return 1

    elif x == "c":

        return 2

    elif x == "d":

        return 3

    elif x == "e":

        return 4

    elif x == "f":

        return 5

    elif x == "g":

        return 6

    elif x == "h":

        return 7

    elif x == "i":

        return 8

    elif x == "j":

        return 9

    elif x == "k":

        return 10

    elif x == "l":

        return 11

    elif x == "m":

        return 12

    elif x == "n":

        return 13

    elif x == "o":

        return 14

    elif x == "p":

        return 15

    elif x == "q":

        return 16

    elif x == "r":

        return 17

    elif x == "s":

        return 18

    elif x == "t":

        return 19

    elif x == "u":

        return 20

    elif x == "v":

        return 21

    elif x == "w":

        return 22

    elif x == "x":

        return 23

    elif x == "y":

        return 24

    elif x == "z":

        return 25

    else:

        return -1



def encode_bin_3(x):

    if x == "T":

        return 1

    elif x == "F":

        return 0

    else:

        return -1



def encode_bin_4(x):

    if x == "Y":

        return 1

    elif x == "N":

        return 0

    else:

        return -1
X_train['ord_1'] = X_train.ord_1.apply(lambda x: encode_ord_1(x))

X_train['ord_2'] = X_train.ord_2.apply(lambda x: encode_ord_2(x))

X_train['ord_3'] = X_train.ord_3.apply(lambda x: encode_ord_3(x))

X_train['ord_4'] = X_train.ord_4.str.lower().apply(lambda x: encode_ord_3(x))

X_val['ord_1'] = X_val.ord_1.apply(lambda x: encode_ord_1(x))

X_val['ord_2'] = X_val.ord_2.apply(lambda x: encode_ord_2(x))

X_val['ord_3'] = X_val.ord_3.apply(lambda x: encode_ord_3(x))

X_val['ord_4'] = X_val.ord_4.str.lower().apply(lambda x: encode_ord_3(x))
# Label encoding to all columns

columns = X_train.columns



for i in columns:

    if X_train[i].dtype == object:        

        label_encoder = LabelEncoder()

        label_encoder.fit(X_train[i])

        X_train[i] = label_encoder.transform(X_train[i])

        X_val[i] = label_encoder.transform(X_val[i])



X_train.shape, X_val.shape
# Standardize the values

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)



# Measure performance on Validation dataset

logit = LogisticRegression()

logit.fit(X_train, y_train)

y_pred =  logit.predict_proba(X_val)



score = roc_auc_score(y_val, y_pred[:,1])

print("Experiment 1 model AUC Score: {}".format(score))



# Experiment 1 perform better than baseline model.

# Our efforts resulting improvement
# Train on all dataset and make submission



train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')



# Drop id on train and test dataset

train = train.drop('id', axis = 1)

test = test.drop('id', axis = 1)



# Select only baseline features

baseline_features = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'day', 'month', 'nom_0',

                     'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_7', 'nom_8',

                     'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']



train_target = train['target']

df_train = train[baseline_features]

df_test = test[baseline_features]



# Impute with Simple

columns = df_train.columns



for i in columns:

    

    if df_train[i].dtype == object:

        imputer = SimpleImputer(missing_values=np.nan, strategy= 'constant', add_indicator= True)

    else:

        imputer = SimpleImputer(missing_values=np.nan, strategy= 'constant', fill_value= -1, add_indicator= True)



    imputer.fit(df_train[i].to_numpy().reshape(-1,1))

    

    df_train[i] = imputer.transform(df_train[i].to_numpy().reshape(-1,1))

    df_test[i] = imputer.transform(df_test[i].to_numpy().reshape(-1,1))

    

# Ordinal Encoding

df_train['ord_1'] = df_train.ord_1.apply(lambda x: encode_ord_1(x))

df_train['ord_2'] = df_train.ord_2.apply(lambda x: encode_ord_2(x))

df_train['ord_3'] = df_train.ord_3.apply(lambda x: encode_ord_3(x))

df_train['ord_4'] = df_train.ord_4.str.lower().apply(lambda x: encode_ord_3(x))

df_test['ord_1'] = df_test.ord_1.apply(lambda x: encode_ord_1(x))

df_test['ord_2'] = df_test.ord_2.apply(lambda x: encode_ord_2(x))

df_test['ord_3'] = df_test.ord_3.apply(lambda x: encode_ord_3(x))

df_test['ord_4'] = df_test.ord_4.str.lower().apply(lambda x: encode_ord_3(x))



# Label Encoding to only object features

for i in columns:

    if df_train[i].dtype == object:        

        label_encoder = LabelEncoder()

        label_encoder.fit(df_train[i])

        df_train[i] = label_encoder.transform(df_train[i])

        df_test[i] = label_encoder.transform(df_test[i])

    

# Standardize the values

scaler = StandardScaler()

df_train = scaler.fit_transform(df_train)

df_test = scaler.transform(df_test)



# Training model

logit = LogisticRegression()

logit.fit(df_train, train_target)



# Predict

y_pred = logit.predict_proba(df_test)
# Make file for submission

exp1_submission = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')

exp1_submission['target'] = y_pred[:,1]

exp1_submission.to_csv('exp1_model.csv', index=False)
# Pearson correlation of each features to target

cor_mtx = pd.concat( [pd.DataFrame(df_train, columns = columns), train_target], axis = 1).corr()

plt.subplots()

sns.heatmap(cor_mtx)



print(cor_mtx['target'].sort_values(ascending=False))
# Read file

train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')



# Drop id on train and test dataset

train = train.drop('id', axis = 1)

test = test.drop('id', axis = 1)



# Select only baseline features

baseline_features = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'day', 'month', 'nom_0',

                     'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_7', 'nom_8',

                     'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']



train_target = train['target']

train = train[baseline_features]

test = test[baseline_features]



# Impute with constant

columns = train.columns



for i in columns:

    

    if train[i].dtype == object:

        imputer = SimpleImputer(missing_values=np.nan, strategy= 'constant', add_indicator= True)

    else:

        imputer = SimpleImputer(missing_values=np.nan, strategy= 'constant', fill_value= -1, add_indicator= True)



    imputer.fit(train[i].to_numpy().reshape(-1,1))

    

    train[i] = imputer.transform(train[i].to_numpy().reshape(-1,1))

    test[i] = imputer.transform(test[i].to_numpy().reshape(-1,1))



train.shape, train_target.shape, test.shape

# We have 600k samples for training and 400k for test
# Feature Engineering



# Create interactions

train['nom_1_nom_2'] = train.nom_1 + "_" + train.nom_2

train['nom_1_nom_3'] = train.nom_1 + "_" + train.nom_3

train['nom_1_nom_4'] = train.nom_1 + "_" + train.nom_4

train['nom_2_nom_3'] = train.nom_2 + "_" + train.nom_3

train['nom_2_nom_4'] = train.nom_2 + "_" + train.nom_4

train['nom_3_nom_4'] = train.nom_3 + "_" + train.nom_4

test['nom_1_nom_2'] = test.nom_1 + "_" + test.nom_2

test['nom_1_nom_3'] = test.nom_1 + "_" + test.nom_3

test['nom_1_nom_4'] = test.nom_1 + "_" + test.nom_4

test['nom_2_nom_3'] = test.nom_2 + "_" + test.nom_3

test['nom_2_nom_4'] = test.nom_2 + "_" + test.nom_4

test['nom_3_nom_4'] = test.nom_3 + "_" + test.nom_4



# Create cyclical features from day and month

train['day_sin7'] = np.sin(2*np.pi*train['day']/7)

train['day_sin14'] = np.sin(2*np.pi*train['day']/14)

train['day_sin14add'] = np.sin(2*np.pi*train['day']/14)*3.5

train['month_sin12'] = np.sin(2*np.pi*train['month']/12)

train['month_sin24'] = np.sin(2*np.pi*train['month']/24)

train['month_sin24_add'] = np.sin(2*np.pi*train['month']/24)*6

test['day_sin7'] = np.sin(2*np.pi*test['day']/7)

test['day_sin14'] = np.sin(2*np.pi*test['day']/14)

test['day_sin14add'] = np.sin(2*np.pi*test['day']/14)*3.5

test['month_sin12'] = np.sin(2*np.pi*test['month']/12)

test['month_sin24'] = np.sin(2*np.pi*test['month']/24)

test['month_sin24_add'] = np.sin(2*np.pi*test['month']/24)*6



# Ordinal Encoding

train['ord_1'] = train.ord_1.apply(lambda x: encode_ord_1(x))

train['ord_2'] = train.ord_2.apply(lambda x: encode_ord_2(x))

train['ord_3'] = train.ord_3.apply(lambda x: encode_ord_3(x))

train['ord_4'] = train.ord_4.str.lower().apply(lambda x: encode_ord_3(x))

test['ord_1'] = test.ord_1.apply(lambda x: encode_ord_1(x))

test['ord_2'] = test.ord_2.apply(lambda x: encode_ord_2(x))

test['ord_3'] = test.ord_3.apply(lambda x: encode_ord_3(x))

test['ord_4'] = test.ord_4.str.lower().apply(lambda x: encode_ord_3(x))
# Split train validation dataset

X_train, X_val, y_train, y_val = train_test_split(train,

                                                  train_target,

                                                  test_size = 0.2,

                                                  stratify = train_target,

                                                  random_state = 41)
# Label encoding to all columns

columns = X_train.columns



for i in columns:

    if X_train[i].dtype == object:        

        label_encoder = LabelEncoder()

        label_encoder.fit(X_train[i])

        X_train[i] = label_encoder.transform(X_train[i])

        X_val[i] = label_encoder.transform(X_val[i])



# Standardize the values

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)



# Measure performance on Validation dataset

logit = LogisticRegression()

logit.fit(X_train, y_train)

y_pred =  logit.predict_proba(X_val)



score = roc_auc_score(y_val, y_pred[:,1])

print("Experiment 2 model AUC Score: {}".format(score))



# Experiment 2 perform better than Experiment 1 model.

# Our efforts resulting improvement of (0.7305 - 0.7272) = 0.0033!
# Read file

train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')



# Drop id on train and test dataset

train = train.drop('id', axis = 1)

test = test.drop('id', axis = 1)



# Select only baseline features

baseline_features = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'day', 'month', 'nom_0',

                     'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_7', 'nom_8',

                     'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']



train_target = train['target']

train = train[baseline_features]

test = test[baseline_features]



# Impute with constant

columns = train.columns



for i in columns:

    

    if train[i].dtype == object:

        imputer = SimpleImputer(missing_values=np.nan, strategy= 'constant', add_indicator= True)

    else:

        imputer = SimpleImputer(missing_values=np.nan, strategy= 'constant', fill_value= -1, add_indicator= True)



    imputer.fit(train[i].to_numpy().reshape(-1,1))

    

    train[i] = imputer.transform(train[i].to_numpy().reshape(-1,1))

    test[i] = imputer.transform(test[i].to_numpy().reshape(-1,1))



# Feature Engineering



# Create interactions

train['nom_1_nom_2'] = train.nom_1 + "_" + train.nom_2

train['nom_1_nom_3'] = train.nom_1 + "_" + train.nom_3

train['nom_1_nom_4'] = train.nom_1 + "_" + train.nom_4

train['nom_2_nom_3'] = train.nom_2 + "_" + train.nom_3

train['nom_2_nom_4'] = train.nom_2 + "_" + train.nom_4

train['nom_3_nom_4'] = train.nom_3 + "_" + train.nom_4

test['nom_1_nom_2'] = test.nom_1 + "_" + test.nom_2

test['nom_1_nom_3'] = test.nom_1 + "_" + test.nom_3

test['nom_1_nom_4'] = test.nom_1 + "_" + test.nom_4

test['nom_2_nom_3'] = test.nom_2 + "_" + test.nom_3

test['nom_2_nom_4'] = test.nom_2 + "_" + test.nom_4

test['nom_3_nom_4'] = test.nom_3 + "_" + test.nom_4



# Create cyclical features from day and month

train['day_sin7'] = np.sin(2*np.pi*train['day']/7)

train['day_sin14'] = np.sin(2*np.pi*train['day']/14)

train['day_sin14add'] = np.sin(2*np.pi*train['day']/14)*3.5

train['month_sin12'] = np.sin(2*np.pi*train['month']/12)

train['month_sin24'] = np.sin(2*np.pi*train['month']/24)

train['month_sin24_add'] = np.sin(2*np.pi*train['month']/24)*6

test['day_sin7'] = np.sin(2*np.pi*test['day']/7)

test['day_sin14'] = np.sin(2*np.pi*test['day']/14)

test['day_sin14add'] = np.sin(2*np.pi*test['day']/14)*3.5

test['month_sin12'] = np.sin(2*np.pi*test['month']/12)

test['month_sin24'] = np.sin(2*np.pi*test['month']/24)

test['month_sin24_add'] = np.sin(2*np.pi*test['month']/24)*6



# Ordinal Encoding

train['ord_1'] = train.ord_1.apply(lambda x: encode_ord_1(x))

train['ord_2'] = train.ord_2.apply(lambda x: encode_ord_2(x))

train['ord_3'] = train.ord_3.apply(lambda x: encode_ord_3(x))

train['ord_4'] = train.ord_4.str.lower().apply(lambda x: encode_ord_3(x))

test['ord_1'] = test.ord_1.apply(lambda x: encode_ord_1(x))

test['ord_2'] = test.ord_2.apply(lambda x: encode_ord_2(x))

test['ord_3'] = test.ord_3.apply(lambda x: encode_ord_3(x))

test['ord_4'] = test.ord_4.str.lower().apply(lambda x: encode_ord_3(x))



# Update columns

columns = train.columns



# Label Encoding to only object features

for i in columns:

    if train[i].dtype == object:        

        label_encoder = LabelEncoder()

        label_encoder.fit(train[i])

        train[i] = label_encoder.transform(train[i])

        test[i] = label_encoder.transform(test[i])

    

# Standardize the values

scaler = StandardScaler()

train = scaler.fit_transform(train)

test = scaler.transform(test)



# Training model

logit = LogisticRegression()

logit.fit(train, train_target)



# Predict

y_pred = logit.predict_proba(test)
# Make file for submission

exp2_submission = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')

exp2_submission['target'] = y_pred[:,1]

exp2_submission.to_csv('exp2_model.csv', index=False)
cor_mtx = pd.concat([pd.DataFrame(train, columns = columns), train_target], axis = 1).corr()

sns.heatmap(cor_mtx)



print(cor_mtx['target'].sort_values(ascending = False))
# Read file

train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')



# Drop id on train and test dataset

train = train.drop('id', axis = 1)

test = test.drop('id', axis = 1)



# Select only baseline features

exp3_features = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'day', 'month', 'nom_0',

                     'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',

                     'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']



train_target = train['target']

train = train[exp3_features]

test = test[exp3_features]



# Replacing nom_6 value 'a885aacec' in test_dataset with 'missing_value' because the value is not seen at training data

test.loc[test.nom_6 == "a885aacec", 'nom_6'] = "missing_value"





# Label encode first bin_3 and bin_4

train['bin_3'] = train.bin_3.apply(lambda x: encode_bin_3(x))

train['bin_4'] = train.bin_4.apply(lambda x: encode_bin_4(x))

test['bin_3'] = test.bin_3.apply(lambda x: encode_bin_3(x))

test['bin_4'] = test.bin_4.apply(lambda x: encode_bin_4(x))



# Impute with constant

columns = train.columns



for i in columns:

    

    if train[i].dtype == object:

        imputer = SimpleImputer(missing_values=np.nan, strategy= 'constant', add_indicator= True)

    else:

        imputer = SimpleImputer(missing_values=np.nan, strategy= 'constant', fill_value= -1, add_indicator= True)



    imputer.fit(train[i].to_numpy().reshape(-1,1))

    

    train[i] = imputer.transform(train[i].to_numpy().reshape(-1,1))

    test[i] = imputer.transform(test[i].to_numpy().reshape(-1,1))



# Feature Engineering

# Create interactions

train['nom_1_nom_2'] = train.nom_1 + "_" + train.nom_2

train['nom_1_nom_3'] = train.nom_1 + "_" + train.nom_3

train['nom_1_nom_4'] = train.nom_1 + "_" + train.nom_4

train['nom_2_nom_3'] = train.nom_2 + "_" + train.nom_3

train['nom_2_nom_4'] = train.nom_2 + "_" + train.nom_4

train['nom_3_nom_4'] = train.nom_3 + "_" + train.nom_4

test['nom_1_nom_2'] = test.nom_1 + "_" + test.nom_2

test['nom_1_nom_3'] = test.nom_1 + "_" + test.nom_3

test['nom_1_nom_4'] = test.nom_1 + "_" + test.nom_4

test['nom_2_nom_3'] = test.nom_2 + "_" + test.nom_3

test['nom_2_nom_4'] = test.nom_2 + "_" + test.nom_4

test['nom_3_nom_4'] = test.nom_3 + "_" + test.nom_4



# Interaction Exp 3

train['bin_all_sum'] = train.bin_0 + train.bin_1 + train.bin_2 + train.bin_3 + train.bin_4

train['bin_all_mul'] = train.bin_0 * train.bin_1 * train.bin_2 * train.bin_3 * train.bin_4

test['bin_all_sum'] = test.bin_0 + test.bin_1 + test.bin_2 + test.bin_3 + test.bin_4

test['bin_all_mul'] = test.bin_0 * test.bin_1 * test.bin_2 * test.bin_3 * test.bin_4



# Create cyclical features from day and month

train['day_sin7'] = np.sin(2*np.pi*train['day']/7)

train['day_sin14'] = np.sin(2*np.pi*train['day']/14)

train['day_sin14add'] = np.sin(2*np.pi*train['day']/14)*3.5

train['month_sin12'] = np.sin(2*np.pi*train['month']/12)

train['month_sin24'] = np.sin(2*np.pi*train['month']/24)

train['month_sin24_add'] = np.sin(2*np.pi*train['month']/24)*6

test['day_sin7'] = np.sin(2*np.pi*test['day']/7)

test['day_sin14'] = np.sin(2*np.pi*test['day']/14)

test['day_sin14add'] = np.sin(2*np.pi*test['day']/14)*3.5

test['month_sin12'] = np.sin(2*np.pi*test['month']/12)

test['month_sin24'] = np.sin(2*np.pi*test['month']/24)

test['month_sin24_add'] = np.sin(2*np.pi*test['month']/24)*6



# Ordinal Encoding

train['ord_1'] = train.ord_1.apply(lambda x: encode_ord_1(x))

train['ord_2'] = train.ord_2.apply(lambda x: encode_ord_2(x))

train['ord_3'] = train.ord_3.apply(lambda x: encode_ord_3(x))

train['ord_4'] = train.ord_4.str.lower().apply(lambda x: encode_ord_3(x))

test['ord_1'] = test.ord_1.apply(lambda x: encode_ord_1(x))

test['ord_2'] = test.ord_2.apply(lambda x: encode_ord_2(x))

test['ord_3'] = test.ord_3.apply(lambda x: encode_ord_3(x))

test['ord_4'] = test.ord_4.str.lower().apply(lambda x: encode_ord_3(x))



# Ordinal Encoding Square

# First normalize with maximum label for faster convergence,

# Subtract with 0.5 and square it

train['ord_1_sqr_mid'] = ((train.ord_1 / 4) - 0.5)**2

train['ord_2_sqr_mid'] = ((train.ord_2 / 5) - 0.5)**2

train['ord_3_sqr_mid'] = ((train.ord_3 / 25) - 0.5)**2

train['ord_4_sqr_mid'] = ((train.ord_4 / 25) - 0.5)**2

test['ord_1_sqr_mid'] = ((test.ord_1 / 4) - 0.5)**2

test['ord_2_sqr_mid'] = ((test.ord_2 / 5) - 0.5)**2

test['ord_3_sqr_mid'] = ((test.ord_3 / 25) - 0.5)**2

test['ord_4_sqr_mid'] = ((test.ord_4 / 25) - 0.5)**2

# Square bot not centered

train['ord_1_sqr'] = ((train.ord_1 / 4))**2

train['ord_2_sqr'] = ((train.ord_2 / 5))**2

train['ord_3_sqr'] = ((train.ord_3 / 25))**2

train['ord_4_sqr'] = ((train.ord_4 / 25))**2

test['ord_1_sqr'] = ((test.ord_1 / 4))**2

test['ord_2_sqr'] = ((test.ord_2 / 5))**2

test['ord_3_sqr'] = ((test.ord_3 / 25))**2

test['ord_4_sqr'] = ((test.ord_4 / 25))**2

# Log Transform

train['ord_1_log'] = np.log1p((train.ord_1 / 4))

train['ord_2_log'] = np.log1p((train.ord_2 / 5))

train['ord_3_log'] = np.log1p((train.ord_3 / 25))

train['ord_4_log'] = np.log1p((train.ord_4 / 25))

test['ord_1_log'] = np.log1p((test.ord_1 / 4))

test['ord_2_log'] = np.log1p((test.ord_2 / 5))

test['ord_3_log'] = np.log1p((test.ord_3 / 25))

test['ord_4_log'] = np.log1p((test.ord_4 / 25))
# Splitting train data for validation

X_train, X_val, y_train, y_val = train_test_split(train,

                                                 train_target,

                                                 test_size = 0.2,

                                                 stratify = train_target,

                                                 random_state = 41)
# Label encoding to all columns

columns = X_train.columns



for i in columns:

    if X_train[i].dtype == object:        

        label_encoder = LabelEncoder()

        X_train[i] = label_encoder.fit_transform(X_train[i])

        label_encoder = LabelEncoder()

        X_val[i] = label_encoder.fit_transform(X_val[i])

        

# Create ordinal square after label encoding - centered

X_train['ord_5_sqr_mid'] = ((X_train.ord_5 / X_train.ord_5.max()) - 0.5)**2

X_val['ord_5_sqr_mid'] = ((X_val.ord_5 / X_train.ord_5.max()) - 0.5)**2

# Create ordinal square after label encoding - not centered

X_train['ord_5_sqr'] = ((X_train.ord_5 / X_train.ord_5.max()) )**2

X_val['ord_5_sqr'] = ((X_val.ord_5 / X_train.ord_5.max()) )**2

# Log transform

X_train['ord_5_log'] = np.log1p((X_train.ord_5 / X_train.ord_5.max()))

X_val['ord_5_log'] = np.log1p((X_val.ord_5 / X_train.ord_5.max()))



# Standardize the values

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_val = scaler.transform(X_val)



# Measure performance on Validation dataset

logit = LogisticRegression()

logit.fit(X_train, y_train)

y_pred =  logit.predict_proba(X_val)



score = roc_auc_score(y_val, y_pred[:,1])

print("Experiment 3 model AUC Score: {}".format(score))



# Experiment 3 perform better than Experiment 2 model.
# Read file

train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')



# Drop id on train and test dataset

train = train.drop('id', axis = 1)

test = test.drop('id', axis = 1)



# Select only baseline features

exp3_features = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'day', 'month', 'nom_0',

                     'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',

                     'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']



train_target = train['target']

train = train[exp3_features]

test = test[exp3_features]



# Replacing nom_6 value 'a885aacec' in test_dataset with 'missing_value' because the value is not seen at training data

test.loc[test.nom_6 == "a885aacec", 'nom_6'] = "missing_value"



# Label encode first bin_3 and bin_4

train['bin_3'] = train.bin_3.apply(lambda x: encode_bin_3(x))

train['bin_4'] = train.bin_4.apply(lambda x: encode_bin_4(x))

test['bin_3'] = test.bin_3.apply(lambda x: encode_bin_3(x))

test['bin_4'] = test.bin_4.apply(lambda x: encode_bin_4(x))



# Impute with constant

columns = train.columns



for i in columns:

    

    if train[i].dtype == object:

        imputer = SimpleImputer(missing_values=np.nan, strategy= 'constant', add_indicator= True)

    else:

        imputer = SimpleImputer(missing_values=np.nan, strategy= 'constant', fill_value= -1, add_indicator= True)



    imputer.fit(train[i].to_numpy().reshape(-1,1))

    

    train[i] = imputer.transform(train[i].to_numpy().reshape(-1,1))

    test[i] = imputer.transform(test[i].to_numpy().reshape(-1,1))



# Feature Engineering

# Create interactions

train['nom_1_nom_2'] = train.nom_1 + "_" + train.nom_2

train['nom_1_nom_3'] = train.nom_1 + "_" + train.nom_3

train['nom_1_nom_4'] = train.nom_1 + "_" + train.nom_4

train['nom_2_nom_3'] = train.nom_2 + "_" + train.nom_3

train['nom_2_nom_4'] = train.nom_2 + "_" + train.nom_4

train['nom_3_nom_4'] = train.nom_3 + "_" + train.nom_4

test['nom_1_nom_2'] = test.nom_1 + "_" + test.nom_2

test['nom_1_nom_3'] = test.nom_1 + "_" + test.nom_3

test['nom_1_nom_4'] = test.nom_1 + "_" + test.nom_4

test['nom_2_nom_3'] = test.nom_2 + "_" + test.nom_3

test['nom_2_nom_4'] = test.nom_2 + "_" + test.nom_4

test['nom_3_nom_4'] = test.nom_3 + "_" + test.nom_4



# Interaction Exp 3

train['bin_all_sum'] = train.bin_0 + train.bin_1 + train.bin_2 + train.bin_3 + train.bin_4

train['bin_all_mul'] = train.bin_0 * train.bin_1 * train.bin_2 * train.bin_3 * train.bin_4

test['bin_all_sum'] = test.bin_0 + test.bin_1 + test.bin_2 + test.bin_3 + test.bin_4

test['bin_all_mul'] = test.bin_0 * test.bin_1 * test.bin_2 * test.bin_3 * test.bin_4



# Create cyclical features from day and month

train['day_sin7'] = np.sin(2*np.pi*train['day']/7)

train['day_sin14'] = np.sin(2*np.pi*train['day']/14)

train['day_sin14add'] = np.sin(2*np.pi*train['day']/14)*3.5

train['month_sin12'] = np.sin(2*np.pi*train['month']/12)

train['month_sin24'] = np.sin(2*np.pi*train['month']/24)

train['month_sin24_add'] = np.sin(2*np.pi*train['month']/24)*6

test['day_sin7'] = np.sin(2*np.pi*test['day']/7)

test['day_sin14'] = np.sin(2*np.pi*test['day']/14)

test['day_sin14add'] = np.sin(2*np.pi*test['day']/14)*3.5

test['month_sin12'] = np.sin(2*np.pi*test['month']/12)

test['month_sin24'] = np.sin(2*np.pi*test['month']/24)

test['month_sin24_add'] = np.sin(2*np.pi*test['month']/24)*6



# Ordinal Encoding

train['ord_1'] = train.ord_1.apply(lambda x: encode_ord_1(x))

train['ord_2'] = train.ord_2.apply(lambda x: encode_ord_2(x))

train['ord_3'] = train.ord_3.apply(lambda x: encode_ord_3(x))

train['ord_4'] = train.ord_4.str.lower().apply(lambda x: encode_ord_3(x))

test['ord_1'] = test.ord_1.apply(lambda x: encode_ord_1(x))

test['ord_2'] = test.ord_2.apply(lambda x: encode_ord_2(x))

test['ord_3'] = test.ord_3.apply(lambda x: encode_ord_3(x))

test['ord_4'] = test.ord_4.str.lower().apply(lambda x: encode_ord_3(x))



# Ordinal Encoding Square

# First normalize with maximum label for faster convergence,

# Subtract with 0.5 and square it

train['ord_1_sqr_mid'] = ((train.ord_1 / 4) - 0.5)**2

train['ord_2_sqr_mid'] = ((train.ord_2 / 5) - 0.5)**2

train['ord_3_sqr_mid'] = ((train.ord_3 / 25) - 0.5)**2

train['ord_4_sqr_mid'] = ((train.ord_4 / 25) - 0.5)**2

test['ord_1_sqr_mid'] = ((test.ord_1 / 4) - 0.5)**2

test['ord_2_sqr_mid'] = ((test.ord_2 / 5) - 0.5)**2

test['ord_3_sqr_mid'] = ((test.ord_3 / 25) - 0.5)**2

test['ord_4_sqr_mid'] = ((test.ord_4 / 25) - 0.5)**2

# Square bot not centered

train['ord_1_sqr'] = ((train.ord_1 / 4))**2

train['ord_2_sqr'] = ((train.ord_2 / 5))**2

train['ord_3_sqr'] = ((train.ord_3 / 25))**2

train['ord_4_sqr'] = ((train.ord_4 / 25))**2

test['ord_1_sqr'] = ((test.ord_1 / 4))**2

test['ord_2_sqr'] = ((test.ord_2 / 5))**2

test['ord_3_sqr'] = ((test.ord_3 / 25))**2

test['ord_4_sqr'] = ((test.ord_4 / 25))**2

# Log Transform

train['ord_1_log'] = np.log1p((train.ord_1 / 4))

train['ord_2_log'] = np.log1p((train.ord_2 / 5))

train['ord_3_log'] = np.log1p((train.ord_3 / 25))

train['ord_4_log'] = np.log1p((train.ord_4 / 25))

test['ord_1_log'] = np.log1p((test.ord_1 / 4))

test['ord_2_log'] = np.log1p((test.ord_2 / 5))

test['ord_3_log'] = np.log1p((test.ord_3 / 25))

test['ord_4_log'] = np.log1p((test.ord_4 / 25))



# Update columns

columns = train.columns



# Label Encoding to only object features

for i in columns:

    if train[i].dtype == object:        

        label_encoder = LabelEncoder()

        label_encoder.fit(train[i])

        train[i] = label_encoder.transform(train[i])

        test[i] = label_encoder.transform(test[i])



# Create ordinal square after label encoding - centered

train['ord_5_sqr_mid'] = ((train.ord_5 / train.ord_5.max()) - 0.5)**2

test['ord_5_sqr_mid'] = ((test.ord_5 / train.ord_5.max()) - 0.5)**2

# Create ordinal square after label encoding - not centered

train['ord_5_sqr'] = ((train.ord_5 / train.ord_5.max()) )**2

test['ord_5_sqr'] = ((test.ord_5 / train.ord_5.max()) )**2

# Log transform

train['ord_5_log'] = np.log1p((train.ord_5 / train.ord_5.max()))

test['ord_5_log'] = np.log1p((test.ord_5 / train.ord_5.max()))



# Update columns

columns = train.columns



# Standardize the values

scaler = StandardScaler()

train = scaler.fit_transform(train)

test = scaler.transform(test)



# Training model

logit = LogisticRegression()

logit.fit(train, train_target)



# Predict

y_pred = logit.predict_proba(test)
# Make file for submission

exp3_submission = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')

exp3_submission['target'] = y_pred[:,1]

exp3_submission.to_csv('exp3_model.csv', index=False)
cor_mtx = pd.concat([pd.DataFrame(train, columns = columns) ,train_target], axis = 1).corr()

sns.heatmap(cor_mtx)



print(cor_mtx['target'].sort_values(ascending=False))
# Read file

train = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')



# Drop id on train and test dataset

train = train.drop('id', axis = 1)

test = test.drop('id', axis = 1)



# Select only baseline features

exp3_features = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'day', 'month', 'nom_0',

                     'nom_1', 'nom_2', 'nom_3', 'nom_4', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9',

                     'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']



train_target = train['target']

train = train[exp3_features]

test = test[exp3_features]



# Replacing nom_6 value 'a885aacec' in test_dataset with 'missing_value' because the value is not seen at training data

test.loc[test.nom_6 == "a885aacec", 'nom_6'] = "missing_value"



# Label encode first bin_3 and bin_4

train['bin_3'] = train.bin_3.apply(lambda x: encode_bin_3(x))

train['bin_4'] = train.bin_4.apply(lambda x: encode_bin_4(x))

test['bin_3'] = test.bin_3.apply(lambda x: encode_bin_3(x))

test['bin_4'] = test.bin_4.apply(lambda x: encode_bin_4(x))



# Impute with constant

columns = train.columns



for i in columns:

    

    if train[i].dtype == object:

        imputer = SimpleImputer(missing_values=np.nan, strategy= 'constant', add_indicator= True)

    else:

        imputer = SimpleImputer(missing_values=np.nan, strategy= 'constant', fill_value= -1, add_indicator= True)



    imputer.fit(train[i].to_numpy().reshape(-1,1))

    

    train[i] = imputer.transform(train[i].to_numpy().reshape(-1,1))

    test[i] = imputer.transform(test[i].to_numpy().reshape(-1,1))
# Feature Engineering



# define one-hot columns

onehot_columns = train.dtypes[train.dtypes == object].index.to_list()



# See number of samples on training data

num_train = len(train)



df_all = train.append(test)



one_hot = pd.get_dummies(

    df_all[onehot_columns],

    columns=onehot_columns,

    drop_first=True,

    dummy_na=True,

    sparse=True,

    dtype="int8",

 # ).to_numpy()

).sparse.to_coo().toarray()



one_hot_train = one_hot[:num_train]

one_hot_test = one_hot[num_train:]



one_hot_train = scipy.sparse.coo_matrix(one_hot_train)

one_hot_test = scipy.sparse.coo_matrix(one_hot_test)
# Create interactions

train['nom_1_nom_2'] = train.nom_1 + "_" + train.nom_2

train['nom_1_nom_3'] = train.nom_1 + "_" + train.nom_3

train['nom_1_nom_4'] = train.nom_1 + "_" + train.nom_4

train['nom_2_nom_3'] = train.nom_2 + "_" + train.nom_3

train['nom_2_nom_4'] = train.nom_2 + "_" + train.nom_4

train['nom_3_nom_4'] = train.nom_3 + "_" + train.nom_4

test['nom_1_nom_2'] = test.nom_1 + "_" + test.nom_2

test['nom_1_nom_3'] = test.nom_1 + "_" + test.nom_3

test['nom_1_nom_4'] = test.nom_1 + "_" + test.nom_4

test['nom_2_nom_3'] = test.nom_2 + "_" + test.nom_3

test['nom_2_nom_4'] = test.nom_2 + "_" + test.nom_4

test['nom_3_nom_4'] = test.nom_3 + "_" + test.nom_4



# Interaction Exp 3

train['bin_all_sum'] = train.bin_0 + train.bin_1 + train.bin_2 + train.bin_3 + train.bin_4

train['bin_all_mul'] = train.bin_0 * train.bin_1 * train.bin_2 * train.bin_3 * train.bin_4

test['bin_all_sum'] = test.bin_0 + test.bin_1 + test.bin_2 + test.bin_3 + test.bin_4

test['bin_all_mul'] = test.bin_0 * test.bin_1 * test.bin_2 * test.bin_3 * test.bin_4



# Create cyclical features from day and month

train['day_sin7'] = np.sin(2*np.pi*train['day']/7)

train['day_sin14'] = np.sin(2*np.pi*train['day']/14)

train['day_sin14add'] = np.sin(2*np.pi*train['day']/14)*3.5

train['month_sin12'] = np.sin(2*np.pi*train['month']/12)

train['month_sin24'] = np.sin(2*np.pi*train['month']/24)

train['month_sin24_add'] = np.sin(2*np.pi*train['month']/24)*6

test['day_sin7'] = np.sin(2*np.pi*test['day']/7)

test['day_sin14'] = np.sin(2*np.pi*test['day']/14)

test['day_sin14add'] = np.sin(2*np.pi*test['day']/14)*3.5

test['month_sin12'] = np.sin(2*np.pi*test['month']/12)

test['month_sin24'] = np.sin(2*np.pi*test['month']/24)

test['month_sin24_add'] = np.sin(2*np.pi*test['month']/24)*6



# Ordinal Encoding

train['ord_1'] = train.ord_1.apply(lambda x: encode_ord_1(x))

train['ord_2'] = train.ord_2.apply(lambda x: encode_ord_2(x))

train['ord_3'] = train.ord_3.apply(lambda x: encode_ord_3(x))

train['ord_4'] = train.ord_4.str.lower().apply(lambda x: encode_ord_3(x))

test['ord_1'] = test.ord_1.apply(lambda x: encode_ord_1(x))

test['ord_2'] = test.ord_2.apply(lambda x: encode_ord_2(x))

test['ord_3'] = test.ord_3.apply(lambda x: encode_ord_3(x))

test['ord_4'] = test.ord_4.str.lower().apply(lambda x: encode_ord_3(x))



# Ordinal Encoding Square

# First normalize with maximum label for faster convergence,

# Subtract with 0.5 and square it

train['ord_1_sqr_mid'] = ((train.ord_1 / 4) - 0.5)**2

train['ord_2_sqr_mid'] = ((train.ord_2 / 5) - 0.5)**2

train['ord_3_sqr_mid'] = ((train.ord_3 / 25) - 0.5)**2

train['ord_4_sqr_mid'] = ((train.ord_4 / 25) - 0.5)**2

test['ord_1_sqr_mid'] = ((test.ord_1 / 4) - 0.5)**2

test['ord_2_sqr_mid'] = ((test.ord_2 / 5) - 0.5)**2

test['ord_3_sqr_mid'] = ((test.ord_3 / 25) - 0.5)**2

test['ord_4_sqr_mid'] = ((test.ord_4 / 25) - 0.5)**2

# Square bot not centered

train['ord_1_sqr'] = ((train.ord_1 / 4))**2

train['ord_2_sqr'] = ((train.ord_2 / 5))**2

train['ord_3_sqr'] = ((train.ord_3 / 25))**2

train['ord_4_sqr'] = ((train.ord_4 / 25))**2

test['ord_1_sqr'] = ((test.ord_1 / 4))**2

test['ord_2_sqr'] = ((test.ord_2 / 5))**2

test['ord_3_sqr'] = ((test.ord_3 / 25))**2

test['ord_4_sqr'] = ((test.ord_4 / 25))**2

# Log Transform

train['ord_1_log'] = np.log1p((train.ord_1 / 4))

train['ord_2_log'] = np.log1p((train.ord_2 / 5))

train['ord_3_log'] = np.log1p((train.ord_3 / 25))

train['ord_4_log'] = np.log1p((train.ord_4 / 25))

test['ord_1_log'] = np.log1p((test.ord_1 / 4))

test['ord_2_log'] = np.log1p((test.ord_2 / 5))

test['ord_3_log'] = np.log1p((test.ord_3 / 25))

test['ord_4_log'] = np.log1p((test.ord_4 / 25))



# Update columns

columns = train.columns



# Target Encoding to only object features

for i in columns:

    if train[i].dtype == object:

        target_encoder = cat_encoder.TargetEncoder(smoothing = 0.1)

        target_encoder.fit(train[i], train_target)

        train[i+"_target"] = target_encoder.transform(train[i])

        test[i+"_target"] = target_encoder.transform(test[i])

        

# Update columns

columns = train.columns



# Label Encoding to only object features

for i in columns:

    if train[i].dtype == object:        

        label_encoder = LabelEncoder()

        label_encoder.fit(train[i])

        train[i] = label_encoder.transform(train[i])

        test[i] = label_encoder.transform(test[i])



# Create ordinal square after label encoding - centered

train['ord_5_sqr_mid'] = ((train.ord_5 / train.ord_5.max()) - 0.5)**2

test['ord_5_sqr_mid'] = ((test.ord_5 / train.ord_5.max()) - 0.5)**2

# Create ordinal square after label encoding - not centered

train['ord_5_sqr'] = ((train.ord_5 / train.ord_5.max()) )**2

test['ord_5_sqr'] = ((test.ord_5 / train.ord_5.max()) )**2

# Log transform

train['ord_5_log'] = np.log1p((train.ord_5 / train.ord_5.max()))

test['ord_5_log'] = np.log1p((test.ord_5 / train.ord_5.max()))



# Update columns

columns = train.columns



# Standardize the values

scaler = StandardScaler()

train = scaler.fit_transform(train)

test = scaler.transform(test)
# Horizontal stack array result from one-hot encoding

train = scipy.sparse.hstack(([one_hot_train, train])).tocsr()

test = scipy.sparse.hstack(([one_hot_test, test])).tocsr()



# Training model

logit = LogisticRegression()

logit.fit(train, train_target)



# Predict

y_pred = logit.predict_proba(test)
# Make file for submission

exp4_submission = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/sample_submission.csv')

exp4_submission['target'] = y_pred[:,1]

exp4_submission.to_csv('exp4_model.csv', index=False)



# Experimentation 4 result better than experiment 3 on public leaderboard. It achieves score 0.77515 using plain LogisticRegression