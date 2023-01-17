import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

df = pd.read_csv('../input/accepted_2007_to_2017Q3.csv.gz', compression='gzip', low_memory=True)

# remove columns that have only one distinct value
drop_list = []
for col in df.columns:
    if df[col].nunique() == 1:
        drop_list.append(col)

drop_list
#drop columns
df.drop(labels=drop_list, axis=1, inplace=True)
# remove columns that have values for less than 10% rows
drop_list = []
for col in df.columns:
    if df[col].notnull().sum() / df.shape[0] < 0.1:
        drop_list.append(col)

drop_list
df.drop(labels=drop_list, axis=1, inplace=True)
# remove irrelevant features
df.drop(labels=['dti', 'id', 'emp_title', 'title', 'issue_d', 'funded_amnt_inv','out_prncp_inv', 'last_credit_pull_d', 'earliest_cr_line','fico_range_low','last_fico_range_low','next_pymnt_d','disbursement_method'], axis=1, inplace=True)

###Some features give away the loan status. Remove these columns
df.drop(labels=['collection_recovery_fee', 'debt_settlement_flag', 'last_pymnt_amnt', 'last_pymnt_d', 'recoveries', 
                 'total_pymnt', 'total_pymnt_inv', 'total_rec_int', 'total_rec_late_fee', 'total_rec_prncp'], axis=1, inplace=True)

#columns with text data
text_cols = []
for col in df.columns:
    if df[col].dtype == np.object:
        text_cols.append(col)

text_cols
#Remove rows without term
df.dropna(subset=["term"], how='all', inplace=True)

#Remove rows with bLANK Ssub grade
df.dropna(subset=["sub_grade"], how='all', inplace=True)
#Convert term into integer
floatval = lambda s: np.float(s[1:3])
df['term'] = df['term'].apply(floatval) 

#populate -1 for mths_since_last_delinq and mths_since_last_record, mths_since_last_major_derog if value is null

df['mths_since_last_delinq'] = df['mths_since_last_delinq'].fillna(-1)
df['mths_since_last_record'] = df['mths_since_last_record'].fillna(-1)
df['mths_since_last_major_derog'] = df['mths_since_last_major_derog'].fillna(-1)
## convert subgrade
grade_dict = {'A':0.0, 'B':1.0, 'C':2.0, 'D':3.0, 'E':4.0, 'F':5.0, 'G':6.0}
def grade_to_float(s):
    return 5 * grade_dict[s[0]]  + np.float(s[1]) - 1
df['sub_grade'] = df['sub_grade'].apply(lambda s: grade_to_float(s))

#grade is implied by sub grade
df.drop(labels=['grade'], axis=1, inplace=True)
#convert employment length to float
def emp_conv(s):
    try:
        if pd.isnull(s):
            return s
        elif s[0] == '<':
            return 0.0
        elif s[:2] == '10':
            return 10.0
        else:
            return np.float(s[0])
    except TypeError:
        return np.float64(s)

df['emp_length'] = df['emp_length'].apply(lambda s: emp_conv(s))
df['emp_length'].value_counts()
#Convert zip code to float and drop state 
df.dropna(subset=["zip_code"], how='all', inplace=True)
df['zip_code'] = df['zip_code'].apply(lambda s:np.float(s[:3]))
df.drop(labels='addr_state', axis=1, inplace=True)
#convert the 'loan_status' column to a 0/1 'charged_off' column. 
df['loan_status'] = df['loan_status'].apply(lambda s: np.float(s == 'Charged Off'))
#rename loan status column
df.rename(columns={'loan_status':'charged_off'}, inplace=True)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#convert categorical features into dummy variable
cat_feats = []
for col in df.columns:
    if df[col].dtype == np.object:
        cat_feats.append(col)

cat_feats
df = pd.get_dummies(df, columns=cat_feats, drop_first=True)
#train and validation data

X = df.drop(labels=['charged_off'], axis=1) # Features
Y = df['charged_off'] # Target variable
# completion of data i.e. how much percentage of rows are null for each feature
pd.DataFrame((X.notnull().sum() / X.shape[0]).sort_values(), columns=['Fraction not null'])
#Impute missing data with mean, median, or constant value
# fill missing values with mean column values
X.fillna(X.mean(), inplace=True)
## feature scaling - Mean
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)
X = pd.DataFrame(scaler.transform(X), columns=X.columns)
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
from keras import layers, regularizers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, Sequential
import keras.backend as K
from keras.optimizers import SGD
#custom matrix
def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)
#Neural Network
model = Sequential()

model.add(Dense(3,input_dim=X.shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

##fine tune stochastic gradient descent optimizer parameterse
sgd = SGD(lr=0.001, momentum=0.8, decay=0.0, nesterov=False)
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy',  recall, precision])
history = model.fit(X, Y, validation_split=0.2, epochs=6, batch_size=256)


# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

