# Importing necessary modules

import numpy as np

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.callbacks import ModelCheckpoint

from keras.models import Sequential

from keras.layers import Dense

from matplotlib import pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')

warnings.filterwarnings('ignore', category=DeprecationWarning)

warnings.filterwarnings('ignore', category=FutureWarning)



pd.options.mode.chained_assignment = None
train = pd.read_csv('../input/loan-data/Loan_Training_data.csv')

test = pd.read_csv('../input/loan-data/Loan_Test_Data.csv')
train.head()
train.Loan_Status.value_counts()
train.isnull().sum()
sns.barplot(train.Gender, train.LoanAmount, hue=train.Loan_Status)

plt.legend(loc='upper right')

plt.title('Loan Amount vs Gender - grouped based on Loan Status')
# Changing categorical names for the sake of easier understanding

train.Married = train.Married.apply(lambda x: 'Married' if x == 'Yes' else 'Unmarried')



sns.barplot(train.Gender, train.LoanAmount, hue=train.Married, hue_order = ['Married', 'Unmarried'])

plt.legend(loc='upper left')

plt.title('Loan Amount vs Gender - grouped based on Marital Status')
sns.scatterplot(train.ApplicantIncome, train.LoanAmount, hue=train.Loan_Status)

plt.xticks(rotation=45)
sns.countplot(train.Education, hue=train.Loan_Status)
sns.countplot(train.Property_Area, hue=train.Loan_Status)
sns.pairplot(train, hue='Loan_Status', palette='Set2', diag_kind='kde')
def target_split(train): 

    train_mod = train[~train['LoanAmount'].isnull()] # Loan Amount has few null values but they should not be imputed

    train_mod.drop('Loan_ID', axis = 1, inplace=True) # Dropping ID column as it is not relevant to the model



    y = train_mod.Loan_Status

    train_mod.drop('Loan_Status', axis = 1, inplace=True)



    y = y.apply(lambda x: 1 if x == 'Y' else 0) # Changing categories to numerical values

    

    return train_mod, y
def impute(train):

    cols = train.columns

    nan_cols = []

    for col in cols:

        if(train[col].isnull().sum() > 0):

            nan_cols.append(col)

    # nan_cols contains the list of columns having null values

    

    argmax_in_nan = {}

    for col in nan_cols:

        argmax_in_nan[col] = None

        argmax_in_nan[col] = train[col].value_counts().idxmax() # Getting the most frequent value in the column

        

        train[col].fillna(argmax_in_nan[col], inplace=True)

            

    return train
def scaler(train):

    num_cols = [col for col in train.select_dtypes(exclude='object').columns]

    scaler = MinMaxScaler()

    for col in num_cols:

        if (col != 'Credit_History'): # Credit_History belongs to int64 datatype but it is a categorical value. So it should not be scaled.

            train[col] = scaler.fit_transform(train[[col]])

            

    return train
def cat_enc(train):

    cat_cols = [col for col in train.select_dtypes(include='object').columns]

    

    for col in cat_cols:

        dummies = pd.get_dummies(train[col], prefix=col)

        train = pd.concat([train,dummies], axis=1)

        train.drop([col],axis = 1 , inplace=True)

    

    return train
def preprocess(train):

    train, y = target_split(train)

    train = impute(train)

    train = scaler(train)

    train = cat_enc(train)

    

    return train, y
train_mod, y = preprocess(train)
train_mod.head()
np.random.seed(0)



model = Sequential()



model.add(Dense(48, kernel_initializer='normal',input_dim = train_mod.shape[1], activation='relu'))

model.add(Dense(96, kernel_initializer='normal',activation='relu'))

model.add(Dense(96, kernel_initializer='normal',activation='relu'))



model.add(Dense(1, kernel_initializer='normal',activation='linear'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
checkpoint_name = 'Weights-{epoch:02d}--{val_loss:.2f}.hdf5'

checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')

callbacks_list = [checkpoint]
model.fit(train_mod, y, epochs=50, batch_size=37, validation_split = 0.2, callbacks=callbacks_list)
!ls .
import os



best_weight_file = str()

val_loss = 100

for filename in os.listdir():

    if(filename.startswith('W')):

        name, ext = os.path.splitext(filename)

        if(int(name[-2:]) < val_loss):

            val_loss = int(name[-2:])

            best_weight_file = filename

            

print(best_weight_file)
model.load_weights(best_weight_file)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



scores = model.evaluate(train_mod, y, verbose=0)

print("Accuracy of model: %.2f%%" % (scores[1]*100))