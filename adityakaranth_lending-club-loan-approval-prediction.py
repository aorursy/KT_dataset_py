# Basic Imports

import numpy as np

import pandas as pd

import os



# Visualization 

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')

matplotlib.rcParams['figure.figsize'] = (12,8)  



# Model

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from datetime import datetime



import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout

from tensorflow.keras.constraints import max_norm

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard



# Evaluation

from sklearn.metrics import classification_report,confusion_matrix
tf.__version__
# Loading the dataset

df_info = pd.read_csv('../input/lendingclub-data-sets/lending_club_info.csv', index_col='LoanStatNew')

df = pd.read_csv('../input/lendingclub-data-sets/lending_club_loan_two.csv')
# df_info contains description about all the features present in dataset(df)

# Target is loan_status

df_info.head()
df.head()
df.info()
# Function to get information about a feature

def feature_info(col_name):

    return '{} : {}'.format(col_name,df_info.loc[col_name]['Description'])



# Info about Target ('loan_status')

feature_info('loan_status')
df.describe().T
# Get numeric columns

num_cols = df.select_dtypes(include=[np.number]).columns.values

print('Numeric cols :',num_cols)
# Get Non-numeric/categorical columns

cat_cols = df.select_dtypes(exclude=[np.number]).columns.values

print('Categorical cols :',cat_cols)
# Check for missing data

sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')

plt.tight_layout()
# Number of missing values

print(df.isnull().sum().sort_values(ascending=False))
# Number of missing %

print(round((df.isnull().mean()*100).sort_values(ascending=False),2))
# Countplot of Target (loan_status)

sns.countplot(df['loan_status'])
# From above graph we have imbalanced data, where 80% of data points are Fully Paid and 20% charged off

100*df['loan_status'].value_counts()/(df['loan_status'].count())
# Histogram of loan_amnt

sns.distplot(df['loan_amnt'], kde=False)
# Correlation plot

sns.heatmap(df.dropna().corr(), annot=True, cmap='viridis')
# "installment" almost perfect correlation with the "loan_amount" feature

print(feature_info('installment'),'\n', feature_info('loan_amnt'))
# As loan_amnt amount increases installments increases

sns.scatterplot('installment', 'loan_amnt', data=df)
# open credit lines v/s total number of credit lines in the borrower's credit file.

print(feature_info('total_acc'), '\n',feature_info('open_acc'),'\n')

sns.scatterplot('open_acc', 'total_acc', data=df.dropna())

plt.tight_layout()
# Boxplot showing the relationship between the loan_status and the loan_amnt

sns.boxplot('loan_status','loan_amnt',data=df)
# summary statistics for the loan amount, grouped by the loan_status.

df.groupby('loan_status')['loan_amnt'].describe()

# Charged Off loans "loan amount" is a bit higher than Fully Paid loans
# Explore the "grade" column that LendingClub attributes to the loans.

print(np.sort(df['grade'].unique()))
# Explore the "sub_grade" column that LendingClub attributes to the loans.

print(np.sort(df['sub_grade'].unique()))
# Countplot of grade per loan_status. 

sns.countplot(df['grade'].sort_values(), hue=df['loan_status'])
# Countplot of sub_grade per loan_status. 

plt.figure(figsize=(15,6))

sns.countplot(df['sub_grade'].sort_values(), palette='rainbow',hue=df['loan_status'])
# Isolating F and G and countplot just for those subgrades

f_g = df[df['sub_grade'].isin(np.sort(df['sub_grade'].unique())[-10:])]

sns.countplot(f_g['sub_grade'].sort_values(), palette='rainbow', hue=f_g['loan_status'])



# F and G sub grades often doesn't payback their loans
# Copy of df

df1 = df.copy()
# Function to check missing data after handling missing data

def check_null():

  return round(df1.isna().mean()*100,2).sort_values(ascending = False)
# Around 10% of mort_acc is missing

feature_info('mort_acc')
df1['mort_acc'].value_counts()
# Features that correlates with 'mort_acc'

df1.corr()['mort_acc'].sort_values(ascending=False)
# 'total_acc' have high correlation with 'mort_acc'

# Lookup table for mort_acc values according to total_acc

mort_acc_avg = df1.groupby('total_acc')['mort_acc'].mean()

mort_acc_avg
# Function "fill" 

def fill(x):

  '''

  Take a row and if mort_acc value is missing fill it with w.r.t total_acc,

  with help of look up table above

  if mort_acc not null, no change.

  '''

  mort_account = x[0]

  total_account = x[1]

  if pd.isnull(mort_account):

    return mort_acc_avg[total_account]

  else:

    return mort_account
# Fill missing values using 'fill' function

df1['mort_acc'] = df1[['mort_acc', 'total_acc']].apply(fill, axis=1)
# Verify changes 

df1['mort_acc'].isnull().sum()
check_null()
print(feature_info('emp_title'),'\n')

print('No. of unique job titles : ', df1['emp_title'].nunique())
# Too many unique job titles to convert in to a dummy variable feature. 

# Dropping "emp_title"

df1.drop('emp_title',axis=1, inplace=True)
check_null()
print(feature_info('emp_length'),'\n')

print('Different Employment lengths: \n', sorted(df1['emp_length'].dropna().unique()))
# Countplot of 'emp_lenth' with given order "emp_length_order"

emp_length_order = [ '< 1 year','1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years']

sns.countplot(df1['emp_length'], order=emp_length_order, hue=df1['loan_status'])



# The fully paid and Charged Off ratio looks same all emp_lengths
# Charged Off emp loan_status w.r.t emp_length 

emp_co = df1[df1['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']

print(emp_co)
# Fully Paid emp loan_status w.r.t emp_length 

emp_fp = df1[df1['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']

print(emp_fp)
# Charged off loan_status Ratio w.r.t emp_length 

emp_len = emp_co/(emp_fp+emp_co)

print(emp_len)
# Charge off rates are similar across all employment lengths

emp_len.plot(kind='bar')
# Dropping the emp_length column as it doen't contain much information

df1.drop('emp_length',axis=1, inplace=True)
check_null()
print(feature_info('title'),'\n')



print('No. of different titles: ', df1['title'].dropna().nunique())

print('\nDifferent titles:\n',df1['title'].dropna().value_counts())
# 48817 different titles to convert in to dummies

# Dropping title feature

df1.drop('title',axis=1, inplace=True) 
check_null()
# pub_rec_bankruptcies and revol_util have only 0.14% and 0.07 % of missing values respectively



print(feature_info('pub_rec_bankruptcies'),'\n')

print('Different values of pub_rec_bankruptcies: ', df1['pub_rec_bankruptcies'].dropna().unique())

print('\npub_rec_bankruptcies counts:\n',df1['pub_rec_bankruptcies'].dropna().value_counts())
# Mean and Median values of pub_rec_bankruptcies and revol_util



print('pub_rec_bankruptcies:\n',df['pub_rec_bankruptcies'].describe())

print('\n\nrevol_util:\n',df['revol_util'].describe())
# Fill the missing values with median

df1 = df1.fillna(df.median())
check_null()
# Get all Categorical cols of df1

cat_cols1 = df1.select_dtypes(exclude=[np.number]).columns.values

print('Categorical cols of df1:',cat_cols1)
print(feature_info('term'),'\n')

df1['term'].unique()
# Converting ' 36 months' to 36, ' 60 months' to 60

df1['term'] = df1['term'].map({' 36 months':36, ' 60 months':60})

df1['term'].unique()
print(feature_info('grade'))

print(df1['grade'].unique(),'\n')

print(feature_info('sub_grade'))

print(df1['sub_grade'].unique())
# As subgrade itself cointains grades,

# dropping 'grade' feature

df1.drop('grade', axis=1, inplace=True)
print(feature_info('home_ownership'),'\n')

print('Unique home_ownership :',df1['home_ownership'].unique(),'\n')

print('Value counts of home_ownership :\n',df1['home_ownership'].value_counts())
# Merging "NONE" and "ANY" to 'OTHERS'

df1['home_ownership'] = df1['home_ownership'].replace(['NONE','ANY'],'OTHER')

print('Unique home_ownership :',df1['home_ownership'].unique(),'\n')

print('Value counts of home_ownership :\n',df1['home_ownership'].value_counts())
print(feature_info('verification_status'),'\n')

print('Unique verification_status :',df1['verification_status'].unique(),'\n')

print('Value counts of verification_status :\n',df1['verification_status'].value_counts())
sns.countplot(df1['verification_status'], hue=df['loan_status'])
# As we are predicting whether or not a loan would be issued when using our model

# We wouldn't know beforehand whether or not a loan would be issued. 

# dropping this feature



print(feature_info('issue_d'),'\n')

df1.drop('issue_d', axis=1, inplace=True)
# Target Feature



print(feature_info('loan_status'),'\n')

print('Unique loan_status :',df1['loan_status'].unique(),'\n')

print('Value counts of loan_status :\n',df1['loan_status'].value_counts())
# New 'load_repaid' column with 1 if the loan status was "Fully Paid" and a 0 if it was "Charged Off".

df1['loan_repaid'] = df1['loan_status'].apply(lambda x:1 if x=='Fully Paid' else 0)



#Dropping the original loan_status column

df1.drop(['loan_status'], axis=1, inplace=True)

print('Unique loan_repaid :',df1['loan_repaid'].unique(),'\n')

print('Value counts of loan_repaid :\n',df1['loan_repaid'].value_counts())
# Correlation of the numeric features to the new loan_repaid column

df1.corr()['loan_repaid'].sort_values()[:-1].plot.bar()
print(feature_info('purpose'),'\n')

print('Unique purpose :',df1['purpose'].unique(),'\n')

print('Value counts of purpose :\n',df1['purpose'].value_counts())
print(feature_info('earliest_cr_line'),'\n')

print('Unique earliest_cr_line count :',df1['earliest_cr_line'].nunique(),'\n')

print('Value counts of earliest_cr_line :\n',df1['earliest_cr_line'].value_counts())
# Extracting the year from the "earliest_cr_line" feature.

# New numeric feature 'earliest_cr_year'

# dropping the original earliest_cr_line feature.



df1['earliest_cr_year'] = df1['earliest_cr_line'].apply(lambda date:int(date[-4:]))

df1.drop('earliest_cr_line', axis=1, inplace=True)
print(feature_info('initial_list_status'),'\n')

print('Unique initial_list_status count :',df1['initial_list_status'].unique(),'\n')

print('Value counts of initial_list_status :\n',df1['initial_list_status'].value_counts())
print(feature_info('application_type'),'\n')

print('Unique application_type count :',df1['application_type'].unique(),'\n')

print('Value counts of application_type :\n',df1['application_type'].value_counts())
df1['address']
# Exctracting only zipcode from address into a new "zip_code" column



df1['zip_code'] = df1['address'].apply(lambda x: x[-5:])

df1.drop('address', axis=1, inplace=True)
print('Value counts of zip_code :\n',df1['zip_code'].value_counts())
df1.corr()['installment'].sort_values(ascending=False)
# Dropping "installment" as it correlates above 90% with "loan_amnt"

df1.drop('installment', axis=1, inplace=True)
# df2 as copy of df1

df2 = df1.copy()
df2
# Take all categorical column of df2 in cat_cols2

cat_cols2 = df2.select_dtypes(exclude=[np.number]).columns.values

print(cat_cols2)
# Get dummies for categorical columns

dummies = pd.get_dummies(df1[cat_cols2], drop_first=True)



# Drop actual categorical columns

df2 = df2.drop(cat_cols2,axis=1)



# Concatenate dummy columns with df2

df2 = pd.concat([df2,dummies],axis=1)
df2.head()
print('Total rows : ', len(df2)) 

print('Total columns : ', len(df2.columns)) 

print('columns : \n', df2.columns.values) 
# last row of df2 as new customer

new_cust = df2.iloc[-1]



# All rows excluding last data point

X = df2.iloc[:-1].drop('loan_repaid', axis=1).values

y = df2.iloc[:-1]['loan_repaid'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=22)
print('X_train : {} \ny_train : {}'.format(X_train.shape,y_train.shape))
print('X_test : {} \ny_test : {}'.format(X_test.shape,y_test.shape))
# Using MinMaxScaler to convert values between min:0 and max:1

scaler = MinMaxScaler()
X_train_sca = scaler.fit_transform(X_train)

X_test_sca = scaler.transform(X_test)
# Creating Model



model = Sequential()

# Input layer

model.add(Dense(77, activation='relu', kernel_constraint=max_norm(3)))

model.add(Dropout(0.3)) # Drop 30% of neurons randomly

# Hidden layers

model.add(Dense(38, activation='relu', kernel_constraint=max_norm(3)))

model.add(Dropout(0.3))

model.add(Dense(19, activation='relu', kernel_constraint=max_norm(3)))

model.add(Dropout(0.3))

# Output layer

model.add(Dense(1, activation='sigmoid'))
# Compile the created model

model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])



# Monitor the "validation_loss" and 

# when "min" value is reached during training, wait for "10" epochs and stop training

early_stop = EarlyStopping(monitor='val_loss',

                    verbose=1,

                    mode='min',

                    patience=10)



# Tensor Board

logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))

tfboard = TensorBoard(logdir,

                      histogram_freq=1,

                      update_freq = 'epoch')



# Fit the model

model.fit(X_train_sca,

        y_train,

        batch_size=256,

        epochs=100,

        validation_data=(X_test_sca, y_test),

        callbacks=[early_stop, tfboard])
model.summary()
# New dataframe "losses" with model.history.history data

losses = pd.DataFrame(model.history.history)

losses
# Plot training loss v/s validation loss

losses[['loss','val_loss']].plot()
# Plot training accuracy v/s validation accuracy

losses[['accuracy','val_accuracy']].plot()
# Print the final loss and accuracy on test set.

print(f'loss: {model.evaluate(X_test_sca, y_test, verbose=0)[0]}\naccuracy: {model.evaluate(X_test_sca, y_test, verbose=0)[1]}')





# Accuracy of 89% is considering an imbalanced data (80% of data points are Fully Paid and 20% charged off)
# Predictions on test set

y_pred = model.predict_classes(X_test_sca)

print(y_pred)
# classification report

print(classification_report(y_test,y_pred))
# confusion matrix

print(confusion_matrix(y_test,y_pred))
# Load tenserboard

%load_ext tensorboard
%tensorboard --logdir logs
# As satisfied with obtained accuracy,

# Before predicting for new customer, training on whole dataset (X and y) without splitting.



X_sca = scaler.fit_transform(X)



# Creating final model

final_model = Sequential()

# Input layer

final_model.add(Dense(77, activation='relu', kernel_constraint=max_norm(3)))

final_model.add(Dropout(0.3)) # Drop 30% of neurons randomly

# Hidden layers

final_model.add(Dense(38, activation='relu', kernel_constraint=max_norm(3)))

final_model.add(Dropout(0.3))

final_model.add(Dense(19, activation='relu', kernel_constraint=max_norm(3)))

final_model.add(Dropout(0.3))

# Output layer

final_model.add(Dense(1, activation='sigmoid'))





# Compile the created model

final_model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])





# Fit the model

final_model.fit(X_sca, y,

        batch_size=256,

        epochs=30)
# final_model.save('loan_approval.h5')  
# New customer "new_cust"

new_cust
# Function "assess_customer"

def assess_customer(cust):



  '''

  Take new customer and reshape the values to shape that model was trained on. (1,77)

  Transform the new customer attributes using same MinMaxScaler object

  return the output based on predicted class.

  '''



  cust = cust.values.reshape(1,77)

  cust_sca = scaler.transform(cust)



  if final_model.predict_classes(cust_sca)==1:

    print ("Customer is likely to pay back the loan \nLoan can be approved")

  else:

    print ("Customer is not likely to pay back the loan \nLoan can't be approved")
# Model predicted

assess_customer(new_cust.drop('loan_repaid'))
# Actual

new_cust['loan_repaid']