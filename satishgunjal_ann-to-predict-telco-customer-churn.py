import pandas as pd

import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras

from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline
df = pd.read_csv("/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print(f'No of rows/training examples: {df.shape[0]}, No of columns/features: {df.shape[1]}')

df.sample(5)
df.describe()
df.isna().sum()
# Count of customer that will churn

cust_churn_yes_count = df[df.Churn == 'Yes'].shape[0]

# Count of customer that will not churn (retain)

cust_churn_no_count = df[df.Churn == 'No'].shape[0]



# Percentage of customer that will churn

cust_churn_yes_percent = round((cust_churn_yes_count / (cust_churn_yes_count + cust_churn_no_count) * 100),2)

# Percentage of customer that will not churn (retain)

cust_churn_no_percent = round((cust_churn_no_count / (cust_churn_yes_count + cust_churn_no_count) * 100 ),2)



plt.figure(figsize=(10,6))

ax = sns.countplot(df['Churn'])

ax.set_title(f'{cust_churn_yes_percent} % ({cust_churn_yes_count} nos) of cust will churn & {cust_churn_no_percent} % ({cust_churn_no_count} nos) of cust will retain')
plt.figure(figsize=(10,6))

ax = sns.countplot(x= 'gender', hue='Churn', data=df)

ax.set_title(f'Effect of Gender on customer churn')
plt.figure(figsize=(10,6))

ax = sns.countplot(x= 'SeniorCitizen', hue='Churn', data=df)

ax.set_title(f'SeniorCitizens have higher customer churn')

plt.xlabel('SeniorCitizens(0: No, 1: Yes)')
plt.figure(figsize=(10,6))

ax = sns.countplot(x= 'InternetService', hue='Churn', data=df)

ax.set_title(f'Effect of internet service on customer churn')
tenure_churn_yes = df[df.Churn == 'Yes'].tenure

tenure_churn_no = df[df.Churn == 'No'].tenure



plt.figure(figsize=(10,6))

plt.hist([tenure_churn_yes, tenure_churn_no], color=['orange', 'blue'], label= ['Leaving', 'Staying'])

plt.xlabel('Tenure(Months)')

plt.ylabel('No of Customers')

plt.title('Customers Staying vs Leaving Based on Tenure')

plt.legend()
monthly_charges_churn_yes = df[df.Churn == 'Yes'].MonthlyCharges

monthly_charges_churn_no = df[df.Churn == 'No'].MonthlyCharges





plt.figure(figsize=(10,6))

plt.hist([monthly_charges_churn_yes, monthly_charges_churn_no], color=['orange', 'blue'], label= ['Leaving', 'Staying'])

plt.xlabel('Monthly Charges')

plt.ylabel('No of Customers')

plt.title('Customers Staying vs Leaving Based on Monthly Charges')

plt.legend()
df1 = df.drop('customerID', axis = 'columns')

df1.shape # Print the shape of new dataframe
df1.dtypes
# Print the values in TotalCharges

df1.TotalCharges.unique()
# Print rows with missing TotalCharges values

df1[df1.TotalCharges == ' ']
df1.TotalCharges =  df1.TotalCharges.replace(r' ', '0')

df1[df1.tenure == 0]
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)

print(f'New datatype of TotalCharges : { df1.TotalCharges.dtype}')    
def print_unique_col_values(df):

    """Print unique values from categorical columns of the given dataframe"""

    print('Unique values from categorical columns,\n')

    for column in df.columns:

        if(df[column].dtypes == 'object'): 

            print(f'column: {column}, Unique vlaues: {df[column].unique()}')

        

print_unique_col_values(df1)
df1.replace('No internet service', 'No', inplace = True)

df1.replace('No phone service', 'No', inplace = True)

# Lets print unique values again

print_unique_col_values(df1)
# Converting churn to numeric

df1['Churn'].replace({'Yes': 1,'No': 0},inplace=True)
# Create df2 for cleaned dataset

df2 = pd.get_dummies(data = df1)



print(f'So we have added {df2.shape[1]- df1.shape[1]} more columns to our list. New shape : {df2.shape}')

df2.sample(5)
cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']



scaler = MinMaxScaler()

df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale]) # Fit to data, then transform it

df2[cols_to_scale].describe()
# Create feature matrix X without label column 'Churn'

X = df2.drop('Churn',axis = 'columns')

# Create label vector y

y = df2['Churn']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

print(f'X_train: {X_train.shape}, y_train: {y_train.shape}')

print(f'X_test: {X_test.shape}, y_test: {y_test.shape}')



# Lets have a look at our training datatset

X_train.sample(5)
model = tf.keras.Sequential([

    tf.keras.layers.Dense(38, input_shape= (38,), activation= 'relu'),

    tf.keras.layers.Dense(14, activation= 'relu'),

    tf.keras.layers.Dense(1, activation= 'sigmoid')     

])



model.compile(optimizer= 'adam',

             loss='binary_crossentropy',

             metrics=['accuracy'])
model.fit(X_train, y_train, epochs= 100)
model.evaluate(X_test, y_test)
predictions = model.predict(X_test)

predictions[:5]
y_pred = []



for val in predictions:

    if val > 0.5:

        y_pred.append(1)

    else:

        y_pred.append(0)

            

y_pred[:10]
df_true_pred = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred}) 

df_true_pred[:10]
print(classification_report(y_test,y_pred))
cm = tf.math. confusion_matrix(labels= y_test, predictions= y_pred)

plt.figure(figsize = (10, 7))

sns.heatmap(cm, annot= True, fmt= 'd')

plt.xlabel('Predicted')

plt.ylabel('Truth')