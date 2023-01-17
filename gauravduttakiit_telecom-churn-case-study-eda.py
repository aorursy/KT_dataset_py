# Suppressing Warnings

import warnings

warnings.filterwarnings('ignore')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing Pandas and NumPy

import pandas as pd, numpy as np, seaborn as sns,matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
# Importing all datasets

churn_data = pd.read_csv("/kaggle/input/telecom-churn-data-sets/churn_data.csv")

churn_data.head()
customer_data = pd.read_csv("/kaggle/input/telecom-churn-data-sets/customer_data.csv")

customer_data.head()
internet_data = pd.read_csv("/kaggle/input/telecom-churn-data-sets/internet_data.csv")

internet_data.head()
# Merging on 'customerID'

df_1 = pd.merge(churn_data, customer_data, how='inner', on='customerID')
# Final dataframe with all predictor variables

telecom = pd.merge(df_1, internet_data, how='inner', on='customerID')
# Let's see the head of our master dataset

telecom.head()
# Let's check the dimensions of the dataframe

telecom.shape
# let's look at the statistical aspects of the dataframe

telecom.describe()
# Let's see the type of each column

telecom.info()
#The varaible was imported as a string we need to convert it to float

# telecom['TotalCharges'] = telecom['TotalCharges'].astype(float) 

telecom.TotalCharges = pd.to_numeric(telecom.TotalCharges, errors='coerce')
telecom.info()


plt.figure(figsize=(20,40))

plt.subplot(10,2,1)

ax = sns.distplot(telecom['tenure'], hist=True, kde=False, 

             bins=int(180/5), color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

ax.set_ylabel('# of Customers')

ax.set_xlabel('Tenure (months)')

plt.subplot(10,2,2)

ax = sns.countplot(x='PhoneService', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,3)

ax =sns.countplot(x='Contract', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,3)

ax =sns.countplot(x='Contract', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,4)

ax =sns.countplot(x='PaperlessBilling', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,5)

ax =sns.countplot(x='PaymentMethod', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,6)

ax =sns.countplot(x='Churn', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,7)

ax =sns.countplot(x='gender', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,8)

ax =sns.countplot(x='SeniorCitizen', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,9)

ax =sns.countplot(x='Partner', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,10)

ax =sns.countplot(x='Dependents', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,11)

ax =sns.countplot(x='MultipleLines', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,12)

ax =sns.countplot(x='InternetService', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,13)

ax =sns.countplot(x='OnlineSecurity', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,14)

ax =sns.countplot(x='OnlineBackup', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,15)

ax =sns.countplot(x='DeviceProtection', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,16)

ax =sns.countplot(x='TechSupport', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,17)

ax =sns.countplot(x='StreamingTV', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,18)

ax =sns.countplot(x='StreamingMovies', data=telecom)

ax.set_ylabel('# of Customers')

plt.subplot(10,2,19)

ax = sns.distplot(telecom['MonthlyCharges'], hist=True, kde=False, 

             bins=int(180/5), color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

ax.set_ylabel('# of Customers')

ax.set_xlabel('MonthlyCharges')

plt.subplot(10,2,20)

ax = sns.distplot(telecom['TotalCharges'], hist=True, kde=False, 

             bins=int(180/5), color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

ax.set_ylabel('# of Customers')

ax.set_xlabel('TotalCharges');
sns.pairplot(telecom)

plt.show()
plt.figure(figsize=(25, 10))

plt.subplot(1,3,1)

sns.boxplot(x = 'tenure', y = 'Churn', data=telecom)

plt.subplot(1,3,2)

sns.boxplot(x = 'MonthlyCharges', y = 'Churn', data=telecom)

plt.subplot(1,3,3)

sns.boxplot(x = 'TotalCharges', y = 'Churn', data=telecom)

plt.show()
# List of variables to map



varlist =  ['PhoneService', 'PaperlessBilling', 'Churn', 'Partner', 'Dependents']



# Defining the map function

def binary_map(x):

    return x.map({'Yes': 1, "No": 0})



# Applying the function to the housing list

telecom[varlist] = telecom[varlist].apply(binary_map)
telecom.head()
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(telecom[['Contract', 'PaymentMethod', 'gender', 'InternetService']], drop_first=True)



# Adding the results to the master dataframe

telecom = pd.concat([telecom, dummy1], axis=1)
telecom.head()
# Creating dummy variables for the remaining categorical variables and dropping the level with big names.



# Creating dummy variables for the variable 'MultipleLines'

ml = pd.get_dummies(telecom['MultipleLines'], prefix='MultipleLines')

# Dropping MultipleLines_No phone service column

ml1 = ml.drop(['MultipleLines_No phone service'], 1)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,ml1], axis=1)



# Creating dummy variables for the variable 'OnlineSecurity'.

os = pd.get_dummies(telecom['OnlineSecurity'], prefix='OnlineSecurity')

os1 = os.drop(['OnlineSecurity_No internet service'], 1)

# Adding the results to the master dataframe

telecom = pd.concat([telecom,os1], axis=1)



# Creating dummy variables for the variable 'OnlineBackup'.

ob = pd.get_dummies(telecom['OnlineBackup'], prefix='OnlineBackup')

ob1 = ob.drop(['OnlineBackup_No internet service'], 1)

# Adding the results to the master dataframe

telecom = pd.concat([telecom,ob1], axis=1)



# Creating dummy variables for the variable 'DeviceProtection'. 

dp = pd.get_dummies(telecom['DeviceProtection'], prefix='DeviceProtection')

dp1 = dp.drop(['DeviceProtection_No internet service'], 1)

# Adding the results to the master dataframe

telecom = pd.concat([telecom,dp1], axis=1)



# Creating dummy variables for the variable 'TechSupport'. 

ts = pd.get_dummies(telecom['TechSupport'], prefix='TechSupport')

ts1 = ts.drop(['TechSupport_No internet service'], 1)

# Adding the results to the master dataframe

telecom = pd.concat([telecom,ts1], axis=1)



# Creating dummy variables for the variable 'StreamingTV'.

st =pd.get_dummies(telecom['StreamingTV'], prefix='StreamingTV')

st1 = st.drop(['StreamingTV_No internet service'], 1)

# Adding the results to the master dataframe

telecom = pd.concat([telecom,st1], axis=1)



# Creating dummy variables for the variable 'StreamingMovies'. 

sm = pd.get_dummies(telecom['StreamingMovies'], prefix='StreamingMovies')

sm1 = sm.drop(['StreamingMovies_No internet service'], 1)

# Adding the results to the master dataframe

telecom = pd.concat([telecom,sm1], axis=1)
telecom.head()
# We have created dummies for the below variables, so we can drop them

telecom = telecom.drop(['Contract','PaymentMethod','gender','MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',

       'TechSupport', 'StreamingTV', 'StreamingMovies'], 1)
# Checking for outliers in the continuous variables

num_telecom = telecom[['tenure','MonthlyCharges','SeniorCitizen','TotalCharges']]
# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%

num_telecom.describe(percentiles=[.25, .5, .75, .90, .95, .99])
# Adding up the missing values (column-wise)

telecom.isnull().sum()
print('No. of Null Records for TotalCharges:',telecom.TotalCharges.isnull().sum())
print('No. of Records for TotalCharges:',len(telecom))
print('No. of non Records for TotalCharges:',len(telecom)-telecom.TotalCharges.isnull().sum())
# Checking the percentage of missing values

round(100*(telecom.isnull().sum()/len(telecom.index)), 2)
telecom = telecom.dropna()

telecom = telecom.reset_index(drop=True)



# Checking percentage of missing values after removing the missing values

round(100*(telecom.isnull().sum()/len(telecom.index)), 2)
from sklearn.model_selection import train_test_split
# Putting feature variable to X

X = telecom.drop(['Churn','customerID'], axis=1)



X.head()
# Putting response variable to y

y = telecom['Churn']



y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



X_train[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(X_train[['tenure','MonthlyCharges','TotalCharges']])



X_train.head()
### Checking the Churn Rate

churn = (sum(telecom['Churn'])/len(telecom['Churn'].index))*100

churn
# Importing matplotlib and seaborn

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Let's see the correlation matrix 

plt.figure(figsize = (25,25))        # Size of the figure

sns.heatmap(telecom.corr(),annot = True,cmap="tab20c")

plt.show()
plt.figure(figsize=(10,8))

telecom.corr()['Churn'].sort_values(ascending = False).plot(kind='bar');
X_test = X_test.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No',

                       'StreamingTV_No','StreamingMovies_No'], 1)

X_train = X_train.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No',

                         'StreamingTV_No','StreamingMovies_No'], 1)
plt.figure(figsize = (25,25))

sns.heatmap(X_train.corr(),annot = True,cmap="tab20c")

plt.show()