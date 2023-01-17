# Supress Warnings



import warnings

warnings.filterwarnings('ignore')



# Import the numpy and pandas package



import numpy as np

import pandas as pd



# Data Visualisation



import matplotlib.pyplot as plt 

 
# Importing all datasets

churn_data = pd.read_csv('../input/churn_data.csv')

customer_data = pd.read_csv('../input/customer_data.csv')

internet_data = pd.read_csv('../input/internet_data.csv')
# Merging on 'customerID'

df_1 = pd.merge(churn_data, customer_data, how='inner', on='customerID')



# Final dataframe with all predictor variables

dataset = pd.merge(df_1, internet_data, how='inner', on='customerID')



# Let's see the head of our master dataset

dataset.head()



# let's look at the statistical aspects of the dataframe

dataset.describe()



# Let's see the type of each column

dataset.info()
# Checking Null values

dataset.isnull().sum()*100/dataset.shape[0]



#Replacing NAN values in totalcharges

dataset['TotalCharges'].describe()

dataset['TotalCharges'] = dataset['TotalCharges'].replace(' ', np.nan)

dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'])



value = (dataset['TotalCharges']/dataset['MonthlyCharges']).median()*dataset['MonthlyCharges']

dataset['TotalCharges'] = value.where(dataset['TotalCharges'] == np.nan, other =dataset['TotalCharges'])

dataset['TotalCharges'].describe()





#Model Building

#Data Preparation

#Converting some binary variables (Yes/No) to 0/1

# List of variables to map



varlist =  ['PhoneService', 'PaperlessBilling', 'Churn', 'Partner', 'Dependents']
# Defining the map function

def binary_map(x):

    return x.map({'Yes': 1, "No": 0})
# Applying the function to the var list

dataset[varlist] = dataset[varlist].apply(binary_map)

dataset.head()
#For categorical variables with multiple levels, create dummy features (one-hot encoded)

# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(dataset[['Contract', 'PaymentMethod', 'gender', 'InternetService']], drop_first=True)



# Adding the results to the master dataframe

dataset = pd.concat([dataset, dummy1], axis=1)

dataset.head()
# Creating dummy variables for the remaining categorical variables and dropping the level with big names.



# Creating dummy variables for the variable 'MultipleLines'

ml = pd.get_dummies(dataset['MultipleLines'], prefix='MultipleLines')

# Dropping MultipleLines_No phone service column

ml1 = ml.drop(['MultipleLines_No phone service'], 1)

#Adding the results to the master dataframe

dataset = pd.concat([dataset,ml1], axis=1)



# Creating dummy variables for the variable 'OnlineSecurity'.

os = pd.get_dummies(dataset['OnlineSecurity'], prefix='OnlineSecurity')

os1 = os.drop(['OnlineSecurity_No internet service'], 1)

# Adding the results to the master dataframe

dataset = pd.concat([dataset,os1], axis=1)



# Creating dummy variables for the variable 'OnlineBackup'.

ob = pd.get_dummies(dataset['OnlineBackup'], prefix='OnlineBackup')

ob1 = ob.drop(['OnlineBackup_No internet service'], 1)

# Adding the results to the master dataframe

dataset = pd.concat([dataset,ob1], axis=1)



# Creating dummy variables for the variable 'DeviceProtection'. 

dp = pd.get_dummies(dataset['DeviceProtection'], prefix='DeviceProtection')

dp1 = dp.drop(['DeviceProtection_No internet service'], 1)

# Adding the results to the master dataframe

dataset = pd.concat([dataset,dp1], axis=1)



# Creating dummy variables for the variable 'TechSupport'. 

ts = pd.get_dummies(dataset['TechSupport'], prefix='TechSupport')

ts1 = ts.drop(['TechSupport_No internet service'], 1)

# Adding the results to the master dataframe

dataset = pd.concat([dataset,ts1], axis=1)



# Creating dummy variables for the variable 'StreamingTV'.

st =pd.get_dummies(dataset['StreamingTV'], prefix='StreamingTV')

st1 = st.drop(['StreamingTV_No internet service'], 1)

# Adding the results to the master dataframe

dataset = pd.concat([dataset,st1], axis=1)



# Creating dummy variables for the variable 'StreamingMovies'. 

sm = pd.get_dummies(dataset['StreamingMovies'], prefix='StreamingMovies')

sm1 = sm.drop(['StreamingMovies_No internet service'], 1)

# Adding the results to the master dataframe

dataset = pd.concat([dataset,sm1], axis=1)

dataset.head()

# We have created dummies for the below variables, so we can drop them

dataset = dataset.drop(['Contract','PaymentMethod','gender','MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',

       'TechSupport', 'StreamingTV', 'StreamingMovies'], 1)

dataset.info()
# Checking for outliers in the continuous variables

num_telecom = dataset[['tenure','MonthlyCharges','SeniorCitizen','TotalCharges']]

# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%

num_telecom.describe(percentiles=[.25, .5, .75, .90, .95, .99])

# Checking up the missing values (column-wise)

dataset.isnull().sum()

# Removing NaN TotalCharges rows

dataset = dataset[~np.isnan(dataset['TotalCharges'])]
# Checking percentage of missing values after removing the missing values

round(100*(dataset.isnull().sum()/len(dataset.index)), 2)


# Putting feature variable to X

from sklearn.model_selection import train_test_split #use 'cross_validation' instead of

                                                     #'model_selection' Executing in jupyter or spyder 

X = dataset.drop(['Churn','customerID'], axis=1)

X.head()



# Putting response variable to y

y = dataset['Churn']



y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

#Feature Scaling

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



X_train[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(X_train[['tenure','MonthlyCharges','TotalCharges']])



X_train.head()
#Model Building

# Logistic regression model

import statsmodels.api as sm

logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())

logm1.fit().summary()



#Feature Selection Using RFE

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

from sklearn.feature_selection import RFE

rfe = RFE(logreg, 15)             # running RFE with 13 variables as output

rfe = rfe.fit(X_train, y_train)

rfe.support_



list(zip(X_train.columns, rfe.support_, rfe.ranking_))





col = X_train.columns[rfe.support_]

X_train.columns[~rfe.support_]

#Adding a constant



X_train_sm = sm.add_constant(X_train[col])

logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())

res = logm2.fit()

res.summary()

# Getting the predicted values on the train set

y_train_pred = res.predict(X_train_sm)

y_train_pred[:10]





y_train_pred_final = pd.DataFrame({'Churn':y_train.values, 'Churn_Prob':y_train_pred})

y_train_pred_final['CustID'] = y_train.index

y_train_pred_final.head()
#Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

y_train_pred_final['predicted'] = y_train_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.5 else 0)



# Let's see the head

y_train_pred_final.head()
# Confusion matrix

from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(y_train_pred_final.Churn, y_train_pred_final.predicted )

print(confusion_matrix)

# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Churn, y_train_pred_final.predicted))

#Making predictions on the test set

X_test[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(X_test[['tenure','MonthlyCharges','TotalCharges']])

X_test = X_test[col]

X_test.head()



X_test_sm = sm.add_constant(X_test)

y_test_pred = res.predict(X_test_sm)

y_test_pred[:10]
# Converting y_pred to a dataframe which is an array

y_pred_1 = pd.DataFrame(y_test_pred)

y_pred_1.head()
# Converting y_test to dataframe

y_test_df = pd.DataFrame(y_test)
# Putting CustID to index

y_test_df['CustID'] = y_test_df.index

# Removing index for both dataframes to append them side by side 

y_pred_1.reset_index(drop=True, inplace=True)

y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1

y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)

y_pred_final.head()
# Rearranging the columns

y_pred_final = y_pred_final.reindex_axis(['CustID','Churn','Churn_Prob'], axis=1)

# Let's see the head of y_pred_final

y_pred_final.head()

y_pred_final['final_predicted'] = y_pred_final.Churn_Prob.map(lambda x: 1 if x > 0.42 else 0)

y_pred_final.head()

# Let's check the overall accuracy.

metrics.accuracy_score(y_pred_final.Churn, y_pred_final.final_predicted)