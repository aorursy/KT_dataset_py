# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



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
df = pd.read_csv('../input/bank-loan2/madfhantr.csv')

test = pd.read_csv('../input/bank-loan2/madhante.csv')
df.head()
#see my data set



df.shape
df.describe(include='all')
# Import missingno

import missingno as msno



# Plot missingness dendrogram of diabetes

msno.dendrogram(df)



# Show plot

plt.show()
null_columns=df.columns[df.isnull().any()]



print(df[df.isnull().any(axis=1)][null_columns].head())
#fill missing value use mode method

df['Gender'].fillna((df['Gender'].mode()[0]),inplace=True)

#fill Self_Employed  

df['Self_Employed'].fillna((df['Self_Employed'].mode()[0]),inplace=True)



df['LoanAmount'].fillna((df['LoanAmount'].mode()[0]),inplace=True)



# Loan_Amount_Term  

df['Loan_Amount_Term'].fillna((df['Loan_Amount_Term'].mode()[0]),inplace=True)



# Credit_History  



df['Credit_History'].fillna((df['Credit_History'].mode()[0]),inplace=True)





df['Dependents'].fillna((df['Dependents'].mode()[0]),inplace=True)



# replacing '+' from Dependent column

df['Dependents']=df['Dependents'].apply(lambda x:str(x).replace('+','')if '+' in str(x) else str(x))

df['Dependents']=df['Dependents'].apply(lambda x:int(x))



#fill marrird col

df['Married'].fillna((df['Married'].mode()[0]),inplace=True)









# Conform your data is not any null value 

#lets check it
df.isnull().sum()
df[df.dtypes[(df.dtypes=="float64")|(df.dtypes=="int64")]

                        .index.values].hist(figsize=[11,11])
# let creat x and y value

x_with_scale = df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']]



y_with_scale = df['Loan_Status']
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



#Initializing and Fitting a k-NN model

knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(x_with_scale,y_with_scale.values.ravel())

# Checking the performance of our model on the testing data set

print("\nAccuracy score on test set :", accuracy_score(y_with_scale,knn.predict(x_with_scale)))
from sklearn.preprocessing import MinMaxScaler

min_max=MinMaxScaler()
x_with_scale_minmax=min_max.fit_transform(x_with_scale)
# Fitting k-NN on our scaled data set

knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(x_with_scale_minmax,y_with_scale)

# Checking the model's accuracy

accuracy_score(y_with_scale,knn.predict(x_with_scale_minmax))
from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression(solver='liblinear', penalty='l1')

log_reg.fit(x_with_scale,y_with_scale.values.ravel())

# Checking the performance of our model on the testing data set

print("\nAccuracy score on test set :", accuracy_score(y_with_scale,log_reg.predict(x_with_scale)))
from sklearn.linear_model import LogisticRegression



log_reg_1 = LogisticRegression(solver='liblinear', penalty='l1')

log_reg_1.fit(x_with_scale_minmax,y_with_scale.values.ravel())



# Checking the performance of our model on the testing data set

print("\nAccuracy score on test set :", accuracy_score(y_with_scale,log_reg_1.predict(x_with_scale_minmax)))
#import library

# Import StandardScaler from scikit-learn

from sklearn.preprocessing import StandardScaler



# Create the scaler

ss = StandardScaler()
# let creat x and y value

x_with_stand = df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']]



y_with_stand = df['Loan_Status']
# Apply the scaler to the DataFrame subset

x_with_stand_scaled = ss.fit_transform(x_with_stand)
# import SVC classifier

from sklearn.svm import SVC





# import metrics to compute accuracy

from sklearn.metrics import accuracy_score





# instantiate classifier with default hyperparameters

svc_scale=SVC() 



# fit classifier to training set

svc_scale.fit(x_with_stand_scaled,y_with_stand)



# Checking the performance of our model on the testing data set

print("\nAccuracy score on test set :", accuracy_score(y_with_stand,svc_scale.predict(x_with_stand_scaled)))
from sklearn.linear_model import LogisticRegression

log_stand=LogisticRegression(penalty='l2',C=.01)
log_stand.fit(x_with_stand_scaled,y_with_stand)



# Checking the performance of our model on the testing data set

print("\nAccuracy score on test set :", accuracy_score(y_with_stand,log_stand.predict(x_with_stand_scaled)))

# Fitting k-NN on our scaled data set

knn_stand=KNeighborsClassifier(n_neighbors=5)

knn_stand.fit(x_with_stand_scaled,y_with_stand)

# Checking the model's accuracy

print("\nAccuracy score on test set :", accuracy_score(y_with_stand,knn_stand.predict(x_with_stand_scaled)))
df.info()
#set target and feture value
df.head()
X_1 = df.drop(['Loan_Status'], axis=1)



y_1 = df['Loan_Status']
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



#Initializing and Fitting a k-NN model

knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(X_1,y_1)





# for each column

for c in list(df.columns):

    

    # get a list of unique values

    n = df[c].unique()

    

    # if number of unique values is less than 30, print the values. Otherwise print the number of unique values

    if len(n)<5:

        print(c)

        print(n)

    else:

        print(c + ': ' +str(len(n)) + ' unique values')



# Replace'no' with 0 and 'yes' with 1 in 'Vmail_Plan'

df['Loan_Status'] = df['Loan_Status'].replace({'N': 0 , 'Y': 1})



# Replace 'no' with 0 and 'yes' with 1 in 'Churn'

df['Married'] = df['Married'].replace({'No': 0 , 'Yes': 1})
# Replace 'no' with 0 and 'yes' with 1 in 'Churn'

df['Self_Employed'] = df['Self_Employed'].replace({'No': 0 , 'Yes': 1})
# Importing LabelEncoder and initializing it

from sklearn.preprocessing import LabelEncoder



# Set up the LabelEncoder object

enc = LabelEncoder()



# Apply the encoding to the "Accessible" column

df["Education_enc"] = enc.fit_transform(df["Education"])
# Compare the two columns

print(df[["Education_enc", "Education"]].head())
#nex endconder



# Importing LabelEncoder and initializing it

from sklearn.preprocessing import LabelEncoder



# Set up the LabelEncoder object

enc = LabelEncoder()



# Apply the encoding to the "Accessible" column

df["Gender_enc"] = enc.fit_transform(df["Gender"])

df.head()
#save dataset

#filtered_loans.to_csv("processed_data/cleaned_loans_2007.csv",index=False)



df.columns
# Drop the unnecessary features

df = df.drop(df[['Loan_ID','Education','Gender',]], axis=1)
df.head()
df.info()
# Get dummies and save them inside a new DataFrame

Property_Area = pd.get_dummies(df.Property_Area)



# Take a quick look to the first 5 rows of the new DataFrame called departments

print(Property_Area.head())
# Drop the old column "department" as you don't need it anymore

df = df.drop("Property_Area", axis=1)



# Join the new dataframe "Property_Area" to your  dataset: 

df = df.join(Property_Area)
df.head()
# Set the target and features



# Choose the dependent variable column (Loan_Status) and set it as target

target = df.Loan_Status



# Drop column Loan_Status and set everything else as features

features = df.drop("Loan_Status",axis=1)
features.describe()
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

se = pd.DataFrame(scale.fit_transform(features))
features= pd.concat([features, se], axis=1)

features.drop(['Married',

         'Dependents',

         'Self_Employed',

         'ApplicantIncome',

         'CoapplicantIncome',

         'LoanAmount',

         'Loan_Amount_Term',

         'Credit_History',

         'Education_enc',

         'Gender_enc',

         'Rural',

         'Semiurban',

         'Urban'], axis=1, inplace=True)
features.rename(columns={0:'Married',

                          1:'Dependents',

                          2:'Self_Employed',

                          3:'ApplicantIncome',

                          4:'CoapplicantIncome',

                          5:'LoanAmount',

                          6:'Loan_Amount_Term',

                          7:'Credit_History',

                          8:'Education_enc',

                          9:'Gender_enc',

                          10:'Rural',

                          11:'Semiurban',

                          12:'Urban'}, inplace=True)

features.head()
# Import the function for splitting dataset into train and test

from sklearn.model_selection import train_test_split



# Use that function to create the splits both for target and for features

# Set the test sample to be 25% of your observations

target_train, target_test, features_train, features_test = train_test_split(target,features,test_size=0.25,random_state=42)
# Import the classification algorithm

from sklearn.tree import DecisionTreeClassifier



# Initialize it and call model by specifying the random_state parameter

model = DecisionTreeClassifier(random_state=42)



# Apply a decision tree model to fit features to the target

model.fit(features_train, target_train)
# Apply a decision tree model to fit features to the target in the training set

model.fit(features_train,target_train)



# Check the accuracy score of the prediction for the training set

model.score(features_train,target_train)*100



# Check the accuracy score of the prediction for the test set

model.score(features_test,target_test)*100
# Fitting k-NN on our scaled data set

knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(features_train,target_train)

# Checking the model's accuracy

accuracy_score(target_train,knn.predict(features_train))
from sklearn.linear_model import LogisticRegression



log_reg_all = LogisticRegression(solver='liblinear', penalty='l1')

log_reg_all.fit(features_train,target_train)





# Checking the performance of our model on the testing data set

print("\nAccuracy score on test set :", accuracy_score(target_train,log_reg_all.predict(features_train)))
