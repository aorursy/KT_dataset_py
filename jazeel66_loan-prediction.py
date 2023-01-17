## This is just a practice problem for predicting loan application approvals with a relatively small dataset

## We'll use pandas for data manupulations and wrangling and sklearn for fitting machine learning models.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegressionCV, SGDClassifier, LogisticRegression

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix
df= pd.read_csv('../input/train_u6lujuX_CVtuZ9i (1).csv')
df.head()

# checking the dataset
df.drop('Loan_ID', axis=1, inplace= True)

#we'll drop the ID variables as it does not add to the model
df.info()
obj_cols= [*df.select_dtypes('object').columns]

obj_cols.remove('Loan_Status')

# as Loan_Status is a target variable
obj_cols
plt.figure(figsize=(24, 18))



for idx, cols in enumerate(obj_cols):

    

    plt.subplot(3, 3, idx+1)

    

    sns.countplot(cols, data= df, hue='Loan_Status')
num_cols= [*df.select_dtypes(['Int64', 'Float64']).columns]

num_cols.remove('Loan_Amount_Term')

num_cols.remove('Credit_History')

num_cols
plt.figure(figsize=(24, 18))

count = 1



for cols in num_cols:

    

    plt.subplot(3, 2, count)

    

    sns.boxenplot(x='Loan_Status', y= cols, data= df)

    

    count +=1

    

    plt.subplot(3, 2, count)

    

    sns.distplot(df.loc[df[cols].notna(), cols])

    

    count+=1
df.describe()

# we see that LoanAmount, Loan_Amount_Term and Credit History have a some missing values.

"""we also see that there could be possible outliers in the dataset for ApplicantIncome, CoapplicantIncome 

and Loan_Amount."""

# We see that there are places where there is 0 for Coapplicant income, this might help us create a new feature

# Credit history seems to be a binary filed with just 0s and 1s, we will consider this a categorical feature.
df.isna().sum()

"""we see that there are missing values in more than the columns list above, this can also be found from 

df.info() method"""

"""# we'll use a simple imputer for missing values and create a new column for Loan_Amount_Term 

where 1 idicates term 360 months and 0 for rest"""
df.Loan_Status.replace({'Y': 0, 'N': 1}, inplace= True)
df['Loan_Status']= df.Loan_Status.astype(int)
dummies= pd.get_dummies(df, drop_first=True)
# we will now impute values



SimImp = SimpleImputer()



train= pd.DataFrame(SimImp.fit_transform(dummies), columns=dummies.columns)
train.sample(10)
train.info()

# we see that all missing values have been replaced
train['Loan_Term_360']= np.where(train.Loan_Amount_Term == 360, 1, 0)

# we'll create a binary column here for loan amount term and check the data with a count plot

sns.countplot(y='Loan_Term_360', data= train, hue='Loan_Status')

# looks like there is some importance and can be used in the model
train.drop('Loan_Amount_Term', inplace= True, axis= 1)
train.head()

# we'll also create a new variable to called "NoCoapplicantIncome" to check it's significane in the model.
NoCoapplicantIncome= np.where(train['CoapplicantIncome']== 0, 1, 0)
sns.countplot(y=NoCoapplicantIncome, hue=train.Loan_Status)

"""we see that regardless of coapplicant income loans have been rejected in equal amounts,

this variable might not help"""
#we'll split the data to train and test set



obj_train = train.drop(num_cols, axis=1)



# for this model we'll only use the categorical features for training 



X, y = obj_train.drop('Loan_Status', axis=1), obj_train.Loan_Status



X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=123, stratify= y)
logit= LogisticRegressionCV()

logit.fit(X_train, y_train)



logit_pred= logit.predict(X_test)



print(accuracy_score(y_test, logit_pred))



confusion_matrix(y_test, logit_pred)
sgd_clf = SGDClassifier()



sgd_clf.fit(X_train, y_train)



sgd_pred= sgd_clf.predict(X_test)



print(accuracy_score(y_test, sgd_pred))



confusion_matrix(y_test, sgd_pred)
