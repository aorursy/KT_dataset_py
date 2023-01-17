# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
xtrain=pd.read_csv('/kaggle/input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
df=xtrain #Copying original dataset to another df
df
df=df.replace(['Male','Female'],[1,0])
df=df.replace(['Yes','No'],[1,0])
df['Loan_Status']=df['Loan_Status'].replace(['Y','N'],[1,0])
df

df.info()
df.describe()
df.nunique()
df['Gender'].value_counts() #11 missing values---bias towards 489
df['Married'].value_counts() #3 missing values--- replace by 1/ drop them. but i am replacing them
df['Dependents'].value_counts() #15 missing values---bias towards 345
df['Dependents']=df['Dependents'].replace(['3+'],[3])
df
df['Self_Employed'].value_counts() #32 missing values--- bias towards 500 
df['Loan_Amount_Term'].value_counts() #14 missing values--- bias towards 512
df['Credit_History'].value_counts() #50 missing values--- bias towards 475
15/614,32/614,14/614,50/614
# Less than 1% missing values are there
# Replacing all by mode is apt solution except for Loan amt which is replaced by mean.
# Apply feature scaling

(df['Dependents']) # TYpe is object. SO convert into float
df.Dependents=df.Dependents.apply(pd.to_numeric)
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(df.iloc[:,[1,2,3,5,9,10]])
df.iloc[:,[1,2,3,5,9,10]] = imputer.transform(df.iloc[:,[1,2,3,5,9,10]])
gender_mode=df['Gender'].mode()
employed_mode=df['Self_Employed'].mode()
LoanAmount_mean=df['LoanAmount'].mean()
LoanAmountTerm_mean=df['Loan_Amount_Term'].mean()
credit_mode=df['Credit_History'].mode()
dep_mode=df['Dependents'].mode()


df.info()
#Dropping LoanId as it dosen't affect prediction
df=df.drop(columns='Loan_ID')
df
columntitle=["Education","Property_Area","Gender","Married","Dependents","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History","Loan_Status"]
df=df.reindex(columns=columntitle)
df
pd.isnull(df).sum() > 0 
#Check nan valuesin any column before label encoding
imputer_loan=SimpleImputer(missing_values=np.nan,strategy="mean")
imputer_loan.fit(df.iloc[:,[8]])
df.iloc[:,[8]] = imputer_loan.transform(df.iloc[:,[8]])
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
label_enc=LabelEncoder()
df['Education']=label_enc.fit_transform(df['Education'])
df['Property_Area']=label_enc.fit_transform(df['Property_Area'])

# Instead of this can apply colmntransformer only

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('onehot', OneHotEncoder(categories='auto'), [0,1])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
X = ct.fit_transform(df)
ct

# 2 diffnames for education and 3 diff categories for propertyname

column_name_afterEncoding=columntitle
column_name_afterEncoding.insert(0,"Education")
column_name_afterEncoding.insert(2,"Property_Area")
column_name_afterEncoding.insert(3,"Property_Area")
print(column_name_afterEncoding)
encoded_df=pd.DataFrame(X,columns=column_name_afterEncoding)
encoded_df

df['Education'].notnull().nunique()
encoded_df_X=encoded_df.iloc[:,:14]
encoded_df_X
encoded_df_Y=encoded_df.iloc[:,-1:]
encoded_df_Y
from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(random_state=0)

classifier.fit(encoded_df_X,encoded_df_Y)
xtest=pd.read_csv('/kaggle/input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv')
xtest
xtest.info()
xtest=xtest.drop(columns=['Loan_ID'])
xtest['Dependents']=xtest['Dependents'].replace(['3+'],[3])
# xtest['Loan_Status']=xtest['Loan_Status'].replace(['Y','N'],[1,0])
xtest=xtest.replace(['Yes','No'],[1,0])
xtest=xtest.replace(['Male','Female'],[1,0])
xtest.Dependents=xtest.Dependents.apply(pd.to_numeric)
(xtest['Dependents']) # TYpe is object. SO convert into float
xtest
xtest.info()
# Gender
xtest['Gender'].fillna(gender_mode[0],inplace=True) #[0] is used as series is o/p
# Self_Employed
xtest['Self_Employed'].fillna(employed_mode[0],inplace=True)
# Loan_Amount
xtest['LoanAmount'].fillna(LoanAmount_mean,inplace=True)
# Loan_Amount_Term
xtest['Loan_Amount_Term'].fillna(LoanAmountTerm_mean,inplace=True)
# Credit_History
xtest['Credit_History'].fillna(credit_mode[0],inplace=True)
# Dependents
xtest['Dependents'].fillna(dep_mode[0],inplace=True)

xtest.info()
# Rearranging columns acc. to  df
columntitle=["Education","Property_Area","Gender","Married","Dependents","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]
xtest=xtest.reindex(columns=columntitle)


xtest
# Categorical_Var
xtest['Education']=label_enc.fit_transform(xtest['Education']) #actually TANSFORM ONLY TO BEUSED NOT FIT_TANSFORM
xtest['Property_Area']=label_enc.fit_transform(xtest['Property_Area'])
xtest

ct1 = ColumnTransformer(
    [('onehot', OneHotEncoder(categories='auto'), [0,1])],   # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                                         # Leave the rest of the columns untouched
)
X1 = ct1.fit_transform(xtest)

X1.shape

Xtest_columntitle=["Education","Property_Area","Gender","Married","Dependents","Self_Employed","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]

Xtest_column_name_afterEncoding=Xtest_columntitle
Xtest_column_name_afterEncoding.insert(0,"Education")
Xtest_column_name_afterEncoding.insert(2,"Property_Area")
Xtest_column_name_afterEncoding.insert(3,"Property_Area")
encode_xtest=pd.DataFrame(X1,columns=Xtest_column_name_afterEncoding)
encode_xtest
encode_xtest.info()
x_test_pred=classifier.predict(encode_xtest)
x_test_pred
pd.isnull(encode_xtest).sum() > 0 

p2= pd.DataFrame({'Sur': x_test_pred.astype(int)})
p2
p2['Sur'].value_counts()
p2.to_csv('loan.csv',index=False)




df=df.dropna()
df=df.reset_index()
df=df.drop(columns='index')
df=df.drop(columns='Loan_ID')
df=df.drop(columns=['Education','Property_Area'])
df
x=df.iloc[:,0:9]
x
y=df.iloc[:,-1]
y=pd.DataFrame(y)
y

xtest=pd.read_csv('/kaggle/input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv')
xtest
xtest.info()
xtest=xtest.drop(columns=['Loan_ID','Education','Property_Area'])
xtest
xtest=xtest.replace(['Male','Female'],[1,0])
xtest=xtest.replace(['Yes','No'],[1,0])
xtest=xtest.dropna()
xtest=xtest.reset_index()
xtest
xtest=xtest.drop(columns='index')
# df=df.drop(columns='index')
xtest['Dependents']=xtest['Dependents'].replace(['3+'],[3])
x_test_pred=classifier.predict(xtest)
x_test_pred
# from sklearn.metrics import confusion_matrix 

# cm=confusion_matrix()
