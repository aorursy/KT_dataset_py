import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statistics as st

%matplotlib inline
import os
train = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')

test = pd.read_csv('../input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv')

test
train.describe()

test.describe()
train.isnull().sum()

test.isnull().sum()

train.columns
# combined data for cleansing 

def combined_data():

    train = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')

    test = pd.read_csv('../input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv')

    target = train.Loan_Status

    train.drop('Loan_Status',axis = 1,inplace = True)

    combined = train.append(test)

    combined.reset_index(inplace=True)

    combined.drop(['index', 'Loan_ID'], inplace=True, axis=1)

    return combined



combined = combined_data()

combined

print(combined.isnull().sum())

# data Cleansing of combined data

combined.fillna({

    'Gender':st.mode(combined.Gender),

    'Married':st.mode(combined.Married),

    'Dependents':st.mode(combined.Dependents),

    'Self_Employed':st.mode(combined.Self_Employed),

    'LoanAmount':np.mean(combined.LoanAmount),

    'Credit_History':np.mean(combined.Credit_History),

    'Loan_Amount_Term':np.mean(combined.Loan_Amount_Term)

},inplace=True)

print(combined.isnull().sum())

# plt.plot(combined.isnull().sum()) # this time it is zero

combined
# Data Transformation

# One Hot Encoding for qualitative data

# Gender male = 1 and female = 0

def Encoding_Gender():

    combined.Gender = combined.Gender.map({'Male':1,'Female':0})

# Married = 1 and single = 0

def Encoding_Martial():

    combined.Married = combined.Married.map({'No':0,'Yes':1})

def Encoding_Dependents():

    combined['Single'] = combined.Dependents.map(lambda d:1 if d=='1' else 0)

    combined['Small_Family'] = combined.Dependents.map(lambda d:1 if d=='2' else 0)

    combined['Large_Family'] = combined.Dependents.map(lambda d:1 if d=='3+' else 0)

    combined.drop(['Dependents'], axis=1, inplace=True)

def Encoding_Education():

    combined.Education = combined.Education.map({'Graduate':1,'Not Graduate':0})

def Encoding_Self_Employed():

    combined.Self_Employed = combined.Self_Employed.map({'No':0,'Yes':1})

def Encoding_Total_Income():

    combined['Total_Income'] = combined.ApplicantIncome + combined.CoapplicantIncome

    combined.drop(['ApplicantIncome','CoapplicantIncome'],axis=1,inplace=True)

def Encoding_loan_amount():

    combined['Dept_Income_Ratio'] = combined.Total_Income / combined.LoanAmount

    
Encoding_Gender()
Encoding_Martial()
Encoding_Dependents()
Encoding_Education()
Encoding_Self_Employed()
Encoding_Total_Income()
Encoding_loan_amount()
combined
approved_term = train[train['Loan_Status']=='Y']['Loan_Amount_Term'].value_counts()

unapproved_term = train[train['Loan_Status']=='N']['Loan_Amount_Term'].value_counts()

df = pd.DataFrame([approved_term,unapproved_term])

df.index = ['Approved','Unapproved']

df.plot(kind='bar', figsize=(10,8))
def Process_LoanAmount_term():

    combined['very_short_term'] = combined.Loan_Amount_Term.map(lambda d: 1 if d > 1 and d <= 60 else 0)

    combined['short_term'] = combined.Loan_Amount_Term.map(lambda d:1 if d>60 and d<180 else 0)

    combined['Long_term'] = combined.Loan_Amount_Term.map(lambda d:1 if d>=180 and d<300 else 0)

    combined['very_long_term'] = combined.Loan_Amount_Term.map(lambda d:1 if d>=300 else 0)

    combined.drop(['Loan_Amount_Term'], axis=1, inplace=True)

def Process_Credit_History():

    combined['Credit_History_bad'] = combined.Credit_History.map(lambda x: 1 if x == 0 else 0)

    combined['Credit_History_good'] = combined.Credit_History.map(lambda x: 1 if x == 1 else 0)

    combined['Credit_History_unknown'] = combined.Credit_History.map(lambda x:1 if x== 2 else 0)

    combined.drop(['Credit_History'], axis=1, inplace= True)

def Process_Proparty_Area():

    combined['Property_Rural'] = combined.Property_Area.map(lambda x: 1 if x == 'Rural' else 0)

    combined['Property_Urban'] = combined.Property_Area.map(lambda x: 1 if x == 'Urban' else 0)

    combined['Property_SemiUrban'] = combined.Property_Area.map(lambda x: 1 if x == 'Semiurban' else 0)

    combined.drop(['Property_Area'], axis=1, inplace= True)
Process_LoanAmount_term()
Process_Credit_History()
Process_Proparty_Area()
combined.head()
# Scaling the LoanAmount, Totalincome and Dept Income Ratio

def feature_scaling(df):

    df = df - df.min()

    df = df / df.max()

    return df
combined.LoanAmount = feature_scaling(combined.LoanAmount)

combined.Total_Income = feature_scaling(combined.Total_Income)

combined.Dept_Income_Ratio = feature_scaling(combined.Dept_Income_Ratio)
combined.head(30)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score
def train_test():

    target = train.Loan_Status.map({'Y':1,'N':0})

    train_ = combined.head(614)

    test = combined.iloc[len(target):]

    return train_,target,test
train_,target,test = train_test()
arguments = {

    'bootstrap':False,

    'min_samples_leaf':3,

    'n_estimators':50,

    'min_samples_split': 10,

    'max_features': 'sqrt',

    'max_depth':6

}

model = RandomForestClassifier(**arguments)

model.fit(train_,target)
# k fold cross validtion

x_val = cross_val_score(model, train_, target, cv = 5, scoring='accuracy')

np.mean(x_val)
##np.vectorize Define a vectorized function which takes a nested sequence of objects or numpy arrays as inputs and returns a single numpy array or a tuple of numpy array

output = model.predict(test)

output_df = pd.DataFrame()

result_df = pd.read_csv('../input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv')

output_df['Loan_ID'] = result_df.Loan_ID

output_df['Loan_Status'] = np.vectorize(lambda x:'Yes' if x == 1 else 'NO')(output)

output_df[['Loan_ID','Loan_Status']].to_csv('output.csv',index=False)
pd.read_csv("output.csv")
import pandas as pd

test_Y3wMUE5_7gLdaTN = pd.read_csv("../input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv")

train_u6lujuX_CVtuZ9i = pd.read_csv("../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv")
import pandas as pd

test_Y3wMUE5_7gLdaTN = pd.read_csv("../input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv")

train_u6lujuX_CVtuZ9i = pd.read_csv("../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv")