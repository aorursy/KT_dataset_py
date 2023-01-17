# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("/kaggle/input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv").fillna('0')



test_df = pd.read_csv("/kaggle/input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv").fillna('0')



# I think the fillna('0') introduced the str in the Credit_History. look into it.







train_df.head()
train_df['Loan_Status'].replace({'N':0, 'Y':1}, inplace =True)
train_df['Loan_Status'] = train_df.Loan_Status.astype(int)
train_df['Loan_Status'].head()
train_df['Self_Employed'].replace({'No':0, 'Yes':1}, inplace =True)
train_df['Self_Employed'] = train_df.Self_Employed.astype(int)
train_df['Self_Employed'].head()
train_df.drop('Loan_ID', axis=1, inplace= True)
train_df['Loan_Amount_Term'] = train_df.Loan_Amount_Term.astype(int)
train_df['LoanAmount'] = train_df.LoanAmount.astype(int)
train_df['Dependents'] = train_df['Dependents'].replace('3+', 3)
train_df['Dependents'] = train_df.Dependents.astype(int)
train_df.head(30)
missing_values = train_df.isnull().sum()

missing_values
train_df.Credit_History.unique()
# the problem is with y label (Credit_History)



isinstance('Credit_History', str)
train_df['Credit_History'] = train_df['Credit_History'].replace('0', 0)
#from sklearn.preprocessing import LabelEncoder



#cat_feature = ['Gender','Married', 'Education', 'Self_Employed', 'Loan_Status' ]

#encoder = LabelEncoder()



# Apply the label encoder to each column

#encoded = train_df[cat_feature].apply(encoder.fit_transform)

target_1 = train_df["Gender"].unique()

for index in range(0, len(target_1)):

    train_df["Gender"].replace(target_1[index], index, inplace=True) # this is now a 64bits int
train_df['Gender'].head()
target_2 = train_df["Married"].unique()

for index in range(0, len(target_2)):

    train_df["Married"].replace(target_2[index], index, inplace=True)
train_df['Married'].head()
target_3 = train_df['Education'].unique()

for index in range(0, len(target_3)):

    train_df['Education'].replace(target_3[index], index, inplace=True)
train_df['Education'].head()
#feature_1 = ['Loan_Amount_Term', 'Dependents', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',]



#feature_2 = train_df[feature_1].join(encoded)
train_df.head(10)
train_df.columns

#train_df['CoapplicantIncome'] = train_df['CoapplicantIncome'].replace('+', 0) # check this to know how it affects the dataset
train_df['Loan_Status'].unique()
# Education is not included in this features



features = ['Gender', 'Married', 'Loan_Amount_Term', 

            'Dependents', 'ApplicantIncome', 'CoapplicantIncome','LoanAmount',

             'Self_Employed','Credit_History' ]
y = train_df.Loan_Status



X = train_df[features]
# splitting the dataset



from sklearn.model_selection import train_test_split 



X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state =42, test_size=0.2)
from xgboost import XGBRegressor



model = XGBRegressor()

model.fit(X_train, y_train)  # need to findout the way to remove the str before fitting the data.
from sklearn.metrics import mean_absolute_error

prediction = model.predict(X_valid)

prediction

#mae = mean_absolute_error(y_valid, prediction)

#mae

# the mean absolute error is 35%
from sklearn.ensemble import RandomForestRegressor





forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(X_train, y_train)

preds = forest_model.predict(X_valid)

preds

#print(mean_absolute_error(y_valid, preds)) 

# with mean absolute error we have 33%. Remember the lower the mae, the better
#predictions = forest_model.score(X_valid, y_valid) # this is unacceptable

#predictions



# this score is too low, which suggests that the model is not accurate.

from sklearn import svm

clf1 = svm.SVC()

clf1.fit(X_train, y_train)

clf1.score(X_valid, y_valid)
from sklearn.metrics import f1_score

    

f1_score(y_valid, clf1.predict(X_valid),  average='micro' )  # with 'weight' average the score is about 52%