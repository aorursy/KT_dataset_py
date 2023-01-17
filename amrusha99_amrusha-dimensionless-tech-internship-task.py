import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

train_data = pd.read_csv('/kaggle/input/credit-risk-modeling-case-study/CRM_TrainData.csv')
train_data.head()
#Dropping columns

train_data=train_data.drop('Loan ID', axis=1)

train_data=train_data.drop('Customer ID', axis=1)

train_data=train_data.drop('Months since last delinquent', axis=1)
train_data.dtypes
#Dropping rows with na values

train_data=train_data.dropna()
#Replacing special characters/non-dtype characters

train_data['Monthly Debt'] = pd.to_numeric(train_data['Monthly Debt'].astype(str).str.replace('$',''), errors='coerce').fillna(0).astype(float)

train_data['Maximum Open Credit'] = pd.to_numeric(train_data['Maximum Open Credit'].astype(str).str.replace('#VALUE!',''), errors='coerce').fillna(0).astype(float)
train_data.dtypes
train_data.shape
#Separating dependent and independent variables

X=train_data.drop('Loan Status', axis=1)

y=train_data['Loan Status']
#Scaling the data

from sklearn.preprocessing import MinMaxScaler

X_cols=['Current Loan Amount','Credit Score','Annual Income','Monthly Debt','Years of Credit History','Number of Open Accounts','Number of Credit Problems','Current Credit Balance','Maximum Open Credit','Bankruptcies','Tax Liens']

minmax=MinMaxScaler()

X_quant=minmax.fit_transform(X[X_cols])

X_quant=pd.DataFrame(X, columns=X_cols)
X_quant.shape
X_quant.head()
X=pd.concat([X_quant, X['Term'], X['Years in current job'], X['Home Ownership'], X['Purpose']], axis=1)
X.shape
#One Hot Encoding

X=pd.get_dummies(X, drop_first=True)
X.shape
#Label encoding binary dependent variable

from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder()  

train_data['Loan Status']= label_encoder.fit_transform(train_data['Loan Status']) 

y=train_data['Loan Status']
y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
from sklearn.ensemble import RandomForestClassifier

# Building model with 1000 decision trees

rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
rf_mod=rf.fit(X_train, y_train)
test_score = rf.score(X_test,y_test)

test_score
test_data = pd.read_csv('/kaggle/input/credit-risk-modeling-case-study/CRM_TestData.csv')
test_data.head()
test_data.shape
test_data.isna().sum()
#Dropping columns due to too many na values or irrelevance

test_data=test_data.drop('Customer ID', axis=1)

test_data=test_data.drop('Months since last delinquent', axis=1) 

test_data=test_data.drop('Unnamed: 2',axis=1)
test_data['Maximum Open Credit'] = pd.to_numeric(test_data['Maximum Open Credit'].astype(str).str.replace('#VALUE!',''), errors='coerce').fillna(0).astype(float)
test_data['Monthly Debt'] = pd.to_numeric(test_data['Monthly Debt'].astype(str).str.replace('$',''), errors='coerce').fillna(0).astype(float)
test_data.isna().sum()
test_data.dtypes
#Filling na values with respective median of columns 

test_data['Credit Score'] = test_data['Credit Score'].fillna(test_data["Credit Score"].median()) 

test_data['Annual Income'] = test_data['Annual Income'].fillna(test_data["Annual Income"].median())

test_data['Bankruptcies'] = test_data['Bankruptcies'].fillna(test_data["Bankruptcies"].median())

test_data['Tax Liens'] = test_data['Tax Liens'].fillna(test_data["Tax Liens"].median())
test_data.isna().sum()
#Replacing na values with most of the categorical values

x=test_data['Years in current job']

test_data=test_data.apply(lambda x: x.fillna(x.value_counts().index[0]))
test_data.isna().sum()
test_data.shape
loan_ids=test_data['Loan ID']
X_df=test_data.drop('Loan ID', axis=1)
X_df.dtypes
#Fitting model on Training set (CRM_TrainData)

rf.fit(X,y)
#Scaling relevant features in test data

from sklearn.preprocessing import MinMaxScaler

X_df_cols=['Current Loan Amount','Credit Score','Annual Income','Monthly Debt','Years of Credit History','Number of Open Accounts','Number of Credit Problems','Current Credit Balance','Maximum Open Credit','Bankruptcies','Tax Liens']

minmax=MinMaxScaler()

X_df_quant=minmax.fit_transform(X_df[X_df_cols])

X_df_quant=pd.DataFrame(X_df, columns=X_df_cols)
X_df=pd.concat([X_df_quant, X_df['Term'], X_df['Years in current job'], X_df['Home Ownership'], X_df['Purpose']], axis=1)
#One Hot Encoding

X_df=pd.get_dummies(X_df, drop_first=True)
X_df.shape
#Predicting Y for Test Data

y_pred=rf.predict(X_df)
y_pred=pd.DataFrame(y_pred)
y_pred
loan_ids=pd.DataFrame(loan_ids)
loan_ids.to_csv("Test_Loan_IDS_2.csv") 
y_pred.to_csv("Y_pred_test_2.csv") # O : "Charged Off" , 1 : "Fully Paid"
#The y_preds were converted back from 0 and 1 to Charged Off and Fully Paid using IF statement in Excel.

#Test_Loan_IDS_2 and the converted Loan Status values were concatenated in a final csv used for Submission.

#Final csv submitted: Amrusha_dimTechint_NMIMS.csv