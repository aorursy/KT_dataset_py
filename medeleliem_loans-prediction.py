import numpy as np
import pandas as pd
train_df = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')
train_df.head()
train_df=train_df.drop("Loan_ID", axis=1)
train_df['Loan_Status']=pd.get_dummies(train_df['Loan_Status'])
train_df.info()
train_df["Credit_History"] = train_df["Credit_History"].astype(object)
train_df.isnull().sum()
train_df['LoanAmount']=train_df['LoanAmount'].fillna(train_df['LoanAmount'].mean())
train_df['Loan_Amount_Term']=train_df['Loan_Amount_Term'].fillna(train_df['Loan_Amount_Term'].mean())
train_df.shape
train_df=train_df.dropna()
import seaborn as sns
import matplotlib.pyplot as plt

categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History']
fig,axes = plt.subplots(4,2,figsize=(12,15))
for idx,cat_col in enumerate(categorical_columns):
    row,col = idx//2,idx%2
    sns.countplot(x=cat_col,data=train_df,hue='Loan_Status',ax=axes[row,col])


plt.subplots_adjust(hspace=1)
cat_df = train_df[['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History','Loan_Status']]
num_df = train_df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Loan_Status']]
cat_df.head()
num_df.head()
num_df.describe()
num_df.corr(method = 'spearman')
dummy_variable_1 = pd.get_dummies(cat_df['Dependents'])
dummy_variable_1.head()
import seaborn as sns

corr = train_df.corr(method = 'spearman')

sns.heatmap(corr, annot = True)

plt.show()
from sklearn import tree
X=train_df[[]]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)