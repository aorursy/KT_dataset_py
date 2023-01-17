import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv('../input/bank-loan2/madfhantr.csv')#load dataset
df.head(10)#to see first 10 rows
df.info()
df.shape
# let's describe the data

df.describe().T
# Let's check for null values

df.isnull().sum()
# let's deal with the null values

df['Gender'].fillna((df['Gender'].mode()[0]),inplace=True)
sns.heatmap(df.isnull())
df['Married'].fillna((df['Married'].mode()[0]),inplace=True)
df['Dependents'].value_counts()
df['Dependents'].fillna((df['Dependents'].mode()[0]),inplace=True)
# replacing '+' from Dependent column

df['Dependents']=df['Dependents'].apply(lambda x:str(x).replace('+','')if '+' in str(x) else str(x))

df['Dependents']=df['Dependents'].apply(lambda x:int(x))


df.isnull().sum()
df['Self_Employed'].fillna((df['Self_Employed'].mode()[0]),inplace=True)
df['LoanAmount'].fillna((df['LoanAmount'].median()),inplace=True)
df['Loan_Amount_Term'].fillna((df['Loan_Amount_Term'].median()),inplace=True)
df['Credit_History'].fillna((df['Credit_History'].median()),inplace=True)
# Let's check onece again if any column left with the null values

df.isnull().sum()
df.info()
df['Gender'].value_counts()
df['Gender']=df['Gender'].replace({'Male':0,'Female':1})
df.head()
df['Gender'].value_counts()
df['Education'].value_counts()
df['Education']=df['Education'].replace({'Graduate':0,'Not Graduate':1})
df.head()
df['Self_Employed'].value_counts()
df['Self_Employed']=df['Self_Employed'].replace({'No':0,'Yes':1})
df['Married'].value_counts()
df['Married']=df['Married'].replace({'Yes':0,'No':1})
df.head()
df['Loan_Status']=df['Loan_Status'].replace({'N':0,'Y':1})
df.head()
df1=pd.get_dummies(df,drop_first=True)
df1.head()
df1.columns
df1['Loan_Status'].value_counts()
X=df1.drop('Loan_Status',axis=1)
y=df1['Loan_Status']
X.shape,y.shape
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
X_train.shape,y_train.shape
from sklearn.linear_model import LogisticRegression

log=LogisticRegression()
log.fit(X_train,y_train)
pred=log.predict(X_test)
from sklearn.metrics import r2_score

print('the score is',log.score(X_test,y_test))
print('the r2score is ',r2_score(y_test,pred))
from sklearn.metrics import accuracy_score

print('the model accuracy is',accuracy_score(y_test,pred))
log.intercept_
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred)
from sklearn.metrics import classification_report

print(classification_report(y_test,pred))
from sklearn import metrics
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(6, 4))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return fpr, tpr, thresholds
draw_roc(y_test,pred)