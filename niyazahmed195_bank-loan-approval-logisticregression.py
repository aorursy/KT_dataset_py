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
import sklearn

df=pd.read_csv('/kaggle/input/bank-loan-approval/train.csv')
df
df.info()
df.isnull().sum()
df.describe()
df=df.drop(['Loan_ID'],axis=1)
import seaborn as sns

sns.countplot(df.Gender)
df['Gender'].fillna('Male',inplace=True)
sns.countplot(df.Married)
df['Married'].fillna('Yes',inplace=True)
sns.countplot(df.Dependents)
df['Dependents'].fillna('0',inplace=True)
sns.countplot(df.Education)
sns.countplot(df.Self_Employed)
df['Self_Employed'].fillna('No',inplace=True)
df['Loan_Amount_Term'].fillna(360.0,inplace=True)
df.dropna(subset=['Credit_History','LoanAmount'], inplace=True)
df.isnull().sum()
def convert(x):

    if x=='3+':

        return int(x[0])

    else:

        return int(x)
df['Dependents']=df['Dependents'].map(convert)
df['EMI']=(df['LoanAmount']*0.09*(1.09**df['Loan_Amount_Term']))/(1.09**(df['Loan_Amount_Term']-1))
sns.distplot(df.ApplicantIncome)
df['ApplicantIncome']=df['ApplicantIncome'].map(lambda x:float(x/1000))

import numpy as np

df['Income_log']=np.log(df.ApplicantIncome)
sns.distplot(df.Income_log)
df.Income_log.describe()
sns.distplot(df['EMI'])
df.EMI.describe()
df['EMI_log']=np.log(df['EMI'])
sns.distplot(df['EMI_log'])
df=df[df['EMI_log']>0]
df=df[df['Income_log']>0]
from scipy.stats import chi2_contingency 

def  calc_csqu(cate_data,d2):

    res={}

    for d1 in cate_data:

        dataset_table=pd.crosstab(df[d1],df[d2])

        data=dataset_table.values   

        stat, p, dof, expected = chi2_contingency(data)

        res[d1]=p

    return res
cate_data=['Education','Gender','Self_Employed','Property_Area','Credit_History','Married']

all_cate_p=calc_csqu(cate_data,'Loan_Status')



for key,value in all_cate_p.items():

    print(key,'   :',value)
df.drop(['Gender','Self_Employed'],inplace=True,axis=1)
from scipy.stats import f_oneway

def anova_test(nume_data):

    yes=df[df.Loan_Status=='Y']

    no=df[df.Loan_Status=='N']

    res={}

    for d1 in nume_data:

        _,p=f_oneway(yes[d1],no[d1])

        res[d1]=p

    return res
df.head(3)
nume_data=['Dependents','Income_log','EMI_log']

res=anova_test(nume_data)

for key,values in res.items():

    print(key)

    print(values)
df.drop(['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','EMI'],inplace=True,axis=1)
df.head(2)
df=pd.get_dummies(df,drop_first=True)



df.head(2)
y=df['Loan_Status_Y']

df.drop(['Loan_Status_Y'],inplace=True,axis=1)

X=df
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from imblearn.over_sampling import SMOTE

over=SMOTE(sampling_strategy=0.5826530612244898)#used GridSearchCV

X_train,y_train=over.fit_resample(X_train,y_train)

model=LogisticRegression(C=1,solver='liblinear')

model.fit(X_train, y_train)
y_pred=model.predict_proba(X_test)
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

y_pred_1=pd.Series(y_pred[:,1])

fpr, tpr, thresholds = roc_curve(y_test, y_pred_1)

import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()

plot_roc_curve(fpr, tpr)
from sklearn.metrics import accuracy_score

accuracy_ls = []

for thres in thresholds:

    y_pred_2 = np.where(y_pred_1>thres,1,0)

    accuracy_ls.append(accuracy_score(y_test, y_pred_2, normalize=True))

    

accuracy_ls = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls)],

                        axis=1)

accuracy_ls.columns = ['thresholds', 'accuracy']

accuracy_ls.sort_values(by='accuracy', ascending=False, inplace=True)

a=accuracy_ls['thresholds'][30]

y_pred_2 = np.where(y_pred_1>a,1,0)
from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred_2)