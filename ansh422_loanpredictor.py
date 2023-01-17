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
train_df=pd.read_csv('/kaggle/input/loan-prediction-practice-av-competition/train_csv.csv')
test_df=pd.read_csv('/kaggle/input/loan-prediction-practice-av-competition/test.csv.csv')
print(train_df.shape)
print(test_df.shape)
train_df.head().T
train_df.describe().T
train_df.info()
train_df.describe(include=['object'])
train_df.isnull().sum(axis=0)
test_df.isnull().sum()
train_df.nunique()
test_df.nunique()
train_df['Property_Area'].value_counts()
target_map={"Y":1, "N": 0}
dataset=[train_df]
for data in dataset:
    data['Loan_Status']=data['Loan_Status'].map(target_map)
cat_cols=['Gender','Married','Dependents','Self_Employed','Credit_History']
for col in cat_cols:
    train_df[col].fillna(train_df[col].mode()[0],inplace=True)
    test_df[col].fillna(test_df[col].mode()[0],inplace=True)
target=train_df['Loan_Status']
train_df=train_df.drop('Loan_Status',1)
gender_map={"Male": 1,"Female": 0}
marry_map={"Yes":1,"No":0}
education_map={"Graduate": 1,"Not Graduate":0}
property_map={"Semiurban":2,"Urban":1,"Rural":0}
dataset=[train_df]
for data in dataset:
    data['Gender']=data['Gender'].map(gender_map)
    data['Married']=data['Married'].map(marry_map)
    data['Self_Employed']=data['Self_Employed'].map(marry_map)
    data['Education']=data['Education'].map(education_map)
    data['Property_Area']=data['Property_Area'].map(property_map)
#dependents contains numeric value except 3+, so we just need to replace 3+ with 3 and then  convert their type to numeric
train_df = train_df.replace({'Dependents': r'3+'}, {'Dependents': 3}, regex=True)
train_df['Dependents']=train_df['Dependents'].astype('float64')
#test_df = train_df.replace({'Dependents': r'3+'}, {'Dependents': 3},regex=True)
#test_df['Dependents']=test_df['Dependents'].astype('float64')
train_df.info()
gender_map={"Male": 1,"Female": 0}
marry_map={"Yes":1,"No":0}
education_map={"Graduate": 1,"Not Graduate":0}
property_map={"Semiurban":2,"Urban":1,"Rural":0}
dataset=[test_df]
for data in dataset:
    data['Gender']=data['Gender'].map(gender_map)
    data['Married']=data['Married'].map(marry_map)
    data['Self_Employed']=data['Self_Employed'].map(marry_map)
    data['Education']=data['Education'].map(education_map)
    data['Property_Area']=data['Property_Area'].map(property_map)
#dependents contains numeric value except 3+, so we just need to replace 3+ with 3 and then  convert their type to numeric
test_df = test_df.replace({'Dependents': r'3+'}, {'Dependents': 3},regex=True)
test_df['Dependents']=test_df['Dependents'].astype('float64')
test_df.info()
test_df.isnull().sum()
train_df.isnull().sum()
train_df['Loan_Amount_Term'].value_counts()
train_df['LoanAmount'].value_counts()
train_df['Loan_Amount_Term'].fillna(360,inplace=True)
train_df['LoanAmount'].fillna(train_df['LoanAmount'].median(),inplace=True)
train_df.isnull().sum()
train_df.info()
test_df.isnull().sum()
test_df['Loan_Amount_Term'].value_counts()
test_df['LoanAmount'].value_counts()
test_df['Loan_Amount_Term'].fillna(360,inplace=True)
test_df['LoanAmount'].fillna(test_df['LoanAmount'].median(),inplace=True)
test_df.isnull().sum()
test_df.info()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inLine
corr=train_df.corr()
colormap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr,cmap=colormap,xticklabels=corr.columns,yticklabels=corr.columns,annot=True)
plt.show()
train_df=train_df.drop('Loan_ID',1)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train,y_val= train_test_split(train_df,target,test_size=0.30, random_state=np.random.randint(0,100))
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
dt_base=DecisionTreeClassifier(max_depth=10,random_state=4)
dt_base.fit(X_train,y_train)

from sklearn import metrics
y_pred=dt_base.predict(X_val)
acc = metrics.accuracy_score(y_val,y_pred)
print(acc)
dt_base.tree_.node_count
param_grid = {
    'max_depth' : range(4,25),
    'min_samples_leaf' : range(20,200,10),
    'min_samples_split' : range(20,200,10),
    'criterion' : ['gini','entropy'] 
}
n_folds = 5
from sklearn.model_selection import GridSearchCV

dt = DecisionTreeClassifier(random_state=np.random.randint(0,100))
grid = GridSearchCV(dt, param_grid, cv = n_folds, return_train_score=True,verbose=3)
#grid.fit(X_train,y_train)
#grid.best_params_
best_tree=DecisionTreeClassifier(criterion='gini',max_depth=4,min_samples_leaf=20,min_samples_split=80,random_state=np.random.randint(0,100))
best_tree.fit(X_train,y_train)
y_pred_best=best_tree.predict(X_val)
acc = metrics.accuracy_score(y_val,y_pred_best)
print(acc)
test_df.info()
loanID=test_df['Loan_ID']
test_df=test_df.drop('Loan_ID',1)
y_final_tree=best_tree.predict(test_df)
submission = pd.DataFrame({
        "Loan_Id": loanID,
        "Loan_Status": y_final_tree
    })
submission.head(10)
submission.to_csv('submission_tree.csv', index=False)