import numpy as np
import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
Variable_Description=pd.DataFrame()
Variable_Description['Name']=['Loan_ID','Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']
Variable_Description['Description']= ['Unique Loan ID','Male/ Female','Applicant married (Y/N)',
                                      'Number of dependents','Applicant Education (Graduate/ Under Graduate)',
                                      'Self employed (Y/N)','Applicant income','Coapplicant income',
                                      'Loan amount in thousands','Term of loan in months','credit history meets guidelines',
                                      'Urban/ Semi Urban/ Rural','Loan approved (Y/N) (Target)']
Variable_Description
train = pd.read_csv('/kaggle/input/loan-prediction-practice-av-competition/train_csv.csv')
train.head()
test = pd.read_csv('/kaggle/input/loan-prediction-practice-av-competition/test.csv.csv')
test.head()
df=pd.concat([train,test],axis=0,sort=False,ignore_index=True)
df.head()
train.shape
test.shape
train.nunique()
# Since Loan ID is unique for every customer it won't play any role in predicting the loan status, hence we can remove this column.
train=train.drop('Loan_ID',axis=1)
train.info()
X=train.drop('Loan_Status',axis=1)
y=train['Loan_Status']
fig, axes = plt.subplots(2, 3)
axes = axes.flatten()

j=0
for i in train.drop('Loan_Status',axis=1).columns:    
    if train[i].dtype=='O':
        train[i].value_counts().plot(kind='pie',autopct='%.2f%%',ax=axes[j],figsize=(20,10))
        j+=1
        
plt.show()                      
fig, axes = plt.subplots(2, 3, figsize = (15,7))
axes = axes.flatten()

j=0
for i in train.drop('Loan_Status',axis=1).columns:    
    if train[i].dtype=='O':
        sns.countplot(x=train[i],hue=train['Loan_Status'],ax=axes[j])
        j+=1
        
plt.show()
sns.countplot(x=train['Loan_Amount_Term'])
plt.show()
sns.countplot(x=train['Credit_History'],hue=train['Loan_Status'])  
plt.show()
corr=train.corr()
sns.heatmap(corr,annot=True)
plt.show()
df.isnull().sum()
# Filling null values with random and median values
df['Gender']=df['Gender'].fillna(np.random.choice(df['Gender']))
df['Married']=df['Married'].fillna(np.random.choice(df['Married']))
df['Dependents']=df['Dependents'].fillna(np.random.choice(df['Dependents']))
df['Credit_History']=df['Credit_History'].fillna(np.random.choice(df['Credit_History']))
df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df.Loan_Amount_Term.median())
df.groupby('Self_Employed').median()
df.loc[(df['ApplicantIncome']<4758) & (df['CoapplicantIncome']>=646),'Self_Employed']=df.loc[(df['ApplicantIncome']<4758) & (df['CoapplicantIncome']>=646),'Self_Employed'].fillna('No')
df.loc[(train['ApplicantIncome']>=4758) & (df['CoapplicantIncome']<646),'Self_Employed']=df.loc[(df['ApplicantIncome']>=4758) & (df['CoapplicantIncome']<646),'Self_Employed'].fillna('Yes')
df['Self_Employed'].isnull().sum()
# The rest of the null values can be filled with random values
df['Self_Employed']=df['Self_Employed'].fillna(np.random.choice(df['Self_Employed']))
df['Self_Employed'].isnull().sum()
corr=df.corr()
corr['LoanAmount']
# Since Loan amount have highest correlation with ApplicantIncome we will fill the null values with these
from sklearn.ensemble import RandomForestRegressor

rf=RandomForestRegressor(random_state=0)

# New data frame without null values of loan amount
train_without_null=df[['LoanAmount','ApplicantIncome','CoapplicantIncome']].dropna()
X_train=train_without_null[['ApplicantIncome','CoapplicantIncome']]
y_train=train_without_null['LoanAmount']


# applying model
rf.fit(X_train,y_train)

# Predicting on the whole data set and saving it as a new column
df['LoanAmount_pred']=pd.DataFrame(rf.predict(df[['ApplicantIncome','CoapplicantIncome']]))
# Replacing only the null values with the predicted values 
df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount_pred'])
from sklearn.metrics import r2_score
r2_score(df['LoanAmount'],df['LoanAmount_pred'])
df.drop('LoanAmount_pred',axis=1,inplace=True)
df.isnull().sum()
# We convert the dependents column to integer value and change 3+ to 3
df['Dependents']=df['Dependents'].replace('3+',3).astype('int64')
# Checking for the columns whose skewness lies outside -0.5 and +0.5.
df.skew()[abs(df.skew())>0.5]
# Distribution of highly skewed numerical features.
for i in ['ApplicantIncome' ,'CoapplicantIncome','LoanAmount']:
    sns.distplot(df[i])
    plt.show()
# To use boxcox the data must be positive, so we check if the minimum values is less than 0 or not
for i in  ['ApplicantIncome' ,'CoapplicantIncome','LoanAmount','Loan_Amount_Term']:
    if df[i].min()<=0:
        print(i)
    
# Since 'ApplicantIncome' and 'CoapplicantIncome' have 0/negative value we make the minimum value to 1
df['ApplicantIncome']=(df['ApplicantIncome']-df['ApplicantIncome'].min()+1)
df['CoapplicantIncome']=(df['CoapplicantIncome']-df['CoapplicantIncome'].min()+1)
# Now we apply boxcox to the skewed columns
from scipy import stats 
from scipy.stats import boxcox
lambdas={}
for i in ['ApplicantIncome' ,'CoapplicantIncome','LoanAmount','Loan_Amount_Term']:
    df[i],lambdas[i]=stats.boxcox(df[i])
    
df=df.drop('Loan_ID',axis=1)
df['Loan_Status']=df['Loan_Status'].replace({'Y':1,'N':0})
df=pd.get_dummies(df,drop_first=True)
train=df.iloc[0:614,:]
test=df.iloc[614:,:]

X_train=train.drop('Loan_Status',axis=1)
y_train=train['Loan_Status']
X_test=test.drop('Loan_Status',axis=1)
y_test=test['Loan_Status']
# Checking if the data is balanced or not
train['Loan_Status'].value_counts()
# ! pip install imblearn
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state = 2) 
X_train_smote, y_train_smote = sm.fit_sample(X_train, y_train) 
print('Before')
print()
print(y.value_counts())

print('After')
print()
print(y_train_smote.value_counts())

#split into 70:30 ratio 
X_train_sm, X_val_sm, y_train_sm, y_val_sm = train_test_split(X_train_smote, y_train_smote, test_size = 0.3, random_state = 0)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier


models = []

models.append(('LR', LogisticRegression()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RFC', RandomForestClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('XGBoost', XGBClassifier()))
models.append(('GBoost',GradientBoostingClassifier()))
models.append(('ADA',AdaBoostClassifier()))


from sklearn.model_selection import cross_val_score

cv_scores={}
for name, model in models:
    cv_results = cross_val_score(model, X_train_smote, y_train_smote, cv=5, scoring='accuracy',n_jobs=-1)
    cv_scores[name]="%f (%f)" % (cv_results.mean(), cv_results.std())
cv_scores
# Hyper parameter tuninig for random forest
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier()
params= {'n_estimators':sp_randint(0,150),
         'max_depth':sp_randint(2,12),
        'min_samples_split':sp_randint(5,50)} 
rsearch= RandomizedSearchCV(RFC,param_distributions=params,cv=3,scoring='accuracy',n_jobs=-1)
rsearch.fit(X_train_smote,y_train_smote)
rsearch.best_params_
model = RandomForestClassifier(**rsearch.best_params_)
model.fit(X_train_smote, y_train_smote)
y_pred_RFC_H= model.predict(X_test)
# Feature Importances_
imp = pd.DataFrame(model.feature_importances_, columns=["Importance"])
imp.index = X_train_smote.columns
imp=imp.sort_values(by='Importance', ascending=True)
imp.plot(kind='barh',figsize=(10,5))
plt.show()
# Solution
test = pd.read_csv('/kaggle/input/loan-prediction-practice-av-competition/test.csv.csv')
y_pred_RFC_H=pd.DataFrame(y_pred_RFC_H).replace({0:'N',1:'Y'})
solution = pd.DataFrame({'Loan_ID':test['Loan_ID'],'Loan_Status':y_pred_RFC_H[0]})
solution
#solution.to_csv('y_pred_RFC_H.csv',index=False)