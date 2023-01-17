#Problem Statement
# Steps for solving the problem

#Loading Packages
import pandas as pd 
import numpy as np                     # For mathematical calculations 
import seaborn as sns                  # For data visualization 
import matplotlib.pyplot as plt        # For plotting graphs 
%matplotlib inline 
import warnings                        # To ignore any warnings warnings.filterwarnings("ignore")
train_original = pd.read_csv("../input/loan-prediction-train-data/train_ctrUa4K.csv")
train = train_original
train.head()
# Looking into the number of rows and columns of data set train which syas training data set have 614
#rows and 13 columns
train.shape
#Reading the name of columns in data set 
train.columns
test = pd.read_csv("../input/loan-prediction-test-data/test_lAUu6dG.csv")
test.head()
test.shape  #
test.columns
# print data types for each variable in train data set 
train.dtypes

# looking for  the target variable distribution in the train data set
train['Loan_Status'].value_counts()
# Normalize can be set to True to print proportions instead of number 
train['Loan_Status'].value_counts(normalize=True)
# Add title and axis names
plt.title('Loan Status Bar Plot')
plt.xlabel('Loan Status Y - Yes or N- No')
plt.ylabel('Loan Status Count')

train['Loan_Status'].value_counts().plot.bar(color=['green', 'red'],edgecolor='blue')
plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(title='Gender')
plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title='Married')
plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Self Employed')
plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title='Credit History')
plt.figure(2)
plt.subplot(1,3,1)
train['Dependents'].value_counts(normalize=True).plot(figsize=(30,10),kind='bar',title= 'Dependents')
plt.subplot(1,3,2)
train['Education'].value_counts(normalize=True).plot(kind='bar',title= 'Education')
plt.subplot(1,3,3)
train['Property_Area'].value_counts(normalize=True).plot(kind='bar',title= 'Property Area')

plt.figure(1)
plt.subplot(131)
sns.distplot(train['ApplicantIncome'],label="Applicant Income analysis")
plt.subplot(133)
train['ApplicantIncome'].plot(kind='box',figsize=(16,5),label="Applicant Income analysis")
train.boxplot(column='ApplicantIncome',by='Education')
plt.suptitle("")
plt.figure(1)
plt.subplot(121)
sns.distplot(train['CoapplicantIncome'])
plt.subplot(122)
train['CoapplicantIncome'].plot(kind='box', figsize=(16,5))
plt.figure()
plt.subplot(121)
sns.distplot(train['LoanAmount'])
plt.subplot(122)
train['LoanAmount'].plot(kind='box',figsize=(16,5))
Gender = pd.crosstab(train['Gender'],train['Loan_Status'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

Married = pd.crosstab(train['Married'],train['Loan_Status'])
Married.div(Married.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)

Dependents = pd.crosstab(train['Dependents'],train['Loan_Status'])
Dependents.div(Dependents.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)

Education = pd.crosstab(train['Education'],train['Loan_Status'])
Education.div(Education.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)

Self_Employed = pd.crosstab(train['Self_Employed'],train['Loan_Status'])
Self_Employed.div(Self_Employed.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)

Credit_History = pd.crosstab(train['Credit_History'],train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)

Property_Area = pd.crosstab(train['Property_Area'],train['Loan_Status'])
Property_Area.div(Property_Area.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)

train.groupby('Loan_Status')['ApplicantIncome'].mean().plot(kind='bar',title="Loan Status w.r.t Applicant Income")
#Declared four bins for for diff group
bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High', 'Very high']
train['Income_bin']=pd.cut(train['ApplicantIncome'],bins,labels=group)
train.head()
income_bin = pd.crosstab(train['Income_bin'],train['Loan_Status'])
income_bin.div(income_bin.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True,grid=True)
plt.xlabel('Income group')
plt.ylabel('Percentage')


#train.groupby('Loan_Status')['CoapplicantIncome'].mean().plot(kind='bar')
bins=[0,1000,3000,42000] 
group=['Low','Average','High'] 
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)
Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
plt.xlabel('Income group')
plt.ylabel('Percentage')
train['total_income'] = train['ApplicantIncome'] + train['CoapplicantIncome']
bins = [0,2500,4000,6000,8100]
group = ['Low','Average','High', 'Very high']
train['total_income_bin'] = pd.cut(train['total_income'],bins,labels=group)

# crosstab plot between total_income_bin vs Loasn_Status
total_income_bin = pd.crosstab(train['total_income_bin'],train['Loan_Status'])

total_income_bin.div(total_income_bin.sum(1).astype(float),axis=0).plot(kind='bar',stacked=True)
train.groupby('Loan_Status')['LoanAmount'].mean().plot(kind='bar')
bins=[0,100,200,700] 
group=['Low','Average','High'] 
train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status']) 
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('LoanAmount')
plt.ylabel('Percentage')
train.columns
train = train.drop(['Income_bin', 'Coapplicant_Income_bin', 'total_income_bin', 'LoanAmount_bin', 'total_income'], axis=1)
train['Dependents'].replace('3+',3,inplace=True)
test['Dependents'].replace('3+',3,inplace=True)
train['Loan_Status'].replace('Y',1,inplace=True)
train['Loan_Status'].replace('N',0,inplace=True)

matrix = train.corr() 
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");
train.isnull().sum()
train['Gender'].fillna(train['Gender'].mode()[0],inplace=True)
train['Married'].fillna(train['Married'].mode()[0],inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0],inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace=True)

#----------- For Test Data
test['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
train['Loan_Amount_Term'].value_counts()
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(),inplace=True) 
# Replaced null value by Median as mean won't be best representative as outliers are there
train.isnull().sum()
train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=100) 
#Below graph is approx normal distributed. For the test data set also we need to take log of LoanAmount field 
#to make the distribution normal
test['LoanAmount_log'] = np.log(test['LoanAmount'])
train.head()
train = train.drop('Loan_ID',axis=1)
test = test.drop('Loan_ID',axis=1)
x = train.drop('Loan_Status',axis=1)
y = train['Loan_Status']
y.head()
x = pd.get_dummies(x)
train = pd.get_dummies(train)
test = pd.get_dummies(test)
#We will use the train_test_split function from sklearn to divide our train dataset. So, first let us import train_test_split.
# x_cv and y_cv is validation data set component
from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(x,y, test_size =0.3)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
#Let’s predict the Loan_Status for validation set and calculate its accuracy.
pred_cv = model.predict(x_cv)

accuracy_score(y_cv,pred_cv)
pred_test = model.predict(test)

#Lets import result from test file.

submission=pd.read_csv("../input/submission/sample_submission_49d68Cx.csv")
test_original = pd.read_csv("../input/loan-prediction-test-data/test_lAUu6dG.csv")
submission['Loan_Status']=pred_test 
submission['Loan_ID']=test_original['Loan_ID']
submission['Loan_Status'].replace(0,'N',inplace=True)
submission['Loan_Status'].replace(1,'Y',inplace=True)
#converting submission fiel to csv file
pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')
#Let’s import StratifiedKFold from sklearn and fit the model.
from sklearn.model_selection import StratifiedKFold
i = 1
accuracy = []

skf = StratifiedKFold(n_splits=5, random_state=1, shuffle = True)
#skf.get_n_splits(x,y)

for train_index,test_index in skf.split(x,y):
    #print("train: ",train_index," Validation : ",test_index)
    print('\n{} of kfold {}'.format(i,skf.n_splits))
    x1_train,x1_val = x.iloc[train_index],x.iloc[test_index]
    y1_train,y1_val = y.iloc[train_index],y.iloc[test_index]
    model = LogisticRegression(max_iter=200 ,random_state=1)
    model.fit(x1_train,y1_train)
    prediction_test = model.predict(x1_val)
    score = accuracy_score(y1_val,prediction_test)
    print('accuracy_score',score)
    i+=1 
    pred_test = model.predict(test) 
    #print("pred_test",pred_test)
    pred=model.predict_proba(x1_val)[:,1]
    #print("pred",pred)
    
#print(score)

from sklearn import metrics
fpr, tpr, _ = metrics.roc_curve(y1_val, pred)
auc = metrics.roc_auc_score(y1_val,pred)
plt.figure(figsize=(12,8)) 
plt.plot(fpr,tpr,label="validation, auc="+str(auc)) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.legend(loc=4) 
plt.show()
#We got an auc value of 0.721

submission['Loan_Status']=pred_test 
submission['Loan_ID']=test_original['Loan_ID']

#Remember we need predictions in Y and N. So let’s convert 1 and 0 to Y and N.

submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)
#Lets convert the submission to .csv format and make submission to check the accuracy on the leaderboard.

pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('Logistickfold.csv',index=False)
submission.head
train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome'] 
test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']
#Let’s check the distribution of Total Income.

sns.distplot(train['Total_Income']);

train['Total_Income_log'] = np.log(train['Total_Income']) 
sns.distplot(train['Total_Income_log']); 
test['Total_Income_log'] = np.log(test['Total_Income'])

train['EMI']=train['LoanAmount']/train['Loan_Amount_Term'] 
test['EMI']=test['LoanAmount']/test['Loan_Amount_Term']

#Let’s check the distribution of EMI variable.

sns.distplot(train['EMI']);
# us create Balance Income feature now and check its distribution.

train['Balance Income']=train['Total_Income']-(train['EMI']*1000) # Multiply with 1000 to make the units equal test['Balance Income']=test['Total_Income']-(test['EMI']*1000)
sns.distplot(train['Balance Income']);

train_final=train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1) 
test_final=test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)
