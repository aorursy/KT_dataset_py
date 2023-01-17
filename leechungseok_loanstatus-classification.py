import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
dataframe = pd.read_csv('../input/my-dataset/credit_train.csv')
dataframe
for i in range(100000 ,100514):  #하단에 붙어있는 결측값 제거
  dataframe = dataframe.drop([i])

dataframe
np.around(dataframe.describe(),1)
#결측값치 확인
dataframe[list(dataframe.columns)].isnull().sum()
#Tax_lien
print(dataframe['Tax Liens'].value_counts())
print(dataframe['Tax Liens'].unique())
print(dataframe['Tax Liens'].isnull().sum())
#Tax_Lien에서 결측치는 최빈값인 0으로 가정
dataframe['Tax Liens'] = dataframe['Tax Liens'].fillna(0)
#Bankruptcies
print(dataframe['Bankruptcies'].value_counts())
print(dataframe['Bankruptcies'].unique())
print(dataframe['Bankruptcies'].isnull().sum())
#Bankruptcies에서 결측치는 최빈값인 0으로 가정
dataframe['Bankruptcies'] = dataframe['Bankruptcies'].fillna(0)
#Case No.1: 'Years in current job'이 NaN인 사람 중에서 'Annual Income' 역시 NaN인 경우
dataframe[dataframe['Years in current job'].isnull()]['Annual Income'].isnull().sum()
#Case No.1인 경우 810 -> 이 경우에는 'Years in current job'을 'None'으로 채운다.
dataframe.loc[dataframe['Annual Income'].isnull(), 'Years in current job'] = 'None'

#Case No.2: 'Years in current job'이 NaN이지만 "Annual Income"은 있는 경우
dataframe.loc[dataframe['Years in current job'].isnull(),'Years in current job'] = 'casual job'
#Months since last delinquent | NaN이 가장 높은 수치고, 그다음 숫자가 높을 수록 좋은 것
#신용관련 정보가 20년까지 저장된다고 가정 -> 모든 값에 -12*20을 해서 NaN인 사람을 0으로 수렴
dataframe.loc[dataframe['Months since last delinquent'].isnull(),'Months since last delinquent'] = 240
dataframe['Months since last delinquent'] = dataframe['Months since last delinquent'] - 240
#숫자가 똑같은 Annual Income과 Credit Score 결측치에 대한 검증
NaNcount_AnnualIncome = []
NaNcount_CreditScore = []
NaNcount = []
for i in range(len(dataframe)):
    if np.isnan(dataframe['Annual Income'][i]) == True:
        NaNcount_AnnualIncome += [i]

for i in range(len(dataframe)):
    if np.isnan(dataframe['Credit Score'][i]) == True:
        NaNcount_CreditScore += [i]
        
    
for i in range(len(NaNcount_AnnualIncome)):
    if NaNcount_AnnualIncome[i] == NaNcount_CreditScore[i]:
        NaNcount += [i]

        
print('NaNcount_AnnualIncome의 위치정보리스트 길이:',len(NaNcount_AnnualIncome))
print('NaNcount_CreditScore의 위치정보리스트 길이:',len(NaNcount_CreditScore))
print('NaNcount의 위치정보리스트 길이:',len(NaNcount))
dataframe.loc[dataframe['Credit Score'].isnull(),'Credit Score'] = 0
dataframe.loc[dataframe['Annual Income'].isnull(),'Annual Income'] = 0
#수정된 데이터의 결측치 확인
dataframe[list(dataframe.columns)].isnull().sum()
np.around(dataframe.describe(),1)
dataframe['Maximum Open Credit'].describe().astype('int')
dataframe = dataframe.dropna()
print(dataframe.isnull().sum())
print(len(dataframe))
#Credit Score 원래 범위는 300~ 850 그중에서 일반적으로 700점이 넘으면 좋다고 판단
#이상치 확인을 위한 시각화 (최대치인 850이 넘은 레코드 대상)
sns.distplot(dataframe.loc[dataframe['Credit Score']>850, ['Credit Score']])
#주요 점수대인 700~750 사이가 가장많이 분포하는 것으로보아 '0'이 하나 더 붙어있음을 유추
dataframe.loc[dataframe['Credit Score']>850, ['Credit Score']] = (dataframe.loc[dataframe['Credit Score']>850, ['Credit Score']])/10
#Current Loan amount 데이터 이상치 확인을 위한 시각화
sns.distplot(np.around(dataframe['Current Loan Amount'], 1))
# Current Loan Amount의 Max 값인 99999999가 이상치임을 확인
temp = dataframe.loc[dataframe['Current Loan Amount']==99999999, ['Current Loan Amount', 'Term', 'Purpose', 'Loan Status']]
print(temp['Purpose'].value_counts())
print('-'*20)
print(temp['Loan Status'].value_counts())
print('-'*20)
print(temp['Term'].value_counts())
dataframe = dataframe.loc[dataframe['Current Loan Amount']!=99999999]
dataframe
Continuous = ['Loan Status', 'Current Loan Amount', 'Credit Score', 'Annual Income', 'Monthly Debt', 'Years of Credit History', 'Months since last delinquent', 'Number of Open Accounts', 'Number of Credit Problems', 'Current Credit Balance', 'Maximum Open Credit', 'Bankruptcies', 'Tax Liens']
Category = ['Loan Status', 'Term', 'Years in current job', 'Home Ownership', 'Purpose']

sns.pairplot(dataframe[Continuous], hue='Loan Status')
plt.figure(figsize=(13,13))
for i in range(1, 11):
  plt.subplot(5,2,i)
  sns.distplot(dataframe.loc[dataframe['Loan Status']=='Fully Paid', Continuous[i]], label='Fully Paid')
  sns.distplot(dataframe.loc[dataframe['Loan Status']=='Charged Off', Continuous[i]], label= 'Charged Off')
  plt.legend(loc = 'best')
plt.show()
plt.figure(figsize=(15 ,15))
for i in range(1,7):
  plt.subplot(2,3,i)
  sns.countplot(x=Category[i], data =dataframe, hue='Loan Status')
plt.show()
Continuous.remove('Loan Status')
Category.remove('Loan Status')
#연속형 데이터들 MinMax 방식으로 정규화 진행
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler().fit(dataframe[Continuous])
scaled_dataframe = pd.DataFrame(scale.transform(dataframe[Continuous]), columns= Continuous, index= dataframe.index)
#범주형 데이터들 더미화 진행
df_dummies = pd.get_dummies(dataframe[Category], drop_first= True)
df_dummies
findataframe = pd.concat([scaled_dataframe, df_dummies], axis= 1)
findataframe
temp = list(findataframe.columns)
temp.sort()
temp
loanlist = ['Term_Short Term',
            'Current Loan Amount',
            'Maximum Open Credit',
            'Number of Open Accounts',
            'Purpose_Buy House',
            'Purpose_Buy a Car',
            'Purpose_Debt Consolidation',
            'Purpose_Educational Expenses',
            'Purpose_Home Improvements',
            'Purpose_Medical Bills',
            'Purpose_Other',
            'Purpose_Take a Trip',
            'Purpose_major_purchase',
            'Purpose_moving',
            'Purpose_other',
            'Purpose_renewable_energy',
            'Purpose_small_business',
            'Purpose_vacation',
            'Purpose_wedding',]

creditlist = ['Credit Score',
              'Years of Credit History',
              'Months since last delinquent',
              'Number of Credit Problems',
              'Bankruptcies',
              'Tax Liens']

incomelist = ['Annual Income',
              'Current Credit Balance',
              'Years in current job_10+ years',
              'Years in current job_2 years',
              'Years in current job_3 years',
              'Years in current job_4 years',
              'Years in current job_5 years',
              'Years in current job_6 years',
              'Years in current job_7 years',
              'Years in current job_8 years',
              'Years in current job_9 years',
              'Years in current job_< 1 year',
              'Years in current job_None',
              'Years in current job_casual job',
              'Home Ownership_Home Mortgage',
              'Home Ownership_Own Home',
              'Home Ownership_Rent']
#dataset 분리
dataset_loan = findataframe[loanlist]
dataset_credit = findataframe[creditlist]
dataset_income = findataframe[incomelist]
#Target 설정
target = dataframe['Loan Status']
target = target.replace('Fully Paid', 1)
target = target.replace('Charged Off', 0)

#train, test set 분리
from sklearn.model_selection import train_test_split
xtrain_loan, xtest_loan, ytrain, ytest = train_test_split(dataset_loan, target, test_size=0.2, stratify = target)

#같은 레코드를 분리
train_indexlist = list(xtrain_loan.index)
test_indexlist = list(xtest_loan.index)
xtrain_credit = dataset_credit.loc[train_indexlist]
xtest_credit = dataset_credit.loc[test_indexlist]
xtrain_income = dataset_income.loc[train_indexlist]
xtest_income = dataset_income.loc[test_indexlist]

from sklearn.linear_model import LogisticRegression
log_loan = LogisticRegression().fit(xtrain_loan, ytrain)
log_credit = LogisticRegression().fit(xtrain_credit, ytrain)
log_income = LogisticRegression().fit(xtrain_income, ytrain)
print(log_loan.score(xtrain_loan, ytrain))
print(log_loan.score(xtest_loan, ytest))
print('-'*20)
print(log_credit.score(xtrain_credit, ytrain))
print(log_credit.score(xtest_credit, ytest))
print('-'*20)
print(log_income.score(xtrain_income, ytrain))
print(log_income.score(xtest_income, ytest))
from sklearn.ensemble import RandomForestClassifier
rdf_loan = RandomForestClassifier(max_depth=10).fit(xtrain_loan, ytrain)
rdf_credit = RandomForestClassifier(max_depth=10).fit(xtrain_credit, ytrain)
rdf_income = RandomForestClassifier(max_depth=10).fit(xtrain_income, ytrain)
print(rdf_loan.score(xtrain_loan, ytrain))
print(rdf_loan.score(xtest_loan, ytest))
print('-'*20)
print(rdf_credit.score(xtrain_credit, ytrain))
print(rdf_credit.score(xtest_credit, ytest))
print('-'*20)
print(rdf_income.score(xtrain_income, ytrain))
print(rdf_income.score(xtest_income, ytest))
import tensorflow as tf
from tensorflow.keras import layers
model_loan = tf.keras.Sequential()
model_loan.add(layers.Input(shape=xtrain_loan.shape[1]))
model_loan.add(layers.Dense(190, activation='relu'))
model_loan.add(layers.Dense(19, activation='relu'))
model_loan.add(layers.Dense(1, activation='sigmoid'))

model_loan.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
history_loan = model_loan.fit(xtrain_loan, ytrain, epochs=100)
model_credit = tf.keras.Sequential()
model_credit.add(layers.Input(shape=xtrain_credit.shape[1]))
model_credit.add(layers.Dense(60, activation='relu'))
model_credit.add(layers.Dense(6, activation='relu'))
model_credit.add(layers.Dense(1, activation='sigmoid'))

model_credit.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
history_credit = model_credit.fit(xtrain_credit, ytrain, epochs=100)
model_income = tf.keras.Sequential()
model_income.add(layers.Input(shape=xtrain_income.shape[1]))
model_income.add(layers.Dense(170, activation='relu'))
model_income.add(layers.Dense(17, activation='relu'))
model_income.add(layers.Dense(1, activation='sigmoid'))

model_income.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
history_income = model_income.fit(xtrain_income, ytrain, epochs=100)

prediction_loan = model_loan.predict(xtest_loan)
prediction_credit = model_credit.predict(xtest_credit)
prediction_income = model_income.predict(xtest_income)
pred_loan = pd.DataFrame(prediction_loan, index= ytest.index)
pred_credit =pd.DataFrame(prediction_credit, index= ytest.index)
pred_income =pd.DataFrame(prediction_income, index= ytest.index)
predsum = pd.concat([pred_loan, pred_credit, pred_income], axis =1)
predsum.columns = ['loan', 'credit', 'income']
predsum
finx_train, finx_test, finy_train, finy_test = train_test_split(predsum, ytest, test_size=0.2, stratify = ytest)
model_total = tf.keras.Sequential()
model_total.add(layers.Input(shape=finx_train.shape[1]))
model_total.add(layers.Dense(30, activation='relu'))
model_total.add(layers.Dense(3, activation='relu'))
model_total.add(layers.Dense(1, activation='sigmoid'))

model_total.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
history_total = model_total.fit(finx_train, finy_train, epochs=100)
fin_pred = model_total.predict(finx_test)
fin_pred = fin_pred.reshape(3541,)
fin_pred
result=[]
for i in range(len(fin_pred)):
  if fin_pred[i]>=0.5:
    result += [1]
  elif fin_pred[i]<0.5:
    result += [0]

from sklearn.metrics import confusion_matrix
pd.DataFrame(confusion_matrix(finy_test, result), index=['Real_0', 'Real_1'], columns=['Pred_0', 'Pred_1'])