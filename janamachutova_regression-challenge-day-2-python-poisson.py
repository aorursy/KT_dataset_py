import pandas as pd

import chardet

import numpy as np



from io import StringIO





#kaggle = pd.read_csv('multipleChoiceResponses.csv')#("../input/kaggle-survey-2017/multipleChoiceResponses.csv")

with open ("../input/kaggle-survey-2017/multipleChoiceResponses.csv", 'rb') as rawdata:

    result = chardet.detect(rawdata.read(52156))

    

# check what the character encoding might be

print(result)



with open ("../input/kaggle-survey-2017/multipleChoiceResponses.csv", 'rb') as rawdata:

    kaggleString = rawdata.read().decode(result['encoding'],'replace')



kaggle=pd.read_csv(StringIO(kaggleString),low_memory=False)



stackOverflow = pd.read_csv('../input/so-survey-2017/survey_results_public.csv')
has_compensation = kaggle[['CompensationAmount','Age']].copy()

has_compensation['CleanedCompensationAmount'] = has_compensation.CompensationAmount.map(lambda x : str(x).replace(',',''))

has_compensation.CleanedCompensationAmount = has_compensation.CleanedCompensationAmount.map(lambda x : x.replace('-','0'))

has_compensation.CleanedCompensationAmount = has_compensation.CleanedCompensationAmount.astype(float)
has_compensation = has_compensation[np.isnan(has_compensation.CleanedCompensationAmount)==False]

has_compensation.CleanedCompensationAmount = has_compensation.CleanedCompensationAmount.map(lambda x : int(x))
has_compensation = has_compensation.dropna(subset=['Age'])
from sklearn.linear_model import PoissonRegressor

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
model = PoissonRegressor()

X = has_compensation[['Age']]

y = has_compensation.CleanedCompensationAmount

model.fit(X, y)

has_compensation['Pred'] = model.predict(X)
sns.regplot(x='Age', y='CleanedCompensationAmount', data=has_compensation).set_title('Regression Plot')
sns.residplot(has_compensation.Age, has_compensation.CleanedCompensationAmount).set_title('Residual Plot')
ax1 = sns.distplot(has_compensation.CleanedCompensationAmount, hist=False, color="r", label="Actual Value")

sns.distplot(has_compensation.Pred, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Distribution Plot')
has_compensation = has_compensation[has_compensation.CleanedCompensationAmount < 150000].copy()

X = has_compensation[['Age']]

y = has_compensation.CleanedCompensationAmount

model.fit(X, y)
has_compensation.Pred = model.predict(X)
sns.regplot(x='Age', y='CleanedCompensationAmount', data=has_compensation).set_title('Regression Plot')
sns.residplot(has_compensation.Age, has_compensation.CleanedCompensationAmount).set_title('Residual Plot')
ax1 = sns.distplot(has_compensation.CleanedCompensationAmount, hist=False, color="r", label="Actual Value")

sns.distplot(has_compensation.Pred, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Distribution Plot')
stackOverflow.StackOverflowMakeMoney.unique()
stackOverflow.Salary.dtype
so = stackOverflow[np.isnan(stackOverflow.Salary)==False][['Salary','StackOverflowMakeMoney']]

so.info()
opinion_dict = {'Strongly disagree':1, 'Disagree':2, 'Somewhat agree':4, 'Agree':5,

       'Strongly agree':6}
so['OpinionNum'] = so.StackOverflowMakeMoney.map(lambda x : opinion_dict[x] if x in opinion_dict else 3)

so.OpinionNum.value_counts()
so.Salary = so.Salary.map(lambda x : x if x > 10 else 10)

sns.distplot(so.Salary,hist=False)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

so['ScaledSalary'] = scaler.fit_transform(so[['Salary']])

sns.distplot(so.ScaledSalary,hist=False)
X = so[['ScaledSalary']]

y = so.OpinionNum

np.seterr(divide='ignore', invalid='ignore')

model = PoissonRegressor()

model.fit(X,y)
so['Pred'] = model.predict(X)

so.head()

print('min {}, max {}'.format(so.Pred.min(), so.Pred.max()))
sns.regplot(x='Salary', y='OpinionNum', data=so).set_title('Regression Plot')
sns.residplot(so.Salary, so.OpinionNum).set_title('Residual Plot')
ax1 = sns.distplot(so.OpinionNum, hist=False, color="r", label="Actual Value")

sns.distplot(so.Pred, hist=False, color="b", label="Fitted Values" , ax=ax1)

plt.title('Distribution Plot')
so[['Salary','OpinionNum']].corr()