




# 
import pandas as pd 
import numpy as np  
import os 

# 데이터 불러오기 
ibmhr = pd.read_csv('/kaggle/input/ibm-hr-analytics-attrition-dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv')
ibmhr.head()
ibmhr.info()
ibmhr.describe()
ibmhr['Attrition'].dtype.name
if ibmhr['Attrition'].dtype.name == 'object':
    values = ibmhr['Attrition'].unique()
    print(values)
for col in ibmhr.columns:
    if ibmhr[col].dtype.name == 'object':
        values = ibmhr[col].unique()
        print(col,end=': ')
        print(values,end='\n\n')
ibmhr['MonthlyIncomeKRW'] = ibmhr['MonthlyIncome']*1200
ibmhr.info()
df = ibmhr.\
groupby(['Attrition','Department']).\
agg({'MonthlyIncomeKRW':np.mean}).\
pivot_table(index=['Department'],columns=['Attrition'])

df.style.format('{:,.1f}')
idx 
col 

df = ibmhr.\
groupby(['Attrition','Department','Gender']).\
agg({'MonthlyIncomeKRW':[np.mean,np.sum]}).\
pivot_table(index=['Gender','Department'],columns=['Attrition'])

df.style.format('{:,.1f}')

df.loc[('Female','Sales'),('MonthlyIncomeKRW','mean','Yes')]





df = ibmhr.\
groupby(['Attrition','Department','YearsAtCompany']).\
agg({'MonthlyIncomeKRW':np.mean}).\
pivot_table(index=['YearsAtCompany','Department'],columns=['Attrition'])

df.style.format('{:,.1f}')







# 숫자로 된 데이터를 보기 쉽게 영문 설명으로 대체해보기
# 'https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset' 에 있는 설명을 기준으로 
ibmhr['Education'] = ibmhr['Education'].replace({1:'Below College',
                                                 2:'College',
                                                 3:'Bachelor',
                                                 4:'Master',
                                                 5:'Doctor'}).\
                                        astype('category')
ibmhr['EnvironmentSatisfaction'] = ibmhr['EnvironmentSatisfaction'].replace({1:'Low',
                                         2:'Medium',
                                         3:'High',
                                         4:'Very High'}).astype('category')

ibmhr['JobInvolvement'] = ibmhr['JobInvolvement'].replace({
    1:'Low',2:'Medium',3:'High',4:'Vecy High'
}).astype('category')

ibmhr['JobSatisfaction'] = ibmhr['JobSatisfaction'].replace({
    1:'Low',2:'Medium',3:'High',4:'Vecy High'
}).astype('category')

ibmhr['RelationshipSatisfaction'] = ibmhr['RelationshipSatisfaction'].replace({
    1:'Low',2:'Medium',3:'High',4:'Vecy High'
}).astype('category')

ibmhr['PerformanceRating'] = ibmhr['PerformanceRating'].replace({
    1:'Low',2:'Good',3:'Excellent',4:'Outstanding'
}).astype('category')

ibmhr['WorkLifeBalance'] = ibmhr['WorkLifeBalance'].replace({
    1:'Bad',2:'Good',3:'Better',4:'Best'
}).astype('category')
ibmhr.info()
# 카테고리 타입 또는 오브젝트 타입의 변수라면, unique한 값을 출력하기 
for col in ibmhr.columns: 
    if ibmhr[col].dtype.name in ['object','category']:
        print(col + ':' + str(ibmhr[col].unique()))
        print('===============')
import seaborn as sns 
# sns.set_theme(style="whitegrid", palette="muted")
ax = sns.swarmplot(data=ibmhr, x="MonthlyIncome", y="WorkLifeBalance", hue="Department")
# ax.set(ylabel="")