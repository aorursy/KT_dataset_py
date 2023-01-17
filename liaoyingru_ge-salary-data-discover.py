# 导入相关的包
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from datetime import datetime as dt
# 读取数据
df = pd.read_csv('../input/S1_GE_Salaries_2016.csv')
df.head()
# 字段总览
df.info()
# 重复行检查
df.duplicated().sum()
df = df[['Name', 'JobTitle', 'AgencyID', 'Agency', 'HireDate', 'AnnualSalary']]
df.info()
df.HireDate = pd.to_datetime(df.HireDate)
df.info()
df = df.query('AnnualSalary != 0')
df.AnnualSalary.describe()
df['AgencyName'] = df['Agency'].map(lambda x: x.split('(')[0].strip())
df['AgencyName'] = df['AgencyName'].replace(['Mayors Office','HLTH-Health Department'],["Mayor's Office",'HLTH-Heatlh Dept.'])
groupbyAgency = df.groupby('AgencyName').describe()['AnnualSalary'][['count','mean']]
groupbyAgency.columns = ['AnnualSalary_count','AnnualSalary_mean']
groupbyAgency = groupbyAgency.query('AnnualSalary_count >= 10')
groupbyAgency.sort_values('AnnualSalary_mean',ascending=False).head()
df['HireTime'] = df['HireDate'].map(lambda x: (dt.strptime('2016-07-01','%Y-%m-%d')-x).days)
plt.figure(figsize=(8,4))
plt.scatter(df['HireTime'],df['AnnualSalary'])
plt.xlabel('Hire Time')
plt.ylabel('Annual Salary')
plt.title('Hire Time & Annual Salary')
plt.legend();
corr_p = pd.DataFrame(columns=['AgencyName','sampleCnt','corrWithAnnualSalary'])
for i in set(df['AgencyName']):
    target = '"'+i+'"'
    sub_df = df.query('AgencyName == '+target)
    sampleCnt = len(sub_df)
    c = sub_df['HireTime'].corr(sub_df['AnnualSalary'])
    corr_p = corr_p.append({'AgencyName':target,'sampleCnt':sampleCnt,'corrWithAnnualSalary':c},ignore_index=True)
print('相关系数最接近1且样本数不少于10的前三个机构')
print(corr_p.sort_values('corrWithAnnualSalary',ascending=False).query('sampleCnt >= 10').head(3))
print()
print('相关系数最接近-1且样本数不少于10的前三个机构')
print(corr_p.sort_values('corrWithAnnualSalary',ascending=True).query('sampleCnt >= 10').head(3))
pos1 = df.query('AgencyName == "TRANS-Crossing Guards"')
pos2 = df.query('AgencyName == "COMP-Real Estate"')
neg1 = df.query('AgencyName == "HLTH-Health Dept. Location 199"')
neg2 = df.query('AgencyName == "Elections"')
plt.figure(figsize=(12,4))

plt.subplot(221)
plt.title('Hire Time & Annual Salary')
plt.scatter(pos1['HireTime'],pos1['AnnualSalary'],label='TRANS-Crossing Guards',c='teal')
plt.ylabel('Annual Salary')
plt.legend()
plt.subplot(222)
plt.title('Hire Time & Annual Salary')
plt.scatter(pos2['HireTime'],pos2['AnnualSalary'],label='COMP-Real Estate')
plt.legend()
plt.subplot(223)
plt.scatter(neg1['HireTime'],neg1['AnnualSalary'],label='HLTH-Health Dept. Location 199')
plt.xlabel('Hire Time')
plt.ylabel('Annual Salary')
plt.legend()
plt.subplot(224)
plt.scatter(neg2['HireTime'],neg2['AnnualSalary'],label='Elections')
plt.xlabel('Hire Time')
plt.legend();
