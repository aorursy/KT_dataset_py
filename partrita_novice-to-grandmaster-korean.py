#  필요한 라이브러리를 불러옵니다.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings # Suppress Deprecation and Incorrect Usage Warnings
warnings.filterwarnings('ignore')
question = pd.read_csv('../input/schema.csv')
response=pd.read_csv('../input/multipleChoiceResponses.csv',encoding='ISO-8859-1')
# question.shape
question.tail()
response.tail()
print('전체 응답자의 수: ',response.shape[0])
print('전체 국가의 수: ',response['Country'].nunique())
print('가장 많이 응답한 국가: ',response['Country'].value_counts().index[0],'with',response['Country'].value_counts().values[0],'respondents')
print('가장 어린 응답자의 나이: ',response['Age'].min(),' 가장 나이 많은 응답자의 나이 : ',response['Age'].max())
import missingno as msno # https://github.com/ResidentMario/missingno
msno.matrix(response)
plt.subplots()
sns.countplot(y=response['GenderSelect'],order=response['GenderSelect'].value_counts().index)
plt.show()
resp_coun=response['Country'].value_counts()[:15].to_frame() # 응답률로 정렬하고 상위 15개만 분리합니다.
sns.barplot(resp_coun['Country'],resp_coun.index)
plt.title('Top 15 Countries by number of respondents')
plt.xlabel('') # x레이블을 지웁니다.
korean = response[response['Country']=='South Korea'] # 국적이 한국인 데이터만
print('한국인 응답자 수:  ' + str(korean.shape[0]))

sns.distplot(response['Age'].dropna()) # 무응답 데이터 제거
print(response[response['Age'] > 0]['Age'].mean()) # 평균 나이
sns.distplot(korean['Age'].dropna())
print(korean[korean['Age'] > 0]['Age'].mean()) # 평균 나이
major_df = pd.DataFrame(response['MajorSelect'].value_counts())# value_counts 를 사용하면 그룹화된 데이터의 카운트 값을 보여준다.
major_df['ratio'] = pd.DataFrame(response['MajorSelect'].value_counts(normalize=True)) # 해당 데이터가 전체 데이터에서 어느 정도의 비율을 차지하는지 알 수 있다.
major_df.head(10) # 상위 10개의 전공
major_df['ratio'].plot(kind='barh') #pandas를 이용한 간단한 시각화
sns.countplot(y='Tenure', data=response, order=response['Tenure'].value_counts().index)
response['CompensationAmount']=response['CompensationAmount'].str.replace(',','')
response['CompensationAmount']=response['CompensationAmount'].str.replace('-','')
rates=pd.read_csv('../input/conversionRates.csv')
rates.drop('Unnamed: 0',axis=1,inplace=True)
salary=response[['CompensationAmount','CompensationCurrency','GenderSelect','Country','CurrentJobTitleSelect']].dropna()
salary=salary.merge(rates,left_on='CompensationCurrency',right_on='originCountry',how='left')
salary['Salary']=pd.to_numeric(salary['CompensationAmount'])*salary['exchangeRate']
print('최고 연봉($)',salary['Salary'].dropna().astype(int).max())
print('최적 연봉($)',salary['Salary'].dropna().astype(int).min())
print('중위 연봉($)',salary['Salary'].dropna().astype(int).median())
plt.subplots()
salary=salary[(salary['Salary']<300000) & (salary['Salary']>1000) ] # 현실적인 연봉 값만 선택
sns.distplot(salary['Salary']).set(xlim=(0, None))
plt.axvline(salary['Salary'].median(), linestyle='dashed') # 중위값 
plt.title('Salary Distribution')
plt.subplots(figsize=(8,12))
sal_coun = salary.groupby('Country')['Salary'].median().sort_values(ascending=False)[:30].to_frame()
sns.barplot('Salary', sal_coun.index, data = sal_coun, palette='Spectral')
plt.axvline(salary['Salary'].median(), linestyle='dashed')
plt.title('Highest Salary Paying Countries')
f,ax=plt.subplots(figsize=(8,12))
tool=response['MLToolNextYearSelect'].str.split(',')
tool_nxt=[]
for i in tool.dropna():
    tool_nxt.extend(i)
pd.Series(tool_nxt).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9)
ax.set_title('ML Tool Next Year')
plt.subplots(figsize=(6,8))
learn=response['LearningPlatformSelect'].str.split(',')
platform=[]
for i in learn.dropna():
    platform.extend(i)
pd.Series(platform).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9)
plt.title('Best Platforms to Learn',size=15)
plt.subplots()
challenge=response['WorkChallengesSelect'].str.split(',')
challenges=[]
for i in challenge.dropna():
    challenges.extend(i)
plt1=pd.Series(challenges).value_counts().sort_values(ascending=False).to_frame()[:5] # 상위 5개만
sns.barplot(plt1[0],plt1.index)
plt.title('Challenges in Data Science')
plt.xlabel('') # x레이블을 지웁니다.
qc = question.loc[question[ 'Column'].str.contains('JobFactor')]
job_factors = [ x for x in response.columns if x.find('JobFactor') != -1]
jfdf = {}
for feature in job_factors:
    a = response[feature].value_counts()
    a = a/a.sum()
    jfdf[feature[len('JobFactor'):]] = a
jfdf = pd.DataFrame(jfdf).transpose()
plt.figure(figsize=(8,12))
sns.heatmap(jfdf.sort_values('Very Important', ascending=False), annot=True, cmap="YlGnBu")