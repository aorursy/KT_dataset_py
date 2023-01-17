import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
#하계 올림픽 파일 읽기
olympic = pd.read_csv('summer.csv', encoding='utf-8')
olympic.head()
#olympic data 정보 확인
olympic.info()
#olympic 칼럼 확인
olympic.columns
#olympic 칼럼 순서 변경
olympic = olympic[['Country', 'Year', 'Sport', 'Discipline', 'Athlete', 'City', 'Gender',
       'Event', 'Medal']]
olympic.head()
#Gedner로 그룹화하여 count
gender_count = olympic.groupby('Gender').count()
gender_count.head()
#Country로 그룹화한 후 메달 개수 순서로 정렬
country = olympic.groupby('Country').count()
country = country.sort_values(by='Medal',ascending=False)
country.head()
#상위 30개 국가만 good_country로 생성
good_country = country.head(30)
good_country['Medal'].sort_values().plot(kind='barh', grid=True, figsize=(10,10))
plt.xlabel('Medal')
plt.show()
#선수들의 메달 갯수 확인해보기
olympic_athlete = olympic.groupby('Athlete').count()
olympic_athlete.sort_values(by='Medal', ascending = False).head()
#마이클펠프스의 기록 확인
olympic.loc[olympic['Athlete'] == 'PHELPS, Michael']
#한국 올림픽 메달 기록 확인
korea = olympic.loc[olympic['Country'] == 'KOR']
korea.head()
korea.info()
#한국 메달 가장 많이 딴 선수 확인해보기
korea_medal = korea.groupby('Athlete').count()
korea_medal.sort_values(by='Medal', ascending=False).head()
#한국에서 메달을 가장많이 확보한 종목
korea['Discipline'].value_counts().plot(kind='barh', grid=True, figsize=(10,10), )
plt.xlabel('Medal')
plt.show()
#단체종목일 경우 가장 첫번째 레코드만 남겨두고 삭제 by drop_duplicates()
korea_discipline = korea.drop_duplicates(subset=['Year', 'Event', 'Medal'])
korea_discipline.info()
korea_discipline['Discipline'].value_counts().plot(kind='barh', grid=True, figsize=(10,10))
plt.xlabel('Medal')
plt.title('KOREA OLYMPIC MEDAL')
plt.show()
#다시 실습내용 진행하기 위해 
olympic.head()
#마찬가지로 단체 종목을 메달 중복 제거한 olympic_count DF 생성
olympic_count = olympic.drop_duplicates(subset=['Year', 'Country', 'Medal', 'Event'])
olympic_count.info()
#index reset
olympic_count = olympic_count.reset_index(drop=True)
olympic_count.tail()
#size를 통해 country로 그룹한후 Year로 그룹핑
country_medal_count = olympic_count.groupby(['Country','Year']).size()
#size의 결과는 series형태이므로 to_frame()을 통해 Data Frame형태로 바꿔줌.
country_medal_count = country_medal_count.to_frame().reset_index()
country_medal_count.head()
#column rename하기
country_medal_count = country_medal_count.rename(columns = {0 : 'Medal'})
country_medal_count.head()
#pivot_table(index, column, value, aggfunc, margins)
country_medal_year = country_medal_count.pivot_table(index = "Country", columns = "Year", values = "Medal",aggfunc = 'sum', margins = True)
country_medal_year.head()
country_medal_year.columns[ : -4]
#2000년 이전 기록 다 삭제
a = country_medal_year.columns[:-4]
for i in a :
    del country_medal_year[i]
country_medal_year.head() 
#상위 30개 국가만 진행
top_country = country_medal_year.sort_values(by='All', ascending = False).head(40)
#2000년 이전 메달 개수 다 합침
top_country = top_country.fillna(0)
#NaN수를 모두 0으로 대입
top_country['before_2000'] = top_country['All'] - top_country[2004] - top_country[2008] - top_country[2012]
top_country
#GDP 파일 열기
gdp = pd.read_csv('dictionary.csv')
gdp.head()
gdp.rename(columns = {'Country' : 'Fullname', 'Code' : 'Country'}, inplace = True)
gdp.head()
#데이터프레임 합치기 
result = pd.merge(top_country, gdp, on='Country')
result
result = result[['Country','Fullname', 'before_2000', 2004, 2008, 2012, 'All', 'Population',
       'GDP per Capita']]
result.head()
#최근 메달 증가율
result['Growth'] = (result[2004] + result[2008]+ result[2012]) / result['before_2000'] * 100

result.sort_values(by='Growth', ascending=False)
result = result.fillna(7815)
result.set_index('Country', inplace=True)
result.head()
#총 메달 개수와 인구와의 관계
np.corrcoef(result['All'], result['Population'])
#메달 증가율과 인구와의 관계
np.corrcoef(result['Growth'], result['Population'])
#총 메달 개수와 GDP의 관계
np.corrcoef(result['All'], result['GDP per Capita'])
#메달 증가율과 GDP의 관계
np.corrcoef(result['Growth'], result['GDP per Capita'])
#platform -> font한글 폰트하기 위해서 

import platform

from matplotlib import font_manager, rc
plt.rcParams['axes.unicode_minus'] = False

if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)

else :
    print('Unknown system.. sorry~~')
result['All'].plot(kind='barh', grid=True, figsize=(10,10))
plt.xlabel('Medal')
plt.title('Country Medal')
plt.show()
plt.figure(figsize=(6,6))
plt.scatter(result['All'], result['Population'], s= 50)
plt.xlabel('메달 수')
plt.ylabel('인구 수')
plt.title('메달수와 인구수와의 관계')
plt.grid()
plt.show()

result['rate_pop'] = result['Population'] / result['Population'].sum()
result
plt.figure(figsize=(6,6))
plt.scatter(result['Growth'], result['Population'], s= 50)
plt.xlabel('메달 증가율')
plt.ylabel('인구 수')
plt.title('메달증가율과 인구 수와의 관계')
plt.grid()
plt.show()
#너무 큰 차이나와 러시아 빼보자
result_except_big = result.drop(['CHN','RUS'],0)
result_except_big
plt.figure(figsize=(6,6))
plt.scatter(result_except_big['Growth'], result_except_big['Population'], s= 50)
plt.xlabel('메달 증가율')
plt.ylabel('인구 수')
plt.title('메달 증가율과 인구 수와의 관계')
plt.grid()
plt.show()
np.corrcoef(result_except_big['Growth'], result_except_big['Population'])
plt.figure(figsize=(6,6))
plt.scatter(result['Growth'], result['GDP per Capita'], s= 50)
plt.xlabel('메달 증가율')
plt.ylabel('GDP')
plt.title('메달 증가율과 GDP와의 관계')
plt.grid()
plt.show()
plt.figure(figsize=(6,6))
plt.scatter(result['All'], result['GDP per Capita'], s= 50)
plt.xlabel('메달 수')
plt.ylabel('GDP')
plt.title('메달수와 GDP와의 관계')
plt.grid()
plt.show()
fp1 = np.polyfit(result['All'], result['GDP per Capita'], 1)
f1 = np.poly1d(fp1)
fx = np.linspace(100, 3000, 100)
plt.figure(figsize=(10,10))
plt.scatter(result['All'], result['GDP per Capita'], s= 50)
plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
plt.xlabel('메달 수')
plt.ylabel('GDP per Capita')
plt.title('메달 수와 GDP의 관계')
plt.grid()
plt.show()
fp1 = np.polyfit(result['Growth'], result['GDP per Capita'], 1)
f1 = np.poly1d(fp1)
fx = np.linspace(0, 200, 100)
plt.figure(figsize=(10,10))
plt.scatter(result['Growth'], result['GDP per Capita'], s= 50)
plt.plot(fx, f1(fx), ls='dashed', lw=3, color='g')
plt.xlabel('메달 증가율')
plt.ylabel('GDP per Capita')
plt.title('메달증가율과 GDP의 관계')
plt.grid()
plt.show()
fp1 = np.polyfit(result['Growth'], result['GDP per Capita'], 1)

f1 = np.poly1d(fp1)
fx = np.linspace(0,200,100)

result['오차'] = np.abs(result['GDP per Capita'] - f1(result['Growth']))

result_sort = result.sort_values(by='오차', ascending=False)
result_sort.head()
plt.figure(figsize=(14,10))
plt.scatter(result['Growth'], result['GDP per Capita'],
           c=result['오차'], s = 50)
plt.plot(fx, f1(fx), ls='dashed', lw =3 , color='g')

for n in range(10) :
    plt.text(result_sort['Growth'][n]*1.02, result_sort['GDP per Capita'][n]*0.98,
            result_sort.index[n], fontsize=15)

plt.xlabel('메달 증가율')
plt.ylabel('GDP per Capita')

plt.colorbar()
plt.grid()
plt.show()


