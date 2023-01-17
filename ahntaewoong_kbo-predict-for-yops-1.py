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
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import warnings
warnings.filterwarnings(action='ignore')

pd.set_option('display.max_column',100)
pd.set_option('display.max_row',500)
from pandas import DataFrame
from pandas import concat

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from operator import itemgetter

from matplotlib import font_manager, rc #그래프 폰트 깨질경우 방지 부분
#폰트 깨지는거 수정 부분
import os
print(os.listdir("../input"))

plt.rcParams['axes.unicode_minus']=False
fontpath= "../input/fontttf/NanumSquareB.ttf"
fp=font_manager.FontProperties(fname=fontpath)

import re
import requests
from bs4 import BeautifulSoup
import time

import math
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import MinMaxScaler


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from math import sqrt
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
#현재 데이터에서 발생하는 type error를 해결하기 위한 함수 생성해둠

def coerce_df_columns_to_numeric(df, column_list):
    df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')
regular_y=pd.read_csv('/kaggle/input/kbodata1/2019_kbo_for_kaggle_v2.csv')
regular_y.head()
Counter(regular_y['cp'])
# fkd1= list(Counter(regular_y['hand2']).keys())
# fkd2= list(Counter(regular_y['hand2']).values())

plt.figure(figsize=(20,9))

# 설치된 폰트 출력 가능
sns.boxplot(x='hand2',y="YOPS", data=regular_y,palette="Spectral")
plt.title('타석위치 별 YOPS Boxplot',fontproperties=fp,size=30)
plt.xticks(fontproperties=fp,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('hand',size=25)
plt.ylabel('다음 시즌 OPS',fontproperties=fp,size=25);
plt.figure(figsize=(20,9))

sns.boxplot(x='tp',y="YOPS", data=regular_y,palette="Spectral")
plt.title('포지션 별 YOPS Boxplot',fontproperties=fp,size=30)
plt.xticks(fontproperties=fp,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('position',size=25)
plt.ylabel('다음 시즌 OPS',fontproperties=fp,size=25)
#스탯티즈에서 가져온 타석 위치에 대한 통일 작업 실시 (좌타,우타,양타만 확인하고 바꾸는 방식으로 진행했다)
regular_y['hand2']=regular_y['hand2'].replace('우투우타','우타')
regular_y['hand2']=regular_y['hand2'].replace('우타우투','우타')
regular_y['hand2']=regular_y['hand2'].replace('우투양타','양타')
regular_y['hand2']=regular_y['hand2'].replace('우투좌타','좌타')

regular_y['hand2']=regular_y['hand2'].replace('좌투좌타','좌타')
regular_y['hand2']=regular_y['hand2'].replace('좌타좌투','좌타')
regular_y['hand2']=regular_y['hand2'].replace('좌타우투','좌타')

regular_y['hand2'].unique()
plt.figure(figsize=(20,9))

plt.subplot(121)
sns.boxplot(x='hand2',y="YOPS", data=regular_y,palette="Spectral")
plt.title('타석위치 별 YOPS Boxplot',fontproperties=fp,size=30)
plt.xticks(fontproperties=fp,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('hand',size=25)
plt.ylabel('다음 시즌 OPS',fontproperties=fp,size=25)

fkd1= list(Counter(regular_y['hand2']).keys())
fkd2= list(Counter(regular_y['hand2']).values())

plt.subplot(122)
sns.barplot(x=fkd1,y=fkd2)
plt.title('타석 위치 별 인원수(중복포함)',fontproperties=fp,size=30)
plt.xticks(fontproperties=fp,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('타석 위치',fontproperties=fp,size=25)
plt.ylabel('인원수',fontproperties=fp,size=25)
for x,y in zip(range(0,len(fkd1)),fkd2):
    if len(str(y))==2:
        plt.text(x-0.05,y,fkd2[x],size=25)
    else:
        plt.text(x-0.175,y,fkd2[x],size=25)
#스탯티즈를 통해서 대체된 포지션의 경우에 대한 포지션 통합 작업
#지명타자의 경우 외야수 출신인 이범호를 제외하고는 전부 지명타자 출신이어서 내야수로 할당하였다.

regular_y['tp']=regular_y['tp'].replace('2루수','내야수')
regular_y['tp']=regular_y['tp'].replace('지명타자','내야수')
regular_y['tp']=regular_y['tp'].replace('유격수','내야수')
regular_y['tp']=regular_y['tp'].replace('3루수','내야수')
regular_y['tp']=regular_y['tp'].replace('1루수','내야수')

regular_y['tp']=regular_y['tp'].replace('우익수','외야수')
regular_y['tp']=regular_y['tp'].replace('중견수','외야수')
regular_y['tp']=regular_y['tp'].replace('좌익수','외야수')

regular_y['tp'].unique()
plt.figure(figsize=(20,9))

plt.subplot(121)
sns.boxplot(x='tp',y="YOPS", data=regular_y,palette="Spectral")
plt.title('포지션 별 YOPS Boxplot',fontproperties=fp,size=30)
plt.xticks(fontproperties=fp,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('position',size=25)
plt.ylabel('다음 시즌 OPS',fontproperties=fp,size=25)

fkd1= list(Counter(regular_y['tp']).keys())
fkd2= list(Counter(regular_y['tp']).values())
plt.subplot(122)
sns.barplot(x=fkd1,y=fkd2)
plt.title('포지션 별 인원수(중복포함)',fontproperties=fp,size=30)
plt.xticks(fontproperties=fp,fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('position',size=25)
plt.ylabel('인원수',fontproperties=fp,size=25)
for x,y in zip(range(0,len(fkd1)),fkd2):
    if len(str(y))==2:
        plt.text(x-0.05,y,fkd2[x],size=25)
    else:
        plt.text(x-0.15,y,fkd2[x],size=25)
del regular_y['cp']
del regular_y['year_born']
del regular_y['p_year']
regular_y[regular_y.isnull()['YOPS']]
regular_y['YOPS']=regular_y['YOPS'].fillna(0.00000)
cor=regular_y.corr()

regular_y2=regular_y.copy()
del regular_y2['BB']
del regular_y2['HBP']
del regular_y2['year']
del regular_y2['1B']
del regular_y2['YAB']
del regular_y2['PA']
del regular_y2['batter_name']
del regular_y2['hand2']
del regular_y2['tp']
cor=regular_y2.corr()

f,ax=plt.subplots(figsize=(20,12))
ax=sns.barplot(x=cor['YOPS'].sort_values(ascending=False)[1:].keys(), y=cor['YOPS'].sort_values(ascending=False)[1:].values,
            palette=['lightblue' if _y >0.3  else 'lightgrey' if (_y > (-0.1) and ( _y <= 0.3)) else 'lightcoral' for _y in list(cor['YOPS'].sort_values(ascending=False)[1:].values) ])
plt.xticks(fontsize=25,rotation=45)
plt.yticks(fontsize=25)
plt.xlabel('variable name',size=30)
plt.ylabel('YOPS correlation',size=30)
plt.title('YOPS와 수치형 변수간의 상관관계',fontproperties=fp,size=40)

tk=[round(i,2) for i in cor['YOPS'].sort_values(ascending=False)[1:].values]
for x,y in zip(range(0,len(tk)),tk):
        plt.text(x-0.375,y+0.0025,tk[x],size=20);
fig = plt.figure(figsize=(20,15))

plt.subplot(2,2,1)
plt.hist(regular_y2['age'])
plt.title('나이 별 분포',fontproperties=fp,size=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('크기',fontproperties=fp,size=25)

plt.subplot(2,2,2)
plt.hist(regular_y2['AB'])
plt.title('타수 별 분포',fontproperties=fp,size=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('크기',fontproperties=fp,size=25)

plt.subplot(2,2,3)
plt.hist(regular_y2['HR'])
plt.title('홈런 수 별 분포',fontproperties=fp,size=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('크기',fontproperties=fp,size=25)

plt.subplot(2,2,4)
plt.hist(regular_y2['war'])
plt.title('war 별 분포',fontproperties=fp,size=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('크기',fontproperties=fp,size=25);
fig = plt.figure(figsize=(20,15))

plt.subplot(2,2,1)
plt.scatter(regular_y2['age'], regular_y2['YOPS'])
plt.title('나이 별 YOPS 분포',fontproperties=fp,size=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('YOPS',size=25)
plt.text(30,1.75,'correlation : 0.25',size=25)

plt.subplot(2,2,2)
plt.scatter(regular_y2['AB'], regular_y2['YOPS'])
plt.title('타수 별 YOPS 분포',fontproperties=fp,size=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('YOPS',size=25)
plt.text(350,1.75,'correlation : 0.49',size=25)

plt.subplot(2,2,3)
plt.scatter(regular_y2['HR'], regular_y2['YOPS'])
plt.title('홈런 수 별 YOPS 분포',fontproperties=fp,size=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('YOPS',size=25)
plt.text(30,1.75,'correlation : 0.47',size=25)

plt.subplot(2,2,4)
plt.scatter(regular_y2['war'], regular_y2['YOPS'])
plt.title('war 별 YOPS 분포',fontproperties=fp,size=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('YOPS',size=25)
plt.text(6,1.75,'correlation : 0.47',size=25);
#도루 횟수 관련 변수 만들기
regular_y['run']=regular_y['SB']+regular_y['CS']

# eqa 구하기
eqa_high=regular_y['H']+regular_y['TB']+1.5*(regular_y['FBP'])+regular_y['run']  #eqa 분자 공식
eqa_low=regular_y['AB']+regular_y['FBP']+regular_y['CS']+(regular_y['run']/3) #eqa 분모 공식
regular_y['eqa']=round(eqa_high/eqa_low,3) #분자/분모
regular_y['eqa']=regular_y['eqa'].fillna(0) # NaN을 0으로 처리

# isop 구하기
regular_y['avg']=regular_y['avg'].replace('-',0.0) #현재 avg중에서 -처리가 된 것이 있어 0으로 대체함
regular_y['SLG']=regular_y['SLG'].fillna(0) # NaN을 0으로 처리
    
coerce_df_columns_to_numeric(regular_y,'avg') #문자로 인식이 되는 문제가 있어 숫자로 변경

regular_y['isop']=regular_y['SLG']-regular_y['avg'] #공식 사용
regular_y['isop']=round(regular_y['isop'],3) #보기 편하기 위해 소수3자리로 반올림

# 운 변수 투입하기
#데이콘 측에서 사용한 행운 변수 추가해보기
regular_y['1b_luck']=regular_y['1B']/(regular_y['AB']-regular_y['HR']-regular_y['SO'])
regular_y['2b_luck']=regular_y['2B']/(regular_y['AB']-regular_y['HR']-regular_y['SO'])
regular_y['3b_luck']=regular_y['3B']/(regular_y['AB']-regular_y['HR']-regular_y['SO'])

#심화적인 변수들중 BB%,SO%가 쓰이는 경우가 있다.

regular_y['BB%']=regular_y['BB']/regular_y['PA']
regular_y['BB%']=regular_y['BB%'].fillna(0.0)
regular_y['BB%']=round(regular_y['BB%'],3)

regular_y['SO%']=regular_y['SO']/regular_y['PA']

#BABIP구하기
regular_y['babip']= round( (regular_y['H']-regular_y['HR']) / (regular_y['AB']-regular_y['SO']-regular_y['HR']+regular_y['fly']) ,3)
regular_y['babip']=regular_y['babip'].fillna(0) # NaN을 0으로 처리
agg={}
for i in range(2006,2019,1):

    regular3=regular_y[ (regular_y['year']==i) & (regular_y['AB']>=100) ]
    regular3=regular3.reset_index(drop=True)

    #인원수를 동일하게 맞춰줘야 되기 때문에 f1 ~f5를 다시 생성하였음

    f1= DataFrame( [( ((regular3['SB'][i]+3) / (regular3['SB'][i]+regular3['CS'][i]+7)) -0.4) * 20 for i in range(len(regular3)) ] ,columns=['f1'])
    f2=DataFrame( [ (math.sqrt( (regular3['SB'][i]+regular3['CS'][i])/ ( (regular3['H'][i]-regular3['2B'][i]-regular3['3B'][i]-regular3['HR'][i]) +regular3['BB'][i]+regular3['HBP'][i]) )) /0.07 for i in range(len(regular3)) ] ,columns=['f2'])
    f3= DataFrame( [ (regular3['3B'][i]/(regular3['AB'][i]-regular3['HR'][i]-regular3['SO'][i]))*625 for i in range(len(regular3)) ] ,columns=['f3'])
    f4= DataFrame( [ (((regular3['R'][i]-regular3['HR'][i])/(regular3['H'][i]+regular3['BB'][i]-regular3['HR'][i]+regular3['HBP'][i]))-0.1)*25 for i in range(len(regular3)) ] ,columns=['f4'])
    f5= DataFrame( [ (0.063-(regular3['GDP'][i]/(regular3['AB'][i]-regular3['HR'][i]-regular3['SO'][i])))/0.007 for i in range(len(regular3)) ] ,columns=['f5'])

    f_total=pd.concat([f1,f2,f3,f4,f5],axis=1)

    scaler = MinMaxScaler()
    scaler.fit(f_total)

    f_total2=DataFrame(scaler.transform(f_total)*10)

    f_total3=pd.concat([regular3[['batter_name','year']],f_total2 ],axis=1)
    f_total3=f_total3.fillna(0) #비어있는 부분을 0으로 채움

    f_total3['spd'] = (f_total3[0]+f_total3[1]+f_total3[2]+f_total3[3]+f_total3[4])/5
    agg[i-2006]=f_total3[['batter_name','year','spd']]
regular3=regular_y[ (regular_y['year']<=2005) & (regular_y['AB']>=100) ]
regular3=regular3.reset_index(drop=True)

#인원수를 동일하게 맞춰줘야 되기 때문에 f1 ~f5를 다시 생성하였음

f1= DataFrame( [( ((regular3['SB'][i]+3) / (regular3['SB'][i]+regular3['CS'][i]+7)) -0.4) * 20 for i in range(len(regular3)) ] ,columns=['f1'])
f2=DataFrame( [ (math.sqrt( (regular3['SB'][i]+regular3['CS'][i])/ ( (regular3['H'][i]-regular3['2B'][i]-regular3['3B'][i]-regular3['HR'][i]) +regular3['BB'][i]+regular3['HBP'][i]) )) /0.07 for i in range(len(regular3)) ] ,columns=['f2'])
f3= DataFrame( [ (regular3['3B'][i]/(regular3['AB'][i]-regular3['HR'][i]-regular3['SO'][i]))*625 for i in range(len(regular3)) ] ,columns=['f3'])
f4= DataFrame( [ (((regular3['R'][i]-regular3['HR'][i])/(regular3['H'][i]+regular3['BB'][i]-regular3['HR'][i]+regular3['HBP'][i]))-0.1)*25 for i in range(len(regular3)) ] ,columns=['f4'])
f5= DataFrame( [ (0.063-(regular3['GDP'][i]/(regular3['AB'][i]-regular3['HR'][i]-regular3['SO'][i])))/0.007 for i in range(len(regular3)) ] ,columns=['f5'])

f_total=pd.concat([f1,f2,f3,f4,f5],axis=1)

scaler = MinMaxScaler()
scaler.fit(f_total)

f_total2=DataFrame(scaler.transform(f_total)*10)

f_total3=pd.concat([regular3[['batter_name','year']],f_total2 ],axis=1)
f_total3=f_total3.fillna(0) #비어있는 부분을 0으로 채움

f_total3['spd'] = (f_total3[0]+f_total3[1]+f_total3[2]+f_total3[3]+f_total3[4])/5
agg[13]=f_total3[['batter_name','year','spd']]
f_real=pd.concat(agg,ignore_index=True)
f_real=f_real.drop_duplicates(keep='first')

regular_y3=pd.merge(regular_y,f_real,how='left',on=['batter_name','year'])

# regular_y3.to_csv("regular3.csv",index=False)
#변화되는 공식이 아닌 일반적인 공식을 사용하였음
regular_y3['spd'].hist(bins=20,figsize=(20,10)) #확인 필요
plt.grid(False)
plt.xticks(size=25)
plt.yticks(size=25)
plt.xlabel('spd value',size=25)
plt.ylabel('인원수(중복) ',fontproperties=fp,size=25)
plt.title('spd 인원 분포',fontproperties=fp,size=30);
#변화되는 공식이 아닌 일반적인 공식을 사용하였음
regular_y3['wOBA']=(0.72*regular_y3['BB'] + 0.75*regular_y3['HBP'] + 0.9*regular_y3['1B'] + 1.24*regular_y3['2B'] + 1.56*regular_y3['3B'] + 1.95*regular_y3['HR'] ) / ( regular_y3['AB'] +regular_y3['FBP'] + regular_y3['fly'] )
regular_y3['wOBA']=regular_y3['wOBA'].fillna(0.0) 

regular_y3['wOBA']=round(regular_y3['wOBA'],3)

regular_y3['wOBA'].hist(bins=20,figsize=(20,10)) #확인 필요
plt.grid(False)
plt.xticks(size=25)
plt.yticks(size=25)
plt.xlabel('wOBA value',size=25)
plt.ylabel('인원수(중복) ',fontproperties=fp,size=25)
plt.title('wOBA 값 변경 전 인원 분포',fontproperties=fp,size=30);
#변화되는 공식이 아닌 일반적인 공식을 사용하였음
regular_y3['wOBA']=(0.72*regular_y3['BB'] + 0.75*regular_y3['HBP'] + 0.9*regular_y3['1B'] + 1.24*regular_y3['2B'] + 1.56*regular_y3['3B'] + 1.95*regular_y3['HR'] ) / ( regular_y3['AB'] +regular_y3['FBP'] + regular_y3['fly'] )
regular_y3['wOBA']=regular_y3['wOBA'].fillna(0.0) 

regular_y3['wOBA']=round(regular_y3['wOBA'],3)
regular_y33=regular_y3.loc[regular_y3['wOBA']<0.6]

regular_y33['wOBA'].hist(bins=20,figsize=(20,10)) #확인 필요
plt.grid(False)
plt.xticks(size=25)
plt.yticks(size=25)
plt.xlabel('wOBA value',size=25)
plt.ylabel('인원수(중복) ',fontproperties=fp,size=25)
plt.title('wOBA 값 변경 후 인원 분포',fontproperties=fp,size=30);
regular_y33['OBP'].hist(bins=20,figsize=(20,10)) #확인 필요
plt.grid(False)
plt.xticks(size=25)
plt.yticks(size=25)
plt.xlabel('OBP',size=25)
plt.ylabel('인원수(중복) ',fontproperties=fp,size=25)
plt.title('OBP 값 인원 분포',fontproperties=fp,size=30);
#각 시즌별 평균 값이 필요하기 때문에 해당 과정을 실시
mra=regular_y3.groupby(['year'])['wOBA', 'R','H','FBP', 'AB', 'fly', 'TB'].sum()
mra2=regular_y3.groupby(['year'])['wOBA', 'R','H','FBP', 'AB', 'fly', 'TB'].size()
mra['size']=mra2

mra.to_csv("mra.csv")
mra=pd.read_csv("mra.csv")
mra
# mra 
#현재 2000년대 연도들은 사이즈가 너무 작아서 평균을 구하는게 힘들어 보임
#그래서 몇개 연도를 합쳐서 그 평균값을 사용하기로 결정함(2004년이전, 2005~2006 통합)
#2003년 이전 합치기
mra.loc[mra['year']<=2003,'wOBA']=(mra.loc[mra['year']<=2003,'wOBA'].sum()) 
mra.loc[mra['year']<=2003,'R']=(mra.loc[mra['year']<=2003,'R'].sum()) 
mra.loc[mra['year']<=2003,'H']=(mra.loc[mra['year']<=2003,'H'].sum()) 
mra.loc[mra['year']<=2003,'FBP']=(mra.loc[mra['year']<=2003,'FBP'].sum()) 
mra.loc[mra['year']<=2003,'AB']=(mra.loc[mra['year']<=2003,'AB'].sum()) 
mra.loc[mra['year']<=2003,'fly']=(mra.loc[mra['year']<=2003,'fly'].sum()) 
mra.loc[mra['year']<=2003,'TB']=(mra.loc[mra['year']<=2003,'TB'].sum()) 
mra.loc[mra['year']<=2003,'size']=(mra.loc[mra['year']<=2003,'size'].sum()) 

#2004년과 2005년 합치기

mra.loc[(mra['year']<=2005) & (mra['year']>=2004) ,'wOBA']= (mra.loc[(mra['year']<=2005) & (mra['year']>=2004) ,'wOBA'].sum()) 
mra.loc[(mra['year']<=2005) & (mra['year']>=2004) ,'R']=(mra.loc[(mra['year']<=2005) & (mra['year']>=2004) ,'R'].sum()) 
mra.loc[(mra['year']<=2005) & (mra['year']>=2004) ,'H']=(mra.loc[(mra['year']<=2005) & (mra['year']>=2004) ,'H'].sum()) 
mra.loc[(mra['year']<=2005) & (mra['year']>=2004) ,'FBP']=(mra.loc[(mra['year']<=2005) & (mra['year']>=2004) ,'FBP'].sum()) 
mra.loc[(mra['year']<=2005) & (mra['year']>=2004) ,'AB']=(mra.loc[(mra['year']<=2005) & (mra['year']>=2004) ,'AB'].sum()) 
mra.loc[(mra['year']<=2005) & (mra['year']>=2004) ,'fly']=(mra.loc[(mra['year']<=2005) & (mra['year']>=2004) ,'fly'].sum())
mra.loc[(mra['year']<=2005) & (mra['year']>=2004) ,'TB']=(mra.loc[(mra['year']<=2005) & (mra['year']>=2004) ,'TB'].sum()) 
mra.loc[(mra['year']<=2005) & (mra['year']>=2004) ,'size']=(mra.loc[(mra['year']<=2005) & (mra['year']>=2004) ,'size'].sum()) 

#2006 2007
mra.loc[(mra['year']<=2007) & (mra['year']>=2006) ,'wOBA']= (mra.loc[(mra['year']<=2007) & (mra['year']>=2006) ,'wOBA'].sum()) 
mra.loc[(mra['year']<=2007) & (mra['year']>=2006) ,'R']=(mra.loc[(mra['year']<=2007) & (mra['year']>=2006) ,'R'].sum()) 
mra.loc[(mra['year']<=2007) & (mra['year']>=2006) ,'H']=(mra.loc[(mra['year']<=2007) & (mra['year']>=2006) ,'H'].sum()) 
mra.loc[(mra['year']<=2007) & (mra['year']>=2006) ,'FBP']=(mra.loc[(mra['year']<=2007) & (mra['year']>=2006) ,'FBP'].sum()) 
mra.loc[(mra['year']<=2007) & (mra['year']>=2006) ,'AB']=(mra.loc[(mra['year']<=2007) & (mra['year']>=2006) ,'AB'].sum()) 
mra.loc[(mra['year']<=2007) & (mra['year']>=2006) ,'fly']=(mra.loc[(mra['year']<=2007) & (mra['year']>=2006) ,'fly'].sum())
mra.loc[(mra['year']<=2007) & (mra['year']>=2006) ,'TB']=(mra.loc[(mra['year']<=2007) & (mra['year']>=2006) ,'TB'].sum()) 
mra.loc[(mra['year']<=2007) & (mra['year']>=2006) ,'size']=(mra.loc[(mra['year']<=2007) & (mra['year']>=2006) ,'size'].sum()) 

#2008 2009
mra.loc[(mra['year']<=2009) & (mra['year']>=2008) ,'wOBA']= (mra.loc[(mra['year']<=2009) & (mra['year']>=2008) ,'wOBA'].sum()) 
mra.loc[(mra['year']<=2009) & (mra['year']>=2008) ,'R']=(mra.loc[(mra['year']<=2009) & (mra['year']>=2008) ,'R'].sum()) 
mra.loc[(mra['year']<=2009) & (mra['year']>=2008) ,'H']=(mra.loc[(mra['year']<=2009) & (mra['year']>=2008) ,'H'].sum()) 
mra.loc[(mra['year']<=2009) & (mra['year']>=2008) ,'FBP']=(mra.loc[(mra['year']<=2009) & (mra['year']>=2008) ,'FBP'].sum()) 
mra.loc[(mra['year']<=2009) & (mra['year']>=2008) ,'AB']=(mra.loc[(mra['year']<=2009) & (mra['year']>=2008) ,'AB'].sum()) 
mra.loc[(mra['year']<=2009) & (mra['year']>=2008) ,'fly']=(mra.loc[(mra['year']<=2009) & (mra['year']>=2008) ,'fly'].sum())
mra.loc[(mra['year']<=2009) & (mra['year']>=2008) ,'TB']=(mra.loc[(mra['year']<=2009) & (mra['year']>=2008) ,'TB'].sum()) 
mra.loc[(mra['year']<=2009) & (mra['year']>=2008) ,'size']=(mra.loc[(mra['year']<=2009) & (mra['year']>=2008) ,'size'].sum()) 
mra['aOBP']=(mra['H']+mra['FBP'])/(mra['AB']+mra['FBP']+mra['fly']) #각 연도의 평균 OBP생성

mra['aSLG']=mra['TB']/mra['AB'] #각 연도의 평균 SLG생성

mra_total=mra[['year','aOBP','aSLG']] #각 연도별 평균 OBP,SLG만 추출

regular4=pd.merge(regular_y3,mra_total,how='left') #각 선수별 OPS+를 적용시키기 위해서 통합 실시

#OPS + 를 구하는 공식을 사용하고 OPS+는 자연수 값이기 때문에 반올림로 정수 처리하였음
regular4['OPS+']= ( (regular4['OBP']/regular4['aOBP']) + (regular4['SLG']/regular4['aSLG']) -1 ) * 100
regular4['OPS+']=round(regular4['OPS+'],0)

regular4['OPS+'].hist(bins=20,figsize=(20,10)) #현재 0이하의 값과 200이상의 이상치들이 보이는 모습이 있음
plt.grid(False)
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel('OPS+ value',size=25)
plt.ylabel('인원수 (중복)',fontproperties=fp,size=25)
plt.title('OPS+ 수정 전 value 인원 분포',fontproperties=fp,size=30);
regular4.loc[regular4['OPS+']>=220,['OPS+']]=0 #200이상의 이상치는 0으로 처리
regular4.loc[regular4['OPS+']<=0,['OPS+']]=0 #0이하의 값은 0으로 처리

#다 사용이 된 평균 OBP,SLG는 삭제함
del regular4['aOBP']
del regular4['aSLG']

regular4['OPS+'].hist(bins=20,figsize=(20,10)) #이상치 제거 이후 OPS+확인 작업
plt.grid(False)
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel('OPS+ value',size=25)
plt.ylabel('인원수 (중복)',fontproperties=fp,size=25)
plt.title('OPS+ value 수정 후 인원 분포',fontproperties=fp,size=30);
regular555=regular4[regular4['OPS']<=1.5]
regular555['OPS'].hist(bins=20,figsize=(20,10)) #이상치 제거 이후 OPS+확인 작업
plt.grid(False)
plt.xticks(size=20)
plt.yticks(size=20)
plt.xlabel('OPS',size=25)
plt.ylabel('인원수 (중복)',fontproperties=fp,size=25)
plt.title('OPS 수정 후 인원 분포',fontproperties=fp,size=30);
regular46=regular4.copy()
regular46=regular46.sort_values(by=['batter_name', 'year']) #이름과 연도 별로 sort

tk=[i for i in regular46['batter_name'][1:]] #각 관측치마다 연차를 부여하기 위해서 생성
tk2=[i for i in regular46['batter_name'][0:]]

co=list() #해당 관측치와 다음 관측치의 이름이 같으면 1년차 -> 2년차의 방식을 빠르게 수행하기 위해서 시행
co.append(1)
for i in range(len(tk)):
    if tk[i]==tk2[i]:
        co.append(co[i]+1)
    else:
        co.append(1)
regular46['work_year']=co

regular46=regular46.sort_index()
regular6=regular46[['batter_name','work_year']]

regular6=regular6.drop_duplicates(keep='first')

c=Counter(Counter(regular6['batter_name']).values()) #이름 갯수로 누적 시즌 횟수 획득
x3=[]
y3=[]
for i in range(len(c)):
    x3.append(sorted(c.items(), key=itemgetter(0))[i][0]) # 1~ 17
    y3.append(sorted(c.items(), key=itemgetter(0))[i][1]) # 해당 시즌 보낸 사람 인원수

f,ax=plt.subplots(figsize=(20,9))

plt.bar(x3,y3,color='grey')
plt.xticks(x3, (list(range(1,20))))
plt.yticks(range(5,80,10))
plt.xlabel('누적 연차',fontproperties=fp,size=25)
plt.ylabel('인원수',fontproperties=fp,size=25)
ax.spines['left'].set_position(('axes', 0.025))
plt.title('누적 연차 인원 분포',fontproperties=fp,size=30)

#ax.spines를 한 이후 축 글자 크기 변경시 사용한 부분
for label in ax.get_xticklabels() + ax.get_yticklabels(): 
    label.set_fontsize(20)
    label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65 ))
    
for x,y in zip(range(1,20),y3):
    if y>=10:
        plt.text(x-0.2,y,y3[x-1],size=20)
    else:
        plt.text(x-0.075,y,y3[x-1],size=20)
        
regular46.to_csv("regular_total.csv",index=False)
total=pd.read_csv("regular_total.csv")
# total=regular46.copy()
total['ball_score']=0.45032

total['ball_score'] = [np.where( total['year'][i]==2009, np.random.normal(0.4297, 0.005, 1)[0],total['ball_score'][i] ) for i in range(len(total))  ]
total['ball_score'] = [np.where( total['year'][i]==2010, np.random.normal(0.4187, 0.005, 1)[0],total['ball_score'][i] ) for i in range(len(total))  ]
total['ball_score'] = [np.where( total['year'][i]==2011, np.random.normal(0.4228, 0.005, 1)[0],total['ball_score'][i] ) for i in range(len(total))  ]
total['ball_score'] = [np.where( total['year'][i]==2012, np.random.normal(0.4172, 0.005, 1)[0],total['ball_score'][i] ) for i in range(len(total))  ]
total['ball_score'] = [np.where( total['year'][i]==2013, np.random.normal(0.4236, 0.005, 1)[0],total['ball_score'][i] ) for i in range(len(total))  ]
total['ball_score'] = [np.where( total['year'][i]==2014, np.random.normal(0.4316, 0.005, 1)[0],total['ball_score'][i] ) for i in range(len(total))  ]
total['ball_score'] = [np.where( total['year'][i]==2015, np.random.normal(0.4244, 0.005, 1)[0],total['ball_score'][i] ) for i in range(len(total))  ]
total['ball_score'] = [np.where( total['year'][i]==2016, np.random.normal(0.4216, 0.005, 1)[0],total['ball_score'][i] ) for i in range(len(total))  ]
total['ball_score'] = [np.where( total['year'][i]==2017, np.random.normal(0.4224, 0.005, 1)[0],total['ball_score'][i] ) for i in range(len(total))  ]
total['ball_score'] = [np.where( total['year'][i]==2018, np.random.normal(0.4187, 0.005, 1)[0],total['ball_score'][i] ) for i in range(len(total))  ]

total['ball_score'] = [np.where( (total['year'][i]<2009) | (total['year'][i]>2018) , np.random.normal(0.42307, 0.005, 1)[0],total['ball_score'][i] ) for i in range(len(total))  ]
total['p_year']=total['year']+1
total['next_ball_score']=0.45032

total['next_ball_score'] = [np.where( total['p_year'][i]<2009, np.random.normal(0.42307, 0.005, 1)[0],total['next_ball_score'][i] ) for i in range(len(total))  ]
total['next_ball_score'] = [np.where( total['p_year'][i]==2009, np.random.normal(0.4297, 0.005, 1)[0],total['next_ball_score'][i] ) for i in range(len(total))  ]
total['next_ball_score'] = [np.where( total['p_year'][i]==2010, np.random.normal(0.4187, 0.005, 1)[0],total['next_ball_score'][i] ) for i in range(len(total))  ]
total['next_ball_score'] = [np.where( total['p_year'][i]==2011, np.random.normal(0.4228, 0.005, 1)[0],total['next_ball_score'][i] ) for i in range(len(total))  ]
total['next_ball_score'] = [np.where( total['p_year'][i]==2012, np.random.normal(0.4172, 0.005, 1)[0],total['next_ball_score'][i] ) for i in range(len(total))  ]
total['next_ball_score'] = [np.where( total['p_year'][i]==2013, np.random.normal(0.4236, 0.005, 1)[0],total['next_ball_score'][i] ) for i in range(len(total))  ]
total['next_ball_score'] = [np.where( total['p_year'][i]==2014, np.random.normal(0.4316, 0.005, 1)[0],total['next_ball_score'][i] ) for i in range(len(total))  ]
total['next_ball_score'] = [np.where( total['p_year'][i]==2015, np.random.normal(0.4244, 0.005, 1)[0],total['next_ball_score'][i] ) for i in range(len(total))  ]
total['next_ball_score'] = [np.where( total['p_year'][i]==2016, np.random.normal(0.4216, 0.005, 1)[0],total['next_ball_score'][i] ) for i in range(len(total))  ]
total['next_ball_score'] = [np.where( total['p_year'][i]==2017, np.random.normal(0.4224, 0.005, 1)[0],total['next_ball_score'][i] ) for i in range(len(total))  ]
total['next_ball_score'] = [np.where( total['p_year'][i]==2018, np.random.normal(0.4187, 0.005, 1)[0],total['next_ball_score'][i] ) for i in range(len(total))  ]
total['next_ball_score'] = [np.where( total['p_year'][i]==2019, np.random.normal(0.41, 0.005, 1)[0],total['next_ball_score'][i] ) for i in range(len(total))  ]
coerce_df_columns_to_numeric(total,'ball_score') #문자로 입력된 것을 숫자로 변경

coerce_df_columns_to_numeric(total,'next_ball_score') #문자로 입력된 것을 숫자로 변경

cor1=total[['ball_score','next_ball_score','YOPS']].corr()

cor1
cor1['ball_score']['YOPS']
fig = plt.figure(figsize=(20,12))

plt.subplot(1,3,1)
plt.scatter(total['ball_score'], total['YOPS'])
plt.title('반발계수와 YOPS 분포',fontproperties=fp,size=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=15)
plt.xlabel('현시즌 반발계수',fontproperties=fp,size=20)
plt.ylabel('YOPS',size=25)
plt.text(0.40,1.85, 'correlation :  ' + str(round(cor1['ball_score']['YOPS'],4)) ,size=25)

plt.subplot(1,3,2)
plt.scatter(total['next_ball_score'], total['YOPS'])
plt.title('다음 시즌 반발계수와 YOPS 분포',fontproperties=fp,size=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=15)
plt.xlabel('다음 시즌 반발계수',fontproperties=fp,size=20)
plt.ylabel('YOPS',size=25)
plt.text(0.395,1.85,'correlation :  ' + str(round(cor1['next_ball_score']['YOPS'],4)),size=25)

plt.subplot(1,3,3)
plt.scatter(total['ball_score'], total['next_ball_score'])
plt.title('반발계수 간 분포',fontproperties=fp,size=25)
plt.xticks(fontsize=20)
plt.yticks(fontsize=15)
plt.xlabel('현 시즌 반발계수',fontproperties=fp,size=25)
plt.ylabel('다음 시즌 반발계수',fontproperties=fp,size=20)
plt.text(0.40,0.445, 'correlation :  ' + str(round(cor1['ball_score']['next_ball_score'],4)) ,size=25);

total.to_csv("real_total.csv",index=False)
total=pd.read_csv("real_total.csv")

del total['BB']
del total['HBP']

del total['PA']

del total['SB']
del total['CS']

del total['OBP']
del total['SLG']

del total['1B']
del total['2B']
del total['3B']

del total['SO']
train=total.copy()
del train['p_year']

train_y2=train[['year','YAB','YOPS']]

y_train=train_y2[train_y2['year']<=2017].reset_index(drop=True)
y_test=train_y2[train_y2['year']==2018].reset_index(drop=True)

del y_train['year']
del y_test['year']

del train['batter_name']
del train['YAB']
del train['YOPS']

train = pd.get_dummies(train,columns=['hand2'])
train = pd.get_dummies(train,columns=['tp'])

X_train=train[train['year']<=2017].reset_index(drop=True)

X_test=train[train['year']==2018].reset_index(drop=True)

del X_train['year']
del X_test['year']
X_train=X_train.fillna(0.0000)

X_test=X_test.fillna(0.0000)

y_train=y_train.fillna(0.0000)

y_test=y_test.fillna(0.0000)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
list1=list(X_train[X_train['OPS']>=1.25].index)
X_train=X_train.drop(list1,axis=0).reset_index(drop=True)
y_train=y_train.drop(list1,axis=0).reset_index(drop=True)

list2=list(X_train[X_train['OPS+']<=25].index)
X_train=X_train.drop(list2,axis=0).reset_index(drop=True)
y_train=y_train.drop(list2,axis=0).reset_index(drop=True)

# list3=list(X_train[ (X_train['AB']<=10) & (X_train['OPS']>=1) ].index)
# X_train=X_train.drop(list3,axis=0).reset_index(drop=True)
# y_train=y_train.drop(list3,axis=0).reset_index(drop=True)

# list4=list(X_train[ (X_train['AB']<=10) & (X_train['OPS']<=0.00001) ].index)
# X_train=X_train.drop(list4,axis=0).reset_index(drop=True)
# y_train=y_train.drop(list4,axis=0).reset_index(drop=True)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
best_grid = RandomForestRegressor(criterion='mae',max_features=3,min_samples_leaf=7, min_samples_split=2, n_estimators=200,random_state=1234)

best_grid.fit(X_train,y_train['YOPS'])
y_pred_train = best_grid.predict(X_train)

print( 'train randomforest regressor mae :' , round(mean_absolute_error(y_train['YOPS'], y_pred_train),4) )
y_pred = best_grid.predict(X_test)

print( 'test randomforest regressor mae :' , round(mean_absolute_error(y_test['YOPS'], y_pred),4) )

print( 'test randomforest regressor 가중치 rmse :' , round( sqrt( (( (((y_test['YOPS']-y_pred)**2)*y_test['YAB']).sum() )  /  y_test['YAB'].sum()) ),4) )
df1 = DataFrame( list(zip(y_test['YOPS'],y_pred,y_test['YAB'])), columns=['YOPS','predict','YAB']  )

df1 = df1[df1['YAB']>=50]

print( '50타수 이상 test randomforest regressor mae :' , round( mean_absolute_error(df1['YOPS'], df1['predict']),4) )
print( '50타수 이상 test randomforest regressor 가중치 rmse :' , round( sqrt( (( (((df1['YOPS']-df1['predict'])**2)*df1['YAB']).sum() )  /  df1['YAB'].sum()) ),4) )
est=XGBRegressor(max_depth=3, learning_rate=0.005, n_estimators=100,
                     objective='reg:linear', booster='gblinear', reg_lambda=1,
                     scale_pos_weight=190, base_score=0.4, random_state=0,
                     seed=None, missing=None)

result=est.fit(X_train, y_train['YOPS'], sample_weight=None, eval_set=None, eval_metric=None,
                  early_stopping_rounds=None, verbose=True, xgb_model=None)
y_pred_train = result.predict(X_train)

print( 'train randomforest regressor mae :' , round(mean_absolute_error(y_train['YOPS'], y_pred_train),4) )
y_pred = result.predict(X_test)

print( 'test randomforest regressor mae :' , round(mean_absolute_error(y_test['YOPS'], y_pred),4) )
print( 'test randomforest regressor 가중치 rmse :' ,round( sqrt( (( (((y_test['YOPS']-y_pred)**2)*y_test['YAB']).sum() )  /  y_test['YAB'].sum()) ),4) )
df1 = DataFrame( list(zip(y_test['YOPS'],y_pred,y_test['YAB'])), columns=['YOPS','predict','YAB']  )

df1 = df1[df1['YAB']>=50]

print( '50타수 이상 test randomforest regressor mae :' , round(mean_absolute_error(df1['YOPS'], df1['predict']),4) )
model2 = LinearRegression().fit(X_train, y_train['YOPS'])

y_pred_train = model2.predict(X_train)

print( 'train randomforest regressor mae :' , round(mean_absolute_error(y_train['YOPS'], y_pred_train),4) )
y_pred = model2.predict(X_test)

print( 'test randomforest regressor mae :' , round(mean_absolute_error(y_test['YOPS'], y_pred),4) )
print( 'test randomforest regressor 가중치 rmse :' ,round( sqrt( (( (((y_test['YOPS']-y_pred)**2)*y_test['YAB']).sum() )  /  y_test['YAB'].sum()) ),4) )
df1 = DataFrame( list(zip(y_test['YOPS'],y_pred,y_test['YAB'])), columns=['YOPS','predict','YAB']  )

df1 = df1[df1['YAB']>=50]

print( '50타수 이상 test randomforest regressor mae :' , round(mean_absolute_error(df1['YOPS'], df1['predict']),4) )
train=total.copy()
del train['p_year']

train_y2=train[['year','YAB','YOPS']]

y_train=train_y2[train_y2['year']<=2015].reset_index(drop=True)
y_test=train_y2[train_y2['year']==2016].reset_index(drop=True)

del y_train['year']
del y_test['year']

del train['batter_name']
del train['YAB']
del train['YOPS']

train = pd.get_dummies(train,columns=['hand2'])
train = pd.get_dummies(train,columns=['tp'])

X_train=train[train['year']<=2015].reset_index(drop=True)

X_test=train[train['year']==2016].reset_index(drop=True)

del X_train['year']
del X_test['year']
X_train=X_train.fillna(0.0000)

X_test=X_test.fillna(0.0000)

y_train=y_train.fillna(0.0000)

y_test=y_test.fillna(0.0000)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
list1=list(X_train[X_train['OPS']>=1.25].index)
X_train=X_train.drop(list1,axis=0).reset_index(drop=True)
y_train=y_train.drop(list1,axis=0).reset_index(drop=True)

list2=list(X_train[X_train['OPS+']<=25].index)
X_train=X_train.drop(list2,axis=0).reset_index(drop=True)
y_train=y_train.drop(list2,axis=0).reset_index(drop=True)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
rf = RandomForestRegressor(criterion='mae',max_features=3,min_samples_leaf=7, min_samples_split=2, n_estimators=200,random_state=1234)

rf.fit(X_train,y_train['YOPS'])
y_pred_train = rf.predict(X_train)

print( 'train randomforest regressor mae :' , round(mean_absolute_error(y_train['YOPS'], y_pred_train),4) )
y_pred = best_grid.predict(X_test)

print( 'test randomforest regressor mae :' , round(mean_absolute_error(y_test['YOPS'], y_pred),4) )
print( 'test randomforest regressor 가중치 rmse :' , round( sqrt( (( (((y_test['YOPS']-y_pred)**2)*y_test['YAB']).sum() )  /  y_test['YAB'].sum()) ),4) )
df1 = DataFrame( list(zip(y_test['YOPS'],y_pred,y_test['YAB'])), columns=['YOPS','predict','YAB']  )

df1 = df1[df1['YAB']>=50]

print( '50타수 이상 test randomforest regressor mae :' , round( mean_absolute_error(df1['YOPS'], df1['predict']),4) )
print( '50타수 이상 test randomforest regressor 가중치 rmse :' , round( sqrt( (( (((df1['YOPS']-df1['predict'])**2)*df1['YAB']).sum() )  /  df1['YAB'].sum()) ),4) )
train=total.copy()
del train['p_year']

train_y2=train[['year','YAB','YOPS']]

y_train=train_y2[train_y2['year']<=2016].reset_index(drop=True)
y_test=train_y2[train_y2['year']==2017].reset_index(drop=True)

del y_train['year']
del y_test['year']

del train['batter_name']
del train['YAB']
del train['YOPS']

train = pd.get_dummies(train,columns=['hand2'])
train = pd.get_dummies(train,columns=['tp'])

X_train=train[train['year']<=2016].reset_index(drop=True)

X_test=train[train['year']==2017].reset_index(drop=True)

del X_train['year']
del X_test['year']
X_train=X_train.fillna(0.0000)

X_test=X_test.fillna(0.0000)

y_train=y_train.fillna(0.0000)

y_test=y_test.fillna(0.0000)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
list1=list(X_train[X_train['OPS']>=1.25].index)
X_train=X_train.drop(list1,axis=0).reset_index(drop=True)
y_train=y_train.drop(list1,axis=0).reset_index(drop=True)

list2=list(X_train[X_train['OPS+']<=25].index)
X_train=X_train.drop(list2,axis=0).reset_index(drop=True)
y_train=y_train.drop(list2,axis=0).reset_index(drop=True)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
rf = RandomForestRegressor(criterion='mae',max_features=3,min_samples_leaf=7, min_samples_split=2, n_estimators=200,random_state=1234)

rf.fit(X_train,y_train['YOPS'])
y_pred_train = rf.predict(X_train)

print( 'train randomforest regressor mae :' , round(mean_absolute_error(y_train['YOPS'], y_pred_train),4) )
y_pred = best_grid.predict(X_test)

print( 'test randomforest regressor mae :' , round(mean_absolute_error(y_test['YOPS'], y_pred),4) )
print( 'test randomforest regressor 가중치 rmse :' , round( sqrt( (( (((y_test['YOPS']-y_pred)**2)*y_test['YAB']).sum() )  /  y_test['YAB'].sum()) ),4) )
df1 = DataFrame( list(zip(y_test['YOPS'],y_pred,y_test['YAB'])), columns=['YOPS','predict','YAB']  )

df1 = df1[df1['YAB']>=50]

print( '50타수 이상 test randomforest regressor mae :' , round( mean_absolute_error(df1['YOPS'], df1['predict']),4) )
print( '50타수 이상 test randomforest regressor 가중치 rmse :' , round( sqrt( (( (((df1['YOPS']-df1['predict'])**2)*df1['YAB']).sum() )  /  df1['YAB'].sum()) ),4) )
train=total.copy()
del train['p_year']

train_y2=train[['year','YAB','YOPS']]

y_train=train_y2[train_y2['year']<=2017].reset_index(drop=True)
y_test=train_y2[train_y2['year']==2018].reset_index(drop=True)

del y_train['year']
del y_test['year']

del train['batter_name']
del train['YAB']
del train['YOPS']

train = pd.get_dummies(train,columns=['hand2'])
train = pd.get_dummies(train,columns=['tp'])

X_train=train[train['year']<=2017].reset_index(drop=True)

X_test=train[train['year']==2018].reset_index(drop=True)

del X_train['year']
del X_test['year']
X_train=X_train.fillna(0.0000)

X_test=X_test.fillna(0.0000)

y_train=y_train.fillna(0.0000)

y_test=y_test.fillna(0.0000)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
list1=list(X_train[X_train['OPS']>=1.25].index)
X_train=X_train.drop(list1,axis=0).reset_index(drop=True)
y_train=y_train.drop(list1,axis=0).reset_index(drop=True)

list2=list(X_train[X_train['OPS+']<=25].index)
X_train=X_train.drop(list2,axis=0).reset_index(drop=True)
y_train=y_train.drop(list2,axis=0).reset_index(drop=True)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
rf = RandomForestRegressor(criterion='mae',max_features=3,min_samples_leaf=7, min_samples_split=2, n_estimators=200,random_state=1234)

rf.fit(X_train,y_train['YOPS'])
y_pred_train = rf.predict(X_train)

print( 'train randomforest regressor mae :' , round(mean_absolute_error(y_train['YOPS'], y_pred_train),4) )
y_pred = best_grid.predict(X_test)

print( 'test randomforest regressor mae :' , round(mean_absolute_error(y_test['YOPS'], y_pred),4) )
print( 'test randomforest regressor 가중치 rmse :' , round( sqrt( (( (((y_test['YOPS']-y_pred)**2)*y_test['YAB']).sum() )  /  y_test['YAB'].sum()) ),4) )
df1 = DataFrame( list(zip(y_test['YOPS'],y_pred,y_test['YAB'])), columns=['YOPS','predict','YAB']  )

df1 = df1[df1['YAB']>=50]

print( '50타수 이상 test randomforest regressor mae :' , round( mean_absolute_error(df1['YOPS'], df1['predict']),4) )
print( '50타수 이상 test randomforest regressor 가중치 rmse :' , round( sqrt( (( (((df1['YOPS']-df1['predict'])**2)*df1['YAB']).sum() )  /  df1['YAB'].sum()) ),4) )
n_feature = X_train.shape[1] #주어진 변수들의 갯수를 구함
index = np.arange(n_feature)
input_var=X_train.columns
score2=DataFrame( list(zip(input_var,best_grid.feature_importances_  )), columns=['name','importance'])
score2=score2.sort_values('importance')
score3=score2.sort_values('importance',ascending=False)

score3.head(6)
score3
plt.rcParams["figure.figsize"] = (20,15)

plt.barh(index, score2['importance'], align='center') #
plt.xticks(size=20)
plt.yticks(index, score2['name'],fontproperties=fp,size=20)
plt.ylim(-1, n_feature)
plt.xlabel('feature importance', size=25)
plt.ylabel('feature', size=25)
plt.title('RandomForest Regressor feature importance', size=30)
plt.show();
