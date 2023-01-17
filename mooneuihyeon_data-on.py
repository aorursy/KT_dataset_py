# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib import font_manager, rc

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session





df = pd.read_csv('/kaggle/input/combat-data/Combat_Data.csv', engine = 'python', encoding = 'euc-kr')

display(df.head())

print(df.columns)
import missingno as msno

#checking if there's any empty values on the chart.



msno.matrix(df)

plt.show()



print(df.isnull().sum()) #결측치 확인
df1 = df.drop(['사료 번호', '연대', '대대', '중대', '기타소속', '특이사항', '의병 지휘부', '한국군사망', '년도', '월', '일', '일본군수'], axis = 1)

df1.tail()
df1['일본군사상자'] = df1['일본군사상자'].fillna(0)

df1['의병수'] = df1['의병수'].fillna(0)

df1['의병사망'] = df1['의병사망'].fillna(0)

df1
df1 = df1.drop('지역', axis = 1)

df1
import missingno as msno

#checking if there's any empty values on the chart.



msno.matrix(df1)

plt.show()



print(df1.isnull().sum()) #결측치 확인
df1.dropna(axis = 0, how = 'any' ,inplace = True)

df1
df2 = df1['의병수'] == '불명'

df2.sum() #'의병수'가 불명인 값은 3개인데 역사적 사실을 왜곡할 수 없으니까(mean 값으로 채워넣는다던가) 이것도 drop
not_string = df1['의병수'] != '불명'

df_final = df1[not_string]



non_string = df1['의병사망'] != '불명'

df_final = df1[non_string]



df2 = df_final['의병수'] == '불명'

df3 = df_final['의병사망'] == '불명'

print(df2.sum())

print(df3.sum())
not_string = df_final['의병수'] != '미상'

df_final = df_final[not_string]



not_string = df_final['의병수'] != '불명'

df_final = df_final[not_string]





not_string = df_final['의병사망'] != '미상'

df_final = df_final[not_string]



not_string = df_final['의병사망'] != '불명'

df_final = df_final[not_string]
df_final.rename(columns = {"날짜": "Date", "지역 분류": "Region", "의병수": "Chosun_soldiers", "일본군사상자": "Dead_Japanese_soldiers", "의병사망": "Dead_Chosun_soldiers"}, inplace = True)

 

print(df_final)

print(list(df_final['Region'].unique()))



df_final = df_final.replace('경기도', 'Gyeonggido')

df_final = df_final.replace('강원도', 'Gangwondo')

df_final = df_final.replace('경상북도', 'Gyeonsangbukdo')

df_final = df_final.replace('경상남도', 'Gyeonsangnamdo')

df_final = df_final.replace('황해도', 'Hwanghaedo')

df_final = df_final.replace('함경도', 'Hamgyeongdo')

df_final = df_final.replace('평안도', 'Pyeongando')

df_final = df_final.replace('충청남도', 'Chungcheongnamdo')

df_final = df_final.replace('충청북도', 'Chungcheongbukdo')

df_final = df_final.replace('전라북도', 'Jeollabukdo')

df_final = df_final.replace('전라남도', 'Jeollanamdo')



print(df_final)

# 한글폰트 적용이 도저히 안 돼서 영어로 이름 바꾼다.
df_final['Dead_Chosun_soldiers'] = df_final['Dead_Chosun_soldiers'].astype('float')

df_final['Chosun_soldiers'] = df_final['Chosun_soldiers'].astype('float')
%matplotlib inline

import matplotlib.pylab as plt



plt.rcParams["figure.figsize"] = (40,28)

plt.rcParams['lines.linewidth'] = 2

plt.rcParams['lines.color'] = 'r'

plt.rcParams['axes.grid'] = True 

plt.rc('font', size = 25)
df_date = df_final.drop('Region', axis = 1)

df_region = df_final.drop('Date', axis = 1)



df_date = df_date.set_index("Date")

df_region = df_region.set_index("Region")



display(df_date)

display(df_region)
df_date['Dead_Japanese_soldiers'].plot(marker='o')

plt.title("Dead Japanese Soldiers by Date")

plt.xlabel("Date")

plt.ylabel("Number")

plt.show()
df_date['Dead_Chosun_soldiers'].plot(marker='o')

plt.title("Dead Chosun soldiers by Date")

plt.xlabel("Date")

plt.ylabel("Number")

plt.show()
df_date['Chosun_soldiers'].plot(marker='o')

plt.title("Chosun soldiers by Date")

plt.xlabel("Date")

plt.ylabel("Number")

plt.show()
df_date.plot(marker='o')

plt.title("Statistics by Date")

plt.xlabel("Date")

plt.ylabel("Number")

plt.show()

print( df_final['Region'].value_counts())
sizes = df_final['Region'].value_counts()

labels = ['Gyeonggido', 'Gangwondo','Gyeonsangbukdo','Chungcheongbukdo','Jeollabukdo','Hwanghaedo', 'Jeollanamdo', 'Chungcheongnamdo', 'Gyeonsangnamdo','Pyeongando', 'Hamgyeongdo']

plt.pie(sizes,labels = df_final['Region'].unique(), autopct='%1.1f%%', shadow=True, startangle=90, radius = 10)

plt.axis('equal')



plt.tight_layout()

plt.show()
print(list(df_final['Region'].unique()))
Gyeonggido = df_final[df_final['Region'].str.contains("Gyeonggido")]

Gyeonggido.plot(marker='o')

plt.title("Statistic in Gyeonggido")

plt.xlabel("Date")

plt.ylabel("Number")

plt.show()
date_with_region = df_final.set_index("Date")

date_with_region
for reg in list(date_with_region['Region'].unique()):

    region = date_with_region[date_with_region['Region'].str.contains(f"{reg}")]



    region.plot(marker='o')

    plt.title(f"Statistics in summary in {reg}")

    plt.xlabel("Date")

    plt.ylabel("Number")

    plt.show()