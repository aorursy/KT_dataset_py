# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
TimeAge = pd.read_csv("/kaggle/input/coronavirusdataset/TimeAge.csv")
Region = pd.read_csv("/kaggle/input/coronavirusdataset/Region.csv")
Time = pd.read_csv("/kaggle/input/coronavirusdataset/Time.csv")
TimeGender = pd.read_csv("/kaggle/input/coronavirusdataset/TimeGender.csv")
Weather = pd.read_csv("/kaggle/input/coronavirusdataset/Weather.csv")
SearchTrend = pd.read_csv("/kaggle/input/coronavirusdataset/SearchTrend.csv")
TimeProvince = pd.read_csv("/kaggle/input/coronavirusdataset/TimeProvince.csv")
PatientInfo = pd.read_csv("/kaggle/input/coronavirusdataset/PatientInfo.csv")
Policy = pd.read_csv("/kaggle/input/coronavirusdataset/Policy.csv")
SeoulFloating = pd.read_csv("/kaggle/input/coronavirusdataset/SeoulFloating.csv")
Case = pd.read_csv("/kaggle/input/coronavirusdataset/Case.csv")
TimeAge.head()
sns.barplot(x='age', y = 'confirmed', data=TimeAge)
plt.title('Age Group and Confirmed Cases of Coronavirus')
sns.set_style("whitegrid")
plt.xlabel("Age Group")
plt.ylabel("Confirmed Coronavirus Cases")
plt.show()
sns.barplot(x='age', y = 'deceased', data=TimeAge)
plt.title('Age Group and Deceased Cases of Coronavirus')
plt.xlabel("Age Group")
plt.ylabel("Confirmed Coronavirus Deaths")
plt.show()
ratio_of_deaths_to_confirmed = TimeAge['deceased']/TimeAge['confirmed']
sns.barplot(TimeAge['age'], ratio_of_deaths_to_confirmed)
plt.title("Ratio of Deaths to Confirmed Coronavirus Cases and Age Group")
plt.xlabel("Age Group")
plt.ylabel("Ratio of Deaths to Confirmed Coronavirus Cases")
sns.set_style("whitegrid")
plt.show()
Time.head()
Time.tail()
sns.lineplot(x="date", y="confirmed", data=Time)
sns.set_style("whitegrid")
plt.title("Confirmed Cases Over Time")
plt.xlabel("Date")
plt.ylabel("Confirmed Cases of Coronavirus")
plt.show()
deceased_to_confirmed_ratio = Time['deceased']/Time['confirmed']
released_to_confirmed_ratio = Time['released']/Time['confirmed']
sns.lineplot(x=Time['date'], y=deceased_to_confirmed_ratio, label='Deceased to Confirmed Ratio')
sns.lineplot(x=Time['date'], y=released_to_confirmed_ratio, label='Released to Confirmed Ratio')
plt.show()

sns.lineplot(x='date', y='deceased', data=Time)
sns.set_style("whitegrid")
plt.title("Deceased Cases Over Time")
plt.xlabel("Date")
plt.ylabel("Deceased From Coronavirus")
plt.show()

TimeGender.head()
male_data = TimeGender[TimeGender['sex'] == 'male']
female_data = TimeGender[TimeGender['sex']=='female']
sns.barplot(x='sex', y= 'confirmed', data=TimeGender)
plt.title("Different Sex Confirmed Cases")
plt.xlabel("Sex")
plt.ylabel("Number of Confirmed Cases")
plt.show()
sns.barplot(x='sex', y='deceased', data=TimeGender)
plt.title("Different Sex Deceased Cases")
plt.xlabel("Sex")
plt.ylabel("Number of Deceased Cases")
plt.show()
print(male_data.head())
print(female_data.head())
sns.lineplot(x=TimeGender['date'], y=male_data['confirmed'], label='Male')
sns.lineplot(x=TimeGender['date'], y=female_data['confirmed'], label='Female')
plt.title("Different Sex Confirmed Cases of Coronavirus Over Time")
plt.xlabel('Date')
plt.ylabel('Number of Confirmed Cases')
plt.show()
sns.lineplot(x=TimeGender['date'], y=male_data['deceased'], label='Male')
sns.lineplot(x=TimeGender['date'], y=female_data['deceased'], label='Female')
plt.title("Different Sex Deceased Cases of Coronavirus Over Time")
plt.xlabel('Date')
plt.ylabel('Number of Deceased Cases')
plt.show()
m_ratio_deceased_to_confirmed = male_data['deceased']/male_data['confirmed']
f_ratio_deceased_to_confirmed = female_data['deceased']/female_data['confirmed']
sns.lineplot(x=TimeGender['date'], y=m_ratio_deceased_to_confirmed, label='Male')
sns.lineplot(x=TimeGender['date'], y = f_ratio_deceased_to_confirmed,label='Female')
plt.title("Deceased To Confirmed Sex Ratios Over Time")
plt.xlabel('Date')
plt.ylabel('Deceased To Confirmed Sex Ratios')
plt.show()
print(Weather.head())
print(Weather.tail())
unique_provinces = Weather.province.unique()
print(unique_provinces)

print(SearchTrend.info())
print(SearchTrend.head())
print(SearchTrend.tail())
sns.lineplot(x='date', y ='cold', data = SearchTrend, label = 'cold')
sns.lineplot(x='date', y ='flu', data=SearchTrend, label= 'flu')
sns.lineplot(x='date', y ='pneumonia', data=SearchTrend, label ='pneumonia')
sns.lineplot(x='date', y ='coronavirus', data=SearchTrend, label='coronavirus')
plt.title("Keyword Search Trends from January 1, 2016 till June 29, 2020")
plt.xlabel("Date")
plt.ylabel("Daily Hits")
plt.show()
sns.lineplot(x='date', y='coronavirus', data=SearchTrend)
plt.title("Coronavirus Search Trend from January 1, 2016 to June 29, 2020")
plt.xlabel("Date")
plt.ylabel("Daily Hits")
plt.show()
print(TimeProvince.head())
print(TimeProvince.tail())
provinces = TimeProvince.province.unique()
print(provinces)
def province_timeplot(pro):
    for i in pro:
        curr_y = TimeProvince['province'] == i
        sns.lineplot(x=curr_y['date'], y=curr_y['confirmed'])
        plt.title(i + " Province Coronavirus Confirmed Cases Time Plot")
        plt.xlabel('Date')
        plt.ylabel('Number of Confirmed Cases')
        plt.show()
province_timeplot(provinces)
Policy.head()
Policy.tail()
SeoulFloating.head()

Case.head()
sns.barplot(x='city', y="confirmed", data=Case)
plt.title("Korean Cities and Confirmed Cases")
plt.xlabel('Korean Cities')
plt.ylabel('Confirmed Cases')
plt.show()
sns.barplot(x='province', y="confirmed", data=Case)
plt.title("Korean Provinces and Confirmed Cases")
plt.xlabel("Korean Provinces")
plt.ylabel("Confirmed Cases")
plt.show()
Case.sort_values('confirmed',inplace=True,ascending=False)