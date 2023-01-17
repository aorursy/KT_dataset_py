# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
hospitals = pd.read_csv("/kaggle/input/uncover/UNCOVER/hifld/hifld/hospitals.csv")
casesbycountry=pd.read_csv("/kaggle/input/uncover/UNCOVER/public_health_england/covid-19-cases-by-county-uas.csv")
casescanada =pd.read_csv("/kaggle/input/uncover/UNCOVER/covid_19_canada_open_data_working_group/public-covid-19-cases-canada.csv")          
cumulativetesting=pd.read_csv("/kaggle/input/uncover/UNCOVER_v4/UNCOVER/covid_19_canada_open_data_working_group/time-series-of-cumulative-testing.csv")
 #casesbycountry.head(10)   
cumulativetesting.head(10)  
cumulativetesting.info()

#Her şehir için test sayısının grafiğini çizdirelim.
unique_city_list= list(cumulativetesting['province'].unique())
cumulativetesting['cumulative_testing'] = cumulativetesting['cumulative_testing'].astype(float)
testing_ratio=[]

for i in unique_city_list:
    x = cumulativetesting[cumulativetesting['province'] == i]
    province_test_ratio = sum(x.cumulative_testing) / len(x)
    testing_ratio.append(province_test_ratio)

    
data = pd.DataFrame({'unique_city_list' : unique_city_list, 'test_ratio': testing_ratio})
new_index=(data['test_ratio'].sort_values(ascending=False)).index.values
sorted_data= data.reindex(new_index) # Şimdi sorted_data dataframe ine göre görselleştirme yapabiliriz.

plt.figure(figsize=(10,8))
sns.barplot(x=sorted_data['unique_city_list'], y= sorted_data['test_ratio'])
plt.xticks(rotation=45)
plt.xlabel('States')
plt.ylabel('Test Ratio')
plt.title('Test ratios for Corona Virus')


