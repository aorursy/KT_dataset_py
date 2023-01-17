# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200131.csv")
data
# Satir Sayisi
print(data.shape[0])



# Sutun Adlari
print(data.columns.tolist())


# Veri Tipleri
print(data.dtypes)
data.info()
data.describe()
data.groupby('Country/Region').mean()
data.groupby('Province/State').mean()
print(data.columns)
data.isnull().sum()
print(data.corr())
data.describe()
sns.pairplot(data)

data['Recovered'].mean()
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score
x=data["Confirmed"].replace(np.NaN, data["Confirmed"].mean())
y=data["Recovered"].replace(np.NaN, data["Recovered"].mean())
x=np.array(x)
y=np.array(y)
reg = linear_model.LinearRegression(normalize='Ture')
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)


reg.fit(X_train, y_train)

print(reg.score(X_train, y_train))
sns.regplot(x=X_train,y=y_train)
province_state_country = pd.pivot_table(data,index=["Province/State"] ,aggfunc=np.sum).sort_values(by='Confirmed', ascending=False)
province_state_country[:10]
province_state_country[:10].plot(kind='bar' ,figsize=(10, 4), width=1,zorder=12)

province_state_country[1:10].plot(kind='bar' ,figsize=(10, 4), width=2)
country_details  = pd.pivot_table(data,index=['Country/Region'] ,aggfunc=np.sum).sort_values(by='Confirmed', ascending=False)
country_details[:5]
country_details[0:5].plot(kind='bar' ,figsize=(10, 4), width=1,zorder=12,rot=2)
country_details[1:6].plot(kind='bar' ,figsize=(10, 4), width=1,zorder=12,rot=1)
data2= pd.read_csv("../input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126_cleaned.csv")
cpd = pd.pivot_table(data2,index=["Country","Province/State",'Date last updated'] )
cpd
cpd.loc[('Australia', ), :]
data3 = pd.read_csv('/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nCoV_20200121_20200201.csv', parse_dates = ['Last Update'])

countries = data3['Country/Region'].unique().tolist()
print(countries)

print("\nTotal countries affected by virus: ",len(countries))
data3['Country/Region'].replace({'Mainland China':'China'},inplace=True)
countries = data3['Country/Region'].unique().tolist()
print(countries)
print("\nTotal countries affected by virus: ",len(countries))
China = data3[data3['Country/Region']=='China']

f, ax = plt.subplots(figsize=(12, 8))

sns.set_color_codes("pastel")
sns.barplot(x="Confirmed", y="Province/State", data=China[1:],
            label="Confirmed", color="r")

sns.set_color_codes("muted")
sns.barplot(x="Recovered", y="Province/State", data=China[1:],
            label="Recovered", color="g")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 400), ylabel="",
       xlabel="Stats")
sns.despine(left=True, bottom=True)