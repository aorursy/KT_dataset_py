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
import pandas as pd
covid_train=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')

covid_train.head()
covid_test=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/test.csv')

covid_test.head()
covid_train.shape
covid_test.shape
covid_train.isnull().sum()
covid_test.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.heatmap(covid_train.isnull(),yticklabels=False, cbar=False, cmap='viridis')
covid_train.Target.value_counts()
sns.set_style('whitegrid')

sns.countplot(x='Target', data=covid_train)
sns.distplot(covid_train['Weight'], bins=30)
plt.figure(figsize=(12, 8))

sns.boxplot(x='Target',y='Population', data=covid_train, palette='winter')
covid_train.describe()
covid_train.info()
covid_train.corr()
sns.heatmap(covid_train.corr(), cmap="YlGnBu", annot=True, linewidths=2)
sns.pairplot(covid_train)
covid_train.groupby('Country_Region')['TargetValue'].mean().plot(kind = 'bar', figsize= (40,20), title= "Countries with COVID-19 MAX", color='red')
df1 = covid_train.Population.groupby(covid_train['Country_Region']).max().sort_values(ascending= False)

df1.head(10)
df10 = pd.DataFrame()

df20 = pd.DataFrame()

df10['population'] = df1.iloc[0:10]

df10['country'] = df10.index

df20['population'] = df1.iloc[11:20]

df20['country'] = df20.index
plt.figure(figsize =(10,10))

plt.subplot(2,1,1)

sns.barplot(x='country', y='population', data=df10, orient ='v')

plt.xlabel('Country')

plt.title('Popoulation Top 10')

plt.subplot(2,1,2)

sns.barplot(x='country', y='population', data=df20, orient ='v')

plt.xlabel('Country')

plt.title('Population Next 10')
Target_df = covid_train.Target.value_counts()

Target_df
Target = covid_train[covid_train['Target'] =='ConfirmedCases']

Conf_Cases = pd.DataFrame()

Conf_Cases['values'] = Target.TargetValue.groupby(Target['Country_Region']).sum().sort_values(ascending= False)

Conf_Cases['country'] = Conf_Cases.index

Conf_Cases.index = np.arange(0,len(Conf_Cases))

data10 = Conf_Cases.iloc[0:10,:]

data20 = Conf_Cases.iloc[11:20,:]
plt.figure(figsize =(10,10))

plt.subplot(2,1,1)

sns.barplot(x='country', y='values', data=data10, orient ='v')

plt.xlabel('Country')

plt.ylabel('Cases')

plt.title('Covid Cases Top 10')

plt.subplot(2,1,2)

sns.barplot(x='country', y='values', data=data20, orient ='v')

plt.xlabel('Country')

plt.ylabel('Cases')

plt.title('Covid Cases Next 10')
Target = covid_train[covid_train['Target']!='ConfirmedCases']

Conf_Cases = pd.DataFrame()

Conf_Cases['values'] = Target.TargetValue.groupby(Target['Country_Region']).sum().sort_values(ascending= False)

Conf_Cases['country'] = Conf_Cases.index

Conf_Cases.index = np.arange(0,len(Conf_Cases))

data10 = Conf_Cases.iloc[0:10,:]

data20 = Conf_Cases.iloc[11:20,:]
plt.figure(figsize =(10,10))

plt.subplot(2,1,1)

sns.barplot(x='country', y='values', data=data10, orient ='v')

plt.xlabel('Country')

plt.ylabel('Deaths')

plt.title('Covid Cases Top 10')

plt.subplot(2,1,2)

sns.barplot(x='country', y='values', data=data20, orient ='v')

plt.xlabel('Country')

plt.ylabel('Deaths')

plt.title('Covid Cases Next 10')
covid_train[covid_train['Country_Region']=='India'].groupby(by='Country_Region').sum()
d_train = covid_train.drop(['County','Province_State','Country_Region','Target','Date'], axis=1)

d_test = covid_train.drop(['County','Province_State','Country_Region','Target','Date'], axis=1)
X = d_train.iloc[:,1:3]

Y = d_train.iloc[:,3]
from sklearn.model_selection import train_test_split



predictors = d_train.drop(['TargetValue', 'Id'], axis=1)

target = d_train['TargetValue']

X_train, X_test, Y_train, Y_test = train_test_split(predictors,target, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

 

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.linear_model import LinearRegression

regression=LinearRegression()

regression.fit(X_train,Y_train)
from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(X_train, Y_train)
DTCscore = model.score(X_train,Y_train)

print("Decision Tree Score: ",DTCscore)
sub=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/submission.csv')

sub.to_csv('submission.csv',index=False)