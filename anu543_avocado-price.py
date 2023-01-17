# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

%matplotlib inline



import matplotlib 

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/avocado-prices/avocado.csv')

df.shape


print("ROWS=" ,df.shape[0])

print("COLUMNS =" ,df.shape[1])

print("Features =\n" ,df.columns)

print("MISSING VALUE_ =\n" ,df.isna().values.sum())

print("UNIQUE VALUES: \n" ,df.nunique())


import missingno as msno

msno.matrix(df)

df.head(5)
# Average price

sns.boxplot(df['AveragePrice'])
# number of outlier rows presents in Average Price columns

df[df['AveragePrice']> 3].shape
#Total Volume

sns.distplot(df['Total Volume'])
fig ,axs = plt.subplots(1 , 4  , figsize =(20 ,7))

for i ,f in enumerate (['Total Bags' ,'Small Bags' ,'Large Bags' ,'XLarge Bags']):

    sns.distplot(df[f] ,ax = axs[i]).set_title(f)
df['type'].value_counts()
plt.pie(df['type'].value_counts() ,shadow= True ,labels=['coventional' ,'organic'] ,autopct= '%2.f' ,explode=(0 ,0.1));
plt.figure(figsize=(20,7))

plt.bar(df['year'] , df['year'])

plt.xticks(rotation = '45')
df['region'].value_counts()
df.select_dtypes(include='object').columns
df.select_dtypes(include=['int' ,'float']).columns
# Handling of Numerical Variable

feature1 = ['AveragePrice', 'Total Volume', '4046', '4225', '4770','Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags']

print(len(df['region'].unique()))

print(len(df['type'].unique()))


from sklearn.preprocessing import LabelEncoder

encode =LabelEncoder()

df['type'] = encode.fit_transform(df['type'])
df['type'].unique()
print(len(df['region'].unique()))
# Creating final DF

df = pd.concat([df[feature1] ,df['type'] ,df['region']] ,axis =1)

df.head(4)
x = df.drop('AveragePrice' , axis =1)

target =df['AveragePrice']
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,target,test_size=0.3,random_state=101) # y1 is my independent var

x_train.shape

y_train.shape
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train = sc.fit_transform(x_train)

x_test =sc.fit_transform(x_test)
from sklearn.linear_model import LinearRegression



model =LinearRegression()

model.fit(x_train ,y_train)

pred = model.predict(x_test)



from sklearn import metrics

from sklearn.metrics import mean_squared_error, r2_score



print('R-Squared:' , metrics.r2_score(y_test, pred))

print('Root Mean Square error:', np.sqrt(metrics.mean_squared_error(y_test, pred)))

pred
import statsmodels.api as sm

p =sm.add_constant(x)

p.head()
stat_model = sm.OLS(target ,x).fit()

print(stat_model.summary())