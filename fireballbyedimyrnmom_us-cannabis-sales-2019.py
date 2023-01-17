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
import pandas as pd 

  

data = {'State':['WA', 'OR', 'CA', 'NV', 'CO', 'MA'], 'Sales':[1100000000, 752000000, 3100000000, 639000000, 1747990628, 394000000], 

        'Tax Revenue':[407000000, 127840000, 288000000, 99184974, 262198594, 42158000]} 

  

df = pd.DataFrame(data) 

df
df.info()
df.describe()
a=df.corr()

a
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats
sns.heatmap(a, annot=True)
x=df['Sales']

y=df['Tax Revenue']
sns.kdeplot(x, label="Sales")

sns.kdeplot(y, label="Tax")

plt.legend()
sns.jointplot(x, y, data=df, kind="kde", color='salmon')
sns.pairplot(df)
sns.set_color_codes()

ax = sns.distplot(x, color="y")
ax = sns.distplot(y, color="r")
plt.plot(x, x + 0, '-g')  # solid green

plt.plot(y, x + 1, '--c') # dashed cyan
plt.scatter(x, y, c='orange', marker='*', data=df)
sns.set(style="whitegrid")

ax = sns.barplot(x, y, data=df, color="green")
# getting Difference 

df['Revenue after taxes'] = x-y

df
#difference with previous row

df1= df.drop('State', axis=1)

df1=df1.diff()

df1
data1 = {'State':['NM', 'SD', 'AZ', 'AR', 'MO', 'MD'], 'Violent crime rate per 100,000 people':[856.6, 404.7, 474.9, 541.1, 502.1, 468.7]} 

  

df2 = pd.DataFrame(data1) 

df2
data2 = {'State':['WA', 'OR', 'CA', 'NV', 'CO', 'MA'], 'Violent crime rate per 100,000 people':[311.5, 285.5, 447.4, 543.6, 397.2, 338.1]} 

  

df3 = pd.DataFrame(data2) 

df3
stateCrime = pd.concat([df2, df3])



stateCrime
p1 = stateCrime.plot(kind='bar')
stateCrime.plot.bar(color='orange')
data3 = {'State':['WA', 'OR', 'CA', 'NV', 'CO', 'MA'],  'Other industry revenue':[7000000000, 1030279086, 14419579138, 8700000000, 286300000, 3000000000], 'Industry':['Wine', 'Liquor', 'Wine', 'Casinos', 'Tobacco', 'Beer Wine & Liquor Stores']} 

  

other = pd.DataFrame(data3) 

other
up = other['Other industry revenue']

labels = other['State']

plt.style.use('ggplot')

plt.xticks(range(len(up)), labels)

plt.xlabel('State')

plt.ylabel('Other industry revenue')

plt.title('Other industries')

plt.bar(range(len(up)), up) 

plt.show()
compareAll= pd.concat([df, df1, df2, df3, other], axis=1)

compareAll
columns1 = [x for x in range(compareAll.shape[1])]  # list of columns' integer indices



#remove the data for illegal states

columns1.remove(7) #removing column integer index 7

columns1.remove(8) #index 8

#remove the repeated columns with state names

columns1.remove(9)

columns1.remove(11)

#remove row differentiation columns

columns1.remove(4)

columns1.remove(5)

columns1.remove(6)

Legal=compareAll.iloc[:, columns1] #return all columns except the defined column

Legal
stats=Legal.corr()

stats
plt.matshow(stats)

plt.xticks(range(len(Legal.columns)), Legal.columns)

plt.yticks(range(len(Legal.columns)), Legal.columns)

plt.colorbar()

plt.show()
s=Legal['Sales']

t=Legal['Revenue after taxes']

plt.scatter(s, t, color='green')

plt.show()
#label Encode 

# Import label encoder 

from sklearn import preprocessing 

  

# encode objects

label_encoder = preprocessing.LabelEncoder() 

  

# Encode labels of specific columns

Legal['State']= label_encoder.fit_transform(Legal['State']) 

Legal['Industry']= label_encoder.fit_transform(Legal['Industry']) 

Legal
sns.pairplot(Legal, hue="State")
Legal.plot.bar()
n=Legal[['Sales','Tax Revenue','Revenue after taxes', 'Other industry revenue']]

n
##Normalize dataframe

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(n)

normalzd = pd.DataFrame(np_scaled)

normalzd
n.plot.bar()
sns.pairplot(n, hue="Tax Revenue")
Legal
Legal['Tax Revenue'].plot.bar()
# merge dataframes: other + df

inds=pd.merge(df, other, on='State', how='left')

inds
stateCrime