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

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
my_data = pd.read_csv("../input/melbourne-housing-snapshot/melb_data.csv")
my_data
my_data.columns
print(my_data['Rooms'].max())
my_data['Total'] = my_data['Rooms'] + my_data['Bathroom'] + my_data['Bathroom']

my_data.Total
my_data.iloc[:10 , : ].index  
print(list(my_data.iloc[:10 , : ].index))
my_data.loc[my_data.Price > 1465000.0]
print(my_data.loc[my_data.Price > 1465000.0, ['Bathroom','Rooms']][:7]) 
print(my_data.loc[:7 , 'BuildingArea': ]) 
print(my_data.loc[:7 , ['Distance','Rooms']]) 
my_data.iloc[0]
my_Table = my_data.iloc[:9 ,[3 , 7  , 8]] 

my_Table
for col in ['Regionname' , 'YearBuilt' , 'BuildingArea' , 'Landsize'] :

    print(f" THe Index For {col} is {my_data.columns.get_loc(col)}")
my_data.iloc[15]
print(my_data.Bedroom2.max())
my_data.loc[my_data.Rooms < 3][:4]
my_data.loc[:10 , "Address"]
my_data.Distance.value_counts()
print(len(list(my_data.Distance.value_counts())))
my_data[my_data.Price < my_data.Price.mean()]
my_data[my_data.Price < my_data.Price.mean()].max()
my_data[my_data.Price < my_data.Price.mean()].max().loc['Price']
my_data.iloc[-1].loc['Address']
for row in range(len(my_data.index)) :

    if my_data.iloc[row].loc['Type'] != "h" and my_data.iloc[row].loc['SellerG'] == 'Biggin'and my_data.iloc[row].loc['Price'] in range(300000,355000):

        print(my_data.iloc[row].loc[["Address"]])
for row in range(0,3486,4) :

    if my_data.iloc[row].loc['Distance'] != 2.5 and my_data.iloc[row].loc['Distance'] > 14.7 : 

        print(my_data.iloc[row].loc[['Address','Bedroom2','Bathroom','Rooms','Price']])
my_data.loc[my_data.Rooms < 3].iloc[[8,16]]
my_data.loc[my_data.Rooms < 3].iloc[4:6]
print(my_data.Bathroom.max())
my_data.loc[my_data.Rooms < 3].loc[my_data.Bedroom2 < 12].loc[my_data.Bedroom2 == 4]
my_data.loc[my_data.Rooms < 3].loc[my_data.Bedroom2 < 12].loc[my_data.Bedroom2 == 4].Bedroom2
my_data.loc[my_data.Rooms < 3].loc[my_data.Bedroom2 < 12].loc[my_data.Bathroom == 3]
my_data.loc[my_data.Rooms < 3].loc[my_data.Bedroom2 < 12].loc[my_data.Bathroom == 3].loc[: , ['Bedroom2','Rooms','Price']]
my_data.loc[my_data.Rooms < 3].loc[my_data.Bedroom2 < 12].loc[my_data.Bathroom == 3].loc[: , ['Bedroom2','Rooms','Price']].stack()
my_data.loc[my_data.Rooms < 3].loc[my_data.Bedroom2 < 12].loc[my_data.Bathroom == 3].loc[: , ['Bedroom2','Rooms','Price']][:3]
my_data.loc[my_data.Rooms < 3].loc[my_data.Bedroom2 < 12].loc[my_data.Bathroom == 3].loc[: , ['Bedroom2','Rooms','Price']].iloc[5]
my_try = pd.DataFrame(my_data.loc[my_data.Rooms < 3].loc[my_data.Bedroom2 < 12].loc[my_data.Bedroom2 == 4][:3],

                     index = ["HM1","HM2","HM3"] )

my_try
my_array = np.array(my_data.loc[my_data.Rooms < 3].loc[my_data.Bedroom2 < 12].loc[my_data.Bedroom2 == 4][:3])

my_array
my_try = pd.DataFrame(my_array,index = ["HM1","HM2","HM3"])

my_try
my_try = pd.DataFrame(my_array,index = ["HM1","HM2","HM3"], columns= my_data.columns)

my_try
my_try.stack()[1]
my_try.stack()['HM1']
my_try.stack()['HM1'][9]
my_try.stack()['HM1'].Postcode
my_try.stack()['HM1'][21]
my_try.stack()['HM1'][[5,19]]
my_data.loc[my_data.Rooms < 3].loc[my_data.Bedroom2 < 12].loc[my_data.Bedroom2 == 4].stack().loc[6465]
my_data.YearBuilt[1]
my_data.loc[my_data.Rooms < 3].loc[my_data.Bedroom2 < 12].loc[my_data.Bedroom2 == 4][:3].stack()
my_data.loc[my_data.Rooms < 3].loc[my_data.Bedroom2 < 12].loc[my_data.Bedroom2 == 4].stack().iloc[15]
my_data.columns
# my_data.keys
my_data.keys()


my_data.values
print(my_data['Rooms'].agg(['max','min','std','mean']))
print(my_data['Price'].agg(['mean']))
print(my_data.iloc[2:8].values)
print(my_data['Price'].describe()['mean'])
print(my_data.Price.sort_values(ascending= False))
my_data.YearBuilt.plot(kind= "pie")
my_data.YearBuilt[:30].plot(kind= "pie")
list(np.random.randint(my_data.Price.mean(),size=10))
# print(list(my_data.YearBuilt))
for col in my_data.columns :

    if col in my_data.select_dtypes(exclude=['object']).columns and my_data[col].isnull().any() == False :

        print(col)
list(np.random.choice(list(my_data.Landsize),size=20)) 

x = my_data.Landsize.iloc[list(np.random.choice(list(my_data.Landsize),size=100))]

x.plot(kind="hist")
my_data.info()
X = my_data.drop(columns='Price',axis=1)
X.isna()
X.isnull()
X.isna().sum()
X.isnull().sum()
X.isna().sum().reset_index() 
X.isnull().sum().reset_index()
my_data['Price'].isna()
my_data['Price'].isna().sum()
sns.kdeplot(X.Car,Label='Car',color='g')
my_data.Car.plot(kind = 'box' ,x= my_data.Car , label= 'N_car')