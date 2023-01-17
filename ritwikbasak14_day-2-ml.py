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

df = pd.read_csv("../input/autompg-dataset/auto-mpg.csv", na_values='?')

df.head(5)

df.describe()
df.columns
df.isnull().any()
horsepower_median = df['horsepower'].median()



df['horsepower'].fillna(horsepower_median,inplace=True)



df['horsepower'].isnull().any()



df.boxplot(column = [df.columns[0],df.columns[1]])
df.boxplot(column = [df.columns[2],df.columns[3]])
df.boxplot(column = [df.columns[4]])
df.boxplot(column = [df.columns[5]])
df.boxplot(column = [df.columns[6]])
df.boxplot(column = [df.columns[7]])
import pandas as pd

length = [12,10,11,14,15,16]

breadth = [9,4,2,6,10,12]

type_room = ['Big','Small','Small','Medium','Big','Big']  

df = pd.DataFrame({'length':length,'breadth':breadth,'type_room':type_room})

df

#creating a new column

df['area'] = df['length'] * df['breadth']

df

city = ['CityA','CityB','CityA','CityC','CityA','CityB']

roll = [12,14,15,16,13,19]

df1 = pd.DataFrame({"city_of_origin":city, "roll":roll})

df1



df1=pd.get_dummies(df1)

df1

city = ['CityA','CityB','CityA','CityC','CityA','CityB']

roll = [12,14,15,16,13,19]

df2 = pd.DataFrame({"city_of_origin":city, "roll":roll})

df2



from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

df2['city_of_origin'] = lb.fit_transform(df2['city_of_origin'])

df2.head()


