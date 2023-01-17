# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#sets rows display to 1000 so you can easly explore your data.
pd.set_option('display.max_rows', 1000)

#loads PoliceKillingsUS.csv file
killed_people = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv',encoding='cp1252')
#prints out head of killed_people file, you can put into .head() how many rows do you want to print out.
print(killed_people.head())

#sums all variables in age column together.
X = killed_people.age.sum()

#replace A which stands for Asian to given work in this example A replaces for Asian and so on.
killed_people.race.replace({'A':'Asian','H':'Hispanic','B':'Black','W':'White' },inplace=True)

#prints out how many rows and columns data's got.
killed_people.shape

#describes ints for example in this case it shows id and age as these are ints. mean show average, min- minimal value in column,
#max - maximal value in column
killed_people.describe()

#function isnull and any shows us if there is missing data in out data set, is there is there will be True if there's not False.
killed_people.isnull().any()
#this function shows us where we got Non Values
killed_people.isnull().sum()

#this function is able to count each column and sum it in this example it counts each value from race column.
killed_people.race.value_counts()

#really cool - gives a percentage of each race.
killed_people.race.value_counts() / killed_people.race.notnull().sum() 

#very simple plot using age as a attribute.
#killed_people.age.plot()

#same as previous but this time it shows data on bar which might be more readable
killed_people.race.value_counts().plot(kind='bar')

###DATA MANIPULATION###

#changes names of column 'name' to 'Name'
killed_people.rename(index = str, columns ={'name':'Name'},inplace = True)

#shows first 60 rows
killed_people[0:200]

#this creates data frame with choosen columns
choosen_columns = killed_people[['race','age','city']]

killed_people[0:100]




women = train_data.loc[(train_data['Sex']=='female') & (train_data['Survived']==1)]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)
