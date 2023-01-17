# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Import the csv file into a variable called dataset

dataset = pd.read_csv('../input/BlackFriday.csv')



#Check few sample data using the  head command. The head command can be useful in understanding the data in each column 

dataset.head()
dataset.describe()
dataset.info()
#Checking for columns with unique values

for col in dataset.columns:

      print('Column {} contains {} unique values'.format(col,dataset[col].nunique()))
#Identifying if column has any null value or not

dataset.isna().any()
#Male v/S Female

explode = (0.1,0)

colors=['Blue','Green']

plt.pie(dataset['Gender'].value_counts(),explode=explode,colors=colors,autopct='%1.1f%%',shadow=True,startangle=140)

plt.title('Male v/s Female Ratio')

plt.legend(labels=['Male','Female'])

plt.tight_layout()

plt.show()
yval=[100000,200000,300000,400000]

plt.figure(figsize=[10,12])

dataset.groupby(['Age','Stay_In_Current_City_Years']).Product_Category_1.sum().plot(kind='bar')

plt.title('Age,Stay_in_city V/S Product Category 1')

plt.yticks(yval)

plt.figure(figsize=[10,10])

plt.title('Age,Stay_in_City V/S Product Category2')

dataset.groupby(['Age','Stay_In_Current_City_Years']).Product_Category_2.sum().plot(kind='bar')

plt.show()
explode=(0.1,0,0)

colors=['Red','Blue','Green']

plt.pie(dataset.groupby('City_Category')['Purchase'].sum(),explode=explode,colors=colors,labels=dataset['City_Category'].unique(),autopct='%1.1f%%',

        shadow=True,startangle=240)

plt.title('City Wise Purchase percentage')

plt.legend()

plt.show()
plt.figure(figsize=(20,20))

dataset.groupby(['Occupation','City_Category']).Purchase.sum().plot(kind='bar')

plt.legend()
plt.figure(figsize=(10,10))

dataset.groupby(['Gender','City_Category','Age']).Purchase.sum().plot(kind='bar')

plt.title('Gender,City,Age wise Purchase comparison')

plt.xlabel('Gender,City,Age')

plt.ylabel('Purchase')

sns.countplot(dataset['Age'],hue=dataset['Gender'])