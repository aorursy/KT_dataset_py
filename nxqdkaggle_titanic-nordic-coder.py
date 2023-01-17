# print

print('Hello world')



# Exercise



# 1. Print the head, tail, sample data



# 2. Print the most expensive data company name



# 3. Print all toyota car details



# 4. Print the number of car models per company



# 5. Sort all car by mileage and horse power



# 6. Average mileage of each car company
import pandas as pd

import numpy as np 



df = pd.read_csv('../input/nxqdautomobile/Automobile_data.csv')

df.head(5)
most_expensive = df [['company','price']][df.price==df['price'].max()]

most_expensive
manufacturers = df.groupby('company')

manufacturers.head()

toyotaDf = manufacturers.get_group('toyota')

toyotaDf
df['company'].value_counts()
carsDf = df.sort_values(by=['price', 'horsepower'], ascending=False)

carsDf.head(5)
manufacturers = df.groupby('company')

mileageDf = manufacturers['company','average-mileage'].mean()

mileageDf
import matplotlib.pyplot as plt

plt.plot([1,2,3,4])

plt.ylabel("Y label")

plt.show()
plt.plot([1, 2, 3, 4], [1, 5, 9, 16])

data = {'a': np.arange(50),

        'c': np.random.randint(0, 50, 50),

        'd': np.random.randn(50)}

data['b'] = data['a'] + 10 * np.random.randn(50)

data['d'] = np.abs(data['d']) * 100

print(data)



plt.scatter('a', 'b', c='c', s='d', data=data)

plt.xlabel('entry a')

plt.ylabel('entry b')

plt.show()
# Loading Numpy and Pandas Libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
training_data = pd.read_csv('../input/train.csv')

training_data.head(5)
'''

    Printing the 5 first samples in training_data dataframe 

'''

training_data.head(5)
'''

    Printing the 6 samples select randomly in training_data dataframe 

'''

training_data.sample(6)
training_data.columns
training_data.dtypes
%matplotlib inline

'''

    Creating dataframes separating survived and not survived passergers

'''

td_not_survived=training_data.loc[(training_data['Survived']==0)]

td_survived=training_data.loc[(training_data['Survived']==1)]
td_not_survived.head(5)
td_survived.sample(10)
df = training_data.groupby(['Sex','Survived']).size()

df=df.unstack()

df.head()

plt.figure();df.plot(kind='bar').set_title('Gender histogram training data')
df = td_survived.groupby('Sex').size()

#df=df.unstack()

df.head()
plt.figure();df.plot(kind='bar').set_title('Survived passengers by gender');
df = td_not_survived.groupby('Sex').size()

plt.figure();df.plot(kind='bar').set_title(' Not Survived passengers by gender');
df = td_survived.groupby('Pclass').size()

plt.figure();df.plot(kind='bar').set_title('Survived passengers by Pclass');
df = td_not_survived.groupby('Pclass').size()

plt.figure();df.plot(kind='bar').set_title('Not Survived passengers by Pclass');
plt.figure();

td_survived.Age.hist()