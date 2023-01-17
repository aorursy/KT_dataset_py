import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
my_filepath = '../input/top250-00-19.csv'

my_data = pd.read_csv(my_filepath)
# Print the first five rows of the data

my_data.head()
plt.figure(figsize = (16, 8))

sns.barplot(y = 'Season', x = 'Transfer_fee', data = my_data)
sns.lineplot(data = my_data, x = 'Season', y = 'Transfer_fee')
sns.jointplot(data = my_data, x = 'Age', y = 'Transfer_fee')
sns.distplot(a = my_data['Age'])
plt.figure(figsize = (16, 12))

sns.boxplot(data = my_data, y = 'Season', x = 'Transfer_fee')