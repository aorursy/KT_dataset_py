import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
data = pd.read_csv(f'../input/AppleStore.csv')
data.isnull().sum()
data.head()
data.info()
data.describe()
data.corr()
f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
x = data['user_rating'] == 5
data[x].head(5)
y = data[np.logical_and(data['price']==0, data['user_rating']==5 )]
y.head(10)
for i in data.prime_genre.unique():
    print('Application Category: ', i)
data.user_rating.plot(kind='hist', bins = 5, figsize=(12,6), color = 'green')
plt.xlabel('User Rating')
plt.ylabel('Frequency')
plt.title('Frequency of User Rating')
plt.show()