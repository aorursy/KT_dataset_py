%matplotlib inline
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
df = pd.read_csv('../input/act_train.csv')
peoples = df['people_id']
print('Number of active people: {}'.format(peoples.nunique()))
threshold = 400
people_counts = peoples.value_counts()
people_counts[people_counts > threshold].plot(kind='bar')
plt.title('People with more than {} activities'.format(threshold))
plt.ylabel('Activity count')
fig = plt.gcf()
fig.set_size_inches(16, 7)
people_counts[people_counts <= threshold].hist(bins=int(threshold / 10))
plt.xlabel('Activity count')
plt.ylabel('People count')
plt.title('Distribution of peoples with less than {} activities'.format(threshold))
fig = plt.gcf()
fig.set_size_inches(16, 7)