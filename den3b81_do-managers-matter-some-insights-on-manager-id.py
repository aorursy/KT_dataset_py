# let's load the usual packages first

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline
train_df = pd.read_json('../input/train.json')

test_df = pd.read_json('../input/test.json')
man_train_list = train_df.manager_id.unique()

man_test_list = test_df.manager_id.unique()

print("Train: {0}".format(len(man_train_list)))

print("Test: {0}".format(len(man_test_list)))
temp1 = train_df.groupby('manager_id').count().iloc[:,-1]

temp2 = test_df.groupby('manager_id').count().iloc[:,-1]

df_managers = pd.concat([temp1,temp2], axis = 1, join = 'outer')

df_managers.columns = ['train_count','test_count']

print(df_managers.head(20))

print(df_managers.sort_values(by = 'train_count', ascending = False).head(10))
fig, axes = plt.subplots(1,2, figsize = (12,5))

temp = df_managers['train_count'].dropna().sort_values(ascending = False).reset_index(drop = True)

axes[0].plot(temp.index+1, temp.cumsum()/temp.sum())

axes[0].set_title('cumulative train_count')



temp = df_managers['test_count'].dropna().sort_values(ascending = False).reset_index(drop = True)

axes[1].plot(temp.index+1, temp.cumsum()/temp.sum())

axes[1].set_title('cumulative test_count')
ix20 = int(len(df_managers['train_count'].dropna())*0.2)

print("TRAIN: 20% of managers ({0}) responsible for {1:2.2f}% of entries".format(ix20,df_managers['train_count'].sort_values(ascending = False).cumsum().iloc[ix20]/df_managers['train_count'].sum()*100))



ix20 = int(len(df_managers['test_count'].dropna())*0.2)

print("TEST: 20% of managers ({0}) responsible for {1:2.2f}% of entries".format(ix20, df_managers['test_count'].sort_values(ascending = False).cumsum().iloc[ix20]/df_managers['test_count'].sum()*100))
man_not_in_test = set(man_train_list) - set(man_test_list)

man_not_in_train = set(man_test_list) - set(man_train_list)



print("{} managers are featured in train.json but not in test.json".format(len(man_not_in_test)))

print("{} managers are featured in test.json but not in train.json".format(len(man_not_in_train)))
print(df_managers.loc[list(man_not_in_test)]['train_count'].describe())

print(df_managers.loc[list(man_not_in_train)]['test_count'].describe())
df_managers.sort_values(by = 'train_count', ascending = False).head(1000).corr()
df_managers.sort_values(by = 'train_count', ascending = False).head(100).plot.scatter(x = 'train_count', y = 'test_count')
temp = df_managers['train_count'].sort_values(ascending = False).head(100)

temp = pd.concat([temp,temp.cumsum()/df_managers['train_count'].sum()*100], axis = 1).reset_index()

temp.columns = ['manager_id','count','percentage']

print(temp)
man_list = df_managers['train_count'].sort_values(ascending = False).head(100).index

ixes = train_df.manager_id.isin(man_list)

df100 = train_df[ixes][['manager_id','interest_level']]

interest_dummies = pd.get_dummies(df100.interest_level)

df100 = pd.concat([df100,interest_dummies[['low','medium','high']]], axis = 1).drop('interest_level', axis = 1)



print("The top100 contributors account for {} entries\n".format(len(df100)))



print(df100.head(10))
import itertools



# 50 most common surnames in the 90s (http://surnames.behindthename.com/top/lists/united-states/1990)

last_names = ['Smith', 'Johnson', 'Williams', 'Jones', 'Brown', 'Davis', 'Miller', 'Wilson', 'Moore', 

 'Taylor', 'Anderson', 'Thomas', 'Jackson', 'White', 'Harris', 'Martin', 'Thompson', 'Garcia', 

 'Martinez', 'Robinson', 'Clark', 'Rodriguez', 'Lewis', 'Lee', 'Walker', 'Hall', 'Allen', 'Young',

 'Hernandez', 'King', 'Wright', 'Lopez', 'Hill', 'Scott', 'Green', 'Adams', 'Baker', 'Gonzalez', 'Nelson', 

 'Carter', 'Mitchell', 'Perez', 'Roberts', 'Turner', 'Phillips', 'Campbell', 'Parker', 'Evans', 'Edwards', 'Collins']



# 10 most common first names for females and males (names.mongabay.com) 

first_names = ['Mary',  'Patricia',  'Linda',  'Barbara',  'Elizabeth',  

               'Jennifer',  'Maria',  'Susan',  'Margaret',  'Dorothy',

               'James', 'John', 'Robert', 'Michael', 'William', 'David',

               'Richard', 'Charles', 'Joseph', 'Thomas']



names = [first + ' ' + last for first,last in (itertools.product(first_names, last_names))]



# shuffle them

np.random.seed(12345)

np.random.shuffle(names)



dictionary = dict(zip(man_list, names))

df100.loc[df100.manager_id.isin(dictionary), 'manager_id' ] = df100['manager_id'].map(dictionary)

print(df100.head())
# see if the name coincides

print(names[:10])

print(df100.groupby('manager_id').count().sort_values(by = 'low', ascending = False).head(10))
gby = pd.concat([df100.groupby('manager_id').mean(),df100.groupby('manager_id').count()], axis = 1).iloc[:,:-2]

gby.columns = ['low','medium','high','count']

gby.sort_values(by = 'count', ascending = False).head(10)
gby.sort_values(by = 'count', ascending = False).drop('count', axis = 1).plot(kind = 'bar', stacked = True, figsize = (15,5))

plt.figure()

gby.sort_values(by = 'count', ascending = False)['count'].plot(kind = 'bar', figsize = (15,5))
gby['skill'] = gby['medium']*1 + gby['high']*2 



print("Top performers")

print(gby.sort_values(by = 'skill', ascending = False).reset_index().head())

print("\nWorst performers")

print(gby.sort_values(by = 'skill', ascending = False).reset_index().tail())
gby.skill.plot(kind = 'hist')

print(gby.mean())