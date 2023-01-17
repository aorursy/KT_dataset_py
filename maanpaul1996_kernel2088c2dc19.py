# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from matplotlib import pyplot as plt

import seaborn as sns



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#read the train file

path = "/kaggle/input/titanic/"

data = pd.read_csv(path+"train.csv", header=0)
#check the columns in our dataframe

data.columns
#set teh white background in our plot

sns.set(style="white", rc={"lines.linewidth": 3})
#fill null valule in age column

data.Age.fillna(method = 'ffill', inplace=True)
#add new column Lived or Died

data['Lived'] = data.Survived.apply(lambda x : 1 if x == 1 else 0)

data['Died'] = data.Survived.apply(lambda x : 1 if x == 0 else 0)
#gender against survival

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,6))

data.groupby('Sex').sum()[['Lived', 'Died']].plot(ax=axes[0] ,kind='bar', stacked=True)

data.groupby('Sex').mean()[['Lived', 'Died']].plot(ax=axes[1], kind='bar', stacked=True)
#class against survival



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,6))

data.groupby('Pclass').agg('sum')[['Lived', 'Died']].plot(ax=axes[0], kind='bar', stacked=True)

data.groupby('Pclass').agg('mean')[['Lived', 'Died']].plot(ax=axes[1], kind='bar', stacked=True)
data['Category'] = data.Age.apply(lambda x : 'Child' if x < 16 else 'Adult' if x < 60 else 'Senior Citizen')
#survival rate against Category



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,6))



data.groupby('Category').sum()[['Lived', 'Died']].plot(ax=axes[0], kind='bar', stacked=True)

data.groupby('Category').mean()[['Lived', 'Died']].plot(ax=axes[1], kind='bar', stacked=True)
#age distribution



age_hist = pd.DataFrame()



age_hist['Age'] = pd.Series(range(0, 100, 10))

age_hist['Count'] = age_hist.Age.apply(lambda x : data[(data.Age < x) & (data.Age > x - 10)].shape[0])

age_hist['Lived'] = age_hist.Age.apply(lambda x : data[(data.Age < x) & (data.Age > x - 10) & (data.Lived == 1)].shape[0])

ax = age_hist.Count.plot(alpha = .75, kind='bar')

ax2 = ax.twinx()

ax2.plot(ax.get_xticks(), age_hist.Lived, alpha = .75, color = 'r')
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20,6))

sns.scatterplot(x='Age', y='Fare', data = data, ax=ax[0])

sns.scatterplot(x='Age', y='Fare', hue='Pclass', data = data, ax=ax[1])

sns.scatterplot(x='Age', y='Fare', hue='Survived', data = data, ax=ax[2])

g = sns.FacetGrid(data, col='Pclass', row='Survived')

g = g.map(plt.hist, 'Age')
spc_data = data.groupby(['Sex','Pclass','Category']).sum()[['Lived','Died']].reset_index()

spc_plot = sns.FacetGrid(hue='Category', row='Sex', col='Pclass', data=data)

spn_plot = spc_plot.map(plt.hist, 'Survived')

#spc_plot.map(plt.scatter, 'Survived')

spc_plot.add_legend()
train_feature, train_label = data[['Age', 'Fare', 'Sex', 'Category']], data.Survived

train_feature.Sex = train_feature.Sex.apply(lambda x : 0 if x == 'male' else 1)

train_feature.Category = train_feature.Category.apply(lambda x : 0 if x == 'Child' else 1 if x == 'Adult' else 2)
survival_model = RandomForestClassifier(n_estimators=10)

survival_model.fit(train_feature, train_label)
#read the train data



test_feature = pd.read_csv(path+"test.csv", header=0)

test_feature.Age.fillna(method = 'ffill', inplace=True)

test_feature.Fare.fillna(method = 'ffill', inplace=True)



gender_summission = pd.DataFrame()

gender_summission['PassengerId'] = test_feature.PassengerId



test_feature['Category'] = test_feature.Age.apply(lambda x : 'Child' if x < 16 else 'Adult' if x < 60 else 'Senior Citizen')



test_feature = test_feature[['Age', 'Fare', 'Sex', 'Category']]



test_feature.Sex = test_feature.Sex.apply(lambda x : 0 if x == 'male' else 1)

test_feature.Category = test_feature.Category.apply(lambda x : 0 if x == 'Child' else 1 if x == 'Adult' else 2)

gender_summission['Survived'] = survival_model.predict(test_feature)
gender_summission.to_csv('gender_submission.csv', index=False)