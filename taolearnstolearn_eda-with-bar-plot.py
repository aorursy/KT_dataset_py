import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.isnull().sum().plot(kind = 'barh')
train.head()
grouped = train.Survived.groupby(train.Sex)

sex_stats = pd.DataFrame({'total number':grouped.size(), 'number of survivor':grouped.sum()})
sex_stats.plot.bar()
train.Age.plot.hist(bins = 50)
age_group = pd.cut(train.Age, bins = 8)

age_grouped = train.Survived.groupby(age_group)

age_stats = pd.DataFrame({'total number':age_grouped.size(), 'number of survivor':age_grouped.sum()})
age_stats.plot.bar()
class_grouped = train.Survived.groupby(train.Pclass)

class_stats = pd.DataFrame({'total number':class_grouped.size(), 'number of survivor':class_grouped.sum()})
class_stats.plot.bar()
train[['Pclass', 'Fare']].corr()
train.Fare.plot.hist()
train['Fare_class'] = train.Fare.apply(lambda x: 'low' if x < 100 else 'high')
fare_class_grouped = train.Survived.groupby(train.Fare_class)

f_stats = pd.DataFrame({'total number':fare_class_grouped.size(), 'number of survivor':fare_class_grouped.sum()})
f_stats.plot.bar()
train['n_relatives'] = train.SibSp + train.Parch
n_map = {0:'None', 1:'Fair', 2:'Fair', 3:'Fair'}
train['relatives'] = train.n_relatives.apply(lambda x:n_map.get(x, 'Many'))
n_relatives_grouped = train.Survived.groupby(train.relatives)

r_stats = pd.DataFrame({'total number':n_relatives_grouped.size(), 'number of survivor':n_relatives_grouped.sum()})
r_stats.plot.bar()
train['parent_or_children_onboard'] = train.Parch.apply(lambda x: 'Yes' if x > 0 else 'No')
n_pc_grouped = train.Survived.groupby(train.parent_or_children_onboard)

r_stats = pd.DataFrame({'total number':n_pc_grouped.size(), 'number of survivor':n_pc_grouped.sum()})
r_stats.plot.bar()