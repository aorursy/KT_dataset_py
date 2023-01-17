# import libraries

%matplotlib inline

import pandas as pd

from matplotlib import pyplot as plt

import matplotlib

import warnings

import seaborn as sns

from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")
# Get the data

data = pd.read_csv('../input/train.csv')
# First look

data.head()
data.info()
# Add 'child' value in 'Sex' variable 

data['Sex'] = data.apply(lambda row: 'child' if row['Age'] < 18 else row['Sex'], axis=1)
# Retrieve entries with NaN value in 'Age' variable.

null_age = data[data.isnull().Age]



# Retrieve entries with no NaN value in 'Age' variable.

notnull_age = data[data.notnull().Age]



# Get their sizes

null_age_len = null_age.shape[0]

notnull_age_len = notnull_age.shape[0]

total = null_age_len + notnull_age_len
# Print values

print(f"The variable 'Age' has {null_age_len} missing values")

print(f"The variable 'Age' has {notnull_age_len} non missing values")

print(f"The total of missing + non missing values is equal to the total number of entries? {total == data.shape[0]}")
# Glance at the entries with NaN value in 'Age' variable

null_age.head()
# what is the age distribution?

fig, axs = plt.subplots(ncols=4, sharey=True, figsize=(10, 5))

sns.distplot(notnull_age['Age'], hist_kws={'color': 'Teal'},bins=10, kde_kws={'color': 'green'}, ax=axs[0], label='All');

axs[0].set(title='All Age Distribution')

axs[0].set(ylabel='Probability Density')

sns.distplot(notnull_age[notnull_age['Sex']== 'male']['Age'], bins=10,hist_kws={'color': 'yellow'}, kde_kws={'color': 'blue'}, ax=axs[1], label='male' );

axs[1].set(title='Male Age Distribution')

sns.distplot(notnull_age[notnull_age['Sex']== 'female']['Age'],bins=10, hist_kws={'color': 'red'}, kde_kws={'color': 'Navy'}, ax=axs[2], label='female');

axs[2].set(title='Female Age Distribution')

sns.distplot(notnull_age[notnull_age['Sex']== 'child']['Age'], bins=10,hist_kws={'color': 'red'}, kde_kws={'color': 'Navy'}, ax=axs[3], label='child');

axs[3].set(title='Children Age Distribution')

plt.show()
## fixing missing Ages.

### take the median age in each gender category

median_male = np.median(notnull_age[notnull_age['Sex']== 'male']['Age'])

median_female = np.median(notnull_age[notnull_age['Sex']== 'female']['Age'])

median_children = np.median(notnull_age[notnull_age['Sex']== 'child']['Age'])

print(f'The median male adult age in our dataset is {median_male}')

print(f'The median female adult age in our dataset is {median_female}')

print(f'The median child age in our dataset is {median_children}')



### replace missing value in 'Age' with their median

data['Age'] = data.apply(lambda row: median_male 

                                if (np.isnan(row['Age']) & (row['Sex'] == 'male'))

                                else median_female

                                if (np.isnan(row['Age']) & (row['Sex'] == 'female'))

                                else median_child

                                if (np.isnan(row['Age']) & (row['Sex'] == 'female'))

                                else row['Age'], axis=1)
# check if there is any na in 'Age' variable

any(np.isnan(data['Age']))
# let's check our data

index_nan_ages = null_age.index.values

data_with_index_null = data.iloc[index_nan_ages]

data_with_index_null.head()
data_with_index_null.tail()
# Remove the 'Cabin' column

data.drop('Cabin', axis=1, inplace=True)



# Remove entries with NaN value in 'Embarked' 

data.dropna(axis=0, inplace=True)
# Check if we have any missing values in our data

data.info()
# look at the data again

data.head()
# Survived passengers

survived = Counter(data['Survived'])

print(f'The number of passengers who survived is {survived[1]}')

print(f'The number of passengers who did not survive is {survived[0]}')

print(f'The likelihood of survival of a given passenger is { np.round(survived[1]/ data.shape[0],2)}')
sns.countplot(x="Survived", data=data)

plt.title('Count of Survived vs not Survived')
male_survivors = data[data['Sex'] == 'male']['Survived']

male_survivors_count = Counter(male_survivors)

female_survivors = data[data['Sex'] == 'female']['Survived']

female_survivors_count = Counter(female_survivors)

children_survivors = data[data['Sex'] == 'child']['Survived']

children_survivors_count = Counter(children_survivors)

total_children = sum(children_survivors_count.values())

total_males = sum(male_survivors_count.values())

total_females = sum(female_survivors_count.values())



print(f'The total number of children is {total_children}')

print(f'The number of children who survived is {children_survivors_count[1]}')

print(f'The number of children who did not survive is {children_survivors_count[0]}')

print(f'The likehood of survival of a given child is {np.round(children_survivors_count[1]/data.shape[0], 3)}')

print(f'The likehood of survival of a given child within the child group is {np.round(children_survivors_count[1]/total_children, 3)} ')



print()

print(f'The total number of male passengers is {total_males}')

print(f'The number of male passengers who survived is {male_survivors_count[1]}')

print(f'The number of male passengers who did not survive is {male_survivors_count[0]}')

print(f'The likehood of survival of a given male passenger is {np.round(male_survivors_count[1]/data.shape[0], 3)}')

print(f'The likehood of survival of a given male passenger within the male group is {np.round(male_survivors_count[1]/total_males, 3)} ')



print()

print(f'The total number of female passengers is {total_females}')

print(f'The number of female passengers who survived is {female_survivors_count[1]}')

print(f'The number of female passengers who did not survive is {female_survivors_count[0]}')

print(f'The likehood of survival of a given female passenger is {np.round(female_survivors_count[1]/data.shape[0], 3)}')

print(f'The likehood of survival of a given female passenger within the female group is {np.round(female_survivors_count[1]/total_females, 3)} ')

sns.set(style='ticks', color_codes=True)

g = sns.FacetGrid(data, hue='Sex', palette="muted", size=6, sharey=False)

g.map(sns.countplot, 'Survived')

g.set_ylabels("Count")

plt.title('Number of passengers by gender category')

plt.legend()
g = sns.factorplot(x="Sex", y='Survived', data=data,

                   size=6, kind="bar", palette="muted")



g.set_ylabels("Survival Probability")

plt.title('Likelihood of survival by gender category')
classes = Counter(data['Pclass'])

for elt in classes:

    print(f'The total number of passengers in class {elt} is {classes[elt]}')

    

sns.countplot(x="Pclass", data=data)

plt.title('Pclass Count')
sns.countplot(x="Pclass", hue='Sex', data=data)

plt.title('Pclass Count')
g = sns.factorplot(x="Pclass", y='Survived', hue="Sex", data=data,

                   size=6, kind="bar", palette="muted")



g.set_ylabels("Survival Probability")

# fare by Pclass

g = sns.factorplot(x="Pclass", y ='Fare', hue="Sex", data=data,

                   size=6, kind="box", palette="muted")
g = sns.factorplot(x="Parch", y='Survived', hue="Sex", data=data,

                   size=6, kind="bar", palette="muted")



g.set_ylabels("survival probability")
g = sns.factorplot(x="SibSp", y='Survived', hue="Sex", data=data,

                   size=6, kind="bar", palette="muted")



g.set_ylabels("survival probability")

g = sns.factorplot(x="Embarked", y='Survived', hue="Sex", data=data,

                   size=6, kind="bar", palette="muted")



g.set_ylabels("survival probability")
g = sns.FacetGrid(data, col='Survived', palette="muted", size=6)

g.map(sns.boxplot, 'Sex', 'Fare', showfliers=False)

plt.legend()
# females

female_survived = data[(data['Sex'] == 'female')&(data['Survived'] == 1)]

female_not_survived = data[(data['Sex'] == 'female')&(data['Survived'] == 0)]



# males

male_survived = data[(data['Sex'] == 'male')&(data['Survived'] == 1)]

male_not_survived = data[(data['Sex'] == 'male')&(data['Survived'] == 0)]



# children

children_survived = data[(data['Sex'] == 'child')&(data['Survived'] == 1)]

children_not_survived = data[(data['Sex'] == 'child')&(data['Survived'] == 0)]



fig, axs = plt.subplots(ncols=3,nrows=2,  figsize=(10, 5), squeeze=False)

sns.distplot(male_survived['Fare'], hist_kws={'color': 'Teal'},bins=10, kde_kws={'color': 'green'}, ax=axs[0,0], label='All');

axs[0,0].set(title='Male survived Fare Distribution')

sns.distplot(female_survived['Fare'], hist_kws={'color': 'Teal'},bins=10, kde_kws={'color': 'red'}, ax=axs[0,1], label='All');

axs[0,1].set(title='Female survived Fare distribution')

sns.distplot(children_survived['Fare'], hist_kws={'color': 'Teal'},bins=10, kde_kws={'color': 'lime'}, ax=axs[0,2], label='All');

axs[0,2].set(title='children survived Fare distribution')





sns.distplot(male_not_survived['Fare'], hist_kws={'color': 'Teal'},bins=10, kde_kws={'color': 'green'}, ax=axs[1,0], label='All');

axs[1,0].set(title='Male not survived Fare Distribution')

sns.distplot(female_not_survived['Fare'], hist_kws={'color': 'Teal'},bins=10, kde_kws={'color': 'red'}, ax=axs[1,1], label='All');

axs[1,1].set(title='Female not survived Fare distribution')

sns.distplot(children_not_survived['Fare'], hist_kws={'color': 'Teal'},bins=10, kde_kws={'color': 'lime'}, ax=axs[1,2], label='All');

axs[1,2].set(title='children not survived Fare distribution')



plt.subplots_adjust(bottom=0.001, top=2, left=1, right=2)



plt.show()

## let's look how many male with fare > 100 have survived

males_higher_fare = data[(data['Fare'] > 100)&(data['Sex'] == 'male')][

    ['Age', 'Fare', 'Pclass', 'Survived']]

total_males = data[data['Sex'] == 'male'].shape[0]

male_higher_fare_count = Counter(males_higher_fare.Survived)

print(f'The total number of adult males with a fare higher than 100 is { sum( male_higher_fare_count.values())}')

print(f'The number of adult males with a fare higher than 100 who survived is {male_higher_fare_count[1]}')

print(f'The number of adult males with a fare higher than 100 who did not survived is {male_higher_fare_count[0]}')

print(f'The number of adult males with a fare more than 100 have a {np.round(male_higher_fare_count[1]*100 /total_males, 3)}% chance of survival within the male group')

male_50_100_fare  = data[((data['Fare'] <= 100) & (data['Fare'] > 50)) &

                          (data['Sex'] == 'male')][['Age', 'Fare', 'Pclass', 'Survived']]



male_50_100_fare_count = Counter(male_50_100_fare.Survived)

print(f'The total number of males with a fare between 50 and 100 is { sum( male_50_100_fare_count.values())}')

print(f'The number of adult males with a fare between 50 and 100 who survived is {male_50_100_fare_count[1]}')

print(f'The number of adult males with a fare between 50 and 100 who did not survived is {male_50_100_fare_count[0]}')

print(f'The number of adult males with a fare between 50 and 100 have a {np.round(male_50_100_fare_count[1]*100 /total_males, 3)} % chance of survival within the male group')

male_25_50_fare  = data[((data['Fare'] < 50) & (data['Fare'] > 25)) &

                          (data['Sex'] == 'male')][['Age', 'Fare', 'Pclass', 'Survived']]



male_25_50_fare_count = Counter(male_25_50_fare.Survived)

print(f'The total number of males with a fare between 25 and 50 is { sum( male_25_50_fare_count.values())}')

print(f'The number of adult males with a fare between 25 and 50 who survived is {male_25_50_fare_count[1]}')

print(f'The number of adult males with a fare between 25 and 50 who did not survived is {male_25_50_fare_count[0]}')

print(f'The number of adult males with a fare between 25 and 50 have a {np.round(male_25_50_fare_count[1]*100 /total_males, 3)} % chance of survival among males')

male_10_25_fare  = data[((data['Fare'] <= 25) & (data['Fare'] > 10)) &

                          (data['Sex'] == 'male')][['Age', 'Fare', 'Pclass', 'Survived']]



male_10_25_fare_count = Counter(male_10_25_fare.Survived)

print(f'The total number of males with a fare between 10 and 25 is { sum( male_10_25_fare_count.values())}')

print(f'The number of adult males with a fare between 10 and 25 who survived is {male_10_25_fare_count[1]}')

print(f'The number of adult males with a fare between 10 and 25 who did not survived is {male_10_25_fare_count[0]}')

print(f'The number of adult males with a fare between 10 and 25 have a {np.round(male_10_25_fare_count[1]*100 /total_males, 3)} % chance of survival within the male group')

male_0_10_fare  = data[(data['Fare'] <= 10) &

                          (data['Sex'] == 'male')][['Age', 'Fare', 'Pclass', 'Survived']]



male_0_10_fare_count = Counter(male_0_10_fare.Survived)

print(f'The total number of males with a fare less than 10 is { sum( male_0_10_fare_count.values())}')

print(f'The number of adult males with a fare less than 10 who survived is {male_0_10_fare_count[1]}')

print(f'The number of adult males with a fare less than 10 who did not survived is {male_0_10_fare_count[0]}')

print(f'The number of adult males with a fare less than 10 have a {np.round(male_0_10_fare_count[1]*100 /total_males, 3)} % chance of survival within the male group')

## TODO: Add some 1D visualization

## TODO: Calculate the significance of survival rate for different categories 

## TODO: Add Additional insights