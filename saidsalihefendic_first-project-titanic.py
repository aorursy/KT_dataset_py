import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



%matplotlib notebook



df = pd.read_csv('../input/train.csv')
df[:10]


len(df)
df.duplicated().value_counts()
df['PassengerId'].is_unique
survival = {0: 'Not Survived', 1: 'Survived'}



df['Survived'].value_counts()
prop_survived = df['Survived'].value_counts() / len(df)

prop_survived = prop_survived.rename(survival)

print(prop_survived)



prop_survived.plot(kind='bar', title='Survival rate')
gender_data = df['Sex']



gender_data[:10]
# We want to see whether we have NaN values for our genders

gender_data.isna().value_counts()
prop_gender = gender_data.value_counts() / len(gender_data)
print(prop_gender)
male_female_survived = df.groupby(by=['Sex', 'Survived']).size().reset_index().rename(columns={0: 'Count'})



male_female_survived
import seaborn as sns



sns.barplot(x='Sex', y='Count', hue='Survived', data=male_female_survived).set_title('Survival rate by Gender')
# now we calculate the percentage

prop_male_female_survived = male_female_survived.copy()



prop_male_female_survived['Count'] = prop_male_female_survived['Count'] / prop_male_female_survived['Count'].sum()

prop_male_female_survived.rename(columns={'Count': 'Percentage'}, inplace=True)



print(prop_male_female_survived)
male_survived = df[df['Sex'] == 'male']



len(male_survived)
male_survived['Survived'].value_counts().rename(survival).plot(kind='bar', title='Survival rate of Male Passengers')
prop_male_survive = male_survived['Survived'].value_counts().rename(survival) / len(male_survived)

prop_male_survive
female_survived = df[df['Sex'] == 'female']



len(female_survived)
female_survived['Survived'].value_counts().rename(survival).plot(kind='bar', title='Survival rate of Female Passengers')
prop_female_survive = female_survived['Survived'].value_counts().rename(survival) / len(female_survived)

prop_female_survive
# dividing into age groups

df['Age'].isna().value_counts()

df['Age'].describe()
from math import modf



# We want the integer part of the mean to take for fillna()

# We also will not change the df, so we will save it to df_modified



# For now, we will assume they are all in the mean

# Note: Include deviation part?

df_modified = df[['PassengerId', 'Survived', 'Sex', 'Age']].copy()



df_modified['Age'] = df_modified['Age'].fillna(modf(df['Age'][df['Age'].notna()].mean())[1])
df_modified[-10:]
# 0 - [0 -15)

# 1 - [15 - 25)

# 2 - [25 - 40)

# 3 - [40 - 65)

# 4 - [65 - 81)



# Age groups here are left_inclusive



# NOTE: Check to do it with pd.Categorical type



age_groups = {0: (0, 15), 1: (15, 25), 2: (25, 40), 3: (40, 65), 4: (65, 81)}



def which_age_group(x):

    for key, age_group in age_groups.items():

        if x >= age_group[0] and x < age_group[1]:

            return key



df_modified['AgeGroup'] = df_modified['Age'].apply(which_age_group).astype('int64')
df_modified[:10]
print(df_modified.groupby('AgeGroup').size().rename(age_groups))



df_modified.groupby('AgeGroup').size().rename(age_groups).plot(kind='bar', title='Number of passengers by Age Group')
# looking at the whole population



age_group_survive = df_modified.groupby(['AgeGroup', 'Survived']).size().reset_index().rename(columns={0: 'Count'})
# To get sense of which age group we are talking about, not looking at indices



age_group_survive['AgeGroup'] = age_group_survive['AgeGroup'].apply(lambda x: age_groups[x])
age_group_survive
age_group_survive['Survived'] = age_group_survive['Survived'].apply(lambda x: survival[x])
age_group_survive
sns.barplot(x='AgeGroup', y='Count', hue='Survived', data=age_group_survive).set_title('Survival rate by Age Group')
age_group_by_sex = df_modified.groupby(['AgeGroup', 'Sex', 'Survived']).size().reset_index().rename(columns={0: 'Count'})
age_group_by_sex['AgeGroup'] = age_group_by_sex['AgeGroup'].apply(lambda x: age_groups[x])
age_group_by_sex
age_group_by_sex['Survived'] = age_group_by_sex['Survived'].apply(lambda x: survival[x])
# NOTE: Check how to plot with combining two columns without the need to create new one



age_group_by_sex['AgeGroupSex'] = age_group_by_sex['AgeGroup'].apply(lambda x: str(x)) + " - " + age_group_by_sex['Sex']
age_group_by_sex
# plotting



a4_dims = (10, 9)

fig, ax = plt.subplots(2, figsize=a4_dims)



survival_by_age_group_males = sns.barplot(ax=ax[0],

                                            x='AgeGroupSex',

                                            y='Count',

                                            hue='Survived',

                                            data=age_group_by_sex[age_group_by_sex['Sex'] == 'male'])



survival_by_age_group_males.set_title('Survival rate by sex and age group - males')



survival_by_age_group_females = sns.barplot(ax=ax[1],

                                            x='AgeGroupSex',

                                            y='Count',

                                            hue='Survived',

                                            data=age_group_by_sex[age_group_by_sex['Sex'] == 'female'])



survival_by_age_group_females.set_title('Survival rate by sex and age group - females')
df.head()
df['SibSp'].describe()
df['SibSp'].value_counts()
df_siblings = df[['PassengerId', 'Survived', 'Sex', 'SibSp', 'Age']].copy()
df_siblings[:6]
df_siblings['SibSp'].notna().value_counts()
df_siblings['Age'] = df_siblings['Age'].fillna(modf(df['Age'][df['Age'].notna()].mean())[1])

df_siblings['AgeGroup'] = df_siblings['Age'].apply(which_age_group).astype('int64').apply(lambda x: age_groups[x])

df_siblings['HasSibSp'] = df_siblings['SibSp'].apply(lambda x: x > 0)
df_siblings[:6]
df_siblings['HasSibSp'].value_counts()
# using the whole population

hassib_survival_rate = df_siblings.groupby(['HasSibSp', 'Survived']).size().reset_index().rename(columns={0: 'Count'})
hassib_survival_rate
hassib_survival_rate['Survived'] = hassib_survival_rate['Survived'].apply(lambda x: survival[x])
sns.barplot(x='HasSibSp', y='Count', hue='Survived', data=hassib_survival_rate).set_title('Survival rate by having siblings or spouse')
passengers_with_siblings = df_siblings[df_siblings['HasSibSp']]

passengers_with_no_siblings = df_siblings[df_siblings['HasSibSp'] == False]
survival_rate_by_sex_and_sibsp = passengers_with_siblings.groupby(['Sex', 'Survived']).size().reset_index().rename(columns={0: 'Count'})

survival_rate_by_sex_and_nosibsp = passengers_with_no_siblings.groupby(['Sex', 'Survived']).size().reset_index().rename(columns={0: 'Count'})
survival_rate_by_sex_and_sibsp
survival_rate_by_sex_and_nosibsp
survival_rate_by_sex_and_sibsp['Survived'] = survival_rate_by_sex_and_sibsp['Survived'].apply(lambda x: survival[x])



survival_rate_by_sex_and_sibsp

# siblings_gender_plot = sns.barplot(x='Sex', y='Count', hue='Survived', data=survival_rate_by_sex_and_sibsp)



# siblings_gender_plot.set_title('Survival rate of having siblings by sex ')



# siblings_gender_plot
survival_rate_by_sex_and_nosibsp['Survived'] = survival_rate_by_sex_and_nosibsp['Survived'].apply(lambda x: survival[x])



survival_rate_by_sex_and_nosibsp
fig, ax = plt.subplots(2, figsize=a4_dims)



survival_has_sib = sns.barplot(ax=ax[0], x='Sex', y='Count', hue='Survived', data=survival_rate_by_sex_and_sibsp)

survival_has_sib.set_title('Survival rate of having siblings by sex')



survival_has_nosib = sns.barplot(ax=ax[1], x='Sex', y='Count', hue='Survived', data=survival_rate_by_sex_and_nosibsp)

survival_has_nosib.set_title('Survival rate of not having siblings by sex')
survival_rate_by_sex_and_sibsp['Count'] = survival_rate_by_sex_and_sibsp['Count'] / survival_rate_by_sex_and_sibsp['Count'].sum()
survival_rate_by_sex_and_nosibsp['Count'] = survival_rate_by_sex_and_nosibsp['Count'] / survival_rate_by_sex_and_nosibsp['Count'].sum()
survival_rate_by_sex_and_nosibsp
# NOTE: Research impact of sibling/spouse on age groups



survival_rate_by_sex_and_sibsp