# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

train.head()
train['Pclass'].value_counts().sort_index().plot(kind='bar', title='Counts of passengers in each class')

# The 3rd class are the major class in our dataset
train['Age'].plot(kind='hist', title='Age distribution')
ages = [0, 14, 30, 60, 80] # My choice for bins was arbitrary 

bins = ['Children', 'Youth', 'Adults', 'Elderly']

train['age_bin'] = pd.cut(train.Age, ages, labels=bins)
# Now let's check the distribution

train['age_bin'].value_counts().plot(kind='barh', title='Passengers by Age bins')
train['Sex'].value_counts().plot(kind='barh', title='Passengers by Sex')
fig, axes = plt.subplots(1, 2, figsize=(16,5))

train['SibSp'].value_counts().plot(kind='bar', ax=axes[0], title='Siblings/Spouse')

train['Parch'].value_counts().plot(kind='bar', title='Parents/Children', ax=axes[1])
is_married = (train.Age > 16) & (train.SibSp == 1) # This feature maybe noisy as it some passengers

                                                  # with siblings may slip in, and also maybe some

                                                  # married passengers have siblings aboard too

                                                  # so it needs to be tuned while training the model

is_kid = train.Age < 14

is_alone = (train.SibSp == 0) & (train.Parch == 0)



train['is_married'] = np.where(is_married, 1, 0)

train['parents_aboard'] = np.where(is_kid, train.Parch, 0)

train['kids_aboard'] = np.where(is_married, train.Parch, 0)

train['family_aboard'] = train['SibSp'] + train['Parch']

train['sibs_aboard'] = np.where(is_kid, train.SibSp, 0)

train['is_alone'] = np.where(is_alone, 1, 0)
# which age group had more chances of survival?

sns.catplot(x='age_bin', y='count', hue='Survived', data=train.groupby([

                                                            'age_bin', 'Survived'])[

                                                            'PassengerId'].count().reset_index().rename(

                                                            columns={'PassengerId':'count'}), kind='bar')



# The children were the group with higher chances of survival. Adults also had better chances compared to

# youth, but I guess we can see better if we take sex into consideration
sns.catplot(x='age_bin', y='count', hue='Survived', col='Sex', data=train.groupby([

                                                            'age_bin', 'Survived', 'Sex'])[

                                                            'PassengerId'].count().reset_index().rename(

                                                            columns={'PassengerId':'count'}), kind='bar')



# Now these two plot tell a very different stroy. Youth and adults in males only had lower chances of 

# survival, on the other hand females fared better in all ages, and that makes sence, as the rule in

# these situations is to save women and children first.

# Now we answered the first two questions, where we found out that females had much higher chances of

# Survival compared to males, and all male age groups had pretty low chances of survival.
# Does different classes have different chances of survival between different age groups and sexes?

# Will our previous inferences about sex and age survival hold when we segregate the classes?



sns.catplot(x='age_bin', y='count', hue='Survived', col='Sex', row='Pclass', data=train.groupby([

                                                            'age_bin', 'Survived', 'Sex', 'Pclass'])[

                                                            'PassengerId'].count().reset_index().rename(

                                                            columns={'PassengerId':'count'}), kind='bar',

                                                            sharex=False)

# How many passengers in each class were married? and how did that affect their survival?

sns.catplot(x='is_married', y='count', hue='Survived', col='Pclass', data=train.groupby([

                                                            'is_married', 'Survived', 'Pclass'])[

                                                            'PassengerId'].count().reset_index().rename(

                                                            columns={'PassengerId':'count'}), kind='bar',

                                                            sharex=False)



# Married people are more present in the 1st and 2nd classes, and married people in the 3rd class had lower

# chances of survival compared to other classes.
# Is a child's survival affected by the presence or absence of this parents?

sns.catplot(x='parents_aboard', y='count', hue='Survived', data=train[is_kid].groupby([

                                                            'parents_aboard', 'Survived'])[

                                                            'PassengerId'].count().reset_index().rename(

                                                            columns={'PassengerId':'count'}), kind='bar',

                                                            sharex=False)



# kids with only one parent had higher chances of survival, but I really can't make sense of why that

# may happen. Let's look at this plot segregatted by classes
sns.catplot(x='parents_aboard', y='count', hue='Survived', col='Pclass', data=train[is_kid].groupby([

                                                            'parents_aboard', 'Survived', 'Pclass'])[

                                                            'PassengerId'].count().reset_index().rename(

                                                            columns={'PassengerId':'count'}), kind='bar',

                                                            sharex=False)



# Now it kind of makes sense. In the 1st class there aren't much kids to begin with. In the 2nd class, most

# kids have only 1 parent, and even those with 2 parents also survived. In the 3rd class, some kids didn't

# have their parents with them, and they had higher chances of survival, while kids with 1 or 2 parents 

# had lower chances. It might be that kids that had parents refused to leave their parents.
# How many passengers in each class were totally alone? and how did that fare for their survival?

sns.catplot(x='is_alone', y='count', hue='Survived', col='Pclass', data=train.groupby([

                                                            'is_alone', 'Survived', 'Pclass'])[

                                                            'PassengerId'].count().reset_index().rename(

                                                            columns={'PassengerId':'count'}), kind='bar',

                                                            sharex=False)



# Being alone in the 3rd class significantly reduces a passengers chances of survival. And also we can see

# that most passengers in the 3rd class were alone. I guess they were also youth, which makes sense why

# they didn't have any family aboard, just like Jack in the movie.
# Were parents more likely to survive with their kids if they weren't in the 3rd class?

# My answer is yes before we even plot, but let's see

sns.catplot(x='kids_aboard', y='count', hue='Survived', col='Pclass', data=train[(is_married) & 

                                                            (train.kids_aboard > 0)].groupby([

                                                            'kids_aboard', 'Survived', 'Pclass'])[

                                                            'PassengerId'].count().reset_index().rename(

                                                            columns={'PassengerId':'count'}), kind='bar',

                                                            sharex=False)



# We can infer from this that parents in the 1st class also had higher chances of survival compared to the

# 2nd and 3rs class parents.