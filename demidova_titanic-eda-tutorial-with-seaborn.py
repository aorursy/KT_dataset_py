import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np
# path to train dataset

train_path = '../input/titanic/train.csv'

# path to test dataset

test_path = '../input/titanic/test.csv'



# Read a comma-separated values (csv) file into pandas DataFrame

train_data = pd.read_csv(train_path)

test_data = pd.read_csv(test_path)



# shape of tha data

print('Train shape: ', train_data.shape)

print('Test shape: ', test_data.shape)
# create a sequence of DataFrame objects

frames = [train_data, test_data]

# Concatenate pandas objects along a particular axis 

all_data = pd.concat(frames, sort = False)

# shape of the data

print('All data shape: ', all_data.shape)

# Show first 4 rows of the concatenated DataFrame

all_data.head(4)
all_data.info()
# check data for NA values

all_data_NA = all_data.isna().sum()

train_NA = train_data.isna().sum()

test_NA = test_data.isna().sum()



pd.concat([train_NA, test_NA, all_data_NA], axis=1, sort = False, keys = ['Train NA', 'Test NA', 'All NA'])
# set size of the plot

plt.figure(figsize=(6, 4.5)) 



# countplot shows the counts of observations in each categorical bin using bars.

# x - name of the categorical variable

ax = sns.countplot(x = 'Survived', data = all_data, palette=["#3f3e6fd1", "#85c6a9"])



# set the current tick locations and labels of the x-axis.

plt.xticks( np.arange(2), ['drowned', 'survived'] )

# set title

plt.title('Overall survival (training dataset)',fontsize= 14)

# set x label

plt.xlabel('Passenger status after the tragedy')

# set y label

plt.ylabel('Number of passengers')



# calculate passengers for each category

labels = (all_data['Survived'].value_counts())

# add result numbers on barchart

for i, v in enumerate(labels):

    ax.text(i, v-40, str(v), horizontalalignment = 'center', size = 14, color = 'w', fontweight = 'bold')

    

plt.show()
all_data['Survived'].value_counts(normalize = True)
# set plot size

plt.figure(figsize=(15, 3))



# plot a univariate distribution of Age observations 

sns.distplot(all_data[(all_data["Age"] > 0)].Age, kde_kws={"lw": 3}, bins = 50)



# set titles and labels

plt.title('Distrubution of passengers age (all data)',fontsize= 14)

plt.xlabel('Age')

plt.ylabel('Frequency')

# clean layout

plt.tight_layout()
# Descriptive statistics include those that summarize the central tendency, 

# dispersion and shape of a dataset’s distribution, excluding NaN values.

age_distr = pd.DataFrame(all_data['Age'].describe())

# Transpose index and columns.

age_distr.transpose()
plt.figure(figsize=(15, 3))



# Draw a box plot to show Age distributions with respect to survival status.

sns.boxplot(y = 'Survived', x = 'Age', data = train_data,

     palette=["#3f3e6fd1", "#85c6a9"], fliersize = 0, orient = 'h')



# Add a scatterplot for each category.

sns.stripplot(y = 'Survived', x = 'Age', data = train_data,

     linewidth = 0.6, palette=["#3f3e6fd1", "#85c6a9"], orient = 'h')



plt.yticks( np.arange(2), ['drowned', 'survived'])

plt.title('Age distribution grouped by surviving status (train data)',fontsize= 14)

plt.ylabel('Passenger status after the tragedy')

plt.tight_layout()
# Descriptive statistics:

pd.DataFrame(all_data.groupby('Survived')['Age'].describe())
all_data[all_data['Age'] == max(all_data['Age'] )]
train_data.loc[train_data['PassengerId'] == 631, 'Age'] = 48

all_data.loc[all_data['PassengerId'] == 631, 'Age'] = 48
# Descriptive statistics:

pd.DataFrame(all_data.groupby('Survived')['Age'].describe())
# set size

plt.figure(figsize=(20, 6))



# set palette

palette = sns.cubehelix_palette(5, start = 3)



plt.subplot(1, 2, 1)

sns.boxplot(x = 'Pclass', y = 'Age', data = all_data,

     palette = palette, fliersize = 0)



sns.stripplot(x = 'Pclass', y = 'Age', data = all_data,

     linewidth = 0.6, palette = palette)

plt.xticks( np.arange(3), ['1st class', '2nd class', '3rd class'])

plt.title('Age distribution grouped by ticket class (all data)',fontsize= 16)

plt.xlabel('Ticket class')





plt.subplot(1, 2, 2)



# To use kdeplot I need to create variables with filtered data for each category

age_1_class = all_data[(all_data["Age"] > 0) & 

                              (all_data["Pclass"] == 1)]

age_2_class = all_data[(all_data["Age"] > 0) & 

                              (all_data["Pclass"] == 2)]

age_3_class = all_data[(all_data["Age"] > 0) & 

                              (all_data["Pclass"] == 3)]



# Ploting the 3 variables that we create

sns.kdeplot(age_1_class["Age"], shade=True, color='#eed4d0', label = '1st class')

sns.kdeplot(age_2_class["Age"], shade=True,  color='#cda0aa', label = '2nd class')

sns.kdeplot(age_3_class["Age"], shade=True,color='#a2708e', label = '3rd class')

plt.title('Age distribution grouped by ticket class (all data)',fontsize= 16)

plt.xlabel('Age')

plt.xlim(0, 90)

plt.tight_layout()

plt.show()
# Descriptive statistics:

pd.DataFrame(all_data.groupby('Pclass')['Age'].describe())
plt.figure(figsize=(20, 5))

palette = "Set3"



plt.subplot(1, 3, 1)

sns.boxplot(x = 'Sex', y = 'Age', data = age_1_class,

     palette = palette, fliersize = 0)

sns.stripplot(x = 'Sex', y = 'Age', data = age_1_class,

     linewidth = 0.6, palette = palette)

plt.title('1st class Age distribution by Sex',fontsize= 14)

plt.ylim(-5, 80)



plt.subplot(1, 3, 2)

sns.boxplot(x = 'Sex', y = 'Age', data = age_2_class,

     palette = palette, fliersize = 0)

sns.stripplot(x = 'Sex', y = 'Age', data = age_2_class,

     linewidth = 0.6, palette = palette)

plt.title('2nd class Age distribution by Sex',fontsize= 14)

plt.ylim(-5, 80)



plt.subplot(1, 3, 3)

sns.boxplot(x = 'Sex', y = 'Age',  data = age_3_class,

     order = ['female', 'male'], palette = palette, fliersize = 0)

sns.stripplot(x = 'Sex', y = 'Age', data = age_3_class,

     order = ['female', 'male'], linewidth = 0.6, palette = palette)

plt.title('3rd class Age distribution by Sex',fontsize= 14)

plt.ylim(-5, 80)



plt.show()
# Descriptive statistics:

age_1_class_stat = pd.DataFrame(age_1_class.groupby('Sex')['Age'].describe())

age_2_class_stat = pd.DataFrame(age_2_class.groupby('Sex')['Age'].describe())

age_3_class_stat = pd.DataFrame(age_3_class.groupby('Sex')['Age'].describe())



pd.concat([age_1_class_stat, age_2_class_stat, age_3_class_stat], axis=0, sort = False, keys = ['1st', '2nd', '3rd'])
all_data['Title'] = all_data['Name'].str.split(',', expand = True)[1].str.split('.', expand = True)[0].str.strip(' ')



plt.figure(figsize=(6, 5))

ax = sns.countplot( x = 'Title', data = all_data, palette = "hls", order = all_data['Title'].value_counts().index)

_ = plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light'  

)



plt.title('Passengers distribution by titles',fontsize= 14)

plt.ylabel('Number of passengers')



# calculate passengers for each category

labels = (all_data['Title'].value_counts())

# add result numbers on barchart

for i, v in enumerate(labels):

    ax.text(i, v+10, str(v), horizontalalignment = 'center', size = 10, color = 'black')

    



plt.tight_layout()

plt.show()

all_data[all_data['Title']=='Ms']
title_dict = {  'Mr':     'Mr',

                'Mrs':    'Mrs',

                'Miss':   'Miss',

                'Master': 'Master',

              

                'Ms':     'Miss',

                'Mme':    'Mrs',

                'Mlle':   'Miss',



                'Capt':   'military',

                'Col':    'military',

                'Major':  'military',



                'Dr':     'Dr',

                'Rev':    'Rev',

                  

                'Sir':    'honor',

                'the Countess': 'honor',

                'Lady':   'honor',

                'Jonkheer': 'honor',

                'Don':    'honor',

                'Dona':   'honor' }



# map titles to category

all_data['Title_category'] = all_data['Title'].map(title_dict)
fig = plt.figure(figsize=(12, 5))





ax1 = fig.add_subplot(121)

ax = sns.countplot(x = 'Title_category', data = all_data, palette = "hls", order = all_data['Title_category'].value_counts().index)

_ = plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light'  

)

plt.title('Passengers distribution by titles',fontsize= 12)

plt.ylabel('Number of passengers')



# calculate passengers for each category

labels = (all_data['Title_category'].value_counts())

# add result numbers on barchart

for i, v in enumerate(labels):

    ax.text(i, v+10, str(v), horizontalalignment = 'center', size = 10, color = 'black')

    



plt.tight_layout()



ax2 = fig.add_subplot(122)

surv_by_title_cat = all_data.groupby('Title_category')['Survived'].value_counts(normalize = True).unstack()

surv_by_title_cat = surv_by_title_cat.sort_values(by=1, ascending = False)

surv_by_title_cat.plot(kind='bar', stacked='True', color=["#3f3e6fd1", "#85c6a9"], ax = ax2)



plt.legend( ( 'Drowned', 'Survived'), loc=(1.04,0))

_ = plt.xticks(

    rotation=45, 

    horizontalalignment='right',

    fontweight='light'  

)





plt.title('Proportion of survived/drowned by titles (train data)',fontsize= 12)



plt.tight_layout()

plt.show()
category_survived = sns.catplot(x="Title_category",  col="Survived",

                data = all_data, kind="count",

                height=4, aspect=.7)



category_survived.set_xticklabels(rotation=45, 

    horizontalalignment='right',

    fontweight='light')



plt.tight_layout()
class_by_title_cat = all_data.groupby('Title_category')['Pclass'].value_counts(normalize = True)

class_by_title_cat = class_by_title_cat.unstack().sort_values(by = 1, ascending = False)

class_by_title_cat.plot(kind='bar', stacked='True', color = ['#eed4d0', '#cda0aa', '#a2708e'])

plt.legend(loc=(1.04,0))

_ = plt.xticks(

    rotation = 45, 

    horizontalalignment = 'right',

    fontweight = 'light'  

)





plt.title('Proportion of 1st/2nd/3rd ticket class in each title category',fontsize= 14)

plt.xlabel('Category of the Title')

plt.tight_layout()
all_data['deck'] = all_data['Cabin'].str.split('', expand = True)[1]

all_data.loc[all_data['deck'].isna(), 'deck'] = 'U'

print('Unique deck letters from the cabin numbers:', all_data['deck'].unique())
fig = plt.figure(figsize=(20, 5))



ax1 = fig.add_subplot(131)

sns.countplot(x = 'deck', data = all_data, palette = "hls", order = all_data['deck'].value_counts().index, ax = ax1)

plt.title('Passengers distribution by deck',fontsize= 16)

plt.ylabel('Number of passengers')



ax2 = fig.add_subplot(132)

deck_by_class = all_data.groupby('deck')['Pclass'].value_counts(normalize = True).unstack()

deck_by_class.plot(kind='bar', stacked='True',color = ['#eed4d0', '#cda0aa', '#a2708e'], ax = ax2)

plt.legend(('1st class', '2nd class', '3rd class'), loc=(1.04,0))

plt.title('Proportion of classes on each deck',fontsize= 16)

plt.xticks(rotation = False)



ax3 = fig.add_subplot(133)

deck_by_survived = all_data.groupby('deck')['Survived'].value_counts(normalize = True).unstack()

deck_by_survived = deck_by_survived.sort_values(by = 1, ascending = False)

deck_by_survived.plot(kind='bar', stacked='True', color=["#3f3e6fd1", "#85c6a9"], ax = ax3)

plt.title('Proportion of survived/drowned passengers by deck',fontsize= 16)

plt.legend(( 'Drowned', 'Survived'), loc=(1.04,0))

plt.xticks(rotation = False)

plt.tight_layout()



plt.show()

all_data[(all_data['deck']=='A') & (all_data['Survived']==0)]
all_data['Family_size'] = all_data['SibSp'] + all_data['Parch'] + 1

family_size = all_data['Family_size'].value_counts()

print('Family size and number of passengers:')

print(family_size)
all_data['Surname'] = all_data['Name'].str.split(',', expand = True)[0]
all_data[all_data['Family_size'] == 7]['Surname'].value_counts()
all_data[(all_data['Family_size'] == 7) & (all_data['Surname']=='Andersson')]
all_data[(all_data['Family_size'] == 7) & (all_data['Surname']=='Andersson')].Ticket.value_counts()
all_data[(all_data['Ticket'] == '3101281') | (all_data['Ticket'] == '347091')]
all_data.loc[all_data['PassengerId'] == 69, ['SibSp', 'Parch', 'Family_size']] = [0,0,1]

all_data.loc[all_data['PassengerId'] == 1106, ['SibSp', 'Parch', 'Family_size']] = [0,0,1]

all_data[(all_data['Ticket'] == '3101281') | (all_data['Ticket'] == '347091')]
all_data[all_data['Family_size'] == 5]['Surname'].value_counts()
all_data[(all_data['Surname'] == 'Kink-Heilmann')&(all_data['Family_size'] == 5)]
fig = plt.figure(figsize = (12,4))



ax1 = fig.add_subplot(121)

ax = sns.countplot(all_data['Family_size'], ax = ax1)



# calculate passengers for each category

labels = (all_data['Family_size'].value_counts())

# add result numbers on barchart

for i, v in enumerate(labels):

    ax.text(i, v+6, str(v), horizontalalignment = 'center', size = 10, color = 'black')

    

plt.title('Passengers distribution by family size')

plt.ylabel('Number of passengers')



ax2 = fig.add_subplot(122)

d = all_data.groupby('Family_size')['Survived'].value_counts(normalize = True).unstack()

d.plot(kind='bar', color=["#3f3e6fd1", "#85c6a9"], stacked='True', ax = ax2)

plt.title('Proportion of survived/drowned passengers by family size (train data)')

plt.legend(( 'Drowned', 'Survived'), loc=(1.04,0))

plt.xticks(rotation = False)



plt.tight_layout()
all_data['Family_size_group'] = all_data['Family_size'].map(lambda x: 'f_single' if x == 1 

                                                            else ('f_usual' if 5 > x >= 2 

                                                                  else ('f_big' if 8 > x >= 5 

                                                                       else 'f_large' )

                                                                 ))                                                       
fig = plt.figure(figsize = (14,5))



ax1 = fig.add_subplot(121)

d = all_data.groupby('Family_size_group')['Survived'].value_counts(normalize = True).unstack()

d = d.sort_values(by = 1, ascending = False)

d.plot(kind='bar', stacked='True', color = ["#3f3e6fd1", "#85c6a9"], ax = ax1)

plt.title('Proportion of survived/drowned passengers by family size (training data)')

plt.legend(( 'Drowned', 'Survived'), loc=(1.04,0))

_ = plt.xticks(rotation=False)





ax2 = fig.add_subplot(122)

d2 = all_data.groupby('Family_size_group')['Pclass'].value_counts(normalize = True).unstack()

d2 = d2.sort_values(by = 1, ascending = False)

d2.plot(kind='bar', stacked='True', color = ['#eed4d0', '#cda0aa', '#a2708e'], ax = ax2)

plt.legend(('1st class', '2nd class', '3rd class'), loc=(1.04,0))

plt.title('Proportion of 1st/2nd/3rd ticket class in family group size')

_ = plt.xticks(rotation=False)



plt.tight_layout()
ax = sns.countplot(all_data['Pclass'], palette = ['#eed4d0', '#cda0aa', '#a2708e'])

# calculate passengers for each category

labels = (all_data['Pclass'].value_counts(sort = False))

# add result numbers on barchart

for i, v in enumerate(labels):

    ax.text(i, v+2, str(v), horizontalalignment = 'center', size = 12, color = 'black', fontweight = 'bold')

    

    

plt.title('Passengers distribution by family size')

plt.ylabel('Number of passengers')

plt.tight_layout()
fig = plt.figure(figsize=(14, 5))



ax1 = fig.add_subplot(121)

sns.countplot(x = 'Pclass', hue = 'Survived', data = all_data, palette=["#3f3e6fd1", "#85c6a9"], ax = ax1)

plt.title('Number of survived/drowned passengers by class (train data)')

plt.ylabel('Number of passengers')

plt.legend(( 'Drowned', 'Survived'), loc=(1.04,0))

_ = plt.xticks(rotation=False)



ax2 = fig.add_subplot(122)

d = all_data.groupby('Pclass')['Survived'].value_counts(normalize = True).unstack()

d.plot(kind='bar', stacked='True', ax = ax2, color =["#3f3e6fd1", "#85c6a9"])

plt.title('Proportion of survived/drowned passengers by class (train data)')

plt.legend(( 'Drowned', 'Survived'), loc=(1.04,0))

_ = plt.xticks(rotation=False)



plt.tight_layout()
sns.catplot(x = 'Pclass', hue = 'Survived', col = 'Sex', kind = 'count', data = all_data , palette=["#3f3e6fd1", "#85c6a9"])



plt.tight_layout()
plt.figure(figsize=(20, 10))

palette=["#3f3e6fd1", "#85c6a9"]



plt.subplot(2, 3, 1)

sns.stripplot(x = 'Survived', y = 'Age', data = age_1_class[age_1_class['Sex']=='male'],

     linewidth = 0.9, palette = palette)

plt.axhspan(0, 16, color = "#e1f3f6")

plt.axhspan(16, 40, color = "#bde6dd")

plt.axhspan(40, 80, color = "#83ceb9")

plt.title('Age distribution (males, 1st class)',fontsize= 14)

plt.xticks( np.arange(2), ['drowned', 'survived'])

plt.ylim(0, 80)



plt.subplot(2, 3, 2)

sns.stripplot(x = 'Survived', y = 'Age', data = age_2_class[age_2_class['Sex']=='male'],

     linewidth = 0.9, palette = palette)

plt.axhspan(0, 16, color = "#e1f3f6")

plt.axhspan(16, 40, color = "#bde6dd")

plt.axhspan(40, 80, color = "#83ceb9")

plt.title('Age distribution (males, 2nd class)',fontsize= 14)

plt.xticks( np.arange(2), ['drowned', 'survived'])

plt.ylim(0, 80)



plt.subplot(2, 3, 3)

sns.stripplot(x = 'Survived', y = 'Age', data = age_3_class[age_3_class['Sex']=='male'],

              linewidth = 0.9, palette = palette)

plt.axhspan(0, 16, color = "#e1f3f6")

plt.axhspan(16, 40, color = "#bde6dd")

plt.axhspan(40, 80, color = "#83ceb9")

plt.title('Age distribution (males, 3rd class)',fontsize= 14)

plt.xticks( np.arange(2), ['drowned', 'survived'])

plt.ylim(0, 80)





plt.subplot(2, 3, 4)

sns.stripplot(x = 'Survived', y = 'Age', data = age_1_class[age_1_class['Sex']=='female'],

     linewidth = 0.9, palette = palette)

plt.axhspan(0, 16, color = "#ffff9978")

plt.axhspan(16, 40, color = "#ffff97bf")

plt.axhspan(40, 80, color = "#ffed97bf")

plt.title('Age distribution (females, 1st class)',fontsize= 14)

plt.xticks( np.arange(2), ['drowned', 'survived'])

plt.ylim(0, 80)



plt.subplot(2, 3, 5)

sns.stripplot(x = 'Survived', y = 'Age', data = age_2_class[age_2_class['Sex']=='female'],

     linewidth = 0.9, palette = palette)

plt.axhspan(0, 16, color = "#ffff9978")

plt.axhspan(16, 40, color = "#ffff97bf")

plt.axhspan(40, 80, color = "#ffed97bf")

plt.title('Age distribution (females, 2nd class)',fontsize= 14)

plt.xticks( np.arange(2), ['drowned', 'survived'])

plt.ylim(0, 80)



plt.subplot(2, 3, 6)

sns.stripplot(x = 'Survived', y = 'Age', data = age_3_class[age_3_class['Sex']=='female'],

              linewidth = 0.9, palette = palette)

plt.axhspan(0, 16, color = "#ffff9978")

plt.axhspan(16, 40, color = "#ffff97bf")

plt.axhspan(40, 80, color = "#ffed97bf")

plt.title('Age distribution (females, 3rd class)',fontsize= 14)

plt.xticks( np.arange(2), ['drowned', 'survived'])

plt.ylim(0, 80)





plt.show()
plt.figure(figsize = (15,4))



plt.subplot (1,3,1)

ax = sns.countplot(all_data['Sex'], palette="Set3")

plt.title('Number of passengers by Sex')

plt.ylabel('Number of passengers')



# calculate passengers for each category

labels = (all_data['Sex'].value_counts())

# add result numbers on barchart

for i, v in enumerate(labels):

    ax.text(i, v+10, str(v), horizontalalignment = 'center', size = 10, color = 'black')

    



plt.subplot (1,3,2)

sns.countplot( x = 'Pclass', data = all_data, hue = 'Sex', palette="Set3")

plt.title('Number of male/female passengers by class')

plt.ylabel('Number of passengers')

plt.legend( loc=(1.04,0))



plt.subplot (1,3,3)

sns.countplot( x = 'Family_size_group', data = all_data, hue = 'Sex', 

              order = all_data['Family_size_group'].value_counts().index , palette="Set3")

plt.title('Number of male/female passengers by family size')

plt.ylabel('Number of passengers')

plt.legend( loc=(1.04,0))

plt.tight_layout()
fig = plt.figure(figsize = (15,4))



ax1 = fig.add_subplot(131)

palette = sns.cubehelix_palette(5, start = 2)

ax = sns.countplot(all_data['Embarked'], palette = palette, order = ['C', 'Q', 'S'], ax = ax1)

plt.title('Number of passengers by Embarked')

plt.ylabel('Number of passengers')



# calculate passengers for each category

labels = (all_data['Embarked'].value_counts())

labels = labels.sort_index()

# add result numbers on barchart

for i, v in enumerate(labels):

    ax.text(i, v+10, str(v), horizontalalignment = 'center', size = 10, color = 'black')

    



ax2 = fig.add_subplot(132)

surv_by_emb = all_data.groupby('Embarked')['Survived'].value_counts(normalize = True)

surv_by_emb = surv_by_emb.unstack().sort_index()

surv_by_emb.plot(kind='bar', stacked='True', color=["#3f3e6fd1", "#85c6a9"], ax = ax2)

plt.title('Proportion of survived/drowned passengers by Embarked (train data)')

plt.legend(( 'Drowned', 'Survived'), loc=(1.04,0))

_ = plt.xticks(rotation=False)





ax3 = fig.add_subplot(133)

class_by_emb = all_data.groupby('Embarked')['Pclass'].value_counts(normalize = True)

class_by_emb = class_by_emb.unstack().sort_index()

class_by_emb.plot(kind='bar', stacked='True', color = ['#eed4d0', '#cda0aa', '#a2708e'], ax = ax3)

plt.legend(('1st class', '2nd class', '3rd class'), loc=(1.04,0))

plt.title('Proportion of clases by Embarked')

_ = plt.xticks(rotation=False)



plt.tight_layout()


sns.catplot(x="Embarked", y="Fare", kind="violin", inner=None,

            data=all_data, height = 6, palette = palette, order = ['C', 'Q', 'S'])

plt.title('Distribution of Fare by Embarked')

plt.tight_layout()
# Descriptive statistics:

pd.DataFrame(all_data.groupby('Embarked')['Fare'].describe())
train_data[train_data['Embarked'].isna()]
sns.catplot(x="Pclass", y="Fare", kind="swarm", data=all_data, palette=sns.cubehelix_palette(5, start = 3), height = 6)



plt.tight_layout()
sns.catplot(x="Pclass", y="Fare",  hue = "Survived", kind="swarm", data=all_data, 

                                    palette=["#3f3e6fd1", "#85c6a9"], height = 6)

plt.tight_layout()
all_data[all_data['Fare'] == min(all_data['Fare'])]