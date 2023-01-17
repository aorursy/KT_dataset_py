import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
train.head()
# Concatenating train and test dataframe

def concat_df(train_df, test_df):

    return pd.concat([train_df, test_df], sort = False).reset_index(drop = True)



# Breaking combined dataframe into train and test

def divide_df(combined_df):

    return combined_df.loc[:890], combined_df.loc[891:].drop(['Survived'], axis=1)
combined = concat_df(train, test)
# yticklabels -> To get the value of each row's y-label

# cbar -> For the scale



# One can change the plot size by altering the figure size from below

plt.figure(figsize = (10, 5))

sns.heatmap(combined.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# Counting missing values for each feature

combined.isnull().sum()
# Looping over features to count missing values

for feature in combined.columns:

    print('Missing values in feature ' + str(feature) + ' : ' + str(len(combined[combined[feature].isnull() == True])))
# Percentage of missing values

for feature in combined.columns:

    print('Missing value percent in '+ feature +': {:.2f}%'.format((combined[combined[feature].isnull() == True].shape[0])/(combined.shape[0]) * 100))
# Defining plot sizes

plt.figure(figsize = (10, 8))



# Creating correlation matrix

corr_mat = combined.corr()



# Plotting the matrix

sns.heatmap(corr_mat, xticklabels = corr_mat.columns, yticklabels = corr_mat.columns, annot=True)
plt.figure(figsize = (10, 8))



# Barplot with 'x' as 'Pclass', 'y' as 'Age'

# 'hue' acts as a third dimension to visualize data

sns.barplot(x = 'Pclass', y = 'Age', hue = 'Sex', data = combined)
# Grouping data on the basis of 'Sex' and 'Pclass'

age_by_pclass_sex = combined.groupby(['Sex', 'Pclass']).median()['Age']



for pclass in range(1, 4):

    for sex in ['female', 'male']:

        print('Median age of Pclass {} {}s: {}'.format(pclass, sex, age_by_pclass_sex[sex][pclass]))



print('--------')

print('Median age of all passengers: {}'.format(combined['Age'].median()))
plt.figure(figsize = (10, 5))

sns.barplot(x = 'SibSp', y = 'Age', hue = 'Sex', data = combined)
sib_arr = combined['SibSp'].unique()

sib_arr.sort()
age_by_sibsp_sex = combined.groupby(['Sex', 'SibSp']).median()['Age']



for sibsp in sib_arr:

    for sex in ['female', 'male']:

        print('Median age of SibSp {} {}s: {}'.format(sibsp, sex, age_by_sibsp_sex[sex][sibsp]))



print('--------')

print('Median age of all passengers: {}'.format(combined['Age'].median()))
# Imputing missing values through lambda function

combined['Age'] = combined.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))
# Setting up condition to check specific values in dataframe

combined[combined['Embarked'].isnull() == True]
sns.countplot(combined[combined['Pclass'] == 1]['Embarked'])
# Applying multiple conditions to the dataframe

sns.countplot(combined[(combined['SibSp'] == 0) & (combined['Parch'] == 0)]['Embarked'])
sns.countplot(combined[combined['Embarked'] == 'S']['SibSp'])
# Imputing missing values using 'fillna' method

combined['Embarked'] = combined['Embarked'].fillna(combined['Embarked'].mode()[0])
combined[combined['Fare'].isnull() == True]
combined[(combined['Pclass'] == 3) & (combined['SibSp'] == 0) & (combined['Parch'] == 0)]
combined[(combined['Pclass'] == 3) & (combined['SibSp'] == 0) & (combined['Parch'] == 0) & (combined['Sex'] == 'male')]['Fare'].median()
combined[(combined['Pclass'] == 3) & (combined['SibSp'] == 0) & (combined['Parch'] == 0) & (combined['Sex'] == 'female')]['Fare'].median()
combined[(combined['Pclass'] == 3) & (combined['SibSp'] == 0) & (combined['Parch'] == 0)]['Fare'].median()
combined['Fare'] = combined['Fare'].fillna(combined[(combined['Pclass'] == 3) & (combined['SibSp'] == 0) & (combined['Parch'] == 0)]['Fare'].median())
combined['Cabin'].unique()
def deck_extractor(combined):

    

    # Function to separate deck name from 'Cabin' feature.

    deck_list = []

    for deck in combined['Cabin']:

        if type(deck) == str:

            deck_list.append(deck[0])

        else:

            # 'M' stands for missing cabin values

            deck_list.append('M')

            

    # Creating a new coloumn in the dataframe 

    combined['Deck'] = deck_list

    

    return combined
combined = deck_extractor(combined)
plt.figure(figsize = (10, 5))

sns.countplot(x = combined[combined['Deck'] != 'M']['Deck'], hue = combined['Pclass'],

             order = combined[combined['Deck'] != 'M']['Deck'].value_counts().index)
# Identifying unique values for a feature

deck_list = combined['Deck'].unique()



# Sorting the array

deck_list.sort()



for deck in deck_list:

    # Ignoring the missing values of decks

    if deck == 'M':

        continue

    print('For passengers from deck ' + str(deck))

    print('Number of first class passengers are: ', len(combined[(combined['Deck'] == deck) & (combined['Pclass'] == 1)]))

    print('Number of second class passengers are: ', len(combined[(combined['Deck'] == deck) & (combined['Pclass'] == 2)]))

    print('Number of third class passengers are: ', len(combined[(combined['Deck'] == deck) & (combined['Pclass'] == 3)]))

    print('----------------')
plt.figure(figsize = (10, 5))

sns.countplot(x = combined[combined['Deck'] != 'M']['Deck'], hue = combined['Survived'],

             order = combined[combined['Deck'] != 'M']['Deck'].value_counts().index)
for deck in deck_list:

    if deck == 'M':

        continue

    print('For passengers from deck ', str(deck))

    print('Percentage of survival: ' + '{:.2f}%'.format((len(combined[(combined['Deck'] == deck) & (combined['Survived'] == 1)])) / (len(combined[(combined['Deck'] == deck) & (combined['Survived'].isnull() == False)])) * 100))

    print('Percentage of death: ' + '{:.2f}%'.format((len(combined[(combined['Deck'] == deck) & (combined['Survived'] == 0)])) / (len(combined[(combined['Deck'] == deck) & (combined['Survived'].isnull() == False)])) * 100))

    print('--------------')
plt.figure(figsize = (10, 8))

sns.countplot(x = combined['Pclass'], hue = combined['Survived'])
print('Total number of first class passengers with survival data known: ', len(combined[(combined['Pclass'] == 1) & (combined['Survived'].isnull() == False)]))

print('Total number of first class passengers survived: ', len(combined[(combined['Pclass'] == 1) & (combined['Survived'] == 1)]))

print('Percentage of first class passengers survived: ' + '{:.2f}%'.format((len(combined[(combined['Pclass'] == 1) & (combined['Survived'] == 1)])) / (len(combined[(combined['Pclass'] == 1) & (combined['Survived'].isnull() == False)])) * 100))
print('Total number of second class passengers with survival data known: ', len(combined[(combined['Pclass'] == 2) & (combined['Survived'].isnull() == False)]))

print('Total number of second class passengers survived: ', len(combined[(combined['Pclass'] == 2) & (combined['Survived'] == 1)]))

print('Percentage of second class passengers survived: ' + '{:.2f}%'.format((len(combined[(combined['Pclass'] == 2) & (combined['Survived'] == 1)])) / (len(combined[(combined['Pclass'] == 2) & (combined['Survived'].isnull() == False)])) * 100))
print('Total number of third class passengers with survival data known: ', len(combined[(combined['Pclass'] == 3) & (combined['Survived'].isnull() == False)]))

print('Total number of third class passengers survived: ', len(combined[(combined['Pclass'] == 3) & (combined['Survived'] == 1)]))

print('Percentage of third class passengers survived: ' + '{:.2f}%'.format((len(combined[(combined['Pclass'] == 3) & (combined['Survived'] == 1)])) / (len(combined[(combined['Pclass'] == 3) & (combined['Survived'].isnull() == False)])) * 100))
combined[combined['Deck'] == 'T']
# Replacing the deck 'T' with 'A'

combined[combined['Deck'] == 'T'] = combined[combined['Deck'] == 'T'].replace('T', 'A') 
# Replacing multiple values in a dataframe

combined['Deck'] = combined['Deck'].replace(['A', 'B', 'C'], 'ABC')

combined['Deck'] = combined['Deck'].replace(['D', 'E'], 'DE')

combined['Deck'] = combined['Deck'].replace(['F', 'G'], 'FG')
# Dropping features after specifying axis along which it is to be dropped

combined = combined.drop(['Cabin'], axis = 1)
combined.isnull().sum()
plt.figure(figsize = (10, 8))

sns.countplot(combined['Survived'])
print('Total number of people who survived: ', len(combined[(combined['Survived'].isnull() == False) & (combined['Survived'] == 1)]))

print('Percent of people who survived: ' + '{:.2f}%'.format(len(combined[(combined['Survived'].isnull() == False) & (combined['Survived'] == 1)]) / (len(combined[(combined['Survived'].isnull() == False)])) * 100))
print('Total number of people who did not survive: ', len(combined[(combined['Survived'].isnull() == False) & (combined['Survived'] == 0)]))

print('Percent of people who did not survive: ' + '{:.2f}%'.format(len(combined[(combined['Survived'].isnull() == False) & (combined['Survived'] == 0)]) / (len(combined[(combined['Survived'].isnull() == False)])) * 100))
# Executing user defined function to break the dataframe

train, test = divide_df(combined)
plt.figure(figsize = (10, 8))

corr_mat = train.corr()

sns.heatmap(corr_mat, xticklabels = corr_mat.columns, yticklabels = corr_mat.columns, annot=True)
plt.figure(figsize = (10, 8))

corr_mat = test.corr()

sns.heatmap(corr_mat, xticklabels = corr_mat.columns, yticklabels = corr_mat.columns, annot=True)
# Creating sub-plots to visualize multiple graphs

fig,ax = plt.subplots(nrows=2, ncols=2, figsize = (20, 10))



# Defining individual graphs with coordinates specified by parameter 'ax'

sns.kdeplot(data = train[train['Survived'] == 1]['Age'], label = 'Survived', ax = ax[0, 0])

sns.kdeplot(data = train[train['Survived'] == 0]['Age'], label = 'Not Survived', ax = ax[0, 0])

sns.kdeplot(data = train['Age'], label = 'Train Set', ax = ax[1, 0])

sns.kdeplot(data = test['Age'], label = 'Test Set', ax = ax[1, 0])

sns.kdeplot(data = train['Fare'], label = 'Train Set', ax = ax[1, 1])

sns.kdeplot(data = test['Fare'], label = 'Test Set', ax = ax[1, 1])

sns.kdeplot(data = train[train['Survived'] == 1]['Fare'], label = 'Survived', ax = ax[0, 1])

sns.kdeplot(data = train[train['Survived'] == 0]['Fare'], label = 'Not Survived', ax = ax[0, 1])



# Setting up labels for the sub-plots

ax[0, 0].set(xlabel="Age")

ax[0, 1].set(xlabel="Fare")

ax[1, 0].set(xlabel="Age")

ax[1, 1].set(xlabel="Fare")



plt.show()
plt.figure(figsize = (10, 5))



# Plotting fare w.r.t. survival

# Labelling each plot using the parameter 'label'

sns.distplot(train[train['Survived'] == 1]['Fare'], label = 'Survived')

sns.distplot(train[train['Survived'] == 0]['Fare'], label = 'Not Survived')

plt.legend()

plt.plot()
train[train['Survived'] == 1]['Fare'].max()
train[train['Survived'] == 0]['Fare'].max()
train[train['Fare'] == 263.0]
sns.countplot(train[train['Fare'] > 263.0]['Survived'])
train[train['Survived'] == 1]['Fare'].min()
sns.countplot(train[train['Fare'] == 0]['Survived'])
train[train['Fare'] == 0]
plt.figure(figsize = (10, 5))

sns.countplot(train[train['Fare'] == 0]['Pclass'])
train[(train['Fare'] == 0) & (train['Survived'] == 1)]
plt.figure(figsize = (10, 5))



# Creating grids in the plot

ax = plt.gca()

ax.xaxis.grid(True)



# Plotting the graph

sns.distplot(train[train['Survived'] == 1]['Age'], label = 'Survived')

sns.distplot(train[train['Survived'] == 0]['Age'], label = 'Not Survived')

plt.xticks(np.arange(0, train['Age'].max(), step = 5))



# Creating a legend for the graph

plt.legend()

plt.plot()
plt.figure(figsize = (10, 8))

sns.countplot(x = train['Pclass'], hue = train['Survived'])
plt.figure(figsize = (10, 5))

sns.countplot(x = train['Sex'], hue = train['Survived'])
print('Percent of males who survived:' + ' {:.2f}%'.format((len(train[(train['Sex'] == 'male') & (train['Survived'] == 1)])) / (len(train)) * 100))

print('Percent of males who did not survive:' + ' {:.2f}%'.format((len(train[(train['Sex'] == 'male') & (train['Survived'] == 0)])) / (len(train)) * 100))

print('------------')

print('Percent of females who survived:' + ' {:.2f}%'.format((len(train[(train['Sex'] == 'female') & (train['Survived'] == 1)])) / (len(train)) * 100))

print('Percent of females who did not survive:' + ' {:.2f}%'.format((len(train[(train['Sex'] == 'female') & (train['Survived'] == 0)])) / (len(train)) * 100))
plt.figure(figsize = (10, 5))

sns.countplot(x = train['SibSp'], hue = train['Survived'])
sib_arr = train['SibSp'].unique()

sib_arr.sort()
for sib in sib_arr:

    print('Percent of passengers with ' + str(sib) + ' siblings/spouce who survived: {:.2f}%'.format((len(train[(train['SibSp'] == sib) & (train['Survived'] == 1)])) / (len(train[(train['SibSp'] == sib)])) * 100))

    print('Percent of passengers with ' + str(sib) + ' siblings/spouce who did not survive: {:.2f}%'.format((len(train[(train['SibSp'] == sib) & (train['Survived'] == 0)])) / (len(train[(train['SibSp'] == sib)])) * 100))

    print('------------')
plt.figure(figsize = (10, 5))

sns.countplot(x = train['Parch'], hue = train['Survived'])
parc_arr = train['Parch'].unique()

parc_arr.sort()
for parc in parc_arr:

    print('Percent of passengers with ' + str(parc) + ' parents/children who survived: {:.2f}%'.format((len(train[(train['Parch'] == parc) & (train['Survived'] == 1)])) / (len(train[(train['Parch'] == parc)])) * 100))

    print('Percent of passengers with ' + str(parc) + ' parents/children who did not survive: {:.2f}%'.format((len(train[(train['Parch'] == parc) & (train['Survived'] == 0)])) / (len(train[(train['Parch'] == parc)])) * 100))

    print('------------')
plt.figure(figsize = (10, 5))

sns.countplot(x = train['Embarked'], hue = train['Survived'])
for port in train['Embarked'].unique():

    print('Percent of passengers embarked on ' + str(port) + ' who survived: {:.2f}%'.format((len(train[(train['Embarked'] == port) & (train['Survived'] == 1)])) / (len(train[(train['Embarked'] == port)])) * 100))

    print('Percent of passengers embarked on ' + str(port) + ' who did not survive: {:.2f}%'.format((len(train[(train['Embarked'] == port) & (train['Survived'] == 0)])) / (len(train[(train['Embarked'] == port)])) * 100))

    print('------------')
plt.figure(figsize = (10, 5))

sns.countplot(x = train[train['Deck'] != 'M']['Deck'], hue = train['Survived'])
for deck in train['Deck'].unique():

    if deck == 'M':

        continue

    print('Percent of passengers on deck ' + str(deck) + ' who survived: {:.2f}%'.format((len(train[(train['Deck'] == deck) & (train['Survived'] == 1)])) / (len(train[(train['Deck'] == deck)])) * 100))

    print('Percent of passengers on deck ' + str(deck) + ' who did not survive: {:.2f}%'.format((len(train[(train['Deck'] == deck) & (train['Survived'] == 0)])) / (len(train[(train['Deck'] == deck)])) * 100))

    print('------------')
plt.figure(figsize = (10, 5))

ax = plt.gca()

ax.yaxis.grid(True)

sns.barplot(x = train['Pclass'], y = train['Age'])
plt.figure(figsize = (10, 5))

sns.barplot(x = train['Pclass'], y = train['Fare'], hue = train['Survived'])
fig, ax = plt.subplots(ncols = 2, figsize = (20, 6))

plt.figure(figsize = (10, 5))

sns.barplot(x = train['SibSp'], y = train['Age'], hue = train['Survived'], ax = ax[0])

sns.barplot(x = train['Parch'], y = train['Age'], hue = train['Survived'], ax = ax[1])

plt.show()
plt.figure(figsize = (15, 8))

sns.scatterplot(x = train['Age'], y = train['Fare'], hue = train['Survived'])

plt.xlabel('Age')

plt.ylabel('Fare')
fig, ax = plt.subplots(ncols = 2, figsize = (20, 6))

sns.distplot(train[(train['Age'] <= 10) & (train['Survived'] == True)]['Fare'], label = 'Survived', ax = ax[0])

sns.distplot(train[(train['Age'] <= 10) & (train['Survived'] == False)]['Fare'], label = 'Not Survived', ax = ax[0])

sns.distplot(train[(train['Age'] > 10) & (train['Survived'] == True) & (train['Fare'] <= 200)]['Fare'], label = 'Survived', ax = ax[1])

sns.distplot(train[(train['Age'] > 10) & (train['Survived'] == False) & (train['Fare'] <= 200)]['Fare'], label = 'Survived', ax = ax[1])

ax[0].title.set_text('Passengers less than 10 years of age')

ax[1].title.set_text('Passengers more than 10 years of age')

ax[0].legend()

ax[1].legend()

plt.xlabel('Fare')
# Specifying the color scheme for the datapoints

color_dict = dict({'male': 'blue', 

                   'female': 'red'})



fig, ax = plt.subplots(ncols = 2, figsize = (15,5))

sns.scatterplot(x = train['Age'], y = train['Fare'], hue = train[train['Survived'] == 1]['Sex'], palette = color_dict, ax = ax[0])

sns.scatterplot(x = train['Age'], y = train['Fare'], hue = train[train['Survived'] == 0]['Sex'], palette = color_dict, ax = ax[1])

plt.xlabel('Age')

plt.ylabel('Fare')

ax[0].title.set_text('Survived')

ax[1].title.set_text('Not Survived')
fig, ax = plt.subplots(ncols = 2, figsize = (20, 6))

sns.countplot(train[(train['Age'] <= 10) & (train['Sex'] == 'male')]['Survived'], ax = ax[0])

sns.countplot(train[(train['Age'] <= 10) & (train['Sex'] == 'female')]['Survived'], ax = ax[1])

ax[0].title.set_text('Male children less than 10 years of age')

ax[1].title.set_text('Female children less than 10 years of age')
print('Percentage of male children who survived: {:.2f}%'.format(((len(train[(train['Age'] <= 10) & (train['Sex'] == 'male') & (train['Survived'] == True)])) / (len(train[(train['Age'] <= 10) & (train['Sex'] == 'male')])) * 100)))

print('Percentage of male children who did not survive: {:.2f}%'.format(((len(train[(train['Age'] <= 10) & (train['Sex'] == 'male') & (train['Survived'] == False)])) / (len(train[(train['Age'] <= 10) & (train['Sex'] == 'male')])) * 100)))

print('---------------')

print('Percentage of female children who survived: {:.2f}%'.format(((len(train[(train['Age'] <= 10) & (train['Sex'] == 'female') & (train['Survived'] == True)])) / (len(train[(train['Age'] <= 10) & (train['Sex'] == 'female')])) * 100)))

print('Percentage of female children who did not survive: {:.2f}%'.format(((len(train[(train['Age'] <= 10) & (train['Sex'] == 'female') & (train['Survived'] == False)])) / (len(train[(train['Age'] <= 10) & (train['Sex'] == 'female')])) * 100)))
plt.figure(figsize = (10, 5))

sns.countplot(train['SibSp'], hue = train['Parch'])

plt.legend(loc = 'upper right', title = 'Parch')
plt.figure(figsize = (10, 5))

sns.barplot(x = train['SibSp'], y = train['Fare'])
print('Minimum fare for SibSp = 3 is: ', train[train['SibSp'] == 3]['Fare'].min())

print('Minimum fare for SibSp = 8 is: ', train[train['SibSp'] == 8]['Fare'].min())

print('----------------')

print('Maximum fare for SibSp = 3 is: ', train[train['SibSp'] == 3]['Fare'].max())

print('Maximum fare for SibSp = 8 is: ', train[train['SibSp'] == 8]['Fare'].max())
train[train['SibSp'] == 8]
plt.figure(figsize = (10, 5))

sns.barplot(x = train['SibSp'], y = train['Fare'], hue = train['Survived'])
plt.figure(figsize = (10, 5))

sns.barplot(x = train['Parch'], y = train['Fare'])
plt.figure(figsize = (10, 5))

sns.barplot(x = train['Parch'], y = train['Fare'], hue = train['Survived'])
train[train['Parch'] == 4]
plt.figure(figsize = (10, 5))

sns.distplot(combined['Age'])
plt.figure(figsize = (10, 5))

sns.boxplot(combined['Age'])
age_quartile_1, age_quartile_3 = np.percentile(combined['Age'], [25, 75])



print(age_quartile_1, age_quartile_3)
IQR_age = age_quartile_3 - age_quartile_1



print(IQR_age)
lower_bound_age = age_quartile_1 - (1.5 * IQR_age)

upper_bound_age = age_quartile_3 + (1.5 * IQR_age)



print('Age Lower Bound (IQR): ', lower_bound_age)

print('Age Upper Bound (IQR): ', upper_bound_age)
print('Percentage of non-outliers according to IQR: {:.2f}%'.format(len(combined[(combined['Age'] >= lower_bound_age) & (combined['Age'] <= upper_bound_age)]) / (len(combined)) * 100))
from scipy import stats
unique, counts = np.unique((stats.zscore(combined['Age']) > -3) & (stats.zscore(combined['Age']) < 3), return_counts = True)

count_dict = dict(zip(unique, counts))

print('Percentage of non-outliers according to Z-Scores: {:.2f}%'.format((count_dict[True]) / (count_dict[True] + count_dict[False]) * 100))
combined[(stats.zscore(combined['Age']) >= -3) & (stats.zscore(combined['Age']) <= 3)]
print('Age Lower Bound (Z-Score): ', combined[(stats.zscore(combined['Age']) >= -3) & (stats.zscore(combined['Age']) <=3)]['Age'].min())

print('Age Upper Bound (Z-Score): ', combined[(stats.zscore(combined['Age']) >= -3) & (stats.zscore(combined['Age']) <= 3)]['Age'].max())
plt.figure(figsize = (10, 5))

sns.distplot(combined['Fare'])
plt.figure(figsize = (10, 5))

sns.boxplot(combined['Fare'])
fare_quartile_1, fare_quartile_3 = np.percentile(combined['Fare'], [25, 75])



print(fare_quartile_1, fare_quartile_3)
IQR_fare = fare_quartile_3 - fare_quartile_1



print(IQR_fare)
lower_bound_fare = fare_quartile_1 - (1.5 * IQR_fare)

upper_bound_fare = fare_quartile_3 + (1.5 * IQR_fare)



print('Fare Lower Bound (IQR): ', lower_bound_fare)

print('Fare Upper Bound (IQR): ', upper_bound_fare)
print('Percentage of non-outliers according to IQR: {:.2f}%'.format(len(combined[(combined['Fare'] >= lower_bound_fare) & (combined['Fare'] <= upper_bound_fare)]) / (len(combined)) * 100))
unique_fare, counts_fare = np.unique((stats.zscore(combined['Fare']) > -3) & (stats.zscore(combined['Fare']) < 3), return_counts = True)

fare_count_dict = dict(zip(unique_fare, counts_fare))

print('Percentage of non-outliers according to Z-Scores: {:.2f}%'.format((fare_count_dict[True]) / (fare_count_dict[True] + fare_count_dict[False]) * 100))
print('Fare Lower Bound (Z-Score): ', combined[(stats.zscore(combined['Fare']) >= -3) & (stats.zscore(combined['Fare']) <= 3)]['Fare'].min())

print('Fare Upper Bound (Z-Score): ', combined[(stats.zscore(combined['Fare']) >= -3) & (stats.zscore(combined['Fare']) <= 3)]['Fare'].max())
combined_dummy = combined.copy()
combined[stats.zscore(combined['Age']) > 3]
combined_dummy = combined_dummy.replace(dict.fromkeys(combined_dummy[stats.zscore(combined_dummy['Age']) > 3]['Age'], 67.0))
combined_dummy = combined_dummy.replace(dict.fromkeys(combined_dummy[stats.zscore(combined_dummy['Fare']) > 3]['Fare'], 164.8667))
fig, ax = plt.subplots(ncols = 2, figsize = (20, 6))

sns.kdeplot(combined['Age'], ax = ax[0], label = 'Before Filter', color = 'red')

sns.kdeplot(combined_dummy['Age'], ax = ax[0], label = 'After Filter', color = 'blue')

sns.kdeplot(combined['Fare'], ax = ax[1], label = 'Before Filter', color = 'red')

sns.kdeplot(combined_dummy['Fare'], ax = ax[1], label = 'After Filter', color = 'blue')

ax[0].title.set_text('Distribution of Age')

ax[1].title.set_text('Distribution of Fare')

ax[0].legend(loc = 'upper right')

ax[1].legend(loc = 'upper right')
plt.figure(figsize = (10, 5))

sns.kdeplot(combined['Fare'], label = 'Before Filtering', color = 'red')

sns.kdeplot(combined_dummy['Fare'], label = 'After Filtering', color = 'blue')

plt.axvline(x = 164.8667, color = 'yellow')
# Creating bins for continous variables

# Number of bins can be specified. For this case number of bins are 13

combined_dummy['Fare Bins'] = pd.qcut(combined_dummy['Fare'], 13)

combined['Fare Bins'] = pd.qcut(combined['Fare'], 13)



fig, axs = plt.subplots(nrows = 2, figsize=(20, 10))

plt.subplots_adjust(bottom = 0.1)

sns.countplot(x='Fare Bins', hue='Survived', data=combined_dummy, ax = axs[0])

sns.countplot(x='Fare Bins', hue='Survived', data=combined, ax = axs[1])



axs[0].title.set_text('After Filtering')

axs[1].title.set_text('Before Filtering')
combined_dummy['Age Bins'] = pd.qcut(combined_dummy['Age'], 9)

combined['Age Bins'] = pd.qcut(combined['Age'], 9)



fig, axs = plt.subplots(nrows = 2, figsize=(20, 10))

plt.subplots_adjust(bottom = 0.1)

plt.figure(figsize = (20, 6))

sns.countplot(x='Age Bins', hue='Survived', data=combined_dummy, ax = axs[0])

sns.countplot(x='Age Bins', hue='Survived', data=combined, ax = axs[1])



axs[0].title.set_text('After Filtering')

axs[1].title.set_text('Before Filtering')
name_title = []

for name in combined_dummy['Name']:

    for i in name.split():

        if '.' in i:

            name_title.append(i)



unique_name_title = list(set(name_title))         
unique_name_title
strange_title = ['Mlle.', 'Mme.', 'Jonkheer.', 'L.', 'Rev.']
for name in combined_dummy['Name']:

    for title in strange_title:    

        if title in name.split():

            print(name)
unique_name_title.remove('L.')

name_title.remove('L.')



#printing the name list

unique_name_title
occurence_list = []

for title in unique_name_title:

    print('{} has {} occurences.'.format(title, name_title.count(title)))

    occurence_list.append(name_title.count(title))
title_dict = dict(zip(unique_name_title, occurence_list))

title_df = pd.DataFrame(title_dict.items(), columns = ['Title', 'Occurences'])
title_df.head()
plt.figure(figsize = (15, 8))

sns.barplot(x = 'Title', y = 'Occurences', data = title_df)

plt.title('Title count')

plt.show()
combined_dummy['Title'] = name_title
plt.figure(figsize = (10, 5))

sns.countplot(x = 'Title', data = combined_dummy[combined_dummy['Survived'] == True])
def generic_title(combined_dummy):

    # Function to generalize 'Title' feature

    title_arr = combined_dummy['Title']

    generic_title_arr = []

    

    for title in combined_dummy['Title']:

        if title == 'Mr.':

            generic_title_arr.append('Mr')

        elif title == 'Mrs.' or title == 'Miss.' or title == 'Countess.' or title == 'Ms.' or title == 'Mme.' or title == 'Lady.' or title == 'Mlle.' or title == 'Dona.':

            generic_title_arr.append('Mrs/Miss/Ms')

        elif title == 'Major.' or title == 'Col.' or title == 'Dr.' or title == 'Sir.' or title == 'Jonkheer.' or title == 'Rev.' or title == 'Capt.' or title == 'Don.':

            generic_title_arr.append('Dr/Military/Noble/Clergy')

        elif title == 'Master.':

            generic_title_arr.append('Master')

    

    if len(combined_dummy) != len(generic_title_arr):

        print('Check the titles again!')

        return False

    

    combined_dummy['Generic Title'] = generic_title_arr

    

    return combined_dummy
combined_dummy = generic_title(combined_dummy)
combined_dummy.head()
plt.figure(figsize = (10, 5))

sns.countplot(x = 'Generic Title', hue = combined_dummy['Survived'], data = combined_dummy,

              order = combined_dummy['Generic Title'].value_counts().index)
generic_title_arr = combined_dummy['Generic Title'].unique()

generic_title_arr.sort()
for title in generic_title_arr:

    print('Percent of passengers with a generic title of ' + str(title) + ' who survived: {:.2f}%'.format((len(combined_dummy[(combined_dummy['Generic Title'] == title) & (combined_dummy['Survived'] == 1)])) / (len(combined_dummy[(combined_dummy['Generic Title'] == title) & (combined_dummy['Survived'].isnull() == False)])) * 100))

    print('Percent of passengers with a generic title of ' + str(title) + ' who did not survive: {:.2f}%'.format((len(combined_dummy[(combined_dummy['Generic Title'] == title) & (combined_dummy['Survived'] == 0)])) / (len(combined_dummy[(combined_dummy['Generic Title'] == title) & (combined_dummy['Survived'].isnull() == False)])) * 100))

    print('------------')
def family_size(train_dummy):

    sibsp_size = train_dummy['SibSp']

    parch_size = train_dummy['Parch']

    

    sibsp_size = list(sibsp_size)

    parch_size = list(parch_size)

    

    family_size = []

    

    for i in range(len(train_dummy)):

        family_size.append(sibsp_size[i] + parch_size[i] + 1)

        

    train_dummy['Family Size'] = family_size

    

    return train_dummy
combined_dummy = family_size(combined_dummy)
plt.figure(figsize = (10, 5))

sns.countplot(x = combined_dummy['Family Size'], hue = combined_dummy['Survived'])
family_arr = combined_dummy['Family Size'].unique()

family_arr.sort()
for family in family_arr:

    print('Percent of passengers with a family size of ' + str(family) + ' who survived: {:.2f}%'.format((len(combined_dummy[(combined_dummy['Family Size'] == family) & (combined_dummy['Survived'] == 1)])) / (len(combined_dummy[(combined_dummy['Family Size'] == family) & (combined_dummy['Survived'].isnull() == False)])) * 100))

    print('Percent of passengers with a family size of ' + str(family) + ' who did not survive: {:.2f}%'.format((len(combined_dummy[(combined_dummy['Family Size'] == family) & (combined_dummy['Survived'] == 0)])) / (len(combined_dummy[(combined_dummy['Family Size'] == family) & (combined_dummy['Survived'].isnull() == False)])) * 100))

    print('------------')
def family_category(train_dummy):

    family_size = train_dummy['Family Size']

    family_size = list(family_size)

    

    family_category = []

    

    for i in range(len(family_size)):

        if family_size[i] == 1:

            family_category.append('Alone')

        elif family_size[i] == 2 or family_size[i] == 3 or family_size[i] == 4:

            family_category.append('Small')

        elif family_size[i] == 5 or family_size[i] == 6:

            family_category.append('Medium')

        elif family_size[i] == 7 or family_size[i] == 8 or family_size[i] == 11:

            family_category.append('Large')

        else:

            print('Invalid case for family size: ', family_size[i])

            

    train_dummy['Family Category'] = family_category

    

    return train_dummy
combined_dummy = family_category(combined_dummy)
combined_dummy.head()
plt.figure(figsize = (10, 5))

sns.countplot(x = combined_dummy['Family Category'], hue = combined_dummy['Survived'],

              order = combined_dummy['Family Category'].value_counts().index)
category_arr = combined_dummy['Family Category'].unique()

category_arr.sort()
for category in category_arr:

    print('Percent of passengers categorized as ' + str(category) + ' who survived: {:.2f}%'.format((len(combined_dummy[(combined_dummy['Family Category'] == category) & (combined_dummy['Survived'] == 1)])) / (len(combined_dummy[(combined_dummy['Family Category'] == category) & (combined_dummy['Survived'].isnull() == False)])) * 100))

    print('Percent of passengers categorized as ' + str(category) + ' who did not survive: {:.2f}%'.format((len(combined_dummy[(combined_dummy['Family Category'] == category) & (combined_dummy['Survived'] == 0)])) / (len(combined_dummy[(combined_dummy['Family Category'] == category) & (combined_dummy['Survived'].isnull() == False)])) * 100))

    print('------------')
combined_dummy.head()
combined_dummy[combined_dummy['Ticket'] == '113572']
combined_dummy['Ticket Frequency'] = combined_dummy.groupby('Ticket')['Ticket'].transform('count')
plt.figure(figsize = (10, 5))

sns.countplot(x = combined_dummy['Ticket Frequency'], hue = combined_dummy['Survived'],

              order = combined_dummy['Ticket Frequency'].value_counts().index)
ticket_freq_arr = combined_dummy['Ticket Frequency'].unique()

ticket_freq_arr.sort()
for ticket_freq in ticket_freq_arr:

    print('Percent of passengers travelling in a group of ' + str(ticket_freq) + ' who survived: {:.2f}%'.format((len(combined_dummy[(combined_dummy['Ticket Frequency'] == ticket_freq) & (combined_dummy['Survived'] == 1)])) / (len(combined_dummy[(combined_dummy['Ticket Frequency'] == ticket_freq) & (combined_dummy['Survived'].isnull() == False)])) * 100))

    print('Percent of passengers travelling in a group of ' + str(ticket_freq) + ' who did not survive: {:.2f}%'.format((len(combined_dummy[(combined_dummy['Ticket Frequency'] == ticket_freq) & (combined_dummy['Survived'] == 0)])) / (len(combined_dummy[(combined_dummy['Ticket Frequency'] == ticket_freq) & (combined_dummy['Survived'].isnull() == False)])) * 100))

    print('------------')
combined_dummy.head()
print('Percentage of women who survived in 1st class: {:.2f}%'.format((len(combined_dummy[(combined_dummy['Survived'].isnull() == False) & (combined_dummy['Sex'] == 'female') & (combined_dummy['Pclass'] == 1) & (combined_dummy['Survived'] == 1)])) / (len(combined_dummy[(combined_dummy['Survived'].isnull() == False) & (combined_dummy['Sex'] == 'female') & (combined_dummy['Pclass'] == 1)]['Survived'])) * 100))

print('Percentage of women who survived in 2nd class: {:.2f}%'.format((len(combined_dummy[(combined_dummy['Survived'].isnull() == False) & (combined_dummy['Sex'] == 'female') & (combined_dummy['Pclass'] == 2) & (combined_dummy['Survived'] == 1)])) / (len(combined_dummy[(combined_dummy['Survived'].isnull() == False) & (combined_dummy['Sex'] == 'female') & (combined_dummy['Pclass'] == 2)]['Survived'])) * 100))

print('Percentage of women who survived in 3rd class: {:.2f}%'.format((len(combined_dummy[(combined_dummy['Survived'].isnull() == False) & (combined_dummy['Sex'] == 'female') & (combined_dummy['Pclass'] == 3) & (combined_dummy['Survived'] == 1)])) / (len(combined_dummy[(combined_dummy['Survived'].isnull() == False) & (combined_dummy['Sex'] == 'female') & (combined_dummy['Pclass'] == 3)]['Survived'])) * 100))
percent_survival_list = [96.81, 92.11, 50.00]

pclass_list = [1, 2, 3]



plt.figure(figsize = (10, 5))

sns.lineplot(x = pclass_list, y = percent_survival_list)
from sklearn.preprocessing import LabelEncoder
label_encode_features = ['Age Bins', 'Fare Bins']



for feature in label_encode_features:

    combined_dummy[str(feature) + ' Encoded'] = LabelEncoder().fit_transform(combined_dummy[feature])
from sklearn.preprocessing import OneHotEncoder
one_hot_encoded_features = ['Sex', 'Deck', 'Embarked', 'Generic Title', 'Family Category']



encoded_features = []



for feature in one_hot_encoded_features:

    encoded_feature_array =  OneHotEncoder().fit_transform(combined_dummy[feature].values.reshape(-1, 1)).toarray()

    column_names = [str(feature) + '_{}'.format(n) for n in range((combined_dummy[feature].nunique()))]

    one_hot_df = pd.DataFrame(encoded_feature_array, columns = column_names)

    one_hot_df.index = combined_dummy.index

    encoded_features.append(one_hot_df)

    

combined_dummy = pd.concat([combined_dummy, *encoded_features], axis=1)
combined_dummy.head()
drop_columns = ['PassengerId', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked', 'Deck', 'Fare Bins', 'Age Bins', 'Title', 'Generic Title', 'Family Category', 'Family Size']



combined_dummy.drop(columns = drop_columns, axis = 1, inplace = True)
combined_dummy.head()
train_df = combined_dummy[combined_dummy['Survived'].isnull() == False]

test_df = combined_dummy[combined_dummy['Survived'].isnull() == True]
# Creating training dataframe

X_train = train_df.drop('Survived', axis = 1, inplace = False)

y_train = train_df['Survived'].values



# Creating the test dataframe

X_test = test_df.drop('Survived', axis = 1, inplace = False)
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train)

X_test = StandardScaler().fit_transform(X_test)
from sklearn.linear_model import LogisticRegression
# Fitting the model on training data

log_reg_model = LogisticRegression()

log_reg_model.fit(X_train, y_train)
from sklearn.model_selection import cross_val_score
log_reg_accuracy = cross_val_score(log_reg_model, X_train, y_train).mean() * 100
print('Accuracy: {:.2f}%'.format(log_reg_accuracy))
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_train,  log_reg_model.predict(X_train)))
print(classification_report(y_train, log_reg_model.predict(X_train)))
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
param_grid = [    

    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],

    'C' : np.logspace(-4, 4, 20),

    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],

    'max_iter' : [100, 1000,2500, 5000]

    }

]
log_reg_random = RandomizedSearchCV(log_reg_model, param_distributions = param_grid, cv = 5, verbose=True, n_jobs=-1)
best_log_reg_random = log_reg_random.fit(X_train,y_train)
best_log_reg_random.best_estimator_
def random_search_cv(model, param_grid):

    # Creating a Randomized Search Model

    model_random = RandomizedSearchCV(model, param_distributions = param_grid, cv = 3, verbose=True, n_jobs=-1)

    

    # Fitting the model to the data

    best_model_random = model_random.fit(X_train,y_train)

    print('Best Estimator: \n', best_model_random.best_estimator_)

    

    return best_model_random.best_estimator_
def model_evaluation(model):

    # Calculating accuracy

    model_accuracy = cross_val_score(model, X_train, y_train).mean() * 100

    print('Accuracy: {:.2f}%'.format(model_accuracy))

    

    # Printing the confusion matrix

    print('\nConfusion Matrix: ')

    print(confusion_matrix(y_train,  model.predict(X_train)))

    

    # Printing the classification report

    print('\nClassification Report: ')

    print(classification_report(y_train, model.predict(X_train)))
model_evaluation(best_log_reg_random.best_estimator_)
param_grid = [    

    {'penalty' : ['l1', 'l2', 'elasticnet', 'none'],

    'C' : np.logspace(1, 2, 20),

    'solver' : ['lbfgs','newton-cg','liblinear','sag','saga'],

    'max_iter' : [800, 900, 1000, 1500, 1800]

    }

]
def grid_search_cv(model, param_grid):

    # Creating a Grid Search Model

    model_grid = GridSearchCV(model, param_grid= param_grid, cv = 5, verbose=True, n_jobs=-1)

    

    # Fitting the model to the data

    best_model_grid = model_grid.fit(X_train,y_train)

    print('Best Estimator: \n', best_model_grid.best_estimator_)

    

    return best_model_grid.best_estimator_
best_log_reg_estimator = grid_search_cv(log_reg_model, param_grid)
model_evaluation(best_log_reg_estimator)
from sklearn.svm import SVC
# Fitting the model on training data

svc_model = SVC()

svc_model.fit(X_train, y_train)
print('Accuracy: {:.2f}%'.format(cross_val_score(svc_model, X_train, y_train).mean() * 100))
print(confusion_matrix(y_train,  svc_model.predict(X_train)))
print(classification_report(y_train, svc_model.predict(X_train)))
param_grid = [{

    'C': [0.1, 1, 10, 100, 1000],

    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],

    'kernel': ['rbf', 'poly', 'linear', 'sigmoid'],

    'degree': [2, 3, 5, 7, 10]

}]
random_svc_best_model = random_search_cv(svc_model, param_grid)
model_evaluation(random_svc_best_model)
param_grid = [{

    'C': [80, 85, 90],

    'gamma': [0.0001, 0.0025, 0.003],

    'kernel': ['rbf', 'linear', 'sigmoid'],

}]
grid_svc_best_model = grid_search_cv(svc_model, param_grid)
model_evaluation(grid_svc_best_model)
from sklearn.tree import DecisionTreeClassifier
# Fitting the model on training data

dt_model = DecisionTreeClassifier()

dt_model.fit(X_train, y_train)
print('Accuracy: {:.2f}%'.format(cross_val_score(dt_model, X_train, y_train).mean() * 100))
print(confusion_matrix(y_train,  dt_model.predict(X_train)))
print(classification_report(y_train, dt_model.predict(X_train)))
param_grid = [    

    {'splitter' : ['best', 'random'],

     'max_depth' : np.linspace(1, 32, 32, endpoint=True),

     'min_samples_split' : [i for i in range(1, 11)],

     'min_samples_leaf' : np.linspace(0.1, 0.5, 10, endpoint=True),

     'max_features' : list(range(1,X_train.shape[1])),

    }

]
random_dt_best_model = random_search_cv(dt_model, param_grid)
model_evaluation(random_dt_best_model)
dt_model = DecisionTreeClassifier(max_features = 8, splitter = 'best', min_samples_split = 10)
param_grid = [    

    {'max_depth' : np.linspace(1, 5, 5, endpoint=True),

     'min_samples_leaf' : np.linspace(0.01, 0.1, 10, endpoint=True),

    }

]
grid_dt_best_model = grid_search_cv(dt_model, param_grid)
model_evaluation(grid_dt_best_model)
from sklearn.ensemble import RandomForestClassifier
# Fitting the model on training data

rf_model = RandomForestClassifier(n_estimators=1000)

rf_model.fit(X_train, y_train)
print('Accuracy: {:.2f}%'.format(cross_val_score(rf_model, X_train, y_train).mean() * 100))
print(confusion_matrix(y_train,  rf_model.predict(X_train)))
print(classification_report(y_train, rf_model.predict(X_train)))
param_grid = [

  {'n_estimators': [int(x) for x in np.linspace(start = 1500, stop = 2500, num = 10)],

    'max_features': ['auto', 'sqrt'],

    'max_depth': [int(x) for x in np.linspace(5, 50, num = 11)],

    'min_samples_split': [2, 3, 5, 7, 10],

    'min_samples_leaf': [3, 4, 6, 8],

    'random_state' : [42, 43],

    'bootstrap': [True, False]

  }

]
random_rf_best_model = random_search_cv(rf_model, param_grid)
model_evaluation(random_rf_best_model)
rf_model = RandomForestClassifier(max_features='sqrt', random_state=43, max_depth=17, min_samples_leaf=3, min_samples_split=7)
param_grid = [

  { 'n_estimators' : [int(x) for x in np.linspace(start = 23, stop = 30, num = 7)]  }

]
grid_rf_best_model = grid_search_cv(rf_model, param_grid)
model_evaluation(grid_rf_best_model)
from sklearn.neighbors import KNeighborsClassifier
# Fitting the model on training data

knn_model = KNeighborsClassifier(n_neighbors = 5)

knn_model.fit(X_train, y_train)
print('Accuracy: {:.2f}%'.format(cross_val_score(knn_model, X_train, y_train).mean() * 100))
print(confusion_matrix(y_train,  knn_model.predict(X_train)))
print(classification_report(y_train, knn_model.predict(X_train)))
param_grid = [

  {'n_neighbors' : [3, 5, 7, 9],

   'algorithm' : ['auto', 'ball_tree', 'kd_tree'], 

   'leaf_size': [10, 20, 30, 40],

  }

]
random_knn_best_model = random_search_cv(knn_model, param_grid)
model_evaluation(random_knn_best_model)
n_neighbours = [3, 5, 7, 9, 11, 13, 15, 17]

accuracy_list = []



for neighbour in n_neighbours:

    knn_model = KNeighborsClassifier(algorithm='ball_tree', leaf_size=20, metric='minkowski',

                     metric_params=None, n_jobs=None, n_neighbors=neighbour, p=2,

                     weights='uniform')

    accuracy_list.append(cross_val_score(knn_model, X_train, y_train).mean() * 100)

    print('For neighbour: ' + str(neighbour) + ' accuracy is {:.2f}%'.format(cross_val_score(knn_model, X_train, y_train, cv = 5).mean() * 100))

    

plt.figure(figsize = (10, 5))

plt.xticks(np.arange(3, 18, 2))

plt.plot(n_neighbours, accuracy_list)
knn_model = KNeighborsClassifier(n_neighbors=5)
param_grid = [

  {

      'algorithm' : ['auto', 'ball_tree', 'kd_tree'], 

      'leaf_size': [5, 6, 7, 9, 11],

      'p': [1, 2]

  }

]
grid_knn_best_model = grid_search_cv(knn_model, param_grid)
model_evaluation(grid_knn_best_model)
from sklearn.linear_model import SGDClassifier
# Fitting the model on training data

sgd_model = SGDClassifier()

sgd_model.fit(X_train, y_train)
print('Accuracy: {:.2f}%'.format(cross_val_score(sgd_model, X_train, y_train).mean() * 100))
print(confusion_matrix(y_train,  sgd_model.predict(X_train)))
print(classification_report(y_train, sgd_model.predict(X_train)))
param_grid = [

  {

      'loss' : ['hinge', 'log', 'squared_hinge', 'modified_huber'],

      'alpha' : [0.0001, 0.001, 0.1, 0.5, 1],

      'penalty' : ['l2', 'l1'],

      'max_iter' : [4000, 5000, 6000, 7000]

  }

]
random_sgd_best_model = random_search_cv(sgd_model, param_grid)
model_evaluation(random_sgd_best_model)
sgd_model = SGDClassifier(penalty = 'l2', max_iter = 5000)
param_grid = [

  {

      'loss' : ['hinge', 'log', 'squared_hinge', 'modified_huber'],

      'alpha' : [0.4, 0.5]

  }

]
grid_sgd_best_model = grid_search_cv(sgd_model, param_grid)
model_evaluation(grid_sgd_best_model)
tuned_model_summary = pd.DataFrame({'Model' : ['Logistic Regession', 'Support Vector Classifier', 'Decision Tree', 'Random Forest', 'K-Nearest Neighbours', 'SGDClassifier'],

                       'Accuracy' : [82.72, 82.83, 79.80, 84.51, 83.28, 80.13],

                       'False Negatives' : [71, 62, 85, 35, 46, 59]})



tuned_model_summary
from sklearn.metrics import roc_curve
model_list = [best_log_reg_estimator, grid_svc_best_model, grid_dt_best_model, grid_rf_best_model, grid_knn_best_model, grid_sgd_best_model]

x = np.linspace(0, 1, 11)

y = np.linspace(0, 1, 11)

plt.figure(figsize = (10, 8))



for model in model_list:

    fpr, tpr, thresholds = roc_curve(y_true = y_train, y_score = model.predict(X_train), pos_label=0)

    if model == model_list[0]:

        model_name = 'Logistic Regression'

    elif model == model_list[1]:

        model_name = 'Support Vector Classifier'

    elif model == model_list[2]:

        model_name = 'Decision Tree'

    elif model == model_list[3]:

        model_name = 'Random Forest'

    elif model == model_list[4]:

        model_name = 'K-Nearest Neighbours'

    elif model == model_list[5]:

        model_name = 'SGD Classifier'

    plt.plot(tpr, fpr, label = model_name)

    

plt.plot(x, y, color = 'black')

plt.legend(loc = 'lower right')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()
y_pred = grid_rf_best_model.predict(X_test).astype(int)
submission_df = pd.DataFrame({'PassengerId' : test['PassengerId'],

                              'Survived' : y_pred.tolist()})
submission_df.to_csv('submission.csv', header = True, index = False)