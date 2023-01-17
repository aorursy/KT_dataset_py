from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve, auc



import matplotlib.pyplot as plt

import pandas as pd

import numpy as np



import string



import warnings

warnings.filterwarnings('ignore')



import seaborn as sns

sns.set(style="darkgrid")
def combine_dataframes(train_data, test_data):

    """ Returns a combined dataframe containing testing and training datasets """

    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)



def divide_dataframes(combined_data):

    """ Returns the divided testing and training dataframes from the combined dataset """

    return combined_data.loc[:890], combined_data.loc[891:].drop(['Survived'], axis=1)



# import datasets and store within dataframes

train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')



# combine datasets so that operations can be performed on both simultaneously

combined_data = combine_dataframes(train_data, test_data)



# assign names to the dataframes

train_data.name = 'Training Dataset'

test_data.name = 'Testing Dataset'

combined_data.name = 'Combined Dataset'



dataframes = [train_data, test_data]
print('# of Training Entries: {}'.format(train_data.shape[0]))

print('# of Testing Entries: {}\n'.format(test_data.shape[0]))



print('Training X Shape: {}'.format(train_data.shape))

print('Training y Shape: {}\n'.format(train_data['Survived'].shape[0]))



print('Testing X Shape: {}'.format(test_data.shape))

print('Testing y Shape: {}\n'.format(test_data.shape[0]))



print(train_data.columns)

print(test_data.columns)
# print five random samples from the training dataset

train_data.sample(5)
# show descriptive statistics

train_data.describe()





# Arguments for each feature:

#    * For 'Survived', use "percentiles=[.61, .62]"

#    * For 'Parch', use "percentiles=[.75, .80]"

#    * For 'SibSp', use "percentiles=[.68, .69]"

#    * For 'Age' and 'Fare', use "percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]"
# show descriptive statistics for categorical features

train_data.describe(include=['O'])
# show missing values in training data

print("Missing Values in Training Data")

print("-" * 31)

print(train_data.isnull().sum())

print("_" * 31)
# show missing values in testing data

print("Missing Values in Testing Data")

print("-" * 30)

print(test_data.isnull().sum())

print("-" * 30)
# show correlations between 'age' and other features

correlation_data = combined_data.corr().abs().unstack().sort_values(kind='quicksort', ascending=False).reset_index()

correlation_data.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: "Correlation Coefficient"}, inplace=True)

correlation_data[correlation_data['Feature 1'] == 'Age']
# show median 'age' for each 'Pclass'/'Sex' group

age_by_pclass_sex = combined_data.groupby(['Sex', 'Pclass']).median()['Age']



for pclass in range(1, 4):

    for sex in ['female', 'male']:

        print("Median 'Age' of Pclass {} {}s: {}".format(pclass, sex, age_by_pclass_sex[sex][pclass]))



# show median 'age' for all passengers

print("\nMedian 'Age' of all passengers: {}".format(combined_data['Age'].median()))
# fill missing 'age' values with the shared median of the passengers 'pclass' and 'sex'

combined_data['Age'] = combined_data.groupby(['Sex', 'Pclass'])['Age'].apply(

    lambda x: x.fillna(x.median()))
# show the most frequently occurring port

print("Most frequent value for 'Embarked': {}".format(combined_data['Embarked'].dropna().mode()[0]))
# show entries with missing values for 'embarked'

combined_data[combined_data['Embarked'].isnull()]
# fill missing 'embarked' fields with 'S'

combined_data['Embarked'] = combined_data['Embarked'].fillna('S')
# show entry with missing 'fare' value

combined_data[combined_data['Fare'].isnull()]
# fill missing 'fare' field with the median 'fare' value for third-class males

combined_data['Fare'] = combined_data['Fare'].fillna(

                    combined_data.groupby(['Pclass', 'Parch', 'SibSp'])['Fare'].median()[3][0][0])
# create 'deck' feature by extracting the first letter from the 'cabin' feature

combined_data['Deck'] = combined_data['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')



# move passenger from 'deck' T to A

index = combined_data[combined_data['Deck'] == 'T'].index

combined_data.loc[index, 'Deck'] = 'A'
# create dataframe consisting of a 'count' for each combination of 'deck' and 'pclass'

deck_pclass = combined_data.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex',

        'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin', 'PassengerId', 'Ticket']).rename(

        columns={'Name': 'Count'}).transpose()



def get_pclass_dist(dataframe):

    """ Return the total passengers and class percentages per deck """

    

    # create dict. containing passenger class count for each deck

    deck_dict = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}, 'T': {}}

    

    # create an index of decks

    deck_index = dataframe.columns.levels[0]

    

    # count the number of passengers for each class, for each deck (ignore missing values)

    for deck in deck_index:

        for pclass in range(1, 4):

            try:

                count = dataframe[deck][pclass][0]

                deck_dict[deck][pclass] = count

            except KeyError:

                deck_dict[deck][pclass] = 0



    # convert deck_dict to dataframe and init. deck_percentages var

    deck_counts = pd.DataFrame(deck_dict)

    deck_percentages = {}

    

    # create dict. containing passenger class percentage for each deck

    for col in deck_counts.columns:

        deck_percentages[col] = [(count / deck_counts[col].sum()) * 100 for count in deck_counts[col]]

    

    print(deck_dict)

    

    return deck_dict, deck_percentages



def show_pclass_dist(percentages):

    """ Show the passenger count distribution per deck """

    

    # define class percentages and deck names

    deck_percentages = pd.DataFrame(percentages).transpose()

    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T')

    

    # bar parameters

    bar_count = np.arange(len(deck_names))

    bar_width = 0.85

    

    # extract each deck's class distribution

    pclass_1 = deck_percentages[0]

    pclass_2 = deck_percentages[1]

    pclass_3 = deck_percentages[2]

    

    # plot figure

    plt.figure(figsize=(20, 10))

    plt.bar(bar_count, pclass_1, color='#003f5c', edgecolor='white', width=bar_width, label='1st Class Passengers')

    plt.bar(bar_count, pclass_2, bottom=pclass_1, color='#bc5090', edgecolor='white', width=bar_width, label='2nd Class Passengers')

    plt.bar(bar_count, pclass_3, bottom=pclass_1+pclass_2, color='#ffa600', edgecolor='white', width=bar_width, label='3rd Class Passengers')

    

    # label figure

    plt.xlabel('Deck', size=15, labelpad=20)

    plt.ylabel('Passenger Class Percentage', size=15, labelpad=20)

    plt.xticks(bar_count, deck_names)

    plt.tick_params(axis='x', labelsize=15)

    plt.tick_params(axis='y', labelsize=15)

    

    # figure title and legend

    plt.legend(loc='upper left', bbox_to_anchor=(1,1), prop={'size': 15})

    plt.title('Passenger Class Distribution by Deck', size=18, y=1.05)

    

    plt.show()



deck_count, deck_percentage = get_pclass_dist(deck_pclass)

show_pclass_dist(deck_percentage)
# create dataframe consisting of the 'count' of the number of survivors/perished for each deck

deck_survived = combined_data.groupby(['Deck', 'Survived']).count().drop(columns=['Sex', 'Age',

    'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass', 'Cabin', 'PassengerId', 'Ticket']).rename(

        columns={'Name':'Count'}).transpose()



def get_survived_dist(dataframe):

    """ Return the no. of passengers and percentage of passengers that survived, per deck """

    

    # create dict. containing survival count for each deck

    deck_dict = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M':{}}

    

    # create an index of decks

    deck_index = dataframe.columns.levels[0]

    

    # count the number of passengers that survived for each deck

    for deck in deck_index:

        for survive in range(0, 2):

            deck_dict[deck][survive] = dataframe[deck][survive][0]

    

    # convert deck_dict to dataframe and init. deck_percentages var

    deck_counts = pd.DataFrame(deck_dict)

    deck_percentages = {}

    

    # create dict. containing survival percentage for each deck

    for col in deck_counts.columns:

        deck_percentages[col] = [(count / deck_counts[col].sum()) * 100 for count in deck_counts[col]]

    

    return deck_index, deck_percentages



def show_survived_dist(percentages):

    """ Show the survival distribution per deck """

    

    # define survival percentages and deck names

    deck_percentages = pd.DataFrame(percentages).transpose()

    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M')

    

    # bar parameters

    bar_count = np.arange(len(deck_names))

    bar_width = 0.85

    

    # extract survival rate distribution

    perished = deck_percentages[0]

    survived = deck_percentages[1]

    

    # plot figure

    plt.figure(figsize=(20, 10))

    plt.bar(bar_count, perished, color='#003f5c', edgecolor='white', width=bar_width, label='Perished')

    plt.bar(bar_count, survived, bottom=perished, color='#bc5090', edgecolor='white', width=bar_width, label='Survived')

    

    # label figure

    plt.xlabel('Deck', size=15, labelpad=20)

    plt.ylabel('Survival Percentage', size=15, labelpad=20)

    plt.xticks(bar_count, deck_names)

    plt.tick_params(axis='x', labelsize=15)

    plt.tick_params(axis='y', labelsize=15)

    

    # figure title and legend

    plt.legend(loc='upper left', bbox_to_anchor=(1,1), prop={'size': 15})

    plt.title('Survival Percentage by Deck', size=18, y=1.05)

    

    plt.show()



deck_count, deck_percentage = get_survived_dist(deck_survived)

show_survived_dist(deck_percentage)
# group decks based on correlations and similarities

combined_data['Deck'] = combined_data['Deck'].replace(['A', 'B', 'C'], 'ABC')

combined_data['Deck'] = combined_data['Deck'].replace(['D', 'E'], 'DE')

combined_data['Deck'] = combined_data['Deck'].replace(['F', 'G'], 'FG')



combined_data.drop('Cabin', axis=1)



# apply changes to both datasets

train_data, test_data = divide_dataframes(combined_data)
# count number of passengers that survived and perished from the training dataset

survived = train_data['Survived'].value_counts()[1]

perished = train_data['Survived'].value_counts()[0]



# calculate the survival/perish percentage

survived_percentage = survived / train_data.shape[0] * 100

perished_percentage = perished / train_data.shape[0] * 100



# print this information

print("{:.2f}% of passengers survived in the training dataset".format(survived_percentage))

print("{:.2f}% of passengers perished in the training dataset".format(perished_percentage))





# init. figure and plot

plt.figure(figsize=(10, 8))

sns.countplot(train_data['Survived'], palette=["#003f5c", "#bc5090"])



# configure labels

plt.xlabel('Outcome', size=15, labelpad=15)

plt.ylabel('Passenger Count', size=15, labelpad=15)

plt.xticks((0,1), ['Perished ({0:.2f}%)'.format(perished_percentage), 'Survived ({0:.2f}%)'.format(survived_percentage)])

plt.tick_params(axis='x', labelsize=13)

plt.tick_params(axis='y', labelsize=13)



plt.title("Survival Distribution on Training Dataset", size=15, y=1.05)



plt.show()
# create dataframe containing correlations between features, sorted by ascending

training_correlation = train_data.drop(['PassengerId'], axis=1).corr().abs().unstack().sort_values(

    kind='quicksort', ascending=False).reset_index()



# rename columns

training_correlation.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2",

                                     0: "Correlation Coefficient"}, inplace=True)



#  remove inverted instances of the same comparison (e.g. 'Pclass' & 'Fare' and 'Fare' & 'Pclass')

training_correlation.drop(training_correlation.iloc[1::2].index, inplace=True)



# drop entries where coefficient == 1.0 (comparing the same features)

training_correlation = training_correlation.drop(training_correlation[training_correlation['Correlation Coefficient'] == 1.0].index)



# show pairs with high correlation

training_correlation[training_correlation['Correlation Coefficient'] > 0.1]
# define figure

fig, axs = plt.subplots(nrows=1, figsize=(20,20))



# create heatmap

sns.heatmap(train_data.drop(['PassengerId'], axis=1).corr(), annot=True, square=True,

            cmap=(sns.light_palette("#bc5090", as_cmap=True)), annot_kws={'size': 14})



# label axis

axs.tick_params(axis='x', labelsize=20)

axs.tick_params(axis='y', labelsize=20)

axs.set_title("Correlations in Training Dataset", size=25)



plt.show()
# create dataframe containing correlations between features, sorted by ascending

testing_correlation = test_data.drop(['PassengerId'], axis=1).corr().abs().unstack().sort_values(

    kind='quicksort', ascending=False).reset_index()



# rename columns

testing_correlation.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2",

                                     0: "Correlation Coefficient"}, inplace=True)



#  remove inverted instances of the same comparison (e.g. 'Pclass' & 'Fare' and 'Fare' & 'Pclass')

testing_correlation.drop(testing_correlation.iloc[1::2].index, inplace=True)



# drop entries where coefficient == 1.0 (comparing the same features)

testing_correlation = testing_correlation.drop(testing_correlation[testing_correlation['Correlation Coefficient'] == 1.0].index)



# show pairs with high correlation

testing_correlation[testing_correlation['Correlation Coefficient'] > 0.1]



# show pairs with high correlation

testing_correlation[testing_correlation['Correlation Coefficient'] > 0.1]
# define figure

fig, axs = plt.subplots(nrows=1, figsize=(20,20))



# create heatmap

sns.heatmap(test_data.drop(['PassengerId'], axis=1).corr(), annot=True, square=True,

            cmap=(sns.light_palette("#bc5090", as_cmap=True)), annot_kws={'size': 14})



# label axis

axs.tick_params(axis='x', labelsize=20)

axs.tick_params(axis='y', labelsize=20)

axs.set_title("Correlations in Testing Dataset", size=25)



plt.show()
# store target and continuous variables

cont_ftrs = ['Age', 'Fare']

surv_psgrs = train_data['Survived'] == 1



# init. plot

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20,20))

plt.subplots_adjust(right=1.5)



# plot each feature against survival rate

for i, feature in enumerate(cont_ftrs):

    # distribution of survival amongst features

    sns.distplot(train_data[~surv_psgrs][feature], label='Perished', hist=True, color='#003f5c', ax=axs[0][i])

    sns.distplot(train_data[ surv_psgrs][feature], label='Survived', hist=True, color='#bc5090', ax=axs[0][i])

    

    # distribution of features in the dataset

    sns.distplot(train_data[feature], label='Training Dataset', hist=False, color='#003f5c', ax=axs[1][i])

    sns.distplot( test_data[feature], label='Testing Dataset',  hist=False, color='#bc5090', ax=axs[1][i])

    

    # set labels and change their size

    axs[0][i].set_xlabel('')

    axs[1][i].set_xlabel('')

    

    for j in range(2):

        axs[i][j].tick_params(axis='x', labelsize=20)

        axs[i][j].tick_params(axis='y', labelsize=20)

    

    # add legend and titles

    axs[0][i].legend(loc='upper right', prop={'size': 20})

    axs[1][i].legend(loc='upper right', prop={'size': 20})

    

    axs[0][i].set_title('Distribution of Survival in {}'.format(feature), size=20, y=1.05)



axs[1][0].set_title('Distribution of {}'.format('Age'), size=20, y=1.05)

axs[1][1].set_title('Distribution of {}'.format('Fare'), size=20, y=1.05)

    

plt.show()
# store categorical features

cat_ftrs = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp', 'Deck']



# define figure

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20,20))

plt.subplots_adjust(right=1.5, top=1.25)



# plot each feature against survival

for i, feature in enumerate(cat_ftrs, 1):

    plt.subplot(2, 3, i)

    sns.countplot(x=feature, hue='Survived', palette=["#003f5c", "#bc5090"], data=train_data)

    

    # set up labels

    plt.xlabel('{}'.format(feature), size=20, labelpad=15)

    plt.ylabel('Passenger Count', size=20, labelpad=15)

    plt.tick_params(axis='x', labelsize=20)

    plt.tick_params(axis='y', labelsize=20)

    

    # add legend and title

    plt.legend(['Perished', 'Survived'], loc='upper center', prop={'size': 18})

    plt.title('Survival Rate wrt {}'.format(feature), size=20, y=1.05)



plt.show()
# bin 'Fare'

combined_data['Fare'] = pd.qcut(combined_data['Fare'], 13)
# define figure

fig, axs = plt.subplots(figsize=(22,9))

sns.countplot(x='Fare', hue='Survived', palette=["#003f5c", "#bc5090"], data=combined_data)



# set labels

plt.xlabel('Fare', size=15, labelpad=20)

plt.ylabel('Passenger Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=10)

plt.tick_params(axis='y', labelsize=15)



# set legend and title

plt.legend(['Perished', 'Survived'], loc='upper right', prop={'size': 15})

plt.title('Survival Count wrt {}'.format('Fare', size=15, y=1.05))



plt.show()
# bin 'Age'

combined_data['Age'] = pd.qcut(combined_data['Age'], 10)
# define figure

fig, axs = plt.subplots(figsize=(22,9))

sns.countplot(x='Age', hue='Survived', palette=["#003f5c", "#bc5090"], data=combined_data)



# set labels

plt.xlabel('Age', size=15, labelpad=20)

plt.ylabel('Passenger Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=10)

plt.tick_params(axis='y', labelsize=15)



# set legend and title

plt.legend(['Perished', 'Survived'], loc='upper right', prop={'size': 15})

plt.title('Survival Count wrt {}'.format('Fare', size=15, y=1.05))



plt.show()
# create 'FamilySize' feature

combined_data['FamilySize'] = combined_data['SibSp'] + combined_data['Parch'] + 1
# define figure

fig, axs = plt.subplots(figsize=(20,20), ncols=2, nrows=2)

plt.subplots_adjust(right=1.5)



# plot family size value counts before grouping

sns.barplot(x=combined_data['FamilySize'].value_counts().index, y=combined_data['FamilySize'].value_counts().values, palette=["#003f5c", "#bc5090"], ax=axs[0][0])

sns.countplot(x='FamilySize', hue='Survived', palette=["#003f5c", "#bc5090"], data=combined_data, ax=axs[0][1])



# set titles

axs[0][0].set_title('Family Size Feature Value Counts (Before Grouping)', size=20, y=1.05)

axs[0][1].set_title('Survival Count wrt Family Size (Before Grouping)', size=20, y=1.05)



# map 'FamilySize' into groups

family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}

combined_data['GroupedFamilySize'] = combined_data['FamilySize'].map(family_map)



# plot family size value counts after grouping

sns.barplot(x=combined_data['GroupedFamilySize'].value_counts().index, y=combined_data['GroupedFamilySize'].value_counts().values, palette=["#003f5c", "#bc5090"], ax=axs[1][0])

sns.countplot(x='GroupedFamilySize', hue='Survived', palette=["#003f5c", "#bc5090"], data=combined_data, ax=axs[1][1])



# set titles

axs[1][0].set_title('Family Size Feature Value Counts (After Grouping)', size=20, y=1.05)

axs[1][1].set_title('Survival Count wrt Family Size (After Grouping)', size=20, y=1.05)



# set labels and legend

for i in range(2):

    axs[i][1].legend(['Perished', 'Survived'], loc='upper right', prop={'size': 20})

    

    for j in range(2):

        axs[i][j].tick_params(axis='x', labelsize=20)

        axs[i][j].tick_params(axis='y', labelsize=20)

        axs[i][j].set_xlabel('')

        axs[i][j].set_ylabel('')



plt.show()
# create 'TicketFreq' feature

combined_data['TicketFreq'] = combined_data.groupby('Ticket')['Ticket'].transform('count')
# define figure

fig, axs = plt.subplots(figsize=(12, 9))



# plot survival rate based on ticket frequency

sns.countplot(x='TicketFreq', hue='Survived', palette=["#003f5c", "#bc5090"], data=combined_data)



# set labels

plt.xlabel('Ticket Frequency', size=15, labelpad=20)

plt.ylabel('Passenger Count', size=15, labelpad=20)

plt.tick_params(axis='x', labelsize=15)

plt.tick_params(axis='y', labelsize=15)



# set title and legend

plt.legend(['Perished', 'Survived'], loc='upper right', prop={'size': 15})

plt.title('Suruvival Count wrt {}'.format('TicketFreq'), size=15, y=1.05)



plt.show()
# create 'Title' feature, extracted from 'Name'

combined_data['Title'] = combined_data['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]



# create 'IsMarried' feature, where 'Title'=='Mrs'

combined_data['IsMarried'] = 0

combined_data['IsMarried'].loc[combined_data['Title'] == 'Mrs'] = 1
# define figure

fig, axs = plt.subplots(nrows=2, figsize=(20,20))



# plot title value counts before grouping

sns.barplot(x=combined_data['Title'].value_counts().index, y=combined_data['Title'].value_counts().values, palette=["#003f5c", "#bc5090"], ax=axs[0])



# set labels

axs[0].tick_params(axis='x', labelsize=10)

axs[1].tick_params(axis='x', labelsize=15)



for i in range(2):

    axs[i].tick_params(axis='y', labelsize=15)



# set title

axs[0].set_title('Title Value Counts (Before Grouping)', size=20, y=1.05)



# replace female titles with 'Miss/Mrs/Ms'

combined_data['Title'] = combined_data['Title'].replace(['Miss', 'Mrs', 'Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')



# replace unique titles with 'Dr/Military/Noble/Clergy'

combined_data['Title'] = combined_data['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')



# plot title value counts after grouping

sns.barplot(x=combined_data['Title'].value_counts().index, y=combined_data['Title'].value_counts().values, palette=["#003f5c", "#bc5090"], ax=axs[1])



# set title

axs[1].set_title('Title Value Counts (After Grouping)', size=20, y=1.05)



plt.show()
def extractSurname(data):

    """ Extract surnames from passenger entries using the 'Name' feature """

    

    # store family names

    families = []

    

    for i in range(len(data)):

        # acquire passenger name

        name = data.iloc[i]

        

        # remove brackets from name

        if '(' in name:

            name_no_bracket = name.split('(')[0]

        else:

            name_no_bracket = name

        

        

        # the surname appears at the start of the passenger names, followed by a comma

        family = name_no_bracket.split(',')[0]

        title = name_no_bracket.split(',')[1].strip().split(' ')[0]

        

        # remove punctuation

        for c in string.punctuation:

            family = family.replace(c, '').strip()

        

        families.append(family)

    

    return families



# create 'FamilyName' feature

combined_data['FamilyName'] = extractSurname(combined_data['Name'])



# split combined data back into train/test datasets

train_data, test_data = divide_dataframes(combined_data)

dataframes = [train_data, test_data]
# get a list of family names that occur in both testing and training datasets

non_unique_families = [x for x in train_data['FamilyName'].unique() if x in test_data['FamilyName'].unique()]



# get each family's median survival rate in the training dataset

family_survival_rate = train_data.groupby('FamilyName')['Survived', 'FamilyName', 'FamilySize'].median()



# store the median survival rate for each family in the training dataset

family_rates = {}



# store the median survival rate of each family that has more than one member across both datasets

for i in range(len(family_survival_rate)):

    if family_survival_rate.index[i] in non_unique_families and family_survival_rate.iloc[i, 1] > 1:

        family_rates[family_survival_rate.index[i]] = family_survival_rate.iloc[i, 0]
# calculate the mean survival rate across all passengers in training set

mean_survival_rate = np.mean(train_data['Survived'])



# store family survival rates

train_family_survival_rate = []

train_family_survival_rate_NA = []

test_family_survival_rate = []

test_family_survival_rate_NA = []



for i in range(len(train_data)):

    # if the passenger's family name occurs in 'family_rates' (and thefore the training set)

    if train_data['FamilyName'][i] in family_rates:

        # store the family survival rate

        train_family_survival_rate.append(family_rates[train_data['FamilyName'][i]])

        # mark that the family does exist in the training set

        train_family_survival_rate_NA.append(1)

    else:

        # store mean survival rate

        train_family_survival_rate.append(mean_survival_rate)

        # mark that the family doesn't exist in the training set

        train_family_survival_rate_NA.append(0)



for i in range(len(test_data)):

    # if the passenger's family name occurs in 'family rates' (and therefore the training set)

    if test_data['FamilyName'].iloc[i] in family_rates:

        # store the family survival rate

        test_family_survival_rate.append(family_rates[test_data['FamilyName'].iloc[i]])

        # mark that the family does exist in the training set

        test_family_survival_rate_NA.append(1)

    else:

        # store mean survival rate

        test_family_survival_rate.append(mean_survival_rate)

        # mark that the family doesn't exist in the training set

        test_family_survival_rate_NA.append(0)



# add these new features to the datasets

train_data['FamilySurvivalRate'] = train_family_survival_rate

train_data['FamilySurvivalRateNA'] = train_family_survival_rate_NA

test_data['FamilySurvivalRate'] = test_family_survival_rate

test_data['FamilySurvivalRateNA'] = test_family_survival_rate_NA
# get a list of tickets that occur in both training and testing datasets

non_unique_tickets = [x for x in train_data['Ticket'].unique() if x in test_data['Ticket'].unique()]



# get each ticket's median survival rate in the training dataset

ticket_survival_rate = train_data.groupby('Ticket')['Survived', 'Ticket', 'TicketFreq'].median()



# store the median survival rate for each ticket in the training dataset

ticket_rates = {}



# store the median survival rate of each ticket that has more than one member across both datasets

for i in range(len(ticket_survival_rate)):

    if ticket_survival_rate.index[i] in non_unique_tickets and ticket_survival_rate.iloc[i, 1] > 1:

        ticket_rates[ticket_survival_rate.index[i]] = ticket_survival_rate.iloc[i, 0]
# store ticket survival rates

train_ticket_survival_rate = []

train_ticket_survival_rate_NA = []

test_ticket_survival_rate = []

test_ticket_survival_rate_NA = []



for i in range(len(train_data)):

    # if the passenger's ticket occurs in 'ticket_rates' (and thefore the training set)

    if train_data['Ticket'][i] in ticket_rates:

        # store the ticket survival rate

        train_ticket_survival_rate.append(ticket_rates[train_data['Ticket'][i]])

        # mark that the ticket does exist in the training set

        train_ticket_survival_rate_NA.append(1)

    else:

        # store mean survival rate

        train_ticket_survival_rate.append(mean_survival_rate)

        # mark that the ticket doesn't exist in the training set

        train_ticket_survival_rate_NA.append(0)



for i in range(len(test_data)):

    # if the passenger's ticket occurs in 'ticket_rates' (and thefore the training set)

    if test_data['Ticket'].iloc[i] in ticket_rates:

        # store the ticket survival rate

        test_ticket_survival_rate.append(ticket_rates[test_data['Ticket'].iloc[i]])

        # mark that the ticket does exist in the training set

        test_ticket_survival_rate_NA.append(1)

    else:

        # store mean survival rate

        test_ticket_survival_rate.append(mean_survival_rate)

        # mark that the ticket doesn't exist in the training set

        test_ticket_survival_rate_NA.append(0)



# add these new features to the datasets

train_data['TicketSurvivalRate'] = train_ticket_survival_rate

train_data['TicketSurvivalRateNA'] = train_ticket_survival_rate_NA

test_data['TicketSurvivalRate'] = test_ticket_survival_rate

test_data['TicketSurvivalRateNA'] = test_ticket_survival_rate_NA
for dataframe in [train_data, test_data]:

    dataframe['SurvivalRate'] = (dataframe['TicketSurvivalRate'] + dataframe['FamilySurvivalRate']) / 2

    dataframe['SurvivalRateNA'] = (dataframe['TicketSurvivalRateNA'] + dataframe['FamilySurvivalRateNA']) / 2
# list non-numeric features

non_numeric_features = ['Embarked', 'Sex', 'Deck', 'Title', 'GroupedFamilySize', 'Age', 'Fare']



# label encode these features

for dataframe in dataframes:

    for feature in non_numeric_features:

        dataframe[feature] = LabelEncoder().fit_transform(dataframe[feature])
# list categorical features

categorical_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'GroupedFamilySize']

encoded_features = []



# encode these features

for dataframe in dataframes:

    for feature in categorical_features:

        # one-hot encode features

        encoded_feature = OneHotEncoder().fit_transform(dataframe[feature].values.reshape(-1, 1)).toarray()

        # get the number of unique values in the feature

        n = dataframe[feature].nunique()

        # create new columns for each possible value for the feature

        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]

        # create dataframe containing these new columns

        encoded_dataframe = pd.DataFrame(encoded_feature, columns=cols)

        # ensure new dataframe rows share indexes with main dataframe

        encoded_dataframe.index = dataframe.index

        # store features

        encoded_features.append(encoded_dataframe)



# merge new features with training/testing data

train_data = pd.concat([train_data, *encoded_features[:6]], axis=1)

test_data = pd.concat([test_data, *encoded_features[6:]], axis=1)
train_columns = ['Title', 'Cabin', 'Deck', 'Embarked', 'FamilyName', 'FamilySize', 'GroupedFamilySize', 'Survived', 'Name',

               'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'TicketSurvivalRate',

               'FamilySurvivalRate', 'TicketSurvivalRateNA', 'FamilySurvivalRateNA']



test_columns = ['Title', 'Cabin', 'Deck', 'Embarked', 'FamilyName', 'FamilySize', 'GroupedFamilySize', 'Name',

               'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'TicketSurvivalRate',

               'FamilySurvivalRate', 'TicketSurvivalRateNA', 'FamilySurvivalRateNA']
# scale training/testing data

X_train = StandardScaler().fit_transform(train_data.drop(columns=train_columns))

y_train = train_data['Survived'].values

X_test = StandardScaler().fit_transform(test_data.drop(columns=test_columns))
model = RandomForestClassifier(criterion='gini',

                              n_estimators=1100,

                              max_depth=5,

                              min_samples_split=4,

                              min_samples_leaf=5,

                              max_features='auto',

                              oob_score=True,

                              random_state=42,

                              n_jobs=-1,

                              verbose=1)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
def plot_roc_curve(fper, tper):

    fig, axs = plt.subplots(figsize=(15,15))



    axs.plot(fper, tper, color='#bc5090', label='ROC', lw=2, alpha=0.8)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='#003f5c', alpha=0.8, label='Random Guessing')

    axs.set_xlabel('False Positive Rate', size=15, labelpad=20)

    axs.set_ylabel('True Positive Rate', size=15, labelpad=20)

    axs.tick_params(axis='x', labelsize=15)

    axs.tick_params(axis='y', labelsize=15)

    axs.set_xlim([-0.05, 1.05])

    axs.set_ylim([-0.05, 1.05])

    

    axs.set_title('Receiver Operating Characteristic (ROC) Curve', size=20, y=1.02)

    axs.legend(loc='lower right', prop={'size': 13})

    

    plt.show()



probs = model.predict_proba(X_train)  

probs = probs[:, 1]  

fper, tper, thresholds = roc_curve(y_train, probs) 

plot_roc_curve(fper, tper)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions.astype(int)})

output.to_csv('my_submission.csv', index=False)



output.head()