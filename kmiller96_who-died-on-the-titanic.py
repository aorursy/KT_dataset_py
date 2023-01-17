# IMPORT STATEMENTS.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualisation

import matplotlib.pyplot as plt # data visualisation



import random # Used to sample survival.



# DEFINE GLOBALS.

NUM_OF_ROLLS = 3



df = pd.read_csv("../input/train.csv", index_col=0)

df.head()
# First visualise the general case (i.e. no considerations)

total, survivors = df.shape[0], df[df.Survived==1].shape[0]

survival_rate = float(survivors)/float(total)*100



f, ax = plt.subplots(figsize=(7, 7))

ax.set_title("Proportion of People Who Died On The Titanic")

ax.pie(

    [survival_rate, 100-survival_rate], 

    autopct='%1.1f%%', 

    labels=['Survived', 'Died']

)

None # Removes console output
sns.set_style('white')
f, ax = plt.subplots(figsize=(8, 8))

sns.barplot(

    ax=ax,

    x='Pclass',

    y='Survived',

    hue='Sex',

    data=df,

    capsize=0.05

)

ax.set_title("Survival By Gender and Ticket Class")

ax.set_ylabel("Survival (%)")

ax.set_xlabel("")

ax.set_xticklabels(["First Class", "Second Class", "Third Class"])

None # Suppress console output
sns.set_style("whitegrid")
f, ax = plt.subplots(figsize=(12, 5))

ax = sns.distplot(

    df.Age.dropna().values, bins=range(0, 81, 1), kde=False,

    axlabel='Age (Years)', ax=ax

)
total, classes_count = float(df['Pclass'].shape[0]), df['Pclass'].value_counts()

proportions = list(map(lambda x: classes_count.loc[x]/total*100, [1, 2, 3]))



f, ax = plt.subplots(figsize=(8, 8))

ax.set_title('Proportion of Passengers By Class')

ax.pie(proportions, autopct='%1.1f%%', labels=['First Class', 'Second Class', 'Third Class'])

None  # Removes console output
def probability(df, key_list):

    """Finds the probability of surviving based on the parameters passed in key_list.

    

    The key_list input is structured like so:

        [Ticket Class, Sex]

    

    So for example, an input could be [1, 'male'].

    """

    pclass, sex = key_list

    filtered_df = df[(df.Sex == sex) & (df.Pclass == pclass)]

    return filtered_df['Survived'].mean()



##############################################################################################



sexes = df.Sex.unique()

ticket_classes = df.Pclass.unique()



probability_dict = dict()

for x in ticket_classes:

    for y in sexes:

        key = [x, y]

        probability_dict[str(key)] = probability(df, key)

        

##############################################################################################



def make_guesses(df):

    """Makes guesses on if the passengers survived or died."""

    guesses = list()

    for passenger_index, row in df.iterrows():

        # Find if the passenger survived.

        survival_key = [row.Pclass, row.Sex]

        survival_odds = probability_dict[str(survival_key)]

        survived_rolls = list(map(lambda x: random.random() <= survival_odds, range(NUM_OF_ROLLS)))

        survived = sum(survived_rolls) > NUM_OF_ROLLS/2



        # Add the result to the guesses

        guesses.append(survived)

    return guesses



##############################################################################################



df['Guess'] = make_guesses(df)

df['CorrectGuess'] = df.Guess == df.Survived

df.head()
df.CorrectGuess.mean()
results = list()

for ii in range(10**2):

    guesses = make_guesses(df)

    correct_guesses = (df.Survived == guesses)

    results.append(correct_guesses.mean())

sns.distplot(results, kde=False)

None
df.drop('Guess', axis=1, inplace=True)

df.drop('CorrectGuess', axis=1, inplace=True)

df.head()
f, ax = plt.subplots(figsize=(12, 8))

sns.distplot(

    df.Age.dropna().values, bins=range(0, 81, 1), kde=False,

    axlabel='Age (Years)', ax=ax

)

sns.distplot(

    df[(df.Survived == 1)].Age.dropna().values, bins=range(0, 81, 1), kde=False,

    axlabel='Age (Years)', ax=ax

)

None # Suppress console output.
f, ax = plt.subplots(2, figsize=(12, 8))

# Plot both sexes on different axes

for ii, sex in enumerate(['male', 'female']):

    sns.distplot(

        df[df.Sex == sex].Age.dropna().values, bins=range(0, 81, 1), kde=False,

        axlabel='Age (Years)', ax=ax[ii]

    )

    sns.distplot(

        df[(df.Survived == 1)&(df.Sex == sex)].Age.dropna().values, bins=range(0, 81, 1), kde=False,

        axlabel='Age (Years)', ax=ax[ii]

    )



None # Suppress console output.
f, ax = plt.subplots(2, figsize=(12, 8))

# Plot both sexes on different axes

for ii, sex in enumerate(['male', 'female']):

    sns.distplot(

        df[df.Sex == sex].Age.dropna().values, bins=range(0, 81, 5), kde=False,

        axlabel='Age (Years)', ax=ax[ii]

    )

    sns.distplot(

        df[(df.Survived == 1)&(df.Sex == sex)].Age.dropna().values, bins=range(0, 81, 5), kde=False,

        axlabel='Age (Years)', ax=ax[ii]

    )



None # Suppress console output.
survival_rates, survival_labels = list(), list()

for x in range(0, 90+5, 5):

    aged_df = df[(x <= df.Age)&(df.Age <= x+5)]

    survival_rate = aged_df['Survived'].mean()

    survival_rate = 0.5 if (survival_rate == 0.0 or survival_rate == 1.0) else survival_rate

    

    survival_rates.append(survival_rate if (survival_rate != 0.0 or survival_rate != 1.0) else 0.5)

    survival_labels.append('(%i, %i]' % (x, x+5))



f, ax = plt.subplots(figsize=(12, 8))

ax = sns.barplot(x=survival_labels, y=survival_rates, ax=ax)

ax.set_xticklabels(ax.get_xticklabels(), rotation=50)

None # Suppress console output
def getProbability(passengerId, df):

    """

    Finds the weighted probability of surviving based on the passenger's parameters.

    

    This function finds the passenger's information by looking for their id in the dataframe

    and extracting the information that it needs. Currently the probability is found using a

    weighted mean on the following parameters:

        - Pclass: Higher the ticket class the more likely they will survive.

        - Sex: Women on average had a higher chance of living.

        - Age: Infants and older people had a greater chance of living.

    """

    

    passenger = df.loc[passengerId]

    

    # Survival rate based on sex and ticket class.

    bySexAndClass = df[

        (df.Sex == passenger.Sex) & 

        (df.Pclass == passenger.Pclass)

    ].Survived.mean()

    

    # Survival rate based on sex and age.

    byAge = df[

        (df.Sex == passenger.Sex) & 

        ((df.Age//5-1)*5 <= passenger.Age) & (passenger.Age <= (df.Age//5)*5)

    ].Survived.mean()

    

    # Find the weighting for each of the rates.

    parameters = [bySexAndClass, byAge]

    rolls = [5, 4]  # Roll numbers are hardcoded until I figure out the weighting system

    

    probabilities = []

    for Nrolls, prob in zip(rolls, parameters):

        for _ in range(Nrolls):

            probabilities += [prob]

    return probabilities



##############################################################################################



def make_guesses(df):

    """Makes guesses on if the passengers survived or died."""

    guesses = list()

    for passenger_index, _row in df.iterrows():

        # Find if the passenger survived.

        survival_odds = getProbability(passenger_index, df)

        roll_outcomes = []

        for prob in survival_odds:

            roll_outcomes += [random.random() <= prob]

        survived = sum(roll_outcomes) > len(roll_outcomes)/2



        # Add the result to the guesses

        guesses.append(survived)

    return guesses



##############################################################################################



df['Guess'] = make_guesses(df)

df['CorrectGuess'] = df.Guess == df.Survived

df.head()
df.CorrectGuess.mean()
results = list()

for ii in range(10**2):

    guesses = make_guesses(df)

    correct_guesses = (df.Survived == guesses)

    results.append(correct_guesses.mean())

    

    if ii % 10 == 0: print("%i/%i" % (ii, 10**2))

sns.distplot(results, kde=False)

None