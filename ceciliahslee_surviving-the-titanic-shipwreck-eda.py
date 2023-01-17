# Adjust the width of the jupyter notebook container

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))
import numpy as np

import pandas as pd



import matplotlib

import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle

import seaborn as sns



from __future__ import division



%matplotlib inline
titanic_df = pd.read_csv('titanic-data.csv')
sns.set(font_scale=.85)

sns.set_style('darkgrid', 

              {'axes.labelcolor': '.3', 

               'text.color': '.3', 

               'xtick.color': '.3',

               'ytick.color': '.3',

               'axes.facecolor': '#eaeaed'})

plt.rcParams['patch.linewidth'] = 0
# Label percentage on top of each bar

def label_bar_percent(ax):

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x() + p.get_width() / 2.,

                height / 2 ,

                '{:.0%}'.format(height / len(titanic_df)),

                fontsize=8.5, color='white', alpha=.8,

                ha='center', va='center')

        

# Label frequency count on top of each bar

def label_bar_freq(ax, adjust_height):

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x() + p.get_width() / 2.,

                height + adjust_height,

                '%d' % int(height),

                fontsize=8.5, alpha=.5,

                ha='center', va='center')
titanic_df.info()
titanic_df.head(5)
titanic_df.describe()
titanic_df.describe(include=['O'])
titanic_df.isnull().sum()
fig = plt.figure(figsize=[11,8])



# Plot subplots

ax1, ax2, ax3, ax4, ax5, ax6, ax7 = (plt.subplot2grid((3,3), (0,0)), 

                                     plt.subplot2grid((3,3), (0,1)),

                                     plt.subplot2grid((3,3), (0,2)),

                                     plt.subplot2grid((3,3), (1,0), colspan=2),

                                     plt.subplot2grid((3,3), (1,2)),

                                     plt.subplot2grid((3,3), (2,0), colspan=2),

                                     plt.subplot2grid((3,3), (2,2)))

# Set common title and ylabel

fig.suptitle('Distribution of Features', fontsize=9.5)

plt.subplots_adjust(top=.95)

fig.text(0.09, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=9)



## ax1: Survived 

sns.countplot(x="Survived", data=titanic_df, ax=ax1)

ax1.set_ylabel('')

ax1.set_xticks([0, 1])

ax1.set_xticklabels(('No', 'Yes'))



## ax2: Pclass 

sns.countplot(x="Pclass", data=titanic_df, ax=ax2)

ax2.set_xlabel('Ticket Class')

ax2.set_ylabel('')



## ax3: Sex 

sns.countplot(x="Sex", data=titanic_df, ax=ax3)

ax3.set_ylabel('')

ax3.set_xticks([0, 1])

ax3.set_xticklabels(('Male', 'Female'))



## ax4: Age 

ax4.hist(titanic_df.Age.dropna(), bins=17) # We temporarily drop the missing values at this stage.

ax4.set_xlabel('Age')

ax4.set_xticks(np.arange(0, 81, 5))



## ax5: SipSp, Parch 

family_vstack = np.vstack([titanic_df.SibSp, titanic_df.Parch]).T

ax5.hist(family_vstack, bins=np.arange(10)-0.5, normed=False,

         label=['Siblings or spouses', 'Parents or children'])

ax5.legend(loc='upper right')

ax5.set_xlabel('Family Aboard')

ax5.set_xticks(np.arange(0, 9))



## ax6: Fare 

ax6.hist(titanic_df.Fare, bins=30)

ax6.set_xlabel('Fare')

ax6.set_xticks(np.arange(0, 601, 50))



## ax7: Embarked 

sns.countplot(x="Embarked", data=titanic_df, ax=ax7)

ax7.set_xlabel('Port of Embarktion')

ax7.set_ylabel('')

ax7.set_xticks([0, 1, 2])

        

# Apply labeling functions to axes

apply_label_bar_percent = [label_bar_percent(ax) for ax in [ax1, ax2, ax3, ax7]]

label_bar_freq(ax4, 3)

label_bar_freq(ax5, 16)

label_bar_freq(ax6, 12)

        

plt.show()
fig = plt.figure(figsize=(6,4))

p_heatmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(titanic_df.drop('PassengerId',axis=1).corr(), vmax=.6, square=True, annot=True, cmap=p_heatmap, annot_kws={'size': 9})

plt.title('Pearson\'s Correlation of Features', fontsize=9.5)

plt.show()
fig, ax = plt.subplots(figsize=[4,2.2])

ax = sns.regplot(x='Pclass', y='Fare', data=titanic_df, y_jitter=.2, x_jitter=.2)

ax.set_title('Correlation between Ticket Class and Fare', fontsize=9.5)

ax.set_xlabel('Ticket Class')

ax.set_ylabel('Fare')

ax.set_xticks(np.arange(1, 4, 1))

ax.set_yticks(np.arange(0, 601, 100))

plt.show()
fig, ax = plt.subplots(figsize=[7.5,2])

g = sns.boxplot(x='Fare', y='Pclass', data=titanic_df, whis=1.5, linewidth=1.3, orient='h')

g = sns.stripplot(x='Fare', y='Pclass', data=titanic_df, jitter=.2, color='.2', size=1.8, alpha=.3, orient='h')

g.set_title('Distribution of Fare by Ticket Class', fontsize=9.5)

ax.set_xlabel('Fare')

ax.set_ylabel('Ticket Class')

plt.show()
titanic_df[titanic_df.Pclass == 1].Fare.max() / titanic_df[titanic_df.Pclass == 1].Fare.std()
fig, ax = plt.subplots(figsize=[4,2.5])

g = sns.boxplot(x='Pclass', y='Age', data=titanic_df, whis=np.inf, linewidth=1.3)

g = sns.stripplot(x='Pclass', y='Age', data=titanic_df, jitter=.2, color='.2', size=1.8, alpha=.3)

g.set_title('Distribution of Age by Ticket Class', fontsize=9.5)

g.set_xlabel('Ticket Class')

plt.show()
g = sns.factorplot(x='Sex', col='Pclass', data=titanic_df, kind='count', size=2.2, aspect=1.2)

g.set_axis_labels('', 'Frequency')

g.fig.suptitle('Distribution of Gender by Ticket Class', y=1.01, fontsize=9.5)

g.fig.text(.5, .1, 'Sex', ha='center', fontsize=9.5)

plt.show()
fig, ax = plt.subplots(figsize=[5.5,2.5])

sns.countplot(x='Pclass', hue='Embarked', data=titanic_df, ax=ax)

ax.set_title('Distribution of Port of Embarktion by Ticket Class', fontsize=9.5)

ax.set_xlabel('Ticket Class')

ax.set_ylabel('Frequency')

label_bar_freq(ax, 10)

plt.show()
titanic_df.Name.head(8)
titanic_df['Title'] = titanic_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
titanic_df.Title.head()
titanic_df.Title.isnull().sum()
fig, ax = plt.subplots(figsize=[11,2])

ax = sns.countplot('Title', data=titanic_df)

ax.set_title('Distribution of Titles', fontsize=9.5)

ax.set_ylabel('Frequency')

label_bar_freq(ax, 18)

plt.show()
Title_Dictionary = {"Capt":       "Officer",

                    "Col":        "Officer",

                    "Major":      "Officer",

                    "Jonkheer":   "Royalty",

                    "Don":        "Royalty",

                    "Sir" :       "Royalty",

                    "Dr":         "Officer",

                    "Rev":        "Officer",

                    "the Countess":"Royalty",

                    "Dona":       "Royalty",

                    "Mme":        "Mrs",

                    "Mlle":       "Miss",

                    "Ms":         "Mrs",

                    "Mr" :        "Mr",

                    "Mrs" :       "Mrs",

                    "Miss" :      "Miss",

                    "Master" :    "Master",

                    "Lady" :      "Royalty"}

    

titanic_df['Title'] = titanic_df.Title.map(Title_Dictionary)
fig, ax = plt.subplots(figsize=[6.5,2])

ax = sns.countplot('Title', data=titanic_df)

ax.set_title('Distribution of Titles', fontsize=9.5)

ax.set_ylabel('Frequency')

label_bar_freq(ax, 18)

plt.show()
fig, ax = plt.subplots(figsize=[6.5,2])



Mr_Pclass, Mrs_Pclass, Miss_Pclass, Master_Pclass, Royalty_Pclass, Officer_Pclass = (pd.crosstab(titanic_df.Title, titanic_df.Pclass).loc[v] 

                                                                                     for v in ['Mr', 'Mrs', 'Miss', 'Master', 'Royalty', 'Officer'])



Y = np.arange(1, 4, 1)



ax.barh(Y, Mr_Pclass, color = '#5975A4', align='center')

ax.barh(Y, Mrs_Pclass, color = '#5F9E6E', align='center', left = Mr_Pclass)

ax.barh(Y, Miss_Pclass, color = '#B55D60', align='center', left = Mr_Pclass + Mrs_Pclass)

ax.barh(Y, Master_Pclass, color = '#857AAA', align='center', left = Mr_Pclass + Mrs_Pclass + Miss_Pclass)

ax.barh(Y, Royalty_Pclass, color = '#C1B37F', align='center', left = Mr_Pclass + Mrs_Pclass + Miss_Pclass + Master_Pclass)

ax.barh(Y, Officer_Pclass, color = '#71AEC0', align='center', left = Mr_Pclass + Mrs_Pclass + Miss_Pclass + Master_Pclass + Royalty_Pclass)



ax.set_title('Distribution of Titles in Ticket Class', fontsize=9.5)

ax.set_yticks([1, 2, 3])

ax.set_xlabel('Frequency')

ax.set_ylabel('Ticket Class')

ax.legend(['Mr', 'Mrs', 'Miss', 'Master', 'Royalty', 'Officier'], loc='best')



plt.show()
fig, ax = plt.subplots(figsize=[4.5,2.5])

sns.boxplot(x='Title', y='Age', data=titanic_df, ax=ax, whis=np.inf, linewidth=1.2)

sns.stripplot(x='Title', y='Age', data=titanic_df, ax=ax, jitter=.2, color='.2', size=1.8, alpha=.3)

ax.set_title('Distribution of Age by Title', fontsize=9.5)

plt.show()
titanic_df[titanic_df.Title == 'Master'].Age.max()
titanic_df['Lastname'] = titanic_df.Name.str.extract('([A-Za-z]+)\,', expand=False)
titanic_df.Lastname.head()
titanic_df.Lastname.isnull().sum()
titanic_df.Lastname.describe()
titanic_df[titanic_df.Lastname == 'Andersson']
titanic_df.groupby('Lastname').PassengerId.count().sort_values(ascending=False)[:8]
titanic_df[titanic_df.Lastname == 'Skoog']
fig, ax = plt.subplots(figsize=[3,2.5])

sns.boxplot(x='Sex', y='Age', data=titanic_df, ax=ax, whis=np.inf, linewidth=1.2)

sns.stripplot(x='Sex', y='Age', data=titanic_df, ax=ax, jitter=.2, color='.2', size=1.8, alpha=.3)

ax.set_title('Distribution of Age by Sex', fontsize=9.5)

plt.show()
fig, ax = plt.subplots(figsize=[8.5,2])

ax = sns.distplot(titanic_df.Fare[titanic_df.Sex == 'male'].dropna())

ax = sns.distplot(titanic_df.Fare[titanic_df.Sex == 'female'].dropna())

ax.set_title('Distribution of Fare by Sex', fontsize=9.5)

ax.legend(['Male', 'Female'])

plt.show()
fig, ax = plt.subplots(figsize=[5,2])

sns.countplot(x='Sex', hue='Embarked', data=titanic_df, ax=ax)

ax.set_title('Distribution of Port of Embarktion by Sex', fontsize=9.5)

ax.set_ylabel('Frequency')

label_bar_freq(ax, 10)

plt.show()
fig, ax = plt.subplots(figsize=[4,2.5])

ax = sns.regplot(x='SibSp', y='Parch', data=titanic_df, y_jitter=.2, x_jitter=.2)

ax.set_title('Correlation between Siblings or Spouses and \n Parents or Children aboard', fontsize=9.5)

ax.set_xlabel('Siblings or Spouses aboard')

ax.set_ylabel('Parents or Children aboard')

plt.show()
titanic_df['Family'] = titanic_df.SibSp + titanic_df.Parch + 1
fig, ax = plt.subplots(figsize=[7,2])

ax = sns.countplot('Family', data=titanic_df)

ax.set_title('Distribution of Size of Family', fontsize=9.5)

ax.set_xlabel('Size of Family')

ax.set_ylabel('Frequency')

label_bar_freq(ax, 18)

plt.show()
fig, ax = plt.subplots(figsize=[4,2.5])

ax = sns.regplot(x='Pclass', y='Family', data=titanic_df, y_jitter=.1, x_jitter=.1)

ax.set_title('Correlation between Family and Ticket Class', fontsize=9.5)

ax.set_xlabel('Ticket Class')

ax.set_ylabel('Family')

plt.show()
fig, ax = plt.subplots(figsize=[4,2.5])

ax = sns.regplot(x='Age', y='Family', data=titanic_df, y_jitter=.1, x_jitter=.1)

ax.set_title('Correlation between Family and Age', fontsize=9.5)

ax.set_xlabel('Age')

ax.set_ylabel('Family')

plt.show()
fig = plt.figure(figsize=(4.5,4))

ax = sns.heatmap(titanic_df.drop(['PassengerId', 'Survived'], axis=1).corr(), vmax=.6, square=True, annot=True, cmap=p_heatmap, annot_kws={'size': 9})

plt.title('Correlation of Numerical Features with Age', fontsize=9.5)



hightlighter = [ax.add_patch(Rectangle((1, i), 1, 1, fill=False, edgecolor='#AF3D3F', lw=2)) for i in [5, 3, 2, 0]]



plt.show()
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=[12,2], sharey=True)



sns.boxplot(x='Pclass', y='Age', data=titanic_df, ax=ax1, whis=np.inf, linewidth=1.3)

sns.stripplot(x='Pclass', y='Age', data=titanic_df, ax=ax1, jitter=.2, color='.2', size=1.8, alpha=.3)



sns.boxplot(x='SibSp', y='Age', data=titanic_df, ax=ax2, whis=np.inf, linewidth=1.2)

sns.stripplot(x='SibSp', y='Age', data=titanic_df, ax=ax2, jitter=.2, color='.2', size=1.8, alpha=.15)



sns.boxplot(x='Parch', y='Age', data=titanic_df, ax=ax3, whis=np.inf, linewidth=1.2)

sns.stripplot(x='Parch', y='Age', data=titanic_df, ax=ax3, jitter=.2, color='.2', size=1.8, alpha=.15)



sns.boxplot(x='Family', y='Age', data=titanic_df, ax=ax4, whis=np.inf, linewidth=1.2)

sns.stripplot(x='Family', y='Age', data=titanic_df, ax=ax4, jitter=.2, color='.2', size=1.8, alpha=.15)



hide_common_ylabel = [ax.set_ylabel('') for ax in [ax2, ax3, ax4]]

fig.suptitle('Correlation of Numerical Features with Age', fontsize=9.5)



sns.despine()



plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[11,2], sharey=True)



sns.boxplot(x='Sex', y='Age', data=titanic_df, ax=ax1, whis=np.inf, linewidth=1.3)

sns.stripplot(x='Sex', y='Age', data=titanic_df, ax=ax1, jitter=.2, color='.2', size=1.8, alpha=.3)



sns.boxplot(x='Embarked', y='Age', data=titanic_df, ax=ax2, whis=np.inf, linewidth=1.3)

sns.stripplot(x='Embarked', y='Age', data=titanic_df, ax=ax2, jitter=.2, color='.2', size=1.8, alpha=.28)



sns.boxplot(x='Title', y='Age', data=titanic_df, ax=ax3, whis=np.inf, linewidth=1.2)

sns.stripplot(x='Title', y='Age', data=titanic_df, ax=ax3, jitter=.2, color='.2', size=1.8, alpha=.2)



hide_common_ylabel = [ax.set_ylabel('') for ax in [ax2, ax3, ax4]]

fig.suptitle('Correlation of Categorical Features with Age', fontsize=9.5)



plt.show()
missing_age_index = list(titanic_df.Age[titanic_df['Age'].isnull()].index)
titanic_df.ix[missing_age_index[0:5]]
def guess_age(i):

    '''(int) -> float

    Return the guessed age with the given index of passenger from missing_age_index.

    '''

    # Get passenger data from dataframe

    passenger = titanic_df.iloc[missing_age_index[i]]

    

    # Find matching passengers from dataframe with the same Pclass, Title, SibSp, Parch.

    corr_passenger = titanic_df[(titanic_df.Pclass == passenger.Pclass) & 

                                (titanic_df.Title == passenger.Title) & 

                                (titanic_df.SibSp == passenger.SibSp) & 

                                (titanic_df.Parch == passenger.Parch) & 

                                (titanic_df.Sex == passenger.Sex) & 

                                (titanic_df.Age > 0)]

    

    # If no matching passengers found, matching criteria is loosen to only Pclass and Title.

    if len(corr_passenger) == 0:

        corr_passenger = titanic_df[(titanic_df.Pclass == passenger.Pclass) & 

                                    (titanic_df.Title == passenger.Title) & 

                                    (titanic_df.Sex == passenger.Sex) & 

                                    (titanic_df.Age > 0)]

    

    # Return the age of a random-sampled matching passengers, i.e. the guessed age.

    return corr_passenger.Age.sample(n=1).iloc[0]
for i in range(len(missing_age_index)):

    titanic_df.ix[missing_age_index[i], 'Age'] = guess_age(i)
titanic_df.Age.isnull().sum()
titanic_df.ix[missing_age_index[0:5]]
fig, ax = plt.subplots(figsize=[6.5,2.5])

ax.hist(titanic_df.Age, bins=17) # bins=17

ax.set_title('Distribution of Age with Missing Values Completed', fontsize=9.5)

ax.set_xlabel('Age')

ax.set_ylabel('Frequency')

ax.set_xticks(np.arange(0, 81, 5))

label_bar_freq(ax, 4)

plt.show()
titanic_df.Ticket.describe()
titanic_df[titanic_df.Ticket == 'CA. 2343']
len(titanic_df[titanic_df.Lastname == 'Sage']) == 7
titanic_df.groupby('Ticket').PassengerId.count().sort_values(ascending=False)[:10]
titanic_df[titanic_df.Ticket == '1601']
titanic_df[titanic_df.Ticket == 'S.O.C. 14879']
titanic_df.Ticket.head()
def cleanTicket(ticket):

    ticket = ticket.replace('.' , '')

    ticket = ticket.replace('/' , '')

    ticket = ticket.split()

    ticket = map(lambda t: t.strip(), ticket)

    ticket = list(filter(lambda t: not t.isdigit(), ticket))

    if len(ticket) > 0:

        return ticket[0]

    else: 

        return np.nan
titanic_df['Tclass'] = titanic_df['Ticket'].map(cleanTicket)
fig, ax = plt.subplots(figsize=[11.5,2])

Tclass_order = list(titanic_df.groupby(['Tclass']).PassengerId.count().sort_values(ascending=False).index)

g = sns.countplot('Tclass', data=titanic_df, ax=ax, order=Tclass_order)

g.set_title('Distribution of Ticket Prefix', fontsize=9.5)

g.set_xlabel('Ticket Prefix')

g.set_ylabel('Frequency')

label_bar_freq(ax, 1.5)

for item in g.get_xticklabels():

    item.set_rotation(45)
titanic_df.Tclass.describe()
titanic_df.Tclass.isnull().sum()
titanic_df[titanic_df.Tclass == 'PC'][:8]
titanic_df[titanic_df.Tclass == 'PC'].Pclass.unique()
titanic_df[titanic_df.Tclass == 'CA'].Pclass.unique()
titanic_df[titanic_df.Tclass == 'A5'].Pclass.unique()
fig, ax = plt.subplots(figsize=[11.5,1.5])

g = sns.stripplot(x='Tclass', y='Pclass', data=titanic_df, order=Tclass_order)

g.set_title('Distribution of Ticket Class by Ticket Prefix', fontsize=9.5)

g.set_xlabel('Ticket Prefix')

g.set_ylabel('Ticket Class')

g.set_yticks([1, 2, 3])

for item in g.get_xticklabels():

    item.set_rotation(45)
titanic_df.Fare.describe()
def find_outliers(df, column, filter_column, filter_value):

    '''(DataFrame, str, str, any) -> DataFrame

    Return a DataFrame with outliers of column under a condition when filter_column == filter_value in a df.

    Precondition: filter_value must be values in filter_column.

    '''

    filter_df = df[df[filter_column] == filter_value].copy()

    filter_df['x-Mean'] = abs(filter_df[column] - filter_df[column].mean())

    filter_df['1.96*std'] = 1.96 * filter_df[column].std()  

    filter_df['Outlier'] = abs(filter_df[column] - filter_df[column].mean()) > 1.96 * filter_df[column].std()

    return filter_df[filter_df.Outlier == True].sort_values(column, ascending=False)
find_outliers(titanic_df, 'Fare', 'Pclass', 1).drop(['x-Mean', '1.96*std'], axis=1)
find_outliers(titanic_df, 'Fare', 'Pclass', 1).groupby('Tclass').PassengerId.count()
find_outliers(titanic_df, 'Fare', 'Pclass', 2).groupby('Ticket').PassengerId.count()
find_outliers(titanic_df, 'Fare', 'Pclass', 2).groupby('Tclass').PassengerId.count()
find_outliers(titanic_df, 'Fare', 'Pclass', 3).groupby('Ticket').PassengerId.count()
fig, ax = plt.subplots(figsize=[13.5,2])

Tclass_order = list(titanic_df.groupby(['Tclass']).PassengerId.count().sort_values(ascending=False).index)

g = sns.barplot(x='Tclass', y='Fare', hue='Pclass', data=titanic_df, order=Tclass_order, errwidth=0)

g.set_title('Distribution of Mean Fare by Ticket Prefix', fontsize=9.5)

g.set_xlabel('Ticket Prefix')

g.set_ylabel('Mean Fare')

for item in g.get_xticklabels():

    item.set_rotation(45)
fig, ax = plt.subplots(figsize=[8,1.4])

sns.boxplot(x=titanic_df.Fare, y=titanic_df[titanic_df.Pclass == 1].Tclass, ax=ax, whis=1.5, linewidth=1.2, orient='h')

sns.stripplot(x=titanic_df.Fare, y=titanic_df[titanic_df.Pclass == 1].Tclass, ax=ax, jitter=.2, color='.2', size=1.8, alpha=.3, orient='h')

ax.set_title('Distribution of Fare by Ticket Prefix at 1st Class', fontsize=9.5)

ax.set_xlabel('Fare')

ax.set_ylabel('Ticket Prefix')

plt.show()
fig, ax = plt.subplots(figsize=[8,6.2])

sns.boxplot(x=titanic_df.Fare, y=titanic_df[titanic_df.Pclass == 2].Tclass, ax=ax, whis=1.5, linewidth=1.2, orient='h')

sns.stripplot(x=titanic_df.Fare, y=titanic_df[titanic_df.Pclass == 2].Tclass, ax=ax, jitter=.2, color='.2', size=1.8, alpha=.3, orient='h')

ax.set_title('Distribution of Fare by Ticket Prefix at 2nd Class', fontsize=9.5)

ax.set_xlabel('Fare')

ax.set_ylabel('Ticket Prefix')

plt.show()
fig, ax = plt.subplots(figsize=[8,7])

sns.boxplot(x=titanic_df.Fare, y=titanic_df[titanic_df.Pclass == 3].Tclass, ax=ax, whis=1.5, linewidth=1.2, orient='h')

sns.stripplot(x=titanic_df.Fare, y=titanic_df[titanic_df.Pclass == 3].Tclass, ax=ax, jitter=.2, color='.2', size=1.8, alpha=.3, orient='h')

ax.set_title('Distribution of Fare by Ticket Prefix at 3rd Class', fontsize=9.5)

ax.set_xlabel('Fare')

ax.set_ylabel('Ticket Prefix')

plt.show()
titanic_df.Cabin.describe()
titanic_df.Cabin.head(10)
titanic_df['Deck'] = titanic_df.Name.str.extract('([A-Z])', expand=False)
titanic_df.Deck.head()
titanic_df.Deck.isnull().sum()
titanic_df.Deck.describe()
titanic_df.Deck.unique()
fig, ax = plt.subplots(figsize=[11.5,2])

Deck_order = list(titanic_df.groupby(['Deck']).PassengerId.count().sort_values(ascending=False).index)

g = sns.countplot('Deck', data=titanic_df, ax=ax, order=Deck_order)

g.set_title('Distribution of Deck', fontsize=9.5)

g.set_xlabel('Deck')

g.set_ylabel('Frequency')

label_bar_freq(ax, 3)
fig, ax = plt.subplots(figsize=[9.5,2])

titanic_df.groupby(['Deck', 'Pclass']).size().unstack().plot.bar(stacked=True, ax=ax, rot=0)

ax.set_title('Distribution of Pclass by Deck', fontsize=9.5)

ax.set_ylabel('Frequency')

ax.set_yticks(np.arange(0, 86, 20))

leg = ax.legend(fontsize = 'small')

leg.set_title('Pclass', prop = {'size':'small'})

ax.grid(False, axis='x')

plt.show()
titanic_df[(titanic_df.Deck == 'Q') | (titanic_df.Deck == 'U')]
fig, ax = plt.subplots(figsize=[11,2])

Deck_order = list(''.join(sorted(titanic_df.Deck.unique())))

g = sns.barplot(x='Deck', y='Fare', hue='Pclass', data=titanic_df, errwidth=0, order=Deck_order)

g.set_title('Distribution of Mean Fare in Different Ticket Class by Deck', fontsize=9.5)

g.set_xlabel('Deck')

g.set_ylabel('Mean Fare')

leg = g.legend(loc='upper right')

leg.set_title('Pclass', prop = {'size':'small'})

plt.show()
fig, ax = plt.subplots(figsize=[9.5,2.5])

tclass_colors = ['#1395BA', '#117899', '#0F5B78', '#0D3C55', '#C02E1D', '#D94E1F', '#F16C20', '#EF8B2C', '#ECAA38', '#EBC844', '#A2B86C', '#5CA793']

titanic_df.groupby('Tclass').filter(lambda x: len(x) > 4).groupby(['Deck', 'Tclass']).size().unstack().plot.bar(stacked=True, ax=ax, rot=0, color=tclass_colors)

ax.set_title('Distribution of Tclass by Deck', fontsize=9.5)

ax.set_ylabel('Frequency')

leg = ax.legend(fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5), ncol=3)

leg.set_title('Ticket Prefix', prop={'size':'small'})

ax.grid(False, axis='x')

plt.show()
fig, ax = plt.subplots(figsize=[13.5,2])

g = sns.barplot(x='Deck', y='Age', hue='Pclass', data=titanic_df, order=Deck_order, errwidth=0)

g.set_title('Distribution of Mean Age by Deck', fontsize=9.5)

g.set_xlabel('Deck')

g.set_ylabel('Mean Age')

plt.show()
fig, ax = plt.subplots(figsize=[9.5,2])

titanic_df.groupby(['Deck', 'Embarked']).size().unstack().plot.bar(stacked=True, ax=ax, rot=0)

ax.set_title('Distribution of Port of Embarktion by Deck', fontsize=9.5)

ax.set_ylabel('Frequency')

ax.set_yticks(np.arange(0, 86, 20))

leg = ax.legend(fontsize = 'small')

leg.set_title('Embarked', prop = {'size':'small'})

ax.grid(False, axis='x')

plt.show()
titanic_df.Embarked.fillna('S', inplace=True)
titanic_df.Embarked.isnull().sum()
fig, ax = plt.subplots(figsize=[4,2.5])

titanic_df.groupby(['Embarked', 'Pclass']).size().unstack().plot.bar(stacked=True, ax=ax, rot=0)

ax.set_title('Distribution of Pclass by Embarked', fontsize=9.5)

ax.set_ylabel('Frequency')

leg = ax.legend(loc='best', fontsize = 'small')

leg.set_title('Pclass', prop = {'size':'small'})

ax.grid(False, axis='x')

plt.show()
fig, ax = plt.subplots(figsize=[4,2.5])

sns.boxplot(x='Embarked', y='Fare', data=titanic_df, whis=np.inf, linewidth=1.3)

sns.stripplot(x='Embarked', y='Fare', data=titanic_df, jitter=.2, color='.2', size=1.8, alpha=.28)

ax.set_title('Distribution of Pclass by Embarked', fontsize=9.5)

plt.show()
fig, ax = plt.subplots(figsize=[11.5,2])

titanic_df.groupby(['Tclass', 'Embarked']).size().unstack().plot.bar(stacked=True, ax=ax, rot=45)

ax.set_title('Distribution of Port of Embarktion by Deck', fontsize=9.5)

ax.set_ylabel('Frequency')

leg = ax.legend(fontsize = 'small')

leg.set_title('Embarked', prop = {'size':'small'})

ax.grid(False, axis='x')

plt.show()
fig, ax = plt.subplots(figsize=[4,2])

g = sns.countplot(x='Pclass', hue='Survived', data=titanic_df)

g.set_title('Distribution of Survival by Ticket Class', fontsize=9.5)

g.set_xlabel('Ticket Class')

g.set_ylabel('Frequency')

plt.show()
surv_rate_pclass = pd.DataFrame()

surv_rate_pclass['Pclass'] = [1, 2, 3]
def create_surv_rate_df(surv_rate_df, column):

    for i in range(len(surv_rate_df)):

        surv_rate_df.ix[i, 'Survival'] = titanic_df[(titanic_df[column] == surv_rate_df[column][i]) & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[titanic_df[column] == surv_rate_df[column][i]].PassengerId.count()
create_surv_rate_df(surv_rate_pclass, 'Pclass')
surv_rate_pclass
fig, ax = plt.subplots(figsize=[4,2])

sns.barplot(x='Pclass', y='Survival', data=surv_rate_pclass, ax=ax)

ax.set_title('Survival Rate by Ticket Class', fontsize=9.5)

ax.set_xlabel('Ticket Class')

ax.set_ylabel('Survival Rate')

plt.show()
titanic_df.Pclass.corr(titanic_df.Survived)
g = sns.FacetGrid(titanic_df, hue='Survived', size=2.6, aspect=2.1, sharex=False)

g.map(plt.hist, 'Fare', alpha=.6)

g.fig.suptitle('Distribution of Survival by Fare', fontsize=9.5, y=1.01)

g.set_ylabels('Frequency')

plt.show()
fig, ax = plt.subplots(figsize=[8,2.5])



np.seterr(divide='ignore', invalid='ignore')



titanic_df_survived = titanic_df[titanic_df['Survived'] == 1]

titanic_df_not_survived = titanic_df[titanic_df['Survived'] == 0]



fare_bins = np.linspace(0, 600, 50)

survived_hist_fare = np.histogram(titanic_df_survived['Fare'], bins=fare_bins, range=(0,600))

not_survive_hist_fare = np.histogram(titanic_df_not_survived['Fare'], bins=fare_bins, range=(0,600))



surv_rates_fare = survived_hist_fare[0] / (survived_hist_fare[0] + not_survive_hist_fare[0])



ax.bar(fare_bins[:-1], surv_rates_fare, width=fare_bins[1] - fare_bins[0])

ax.set_title('Distribution of Survival Rate by Fare', fontsize=9.5)

ax.set_xlabel('Fare')

ax.set_ylabel('Survival Rate')

plt.show()
titanic_df.Fare.corr(titanic_df.Survived)
g = sns.FacetGrid(titanic_df, col='Pclass', hue='Survived', size=2.3, aspect=1.2, sharex=False)

g.map(plt.hist, 'Fare', alpha=.6)

g.set_xlabels('')

g.fig.suptitle('Distribution of Survival by Ticket Class and Fare', fontsize=9.5, y=1.01)

g.fig.text(.5, .1, 'Fare', ha='center', fontsize=9.5)

g.set_ylabels('Frequency')

g.add_legend()

plt.show()
def plot_surv_rate_fare_pclass(pclass, ax):

    survived_hist_fare = np.histogram(titanic_df_survived[titanic_df_survived.Pclass == pclass].Fare, bins=fare_bins, range=(0,600))

    not_survive_hist_fare = np.histogram(titanic_df_not_survived[titanic_df_not_survived.Pclass == pclass].Fare, bins=fare_bins, range=(0,600))



    surv_rates_fare = survived_hist_fare[0] / (survived_hist_fare[0] + not_survive_hist_fare[0])



    return ax.bar(fare_bins[:-1], surv_rates_fare, width=fare_bins[1] - fare_bins[0])
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[11,2], sharex=True, sharey=True)



for ax, i in ([ax1, 1], [ax2, 2], [ax3, 3]):

    plot_surv_rate_fare_pclass(i, ax)

    ax.set_title('Pclass = {}'.format(i), fontsize=9.5)

    

ax1.set_ylabel('Survival Rate')



fig.suptitle('Survival Rate by Ticket Class and Fare', fontsize=9.5, y=1.07)

fig.text(.5, -.05, 'Fare', ha='center', fontsize=9.5)



plt.show()
titanic_df.ix[titanic_df.Cabin.isnull(), 'HasCabin'] = 0

titanic_df.ix[titanic_df.Cabin.notnull(), 'HasCabin'] = 1

titanic_df.HasCabin = titanic_df.HasCabin.astype(int)
fig, ax = plt.subplots(figsize=[4,2])

sns.countplot(x='HasCabin', hue='Pclass', data=titanic_df, ax=ax)

ax.set_title('Distribution of Cabin by Ticket Class', fontsize=9.5)

ax.set_xlabel('Cabin')

ax.set_ylabel('Frequency')

plt.show()
surv_rate_hascabin = pd.DataFrame()

surv_rate_hascabin['HasCabin'] = [0, 1]

create_surv_rate_df(surv_rate_hascabin, 'HasCabin')
surv_rate_hascabin
fig, ax = plt.subplots(figsize=[4,2])

sns.barplot(x='HasCabin', y='Survival', data=surv_rate_hascabin, ax=ax)

ax.set_title('Survival Rate by Cabin', fontsize=9.5)

ax.set_xlabel('Cabin')

ax.set_ylabel('Survival Rate')

plt.show()
surv_rate_compare_5 = pd.DataFrame()
surv_rate_compare_5['Pclass'] = [1, 1, 2, 2, 3, 3]
surv_rate_compare_5['HasCabin'] = [0, 1] * 3
surv_rate_compare_5.ix[0, 'Survival'] = titanic_df[(titanic_df.HasCabin == 0) & (titanic_df.Pclass == 1) & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[(titanic_df.HasCabin == 0) & (titanic_df.Pclass == 1) ].PassengerId.count()

surv_rate_compare_5.ix[1, 'Survival'] = titanic_df[(titanic_df.HasCabin == 1) & (titanic_df.Pclass == 1) & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[(titanic_df.HasCabin == 1) & (titanic_df.Pclass == 1) ].PassengerId.count()

surv_rate_compare_5.ix[2, 'Survival'] = titanic_df[(titanic_df.HasCabin == 0) & (titanic_df.Pclass == 2) & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[(titanic_df.HasCabin == 0) & (titanic_df.Pclass == 2) ].PassengerId.count()

surv_rate_compare_5.ix[3, 'Survival'] = titanic_df[(titanic_df.HasCabin == 1) & (titanic_df.Pclass == 2) & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[(titanic_df.HasCabin == 1) & (titanic_df.Pclass == 2) ].PassengerId.count()

surv_rate_compare_5.ix[4, 'Survival'] = titanic_df[(titanic_df.HasCabin == 0) & (titanic_df.Pclass == 3) & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[(titanic_df.HasCabin == 0) & (titanic_df.Pclass == 3) ].PassengerId.count()

surv_rate_compare_5.ix[5, 'Survival'] = titanic_df[(titanic_df.HasCabin == 1) & (titanic_df.Pclass == 3) & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[(titanic_df.HasCabin == 1) & (titanic_df.Pclass == 3) ].PassengerId.count()
surv_rate_compare_5
fig, ax = plt.subplots(figsize=[4,2])

sns.barplot(x='Pclass', y='Survival', hue='HasCabin', data=surv_rate_compare_5, ax=ax)

ax.set_title('Survival Rate by Sex and Ticket Class', fontsize=9.5)

ax.set_xlabel('Ticket Class')

ax.set_ylabel('Survival Rate')

plt.show()
surv_rate_embarked = pd.DataFrame()
surv_rate_embarked['Embarked'] = ['S', 'C', 'Q']
surv_rate_embarked.ix[0, 'Survival'] = titanic_df[(titanic_df.Embarked == 'S') & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[titanic_df.Embarked == 'S'].PassengerId.count()

surv_rate_embarked.ix[1, 'Survival'] = titanic_df[(titanic_df.Embarked == 'C') & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[titanic_df.Embarked == 'C'].PassengerId.count()

surv_rate_embarked.ix[2, 'Survival'] = titanic_df[(titanic_df.Embarked == 'Q') & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[titanic_df.Embarked == 'Q'].PassengerId.count()
surv_rate_embarked
fig, ax = plt.subplots(figsize=[4,2])

sns.barplot(x='Embarked', y='Survival', data=surv_rate_embarked, ax=ax)

ax.set_title('Survival Rate by Port of Embarktion', fontsize=9.5)

ax.set_xlabel('Port of Embarktion')

ax.set_ylabel('Survival Rate')

plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[7,2], sharex=True, sharey=True)



ddf_Pclass_Embarked0 = pd.crosstab(titanic_df[titanic_df.Survived == 0].Pclass, titanic_df[titanic_df.Survived == 0].Embarked)

ddf_Pclass_Embarked0.divide(ddf_Pclass_Embarked0.sum(axis=1), axis=0).plot.area(stacked=True, ax=ax1, color=['#55A868', '#4C72B0', '#B55D60'])

ax1.set_title('Survived = 0', fontsize=9.5)

ax1.legend('')

ax1.set_xlabel('')

ax1.set_ylabel('Percent (%)')



ddf_Pclass_Embarked1 = pd.crosstab(titanic_df[titanic_df.Survived == 1].Pclass, titanic_df[titanic_df.Survived == 1].Embarked)

ddf_Pclass_Embarked1.divide(ddf_Pclass_Embarked1.sum(axis=1), axis=0).plot.area(stacked=True, ax=ax2, color=['#55A868', '#4C72B0', '#B55D60'])

ax2.set_title('Survived = 1', fontsize=9.5)

ax2.set_xlabel('')



leg = ax2.legend(fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5))

leg.set_title('Embarked', prop={'size':'small'})



fig.suptitle('Composition of Port of Embarktion in Percentage by Ticket Class', fontsize=9.5, y=1.06)

fig.text(.5, -.05, 'Ticket Class', ha='center', fontsize=9.5)



ax1.set_xticks(np.arange(1, 4, 1))



plt.show()
fig, ax = plt.subplots(figsize=[11.5,2])

g = sns.countplot(x='Tclass', hue='Survived', data=titanic_df, order=Tclass_order)

g.set_title('Distribution of Survival by Ticket Class', fontsize=9.5)

g.set_xlabel('Ticket Prefix')

g.set_ylabel('Frequency')

g.legend().set_title('Survived', prop={'size':'small'})

for item in g.get_xticklabels():

    item.set_rotation(45)

plt.show()
fig, ax = plt.subplots(figsize=[8.5,2])

ddf_Tclass = pd.DataFrame(pd.get_dummies(titanic_df.Tclass).corrwith(titanic_df.Survived).sort_values(ascending=False))

ddf_Tclass.plot(kind='bar', ax=ax, legend='')

ax.set_title('Correlation of Ticket Prefix with Survival', fontsize=9.5)

ax.set_xlabel('Ticket Prefix')

ax.set_ylabel('Correlation')

ax.grid(False, axis='x')

plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=[10,6])

g1 = sns.countplot(x='Tclass', hue='Survived', data=titanic_df[titanic_df.Pclass == 1], ax=ax1)

g2 = sns.countplot(x='Tclass', hue='Survived', data=titanic_df[titanic_df.Pclass == 2], ax=ax2)

g3 = sns.countplot(x='Tclass', hue='Survived', data=titanic_df[titanic_df.Pclass == 3], ax=ax3)



g1.set_title('Pclass = 1', fontsize=9.5, y=.85)

g2.set_title('Pclass = 2', fontsize=9.5, y=.85)

g3.set_title('Pclass = 3', fontsize=9.5, y=.85)

g1.set_xlabel(''); g2.set_xlabel(''); g3.set_xlabel('Ticket Prefix')

g1.set_ylabel(''); g2.set_ylabel('Frequency'); g3.set_ylabel('')

g1.legend(''); g3.legend('')

leg = g2.legend(fontsize='small', loc='center left', bbox_to_anchor=(1, 0.5))

leg.set_title('Ticket Prefix', prop={'size':'small'})

fig.suptitle('Distribution of Survival by Ticket Prefix and Ticket Class', fontsize=9.5, y=.94)



plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=[10,6])



for pclass, ax in ([1, ax1], [2, ax2], [3, ax3]):

    tclass_pclass = pd.crosstab(titanic_df[titanic_df.Pclass == pclass].Tclass, titanic_df[titanic_df.Pclass == pclass].Pclass)

    tclass_pclass_survived = pd.crosstab(titanic_df[(titanic_df.Pclass == pclass) & (titanic_df.Survived == 1)].Tclass, titanic_df[(titanic_df.Pclass == pclass) & (titanic_df.Survived == 1)].Pclass)

    (tclass_pclass_survived / tclass_pclass).fillna(0).plot.bar(ax=ax, rot=0)

    ax.legend('')

    ax.set_title('Pclass = {}'.format(pclass), fontsize=9.5, y=.85, x=.01, loc='left')

    

ax3.set_xlabel('Ticket Prefix')

ax2.set_ylabel('Survival Rate')

fig.suptitle('Survival Rate by Ticket Prefix and Ticket Class', fontsize=9.5, y=.94)



plt.show()
fig, ax = plt.subplots(figsize=[4,2])

g = sns.countplot(x='Sex', hue='Survived', data=titanic_df)

g.set_title('Distribution of Survival by Sex', fontsize=9.5)

g.set_xlabel('Sex')

g.set_ylabel('Frequency')

plt.show()
surv_rate_sex = pd.DataFrame()

surv_rate_sex['Sex'] = ['male', 'female']

create_surv_rate_df(surv_rate_sex, 'Sex')
surv_rate_sex
fig, ax = plt.subplots(figsize=[4,2])

sns.barplot(x='Sex', y='Survival', data=surv_rate_sex, ax=ax)

ax.set_title('Survival Rate by Sex', fontsize=9.5)

ax.set_xlabel('Sex')

ax.set_ylabel('Survival Rate')

plt.show()
fig, ax = plt.subplots(figsize=[4,2])

ax.hist(titanic_df.Age[titanic_df.Survived == 0].dropna(), alpha=.6)

ax.hist(titanic_df.Age[titanic_df.Survived == 1].dropna(), alpha=.6)

ax.set_title('Distribution of Survival by Age', fontsize=9)

ax.set_xlabel('Age')

ax.set_ylabel('Frequency')

ax.legend(['0','1']).set_title('Survived', prop={'size':'small'})

plt.show()
fig, ax = plt.subplots(figsize=[6,2])



age_bins = np.linspace(0, 80, 21)

survived_hist_age = np.histogram(titanic_df_survived['Age'], bins=age_bins, range=(0,80))

not_survive_hist_age = np.histogram(titanic_df_not_survived['Age'], bins=age_bins, range=(0,80))



surv_rates_age = survived_hist_age[0] / (survived_hist_age[0] + not_survive_hist_age[0])



ax.bar(age_bins[:-1], surv_rates_age, width=age_bins[1] - age_bins[0])

ax.set_title('Survival Rate by Age', fontsize=9.5)

ax.set_xlabel('Age')

ax.set_ylabel('Survival Rate')

plt.show()
fig, ax = plt.subplots(figsize=[4,2])

g = sns.countplot(x='Title', hue='Survived', data=titanic_df)

g.set_title('Distribution of Survival by Title', fontsize=9.5)

g.set_xlabel('Title')

g.set_ylabel('Frequency')

plt.show()
surv_rate_title = pd.DataFrame()

surv_rate_title['Title'] = ['Mr', 'Mrs', 'Miss', 'Master', 'Royalty', 'Officer']

create_surv_rate_df(surv_rate_title, 'Title')
surv_rate_title
fig, ax = plt.subplots(figsize=[5,2])

sns.barplot(x='Title', y='Survival', data=surv_rate_title, ax=ax)

ax.set_title('Survival Rate by Title', fontsize=9.5)

ax.set_yticks(np.arange(0,1,.1))

ax.set_xlabel('Title')

ax.set_ylabel('Survival Rate')

plt.show()
surv_rate_compare_1 = pd.DataFrame()
surv_rate_compare_1.ix['Pclass = 1', 'Survival'] = titanic_df[(titanic_df.Pclass == 1) & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[(titanic_df.Pclass == 1)].PassengerId.count()

surv_rate_compare_1.ix['Sex = female', 'Survival'] = titanic_df[(titanic_df.Sex == 'female') & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[(titanic_df.Sex == 'female')].PassengerId.count()

surv_rate_compare_1.ix['Age = 0-8', 'Survival'] = titanic_df[(titanic_df.Age <= 8) & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[(titanic_df.Age <= 8)].PassengerId.count()
surv_rate_compare_1
fig, ax = plt.subplots(figsize=[4,2])

surv_rate_compare_1.plot.bar(ax=ax, rot=0, legend='')

ax.set_title('Survival Rate of Upper Class, Female, and Children', fontsize=9.5)

ax.set_xlabel('Category')

ax.set_ylabel('Survival Rate')

plt.show()
surv_rate_compare_2 = pd.DataFrame()
surv_rate_compare_2['Pclass'] = [1, 1, 2, 2, 3, 3]
surv_rate_compare_2['Sex'] = ['male', 'female'] * 3
surv_rate_compare_2.ix[0, 'Survival'] = titanic_df[(titanic_df.Sex == 'male') & (titanic_df.Pclass == 1) & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[(titanic_df.Sex == 'male') & (titanic_df.Pclass == 1) ].PassengerId.count()

surv_rate_compare_2.ix[1, 'Survival'] = titanic_df[(titanic_df.Sex == 'female') & (titanic_df.Pclass == 1) & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[(titanic_df.Sex == 'female') & (titanic_df.Pclass == 1) ].PassengerId.count()

surv_rate_compare_2.ix[2, 'Survival'] = titanic_df[(titanic_df.Sex == 'male') & (titanic_df.Pclass == 2) & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[(titanic_df.Sex == 'male') & (titanic_df.Pclass == 2) ].PassengerId.count()

surv_rate_compare_2.ix[3, 'Survival'] = titanic_df[(titanic_df.Sex == 'female') & (titanic_df.Pclass == 2) & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[(titanic_df.Sex == 'female') & (titanic_df.Pclass == 2) ].PassengerId.count()

surv_rate_compare_2.ix[4, 'Survival'] = titanic_df[(titanic_df.Sex == 'male') & (titanic_df.Pclass == 3) & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[(titanic_df.Sex == 'male') & (titanic_df.Pclass == 3) ].PassengerId.count()

surv_rate_compare_2.ix[5, 'Survival'] = titanic_df[(titanic_df.Sex == 'female') & (titanic_df.Pclass == 3) & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[(titanic_df.Sex == 'female') & (titanic_df.Pclass == 3) ].PassengerId.count()
surv_rate_compare_2
fig, ax = plt.subplots(figsize=[4,2])

sns.barplot(x='Pclass', y='Survival', hue='Sex', data=surv_rate_compare_2, ax=ax)

ax.set_title('Survival Rate by Sex and Ticket Class', fontsize=9.5)

ax.set_xlabel('Ticket Class')

ax.set_ylabel('Survival Rate')

plt.show()
fig, ax = plt.subplots(figsize=[4,2])

ax.hist(titanic_df[(titanic_df.Age <= 8) & (titanic_df.Sex == 'male')].Age, alpha=.6)

ax.hist(titanic_df[(titanic_df.Age <= 8) & (titanic_df.Sex == 'female')].Age, alpha=.6)

ax.set_title('Distribution of Sex by Age (0 - 8)', fontsize=9.5)

ax.set_xlabel('Age')

ax.set_ylabel('Frequency')

ax.legend(['male', 'female']).set_title('Sex', prop={'size': 'small'})

plt.show()
surv_rate_compare_3 = pd.DataFrame()
surv_rate_compare_3.ix['Boy', 'Survival'] = titanic_df[(titanic_df.Age <= 8) & (titanic_df.Sex == 'male') & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[(titanic_df.Age <= 8) & (titanic_df.Sex == 'male')].PassengerId.count()

surv_rate_compare_3.ix['Girl', 'Survival'] = titanic_df[(titanic_df.Age <= 8) & (titanic_df.Sex == 'female') & (titanic_df.Survived == 1)].PassengerId.count() / titanic_df[(titanic_df.Age <= 8) & (titanic_df.Sex == 'female')].PassengerId.count()
surv_rate_compare_3
fig, ax = plt.subplots(figsize=[4,2])

surv_rate_compare_3.plot.bar(ax=ax, rot=0, legend='')

ax.set_title('Survival Rate of Boy and Girl between Age 0 - 8', fontsize=9.5)

ax.set_xlabel('Sex')

ax.set_ylabel('Survival Rate')

plt.show()
titanic_df[(titanic_df.Age <= 8) & (titanic_df.Sex == 'male')][['Pclass', 'Age', 'Family']].describe()
titanic_df[(titanic_df.Age <= 8) & (titanic_df.Sex == 'female')][['Pclass', 'Age', 'Family']].describe()
titanic_df_ch = titanic_df[titanic_df.Age <=8]
titanic_df_ch.groupby('Pclass').PassengerId.count()
surv_rate_compare_4 = pd.DataFrame()
surv_rate_compare_4['Pclass'] = ['1, 2', '1, 2', '3', '3']
surv_rate_compare_4['Sex'] = ['male', 'female'] * 2
surv_rate_compare_4.ix[0, 'Survival'] = titanic_df_ch[(titanic_df_ch.Sex == 'male') & (titanic_df_ch.Pclass <= 2) & (titanic_df_ch.Survived == 1)].PassengerId.count() / titanic_df_ch[(titanic_df_ch.Sex == 'male') & (titanic_df_ch.Pclass <= 2) ].PassengerId.count()

surv_rate_compare_4.ix[1, 'Survival'] = titanic_df_ch[(titanic_df_ch.Sex == 'female') & (titanic_df_ch.Pclass <= 2) & (titanic_df_ch.Survived == 1)].PassengerId.count() / titanic_df_ch[(titanic_df_ch.Sex == 'female') & (titanic_df_ch.Pclass <= 2) ].PassengerId.count()

surv_rate_compare_4.ix[2, 'Survival'] = titanic_df_ch[(titanic_df_ch.Sex == 'male') & (titanic_df_ch.Pclass == 3) & (titanic_df_ch.Survived == 1)].PassengerId.count() / titanic_df_ch[(titanic_df_ch.Sex == 'male') & (titanic_df_ch.Pclass == 3) ].PassengerId.count()

surv_rate_compare_4.ix[3, 'Survival'] = titanic_df_ch[(titanic_df_ch.Sex == 'female') & (titanic_df_ch.Pclass == 3) & (titanic_df_ch.Survived == 1)].PassengerId.count() / titanic_df_ch[(titanic_df_ch.Sex == 'female') & (titanic_df_ch.Pclass == 3) ].PassengerId.count()
surv_rate_compare_4
fig, ax = plt.subplots(figsize=[4,2])

sns.barplot(x='Pclass', y='Survival', hue='Sex', data=surv_rate_compare_4, ax=ax)

ax.set_title('Survival Rate by Sex and Ticket Class at Age 0 - 8', fontsize=9.5)

ax.set_xlabel('Ticket Class')

ax.set_ylabel('Survival Rate')

plt.show()
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[11,2], sharey=True)

sns.countplot(x='SibSp', hue='Survived', data=titanic_df, ax=ax1)

sns.countplot(x='Parch', hue='Survived', data=titanic_df, ax=ax2)

sns.countplot(x='Family', hue='Survived', data=titanic_df, ax=ax3)



ax1.set_title('Distribution of Survival by \nextSiblings or Spouses Aboard', fontsize=9.5)

ax1.set_xlabel('Siblings or Spouses Aboard')

ax1.set_ylabel('Frequency')

ax1.legend('')



ax2.set_title('Distribution of Survival by \nParents or Children Aboard', fontsize=9.5)

ax2.set_xlabel('Parents or Children Aboard')

ax2.set_ylabel('')

ax2.legend('')



ax3.set_title('Distribution of Survival by \nFamily Size', fontsize=9.5)

ax3.set_xlabel('Family Size')

ax3.set_ylabel('')

leg = ax3.legend(fontsize='small', loc='center left', bbox_to_anchor=(1, .55))

leg.set_title('Survived', prop={'size':'small'})



plt.show()
surv_rate_sibsp = pd.DataFrame()

surv_rate_parch = pd.DataFrame()

surv_rate_family = pd.DataFrame()



surv_rate_sibsp['SibSp'] = [0, 1, 2, 3, 4, 5, 8]

surv_rate_parch['Parch'] = [0, 1, 2, 3, 4, 5, 6]

surv_rate_family['Family'] = [1, 2, 3, 4, 5, 6, 7, 8, 11]
create_surv_rate_df(surv_rate_sibsp, 'SibSp')

create_surv_rate_df(surv_rate_parch, 'Parch')

create_surv_rate_df(surv_rate_family, 'Family')
surv_rate_sibsp
surv_rate_parch
surv_rate_family
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[11.5,2], sharey=True)

sns.barplot(x='SibSp', y='Survival', data=surv_rate_sibsp, ax=ax1)

sns.barplot(x='Parch', y='Survival', data=surv_rate_parch, ax=ax2)

sns.barplot(x='Family', y='Survival', data=surv_rate_family, ax=ax3)



ax1.set_title('Survival Rate by \nSiblings or Spouses Aboard', fontsize=9.5)

ax1.set_xlabel('Siblings or Spouses Aboard')

ax1.set_ylabel('Survival Rate')

ax1.legend('')



ax2.set_title('Survival Rate by \nParents or Children Aboard', fontsize=9.5)

ax2.set_xlabel('Parents or Children Aboard')

ax2.set_ylabel('')

ax2.legend('')



ax3.set_title('Survival Rate by \nFamily Size', fontsize=9.5)

ax3.set_xlabel('Family Size')

ax3.set_ylabel('')

ax3.legend('')



plt.show()
shared_tickets = []

for ticket in titanic_df.Ticket.unique():

    if len(titanic_df[titanic_df.Ticket == ticket].Ticket) > 1:

        shared_tickets.append(ticket)
len(shared_tickets)
titanic_df[titanic_df.Ticket == shared_tickets[1]]
titanic_df[titanic_df.Ticket == shared_tickets[19]]
ticket_df = pd.DataFrame()

ticket_df['Ticket'] = titanic_df.Ticket.unique()



for t in titanic_df.Ticket.unique():

    ticket_df.ix[ticket_df.Ticket == t, 'Passengers'] = titanic_df[titanic_df.Ticket == t].PassengerId.count()

    ticket_df.ix[ticket_df.Ticket == t, 'Survival'] = titanic_df[titanic_df.Ticket == t].Survived.sum() / titanic_df[titanic_df.Ticket == t].Survived.count()



ticket_df.Passengers = ticket_df.Passengers.astype(int)
ticket_df.head()
np.sort(ticket_df.Passengers.unique())
np.sort(titanic_df.Family.unique())
fig, ax = plt.subplots(figsize=[4,2.5])

sns.countplot(x='Passengers', data=ticket_df)

ax.set_title('Distribution of Number of Passengers in a Unique Ticket', fontsize=9.5)

ax.set_ylabel('Frequency')

ax.set_xlabel('Number of Passengers in a Unique Ticket')

label_bar_freq(ax, 15)

plt.show()
fig, ax = plt.subplots(figsize=[3,2])

ax.hist(ticket_df[ticket_df.Passengers == 1].Survival)

ax.set_title('Distribution of Survival Rate of \nPassengers with Unique Ticket', fontsize=9.5)

ax.set_ylabel('Frequency')

ax.set_xlabel('Survival Rate')

plt.show()
shared_ticket_df = ticket_df[ticket_df.Passengers > 1]
g = sns.FacetGrid(shared_ticket_df, col='Passengers', col_wrap=3, size=2.3, aspect=1.2, sharex=True)

g.map(plt.hist, 'Survival')

g.set_xlabels('')

g.fig.suptitle('Distribution of Survival Rate by Number of Passengers in a Unique Ticket', fontsize=9.5, y=1.01)

g.fig.text(.5, .05, 'Survival Rate', ha='center', fontsize=9.5)

g.fig.text(0, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=9.5)



plt.show()
fig, ax = plt.subplots(figsize=[4,2.5])

g = sns.barplot(x='Passengers', y='Survival', data=ticket_df, errwidth=0)

ax.set_title('Distribution of Survival Rate by \nNumber of Passengers in a Unique Ticket', fontsize=9.5, y=1.01)

ax.set_ylabel('Survival Rate')

ax.set_xlabel('Number of Passengers in a Unique Ticket')

plt.show()
titanic_df.Cabin.describe()
cabin_df = pd.DataFrame()

cabin_df['Cabin'] = titanic_df.Cabin.unique()



for c in titanic_df.Cabin.unique():

    cabin_df.ix[cabin_df.Cabin == c, 'Passengers'] = titanic_df[titanic_df.Cabin == c].PassengerId.count()

    cabin_df.ix[cabin_df.Cabin == c, 'Survival'] = titanic_df[titanic_df.Cabin == c].Survived.sum() / titanic_df[titanic_df.Cabin == c].Survived.count()



cabin_df = cabin_df.drop(0)

cabin_df.Passengers = cabin_df.Passengers.astype(int)
cabin_df.head()
fig, ax = plt.subplots(figsize=[4,2.5])

sns.countplot(x='Passengers', data=cabin_df)

ax.set_title('Distribution of Number of Passengers in a Cabin', fontsize=9.5)

ax.set_ylabel('Frequency')

ax.set_xlabel('Number of Passengers in a Cabin')

label_bar_freq(ax, 3)

plt.show()
cabin_df.Passengers.unique()
g = sns.FacetGrid(cabin_df, col='Passengers', col_wrap=2, size=2.3, aspect=1.2, sharex=True)

g.map(plt.hist, 'Survival')

g.set_xlabels('')

g.fig.suptitle('Distribution of Survival Rate by Number of Passengers in a Unique Cabin', fontsize=9.5, y=1.01)

g.fig.text(.5, .05, 'Survival Rate', ha='center', fontsize=9.5)

g.fig.text(0, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=9.5)



plt.show()
fig, ax = plt.subplots(figsize=[4,2.5])

g = sns.barplot(x='Passengers', y='Survival', data=cabin_df, errwidth=0)

ax.set_title('Distribution of Survival Rate by \nNumber of Passengers in a Unique Cabin', fontsize=9.5, y=1.01)

ax.set_ylabel('Survival Rate')

ax.set_xlabel('Number of Passengers in a Unique Cabin')

plt.show()
child_survived_df = titanic_df[(titanic_df.Age <= 8) & (titanic_df.Parch >= 1) & (titanic_df.Survived == 1)]

child_not_survived_df = titanic_df[(titanic_df.Age <= 8) & (titanic_df.Parch >= 1) & (titanic_df.Survived == 0)]
def find_parents(child_df, i):

    child = child_df.iloc[i]

    child_ticket = child.Ticket

    return titanic_df[(titanic_df.Ticket == child_ticket) & (titanic_df.Age >= (child.Age + 18))]
par_child_survived_df = pd.DataFrame()

for i in range(len(child_survived_df)):

    par_child_survived_df = par_child_survived_df.append(find_parents(child_survived_df, i))



par_child_not_survived_df = pd.DataFrame()

for i in range(len(child_not_survived_df)):

    par_child_not_survived_df = par_child_not_survived_df.append(find_parents(child_not_survived_df, i))
surv_rate_par_ch = pd.DataFrame()

surv_rate_par_ch['ChSurvived'] = [0,1]

surv_rate_par_ch['ParSurvival'] = [par_child_not_survived_df.Survived.mean(), par_child_survived_df.Survived.mean()]
fig, ax = plt.subplots(figsize=[4,2])

sns.barplot(x='ChSurvived', y='ParSurvival', data=surv_rate_par_ch, ax=ax)

ax.set_title('Survival Rate of Parents by Survival Rate of Children', fontsize=9.5)

ax.set_xlabel('Survival of Children')

ax.set_ylabel('Survival Rate of Parent(s)')

plt.show()