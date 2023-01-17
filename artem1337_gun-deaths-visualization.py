import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="white")



from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('../input/gun-deaths-in-the-united-states-20122014/guns-data.csv', index_col='Unnamed: 0')

df
df.isna().apply(pd.value_counts)  
print(df.sex.value_counts())



sns.barplot(x=df.sex.value_counts().keys(),

            y=df.sex.value_counts().values)

plt.show()
def visualize_age(target_name):

    def plot_age():

        plt.plot([i for i in range(df.age.min(), df.age.max() + 1)],

             [val_count[i] if i in val_count else 0 for i in range(df.age.min(), df.age.max() + 1)],

            linewidth = 3)



    df.age = df.age.fillna(0)

    df.age = df.age.astype(int)

    legend = ['Total']

    val_count = df.age.value_counts()

    plot_age()

    for place in df[target_name].value_counts().keys():

        legend += [place]

        val_count = df[df[target_name] == place].age.value_counts()

        plot_age()



    plt.legend(legend, bbox_to_anchor=(1., 0., 0., 1))

    plt.minorticks_on()

    plt.grid(which='major',

            color = 'k', 

            linewidth = 1)

    plt.grid(which='minor', 

            color = 'k', 

            linestyle = ':')

    plt.xlabel('age')

    plt.ylabel('amount')

    plt.show()



visualize_age('place')

visualize_age('race')

visualize_age('sex')

visualize_age('education')
sns.boxenplot(x='age', data=df)
for race in np.unique(df.race.value_counts().keys()):

    plt.title(race)

    ax = sns.barplot(x=df[df.race == race].education.value_counts().keys(),

                     y=df[df.race == race].education.value_counts().values)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.show()
print(df.intent.value_counts())



ax = sns.barplot(x=df['intent'].value_counts().keys(),

                 y=df['intent'].value_counts().values)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.show()
sex = 'M'

sns.set_color_codes("pastel")

sns.barplot(x=df[df.sex == sex].intent.value_counts().keys(),

            y=df[df.sex == sex].intent.value_counts().values, color="b")

sex = 'F'

sns.set_color_codes("muted")

sns.barplot(x=df[df.sex == sex].intent.value_counts().keys(),

            y=df[df.sex == sex].intent.value_counts().values, color="b")
for police in np.unique(df.police.value_counts().keys()):

    plt.title(f'Police: {police}')

    ax = sns.barplot(x=df[df.police == police].intent.value_counts().keys(),

                     y=df[df.police == police].intent.value_counts().values)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.show()
for race in np.unique(df.race.value_counts().keys()):

    plt.title(race)

    ax = sns.barplot(x=df[df.race == race].intent.value_counts().keys(),

                     y=df[df.race == race].intent.value_counts().values)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.show()
sns.boxplot(x='intent', y='age', data=df)
print(df.year.value_counts())



ax = sns.barplot(x=df.year.value_counts().keys(),

            y=df.year.value_counts().values)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.show()
for type_ in np.unique(df.intent.value_counts().keys()):

    plt.title(type_)

    ax = sns.barplot(x=df[df.intent == type_].year.value_counts().keys(),

                     y=df[df.intent == type_].year.value_counts().values)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    plt.show()
legend = []

for type_ in np.unique(df.intent.value_counts().keys()):

    legend += [type_]

    plt.plot([month for month in range(1, 12+1)],

             [df[df.intent == type_].month.value_counts()[i] for i in range(1, 12+1)], linewidth=3)

plt.title('Total deaths in each month')

plt.minorticks_on()

plt.grid(which='major',

        color = 'k', 

        linewidth = 1)

plt.grid(which='minor', 

        color = 'k', 

        linestyle = ':')

plt.legend(legend, bbox_to_anchor=(1., 0., 0., 1))

plt.show()
df = df.fillna(0)

df.isna().apply(pd.value_counts)