import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="white")



from sklearn.preprocessing import LabelEncoder
df = pd.read_csv('../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv',

                 index_col='PassengerId')

df
df = df.drop(['Firstname', 'Lastname'], axis=1)
df.isna().apply(pd.value_counts)  
df.info()
print(df.Country.value_counts())



ax = sns.barplot(x=df['Country'].value_counts().keys(),

            y=df['Country'].value_counts().values)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.show()
print(df.Sex.value_counts())



ax = sns.barplot(x=df.Sex.value_counts().keys(),

            y=df.Sex.value_counts().values)
# print(df.Age.value_counts())



print('Min:', df.Age.min(),

      '\nMax:', df.Age.max())

val_count = df.Age.value_counts()

plt.title('Age')

plt.plot([i for i in range(df.Age.min(), df.Age.max() + 1)],

         [val_count[i] if i in val_count else 0 for i in range(df.Age.min(), df.Age.max() + 1)],

        linewidth = 3)



plt.minorticks_on()

plt.grid(which='major',

        color = 'k', 

        linewidth = 1)

plt.grid(which='minor', 

        color = 'k', 

        linestyle = ':')
sns.boxplot(x='Age', data=df)
print(df.Category.value_counts())



ax = sns.barplot(x=df.Category.value_counts().keys(),

            y=df.Category.value_counts().values)
print(df.Survived.value_counts())



ax = sns.barplot(x=df.Survived.value_counts().keys(),

            y=df.Survived.value_counts().values)
# converts categorical features to integers

def label_encoder(data_: pd.DataFrame(), columns_name_: list):

    le = LabelEncoder()

    for i in columns_name_:

        le.fit(data_[i])

        data_[i] = le.transform(data_[i])

    return data_



df = label_encoder(df, ['Country', 'Sex', 'Category'])
corr = df.corr()

f, ax = plt.subplots(figsize=(10, 10))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=None, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# fix error "Selected KDE bandwidth is 0"

sns.distributions._has_statsmodels = False

sns.pairplot(df, hue="Survived", palette="husl")