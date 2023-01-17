# kernel adapted from https://www.kaggle.com/residentmario/welcome-to-data-visualization
# loading data

import pandas as pd

reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)

reviews.head(3)
reviews['province'].value_counts().head(10).plot.bar()
# percentage instead of raw numbers

(reviews['province'].value_counts().head(10) / len(reviews)).plot.bar()
# what about using bar chart for numeric feature

reviews['points'].value_counts().sort_index().plot.bar()
reviews['points'].value_counts().sort_index().plot.line()
reviews['points'].value_counts().sort_index().plot.area()
reviews[reviews['price'] < 200]['price'].plot.hist()
reviews['price'].plot.hist()
reviews[reviews['price'] > 1500]
reviews['points'].plot.hist()
reviews[reviews['price'] < 100].sample(100).plot.scatter(x='price', y='points')
# when number of data points gets larger

reviews[reviews['price'] < 100].plot.scatter(x='price', y='points')
reviews[reviews['price'] < 100].plot.hexbin(x='price', y='points', gridsize=15)
wine_counts = pd.read_csv("../input/most-common-wine-scores/top-five-wine-score-counts.csv",

                          index_col=0)

wine_counts.head()
wine_counts.plot.bar(stacked=True)
wine_counts.plot.area()
wine_counts.plot.line()
import pandas as pd

pd.set_option('max_columns', None)

df = pd.read_csv("../input/fifa-18-demo-player-dataset/CompleteDataset.csv", index_col=0)



import re

import numpy as np



footballers = df.copy()

footballers['Unit'] = df['Value'].str[-1]

footballers['Value (M)'] = np.where(footballers['Unit'] == '0', 0, 

                                    footballers['Value'].str[1:-1].replace(r'[a-zA-Z]',''))

footballers['Value (M)'] = footballers['Value (M)'].astype(float)

footballers['Value (M)'] = np.where(footballers['Unit'] == 'M', 

                                    footballers['Value (M)'], 

                                    footballers['Value (M)']/1000)

footballers = footballers.assign(Value=footballers['Value (M)'],

                                 Position=footballers['Preferred Positions'].str.split().str[0])
footballers.head()
import seaborn as sns



sns.lmplot(x='Value', y='Overall', hue='Position', 

           data=footballers.loc[footballers['Position'].isin(['ST', 'RW', 'LW'])], 

           fit_reg=False)
sns.lmplot(x='Value', y='Overall', markers=['o', 'x', '*'], hue='Position',

           data=footballers.loc[footballers['Position'].isin(['ST', 'RW', 'LW'])],

           fit_reg=False

          )
f = (footballers

         .loc[footballers['Position'].isin(['ST', 'GK'])]

         .loc[:, ['Value', 'Overall', 'Aggression', 'Position']]

    )

f = f[f["Overall"] >= 80]

f = f[f["Overall"] < 85]

f['Aggression'] = f['Aggression'].astype(float)



sns.boxplot(x="Overall", y="Aggression", hue='Position', data=f)
f = (

    footballers.loc[:, ['Acceleration', 'Aggression', 'Agility', 'Balance', 'Ball control']]

        .applymap(lambda v: int(v) if str.isdecimal(v) else np.nan)

        .dropna()

).corr()



sns.heatmap(f, annot=True)
from pandas.plotting import parallel_coordinates



f = (

    footballers.iloc[:, 12:17]

        .loc[footballers['Position'].isin(['ST', 'GK'])]

        .applymap(lambda v: int(v) if str.isdecimal(v) else np.nan)

        .dropna()

)

f['Position'] = footballers['Position']

f = f.sample(200)



parallel_coordinates(f, 'Position')
import matplotlib.pyplot as plt

# import dataset 

titanic = pd.read_csv("../input/titanic/train.csv")

titanic.head(10)
len(titanic)
(titanic['Survived'].value_counts()/len(titanic)).sort_index().plot.bar()

plt.title('Percentage of Survivors ')

tit2 = titanic

tit2['Agebin'] = pd.cut(x=tit2['Age'], bins=[0, 15, 65,120], labels=['Children', 'Adult', 'Seniors'])

tit2.head(10)
sns.countplot(x='Agebin',hue="Survived", data=tit2).set_title("Distribution of Passagens by Survival and Age Group")
sns.countplot(x='Sex',hue="Survived", data=tit2).set_title("Distribution of Passagens by Gender and Age Group")
sns.countplot(x='Pclass', hue="Survived", data=tit2).set_title("Distribution of Passagens by Survival and Class")
sns.lmplot(x='Age', y='Fare', hue='Pclass', markers=['o', 'x', '*'],

           data=tit2[tit2['Fare'] <=300], 

           fit_reg=False)