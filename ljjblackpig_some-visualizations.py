import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.lines as mlines

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
james = pd.read_csv('../input/lebron_career.csv')
james.info()
james.head(5)
def Prepare_james(df):

    #Convert the date column into the datetime column (from object column)

    df['date'] = pd.to_datetime(df['date'])

    #Extracting year, month, and the day of the play date from the column

    df['play_year'], df['play_month'], df['play_day'] = df['date'].dt.year, df['date'].dt.month, df['date'].dt.day

    

    #Separate the age column with age and age subdays column, and combine them to make real-time age

    df[['age','age_subdays']]=df['age'].str.split('-',expand=True).replace(np.nan, 0).astype(int)

    df['age'] = df['age'] + df['age_subdays'] / 365

    

    #Deal with the minutes player column

    df['mp'] = pd.to_datetime(df['mp'], format = '%M:%S').dt.minute

    

    #Calculating overall shooting percentage

    df['overall_pct'] = (df['fg'] + df['three'] + df['ft']) / (df['fga'] + df['threeatt'] + df['fta'])

    

    #game rating from: https://www.basketball-reference.com/about/glossary.html#pf

    df['game_rating'] = 0.7 * df['orb'] + 0.3 * df['drb'] + 0.7 * df['ast'] + 0.7 * df['blk'] + df['stl'] - df['tov']

    

    return df
Prepare_james(james)
fig, ax = plt.subplots(figsize = (8, 6))

ax.plot(james.groupby(['age'])['pts'].mean())

plt.xlabel('Age')

plt.ylabel('Points per game')

plt.title('Lebron points per game by age')
sns.distplot(james.pts)
sns.jointplot(x = 'game_rating', y = 'pts', data = james, height = 8, ratio=4, color = "r")

plt.show()
sns.jointplot(x = 'threep', y = 'pts', data = james, height = 8, ratio=4, color = "b")

plt.show()
sns.jointplot(x = 'mp', y = 'pts', data = james, height = 8, ratio=4, color = "g")

plt.show()
fig, ax = plt.subplots(figsize = (8, 6))

ax.plot(james.groupby(['play_year'])['pts'].mean())

plt.xlabel('Year')

plt.ylabel('Points per game')

plt.title('Lebron points per game by year')
fig, ax = plt.subplots(figsize = (16, 6))

mean_pts = james['pts'].mean()

james['mean_row'] = mean_pts

ax.scatter(james['opp'], james['pts'])

mean_line = ax.plot(james['opp'], james['mean_row'] , label='Mean', color = 'r', linestyle='-')

ax.legend()

plt.title("Lebron Points vs each team in the league")
james_pred = james[['pts','mp', 'age', 'overall_pct', 'game_rating', 'minus_plus']]



colormap = plt.cm.RdBu

plt.figure(figsize=(14, 12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(james_pred.corr(), linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)