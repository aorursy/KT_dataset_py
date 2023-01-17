from IPython.display import display, Markdown

import os

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import pandas as pd



df = pd.read_csv('/kaggle/input/higher-or-lower-game/games.csv', parse_dates=['startTime', 'finishTime'])

df.dataframeName = 'higher-lower-data2'



stats = {

    ('captures', 'hours of gameplay'): int(df['duration'].sum() / 360),

    ('including', 'decisions'): df['numGuesses'].sum(),

    ('made by', 'players'): len(df['user'].unique()),

    ('across', 'games'): len(df)

}



for descrip, num in stats.items():

    display(Markdown(f"*{descrip[0]}*\n# {num:,}\n\n{descrip[1]}"))
mean_guesses_targetNum = df.groupby('targetNum', as_index=False)['numGuesses'].mean()

bins = [1,10,20,30,40,50,60,70,80,90,100]

df4 = mean_guesses_targetNum.groupby(pd.cut(mean_guesses_targetNum['targetNum'], bins=bins)).numGuesses.mean()

df4.plot(kind='bar', ylim=(6.5, 8.5), title='Av no. guesses per targetNum').set_ylabel("Av. no. guesses")
suboptimal_games_pct = len(df[df['numGuesses']>7]) / len(df)

print(f'{suboptimal_games_pct:.0%} of games were suboptimal')

non_50_first_guesses = len(df[(df.targetNum==50) & (df.guess1==50)]) / len(df[df.targetNum==50])

print(f'Only {non_50_first_guesses:.0%} of games where 50 was the target were guessed on the first attempt')

print()

print("Guessing optimally (mid-range) every time, you will never take more than 7 guesses...")

import math, time

results = []

numbers = [x for x in range(1,101)]

for x in numbers:

    a = 1

    b = 100

    guess = 50

    optimal_path = [50]

    while guess != x:

        if guess < x:

            a = guess

            guess = math.ceil((a + b) / 2)

        elif guess>x:

            b = guess

            guess = math.floor((a + b) / 2)

        optimal_path.append(guess)

    res = [x, len(optimal_path), optimal_path]

    print(f'Guessing #{x} takes {len(optimal_path)} guesses: {optimal_path}')

    results.append(res)



#res2 = pd.DataFrame(data=res[1:,1:],

#                   index=res[1:,0],

#                   columns=res[0,1:])



game_counts = df['user'].value_counts()

repeat_players = len(game_counts[game_counts>1])

series = pd.Series((repeat_players, len(game_counts) - repeat_players),

                   index=['repeat players', 'one-off players'], name='stickiness')

series.plot.pie(figsize=(6, 6))
mean_guesses_targetNum = df.groupby('targetNum', as_index=False)['numGuesses'].mean()

plt.figure(figsize=(40, 10))



mn = mean_guesses_targetNum['numGuesses']

clrs = ['red' if (x == max(mn)) else 'green' if (x == min(mn)) else 'grey' for x in np.array(mn)]

sns.barplot(x='targetNum', y='numGuesses', data=mean_guesses_targetNum, palette=clrs)



easiest_num, hardest_num = mn.idxmin() + 1, mn.idxmax() + 1

pct_diff = mn.max() / mn.min() * 100

print('#{} is {:.0f}% harder to guess than #{}'.format(hardest_num, pct_diff, easiest_num))

# Daily games played average

df['date'] = df['startTime'].map(lambda x: x.date())

grp_date = df.groupby('date')

games_by_date = pd.DataFrame(grp_date.size(), columns=['num_games'])

print(f"Av. daily no. games:\t\t{round(games_by_date['num_games'].mean())}")



print(f"Average games per player:\t{round(df['user'].value_counts().mean())}")

print(f"Average no. guesses:\t\t{df['numGuesses'].mean()}")



most_common_first_guess = df['guess1'].mode()[0]

print(f"Most common first guess:\t{most_common_first_guess} " +

     "({:.0%})".format(len(df[df['guess1']==most_common_first_guess]) / len(df))

     )

print()

print(f"Top 10 players:\nUser\tNo. games\n{df['user'].value_counts()[:10]}")
player_rank = df.groupby('user', as_index=False)['numGuesses'].mean().sort_values('numGuesses')



#df6 = df['user'].value_counts()

#print(df6)

#normalized_df=(df6-df6.min())/(df6.max()-df6.min())

#print(normalized_df)



#min_max_scaler = preprocessing.MinMaxScaler()

#x_scaled = min_max_scaler.fit_transform(games_played)

#df = pd.DataFrame(x_scaled)
sns.relplot(x="duration", y="numGuesses", data=df)
sns.distplot(df['numGuesses'], kde=False, bins=4).set_title('No. guesses')
plt.figure(figsize=(40, 10))

sns.countplot(x="targetNum", data=df)
# Distribution graphs (histogram/bar graph) of column data

def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):

    nunique = df.nunique()

    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values

    nRow, nCol = df.shape

    columnNames = list(df)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 90)

        plt.title(f'{columnNames[i]} (column {i})')

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    plt.show()

# Correlation matrix

def plotCorrelationMatrix(df, graphWidth):

    filename = df.dataframeName

    df = df.dropna('columns') # drop columns with NaN

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    if df.shape[1] < 2:

        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')

        return

    corr = df.corr()

    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')

    corrMat = plt.matshow(corr, fignum = 1)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

    plt.yticks(range(len(corr.columns)), corr.columns)

    plt.gca().xaxis.tick_bottom()

    plt.colorbar(corrMat)

    plt.title(f'Correlation Matrix for {filename}', fontsize=15)

    plt.show()

# Scatter and density plots

def plotScatterMatrix(df, plotSize, textSize):

    df = df.select_dtypes(include =[np.number]) # keep only numerical columns

    # Remove rows and columns that would lead to df being singular

    df = df.dropna('columns')

    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values

    columnNames = list(df)

    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots

        columnNames = columnNames[:10]

    df = df[columnNames]

    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')

    corrs = df.corr().values

    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):

        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)

    plt.suptitle('Scatter and Density Plot')

    plt.show()

df1 = df

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 20, 10)