import os



import numpy as np

import pandas as pd

import seaborn as sns; sns.set()

import matplotlib.pyplot as plt

import matplotlib.patches as patches
df = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

print(df.shape)

df.head()
df.dtypes
(df.isna().sum() / df.shape[0]).nlargest(12)
def create_football_field(linenumbers=True,

                          endzones=True,

                          highlight_line=False,

                          highlight_line_number=50,

                          highlighted_name='Line of Scrimmage',

                          fifty_is_los=False,

                          figsize=(12, 6.33)):

    """

    Function that plots the football field for viewing plays.

    Allows for showing or hiding endzones.

    """

    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,

                             edgecolor='r', facecolor='darkgreen', zorder=0)



    fig, ax = plt.subplots(1, figsize=figsize)

    ax.add_patch(rect)



    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,

              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],

             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,

              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],

             color='white')

    if fifty_is_los:

        plt.plot([60, 60], [0, 53.3], color='gold')

        plt.text(62, 50, '<- Player Yardline at Snap', color='gold')

    # Endzones

    if endzones:

        ez1 = patches.Rectangle((0, 0), 10, 53.3,

                                linewidth=0.1,

                                edgecolor='r',

                                facecolor='blue',

                                alpha=0.2,

                                zorder=0)

        ez2 = patches.Rectangle((110, 0), 120, 53.3,

                                linewidth=0.1,

                                edgecolor='r',

                                facecolor='blue',

                                alpha=0.2,

                                zorder=0)

        ax.add_patch(ez1)

        ax.add_patch(ez2)

    plt.xlim(0, 120)

    plt.ylim(-5, 58.3)

    plt.axis('off')

    if linenumbers:

        for x in range(20, 110, 10):

            numb = x

            if x > 50:

                numb = 120 - x

            plt.text(x, 5, str(numb - 10),

                     horizontalalignment='center',

                     fontsize=20,  # fontname='Arial',

                     color='white')

            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),

                     horizontalalignment='center',

                     fontsize=20,  # fontname='Arial',

                     color='white', rotation=180)

    if endzones:

        hash_range = range(11, 110)

    else:

        hash_range = range(1, 120)



    for x in hash_range:

        ax.plot([x, x], [0.4, 0.7], color='white')

        ax.plot([x, x], [53.0, 52.5], color='white')

        ax.plot([x, x], [22.91, 23.57], color='white')

        ax.plot([x, x], [29.73, 30.39], color='white')



    if highlight_line:

        hl = highlight_line_number + 10

        plt.plot([hl, hl], [0, 53.3], color='yellow')

        plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),

                 color='yellow')

    return fig, ax
sample = '20181230154135'

fig, ax = create_football_field()

df.query(f"PlayId == {sample} and Team == 'away'").plot(x='X', y='Y', kind='scatter', ax=ax, color='orange', s=30, legend='Away')

df.query(f"PlayId == {sample} and Team == 'home'").plot(x='X', y='Y', kind='scatter', ax=ax, color='blue', s=30, legend='Home')

plt.title(f'Play # {sample}')

plt.legend()

plt.show()
df['PlayId'].value_counts().describe()
df.groupby(['PlayId']).agg({'Dis': 'sum'})['Dis'].nlargest(20).plot(kind='bar', figsize=(20, 5))
df.groupby(['PlayId']).agg({'S': 'mean'})['S'].nlargest(20).plot(kind='bar', figsize=(20, 5))
df['GameId'].value_counts().nlargest(20).plot(kind='bar', figsize=(20, 5))
df['GameId'].value_counts().describe()
numeric_df = df.select_dtypes('number').drop(['GameId', 'PlayId', 'X', 'Y'], axis=1)

print(numeric_df.columns)

numeric_df.head()
(numeric_df / numeric_df.max()).boxplot(figsize=(20, 5), rot=90)
(numeric_df / numeric_df.max()).boxplot(figsize=(20, 5), rot=90)
sns.distplot(numeric_df['Yards'])
sns.distplot(numeric_df['S'])
sns.distplot(numeric_df['A'])
corr = numeric_df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
cat_df = df.select_dtypes('object').drop(['TimeHandoff'], axis=1)

print(cat_df.columns)

cat_df.head()