import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import cm

import seaborn as sns
df = pd.read_excel("../input/premier-league-match-reports-20192020/premierLeague.xlsx")

df.drop(["Penalties", "PassingAccuracy"], axis=1, inplace=True)

df.head(10)
plt.figure(figsize=(14, 14))



mask = np.zeros_like(df.corr())

mask[np.triu_indices_from(mask)] = True

sns.heatmap(df.corr(), annot=True, vmin=-1, vmax=1, fmt=".1f", cmap="Greens", mask=mask, linewidth=0.1)



plt.show()
grouped = df[["Team", "Fouls", "YellowCard", "RedCard", "YellowRed"]].groupby("Team")

discipline = pd.DataFrame(grouped.agg(np.sum).sort_values(by="Fouls"))

discipline["Kicked"] = discipline["RedCard"] + discipline["YellowRed"]

discipline
fig = plt.figure(figsize=(16, 8))

ax = fig.add_subplot()



size = 50 + np.array(discipline["Kicked"].values)*50

cmap = plt.cm.get_cmap("autumn_r")

ax.scatter(x="Fouls", y="YellowCard", s=size, c="Kicked", cmap=cmap, data=discipline)



for column in ["Fouls", "YellowCard", "Kicked"]:

    txt_df = discipline[np.logical_or(discipline[column] == discipline[column].max(),

                                      discipline[column] == discipline[column].min())]

    for i, index in enumerate(txt_df.index):

        ax.text(x=txt_df["Fouls"].loc[index],

                y=txt_df["YellowCard"].loc[index]+2,

                s=txt_df.index[i],

                horizontalalignment='center',

                verticalalignment='bottom')



ax.set_xticks(np.arange(350, 601, 25))

ax.set_yticks(np.arange(30, 101, 10))

ax.set_xlabel("Fouls", fontsize=14)

ax.set_ylabel("Yellow Cards", fontsize=14)

ax.grid(alpha=0.3, linestyle="--")

ax.spines["right"].set_visible(False)

ax.spines["top"].set_visible(False)



map_ = plt.cm.ScalarMappable(cmap=cmap)

cbar = plt.colorbar(map_)

ticks = set(discipline["Kicked"].values)

cbar.set_ticks(np.linspace(0, 1, len(ticks)))

cbar.set_ticklabels(list(ticks))

cbar.set_label("Kicked", fontsize=14)



plt.show()
grouped = df[["Team", "ShotsAccuracy", "SavesAccuracy"]].groupby("Team")

grouped.agg(np.mean).sort_values(by="ShotsAccuracy")
Acc = df[["Team", "ShotsAccuracy", "SavesAccuracy"]].sort_values(by="Team").reset_index(drop=True)



fig = plt.figure(figsize=(20, 10))

spec = fig.add_gridspec(4, 5)

gen = ((i, ii) for i in range(4) for ii in range(5))



for i, team in enumerate(Acc["Team"].unique()):

    ax = fig.add_subplot(spec[next(gen)])

    data = Acc[Acc["Team"] == team]

    sns.distplot(data["ShotsAccuracy"].tolist(), color="blue", hist=False, label="Shot Accuracy")

    sns.distplot(data["SavesAccuracy"].tolist(), color="red", hist=False, label="Save Accuracy")

    ax.legend_.remove()

    ax.spines["bottom"].set_position("zero")

    list(map(lambda x: ax.spines[x].set_visible(False), ["left", "top" ,"right"]))

    ax.set_xticks(np.arange(-0.5, 1.51, 0.5))

    ax.set_yticks(np.arange(0, 4.1, 1))

    plt.setp(ax.get_xticklabels(), visible=False)

    ax.tick_params(axis='x', which='both', length=0)

    if (i % 5 != 0):

        plt.setp(ax.get_yticklabels(), visible=False)

        ax.tick_params(axis='y', which='both', length=0)

    ax.set_title(team)

    ax.hlines(y=np.arange(0, 4.1, 1), xmin=-0.5, xmax=1.5, alpha=0.3, linestyle="--", linewidth=1)

    

    means = [data["ShotsAccuracy"].mean(), data["SavesAccuracy"].mean()]

    ax.vlines(means, ymin=0, ymax=4, color=["blue", "red"], alpha=0.5)

    ax.text(means[0]-0.05, 4, round(means[0], 2), color="blue", va="top", ha="right")

    ax.text(means[1]+0.05, 4, round(means[1], 2), color="red", va="top", ha="left")

    

plt.suptitle("Shot&Save Accuracy", fontsize=16)

fig.text(0.1, 0.5, 'Density', ha='center', va='center', rotation='vertical', fontsize=14)

fig.legend(["Shot Accuracy", "Save Accuracy"], loc=(0.07, 0.95), ncol=2 )



plt.show()
grouped = df[["Team", "Score", "NumofShots", "SucShots"]].groupby("Team")

score = grouped.agg(np.sum).sort_values(by="Score", ascending=False)

score
fig = plt.figure(figsize=(16,8))

ax = fig.add_subplot()



size = np.array(score["Score"].values)*10

cmap = plt.cm.get_cmap("autumn_r")

ax.scatter(x="NumofShots", y="SucShots", c="Score", s=size, cmap=cmap, data=score)



ax.set_xticks(np.arange(300, 751, 50))

ax.set_yticks(np.arange(100, 261, 20))

ax.set_xlabel("Number of Shots", fontsize=12)

ax.set_ylabel("Successful Shots", fontsize=12)

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.grid(alpha=0.3, linestyle="--", linewidth=1)



for column in score.columns:

    df_txt = score.sort_values(by=column)

    for i in [0, 1, -1, -2]:

        ax.annotate(s=df_txt.index[i],

                    xy=(df_txt["NumofShots"].iloc[i], df_txt["SucShots"].iloc[i]),

                    xytext=(df_txt["NumofShots"].iloc[i]-30,

                            df_txt["SucShots"].iloc[i]+(15 if i>=0 else -15)),

                    arrowprops=dict(facecolor='black', width=1, headwidth=10),

                    horizontalalignment='center',

                    verticalalignment='bottom')



map_ = plt.cm.ScalarMappable(cmap=cmap)

cbar = plt.colorbar(map_)

cbar.set_ticks(np.linspace(0, 1, 6))

cbar.set_ticklabels(np.arange(25, 101, 15))

cbar.set_label("Scores", fontsize=14)



plt.show()