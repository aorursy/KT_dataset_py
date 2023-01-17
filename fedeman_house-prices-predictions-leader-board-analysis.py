import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Defaults

import matplotlib as mpl

mpl.rcParams['figure.figsize'] = [16.0, 6.0]
df = pd.read_csv("/kaggle/input/house-price-predictions-public-leader-board/20200806_house-prices-advanced-regression-techniques-publicleaderboard.csv")

df.head()
competitors = df['TeamId'].value_counts().count()

entries = df['TeamId'].value_counts().sum()



print("{} competitors submitted a total of {} times".format(competitors, entries))
# Keep only the best scores per TeamId

min_scores = df.groupby('TeamId').min()['Score'].sort_values().reset_index()
top100 = min_scores['Score'][99]

print("To be in the top 100, your Score should be lower than {:.6f}".format(top100))
top10prc_idx = int(len(min_scores['Score'])/10)

top10prc = min_scores['Score'][top10prc_idx]

print("To be in the 10%, your Score should be lower than {:.6f}".format(top10prc))
min_scores.sample(10)
# Slice the sample

min_scores = min_scores[min_scores['Score'] <=1]
f, ax = plt.subplots()

g = sns.scatterplot(data=min_scores, x='TeamId', y='Score', ax=ax)

g.set(ylim=(-0.01, 1.1))

plt.title("Scores by TeamId", size=18)

sns.despine(ax=ax, bottom=True, left=True)

ax.axhline(top10prc, c='orange', ls='--')

ax.axhline(top100, c='r', ls='--')

ax.legend(["top 10%", "top 100 positions"])
f1, ax1 = plt.subplots()

g1 = sns.scatterplot(data=min_scores, x='TeamId', y='Score', ax=ax1, x_jitter=True)

g1.set(ylim=(-0.01, top10prc*1.1))

plt.title("Scores by TeamId in detail of the top performances", size=18)

ax1.axhline(top10prc, c='orange', ls='--')

ax1.axhline(top100, c='r', ls='--')

ax1.legend(["top 10%", "top 100 positions"])

sns.despine(ax=ax1, bottom=True, left=True)
f2, ax2 = plt.subplots()

plt.title("Percentual score distribution for top scores (<1) in Kaggle competition", size=18)

plt.ylabel("Percentual value")



# Generate bins

bins_plt = np.arange(0,1,0.02)

out = pd.cut(min_scores['Score'], bins=bins_plt, include_lowest=True)

# Generate normalized distribution and plot

out_norm = out.value_counts(sort=False, normalize=True).mul(100)

out_norm.plot.bar(rot=90, ax=ax2)



# Replace x tick labels

bins = out.cat.categories

bins_tuples = [str(c)[1:-1].replace(",", " < ") for c in bins]

ax2.set_xticklabels(bins_tuples)





ax2.text(x=17,y=5,s="(I'm here)")

plt.arrow(x=17,y=4, dx=0, dy=-3)



ax2.text(x=33,y=5,s='(I was here)')

plt.arrow(x=33,y=4, dx=0, dy=-3)



ax2.axvline(6, c='orange', ls='--')

ax2.axvline(5, c='red', ls='--')

ax2.text(x=2,y=30,s="Top 100\npositions", c='r')

ax2.text(x=7,y=30,s="Top 10%", c='orange')



sns.despine(ax=ax2, bottom=True, left=True)



plt.show()