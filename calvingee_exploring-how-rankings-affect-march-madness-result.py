import pandas as pd

pd.plotting.register_matplotlib_converters()

import matplotlib.pyplot as plt

import seaborn as sns

import math

import operator



cbb_data = pd.read_csv("/kaggle/input/ncaa-basketball-top-25-ranked-teams-20082019/CBB Preseason Rankings.csv")
print(cbb_data.loc[cbb_data['Round'] == 7])
print(cbb_data.loc[cbb_data['Rank'] == 1])
q=sns.lmplot(x='Rank', y='Round', hue='Year', data=cbb_data, height=6, aspect=2)

q.set(xlim=(0,26), ylim=(-1,8))

q.set_axis_labels("Preseason Ranking", "Round Lost in Tournament")

q.fig.suptitle("Performance of Preseason Top Ranked Teams in Tournamnet", fontsize = 18)
s=sns.lmplot(x='Rank',y='Round', data=cbb_data.loc[cbb_data['Year'] == 2018])

s.set_axis_labels("PR", "RL")

s.fig.suptitle("Performance of Preseason Top Ranked Teams - 2018", fontsize = 18)
t=sns.lineplot(x='Year',y='Round',data=cbb_data.loc[cbb_data['Team'] == 'DUKEDuke'], label='Duke')

plt.ylabel("RL")

plt.title("Tournament Finish Over the Years", fontsize = 18)

t.set(xlim=(2009,2018))
sns.lineplot(x='Year',y='Round',data=cbb_data.loc[cbb_data['Team'] == 'DUKEDuke'], label='Duke')

r=sns.lineplot(x='Year',y='Round',data=cbb_data.loc[cbb_data['Rank'] == 1], label='#1 Ranked Team')

plt.ylabel("RL")

plt.title("Tournament Finish Over the Years", fontsize = 18)

r.set(xlim=(2009,2018))
results = {}

count = {}



for x in range(len(cbb_data)):

    name = cbb_data['Team'][x]

    exp_finish = 7 - math.log(cbb_data['Rank'][x],2)

    result = cbb_data['Round'][x] - exp_finish

    if name in results:

        results[name] += result

        count[name] += 1

    else:

        results[name] = result

        count[name] = 1



avg_results = {}

for x in results:

    avg_results[x] = results[x] / count[x]



best_team = max(avg_results.items(), key=operator.itemgetter(1))[0]

print (best_team + " with the best average finish of " + str(avg_results[best_team]) + " in " + str(count[best_team]) + " appearance(s).")
print ("Teams ranked by best average finish with 4 or more appearances:")

for x in sorted(avg_results, key = avg_results.__getitem__, reverse = True):

    if count[x] >= 4:

        print ("{}{}{}{}{}".format(x,': Avg = ',avg_results[x], ', count = ', count[x]))
BUT_expected = pd.DataFrame({'Round': 7 - math.log(cbb_data['Rank'][i],2), 'Year': cbb_data['Year'][i]} for i in range(len(cbb_data)) if cbb_data['Team'][i] == 'BUTButler')

sns.lineplot(x = 'Year', y = 'Round', data=cbb_data.loc[cbb_data['Team'] == 'BUTButler'], label = 'Performance')

sns.lineplot(x='Year', y = 'Round', data= BUT_expected, label = 'Expected Finish')

plt.title("Butler Expected Finish vs. Performance", fontsize = 18)

plt.ylim(0,7)
WIS_expected = pd.DataFrame({'Round': 7 - math.log(cbb_data['Rank'][i],2), 'Year': cbb_data['Year'][i]} for i in range(len(cbb_data)) if cbb_data['Team'][i] == 'WISWisconsin')

sns.lineplot(x = 'Year', y = 'Round', data=cbb_data.loc[cbb_data['Team'] == 'WISWisconsin'], label = 'Performance')

sns.lineplot(x='Year', y = 'Round', data= WIS_expected, label = 'Expected Finish')

plt.title("Wisconsin Expected Finish vs. Performance", fontsize = 18)

plt.ylim(0,7)
print (cbb_data.loc[cbb_data['Team'] == 'WISWisconsin'])