import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

plt.style.use('ggplot')
data = pd.read_csv('../input/top250-00-19.csv')
data.head()
league_from = data.groupby(['League_from'])['Transfer_fee'].sum()

top5sell_league = league_from.sort_values(ascending=False).head(5)

top5sell_league = top5sell_league/1000000

top5sell_league.head()
fig, ax = plt.subplots(figsize=(18,6))

ax.bar(top5sell_league.index, top5sell_league.values, color='orange')

ax.set_ylabel("$ millions", color='navy')

ax.set_yticklabels(labels=[i for i in range(0,8000, 1000)], color='navy')

ax.set_xticklabels(labels=top5sell_league.index, color='navy')
league_to = data.groupby(['League_to'])['Transfer_fee'].sum()

top5buy_league = league_to.sort_values(ascending=False).head(5)

top5buy_league = top5buy_league/1000000

top5buy_league.head()
fig, ax = plt.subplots(figsize=(18,6))

ax.bar(top5buy_league.index, top5buy_league.values, color='navy')

ax.set_ylabel("$ millions", color='black')

ax.set_yticklabels(labels=[i for i in range(0,16000, 2000)], color='black')

ax.set_xticklabels(labels=top5buy_league.index, color='black')
diff_league = top5sell_league - top5buy_league

diff_league = diff_league.sort_values(ascending=False)

diff_league.head()
fig, ax = plt.subplots(figsize=(18,6))

ax.bar(diff_league.index, diff_league.values)

ax.set_ylabel("$ millions")
league_summary = pd.concat([top5sell_league, top5buy_league], axis=1)

league_summary = league_summary.assign(diff=diff_league)

new_columns = league_summary.columns.values

new_columns[[0, 1]] = ['sell', 'buy']

league_summary.columns = new_columns

league_summary.head()
fig, ax = plt.subplots(figsize=(18,6))



sales = league_summary["sell"]

buys = league_summary["buy"]

x = league_summary.index

width=0.4

N = len(x)

loc = np.arange(N)



ax.bar(loc, sales, width, bottom=0, label="Sell")

ax.bar(loc+width, buys, width, bottom=0, label="Buy")



ax.set_title("Buys and sales by major leagues")

ax.legend()

ax.set_xticks(loc + width / 2)

ax.set_xticklabels(x)

ax.set_ylabel("$ millions")

ax.autoscale_view()
club_from_sum = data.groupby(['Team_from'])['Transfer_fee'].sum()

club_from_count = data.groupby(['Team_from'])['Transfer_fee'].count()

club_from_mean_price = (club_from_sum/1000000) / club_from_count
plt.figure(figsize=(18,6))

sellers_mean = club_from_mean_price.sort_values(ascending=False)[:20]

g = sns.barplot(sellers_mean.index, sellers_mean.values, palette="Greens_r")

g.set_title("Mean price of sold player per club")

g.set(ylabel="$ millions", xlabel="Team selling a player")

plt.xticks(rotation=90)
club_to_sum = data.groupby(['Team_to'])['Transfer_fee'].sum()

club_to_count = data.groupby(['Team_to'])['Transfer_fee'].count()

club_to_mean_price = (club_to_sum/1000000) / club_to_count
plt.figure(figsize=(18,6))

buy_mean = club_to_mean_price.sort_values(ascending=False)[:20]

g = sns.barplot(buy_mean.index, buy_mean.values, palette=sns.cubehelix_palette(20))

g.set_title("Mean price of bought player per club")

g.set(ylabel="$ millions", xlabel="Team buying a player")

plt.xticks(rotation=90)
diff_club = club_from_sum - club_to_sum

diff_club = diff_club.sort_values(ascending=False)

diff_club = diff_club.dropna()
diff_club = diff_club/1000000

diff_club.head(15)

# in millions
fig, ax = plt.subplots(figsize=(18,6))

make_money = diff_club.sort_values(ascending=False)[:10]

ax.bar(make_money.index, make_money.values, color="orange")

ax.set_title("Clubs that make money on transfer market")

ax.set_ylabel("$ millions")

ax.set_xticklabels(make_money.index, rotation=90)

# ax.autoscale_view()
diff_club.tail(15)

# in millions
fig, ax = plt.subplots(figsize=(18,6))

lose_money = diff_club.sort_values(ascending=True)[:10]

ax.bar(lose_money.index, lose_money.values, color="black")

ax.set_title("Clubs that lose money on transfer market")

ax.set_ylabel("$ millions")

ax.set_xticklabels(lose_money.index, rotation=90)

ax.autoscale_view()
club_from_sum = club_from_sum.sort_values(ascending=False)

club_from_sum = club_from_sum/1000000

club_from_sum.head(15)
plt.figure(figsize=(20,6))

g = sns.barplot(club_from_sum.head(15).index, club_from_sum.head(15).values, palette=sns.color_palette("hls", 15))

g.set_title("Top sales clubs")

g.set(ylabel="$, millions", xlabel="Team selling")
club_from_sum.tail(15)
plt.figure(figsize=(20,6))

g = sns.barplot(club_from_sum.tail(15).index, club_from_sum.tail(15).values, palette=sns.color_palette("Wistia_r", 15))

g.set_title("Bottom sales clubs")

g.set(ylabel="$, millions", xlabel="Team selling")
club_from_mean_price = club_from_mean_price.sort_values(ascending=False)

club_from_mean_price.head(15)

# in millions
plt.figure(figsize=(18,6))

g = sns.barplot(club_from_mean_price.head(15).index, club_from_mean_price.head(15).values, palette=sns.color_palette("YlGnBu_r", 15))

g.set_title("Mean price of a sale, Top")

g.set(ylabel="$, millions", xlabel="Team selling")
club_from_mean_price.tail(15)

# I should've put some minimum borderline of let's say 10 men moved
plt.figure(figsize=(18,6))

g = sns.barplot(club_from_mean_price.tail(15).index, club_from_mean_price.tail(15).values, palette=sns.color_palette("YlGnBu", 15))

g.set_title("Mean price of a sale, Bottom")

g.set(ylabel="$, millions", xlabel="Team selling")

plt.xticks(rotation=45)
club_to_sum = club_to_sum.sort_values(ascending=False)

club_to_sum = club_to_sum/1000000

club_to_sum.head(15)
plt.figure(figsize=(18,6))

g = sns.barplot(club_to_sum.head(15).index, club_to_sum.head(15).values, palette=sns.color_palette("OrRd_r", 15))

g.set_title("Total historical spend on players, Top")

g.set(ylabel="$, millions", xlabel="Team buing")

plt.xticks(rotation=45)
club_to_sum.tail(15)
plt.figure(figsize=(18,6))

g = sns.barplot(club_to_sum.tail(15).index, club_to_sum.tail(15).values, palette=sns.color_palette("OrRd", 15))

g.set_title("Total historical spend on players, Top")

g.set(ylabel="$, millions", xlabel="Team buing")

plt.xticks(rotation=45)
club_to_mean_price = club_to_mean_price.sort_values(ascending=False)

club_to_mean_price.head(15)
plt.figure(figsize=(18,6))

g = sns.barplot(club_to_mean_price.head(15).index, club_to_mean_price.head(15).values, palette=sns.color_palette("magma", 15))

g.set_title("Mean price of a bought player, Top")

g.set(ylabel="$, millions", xlabel="Team buing")

plt.xticks(rotation=45)
club_to_mean_price.tail(15)
plt.figure(figsize=(18,6))

g = sns.barplot(club_to_mean_price.tail(15).index, club_to_mean_price.tail(15).values, palette=sns.color_palette("magma_r", 15))

g.set_title("Mean price of a bought player, Top")

g.set(ylabel="$, millions", xlabel="Team buing")

plt.xticks(rotation=45)