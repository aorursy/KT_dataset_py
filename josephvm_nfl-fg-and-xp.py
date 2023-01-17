import pandas as pd
total_post = pd.read_csv("../input/nfl-kickers-data/combined_post_1966_2019.csv")

total_post.head()
total_post.sort_values(by='Lng', ascending=False)[:10]
total_df = pd.read_csv("../input/nfl-kickers-data/combined_reg_1966_2019.csv")

total_df.drop(['Team'], axis=1, inplace=True)

total_df.set_index('Player', inplace=True)
sums = total_df.drop(['Lng'], axis=1).sum() # might not have teams

num_kickers = len(total_df)

fg_acc = sums['FGM'] / sums['FG Att']

xp_acc = sums['XPM'] / sums['XPA']

print("Total fg accuracy: ", 100 * fg_acc,"%")

print("Total xp accuracy: ", 100 * xp_acc,"%")

sums
total_df.sort_values(by='FGM', ascending=False)[:5]
total_df['FGAA'] = total_df['FGM'] - fg_acc * total_df['FG Att']

total_df['XPAA'] = total_df['XPM'] - xp_acc * total_df['XPA']

total_df.sort_values(by='FGAA', ascending=False)[:10]
acc_50 = sums['50+_M'] / sums['50+_A']

total_df['50+AA'] = total_df['50+_M'] - acc_50 * total_df['50+_A']

total_df['XPAA'] = total_df['XPM'] - xp_acc * total_df['XPA']

total_df.sort_values(by='50+AA', ascending=False)[:10]
acc_dict = {}

for year in range(1966, 2020):

    sums = pd.read_csv("../input/nfl-kickers-data/reg/reg_"+str(year)+".csv").sum()

    acc_dict[year] = sums['FGM'] / sums['FG Att']
acc_series = pd.Series(acc_dict)

acc_series.plot()