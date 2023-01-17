import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import math
df = pd.read_csv('../input/overwatch-league-with-odds/owl-with-odds.csv')
df.head(5)
team_list = df['team_one'].unique()

#Get a count of how many matches for each team

team_count = []

for t in team_list:

    df_team_one = df[df['team_one'] == t]

    df_team_two = df[df['team_two'] == t]

    team_count.append(len(df_team_one) + len(df_team_two))



print(team_count)

print(team_list)
plt.figure(figsize=(8,8))

plt.xticks(rotation=90)

sns.set()

sns.set(style="darkgrid")

ax = sns.barplot(y=team_count, x=team_list)
complete_over_under_list = []



for t in team_list:

    over_under = 0

    over_under_list = []

    df_team_one = df[df['team_one'] == t]

    df_team_two = df[df['team_two'] == t]

    df_team = df_team_one.append(df_team_two)

    id_list = df_team.id.unique()

    for i in id_list:

        winner = ((df_team[df_team['id'] == i]).winner).iloc[0]

        if winner == t:

            over_under = over_under + 1

        else:

            over_under = over_under - 1

        over_under_list.append(over_under)

    complete_over_under_list.append(over_under_list)
t1_over_under = complete_over_under_list[0]

#print(t1_over_under)

#print(team_list[0])


#fig = plt.figure(figsize=(12, 50)



fig, ax = plt.subplots(10, 2, figsize=(12,80))

for t_count in range(len(team_list)):

    r = math.floor(t_count / 2) 

    c = t_count % 2

    #print (r)

    #print (c)

    ax[r,c].plot(complete_over_under_list[t_count])

    ax[r,c].set(xlabel='Match Number', ylabel='Wins over/under even')

    ax[r,c].set_title(team_list[t_count] + " performance over time", fontweight='bold')
#Convert odds to integers and drop rows where we don't have information

subset = ['t1_odds', 't2_odds']

df.dropna(subset=subset ,inplace=True)



df['t1_odds'] = pd.to_numeric(df['t1_odds'], errors='coerce')

df['t2_odds'] = pd.to_numeric(df['t2_odds'], errors='coerce')

df.dropna(subset=subset ,inplace=True)
df['t1_odds'].describe()
df['t2_odds'].describe()
odds_list = []



id_list = df.id.unique()

for i in id_list:

    odds_list.append(((df[df['id'] == i]).t1_odds).iloc[0])

    odds_list.append(((df[df['id'] == i]).t2_odds).iloc[0])

        

fig2, ax2 = plt.subplots()

ax2.set_title("Odds Histogram", fontweight='bold')

ax2.set_xlabel("Odds")

ax2.set_ylabel("Count")

ax2.hist(odds_list)

fig1, ax1 = plt.subplots()

ax1.set_title('Odds Boxplot')

ax1.boxplot(odds_list)

ax1.set_ylabel('Odds')
odds_list_trimmed = [x for x in odds_list if -2000 <= x <= 2000]
print(len(odds_list_trimmed))

print(len(odds_list))
fig3, ax3 = plt.subplots(figsize=(12,10))

plt.xticks(rotation=90)

ax3.set_title("Odds (Trimmed) Histogram", fontweight='bold')

ax3.set_xlabel("Odds")

ax3.set_ylabel("Count")

ax3.hist(odds_list_trimmed, 40)

fig4, ax4 = plt.subplots()

ax4.set_title('Odds (Trimmed) Boxplot')

ax4.boxplot(odds_list_trimmed)

ax4.set_ylabel('Odds')
odds_labels = [

    "Under -2000",

    "-1000 -> -1999",

    "-750 -> -999",

    "-500 -> -749",

    "-450 -> -499",

    "-400 -> -449",

    "-350 -> -399",

    "-300 -> -349",

    "-250 -> -299",

    "-200 -> -249",

    "-150 -> -199",

    "-100 -> -149",

    "+100 -> +149",

    "+150 -> +199",

    "+200 -> +249",

    "+250 -> +299",

    "+300 -> +349",

    "+350 -> +399",

    "+400 -> +449",

    "+450 -> +499",

    "+500 -> +749",

    "+750 -> +999",

    "+1000 -> +1999",    

    

    

]
def return_win_percentage(f_df, f_min, f_max):

    winners = 0

    losers = 0

    #Filter the dataframe

    f_df_orig = f_df.copy()

    f_df = (((f_df[f_df['t1_odds'] >= f_min])))

    f_df = (((f_df[f_df['t1_odds'] < f_max])))

    f_df_2 = (((f_df_orig[f_df_orig['t2_odds'] >= f_min])))

    f_df_2 = (((f_df_2[f_df_2['t2_odds'] < f_max])))

    f_df = f_df.append(f_df_2)

    

    id_list = f_df.id.unique()

    for i in id_list:

        if (((f_df[f_df['id'] == i]).t1_odds).iloc[0]) >= f_min:

            if (((f_df[f_df['id'] == i]).t1_odds).iloc[0]) < f_max:

                if (((f_df[f_df['id'] == i]).team_one).iloc[0]) == (((f_df[f_df['id'] == i]).winner).iloc[0]):

                    winners = winners + 1

                else:

                    losers = losers + 1



        if (((f_df[f_df['id'] == i]).t2_odds).iloc[0]) >= f_min:

            if (((f_df[f_df['id'] == i]).t2_odds).iloc[0]) < f_max:

                if (((f_df[f_df['id'] == i]).team_two).iloc[0]) == (((f_df[f_df['id'] == i]).winner).iloc[0]):

                    winners = winners + 1

                else:

                    losers = losers + 1



                    

    #print(winners)

    #print(losers)

    #print(len(id_list))

        

    return (100*(winners/(winners+losers)))

#    print(len(f_df))

#    print(len(f_df_2))

#    working_df = f_df.append(f_df_2)

#    print(len(working_df))

#    display(working_df)
win_percentages = []



win_percentages.append(return_win_percentage(df, -100000, -2000))

win_percentages.append(return_win_percentage(df, -1999, -999))

win_percentages.append(return_win_percentage(df, -999, -749))

win_percentages.append(return_win_percentage(df, -749, -499))

win_percentages.append(return_win_percentage(df, -499, -449))

win_percentages.append(return_win_percentage(df, -449, -399))

win_percentages.append(return_win_percentage(df, -399, -349))

win_percentages.append(return_win_percentage(df, -349, -299))

win_percentages.append(return_win_percentage(df, -299, -249))

win_percentages.append(return_win_percentage(df, -249, -199))

win_percentages.append(return_win_percentage(df, -199, -149))

win_percentages.append(return_win_percentage(df, -149, -99))

win_percentages.append(return_win_percentage(df, 100, 149))

win_percentages.append(return_win_percentage(df, 150, 199))

win_percentages.append(return_win_percentage(df, 200, 249))

win_percentages.append(return_win_percentage(df, 250, 299))

win_percentages.append(return_win_percentage(df, 300, 349))

win_percentages.append(return_win_percentage(df, 350, 399))

win_percentages.append(return_win_percentage(df, 400, 449))

win_percentages.append(return_win_percentage(df, 450, 499))

win_percentages.append(return_win_percentage(df, 500, 549))

win_percentages.append(return_win_percentage(df, 750, 999))

win_percentages.append(return_win_percentage(df, 1000, 1999))





fig5, ax5 = plt.subplots(figsize=(12,10))

plt.xticks(rotation=90)

ax5.set_title('Win Percentage Base on Odds')

ax5.plot(odds_labels, win_percentages, marker='o')

ax5.set_ylabel('Odds')