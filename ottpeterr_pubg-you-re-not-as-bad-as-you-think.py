# environment preparations
import numpy as np
import pandas as pd

kills_paths = []
kills_paths += ["../input/aggregate/agg_match_stats_0.csv"]
kills_paths += ["../input/aggregate/agg_match_stats_1.csv"]
kills_paths += ["../input/aggregate/agg_match_stats_2.csv"]
# kills_paths += ["../input/aggregate/agg_match_stats_3.csv"]
# kills_paths += ["../input/aggregate/agg_match_stats_4.csv"]

# these are the columns we care about, leaving out the data we won't use
col_filter = [
#                 'match_id',
                'party_size', # 1, 2, 4
#                 'match_mode', # fpp, tpp - theyre all tpp
#                 'player_name',
                'player_kills',
                'team_placement',
                'player_dmg',
                  ]


# combine all the data files into one array
kills = None
for kill_file_path in kills_paths:
    new_kills = pd.read_csv(kill_file_path, usecols=col_filter)
    kills = pd.concat([kills, new_kills])
    
    
# Filtering the data

# solo
kills_solo=kills[kills['party_size']==1]
# kills_duo=kills[kills['party_size']==2]
kills_squad=kills[kills['party_size']==4]

kills_solo.drop(columns=['party_size'])
# kills_duo.drop(columns=['party_size'])
# kills_squad.drop(columns=['party_size'])

# Take a sample
# sample_size = 7500
# kills_solo = kills_solo.sample(sample_size)
# kills_duo = kills_duo.sample(sample_size)
# kills_squad = kills_squad.sample(sample_size)

#save some memory
del kills

print(len(kills_solo))
kills_solo.head()
rank_and_kills = kills_solo[['team_placement', 'player_kills']]
plot = rank_and_kills.plot.scatter(x='team_placement', y='player_kills', color='green')
plot.set_xlabel("rank")
plot.set_ylabel("kills")
plot.grid(color='black', axis=['x', 'y'], linestyle='solid')
# format is ("name", <=, >)
groups = [("Q1", 100,75), ("Q2", 75, 50), ("Q3", 50,25),("Q4", 25,0),("T10", 10,0),("T7", 7,0), ("T5", 5,0), ("T3", 3,0), ("Winner", 1,0)]
print("    Kills\t\t   Damage\n    average\tmedian\t   average\tmedian")
for (name, lte, gt) in groups:
    mean_kills = kills_solo[(kills_solo['team_placement'] > gt) & (kills_solo['team_placement'] <= lte)]['player_kills'].mean()
    median_kills = kills_solo[(kills_solo['team_placement'] > gt) & (kills_solo['team_placement'] <= lte)]['player_kills'].median()
    mean_dmg = kills_solo[(kills_solo['team_placement'] > gt) & (kills_solo['team_placement'] <= lte)]['player_dmg'].mean()
    median_dmg = kills_solo[(kills_solo['team_placement'] > gt) & (kills_solo['team_placement'] <= lte)]['player_dmg'].median()
    print(name+": "+str(mean_kills)[0:5]+"  \t"+str(median_kills)+"\t   "+str(mean_dmg)[0:5]+"  \t"+str(median_dmg))
my_hist = kills_solo[kills_solo['team_placement'] <= 5]['player_kills'].hist(bins=25, range=[0,25], color='green')
my_hist.set_title("Top 5")
my_hist.set_xlabel("num kills")
my_hist.set_ylabel("count")
max_rank = 25
avg_kills = [0]+list(range(1,max_rank+1))
stddev_kills = [0]+list(range(1,max_rank+1))

for rank in avg_kills:
    k = kills_solo[kills_solo['team_placement'] == rank]['player_kills']
    avg_kills[rank] = k.mean()
    stddev_kills[rank] = k.std()

d = {"average kills": avg_kills}
df = pd.DataFrame(data=d)
p = df.plot(yerr=stddev_kills, color='green')
p.set_ylabel("kill count")
p.set_xlabel("rank")
# ignore the error message about a NaN
# it's because there's no data for players ranked as 0th place
kill_count = list(range(0,10))
print("kills\tavg rank")
for k in kill_count:
    print(str(k)+"\t"+str(kills_solo[kills_solo['player_kills'] == k]['team_placement'].mean())[0:5])
print(">"+str(k)+"\t"+str(kills_solo[kills_solo['player_kills'] > kill_count[-1]]['team_placement'].mean())[0:5])
winners = kills_solo[kills_solo['team_placement'] == 1]
first_place_kills = winners['player_kills']
my_hist = first_place_kills.plot.hist(bins=25, range=(0,25), color='gold')
my_hist.set_xlabel("kills")
my_hist.grid(color='black', axis='x', linestyle='solid')
dmg_per_kill = winners['player_dmg'] / winners['player_kills']
dmg_per_kill = dmg_per_kill.replace(np.inf, -4)
plot = dmg_per_kill.plot.hist(bins=102, range=(-4,200), color='gold')
plot.grid(color='black', axis='x', linestyle='solid')
plot.set_xlabel("damage dealt / number of kills")
plot = winners['player_dmg'].plot.hist(bins=50, range=(0,3000), color='gold')
plot.grid(color='black', axis='x', linestyle='solid')
plot.set_xlabel("total damage dealt")