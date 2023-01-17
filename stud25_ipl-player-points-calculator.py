import pandas as pd

df = pd.read_csv("../input/ipl-20172019-scorecard-data/ipl-scorecard-17-18-19.csv")
df = df[["id", "innings_1", "innings_2", "winner"]]
df
match_group = df.groupby('id')
res = []
for mid, group in match_group:
    temp = {}
    temp["id"] = mid
    temp["innings_1"] = group.iloc[0]["innings_1"]
    temp["innings_2"] = group.iloc[0]["innings_2"]
    temp["winner"] = group.iloc[0]["winner"]
    res.append(temp)

#res

final_agg = pd.DataFrame(res)
final_agg
final_agg.to_csv("ipl-match-id-to-teams.csv", index=False)

agg = pd.read_csv("ipl-match-id-to-teams.csv")
agg
df = pd.read_csv("../input/ipl-20172019-scorecard-data/ipl-scorecard-17-18-19.csv")
df.fillna(0,inplace=True)
g = df.groupby('id')
g.get_group("8048_1082591_1_2017")
match_id_to_team = agg
match_id_to_team
def calculate_points(records):
    def _strike_Rate(strike_rates):
        score = 0
        for sr in strike_rates:
            if not sr:
                continue
            if sr == "-":
                continue
            if float(sr) == 0:
                continue
            if float(sr) < 50:
                score = score - 20
            elif float(sr) < 100:
                score = score - 10
            elif float(sr) < 150:
                score = score + 10
            elif float(sr) > 150:
                score = score + 20
            elif float(sr) > 200:
                score = score + 30
        return score

    def _econ_points(econs):
        score = 0
        for e in econs:
            if not e:
                continue
            if int(e) == 0:
                continue
            if int(e) < 6:
                score = score + 20
            elif int(e) < 7:
                score = score + 5
            elif int(e) > 7:
                score = score - 5
        return score

    res = 0
    runs = sum([int(i) for i in records['R'] if i and i != "-"])
    sr = _strike_Rate(records['SR'])
    fours = sum([int(i) * 2 for i in records['4s'] if i and i != "-"] or [0])
    six = sum([int(i) * 4 for i in records['6s']])
    wickets = sum([int(i) * 10 for i in records['W']])
    econ = _econ_points(records['Econ'])
    maiden = sum([int(i) * 10 for i in records['M']])
    no_ball = sum([int(i) for i in records['Nb']])
    wide_ball = sum([int(i) for i in records['Wb']])

    res = res + runs + sr  + six + fours + wickets + econ + maiden - no_ball - wide_ball
    return res

    
agg_points = []
for idx in match_id_to_team["id"]:
    temp_agg = {}
    match_group = g.get_group(idx)
    player_group = match_group.groupby('player_name')
    res = []
    for player, records in player_group:
        temp = {}
        temp["player_name"] = player
        temp["points"] = calculate_points(records[['R', 'B', 'SR', '4s', '6s', 'Rb', 'W', 'Econ', 'M', 'Nb', 'Wb']])
        res.append(temp)
    temp_agg["id"] = idx
    temp_agg["player_performance"] = res
    agg_points.append(temp_agg)
len(agg_points)
final_df = pd.DataFrame(agg_points)
final_df
final_csv = final_df.to_csv("ipl-agg-points.csv", index=False)
df = pd.read_csv("ipl-agg-points.csv")
df
res = []
for agg in agg_points:
    match_id = agg["id"]
    performance = agg["player_performance"]
    for p in performance:
        temp = {}
        temp["match_id"] = match_id
        temp["player_name"] = p["player_name"]
        temp["points"] = p["points"]
        res.append(temp)

#res
    
flat_df = pd.DataFrame(res)
flat_df
flat_df.to_csv("ipl-player-match-to-points.csv", index=False)

flat_df_1 = pd.read_csv("ipl-player-match-to-points.csv")
flat_df_1
df = flat_df_1
df[df["player_name"] == "V Kohli"].head()
## Top 20 players

df.pivot_table(index="player_name", aggfunc="sum").sort_values(['points'], ascending=False)[0:20]