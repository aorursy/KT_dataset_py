import pandas as pd

df = pd.read_csv("../input/ipl-scorecard-data/ipl-scorecard-18-19.csv")
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

res

final_agg = pd.DataFrame(res)
final_agg
final_agg.to_csv("ipl-match-id-to-teams.csv", index=False)

agg = pd.read_csv("ipl-match-id-to-teams.csv")
agg