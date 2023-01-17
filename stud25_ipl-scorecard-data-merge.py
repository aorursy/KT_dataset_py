import pandas as pd





df1 = pd.read_csv("../input/ipl-scorecard-data/ipl-2018-scorecard.csv")

df1.shape
df2 = pd.read_csv("../input/ipl-scorecard-data/ipl-2019-scorecard.csv")

df2.shape
res = pd.concat([df1, df2])

res
res.shape
print(sum([int(i) for i in res[res['player_name'] == "V Kohli"]["R"]]))

print(sum([int(i) for i in res[res['player_name'] == "SK Raina"]["R"]]))

print(sum([int(i) for i in res[res['player_name'] == "MS Dhoni"]["R"]]))

print(sum([int(i) for i in res[res['player_name'] == "RG Sharma"]["R"]]))

print(sum([int(i) for i in res[res['player_name'] == "S Dhawan"]["R"]]))
res.to_csv("ipl-scorecard-18-19.csv", index=False)



df = pd.read_csv("ipl-scorecard-18-19.csv")

df
g = df.groupby('id')
id = "8048_1136561_1_2018"

match_group = g.get_group(id)
player_group = match_group.groupby('player_name')

player_group.get_group('HH Pandya')
def calculate_points(records):

    def _strike_Rate(strike_rates):

        score = 0

        for sr in strike_rates:

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

    runs = sum([int(i) for i in records['R']])

    sr = _strike_Rate(records['SR'])

    fours = sum([int(i) * 2 for i in records['4s']])

    six = sum([int(i) * 4 for i in records['6s']])

    wickets = sum([int(i) * 10 for i in records['W']])

    econ = _econ_points(records['Econ'])

    maiden = sum([int(i) * 10 for i in records['M']])

    no_ball = sum([int(i) for i in records['Nb']])

    wide_ball = sum([int(i) for i in records['Wb']])

        

    res = res + runs + sr + fours + six + wickets + econ + maiden - no_ball - wide_ball

    return res

        



res = []

for player, records in player_group:

    temp = {}

    temp["player_name"] = player

    temp["points"] = calculate_points(records[['R', 'B', 'SR', '4s', '6s', 'Rb', 'W', 'Econ', 'M', 'Nb', 'Wb']])

    res.append(temp)



res
player_group.get_group("MA Wood")
calculate_points(player_group.get_group("MA Wood"))
res_df = pd.DataFrame(res)

res_df.sort_values(by=['points'], inplace=True, ascending=False)

res_df
res_df.sort_values(by=['points'], inplace=True, ascending=False)

res_df[0:10]