import pandas as pd



bball_projects = pd.read_csv("/kaggle/input/nba-stat-projections-2019/FantasyPros_Fantasy_Basketball_Overall_2019_Projections.csv", index_col =0)

df = bball_projects
df = (df.assign(FPTS = lambda x : x.PTS + x.REB * 1.1 + x.AST * 1.5 + (x.BLK + x.STL)* 2 - (((x.PTS - (3 * x['3PM']))/2)/x['FG%']) - x.TO * 2))
df = df.sort_values(by = 'FPTS', ascending = False)

print(df)