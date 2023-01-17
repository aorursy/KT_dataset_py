import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import plotly.plotly as pl

def compile_csv(startYear, endYear):
    df_list = []
    for i in range(startYear, endYear + 1):
        off_filename = "../input/" + str(i) + 'Offense.csv'
        def_filename = "../input/" + str(i) + 'Defense.csv'
        off_df = pd.read_csv(off_filename)
        off_df["Unit"] = ["Offense" if pos != "LS" else "Special" for pos in off_df["Pos"]]
        def_df = pd.read_csv(def_filename)
        def_df["Unit"] = ["Defense" if (pos != "K" and pos != "P") else "Special" for pos in def_df["Pos"]]
        df_list.append(off_df)
        df_list.append(def_df)
    data = pd.concat(df_list)
    return data

df = compile_csv(2000, 2017)
df["Player"] = [x.split("\\")[0] for x in df["Player"]]

#parse out Drafted (tm/rnd/yr) column
df["Drafted (tm/rnd/yr)"] = df["Drafted (tm/rnd/yr)"].where(pd.notnull(df["Drafted (tm/rnd/yr)"]), None)
df["DraftTeam"] = [x.split(" / ")[0] if x != None else None for x in df["Drafted (tm/rnd/yr)"]]
df["DraftRd"] = [x.split(" / ")[1] if x != None else None for x in df["Drafted (tm/rnd/yr)"]]
df["DraftRd"] = df["DraftRd"].str.replace('[a-zA-Z]+', '')
df["DraftPick"] = [x.split(" / ")[2] if x != None else None for x in df["Drafted (tm/rnd/yr)"]]
df["DraftPick"] = df["DraftPick"].str.replace('[a-zA-Z_]+', '')
df = df.drop(["Drafted (tm/rnd/yr)"], axis=1)

df
weight_year = df[['Year', 'Wt']].groupby(['Year']).mean()
weight_year
fig = plt.figure()
ax = fig.gca()
ax.plot(weight_year.index, weight_year['Wt'])
plt.show()
def convert_height(x):
    feet = x.split("-")[0]
    inches = x.split("-")[1]
    height = (int(feet) * 12) + int(inches)
    return height
df['Height'] = df['Height'].apply(convert_height)
df
df_nonan = df.dropna(axis=0)
df_nonan.describe()
# lose almost all the data points - too few players participate in all drills
df.describe()
df2 = df[['Pos', 'Height', 'Wt', '40YD', 'Vertical', 'BenchReps', 'Broad Jump', '3Cone', 'Shuttle','Unit', 'DraftRd']]
df2['DraftRd'] = df2['DraftRd'].fillna(0)
df2['BenchReps'] = df2['BenchReps'].fillna(-1)
df2 = df2.fillna('-1')
df2 = df2.apply(pd.to_numeric, errors='ignore')
df2
from sklearn.preprocessing import MinMaxScaler

plot_data = df2
scaler = MinMaxScaler()
plot_data[['40YD', 'Vertical', 'BenchReps', 'Broad Jump', '3Cone', 'Shuttle']] = scaler.fit_transform(plot_data[['40YD', 'Vertical', 'BenchReps', 'Broad Jump', '3Cone', 'Shuttle']])
# print(plot_data.Pos.unique())
# plot40
fig, axes = plt.subplots(8, 2, figsize=(20, 50))
cols = 0
rows = 0
for pos in plot_data.Pos.unique():
#     print("Pos: {}; grid: ({}, {})".format(pos, rows, cols))
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'black', 'orange', 'brown']
    for i in range(0, 8):
        axes[rows, cols].plot(plot_data[(plot_data['Pos'] == pos) & (plot_data['DraftRd'] == i)]['Vertical'], plot_data[(plot_data['Pos'] == pos)  & (plot_data['DraftRd'] == i)]['40YD'], 'o', color=colors[i], label="Round " + str(i), alpha=0.5)
        axes[rows, cols].set_title("Position: " + pos)
    axes[0, 0].legend()
    
    rows = rows + 1
    if cols == 1:
        if rows > 7:
            break
    if rows > 7:
        cols = cols + 1
        rows = 0
        continue
plt.tight_layout(pad=0.8)
plt.show()
df2
