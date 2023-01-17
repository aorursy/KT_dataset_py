import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df = pd.read_csv('../input/World Cup 2018 Dataset.csv')
df = df.dropna(axis='rows',how='all')
df = df.dropna(axis='columns',how='all')
df
df = df.fillna(0)
df.columns
df.info()
df['score'] = (df['Previous \nappearances'] + 4 * df['Previous\n semifinals'] + 8 * df['Previous\n finals'] + 16 * df['Previous \ntitles'])
df['group_win'] = (df['history with \nfirst opponent\n W-L'] + df['history with\n second opponent\n W-L'] + df['history with\n third opponent\n W-L'])
df['group_goals'] = (df['history with\n first opponent\n goals'] + df['history with\n second opponent\n goals'] + df['history with\n third opponent\n goals'])
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.barh(df.Team, df.score)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.barh(df.Team, df.group_win)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.barh(df.Team, df.group_goals)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

barWidth = 0.9

plt.bar(df.Team, df.score, width = barWidth, label='Total score')
plt.bar(df.Team, df.group_win, width = barWidth, label='Wins to opponenets')
plt.bar(df.Team, df.group_goals, width = barWidth, label='Goals against opponents')

plt.legend()
plt.xticks(rotation=90)
plt.subplots_adjust(bottom= 0.2, top = 0.98)
 
plt.show()
df_tree = df.drop(df.index[[13,25]])
import squarify

df_sub = df_tree.loc[(df_tree!=0).any(axis=1)]

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

squarify.plot(sizes=df_tree['Previous \nappearances'], label=df_tree['Team'])
plt.axis('off')
plt.title('Distribution of appearances')
plt.show()
df.describe()
df.head(4)
#Rank = (70 - df.Current FIFA rank)/7
#Score = (max = 200) /20
#wins(max = 18,min = -17) = (17 + df.group_win)/3.5
#Goals(max = 72,min = -72) = 72 + df.group_goals/14
#Appearance(max = 20) = df.Previous appearances / 2
#Round to integer
df_radar = pd.DataFrame({
    'group': ['Russia','Saudi Arabia','Egypt','Uruguay'],
    'Rank': [1, 1, 6, 7],
    'Score': [1, 0, 0, 4],
    'Wins': [5, 4, 6, 5],
    'Goals': [5, 5, 5, 5],
    'Appearance': [5, 2, 1, 6]
})
# Refer to https://python-graph-gallery.com/radar-chart/

from math import pi

def make_spider( row, title, color):
 
    # number of variable
    categories=list(df_radar)[1:]
    N = len(categories)
 
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
 
    # Initialise the spider plot
    ax = plt.subplot(2,2,row+1, polar=True, )
 
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
 
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
 
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([2,4,6,8], ["2","4","6","8"], color="grey", size=7)
    plt.ylim(0,10)
 
    # Ind1
    values=df_radar.loc[row].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)
 
    # Add a title
    plt.title(title, size=11, color=color, y=1.1)
 
    # ------- PART 2: Apply to all individuals
    # initialize the figure
    my_dpi=96
    plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
 
    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set2", len(df.index))
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
my_palette = plt.cm.get_cmap("Set2", len(df.index))
# Loop to plot
for row in range(0, len(df_radar.index)):
    make_spider(row=row, title='group' + df_radar['group'][row], color=my_palette(row))
df[4:8]
df_radar = pd.DataFrame({
    'group': ['Portugal','Spain','Morocco','Iran'],
    'Rank': [10, 9, 4, 5],
    'Score': [1, 2, 0, 0],
    'Wins': [2, 10, 3, 5],
    'Goals': [3, 8, 5, 5],
    'Appearance': [3, 7, 2, 2]
})
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for row in range(0, len(df_radar.index)):
    make_spider(row=row, title='group' + df_radar['group'][row], color=my_palette(row))
df[8:12]
df_radar = pd.DataFrame({
    'group': ['France','Australia','Peru','Denmark'],
    'Rank': [9, 5, 8, 8],
    'Score': [3, 0, 0, 0],
    'Wins': [6, 4, 5, 5],
    'Goals': [6, 5, 5, 5],
    'Appearance': [7, 2, 2, 2]
})
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for row in range(0, len(df_radar.index)):
    make_spider(row=row, title='group' + df_radar['group'][row], color=my_palette(row))
df[12:16]
df_radar = pd.DataFrame({
    'group': ['Argentina','Iceland','Croatia','Nigeria'],
    'Rank': [9, 7, 8, 3],
    'Score': [5, 0, 0, 0],
    'Wins': [6, 4, 5, 4],
    'Goals': [5, 5, 6, 5],
    'Appearance': [8, 0, 2, 3]
})
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for row in range(0, len(df_radar.index)):
    make_spider(row=row, title='group' + df_radar['group'][row], color=my_palette(row))
df[16:20]
df_radar = pd.DataFrame({
    'group': ['Brazil','Switzerland','Costarica','Serbia'],
    'Rank': [10, 9, 6, 5],
    'Score': [10, 1, 0, 1],
    'Wins': [8, 5, 3, 5],
    'Goals': [7, 5, 4, 5],
    'Appearance': [10, 5, 2, 6]
})
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for row in range(0, len(df_radar.index)):
    make_spider(row=row, title='group' + df_radar['group'][row], color=my_palette(row))
df[20:24]
df_radar = pd.DataFrame({
    'group': ['Germany','Mexico','Sweden','Korea'],
    'Rank': [10, 8, 7, 2],
    'Score': [10, 1, 2, 1],
    'Wins': [7, 4, 5, 4],
    'Goals': [7, 5, 5, 4],
    'Appearance': [9, 8, 6, 5]
})
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for row in range(0, len(df_radar.index)):
    make_spider(row=row, title='group' + df_radar['group'][row], color=my_palette(row))
df[24:28]
df_radar = pd.DataFrame({
    'group': ['Belgium','Panama','Tunisia','England'],
    'Rank': [9, 2, 6, 8],
    'Score': [1, 0, 0, 2],
    'Wins': [0, 5, 5, 10],
    'Goals': [0, 5, 5, 10],
    'Appearance': [6, 0, 2, 7]
})
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for row in range(0, len(df_radar.index)):
    make_spider(row=row, title='group' + df_radar['group'][row], color=my_palette(row))
df[28:32]
df_radar = pd.DataFrame({
    'group': ['Poland','Senegal','Columbia','Japan'],
    'Rank': [9, 7, 8, 2],
    'Score': [1, 0, 0, 0],
    'Wins': [4, 4, 4, 2],
    'Goals': [5, 5, 5, 4],
    'Appearance': [4, 1, 3, 3]
})
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for row in range(0, len(df_radar.index)):
    make_spider(row=row, title='group' + df_radar['group'][row], color=my_palette(row))
match = pd.DataFrame({
    'Team': ['France','Argentina','Uruguay','Portugal','Spain','Russia','Croatia','Denmark','Brazil','Mexico','Belgium','Japan','Switzerland','Sweden','Columbia','England'],
    'Appearance': [7,8,6,3,7,5,2,8,10,8,6,3,5,6,3,7],
    'Rank': [9,9,7,10,9,1,8,8,10,8,9,2,9,7,8,8],
    'Wins': [2,1,3,1,1,2,3,1,2,2,3,1,1,2,2,2],
    'GS': [3,7,5,5,6,8,7,2,5,3,9,4,5,5,5,8],
    'GA': [9,9,10,6,5,6,9,9,9,6,8,6,6,8,7,7]
})
# Refer to https://python-graph-gallery.com/radar-chart/

from math import pi

def make_spider( row, title, color):
 
    # number of variable
    categories=list(match)[1:]
    N = len(categories)
 
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
 
    # Initialise the spider plot
    ax = plt.subplot(2,8,row+1, polar=True, )
 
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
 
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
 
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([2,4,6,8], ["2","4","6","8"], color="grey", size=7)
    plt.ylim(0,10)
 
    # Ind1
    values=match.loc[row].drop('Team').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)
 
    # Add a title
    plt.title(title, size=11, color=color, y=1.4)
 
    # ------- PART 2: Apply to all individuals
    # initialize the figure
    my_dpi=96
    plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
 
    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set2", len(df.index))
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for row in range(0, len(match.index)):
    make_spider(row=row, title='group' + match['Team'][row], color=my_palette(row))
match = pd.DataFrame({
    'Team': ['France','Uruguay','Russia','Croatia','Brazil','Belgium','Sweden','England'],
    'Appearance': [7,6,5,2,10,5,6,7],
    'Rank': [9,7,1,8,10,9,7,8],
    'Wins': [2,3,2,3,2,1,2,2],
    'GS': [3,5,8,7,5,5,5,8],
    'GA': [9,10,6,9,9,6,8,7]
})
# Refer to https://python-graph-gallery.com/radar-chart/

from math import pi

def make_spider( row, title, color):
 
    # number of variable
    categories=list(match)[1:]
    N = len(categories)
 
    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
 
    # Initialise the spider plot
    ax = plt.subplot(2,4,row+1, polar=True, )
 
    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
 
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color='grey', size=8)
 
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([2,4,6,8], ["2","4","6","8"], color="grey", size=7)
    plt.ylim(0,10)
 
    # Ind1
    values=match.loc[row].drop('Team').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, values, color=color, alpha=0.4)
 
    # Add a title
    plt.title(title, size=11, color=color, y=1.4)
 
    # ------- PART 2: Apply to all individuals
    # initialize the figure
    my_dpi=96
    plt.figure(figsize=(1000/my_dpi, 1000/my_dpi), dpi=my_dpi)
 
    # Create a color palette:
    my_palette = plt.cm.get_cmap("Set2", len(df.index))
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

for row in range(0, len(match.index)):
    make_spider(row=row, title='group' + match['Team'][row], color=my_palette(row))
