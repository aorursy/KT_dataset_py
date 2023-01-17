import numpy as np 

import pandas as pd 

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import MultiLabelBinarizer

from matplotlib import pyplot as plt

sns.set_palette('husl')

mlb = MultiLabelBinarizer()
data= pd.read_csv("/kaggle/input/social-power-nba/nba_2016_2017_100.csv")

data.head(10)
data.describe()
res = pd.DataFrame(mlb.fit_transform(data), columns=mlb.classes_)

corr = data.loc[:,["AGE", "GP" , "W" , "L" , "MIN" , "PIE" , "AST_PCT" , "AST_TO" ,  "W_PCT_RANK" , "PTS" , "REB_PCT" ,"PACE" , "PIE" , "FGM" , "FGA" , "SALARY_MILLIONS" ]].corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(10,10))

cmap = sns.diverging_palette(180, 220, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
most_popular = data.sort_values(by="SALARY_MILLIONS",ascending=False)[:10]

display (most_popular)
top10 = most_popular[0:10]

display(top10)
plt.figure(figsize=(8,7))

ax= sns.scatterplot(x="OFF_RATING" , y="DEF_RATING", data= data, color="red")

plt.annotate("Russell Westbrook", xy=(107.9, 104.6), xytext=(95, 104), arrowprops=dict(facecolor="black", shrink=0.05))

plt.annotate("James Harden", xy=(113.6, 107.3), xytext=(119, 111), arrowprops=dict(facecolor="black", shrink=0.05))

plt.annotate("Kevin Durant", xy=(117.2, 101.3), xytext=(119, 93), arrowprops=dict(facecolor="black", shrink=0.05))

plt.annotate("Lebron James", xy=(114.9, 107.1), xytext=(119, 104), arrowprops=dict(facecolor="black", shrink=0.05))

plt.annotate("Damian Lillard", xy=(110.1, 108.9), xytext=(112, 115), arrowprops=dict(facecolor="black", shrink=0.05))

plt.annotate("Dirk Nowitzki", xy=(104.8, 106.5), xytext=(90, 110), arrowprops=dict(facecolor="black", shrink=0.05))

plt.annotate("Carmelo Anthony", xy=(106.1, 111.1), xytext=(90, 113), arrowprops=dict(facecolor="black", shrink=0.05))

plt.title("RELACIÓN OFENSIVA-DEFENSIVA",color='black', fontsize=20)
plt.figure(figsize=(15,7))

sns.barplot(top10["PLAYER_NAME"], data["FGA"],linewidth=1,edgecolor="k"*len(data),color="skyblue",label="Field Goals Attempted")

sns.barplot(top10["PLAYER_NAME"],data["FGM"],linewidth=1, edgecolor="k"*len(data),color="orangered",label="Field Goals Made")

plt.legend(loc="best",prop={"size":10})

plt.title("EFECTIVIDAD EN EL TIRO",color='black', fontsize=20)

plt.ylabel("# de Tiros")

plt.show()
colors = ['gold', 'mediumturquoise']

fig = go.Figure(data=[go.Pie(labels=['Win', 'Lost'],values=[46,35])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20, marker=dict(colors=colors, line=dict(color='#000000', width=3)))

fig.update_layout(title_text="Russell Westbrook")

fig.show()



colors = ['red', 'blue']

fig = go.Figure(data=[go.Pie(labels=['Win', 'Lost'],values=[54,27])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text="James Harden")

fig.show()



colors = ['crimson', 'sandybrown']

fig = go.Figure(data=[go.Pie(labels=['Win', 'Lost'],values=[51,11])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text="Kevin Durant")

fig.show()



colors = ['g', 'salmon']

fig = go.Figure(data=[go.Pie(labels=['Win', 'Lost'],values=[51,23])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text="Lebron James")

fig.show()





colors = ['dodgerblue', 'magenta']

fig = go.Figure(data=[go.Pie(labels=['Win', 'Lost'],values=[35,34])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text="Mike Conley")

fig.show()



colors = ['chartreuse', 'orange']

fig = go.Figure(data=[go.Pie(labels=['Win', 'Lost'],values=[38,37])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text="Damian Lillard")

fig.show()



colors = ['forestgreen', 'goldenrod']

fig = go.Figure(data=[go.Pie(labels=['Win', 'Lost'],values=[47,27])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text="Demar DeRozam")

fig.show()



colors = ['aqua', 'hotpink']

fig = go.Figure(data=[go.Pie(labels=['Win', 'Lost'],values=[46,22])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text="Al Horford")

fig.show()



colors = ['indigo', 'pink']

fig = go.Figure(data=[go.Pie(labels=['Win', 'Lost'],values=[23,31])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text="Dirk Nowitzki")

fig.show()



colors = ['slategray', 'tomato']

fig = go.Figure(data=[go.Pie(labels=['Win', 'Lost'],values=[29,45])])

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title_text="Carmelo Anthony")

fig.show()

plt.figure(figsize=(15,7))

barWidth = 0.2

 

bars1 = [0.388 ,0.543,0.505,0.218,0.329,0.277,0.204,0.239,0.099,0.141]

bars2 = [0.127,0.167,0.123,0.137,0.061,0.075,0.083,0.118,0.14,0.092]

bars3 = [0.619,0.554,0.613,0.651,0.604,0.586,0.552,0.553,0.529,0.535]

    

v1 = np.arange(len(bars1))

v2 = [x + barWidth for x in v1]

v3 = [x + barWidth for x in v2]

 

plt.bar(v1, bars1, color='lawngreen', width=barWidth, edgecolor='white', label='ASSIST')

plt.bar(v2, bars2, color='orange', width=barWidth, edgecolor='white', label='REBOUNDS')

plt.bar(v3, bars3, color='c', width=barWidth, edgecolor='white', label='POINTS')



plt.xlabel('PLAYERS', fontweight='bold')

plt.xticks([r + barWidth for r in range(len(bars1))], ["Lebron James" , "Rusell Westbrook", "James Harden", "Kevin Durant", "Mikey Conley", "Damian Lillard", "DeMar DeRozan", "Al Horford", "Dirk Nowitzki", "Carmelo Anthony"])

 

plt.title("TRIPLE DOUBLE",color='black', fontsize=20)

plt.legend()

plt.show()

        