import pandas as pd
fifa = pd.read_csv("../input/fifa-20-complete-player-dataset/players_20.csv")
keep = ["long_name", "age", "height_cm", "weight_kg", "nationality", "club", "overall", "value_eur", "wage_eur", "player_positions", "preferred_foot", "weak_foot", "skill_moves", "work_rate", "pace", "shooting", "passing", "dribbling", "defending", "physic", "gk_diving", "gk_handling", "gk_kicking", "gk_reflexes", "gk_speed", "gk_positioning"]
fifa = fifa.loc[:, keep]
fifa.head(10)
sum(fifa.duplicated()) # no duplication
fifa["age"].unique()
fifa["height_cm"].unique() 
fifa["weight_kg"].unique()
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(fifa.isnull(), cbar = False)
main_pos = fifa["player_positions"].str.split(",").str[0]
fw = (main_pos == "ST") | (main_pos == "CF")
forward = fifa[fw].head(2500)
forward = forward[["long_name", "overall", "weak_foot", "skill_moves", "work_rate", "pace", "shooting", "passing", "dribbling", "defending", "physic"]]
forward.head(10)
import pandas as pd
dummies = pd.get_dummies(forward.work_rate)
forward = pd.concat([forward, dummies], axis = 1)
X = forward.drop(["long_name", "overall", "work_rate"], axis = 1)
Y = forward["overall"]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 35)
# Backward Elimination

import statsmodels.api as sm
X_train = sm.add_constant(X_train)
cols = list(X_train.columns)

while len(cols) > 0:
    X_train = X_train[cols]
    model = sm.OLS(Y_train, X_train).fit()
    pvals = pd.Series(model.pvalues.values, index = cols)
    fmaxp = pvals.idxmax()
    if (pvals[fmaxp] > 0.05) & (fmaxp != "const"):
        cols.remove(fmaxp)
    else:
        break

print(cols)
opt = ["const", "pace", "shooting", "dribbling", "physic"]
X_opt_train = X_train[opt]

model = sm.OLS(Y_train, X_opt_train).fit()
model.summary()
X_opt_train = X_opt_train.drop(columns = "const")

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_opt_train, Y_train)
X_opt_test = X_test[["pace", "shooting", "dribbling", "physic"]]
regressor.score(X_opt_test, Y_test)
models = ["E(Y) = 0.8473 + 0.0275*pace + 0.5914*shooting + 0.2655*dribbling + 0.1255*physic", "E(Y) = -3.039 + 0.1301*pace + 0.1965*shooting + 0.2676*passing + 0.4199*dribbling + 0.0339*physic", "E(Y) = 1.4226 + 0.0458*pace + 0.1848*shooting + 0.3425*passing + 0.3914*dribbling + 0.0383*physic", "E(Y) = 1.019 + 0.0622*shooting + 0.3936*passing + 0.3356*dribbling + 0.1658*defending + 0.0684*physic", "E(Y) = -0.5254 + 0.1233*pace + 0.0839*shooting + 0.3808*passing + 0.3701*dribbling + 0.0656*physic", "E(Y) = 1.5289 + 0.2279*passing + 0.1272*dribbling + 0.5359*defending + 0.1306*physic", "E(Y) = 1.4127 + 0.0390*passing + 0.0452*dribbling + 0.7300*defending + 0.1852*physic", "E(Y) = 0.5984 + 0.1364*pace + 0.1502*passing + 0.1090*dribbling + 0.5709*defending + 0.0593*physic", "E(Y) = -0.9113 + 0.2430*gk_diving + 0.2154*gk_handling + 0.0543*gk_kicking + 0.2497*gk_reflexes + 0.2570*gk_positioning"]
test = [0.96, 0.985, 0.97, 0.96, 0.97, 0.975, 0.98, 0.97, 0.99]
tt_size = ["1875/625", "525/175", "750/250", "1500/500", "1500/500", "1050/350", "2250/750", "1950/650", "1500/500"]
top3 = ["shooting, dribbling, physic", "dribbling, passing, shooting", "dribbling, passing, shooting", "passing, dribbling, defending", "passing, dribbling, pace", "defending, passing, physic", "defending, physic, dribbling", "defending, passing, pace", "gk_positioning, gk_reflexes, gk_diving"]
index = ["Forward(ST/CF)", "Winger(LW/RW)", "Attacking Midfielder(CAM)", "Central Midfielder(CM)", "Side Midfielder(LM/RM)", "Defensive Midfielder(CDM)", "Centre Back(CB)", "Full Back(LWB, RWB, LB, RB)", "Goalkeeper(GK)"]
d = {"position category": index, "final model": models, "test set score (approx)": test, "train/test size": tt_size, "top 3 attributes": top3}
table = pd.DataFrame(data = d)
table.set_index("position category", inplace = True)
pd.set_option('display.max_colwidth', 150)
table
fig = plt.figure(figsize = (20, 20))
hist_col = ["age", "height_cm", "weight_kg", "overall", "pace", "shooting", "passing", "dribbling", "defending", "physic", "gk_diving", "gk_handling", "gk_kicking", "gk_reflexes", "gk_speed", "gk_positioning"]

d = {}
for i in range(16):
    d["ax{0}".format(i)] = fig.add_subplot(4, 4, i+1)
    d["ax{0}".format(i)] = sns.distplot(fifa[hist_col[i]].dropna())
    d["ax{0}".format(i)].axvline(fifa[hist_col[i]].mean(), color = 'black')
    d["ax{0}".format(i)].set_xlabel(hist_col[i], fontsize = 14)
fig = plt.figure(figsize = (12, 4))
bar_col = ["preferred_foot", "weak_foot", "skill_moves"]

d = {}
for i in range(3):
    d["ax{0}".format(i)] = fig.add_subplot(1, 3, i+1)
    d["ax{0}".format(i)] = sns.countplot(fifa[bar_col[i]], color = "grey")
    
plt.tight_layout()
fig = plt.figure(figsize = (15, 15))
cor = fifa.drop(columns = ["long_name", "nationality", "club", "player_positions", "preferred_foot", "work_rate"]).corr()
ax = sns.heatmap(cor, vmin = -1, vmax = 1, center = 0, cmap = sns.diverging_palette(20, 220, n = 200), annot = True, square = True)
map_d = {'RW': 'Winger(LW, RW)', 'ST': 'Forward(ST, CF)', 'LW': 'Winger(LW, RW)', 'GK': 'Goalkeeper(GK)', 'CAM': 'Central Attacking Midfielder(CAM)', 'CB': 'Centre Back(CB)', 'CM': 'Central Midfielder(CM)', 'CDM': 'Central Defensive Midfielder(CDM)', 'CF': 'Forward(ST, CF)', 'LB': 'Full Back(LB, RB, LWB, RWB)', 'RB': 'Full Back(LB, RB, LWB, RWB)', 'RM': 'Side Midfielder(LM, RM)', 'LM': 'Side Midfielder(LM, RM)', 'LWB': 'Full Back(LB, RB, LWB, RWB)', 'RWB': 'Full Back(LB, RB, LWB, RWB)'}
main_pos_c = main_pos.map(map_d)
fifa["position_category"] = main_pos_c
# top 20
import numpy as np
world_c = fifa.groupby("position_category").head(20)
wcavg = world_c.pivot_table(values = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning'], index = 'position_category', aggfunc = np.mean)

# 75th percentile
def third_q(df):
    x = round(df.shape[0]*0.25)
    return df.iloc[(x-10):(x+10), ]

top_quantile = fifa.groupby("position_category").apply(third_q)
top_quantile = top_quantile.droplevel("position_category")
topqavg = top_quantile.pivot_table(values = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning'], index = 'position_category', aggfunc = np.mean)

# median
def median(df):
    x = round(df.shape[0]*0.5)
    return df.iloc[(x-10):(x+10), ]

mid_quantile = fifa.groupby("position_category").apply(median)
mid_quantile = mid_quantile.droplevel("position_category")
midqavg = mid_quantile.pivot_table(values = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_reflexes', 'gk_speed', 'gk_positioning'], index = 'position_category', aggfunc = np.mean)
# Yeah I know this sucks, but Plotly lacks lots of support, including allowing for-looping for subplots. 

import plotly.graph_objects as go
from plotly.subplots import make_subplots

pos = ['Forward(ST, CF)', 'Winger(LW, RW)', 'Central Attacking Midfielder(CAM)', 'Central Midfielder(CM)', 'Side Midfielder(LM, RM)',
      'Central Defensive Midfielder(CDM)', 'Centre Back(CB)', 'Full Back(LB, RB, LWB, RWB)', 'Goalkeeper(GK)'] 

spec = [[{'type':'polar'}, {'type':'polar'}, {'type':'polar'}],
        [{'type':'polar'}, {'type':'polar'}, {'type':'polar'}],
        [{'type':'polar'}, {'type':'polar'}, {'type':'polar'}]]

fig = make_subplots(rows = 3, cols = 3, start_cell = "top-left", shared_xaxes = "all", shared_yaxes = "all", specs = spec, subplot_titles = pos)

atts = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
gatts = ["gk_diving", "gk_handling", "gk_kicking", "gk_reflexes", "gk_speed", "gk_positioning"]

for i in fig['layout']['annotations']: 
    i['font'] = dict(size = 14)

fig.add_trace(go.Scatterpolar(r = list(wcavg.loc[pos[0], atts]), theta = atts, fill = 'toself', name = 'world-class'), row = 1, col = 1)
fig.add_trace(go.Scatterpolar(r = list(topqavg.loc[pos[0], atts]), theta = atts, fill = 'toself', name = '75th percentile'), row = 1, col = 1)
fig.add_trace(go.Scatterpolar(r = list(midqavg.loc[pos[0], atts]), theta = atts, fill = 'toself', name = 'median'), row = 1, col = 1)

fig.add_trace(go.Scatterpolar(r = list(wcavg.loc[pos[1], atts]), theta = atts, fill = 'toself', name = 'world-class'), row = 1, col = 2)
fig.add_trace(go.Scatterpolar(r = list(topqavg.loc[pos[1], atts]), theta = atts, fill = 'toself', name = '75th percentile'), row = 1, col = 2)
fig.add_trace(go.Scatterpolar(r = list(midqavg.loc[pos[1], atts]), theta = atts, fill = 'toself', name = 'median'), row = 1, col = 2)

fig.add_trace(go.Scatterpolar(r = list(wcavg.loc[pos[2], atts]), theta = atts, fill = 'toself', name = 'world-class'), row = 1, col = 3)
fig.add_trace(go.Scatterpolar(r = list(topqavg.loc[pos[2], atts]), theta = atts, fill = 'toself', name = '75th percentile'), row = 1, col = 3)
fig.add_trace(go.Scatterpolar(r = list(midqavg.loc[pos[2], atts]), theta = atts, fill = 'toself', name = 'median'), row = 1, col = 3)

fig.add_trace(go.Scatterpolar(r = list(wcavg.loc[pos[3], atts]), theta = atts, fill = 'toself', name = 'world-class'), row = 2, col = 1)
fig.add_trace(go.Scatterpolar(r = list(topqavg.loc[pos[3], atts]), theta = atts, fill = 'toself', name = '75th percentile'), row = 2, col = 1)
fig.add_trace(go.Scatterpolar(r = list(midqavg.loc[pos[3], atts]), theta = atts, fill = 'toself', name = 'median'), row = 2, col = 1)

fig.add_trace(go.Scatterpolar(r = list(wcavg.loc[pos[4], atts]), theta = atts, fill = 'toself', name = 'world-class'), row = 2, col = 2)
fig.add_trace(go.Scatterpolar(r = list(topqavg.loc[pos[4], atts]), theta = atts, fill = 'toself', name = '75th percentile'), row = 2, col = 2)
fig.add_trace(go.Scatterpolar(r = list(midqavg.loc[pos[4], atts]), theta = atts, fill = 'toself', name = 'median'), row = 2, col = 2)

fig.add_trace(go.Scatterpolar(r = list(wcavg.loc[pos[5], atts]), theta = atts, fill = 'toself', name = 'world-class'), row = 2, col = 3)
fig.add_trace(go.Scatterpolar(r = list(topqavg.loc[pos[5], atts]), theta = atts, fill = 'toself', name = '75th percentile'), row = 2, col = 3)
fig.add_trace(go.Scatterpolar(r = list(midqavg.loc[pos[5], atts]), theta = atts, fill = 'toself', name = 'median'), row = 2, col = 3)

fig.add_trace(go.Scatterpolar(r = list(wcavg.loc[pos[6], atts]), theta = atts, fill = 'toself', name = 'world-class'), row = 3, col = 1)
fig.add_trace(go.Scatterpolar(r = list(topqavg.loc[pos[6], atts]), theta = atts, fill = 'toself', name = '75th percentile'), row = 3, col = 1)
fig.add_trace(go.Scatterpolar(r = list(midqavg.loc[pos[6], atts]), theta = atts, fill = 'toself', name = 'median'), row = 3, col = 1)

fig.add_trace(go.Scatterpolar(r = list(wcavg.loc[pos[7], atts]), theta = atts, fill = 'toself', name = 'world-class'), row = 3, col = 2)
fig.add_trace(go.Scatterpolar(r = list(topqavg.loc[pos[7], atts]), theta = atts, fill = 'toself', name = '75th percentile'), row = 3, col = 2)
fig.add_trace(go.Scatterpolar(r = list(midqavg.loc[pos[7], atts]), theta = atts, fill = 'toself', name = 'median'), row = 3, col = 2)
      
fig.add_trace(go.Scatterpolar(r = list(wcavg.loc[pos[8], gatts]), theta = gatts, fill = 'toself', name = 'world-class'), row = 3, col = 3)
fig.add_trace(go.Scatterpolar(r = list(topqavg.loc[pos[8], gatts]), theta = gatts, fill = 'toself', name = '75th percentile'), row = 3, col = 3)
fig.add_trace(go.Scatterpolar(r = list(midqavg.loc[pos[8], gatts]), theta = gatts, fill = 'toself', name = 'median'), row = 3, col = 3)

fig.update_layout(height = 1000, width = 1000, title_text = "World-Class v. Top Quarter v. Median", showlegend = False)
epl_teams = ["Manchester City", "Manchester United", "Tottenham Hotspur", "Liverpool", "Chelsea", "Arsenal", "Burnley", "Everton", "Leicester City", "Newcastle United", "Crystal Palace", "Bournemouth", "West Ham United", "Watford", "Brighton & Hove Albion", "Huddersfield Town", "Southampton", "Cardiff City", "Fulham", "Wolverhampton Wanderers"]
laliga_teams = ["FC Barcelona", "Atlético Madrid", "Real Madrid", "Valencia CF", "Villarreal CF", "Real Betis", "Sevilla FC", "Getafe CF", "SD Eibar", "Girona FC", "RCD Espanyol", "Real Sociedad", "RC Celta", "Deportivo Alavés", "Levante UD", "Athletic Club de Bilbao", "CD Leganés", "Real Valladolid CF", "Rayo Vallecano", "SD Huesca"]
bundesliga_teams = ["FC Bayern München", "FC Schalke 04", "TSG 1899 Hoffenheim", "Borussia Dortmund", "Bayer 04 Leverkusen", "RB Leipzig", "VfB Stuttgart", "Eintracht Frankfurt", "Borussia Mönchengladbach", "Hertha BSC", "SV Werder Bremen", "FC Augsburg", "Hannover 96", "1. FSV Mainz 05", "SC Freiburg", "VfL Wolfsburg", "Fortuna Düsseldorf", "1. FC Nürnberg"]
serieA_teams = ["Juventus", "Napoli", "Roma", "Inter", "Lazio", "Milan", "Atalanta", "Fiorentina", "Torino", "Sampdoria", "Sassuolo", "Genoa", "Chievo Verona", "Udinese", "Bologna", "Cagliari", "SPAL", "Parma", "Empoli", "Frosinone"]
ligue1_teams = ["Paris Saint-Germain", "AS Monaco", "Olympique Lyonnais", "Olympique de Marseille", "Stade Rennais FC", "FC Girondins de Bordeaux", "AS Saint-Étienne", "OGC Nice", "FC Nantes", "Montpellier HSC", "Dijon FCO", "En Avant de Guingamp", "Amiens SC", "Angers SCO", "RC Strasbourg Alsace", "Stade Malherbe Caen", "LOSC Lille", "Toulouse Football Club", "Stade de Reims", "Nîmes Olympique"]
fifa["value_eur"] = fifa["value_eur"]/1000000
fifa.rename(columns = {"value_eur": "value_eur_mil"}, inplace = True)

epl = fifa[fifa["club"].apply(lambda x: True if (x in epl_teams) else False)].groupby("club").head(15)
laliga = fifa[fifa["club"].apply(lambda x: True if (x in laliga_teams) else False)].groupby("club").head(15)
bundesliga = fifa[fifa["club"].apply(lambda x: True if (x in bundesliga_teams) else False)].groupby("club").head(15)
serieA = fifa[fifa["club"].apply(lambda x: True if (x in serieA_teams) else False)].groupby("club").head(15)
ligue1 = fifa[fifa["club"].apply(lambda x: True if (x in ligue1_teams) else False)].groupby("club").head(15)
fig = plt.figure(figsize = (20, 12))
cat = ["age", "height_cm", "weight_kg", "overall", "value_eur_mil", "wage_eur"]
title = ["Age", "Height(cm)", "Weight(kg)", "Overall Rating", "Market Value(millions of euros)", "Weekly Wage(euros)"]

d = {}
for i in range(6):
    d["ax{0}".format(i)] = fig.add_subplot(2, 3, i+1)
    d["ax{0}".format(i)] = sns.boxenplot(data = [epl[cat[i]], laliga[cat[i]], bundesliga[cat[i]], serieA[cat[i]], ligue1[cat[i]]], palette = "rainbow")
    d["ax{0}".format(i)].set_title(title[i])
    d["ax{0}".format(i)].set_xticklabels(["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"])
t15 = fifa.groupby("club").head(15)
t15a = t15.groupby("club")["overall"].agg(np.mean).sort_values(ascending = False).head(30)
fig = plt.figure(figsize = (20, 20))

ax1 = fig.add_subplot(3, 1, 1)
ax1.bar(t15a.index, t15a.values, color = "orange")
ax1 = plt.title("Top 30 strongest squads in the world, as of 2018-19 season", size = 20)
ax1 = plt.xticks(rotation = 70)
ax1 = plt.ylabel("Average Overall")
ax1 = plt.axis([None, None, 75, 90])

hpcs = fifa.groupby("club")["wage_eur"].agg(np.mean).sort_values(ascending = False).head(30)
ax2 = fig.add_subplot(3, 1, 2)
ax2.bar(hpcs.index, hpcs.values, color = "green")
ax2 = plt.title("Top 30 highest-paying clubs in the world, as of 2018-19 season", size = 20)
ax2 = plt.xticks(rotation = 70)
ax2 = plt.ylabel("Average weekly wage, in euro")

mvcs = fifa.groupby("club")["value_eur_mil"].sum().sort_values(ascending = False).head(30)
ax3 = fig.add_subplot(3, 1, 3)
ax3.bar(mvcs.index, mvcs.values, color = "purple")
ax3 = plt.title("Top 30 most valuable clubs in the world, as of 2018-19 season", size = 20)
ax3 = plt.xticks(rotation = 70)
ax3 = plt.ylabel("Total market value, in millions of euros")

plt.tight_layout()
import plotly.offline as py
%matplotlib inline

boole = fifa["nationality"].value_counts() >= 15
nations = boole[boole].keys()

def selected(x):
    if x in nations:
        return True
    else:
        return False

qnations = fifa[fifa["nationality"].apply(selected)]
top15 = fifa.groupby("nationality").head(15)
top15a = top15.groupby("nationality")["overall"].agg(np.mean).sort_values()

data = [dict(
        type = 'choropleth',
        locations = top15a.index,
        z = top15a,
        locationmode = 'country names',
        text = "Overall",
        marker = dict(
            line = dict(color = 'rgb(0,0,0)', width = 1)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'Overall')
            )
       ]

layout = dict(
    title = 'National team strengths around the world, based on FIFA 20',
    geo = dict(
        showframe = False,
        showocean = True,
        oceancolor = 'rgb(0,255,255)',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = True,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = True,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )

fig = dict(data = data, layout = layout)
py.iplot(fig, validate = False)
top10 = fifa.groupby("nationality").head(10)
top10s = top10.groupby("nationality")["value_eur_mil"].agg(sum)
top10s = top10s[top10s > 0]
countries = top10s.index

data = [dict(
        type = 'choropleth',
        locations = countries,
        z = top10s,
        locationmode = 'country names',
        text = "M €",
        marker = dict(
            line = dict(color = 'rgb(0,0,0)', width = 1)),
            colorbar = dict(autotick = True, tickprefix = '', 
            title = 'Euro(Millions)')
            )
       ]


layout = dict(
    title = 'Total market value of top 10 players for each nation',
    geo = dict(
        showframe = False,
        showocean = True,
        oceancolor = 'rgb(0,255,255)',
        projection = dict(
        type = 'orthographic',
            rotation = dict(
                    lon = 60,
                    lat = 10),
        ),
        lonaxis =  dict(
                showgrid = True,
                gridcolor = 'rgb(102, 102, 102)'
            ),
        lataxis = dict(
                showgrid = True,
                gridcolor = 'rgb(102, 102, 102)'
                )
            ),
        )

fig = dict(data = data, layout = layout)
py.iplot(fig, validate = False)