from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)    #THIS LINE IS MOST IMPORTANT AS THIS WILL DISPLAY PLOT ON 

#NOTEBOOK WHILE KERNEL IS RUNNING



import numpy as np 

import pandas as pd

import plotly as py

import plotly.express as px

import plotly.graph_objs as go

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

import lightgbm as lgb

from bayes_opt import BayesianOptimization

from sklearn.metrics import roc_auc_score

import shap

from pandas_profiling import ProfileReport 



import warnings  

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/league-of-legends-diamond-ranked-games-10-min/high_diamond_ranked_10min.csv')

df['blueWins'] = df['blueWins'].map({1: 'Blue Side', 0:'Red Side'})



isneg = []

for i in df.blueGoldDiff :

    if i < 0 :

        isneg.append(0)

    else :

        isneg.append(1)

df['blueGoldDiffSituation'] = isneg

isneg = []

for i in df.redGoldDiff :

    if i < 0 :

        isneg.append(0)

    else :

        isneg.append(1)

df['redGoldDiffSituation'] = isneg



isneg = []

for i in df.blueExperienceDiff :

    if i < 0 :

        isneg.append(0)

    else :

        isneg.append(1)

df['blueExperienceDiffSituation'] = isneg

isneg = []

for i in df.redExperienceDiff :

    if i < 0 :

        isneg.append(0)

    else :

        isneg.append(1)

df['redExperienceDiffSituation'] = isneg
display(df.info())

df.head()
report_data = ProfileReport(df.sample(2000))

report_data
color_discrete_map = {'Blue Side': 'rgb(122, 148, 231)', 'Red Side': 'rgb(255, 105, 97)'}



fig = px.histogram(df, x="blueWins",color = 'blueWins', color_discrete_map=color_discrete_map,

                  labels={

                     "blueWins": "Sides","count": "Wins",

                 },

                title="Total Wins per Side",

                hover_name="blueWins",       

                  )

# fig.show()

py.offline.iplot(fig)
color_discrete_map = {'Blue': 'rgb(122, 148, 231)', 'Red': 'rgb(255, 105, 97)'}



layout = go.Layout(

    yaxis=dict(

        range=[5, 45]

    ),

    xaxis=dict(

        range=[100, 200]

    )

)



tmp1 = df[['blueWardsPlaced', 'blueWardsDestroyed']].copy()

tmp1.columns = ['WardsPlaced','WardsDestroyed']

tmp1 = tmp1.astype(float)

tmp1['Side'] = 'Blue'

tmp2 = df[['redWardsPlaced', 'redWardsDestroyed']].copy()

tmp2.columns = ['WardsPlaced','WardsDestroyed']

tmp2 = tmp2.astype(float)

tmp2['Side'] = 'Red'

data = pd.concat([tmp1, tmp2], ignore_index = True).sample(2000)

data



fig1 = px.violin(data, y="WardsPlaced", color = 'Side',  box=True, points='all', color_discrete_map=color_discrete_map, title = 'Wards Placed per Side')

fig2 = px.violin(data, y="WardsPlaced", color = 'Side',  box=True, points='all', color_discrete_map=color_discrete_map, title = 'Wards Placed per Side Zoomed')

fig2.update_layout(

    yaxis=dict(

        range=[5, 45]

    )

)

fig1.show()

fig2.show()
fig1 = px.violin(data, y="WardsDestroyed", color = 'Side',  box=True, points='all', color_discrete_map=color_discrete_map, title = 'Wards Destroyed per Side')

fig2 = px.violin(data, y="WardsDestroyed", color = 'Side',  box=True, points='all', color_discrete_map=color_discrete_map, title = 'Wards Destroyed per Side Zoomed')

fig2.update_layout(

    yaxis=dict(

        range=[0, 10]

    )

)

fig1.show()

fig2.show()
color_discrete_map = {'Blue': 'rgb(122, 148, 231)', 'Red': 'rgb(255, 105, 97)'}



tmp1 = df[['blueFirstBlood']].copy()

tmp1.columns = ['FirstBloods']

tmp1 = tmp1.astype(float)

tmp1['Side'] = 'Blue'

tmp2 = df[['redFirstBlood']].copy()

tmp2.columns = ['FirstBloods']

tmp2 = tmp2.astype(float)

tmp2['Side'] = 'Red'

data = pd.concat([tmp1, tmp2])

data = data.groupby('Side').mean().reset_index()



fig = px.bar(data, x='Side', y='FirstBloods',color = 'Side', color_discrete_map=color_discrete_map, title = 'Mean First Bloods per Side')

fig.show()
col = ['blueKills','blueDeaths','blueAssists','redKills','redDeaths','redAssists']

tmp1 = df[col[0:3]].copy()

tmp1.columns = ['Kills','Death','Assistes']

tmp1['Side'] = 'Blue'

tmp2 = df[col[3:6]].copy()

tmp2.columns = ['Kills','Death','Assistes']

tmp2['Side'] = 'Red'



data = pd.concat([tmp1, tmp2], ignore_index = True)

data = data.groupby('Side').mean().reset_index()

data = pd.melt(data, id_vars=['Side'], value_vars=['Kills','Death','Assistes'])

data.columns = ['Side','KDA','Mean']



fig = px.bar(data, x="Side", y="Mean", color="Side", color_discrete_map=color_discrete_map,

             facet_col="KDA", title = 'Mean KDA per Side'

#              category_orders={"day": ["Thur", "Fri", "Sat", "Sun"],

#                               "time": ["Lunch", "Dinner"]}

            )

fig.show()
col = ['blueKills','blueDeaths','blueAssists','redKills','redDeaths','redAssists']

tmp1 = df[col[0:3]].copy()

tmp1.columns = ['Kills','Death','Assistes']

tmp1['Side'] = 'Blue'

tmp2 = df[col[3:6]].copy()

tmp2.columns = ['Kills','Death','Assistes']

tmp2['Side'] = 'Red'



data = pd.concat([tmp1, tmp2], ignore_index = True)

# data = data.groupby('Side').sum().reset_index()

data = pd.melt(data, id_vars=['Side'], value_vars=['Kills','Death','Assistes'])

data.columns = ['Side','KDA','Total']



fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=data[data['Side'] == 'Red']['KDA'], values=data[data['Side'] == 'Red']['Total'], name="Red Side"),

              1, 1)

fig.add_trace(go.Pie(labels=data[data['Side'] == 'Blue']['KDA'], values=data[data['Side'] == 'Blue']['Total'], name="Blue Side"),

              1, 2)



fig.update_traces(hole=.4, hoverinfo="label+percent+name+value")



fig.update_layout(

    title_text="KDA Proportion per Side",

    annotations=[dict(text='Red', x=0.18, y=0.5, font_size=20, showarrow=False),

                 dict(text='Blue', x=0.82, y=0.5, font_size=20, showarrow=False)])

fig.show()
fig = go.Figure()



fig.add_trace(go.Violin(x=data['KDA'][ data['Side'] == 'Blue' ],

                        y=data['Total'][ data['Side'] == 'Blue' ],

                        legendgroup='Blue', scalegroup='Blue', name='Blue',

                        line_color='rgb(122, 148, 231)')

             )

fig.add_trace(go.Violin(x=data['KDA'][ data['Side'] == 'Red' ],

                        y=data['Total'][ data['Side'] == 'Red' ],

                        legendgroup='Red', scalegroup='Red', name='Red',

                        line_color='rgb(255, 105, 97)')

             )



fig.update_traces(box_visible=True, meanline_visible=True)

fig.update_layout(violinmode='group', title = 'KDA Distribution per Side')

fig.show()
col = ['blueEliteMonsters','blueDragons','blueHeralds','redEliteMonsters','redDragons','redHeralds']

tmp1 = df[col[0:3]].copy()

tmp1.columns = ['EliteMonsters','Dragons','Heralds']

tmp1['Side'] = 'Blue'

tmp2 = df[col[3:6]].copy()

tmp2.columns = ['EliteMonsters','Dragons','Heralds']

tmp2['Side'] = 'Red'



data = pd.concat([tmp1, tmp2], ignore_index = True)

data = data.groupby('Side').mean().reset_index()

data=pd.melt(data, id_vars=['Side'], value_vars=['EliteMonsters','Dragons','Heralds'])

data.columns = ['Side','NeutralGoals','Mean']



fig = px.bar(data, x="Side", y="Mean", color="Side", color_discrete_map=color_discrete_map,

             facet_col="NeutralGoals", title = 'Mean Neutral Goals per Side'

            )

fig.show()
col = ['blueTowersDestroyed', 'redTowersDestroyed']

tmp1 = df[col[0:1]].copy()

tmp1.columns = ['TowersDestroyed']

tmp1['Side'] = 'Blue'

tmp2 = df[col[1:2]].copy()

tmp2.columns = ['TowersDestroyed']

tmp2['Side'] = 'Red'



data = pd.concat([tmp1, tmp2], ignore_index = True)

data = data.groupby('Side').mean().reset_index()

data



fig = px.bar(data, x="Side", y="TowersDestroyed", color="Side", color_discrete_map=color_discrete_map,title = 'Mean Turrets Destroyed per Side'

            )

fig.show()
col = ['blueTotalGold', 'redTotalGold']

tmp1 = df[col[0:1]].copy()

tmp1.columns = ['TotalGold']

tmp1['Side'] = 'Blue'

tmp2 = df[col[1:2]].copy()

tmp2.columns = ['TotalGold']

tmp2['Side'] = 'Red'



data = pd.concat([tmp1, tmp2], ignore_index = True)

data = data.groupby('Side').mean().reset_index()





fig = px.bar(data, x="Side", y="TotalGold", color="Side", color_discrete_map=color_discrete_map,title = 'Mean Gold per Side'

            )

fig.show()



data = pd.concat([tmp1, tmp2], ignore_index = True)

data



fig1 = px.violin(data, y="TotalGold", color = 'Side',  box=True, points='all', color_discrete_map=color_discrete_map, title = 'Total Gold per Side Distribution')

fig1.show()
col = ['blueAvgLevel', 'redAvgLevel']

tmp1 = df[col[0:1]].copy()

tmp1.columns = ['AvgLevel']

tmp1['Side'] = 'Blue'

tmp2 = df[col[1:2]].copy()

tmp2.columns = ['AvgLevel']

tmp2['Side'] = 'Red'



data = pd.concat([tmp1, tmp2], ignore_index = True)

data = data.groupby('Side').mean().reset_index()





fig = px.bar(data, x="Side", y="AvgLevel", color="Side", color_discrete_map=color_discrete_map,title = 'Mean Level per Side'

            )

fig.show()





data = pd.concat([tmp1, tmp2], ignore_index = True).sample(2000)

data



fig1 = px.violin(data, y="AvgLevel", color = 'Side',  box=True, points='all', color_discrete_map=color_discrete_map, title = 'Level per Side Distribution')

fig1.show()
col = ['blueTotalExperience', 'redTotalExperience']

tmp1 = df[col[0:1]].copy()

tmp1.columns = ['TotalExperience']

tmp1['Side'] = 'Blue'

tmp2 = df[col[1:2]].copy()

tmp2.columns = ['TotalExperience']

tmp2['Side'] = 'Red'



data = pd.concat([tmp1, tmp2], ignore_index = True)

data = data.groupby('Side').mean().reset_index()





fig = px.bar(data, x="Side", y="TotalExperience", color="Side", color_discrete_map=color_discrete_map,title = 'Mean Experience per Side'

            )

fig.show()





data = pd.concat([tmp1, tmp2], ignore_index = True).sample(2000)

data



fig1 = px.violin(data, y="TotalExperience", color = 'Side',  box=True, points='all', color_discrete_map=color_discrete_map, title = 'Total Experience per Side Distribution')

fig1.show()
# blueTotalMinionsKilled



col = ['blueTotalMinionsKilled', 'redTotalMinionsKilled']

tmp1 = df[col[0:1]].copy()

tmp1.columns = ['TotalMinionsKilled']

tmp1['Side'] = 'Blue'

tmp2 = df[col[1:2]].copy()

tmp2.columns = ['TotalMinionsKilled']

tmp2['Side'] = 'Red'



data = pd.concat([tmp1, tmp2], ignore_index = True)

data = data.groupby('Side').mean().reset_index()





fig = px.bar(data, x="Side", y="TotalMinionsKilled", color="Side", color_discrete_map=color_discrete_map,title = 'Mean Minions Killede per Side'

            )

fig.show()





data = pd.concat([tmp1, tmp2], ignore_index = True).sample(2000)

data



fig1 = px.violin(data, y="TotalMinionsKilled", color = 'Side',  box=True, points='all', color_discrete_map=color_discrete_map, title = 'Total Minions Killed per Side Distribution')

fig1.show()
# blueTotalJungleMinionsKilled



col = ['blueTotalJungleMinionsKilled', 'redTotalJungleMinionsKilled']

tmp1 = df[col[0:1]].copy()

tmp1.columns = ['TotalJungleMinionsKilled']

tmp1['Side'] = 'Blue'

tmp2 = df[col[1:2]].copy()

tmp2.columns = ['TotalJungleMinionsKilled']

tmp2['Side'] = 'Red'



data = pd.concat([tmp1, tmp2], ignore_index = True)

data = data.groupby('Side').mean().reset_index()





fig = px.bar(data, x="Side", y="TotalJungleMinionsKilled", color="Side", color_discrete_map=color_discrete_map,title = 'Mean Jungle Minions Killed per Side'

            )

fig.show()





data = pd.concat([tmp1, tmp2], ignore_index = True).sample(2000)

data



fig1 = px.violin(data, y="TotalJungleMinionsKilled", color = 'Side',  box=True, points='all', color_discrete_map=color_discrete_map, title = 'Total Jungle Minions Killed per Side Distribution')

fig1.show()
# blueGoldDiff



col = ['blueGoldDiff', 'redGoldDiff']

tmp1 = df[col[0:1]].copy()

tmp1.columns = ['GoldDiff']

tmp1['Side'] = 'Blue'

tmp2 = df[col[1:2]].copy()

tmp2.columns = ['GoldDiff']

tmp2['Side'] = 'Red'



data = pd.concat([tmp1, tmp2], ignore_index = True)

data = data.groupby('Side').mean().reset_index()



fig = px.bar(data, x="Side", y="GoldDiff", color="Side", color_discrete_map=color_discrete_map, title = 'Mean Gold Diff per Side'

            )

fig.show()





data = pd.concat([tmp1, tmp2], ignore_index = True).sample(2000)

data



fig1 = px.violin(data, y="GoldDiff", color = 'Side',  box=True, points='all', color_discrete_map=color_discrete_map, title = 'Gold Diff per Side Distribution')

fig1.show()
col = ['blueGoldDiffSituation', 'redGoldDiffSituation']

tmp1 = df[col[0:1]].copy()

tmp1.columns = ['GoldDiffSituation']

tmp1['Side'] = 'Blue'

tmp2 = df[col[1:2]].copy()

tmp2.columns = ['GoldDiffSituation']

tmp2['Side'] = 'Red'



data = pd.concat([tmp1, tmp2], ignore_index = True)

data = data.groupby('Side').mean().reset_index()



fig = px.bar(data, x="Side", y="GoldDiffSituation", color="Side", color_discrete_map=color_discrete_map, title = 'Gold Difference Situation per Side'

            )

fig.show()

# blueExperienceDiff



col = ['blueExperienceDiff', 'redExperienceDiff']

tmp1 = df[col[0:1]].copy()

tmp1.columns = ['ExperienceDiff']

tmp1['Side'] = 'Blue'

tmp2 = df[col[1:2]].copy()

tmp2.columns = ['ExperienceDiff']

tmp2['Side'] = 'Red'



data = pd.concat([tmp1, tmp2], ignore_index = True)

data = data.groupby('Side').mean().reset_index()



fig = px.bar(data, x="Side", y="ExperienceDiff", color="Side", color_discrete_map=color_discrete_map, title = 'Mean Experience Difference per Side'

            )

fig.show()





data = pd.concat([tmp1, tmp2], ignore_index = True).sample(2000)

data



fig1 = px.violin(data, y="ExperienceDiff", color = 'Side',  box=True, points='all', color_discrete_map=color_discrete_map, title = 'Experience Difference per Side Distribution')

fig1.show()
col = ['blueExperienceDiffSituation', 'redExperienceDiffSituation']

tmp1 = df[col[0:1]].copy()

tmp1.columns = ['ExperienceDiffSituation']

tmp1['Side'] = 'Blue'

tmp2 = df[col[1:2]].copy()

tmp2.columns = ['ExperienceDiffSituation']

tmp2['Side'] = 'Red'



data = pd.concat([tmp1, tmp2], ignore_index = True)

data = data.groupby('Side').mean().reset_index()



fig = px.bar(data, x="Side", y="ExperienceDiffSituation", color="Side", color_discrete_map=color_discrete_map, title = 'Mean Experience Difference Situation per Side'

            )

fig.show()

# blueCSPerMin	



col = ['blueCSPerMin', 'redCSPerMin']

tmp1 = df[col[0:1]].copy()

tmp1.columns = ['CSPerMin']

tmp1['Side'] = 'Blue'

tmp2 = df[col[1:2]].copy()

tmp2.columns = ['CSPerMin']

tmp2['Side'] = 'Red'



data = pd.concat([tmp1, tmp2], ignore_index = True)

data = data.groupby('Side').mean().reset_index()



fig = px.bar(data, x="Side", y="CSPerMin", color="Side", color_discrete_map=color_discrete_map, title = 'Mean CS Per Min per Side'

            )

fig.show()





data = pd.concat([tmp1, tmp2], ignore_index = True).sample(2000)

data



fig1 = px.violin(data, y="CSPerMin", color = 'Side',  box=True, points='all', color_discrete_map=color_discrete_map, title = 'CS Per Min per Side Distribution')

fig1.show()
# blueGoldPerMin



col = ['blueGoldPerMin', 'redGoldPerMin']

tmp1 = df[col[0:1]].copy()

tmp1.columns = ['GoldPerMin']

tmp1['Side'] = 'Blue'

tmp2 = df[col[1:2]].copy()

tmp2.columns = ['GoldPerMin']

tmp2['Side'] = 'Red'



data = pd.concat([tmp1, tmp2], ignore_index = True)

data = data.groupby('Side').mean().reset_index()



fig = px.bar(data, x="Side", y="GoldPerMin", color="Side", color_discrete_map=color_discrete_map, title = 'Mean Gold Per Min per Side'

            )

fig.show()





data = pd.concat([tmp1, tmp2], ignore_index = True).sample(2000)

data



fig1 = px.violin(data, y="GoldPerMin", color = 'Side',  box=True, points='all', color_discrete_map=color_discrete_map, title = 'Gold Per Min per Side Distribution')

fig1.show()
corr = df[[col for col in df.columns if 'blue' in col and col != 'blueWins']].corr()

f,ax = plt.subplots(figsize=(20, 20))

p = sns.heatmap(corr,

                cmap='coolwarm',

                annot=True,

                fmt=".1f",

                annot_kws={'size':10},

                cbar=False,

                ax=ax)

p.set_title('Blue Side Features Correlation')
corr = df[[col for col in df.columns if 'red' in col and col != 'blueWins']].corr()

f,ax = plt.subplots(figsize=(20, 20))

p = sns.heatmap(corr,

                cmap='coolwarm',

                annot=True,

                fmt=".1f",

                annot_kws={'size':10},

                cbar=False,

                ax=ax)
df['blueWins'] = df['blueWins'].map({'Blue Side': 1, 'Red Side': 0})
blue_win = df[[col for col in df.columns if col != 'blueWins']].corrwith(df['blueWins']).to_frame().sort_values(by = 0, ascending = False)

blue_win = pd.concat([blue_win.head(5), blue_win.tail(5)])

blue_win.columns = ['Blue Win Correlation']

blue_win



red_win = df[[col for col in df.columns if col != 'blueWins']].corrwith(df['blueWins'].map({0:1, 1:0})).to_frame().sort_values(by = 0, ascending = False)

red_win = pd.concat([red_win.head(5), red_win.tail(5)])

red_win.columns = ['Red Win Correlation']

red_win



fig = plt.figure(figsize=(25,10))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)



plt.figure(figsize=(6,6))



sns.heatmap(blue_win,

            vmin=-1,

            cmap='coolwarm',

            annot=True,

           ax = ax1);



sns.heatmap(red_win,

            vmin=-1,

            cmap='coolwarm',

            annot=True,

           ax = ax2);
X = df.drop(['blueWins', 'gameId'], axis=1)

y = df.blueWins
# X = MinMaxScaler().fit_transform(df.drop(['blueWins', 'gameId'], axis=1))



def bayes_parameter_opt_lgb(X = X, y = y, init_round=15, opt_round=25, n_folds=5, random_seed=6, n_estimators=10000, learning_rate=0.05, output_process=False):

    # prepare data

    train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)

    # parameters

    def lgb_eval(num_leaves, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):

        params = {'application':'binary','num_iterations': n_estimators, 'learning_rate':learning_rate, 'early_stopping_round':100, 'metric':'auc'}

        params["num_leaves"] = int(round(num_leaves))

        params['feature_fraction'] = max(min(feature_fraction, 1), 0)

        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)

        params['max_depth'] = int(round(max_depth))

        params['lambda_l1'] = max(lambda_l1, 0)

        params['lambda_l2'] = max(lambda_l2, 0)

        params['min_split_gain'] = min_split_gain

        params['min_child_weight'] = min_child_weight

        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])

        return max(cv_result['auc-mean'])

    # range 

    lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (24, 45),

                                            'feature_fraction': (0.1, 0.9),

                                            'bagging_fraction': (0.8, 1),

                                            'max_depth': (5, 8.99),

                                            'lambda_l1': (0, 5),

                                            'lambda_l2': (0, 3),

                                            'min_split_gain': (0.001, 0.1),

                                            'min_child_weight': (5, 50)}, random_state=0)

    # optimize

    lgbBO.maximize(init_points=init_round, n_iter=opt_round)

    

    # output optimization process

    if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")

    

    # return best parameters

    return lgbBO



opt_params = bayes_parameter_opt_lgb(X, y, init_round=5, opt_round=10, n_folds=3, random_seed=6, n_estimators=100, learning_rate=0.05)
params = opt_params.max['params']

params["num_leaves"] = int(round(params['num_leaves']))

params['feature_fraction'] = max(min(params['feature_fraction'], 1), 0)

params['bagging_fraction'] = max(min(params['bagging_fraction'], 1), 0)

params['max_depth'] = int(round(params['max_depth']))

params['lambda_l1'] = max(params['lambda_l1'], 0)

params['lambda_l2'] = max(params['lambda_l2'], 0)

params['min_split_gain'] = max(params['min_split_gain'], 0)

params['min_child_weight'] = max(params['min_child_weight'], 0)

params['num_iterations'] = int(round(10000))

params['learning_rate'] = max(0.05, 0)

params['application'] = 'binary'

params['metric'] = 'auc'

params['objective'] = 'binary'
train_data = lgb.Dataset(data=X, label=y, free_raw_data=True, feature_name = [col for col in df.columns if col!= 'gameId' and col!= 'blueWins'])

model = lgb.train(params, train_data)
ax = lgb.plot_importance(model, max_num_features=10, figsize = (15,15))

plt.show()
pred = np.round(model.predict(X)).astype(int)

data_test = TSNE(n_components=3).fit_transform(X)
predicted = pd.DataFrame(data_test)

predicted['output'] = pred

predicted['output'] = predicted['output'].astype(object)

predicted['output'] = predicted['output'].map({0: "Red Wins",1: "Blue Wins"})

predicted['flag'] = 'predicted'

predicted.columns = ['dim_1','dim_2','dim_3','output', 'flag']



real = pd.DataFrame(data_test)

real['output'] = y.values

real['output'] = real['output'].astype(object)

real['output'] = real['output'].map({0: "Red Wins",1: "Blue Wins"})

real['flag'] = 'real'

real.columns = ['dim_1','dim_2','dim_3','output', 'flag']



data_3d = pd.concat([predicted,real], ignore_index = True).sample(4000)
color_discrete_map = {"Blue Wins": 'rgb(122, 148, 231)', "Red Wins": 'rgb(255, 105, 97)'}



fig1 = px.scatter_3d(predicted, x='dim_1', y='dim_2', z='dim_3', color_discrete_map=color_discrete_map,color='output', title = 'How Model Distinct Wins and Loses')

fig2 = px.scatter_3d(real, x='dim_1', y='dim_2', z='dim_3',color_discrete_map=color_discrete_map,color='output', title = 'Real Distinction Between Wins and Loses')



fig1.show()

fig2.show()