from fastai import *
from fastai.tabular import *
path = '../input/nba-201920-regular-season-stats/NBA_2019-20_regular_season_stats.csv'
df = pd.read_csv(path) #dataframe
df
dep_var = '2019-20_allstar_selections'
cat_names = ['team']
cont_names = ['age','games_played','W','L','MINS','PTS','FGM','FGA','FG_percent','3_PM','3PA','3P_percent','FTM','FTA','FT%','OREB','DREB','REB','AST','TOV','STL','BLK','PF','FP','DD2','TD3','plus-minus']
procs = [FillMissing, Categorify, Normalize] #preprocesses
test = TabularList.from_df(df.iloc[40:70].copy(), path=path, cat_names=cat_names, cont_names=cont_names)
data = (TabularList.from_df(df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
                           .split_by_idx(list(range(40,70)))
                           .label_from_df(cols=dep_var)
                           .add_test(test)
                           .databunch())
data.show_batch(rows=10)
learn = tabular_learner(data, layers=[200,100], metrics=accuracy)
learn.fit(100, 1e-2)
for i in range(100):
    row = df.iloc[i]
    if str(row[30]) is not str(learn.predict(row)[0]):
        print(row[1] + " predicted to be " + str(learn.predict(row)[0]) + " but was " + row[30])