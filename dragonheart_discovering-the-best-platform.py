import pandas as pd

a=[1,2,3]

b=a.copy()

b.append(4)

b

1-1e-10


g_per_platform = []

for platform,data in ign_df.groupby('platform'):

    if data.shape[0] >= 70: #filter platforms with little games

        g_per_platform.append((platform,data.score.mean(),data.shape[0]))

    

g_per_platform = pd.DataFrame.from_records(g_per_platform,

                                    columns=('name','avg_score','n_games'))

g_per_platform = g_per_platform.sort('avg_score')



g_per_platform
# remove platforms with a small number of entries

df= ign_df[ign_df['platform'].isin(g_per_platform.name)]



hist = df['score'].hist(alpha=0.5,bins=40)

hist.set_xlabel('Scores')

hist.set_ylabel('Number of games')
awards = []

award_rates = []

for plat,data in df.groupby('platform'):

    n_awards = data[data['editors_choice'] == 'Y'].shape[0]

    

    awards.append((plat,n_awards,data.shape[0],n_awards/data.shape[0]))

awards = pd.DataFrame.from_records(awards,

                                    columns=('platform','n_awards','n_games','award_rate'))







awards
bar = awards[['award_rate']].plot.bar(x=awards['platform'],

                                legend=False)

bar.set_ylabel('award_rate')
month_list = ['January','February','March','April','May','June',

                 'July','August','September','October',

                     'November','December']



by_month = []

for month,data in ign_df.groupby('release_month'):

    by_month.append((month_list[month-1],data.score.mean(),data.shape[0]))

                    

by_month = pd.DataFrame.from_records(by_month,

                                    columns=('month','avg_score','n_games'))



by_month
by_month = by_month[['month','n_games']]

pie = by_month.plot.pie('n_games',labels=month_list,

                       legend=False)

pie.set_ylabel('Releases per month')
by_year = []

for year,data in ign_df.groupby('release_year'):

    by_year.append((year,data.score.mean(),data.shape[0]))

                    

by_year = pd.DataFrame.from_records(by_year,

                                    columns=('year','avg_score','n_games'))



bar = by_year[['n_games']].plot.bar(x=by_year['year'],

                                legend=False)



by_year
by_year = []

no_duplicates = ign_df[['title','release_year']].drop_duplicates()

for year,data in no_duplicates.groupby('release_year'):

    by_year.append((year,data.shape[0]))

                    

by_year = pd.DataFrame.from_records(by_year,

                                    columns=('year','n_games'))



bar = by_year[['n_games']].plot.bar(x=by_year['year'],

                                legend=False)



by_year