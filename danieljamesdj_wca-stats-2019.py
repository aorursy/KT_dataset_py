# import relevant packages

import numpy as np

import pandas as pd

pd.set_option('display.max_columns', None)

import itertools



# set up path of shared files and helper funtion for loading certain tables as pandas DataFrame

db_path = '../input/wca-db-export/'



def load_table(table_name):

    """ helper function to obtain a pandas DataFrame for table <table_name> """

    return pd.read_csv(db_path + 'WCA_export_{}.tsv'.format(table_name), delimiter='\t')



person_info = load_table('Persons')

person_info = person_info[person_info.subid == 1][['id', 'name', 'countryId']]

person_info = person_info.rename(columns={'id': 'personId', 'name': 'personName', 

                                          'countryId': 'personCountryId'})



def add_person_info(df):

    return person_info.merge(df, on='personId')
results = load_table('Results')

results['year'] = results.competitionId.str[-4:].astype('int')

results_2019 = results[results.year == 2019]

attempts_2019 = results_2019[['personId', 'personName', 'personCountryId'] + ['value' + str(i+1) for i in range(5)]].melt(

    id_vars=['personId', 'personName', 'personCountryId'], value_vars=['value' + str(i+1) for i in range(5)])

attempts_2019 = attempts_2019[attempts_2019.value != 0]

attempts_2019['solved'] = (attempts_2019.value > 0).astype('int')

total_solves = attempts_2019.groupby(['personId', 'personName', 'personCountryId']).solved.sum().reset_index()

total_solves.sort_values('solved', ascending=False).head(10).reset_index(drop=True)
comps = load_table('Competitions')

person_comps_2019 = results_2019[['personId', 'personName', 'personCountryId', 'competitionId']].merge(

    comps[comps.countryId.str[0] != 'X'][['id', 'countryId']], how='inner', left_on='competitionId', right_on='id')

most_countries = person_comps_2019.groupby(['personId', 'personName', 'personCountryId']).countryId.nunique().reset_index()

most_countries.rename(columns={'countryId': 'Number of countries'}).sort_values('Number of countries', ascending=False).head(10).reset_index(drop=True)
golds = results_2019[((results_2019.pos == 1) & (results_2019.roundTypeId.isin(['c', 'f'])) & (results_2019.best > 0))]

most_golds = golds.groupby(['personId', 'personName', 'personCountryId']).pos.count().reset_index()

most_golds.rename(columns={'pos': 'Number of golds'}).sort_values('Number of golds', ascending=False).head(10).reset_index(drop=True)
silvers = results_2019[((results_2019.pos == 2) & (results_2019.roundTypeId.isin(['c', 'f'])) & (results_2019.best > 0))]

most_silvers = silvers.groupby(['personId', 'personName', 'personCountryId']).pos.count().reset_index()

most_silvers.rename(columns={'pos': 'Number of silvers'}).sort_values('Number of silvers', ascending=False).head(10).reset_index(drop=True)
bronzes = results_2019[((results_2019.pos == 3) & (results_2019.roundTypeId.isin(['c', 'f'])) & (results_2019.best > 0))]

most_bronzes = bronzes.groupby(['personId', 'personName', 'personCountryId']).pos.count().reset_index()

most_bronzes.rename(columns={'pos': 'Number of bronzes'}).sort_values('Number of bronzes', ascending=False).head(10).reset_index(drop=True)
podiums = results_2019[((results_2019.pos.isin([1, 2, 3])) & (results_2019.roundTypeId.isin(['c', 'f'])) & (results_2019.best > 0))]

most_podiums = podiums.groupby(['personId', 'personName', 'personCountryId']).pos.count().reset_index()

most_podiums.rename(columns={'pos': 'Number of podiums'}).sort_values('Number of podiums', ascending=False).head(10).reset_index(drop=True)
# SQL used: SELECT co.competition_id, u.name FROM competition_organizers co INNER JOIN users u ON co.organizer_id=u.id

comp_organizers = pd.read_csv('../input/wca-org-del-export/competition_organizers.csv')

organizers_2019 = comp_organizers[comp_organizers.competition_id.str[-4:] == '2019']

most_organized_2019 = organizers_2019.groupby('name').competition_id.count().reset_index()

most_organized_2019.rename(columns={'competition_id': 'Competitions organized'}).sort_values('Competitions organized', ascending=False).head(10).reset_index(drop=True)
comps_2019 = comps[comps.year == 2019]

comps_before_2019 = comps[comps.year < 2019]

countries_before_2019 = comps_before_2019[comps_before_2019.countryId.str[0] != 'X'].countryId.unique()

countries_2019 = comps_2019[comps_2019.countryId.str[0] != 'X'].countryId.unique()

new_countries_2019 = np.setdiff1d(countries_2019, countries_before_2019)

print('Number of new countries 2019:', len(new_countries_2019))

print('Countries:', new_countries_2019)
comps_2019.groupby('cityName').id.count().sort_values(ascending=False).reset_index().rename(columns={'id': 'Number of competitions'}).head(10)
comps_2019.groupby('countryId').id.count().sort_values(ascending=False).reset_index().rename(columns={'id': 'Number of competitions'}).head(10)
results_2019.groupby('personCountryId').personId.nunique().sort_values(ascending=False).reset_index().rename(

    columns={'personId': 'Number of distinct competitors'}).head(10)
results_333bf_2019 = results_2019[results_2019.eventId == '333bf'][['personId', 'personName', 'personCountryId', 'competitionId', 'roundTypeId'] 

                                                                   + ['value' + str(i+1) for i in range(5)]]

attempts_333bf_2019 = results_333bf_2019.melt(id_vars=['personId', 'personName', 'personCountryId', 'competitionId', 'roundTypeId'], 

                                              value_vars=['value' + str(i+1) for i in range(5)]).rename(columns={'variable': 'attempt'})

attempts_333bf_2019 = attempts_333bf_2019[attempts_333bf_2019.value != 0]

attempts_333bf_2019['solved'] = (attempts_333bf_2019.value > 0).astype('int')

total_successes = attempts_333bf_2019.groupby(['personId', 'personName', 'personCountryId']).solved.sum().reset_index()

total_successes.sort_values('solved', ascending=False).head(10).reset_index(drop=True)
comps['start_date'] = pd.to_datetime(comps[['year', 'month', 'day']], format='%Y%m%d')

df_333bf = attempts_333bf_2019.merge(comps[['id', 'start_date']], how='inner', left_on='competitionId', right_on='id')

df_333bf = df_333bf[['personId', 'personName', 'personCountryId', 'start_date', 'roundTypeId', 'attempt', 'solved']]



df_333bf = df_333bf.sort_values(['personId', 'start_date', 'roundTypeId', 'attempt']).reset_index()

df_333bf['grouper'] = ((df_333bf.personId != df_333bf.personId.shift(1)) | (df_333bf.solved != df_333bf.solved.shift(1))).cumsum()

df_333bf['streak'] = df_333bf.groupby('grouper').solved.cumsum().astype('int')



df_333bf[['personId', 'personName', 'personCountryId', 'streak']].groupby(

    ['personId', 'personName', 'personCountryId']).streak.max().sort_values(ascending=False).head(10).reset_index()
num_comps_2019 = results_2019.groupby(['personId', 'personName', 'personCountryId']).competitionId.nunique()

num_comps_2019.sort_values(ascending=False).reset_index().rename(columns={'competitionId': 'Competitions'}).head(10)
df_wrs = results_2019.copy()

df_wrs['single_wr'] = (df_wrs.regionalSingleRecord == 'WR')

df_wrs['average_wr'] = (df_wrs.regionalAverageRecord == 'WR')

df_wrs['wrs'] = df_wrs['single_wr'].astype('int') + df_wrs['average_wr'].astype('int')

df_wrs['comp_wrs'] = df_wrs.groupby('competitionId').wrs.transform('sum')

wrs_seen_person_comps = df_wrs[['personId', 'personName', 'personCountryId', 'competitionId', 'comp_wrs']].drop_duplicates()

wrs_seen = wrs_seen_person_comps.groupby(['personId', 'personName', 'personCountryId']).comp_wrs.sum()

wrs_seen.sort_values(ascending=False).reset_index().rename(columns={'comp_wrs': 'WRs potentially seen'}).head(10)
pid_ba_dates = results[['personId', 'competitionId', 'eventId', 'roundTypeId', 'best', 'average']].merge(

    comps[['id', 'start_date']], how='inner', left_on='competitionId', right_on='id')

pid_ba_nan = pid_ba_dates.replace([-2, -1, 0], np.nan)

pid_ba_nan = pid_ba_nan.sort_values(['personId', 'eventId', 'start_date', 'roundTypeId'])

pid_ba_nan['min_best'] = pid_ba_nan.groupby(['personId', 'eventId']).best.transform('cummin')

pid_ba_nan['min_avg'] = pid_ba_nan.groupby(['personId', 'eventId']).average.transform('cummin')

pid_ba_nan['sgl_pb'] = (pid_ba_nan.best == pid_ba_nan.min_best)

pid_ba_nan['avg_pb'] = (pid_ba_nan.average == pid_ba_nan.min_avg)

pid_ba_nan['pbs'] = pid_ba_nan['sgl_pb'].astype('int') + pid_ba_nan['avg_pb'].astype('int')

pid_ba_pbs_2019 = pid_ba_nan[pid_ba_nan.competitionId.str[-4:].astype('int') == 2019]

pbs_comps_2019 = pid_ba_pbs_2019[['personId', 'competitionId', 'pbs']].groupby(['personId', 'competitionId']).pbs.sum()

add_person_info(pbs_comps_2019).sort_values('pbs', ascending=False).head(10).reset_index()
pbs_comps_2019 = pbs_comps_2019.reset_index()

pbs_comps_2019['pb_flag'] = (pbs_comps_2019.pbs > 0)

pbs_comps_2019['grouper'] = ((pbs_comps_2019.personId != pbs_comps_2019.personId.shift(1)) | 

                             (pbs_comps_2019.pb_flag != pbs_comps_2019.pb_flag.shift(1))).cumsum()

pbs_comps_2019['streak'] = pbs_comps_2019.groupby('grouper').pb_flag.cumsum().astype('int')

add_person_info(pbs_comps_2019[['personId', 'streak']].groupby('personId').streak.max().sort_values(

    ascending=False).head(10).reset_index())
# SQL used: SELECT cd.competition_id, u.name FROM competition_delegates cd INNER JOIN users u ON cd.delegate_id=u.id

comp_delegates = pd.read_csv('../input/wca-org-del-export/competition_delegates.csv')

delegates_2019 = comp_delegates[comp_delegates.competition_id.str[-4:] == '2019']

most_delegated_2019 = delegates_2019.groupby('name').competition_id.count().reset_index()

most_delegated_2019.rename(columns={'competition_id': 'Competitions delegated'}).sort_values('Competitions delegated', ascending=False).head(10).reset_index(drop=True)
averages_333_before_2019 = results[((results.year < 2019) & (results.eventId == '333') & (results.average > 0))][['personId', 'average']]

best_averages_333_before_2019 = averages_333_before_2019.groupby('personId').average.min().reset_index()

averages_333_2019 = results[((results.year == 2019) & (results.eventId == '333') & (results.average > 0))][['personId', 'average']]

best_averages_333_2019 = averages_333_2019.groupby('personId').average.min().reset_index()

averages_333_comparison = best_averages_333_before_2019.merge(best_averages_333_2019, on='personId')

averages_333_comparison['improvement_pc'] = 100 * (averages_333_comparison.average_x - averages_333_comparison.average_y) / averages_333_comparison.average_x

averages_333_comparison = averages_333_comparison.rename(columns={'average_x': 'best_333_average_before_2019','average_y': 'best_333_average_2019'})

add_person_info(averages_333_comparison).nlargest(10, 'improvement_pc').reset_index(drop=True)
pid_best_2019 = pid_ba_pbs_2019[['personId', 'eventId', 'best']].groupby(

    ['personId', 'eventId']).min().dropna()

full_index = list(itertools.product(list(pid_best_2019.reset_index().personId.unique()), 

                      list(pid_best_2019.reset_index().eventId.unique())))

pid_best_2019 = pid_best_2019.reindex(full_index).fillna(9999999999)

pid_best_2019 = pid_best_2019.sort_values(['eventId', 'best']).reset_index()

pid_best_2019['rank'] = pid_best_2019.groupby('eventId').best.rank(ascending=True, method='min').astype('int')

single_ranks_2019 = pd.pivot_table(pid_best_2019[['personId', 'eventId', 'rank']], index='personId', 

                                   columns='eventId', values='rank', aggfunc='first')

single_ranks_2019['SOR'] = pid_best_2019[['personId', 'rank']].groupby('personId').sum()

single_ranks_2019 = add_person_info(single_ranks_2019)

single_ranks_2019.sort_values('SOR').reset_index(drop=True).head(10)
pid_average_2019 = pid_ba_pbs_2019[['personId', 'eventId', 'average']].groupby(

    ['personId', 'eventId']).min().dropna()

full_index = list(itertools.product(list(pid_average_2019.reset_index().personId.unique()), 

                      list(pid_average_2019.reset_index().eventId.unique())))

pid_average_2019 = pid_average_2019.reindex(full_index).fillna(9999999999)

pid_average_2019 = pid_average_2019.sort_values(['eventId', 'average']).reset_index()

pid_average_2019['rank'] = pid_average_2019.groupby('eventId').average.rank(ascending=True, method='min').astype('int')

average_ranks_2019 = pd.pivot_table(pid_average_2019[['personId', 'eventId', 'rank']], index='personId', 

                                   columns='eventId', values='rank', aggfunc='first')

average_ranks_2019['SOR'] = pid_average_2019[['personId', 'rank']].groupby('personId').sum()

average_ranks_2019 = add_person_info(average_ranks_2019)

average_ranks_2019.sort_values('SOR').reset_index().head(10)
single_sor_top100_2019 = single_ranks_2019.sort_values('SOR').reset_index().head(100)

average_sor_top100_2019 = average_ranks_2019.sort_values('SOR').reset_index().head(100)

sor_top100_2019_people = single_sor_top100_2019[['personId', 'personName', 'personCountryId']].append(

    average_sor_top100_2019[['personId', 'personName', 'personCountryId']]).drop_duplicates()

sor_top100_2019_people.groupby('personCountryId').count().personId.sort_values(

    ascending=False).reset_index().head(10).rename(columns={'personId': 'Number of people'})
mostSingleRecords = results_2019[results_2019.regionalSingleRecord == 'WR'].groupby(['personId', 'personName']).regionalSingleRecord.agg(['count'])

mostAverageRecords = results_2019[results_2019.regionalAverageRecord == 'WR'].groupby(['personId', 'personName']).regionalAverageRecord.agg(['count'])

mostRecords = pd.concat([mostSingleRecords, mostAverageRecords]).groupby(['personId', 'personName'])['count'].sum()

mostRecords.sort_values(ascending=False).reset_index().rename(columns={'count': 'WR count'}).head(10)
nonWRsingleRecords = results_2019[results_2019.regionalSingleRecord != 'WR']

nonWRaverageRecords = results_2019[results_2019.regionalAverageRecord != 'WR']

crSingleRecords = nonWRsingleRecords[nonWRsingleRecords.regionalSingleRecord != 'NR']

crAverageRecords = nonWRaverageRecords[nonWRaverageRecords.regionalAverageRecord != 'NR']

mostSingleRecords = crSingleRecords.groupby(['personId', 'personName']).regionalSingleRecord.agg(['count'])

mostAverageRecords = crAverageRecords.groupby(['personId', 'personName']).regionalAverageRecord.agg(['count'])

mostRecords = pd.concat([mostSingleRecords, mostAverageRecords]).groupby(['personId', 'personName'])['count'].sum()

mostRecords.sort_values(ascending=False).reset_index().rename(columns={'count': 'CR count'}).head(10)
mostSingleRecords = results_2019[results_2019.regionalSingleRecord == 'NR'].groupby(['personId', 'personName']).regionalSingleRecord.agg(['count'])

mostAverageRecords = results_2019[results_2019.regionalAverageRecord == 'NR'].groupby(['personId', 'personName']).regionalAverageRecord.agg(['count'])

mostRecords = pd.concat([mostSingleRecords, mostAverageRecords]).groupby(['personId', 'personName'])['count'].sum()

mostRecords.sort_values(ascending=False).reset_index().rename(columns={'count': 'NR count'}).head(10)
attempts_2019 = results_2019[['personId', 'personName', 'competitionId', 'personCountryId'] + ['value' + str(i+1) for i in range(5)]].melt(

    id_vars=['personId', 'personName', 'competitionId', 'personCountryId'], value_vars=['value' + str(i+1) for i in range(5)])

attempts_2019 = attempts_2019[attempts_2019.value > 0].drop(columns='variable')

solvesPerCompetition = attempts_2019.groupby(['personId', 'personName', 'personCountryId', 'competitionId']).count().reset_index()

solvesPerCompetition.sort_values('value', ascending=False).reset_index(drop=True).head(10).rename(columns={'value': 'Number of solves'})
# Going with the mean for simplicity...

attempts_2019 = results_2019[['personId', 'personName', 'eventId', 'personCountryId'] + ['value' + str(i+1) for i in range(5)]].melt(

    id_vars=['personId', 'personName', 'eventId', 'personCountryId'], value_vars=['value' + str(i+1) for i in range(5)])

solves_333_2019 = attempts_2019[(attempts_2019.eventId == '333') & (attempts_2019.value > 0)].drop(columns='variable')

average_333_2019 = solves_333_2019.groupby(['personId', 'personName', 'personCountryId']).value.mean().reset_index()

average_333_2019['value'] = np.round(average_333_2019['value'] / 100, 2)

average_333_2019.sort_values('value').head(10).reset_index(drop=True).rename(columns={'value': 'Overall 3x3x3 mean in 2019'})
results_2019.groupby(['competitionId']).personId.nunique().sort_values(ascending=False).reset_index().head(10).rename(columns={'personId': 'Number of competitors'})
comps_pid_dates = pid_ba_dates[['competitionId', 'personId', 'start_date']].drop_duplicates()

comps_pid_dates['newcomer_date'] = comps_pid_dates[['personId', 'start_date']].groupby('personId').start_date.transform('min')

comps_pid_dates['is_newcomer'] = (comps_pid_dates['start_date'] == comps_pid_dates['newcomer_date'])

newcomers_2019 = comps_pid_dates[comps_pid_dates.competitionId.str[-4:] == '2019'][['competitionId', 'is_newcomer']].groupby('competitionId').is_newcomer.sum()

newcomers_2019.astype('int').sort_values(ascending=False).head(10).reset_index().rename(columns={'is_newcomer': 'Number of newcomers'})
comps_pid_2019 = results_2019[['competitionId', 'personId']].drop_duplicates()

comp_nums = comps_pid_2019.groupby('personId').competitionId.transform('count')

comps_pid_2019_reduced = comps_pid_2019[comp_nums > 10]

comp_pairs_2019 = comps_pid_2019_reduced.merge(comps_pid_2019_reduced, on='competitionId')

comp_pairs_2019 = comp_pairs_2019[comp_pairs_2019.personId_x < comp_pairs_2019.personId_y]

comp_pairs_2019.groupby(['personId_x', 'personId_y']).count().sort_values(

    'competitionId', ascending=False).rename(columns={'competitionId': 'Shared competitions'}).reset_index().head(10)
comp_trips_2019 = comp_pairs_2019.merge(comps_pid_2019_reduced.rename(columns={'personId': 'personId_z'}), on='competitionId')

comp_trips_2019 = comp_trips_2019[((comp_trips_2019.personId_x < comp_trips_2019.personId_z) & (comp_trips_2019.personId_y < comp_trips_2019.personId_z))]

comp_trips_2019.groupby(['personId_x', 'personId_y', 'personId_z']).count().sort_values(

    'competitionId', ascending=False).rename(columns={'competitionId': 'Shared competitions'}).reset_index().head(10)
# Note: podiums from above just includes 2019 podiums

podiums_reduced = podiums[['personId', 'competitionId', 'eventId']]

podium_pairs = podiums_reduced.merge(podiums_reduced, on=['competitionId', 'eventId'])

podium_pairs = podium_pairs[podium_pairs.personId_x < podium_pairs.personId_y]

podium_pairs.groupby(['personId_x', 'personId_y']).eventId.count().sort_values(

    ascending=False).reset_index().rename(columns={'eventId': 'Shared podiums'}).head(10)
podium_trips_2019 = podium_pairs.merge(podiums_reduced.rename(columns={'personId': 'personId_z'}), on=['competitionId', 'eventId'])

podium_trips_2019 = podium_trips_2019[((podium_trips_2019.personId_x < podium_trips_2019.personId_z) 

                                       & (podium_trips_2019.personId_y < podium_trips_2019.personId_z))]

podium_trips_2019.groupby(['personId_x', 'personId_y', 'personId_z']).eventId.count().sort_values(

    ascending=False).reset_index().rename(columns={'eventId': 'Shared podiums'}).head(10)
num_comps_2019 = results_2019[results_2019.personId.str[:4] == '2019'].groupby(['personId', 'personName', 'personCountryId']).competitionId.nunique()

num_comps_2019.sort_values(ascending=False).reset_index().rename(columns={'competitionId': 'Competitions'}).head(10)
newcomer_comps_2019 = comps_pid_dates[(comps_pid_dates.personId.str[:4] == '2019') & (comps_pid_dates.is_newcomer)]

newcomer_results_2019 = results_2019.merge(newcomer_comps_2019, on=['competitionId', 'personId'])

best_newcomer_singles_2019 = newcomer_results_2019.loc[newcomer_results_2019[newcomer_results_2019.best > 0].groupby('eventId').best.idxmin()]

best_newcomer_averages_2019 = newcomer_results_2019.loc[newcomer_results_2019[newcomer_results_2019.average > 0].groupby('eventId').average.idxmin()]
best_newcomer_singles_2019[['eventId', 'personId', 'personName', 'personCountryId', 'competitionId', 'best']].reset_index(drop=True)
best_newcomer_averages_2019[['eventId', 'personId', 'personName', 'personCountryId', 'competitionId', 'average']].reset_index(drop=True)