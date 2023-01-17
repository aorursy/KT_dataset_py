!pip install numpyro
import time

import pandas as pd

import numpy as onp

import matplotlib.pyplot as plt



import jax.numpy as np

from jax import random



import numpyro

import numpyro.distributions as dist

from numpyro.examples.datasets import COVTYPE, load_dataset

from numpyro.infer import HMC, MCMC, NUTS



from jax.random import PRNGKey



numpyro.set_platform("gpu") #Use GPU for MCMC



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))
pbp = None

seasons = list(range(2009, 2020))



for season in seasons:

    path = '/kaggle/input/nflsr-pbp/regular_season/reg_pbp_{}.csv'.format(season)

    sea_pbp = pd.read_csv(path)

    

    if pbp is not None:

        pbp = pd.concat([pbp,sea_pbp],axis=0)

    else:

        pbp = sea_pbp
# some renames

pbp = pbp.rename(columns={'yardline_100':'yards_for_td','ydstogo':'distance'})



pbp['half'] = onp.where(pbp['game_seconds_remaining'] > 1800, 0, 1)
# for this analysis I'm going to omit 4th down. 4th down brings its own set of complications but could be analyzed in future work



pbp = pbp.loc[pbp.down.isin([1,2,3])]



# only care about passes and runs

pbp.groupby(['play_type','down'])['distance'].count()

# since a lot of no plays are runs with holding calls, or passes with interference, I want to keep as many as possible

# the decision to run or pass was there despite the penalty 

pbp = pbp.loc[pbp.play_type.isin(['pass','run','no_play'])]



# clean no plays

pbp.loc[(pbp.play_type=='no_play')&(pbp.desc.str.contains('sacked')), 'play_type'] = 'pass'

pbp.loc[(pbp.play_type=='no_play')&(pbp.desc.str.contains('complete')), 'play_type'] = 'pass'

pbp.loc[(pbp.play_type=='no_play')&(pbp.desc.str.contains('incomplete')), 'play_type'] = 'pass'

pbp.loc[(pbp.play_type=='no_play')&(pbp.desc.str.contains('pass')), 'play_type'] = 'pass'



pbp.loc[(pbp.play_type=='no_play')&(pbp.desc.str.contains('up the middle')), 'play_type'] = 'run'

pbp.loc[(pbp.play_type=='no_play')&(pbp.desc.str.contains('right guard')), 'play_type'] = 'run'

pbp.loc[(pbp.play_type=='no_play')&(pbp.desc.str.contains('right tackle')), 'play_type'] = 'run'

pbp.loc[(pbp.play_type=='no_play')&(pbp.desc.str.contains('right end')), 'play_type'] = 'run'

pbp.loc[(pbp.play_type=='no_play')&(pbp.desc.str.contains('left guard')), 'play_type'] = 'run'

pbp.loc[(pbp.play_type=='no_play')&(pbp.desc.str.contains('left tackle')), 'play_type'] = 'run'

pbp.loc[(pbp.play_type=='no_play')&(pbp.desc.str.contains('left end')), 'play_type'] = 'run'



# verification that remaining "no plays" are pre-snap penalties

pbp.loc[pbp.play_type=='no_play'].desc.head(50).values

pbp = pbp.loc[pbp.play_type.isin(['pass','run'])]

print("There are {} plays in this dataset".format(len(pbp)))

# add head coaches



coaches = None

for season in seasons:

    path = '/kaggle/input/nflsr-pbp/coaches/{}_HC.csv'.format(season)

    sea_coach = pd.read_csv(path)

    sea_coach['Season']=season

    if coaches is not None:

        coaches = pd.concat([coaches,sea_coach],axis=0)

    else:

        coaches = sea_coach

        

coaches.head()
# some data cleaning

# match up team ids

print(onp.sort(pbp.posteam.unique()))



# lower counts means multiple ids

# print(pbp.groupby(['posteam'])['epa'].count().reset_index().sort_values(by=['epa']))



print(onp.sort(coaches.Tm.unique()))



nflR_to_nflR = {

    'JAC':'JAX',

    'SD':'LAC',

    'STL':'LA',

}



pbp['posteam'] = pbp['posteam'].copy().replace(nflR_to_nflR)

pbp['defteam'] = pbp['defteam'].copy().replace(nflR_to_nflR)





coach_to_nflR = {

    'GNB':'GB',

    'KAN':'KC',

    'LAR':'LA',

    'NOR':'NO',

    'NWE':'NE',

    'SDG':'LAC',

    'SFO':'SF',

    'STL':'LA',

    'TAM':'TB'

}



coaches['Tm'] = coaches['Tm'].copy().replace(coach_to_nflR)



still_diff = list(set(pbp.posteam.unique()) - set(coaches.Tm.unique()))

print("There are now {} teams with different abbreviations".format(len(still_diff)))
# goal is to add head coach column to nflscrapR data



# i'll assemble a dataframe that has the following cols:

# [season     week      team       coach]     



# there is definitely a good way to do this with a dataframe pivot but i'll do the slower simple dumber way with 2 for loops

coach_df = []

i = 0

for index, row in coaches.iterrows():

    season = row['Season']

    begin = row['Begin']

    end = row['End'] + 1

    coach = row['Coach']

    team = row['Tm']

    weeks = list(range(begin, end))

    for week in weeks:

        coach_df.append([season, week, team, coach])

            

coach_df = pd.DataFrame(coach_df, columns=['season', 'week', 'posteam', 'head_coach'])



coach_df = coach_df.drop_duplicates()

coach_df.head()
# load in week data separately & merge with pbp data



wk_id = None



for season in seasons:

    path = '/kaggle/input/nflsr-pbp/games_data/regular_season/reg_games_{}.csv'.format(season)

    sea_wk = pd.read_csv(path)

    

    if wk_id is not None:

        wk_id = pd.concat([wk_id,sea_wk],axis=0)

    else:

        wk_id = sea_wk

        

wk_id = wk_id[['game_id','season','week']]

pbp = pd.merge(pbp, wk_id, how='left', on=['game_id','game_id'])



pbp[['season','week','posteam']].head()

pbp = pbp.reset_index(drop=True)

pbp = pd.merge(pbp, coach_df, how='left', left_on=['season','week','posteam'], right_on=['season','week','posteam'])

# forward fill passer, so we know QB on run plays

# not perfect but reasonable



pbp = pbp.sort_values(by=['posteam','season','week','game_seconds_remaining'], ascending=[True,True,True,False])



pbp.passer_player_name = pbp.passer_player_name.copy().fillna(method='ffill')



# change secs remaining to mins, easier for interpretation

#pbp.loc[:,'half_seconds_remaining'] = pbp['half_seconds_remaining'].copy() * 60

pbp = pbp.rename(columns={'half_seconds_remaining':'hsec'})

# target from string to bool

pbp.loc[:,'play_type'] = pbp.play_type.copy().replace({"run":"0","pass":"1"}).astype(int)

cols = ['season','week','posteam','head_coach','passer_player_name','score_differential','down','distance','half','hsec','play_type']



pbp[cols].isnull().sum(axis=0)
from sklearn.preprocessing import LabelEncoder



data = pbp[cols]



data = data.sort_values(by=['season','week'],ascending=[False,False])

data = data[:10000]



# shuffle data 

data = data.sample(frac=1)



le = LabelEncoder()

le.fit(data.posteam)



data.loc[:,'i_team'] = le.transform(data.posteam.copy())



# print("there are {} teams in the dataset".format(data.i_team.max()+1))



# data.groupby(['i_team'])['play_type'].count()
# following https://www.kaggle.com/s903124/numpyro-speed-benchmark



# numpyro.set_host_device_count(2)



d = {"passes": np.array(data.play_type.values),

     "team": np.array(data.i_team.values)}



def model(team, passes=None, link=False):

    a_bar = numpyro.sample("a_bar", dist.Normal(0, 10))

    sigma_a = numpyro.sample("sigma_a", dist.HalfCauchy(5))

#     b_bar = numpyro.sample("b_bar", dist.Normal(0, 10))

#     sigma_b = numpyro.sample("sigma_b", dist.HalfCauchy(5))



#     intercept = numpyro.sample("intercept",dist.Normal(0.3,1))#

    a = numpyro.sample("a", dist.Normal(a_bar, sigma_a), sample_shape=(len(data['i_team'].unique()),))

#     b = numpyro.sample("b", dist.Normal(b_bar, sigma_b), sample_shape=(len(d['batter_code'].unique()),))



    # non-centered paramaterization

#     a = numpyro.sample('a',  dist.TransformedDistribution(dist.Normal(0., 1.), dist.transforms.AffineTransform(a_bar, sigma_a)), sample_shape=(len(d['pitcher_code'].unique()),))

#     b = numpyro.sample('b',  dist.TransformedDistribution(dist.Normal(0., 1.), dist.transforms.AffineTransform(b_bar, sigma_b)), sample_shape=(len(d['batter_code'].unique()),))



    logit_p = a[team] + 0.39 # 0.39 is median 

    if link:

        p = expit(logit_p)

        numpyro.sample("p", dist.Delta(p), obs=p)

    numpyro.sample("pass_rate", dist.Binomial(logits=logit_p), obs=passes)



mcmc = MCMC(NUTS(model), 1000, 1000, num_chains=2)

mcmc.run(PRNGKey(0), d['team'], passes=d['passes'], extra_fields=('potential_energy','mean_accept_prob',))
vars(mcmc)

le.classes_



# there is definitely a better way to do this

i = 0

for team in le.classes_:

    print(team, "average", np.mean(np.asarray(mcmc._states['z']['a'][:,:,i])), "std", np.std(np.asarray(mcmc._states['z']['a'][:,:,i])))

    i+=1

mcmc.print_summary()
# d = {"passes": np.array(data.play_type.values),

#      "team": np.array(data.i_team.values),

#     "score_diff":np.array(data.score_differential.values)}



# def model(team, score_diff, passes=None, link=False):

#     a_bar = numpyro.sample("a_bar", dist.Normal(0, 10))

#     sigma_a = numpyro.sample("sigma_a", dist.HalfCauchy(5))

#     sd_bar = numpyro.sample("b_bar", dist.Normal(0, 10))

#     sigma_sd = numpyro.sample("sigma_b", dist.HalfCauchy(5))



# #     intercept = numpyro.sample("intercept",dist.Normal(0.3,1))#

#     a = numpyro.sample("a", dist.Normal(a_bar, sigma_a), sample_shape=(len(data['i_team'].unique()),))

#     sdiff_b = numpyro.sample("sdiff_b", dist.Normal(sd_bar, sigma_sd))



#     # non-centered paramaterization

# #     a = numpyro.sample('a',  dist.TransformedDistribution(dist.Normal(0., 1.), dist.transforms.AffineTransform(a_bar, sigma_a)), sample_shape=(len(d['pitcher_code'].unique()),))

# #     b = numpyro.sample('b',  dist.TransformedDistribution(dist.Normal(0., 1.), dist.transforms.AffineTransform(b_bar, sigma_b)), sample_shape=(len(d['batter_code'].unique()),))



#     logit_p = + score_diff * sdiff_b + a[team] + 0.39 # 0.39 is median 

#     if link:

#         p = expit(logit_p)

#         numpyro.sample("p", dist.Delta(p), obs=p)

#     numpyro.sample("pass_rate", dist.Binomial(logits=logit_p), obs=passes)



# mcmc = MCMC(NUTS(model), 1000, 1000, num_chains=2)

# mcmc.run(PRNGKey(0), d['team'], d['score_diff'], passes=d['passes'], extra_fields=('potential_energy','mean_accept_prob',))


# sum_df = []

# # there is definitely a better way to do this

# i = 0

# for team in le.classes_:

#     sum_df.append([team, np.round(np.mean(np.asarray(mcmc._states['z']['a'][:,:,i])),3), np.round(np.std(np.asarray(mcmc._states['z']['a'][:,:,i])),3)])

#     i+=1

    

# sum_df = pd.DataFrame(sum_df, columns=['team','mean_effect','uncertainty'])



# sum_df = sum_df.round(3).sort_values(by=['mean_effect'], ascending=False)



# sum_df
# data = pbp[cols]



# data = data.sort_values(by=['season','week'],ascending=[False,False])

# data = data[:25000]



# # shuffle data 

# data = data.sample(frac=1)



# le = LabelEncoder()

# le.fit(data.posteam)



# data.loc[:,'i_team'] = le.transform(data.posteam.copy())



# print("there are {} teams in the dataset".format(data.i_team.max()+1))

# season                    0

# week                      0

# posteam                   0

# head_coach                0

# passer_player_name        0

# score_differential        0

# down                      0

# distance                  0

# half                      0

# half_seconds_remaining    0

# play_type                 0



# d = {"passes": np.array(data.play_type.values),

#      "team": np.array(data.i_team.values),

#     "score_diff":np.array(data.score_differential.values),

#     "down":np.array(data.down.values),

#     "dist":np.array(data.distance.values),

#     "half":np.array(data.half.values),

#     "hsec":np.array(data.hsec.values),}



# def model(team, score_diff, down, ydstogo, half, hsec, passes=None, link=False):

#     a_bar = numpyro.sample("a_bar", dist.Normal(0, 10))

#     sigma_a = numpyro.sample("sigma_a", dist.HalfCauchy(5))

    

#     sd_bar = numpyro.sample("b_bar", dist.Normal(0, 10))

#     sigma_sd = numpyro.sample("sigma_b", dist.HalfCauchy(5))

#     down_bar = numpyro.sample("down_bar", dist.Normal(0, 10))

#     sigma_down = numpyro.sample("sigma_down", dist.HalfCauchy(5))

#     dist_bar = numpyro.sample("dist_bar", dist.Normal(0, 10))

#     sigma_dist = numpyro.sample("sigma_dist", dist.HalfCauchy(5))

#     half_bar = numpyro.sample("half_bar", dist.Normal(0, 10))

#     sigma_half = numpyro.sample("sigma_half", dist.HalfCauchy(5))

#     hsec_bar = numpyro.sample("hsec_bar", dist.Normal(0, 10))

#     sigma_hsec = numpyro.sample("sigma_hsec", dist.HalfCauchy(5))



# #     intercept = numpyro.sample("intercept",dist.Normal(0.3,1))#

#     a = numpyro.sample("a", dist.Normal(a_bar, sigma_a), sample_shape=(len(data['i_team'].unique()),))

    

#     sdiff_b = numpyro.sample("sdiff_b", dist.Normal(sd_bar, sigma_sd))

#     down_b = numpyro.sample("down_b", dist.Normal(down_bar, sigma_down))

#     dist_b = numpyro.sample("dist_b", dist.Normal(dist_bar, sigma_dist))

#     half_b = numpyro.sample("half_b", dist.Normal(half_bar, sigma_half))

#     hsec_b = numpyro.sample("hsec_b", dist.Normal(hsec_bar, sigma_hsec))

    



#     logit_p = + score_diff * sdiff_b + down * down_b + ydstogo * dist_b + half * half_b + hsec * hsec_b + a[team] + 2.73 # 2.73 is median 

#     if link:

#         p = expit(logit_p)

#         numpyro.sample("p", dist.Delta(p), obs=p)

#     numpyro.sample("pass_rate", dist.Binomial(logits=logit_p), obs=passes)



# mcmc = MCMC(NUTS(model), 1000, 1000, num_chains=2)

# mcmc.run(PRNGKey(0), d['team'], d['score_diff'], d['down'], d['dist'], d['half'], d['hsec'], passes=d['passes'], extra_fields=('potential_energy','mean_accept_prob',))



# mcmc.print_summary()


# sum_df = []

# i = 0

# for team in le.classes_:

#     sum_df.append([team, np.round(np.mean(np.asarray(mcmc._states['z']['a'][:,:,i])),3), np.round(np.std(np.asarray(mcmc._states['z']['a'][:,:,i])),3)])

#     i+=1

    

# sum_df = pd.DataFrame(sum_df, columns=['team','mean_effect','uncertainty'])



# sum_df = sum_df.round(3).sort_values(by=['mean_effect'], ascending=False)



# sum_df

data = pbp[cols]



# only Andy Reid, Kyle Shanahan

# sample = data.loc[data.head_coach.isin(['Andy Reid','Kyle Shanahan'])]

andy = data.loc[data.head_coach.isin(['Andy Reid'])]

andy = andy.sort_values(by=['season','week'],ascending=[False,False])

# only need a sample of his plays

andy = andy[:3700]

kyle = data.loc[data.head_coach.isin(['Kyle Shanahan'])]

sample = pd.concat([andy,kyle],axis=0)



# also want a league average control

control = data.loc[~data.head_coach.isin(['Andy Reid','Kyle Shanahan'])]



# while I could separate out head coach and quarterback, for small number of categories I'll just use one indicators



# A.Reid + P.Mahomes: 0

# A.Reid - P.Mahomes: 1

# KS + JG: 2

# KS - JG: 3

# League Avg Control: 4



control = control.sample(frac=1)

control = control[:int((len(sample)/2))] # reasonably balanced 



data = pd.concat([control,sample], axis=0)



del sample

del control

del andy

del kyle

import gc

gc.collect()



data['cat_effect'] = np.nan

data.loc[(data.head_coach=='Andy Reid')&(data.passer_player_name=='P.Mahomes'), 'cat_effect'] = 0

data.loc[(data.head_coach=='Andy Reid')&(data.passer_player_name!='P.Mahomes'), 'cat_effect'] = 1

data.loc[(data.head_coach=='Kyle Shanahan')&(data.passer_player_name=='J.Garoppolo'), 'cat_effect'] = 2

data.loc[(data.head_coach=='Kyle Shanahan')&(data.passer_player_name!='J.Garoppolo'), 'cat_effect'] = 3

data.loc[~data.head_coach.isin(['Andy Reid','Kyle Shanahan']), 'cat_effect'] = 4



print(data.groupby(['cat_effect'])['play_type'].count())

print(len(data))

data.head()
data = data.sample(frac=1)



d = {"passes": np.array(data.play_type.values),

     "cat": np.array(data.cat_effect.astype(int).values),

    "score_diff":np.array(data.score_differential.values),

    "down":np.array(data.down.values),

    "dist":np.array(data.distance.values),

    "half":np.array(data.half.values),

    "hsec":np.array(data.hsec.values),}



def model(team, score_diff, down, ydstogo, half, hsec, passes=None, link=False):

    # only works for cpu

    #numpyro.set_host_device_count(4)

    

    cat_bar = numpyro.sample("cat_bar", dist.Normal(0, 10))

    sigma_cat = numpyro.sample("sigma_cat", dist.HalfCauchy(5))





    intercept = numpyro.sample("intercept",dist.Normal(0,10))

    cats = numpyro.sample("cats", dist.Normal(cat_bar, sigma_cat), sample_shape=(len(data['cat_effect'].unique()),))

    

    sdiff_b = numpyro.sample("sdiff_b", dist.Normal(-0.05, 1))

    down_b = numpyro.sample("down_b", dist.Normal(0.5, 1))

    dist_b = numpyro.sample("dist_b", dist.Normal(0, 1))

    half_b = numpyro.sample("half_b", dist.Normal(0, 1))

    hsec_b = numpyro.sample("hsec_b", dist.Normal(0, 0.01))

    



    logit_p = + score_diff * sdiff_b + down * down_b + ydstogo * dist_b + half * half_b + hsec * hsec_b + cats[team]

    if link:

        p = expit(logit_p)

        numpyro.sample("p", dist.Delta(p), obs=p)

    numpyro.sample("pass_rate", dist.Binomial(logits=logit_p), obs=passes)



mcmc = MCMC(NUTS(model), 2000, 1000, num_chains=4)

mcmc.run(PRNGKey(0), d['cat'], d['score_diff'], d['down'], d['dist'], d['half'], d['hsec'], passes=d['passes'], extra_fields=('potential_energy','mean_accept_prob',))



mcmc.print_summary()
mcmc._states['z']['cats'].shape
# different score differentials

x = np.linspace(-28, 28, 100)



down = 1

ydstogo = 10

half = 1

hsec_remain = 0



down_b = np.mean(np.asarray(mcmc._states['z']['down_b']))

dist_b = np.mean(np.asarray(mcmc._states['z']['dist_b']))

half_b = np.mean(np.asarray(mcmc._states['z']['half_b']))

hsec_b = np.mean(np.asarray(mcmc._states['z']['hsec_b']))

sdiff_b = np.mean(np.asarray(mcmc._states['z']['sdiff_b']))



base = down * down_b + ydstogo * dist_b + half * half_b + hsec_remain * hsec_b + sdiff_b*x



pat = np.mean(np.asarray(mcmc._states['z']['cats'][:,:,0]))

pat_error = 0.11

high_pat = base + pat_error + pat

pat_y = base + pat

low_pat = base - pat_error + pat



npat = np.mean(np.asarray(mcmc._states['z']['cats'][:,:,1]))

npat_error = 0.11

high_npat = base + npat_error + npat

npat_y = base + npat

low_npat = base - npat_error + npat



# inverse logit

high_pat = 1 / (1 + np.exp(-(high_pat)))

pat_y = 1 / (1 + np.exp(-(pat_y)))

low_pat = 1 / (1 + np.exp(-(low_pat)))



high_npat = 1 / (1 + np.exp(-(high_npat)))

npat_y = 1 / (1 + np.exp(-(npat_y)))

low_npat = 1 / (1 + np.exp(-(low_npat)))



fig = plt.figure(figsize=(12,6))



plt.plot(x, pat_y, color='maroon', label='Andy Reid with Patrick Mahomes')

plt.fill_between(x, high_pat, low_pat,alpha=0.5, edgecolor='maroon', facecolor='red')



plt.plot(x, npat_y, color='darkgreen', label='Andy Reid before Patrick Mahomes')

plt.fill_between(x, high_npat, low_npat,alpha=0.5, edgecolor='darkgreen', facecolor='g')



la = np.mean(np.asarray(mcmc._states['z']['cats'][:,:,3]))

la_y = base + la

la_y = 1 / (1 + np.exp(-(la_y)))

plt.plot(x, la_y, color='black', label='League Average')



plt.legend()

plt.suptitle("Does Andy Reid Pass More With Mahomes?", fontsize=18)

plt.title("1st and 10, 1st Play of Second Half")

plt.xlabel("Score Differential")

plt.ylabel("Probability of Pass")



plt.savefig('./ReidWMahomes1st&10.png')



plt.show()
down = 3

ydstogo = 15

half = 2

hsec_remain = 5*60



base = down * down_b + ydstogo * dist_b + half * half_b + hsec_remain * hsec_b + sdiff_b*x



high_pat = base + pat_error + pat

pat_y = base + pat

low_pat = base - pat_error + pat



high_npat = base + npat_error + npat

npat_y = base + npat

low_npat = base - npat_error + npat



# inverse logit

high_pat = 1 / (1 + np.exp(-(high_pat)))

pat_y = 1 / (1 + np.exp(-(pat_y)))

low_pat = 1 / (1 + np.exp(-(low_pat)))



high_npat = 1 / (1 + np.exp(-(high_npat)))

npat_y = 1 / (1 + np.exp(-(npat_y)))

low_npat = 1 / (1 + np.exp(-(low_npat)))



fig = plt.figure(figsize=(12,6))



plt.plot(x, pat_y, color='maroon', label='Andy Reid with Patrick Mahomes')

plt.fill_between(x, high_pat, low_pat,alpha=0.5, edgecolor='maroon', facecolor='red')



plt.plot(x, npat_y, color='darkgreen', label='Andy Reid before Patrick Mahomes')

plt.fill_between(x, high_npat, low_npat,alpha=0.5, edgecolor='darkgreen', facecolor='g')



la = np.mean(np.asarray(mcmc._states['z']['cats'][:,:,3]))

la_y = base + la

la_y = 1 / (1 + np.exp(-(la_y)))

plt.plot(x, la_y, color='black', label='League Average')



plt.legend()

plt.suptitle("Does Andy Reid Pass More With Mahomes?", fontsize=18)

plt.title("3rd and 15, 2nd Half, 5 Min Remaining")

plt.xlabel("Score Differential")

plt.ylabel("Probability of Pass")



plt.savefig('./ReidWMahomesLate.png')



plt.show()
# different score differentials

x = np.linspace(-28, 28, 100)



down = 1

ydstogo = 10

half = 1

hsec_remain = 0



base = down * down_b + ydstogo * dist_b + half * half_b + hsec_remain * hsec_b + sdiff_b*x



jg = np.mean(np.asarray(mcmc._states['z']['cats'][:,:,2]))

jg_error = 0.12

high_jg = base + jg_error + jg

jg_y = base + jg

low_jg = base - jg_error + jg



njg = np.mean(np.asarray(mcmc._states['z']['cats'][:,:,3]))

njg_error = 0.12

high_njg = base + njg_error + njg

njg_y = base + njg

low_njg = base - njg_error + njg



# inverse logit

high_jg = 1 / (1 + np.exp(-(high_jg)))

jg_y = 1 / (1 + np.exp(-(jg_y)))

low_jg = 1 / (1 + np.exp(-(low_jg)))



high_njg = 1 / (1 + np.exp(-(high_njg)))

njg_y = 1 / (1 + np.exp(-(njg_y)))

low_njg = 1 / (1 + np.exp(-(low_njg)))



fig = plt.figure(figsize=(12,6))



plt.plot(x, jg_y, color='maroon', label='Kyle Shanahan with Jimmy Garoppalo')

plt.fill_between(x, high_jg, low_jg,alpha=0.5, edgecolor='tomato', facecolor='maroon')



plt.plot(x, njg_y, color='goldenrod', label='Kyle Shanahan without Jimmy Garoppalo')

plt.fill_between(x, high_njg, low_njg,alpha=0.5, edgecolor='brown', facecolor='goldenrod')



# league average

# la = np.mean(np.asarray(mcmc._states['z']['cats'][:,:,3]))

# la_y = base + la

# la_y = 1 / (1 + np.exp(-(la_y)))

# plt.plot(x, la_y, color='black', label='League Average')



plt.legend()

plt.suptitle("Does Kyle Shanahan Pass More With Jimmy G?", fontsize=18)

plt.title("1st and 10, 1st Play of Second Half")

plt.xlabel("Score Differential")

plt.ylabel("Probability of Pass")



plt.savefig('./KyleWJG.png')



plt.show()
sum_df = []

i = 0

for cat in ['Andy & Pat','Andy W/O Pat','Kyle & Jimmy','Kyle W/O Jimmy','League Avg']:

    sum_df.append([cat, np.mean(np.asarray(mcmc._states['z']['cats'][:,:,i])), np.std(np.asarray(mcmc._states['z']['cats'][:,:,i]))])

    i+=1

    

sum_df = pd.DataFrame(sum_df, columns=['team','mean_effect','uncertainty'])



sum_df = sum_df.round(8).sort_values(by=['mean_effect'], ascending=False)



sum_df





# data = pbp[cols]



# # only Andy Reid, Kyle Shanahan

# # sample = data.loc[data.head_coach.isin(['Andy Reid','Kyle Shanahan'])]

# andy = data.loc[data.head_coach.isin(['Andy Reid'])]

# andy = andy.sort_values(by=['season','week'],ascending=[False,False])

# # only need a sample of his plays

# andy = andy[:3700]

# kyle = data.loc[data.head_coach.isin(['Kyle Shanahan'])]

# sample = pd.concat([andy,kyle],axis=0)



# # also want a league average control

# control = data.loc[~data.head_coach.isin(['Andy Reid','Kyle Shanahan'])]



# # while I could separate out head coach and quarterback, for small number of categories I'll just use one indicators



# # A.Reid + P.Mahomes: 0

# # A.Reid - P.Mahomes: 1

# # KS + JG: 2

# # KS - JG: 3

# # League Avg Control: 4



# control = control.sample(frac=1)

# control = control[:int((len(sample)/2))] # reasonably balanced 



# data = pd.concat([control,sample], axis=0)



# del sample

# del control

# del andy

# del kyle

# import gc

# gc.collect()



# data['cat_effect'] = np.nan

# data.loc[(data.head_coach=='Andy Reid')&(data.passer_player_name=='P.Mahomes'), 'cat_effect'] = 0

# data.loc[(data.head_coach=='Andy Reid')&(data.passer_player_name!='P.Mahomes'), 'cat_effect'] = 1

# data.loc[(data.head_coach=='Kyle Shanahan')&(data.passer_player_name=='J.Garoppolo'), 'cat_effect'] = 2

# data.loc[(data.head_coach=='Kyle Shanahan')&(data.passer_player_name!='J.Garoppolo'), 'cat_effect'] = 3

# data.loc[~data.head_coach.isin(['Andy Reid','Kyle Shanahan']), 'cat_effect'] = 4



# print(data.groupby(['cat_effect'])['play_type'].count())

# print(len(data))