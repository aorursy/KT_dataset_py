# We'll import the files we need from Kaggle and two that have been uploaded: massey_ord_2020 and at_large_2020. These two files are necessary for our 2020 predictions. 
import numpy as np
import pandas as pd 
import lightgbm as lgb
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 
warnings.simplefilter(action='ignore')
np.random.seed(924)

# Importing the files needed to perform the necessary analytics
reg_season = pd.read_csv('/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/MRegularSeasonCompactResults.csv')
ncaa_tourn = pd.read_csv('/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/MNCAATourneyCompactResults.csv')
massey_ord = pd.read_csv('/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/MMasseyOrdinals.csv')
massey_ord_2020 = pd.read_csv('/kaggle/input/ncaa-supplementals/MMasseyOrdinals_2020_only.csv')
ncaa_seeds = pd.read_csv('/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/MNCAATourneySeeds.csv')
conference = pd.read_csv('/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/MTeamConferences.csv')
conference.rename(columns={'TeamID':'Team'}, inplace = True)
teams = pd.read_csv('/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/MTeams.csv')
teams.rename(columns={'TeamID':'Team'}, inplace = True)
at_large = pd.read_csv('/kaggle/input/ncaa-supplementals/NCAA_auto_bids.csv')
at_large_2020 = at_large[(at_large.Season ==2020)]


# Because we'll need this function later and it doesn't fit well anywhere else, we'll create it now. 
# It makes the seeding file usable by removing the location designation 
# from the seed while converting it from a str to an int. 
seeds_df = ncaa_seeds.loc[:, ['TeamID', 'Season', 'Seed']]

def clean_seed(seed):
    s_int = int(seed[1:3])
    return s_int

seeds_df['seed_int'] = seeds_df['Seed'].apply(lambda x: clean_seed(x))
seeds_df.drop(labels=['Seed'], inplace=True, axis=1) # This is the string label
seeds_df.rename(columns={'TeamID':'Team'}, inplace = True)
def duplicate_games(reg_season):    
    reg_season_w = reg_season.rename(columns = {'WTeamID': 'Team',
                                             'WScore': 'Team_Score',                                                             
                                             'LTeamID': 'Opponent',
                                             'LScore': 'Opponent_Score'})

    reg_season_l = reg_season.rename(columns = {'LTeamID': 'Team',
                                            'LScore': 'Team_Score',                                                             
                                            'WTeamID': 'Opponent',
                                            'WScore': 'Opponent_Score'})

    regseason = (reg_season_w, reg_season_l)
    regseason = pd.concat(regseason, ignore_index = True, sort = False)
    regseason = regseason[['Season','DayNum', 'Team', 'Team_Score', 'Opponent', 'Opponent_Score']]
    return regseason

reg_season = duplicate_games(reg_season)
# These inputs are self-explanatory. We create variables for a team’s margin of victory and a win/loss binary. 
# Additionally, we get the averages for points scored and allowed along with a team’s winning percentage.
# One input variable that was surprisingly helpful was a team’s minimal margin of victory for a season. 

reg_season['mov'] = reg_season['Team_Score'] - reg_season['Opponent_Score']
reg_season['win'] = np.where(reg_season.mov > 0,1,0)
reg_season['avg_off'] = reg_season.groupby(['Season','Team'])['Team_Score'].transform('mean')
reg_season['avg_def'] = reg_season.groupby(['Season','Team'])['Opponent_Score'].transform('mean')
reg_season['wp'] = reg_season.groupby(['Season','Team'])['win'].transform('mean')
reg_season['mov_min'] = reg_season.groupby(['Season','Team'])['mov'].transform('min')
# We'll calculate a team's average margin of victory for each season. 
reg_season['mov_diff_avg'] = reg_season.groupby(['Season','Team'])['mov'].transform('mean')

# This step creates the opponent's average margin of victory. 
reg_season['mov_opp_avg'] = reg_season.groupby(['Season','Opponent'])['mov'].transform('mean')

# Finally, we take the average of a team's opponent's average margin of victory. 
# When a team's opponents have a strong average of an average margin of victory, it means a team has played a difficult schedule.
reg_season['schd_strngth'] = reg_season.groupby(['Season','Team'])['mov_opp_avg'].transform('mean')
# We calculate a team's opponent's average score allowed. 
reg_season['avg_def_opp'] = reg_season.groupby(['Season','Opponent'])['Team_Score'].transform('mean') 

# We take the average of that number. 
reg_season['avg_off_adj'] = reg_season.groupby(['Season','Team'])['avg_def_opp'].transform('mean') 

# Finally we take the difference of our average offensive performance versus our opponents average defensive performance. 
reg_season['off_adj'] = reg_season['avg_off'] - reg_season['avg_off_adj'] 
# These two lines help us to determine if a team won its last game of the season. It's a proxy for winning a conference championship.   
reg_season['max_gameday'] = reg_season.groupby(['Season','Team'])['DayNum'].transform('max')
reg_season['final_game'] = np.where(reg_season['max_gameday'] == reg_season['DayNum'],1,0)

# We join in the conference dataset. The conferences and conference groupings are assigned a binary variable. The Pac-10 and Pac-12 are treated as the same conference. 
reg_season = pd.merge(reg_season, conference, left_on = ['Season','Team'],  right_on=['Season','Team'])
reg_season['SEC'] = np.where(reg_season['ConfAbbrev'].isin(['sec']),1,0)
reg_season['P12'] = np.where(reg_season['ConfAbbrev'].isin(['pac_ten','pac_twelve']),1,0)
reg_season['Minors'] = np.where(reg_season['ConfAbbrev'].isin(['wac','ivy','mac','sun_belt','big_west','ovc','southern','a_sun','aec','maac'
                                                               ,'patriot','big_sky','southland','big_south','meac','nec','summit','swac','mid_cont']),1,0)
reg_season['Mid_majors'] = np.where(reg_season['ConfAbbrev'].isin(['a_ten','cusa','mwc','mvc','wcc','horizon','caa','aac']),1,0) 
reg_season = reg_season[(reg_season.Season>=2003)] 

# Group the dataset by team and season. This gives us one record per team per year which is the needed structure for the final model.    
season_grp = reg_season.groupby(['Season','Team'])['mov_diff_avg','avg_off','avg_def','wp','off_adj','mov_min','schd_strngth','SEC','P12','Minors','Mid_majors'].mean().reset_index(drop=False)

# We calculate a modified version of Pythagorean Expectation. These results will overestimate a team's quality of play because of the differences in exponents but that's fine. 
season_grp['pythag'] = (season_grp['avg_off']**9.5)/((season_grp['avg_off']**9.5)+(season_grp['avg_def']**9.2))

# We'll take the difference between a team's modified Pythagorean Expectation and actual win percentage. 
season_grp['pythag_overage'] = season_grp['pythag'] - season_grp['wp']
season_grp = season_grp[(season_grp.Season>=2003)] 

# Referencing back to the reg_season data frame. This code creates a data frame that tells us if a team won its final game.    
won_final = reg_season[(reg_season.final_game ==1)]
won_final['won_final_game'] = np.where(won_final['mov'] > 0,1,0)
won_final = won_final.iloc[:,[0,2,25]]
# Due to some team/mapping issues, there were some teams duplicated and other missing for 2020. An update was made available so we'll filter out 2020 in the 
# original ordinals dataset and then concatenate the updated file as a fix.
massey_ord = massey_ord[(massey_ord.Season <= 2019 )] 
massey_ord_2020  
frames = [massey_ord,massey_ord_2020]
massey_ord = pd.concat(frames)

# We'll create two datasets to create the ratio. Any system could have been used but the WLK system had no data integrity issues for the timeframe around days 99/100. 
# Other systems could be worth looking into.  
  
massey_99 = massey_ord[((massey_ord.RankingDayNum == 99) | (massey_ord.RankingDayNum == 100)) & (massey_ord.SystemName == 'WLK' )]
massey_133 = massey_ord[(massey_ord.RankingDayNum == 133) & (massey_ord.SystemName == 'WLK' ) 
                        |((massey_ord.RankingDayNum == 128) & (massey_ord.SystemName == 'WLK' ) & (massey_ord.Season == 2020)) ]
massey_ratio = pd.merge(massey_99,massey_133,how='inner', on =['Season','TeamID'])
massey_ratio = massey_ratio.iloc[:,[0,3,4,7]]
massey_ratio['trend_ratio'] = massey_ratio['OrdinalRank_x'] / massey_ratio['OrdinalRank_y']
massey_ratio = massey_ratio.iloc[:,[0,1,4]]
# This is a test to only include systems where all seasons exist and existed in 2020.
massey_ord = massey_ord[(massey_ord.Season>=2003)] 
m_test = massey_ord.groupby(['Season', 'SystemName'])['RankingDayNum'].max().reset_index(drop=False)
m_test['seasons_count'] = m_test.groupby(['SystemName'])['RankingDayNum'].transform('count')

# Once we have the systems, we store them in a list and use that list to filter our dataset down to only those results. 
# We remove the voting systems (USA Today and AP polls) because they only have ratings for a small set of teams. 
# In the future, I will look for a way to use them because they 
# are aligned with the perception of a team's quality more so than its actual strength.

systems_list = m_test[(m_test.seasons_count==18) & (m_test.Season==2020)]
ordinal_name = list(systems_list['SystemName'])
ordinal_name.remove('USA') 
ordinal_name.remove('AP') 
massey_ord = massey_ord[massey_ord.SystemName.isin(ordinal_name)]

# We then find the last ranking day of the season and filter to that day. The data is pivoted to make it more usable for analysis. 
massey_ord['max_day_num'] = massey_ord.groupby(['Season', 'SystemName', 'TeamID'])['RankingDayNum'].transform('max')
massey_ord = massey_ord[(massey_ord.RankingDayNum == massey_ord.max_day_num )]
massey_ord = massey_ord.iloc[:,[0,3,4,2]]
massey_ord = pd.pivot_table(massey_ord, index = ['Season','TeamID'], columns = 'SystemName', values = 'OrdinalRank')
# The code below finds the min, max and standard deviation across the row for each team/season for all systems. 
massey_ord['min_ord'] = massey_ord.min(axis=1)
massey_ord['max_ord'] = massey_ord.max(axis=1)
massey_ord['std_dev'] = np.std(massey_ord.iloc[ :,0:9],axis=1)
massey_ord = massey_ord.reset_index(drop=False)

# We join these results with the ratio results from above. Finally, renaming the column makes joining cleaner later. 
massey_ord = pd.merge(massey_ord,massey_ratio,how='inner',on = ['Season','TeamID'])
massey_ord.rename(columns={'TeamID':'Team'}, inplace = True)
# The four datasets are joined. We drop two columns that were needed in early calculations but won't be used for either problem. 
train = pd.merge(conference, season_grp, how='left', left_on = ['Season','Team'], right_on = ['Season','Team'])
train = pd.merge(train, won_final, how='left', left_on = ['Season','Team'], right_on = ['Season','Team'])
train = pd.merge(train,     massey_ord, how='left', left_on = ['Season','Team'], right_on = ['Season','Team'])
train = train[(train.Season >= 2003)]
train.drop(labels=['avg_off','avg_def'], inplace=True, axis=1)

# After extensive testing, these columns returned the best results. 
# We only need seasons 2003-19. It worked out that losing some games due to the cancellation of some conferences tournament games should
# have minimal impact on our model. None of the input variables were directly dependent on conference tournament results. This is an avenue I didn't look into but I suspect 
# knowing a team’s results in its conference tournament would have been predictive.
at_large_train = train[['Season','Team','ConfAbbrev','wp','off_adj','pythag','mov_diff_avg','COL','MOR','RTH','SAG','WLK',
                        'max_ord','schd_strngth','mov_min','SEC','P12','Mid_majors','Minors']]

# This dataset contains our target variable. It also allows us to filter out the teams already guaranteed a place in the tournament due to winning their conference tournament. 
at_large = at_large.iloc[:,[0,1,2,3]]
at_large['Team'] = at_large['Team'].astype(int) 

# We use an outer join. Any team that didn't get an invitation to the NCAA tournament will get an nan for the target column 'at_large'. We'll replace the nan's with zeros. 
at_large_train = pd.merge(at_large, at_large_train, how='outer', left_on = ['Season','Team'], right_on = ['Season','Team'])
at_large_train['at_large'] = np.nan_to_num(at_large_train.at_large)

# We filter out teams receiving an automatic bid and then drop that column. 
at_large_train = at_large_train[(at_large_train.auto_bid != 1)]
at_large_train.drop(labels=['auto_bid'], inplace=True, axis=1)
# The second training dataset to predict the seeding in the NCAA tournament is created similarly to the previous approach. Not all input variables are the same but since we created everything
# needed earlier, after we join all of the files, we'll select the predictive columns by name

seeding_train = pd.merge(seeds_df, season_grp, how='left', left_on = ['Season','Team'], right_on = ['Season','Team'])
seeding_train = pd.merge(seeding_train,     won_final, how='left', left_on = ['Season','Team'], right_on = ['Season','Team'])
seeding_train = pd.merge(seeding_train,     massey_ord, how='left',   left_on = ['Season','Team'], right_on = ['Season','Team'])
seeding_train = seeding_train[(seeding_train.Season >= 2003)]
seeding_train.drop(labels=['avg_off','avg_def'], inplace=True, axis=1)

seeding_train = seeding_train[['Season','Team','seed_int','wp','pythag_overage','mov_diff_avg','COL','DOL','MOR','POM','RTH','SAG',
                        'WLK','WOL','min_ord','max_ord','std_dev','trend_ratio','schd_strngth','Minors','won_final_game']]
cv_df = at_large_train[(at_large_train['Season'] < 2017)]
target = cv_df[(cv_df['Season'] < 2017)]['at_large']
cv_df = cv_df.iloc[:,4:]
lgb_df = lgb.Dataset(cv_df, label=target)

# The results for the cross fold validation have an error rate of 1.87%. Speaking in general terms, given that there are ~ 350 college basketball teams 
# and 32 conference champions receiving automatic bids, we are selecting 36 teams out of 320. If the model doesn't select any teams, 
# it will have an error rate of ~ 11% as it will be right in 284 out of 320 instances. 

param = {'num_leaves': [2], 
         'objective': 'binary',
          'metric':['binary_error'],
          'learning_rate': [85/1000], 
          'min_data': [10],              
          'max_depth' : [-1],
          'min_hessian':7/10,    
          'colsample_bytree':37/100,
          'colsample_bynode':92/100,   
          'lambda_l2': (1/20),    
          'lambda_l1':(1/100), 
          'max_bin':84,                 
          'bagging_fraction':(6/100),       
          'bagging_freq':6,   
          }

clf_mod = lgb.cv(params = param, 
          nfold = 7, 
          train_set  = lgb_df, 
          num_boost_round = 1000,      
          verbose_eval = 10,         
          early_stopping_rounds = 10)

# The best iteration is num_boost_round which we'll use in during the model creation. That number is 41.

# Basic data preparation to create the model 
at_large_df = at_large_train[(at_large_train['Season'] < 2017)]
at_large_target = at_large_df[(at_large_df['Season'] < 2017)]['at_large']
at_large_df = at_large_df.iloc[:,4:]
at_large_pred = at_large_train[(at_large_train['Season'] >= 2017) & (at_large_train['Season'] <= 2020)]   
at_large_pred_join = at_large_pred.iloc[:,[0,1,2,3]].reset_index(drop=True)  
at_large_pred = at_large_pred.iloc[:,4:]
lgb_df = lgb.Dataset(at_large_df, label=at_large_target)  

# We create the model, predict on seasons 2017-19 and join the results back to a data frame with team and season information.  
clf_mod = lgb.train(param, lgb_df, num_boost_round = 41)
model_preds = pd.DataFrame(clf_mod.predict(at_large_pred)) #lgbm
pred_df = pd.merge(at_large_pred_join,model_preds,left_index=True, right_index=True)              
pred_df.rename(columns={0:'pred'}, inplace= True)

# To see the results per year, we need to rank the probabilities by year. 
# By creating a 'pred_rank' column looking at only the top 36 selection and a 'correct' column looking at all selections,
# it makes it simpler to return results. Consider that while this is a classification problem returning probabilities, 
# the cutoff is dynamic for every season. 
pred_df['pred_rank'] = pred_df.groupby(['Season'])['pred'].rank(ascending=False)
pred_df['rank_pred'] = np.where(((pred_df['pred_rank'] <= 36) & (pred_df['at_large']==1)) ,1,0)  
pred_df['correct'] = np.where(((pred_df['pred_rank'] > 36 ) & (pred_df['at_large']== 0))
                              |((pred_df['pred_rank'] <= 36) & (pred_df['at_large']== 1)) ,1,0)   
avg_accuracy = pred_df[(pred_df.Season<2020)]['correct'].mean()
correct_count = pred_df['rank_pred'].sum()                
by_season_splits = pred_df.groupby(['Season'])['rank_pred'].sum().reset_index(drop=False) 

print('Accuracy between 2017-2019: ' + str(round(avg_accuracy,3)))
print('Total Correct Selections: ' + str(round(correct_count,4)) + ' out of 108')
print(by_season_splits[(by_season_splits.Season <=2019)])


feature_imp = pd.DataFrame(sorted(zip(clf_mod.feature_importance(),at_large_df.columns)), columns=['Value','Feature'])
feature_imp = feature_imp[(feature_imp.Value>0)].sort_values('Value', ascending = False)

plt.figure(figsize=(10, 5))
#f, ax = plt.subplots(figsize=(6, 15))
sns.set(style='darkgrid')
ax = sns.barplot(x='Feature', y='Value', data=feature_imp, palette = 'summer') #     color='teal')
plt.title('LightGBM Variable Importance',fontsize=18)
plt.ylabel('LGB Value')
plt.xlabel('Input Variables',fontsize=14) 
plt.show()

# We'll use our prediction data frame and filter it on only 2020. We'll then create two lists, one for all conferences and one for conferences with tournament champions. 
pred_df_2020 = pred_df[(pred_df.Season == 2020)]
conf_list = pred_df_2020['ConfAbbrev'].tolist()
# This step eliminates duplicates
conf_list = list(dict.fromkeys(conf_list))
auto_bid_list = at_large_2020['conf'].tolist()
# We remove everything from list one that is in list two and return a list we can filter on below. 
conf_list = [x for x in conf_list if x not in auto_bid_list]

# We loop through our conferences and find the team with the highest probability to create a data frame. There can be duplicated records for some conference due to ties in the probabilities. 
conf_auto_bid = pd.DataFrame()
for i in conf_list:
    conf_df = pred_df_2020[(pred_df_2020.ConfAbbrev == i)]
    conf_df = conf_df.iloc[:,:6].reset_index(drop=True)
    conf_df['max_conf'] = conf_df.groupby('ConfAbbrev')['pred'].transform(max)
    conf_df = conf_df[(conf_df.pred == conf_df.max_conf)]
    conf_auto_bid = conf_auto_bid.append(conf_df)    

# We need to split our data frame into two. Any conference without duplication will be given the automatic bid. 
conf_auto_bid['conf_count'] = conf_auto_bid.groupby('ConfAbbrev')['pred'].transform('count')
conf_auto_bid_a = conf_auto_bid[(conf_auto_bid.conf_count == 1)]

# We filter on conferences with multiple top teams. I've previously verified the top seeds to the teams in the top_seed_filter list.
# The list will then be used to select the top conference tournament seeded teams. 
conf_auto_bid_b = conf_auto_bid[(conf_auto_bid.conf_count > 1)]
top_seed_filter = [1341,1300,1199,1166,1242,1246]
conf_auto_bid_b = conf_auto_bid_b[conf_auto_bid_b['Team'].isin(top_seed_filter)]

# We'll combine our results into our auto-bid data frame and update the AEC conference results as mentioned above as team 1467 
# had the highest probability but lost. We then assign all teams in this data frame a zero as an 'at-large' selection and a one for an 'auto_bid'
# Lastly, we need to change column locations to match the structure of the at_large_2020 below.
conf_auto_bid_f = conf_auto_bid_a.append(conf_auto_bid_b)
conf_auto_bid_f['Team'] = conf_auto_bid_f['Team'].replace(1467,1436)
conf_auto_bid_f['at_large'] = 0
conf_auto_bid_f = conf_auto_bid_f.iloc[:,:4]
conf_auto_bid_f['auto_bid'] = 1
conf_auto_bid_f =  conf_auto_bid_f.iloc[:,[0,1,2,4,3]]

# This code appends the 13 conference tournament winners to our 19 modeled 'conference champions'. 
at_large_2020.rename(columns={'conf':'ConfAbbrev'}, inplace = True)
at_large_2020 = at_large_2020.append(conf_auto_bid_f)


# Now that we know who our 32 at-large teams are, we can use our at-large model predictions to select the 38 at-large teams for 2020!
# We create a list of the automatic bids and filter the teams in it. 
auto_bid_list = at_large_2020['Team'].tolist()
at_large_2020_pred = pred_df_2020[~pred_df_2020['Team'].isin(auto_bid_list)]

# Now that we have only teams available for an at-large selection, we rank them by the model prediction and take the top 36 teams. 
# We'll print rankings 36-37 to verify there is no tie at the 36th position. There isn't so no tie breaks are needed and we can take the top 36 teams.
at_large_2020_pred['pred_rank'] = at_large_2020_pred.groupby(['Season'])['pred'].rank(ascending=False)
print('The 36th and 37th teams have different probabilites for an at-large selection')
print(at_large_2020_pred[(at_large_2020_pred.pred_rank <= 37) & (at_large_2020_pred.pred_rank >= 36)].sort_values('pred',ascending = False).iloc[:,[0,1,3,4,5]])
print()
# We'll print teams 33-36 to see who were the last four in as us basketball geeks love that kind of thing.
# In order, they are: Texas Tech, Providence, Cinci, Miss St. 
print('Last four in')
print(at_large_2020_pred[(at_large_2020_pred.pred_rank <= 36) & (at_large_2020_pred.pred_rank >= 33)].sort_values('pred',ascending = False).iloc[:,[0,1,3,4,5]])
print()
# We'll also print teams 37-40 to see who was on the 'bubble' and barely missed a selection. 
# They are: Arizona St, NC State, Arkansas, Memphis - all from different conferences. 
# We'll see two teams with probabilities  > 50% left out. This is a top heavy year. 
print('First four out')
print(at_large_2020_pred[(at_large_2020_pred.pred_rank <= 40) & (at_large_2020_pred.pred_rank >= 37)].sort_values('pred',ascending = False).iloc[:,[0,1,3,4,5]])
print()

at_large_2020_pred = at_large_2020_pred[(at_large_2020_pred.pred_rank <= 36)]
at_large_2020_pred['at_large'] = 1
at_large_2020_pred['auto_bid'] = 0
at_large_2020_pred.drop(labels = ['pred','pred_rank','rank_pred','correct'], inplace=True, axis=1)
at_large_2020_pred = at_large_2020_pred.iloc[:,[3,0,1,2,4]]
NCAA_2020_tourn_teams = at_large_2020_pred.append(at_large_2020)

#For some peace of mind, we'll print the number of unique conferences and total teams. The outputs are exactly as we'd hoped.  
print(NCAA_2020_tourn_teams.groupby('Season')['ConfAbbrev'].nunique())
print()
print(NCAA_2020_tourn_teams.groupby('Season')['Team'].count())

train_split = seeding_train[(seeding_train['Season'] < 2017)]
target = train_split[(train_split['Season'] < 2017)]['seed_int']
train_split = train_split.iloc[:,3:]                       
lgb_df = lgb.Dataset(train_split, label=target)

param = {'num_leaves': [2], 
         'objective': 'mae',
         'learning_rate': [43/1000], 
         'min_data': [17],                
          'max_depth' : [-1],
          'min_hessian':0/10,    
          'colsample_bytree':53/100,
          'colsample_bynode':25/100,     
          'lambda_l2': (192/20),    
          'lambda_l1':(0/100), 
          'max_bin':163,                      
          'bagging_fraction':(33/100),        
          'bagging_freq':36,       
          'verbose':1}

clf_mod_2 = lgb.cv(params = param, 
          nfold = 7, 
          train_set  = lgb_df, 
          num_boost_round = 3000,
          verbose_eval = 100,
          early_stopping_rounds = 100)

# The best iteration is 514 which is what we'll use to train the model.   
# The cross validation error is < 1. We'll see how this translates in the validation set. 
# We performed cross validation above and now we'll compare those results with the unseen data from years 2017-2019. If our results don't 
# resemble the results above, we know we've overfit. It's a little abstract to know if we actually have overfit due to the complexity of 
# the problem, so we'll rely on the knowledge that modeling as a multiclass problem cross validated in the mid 40%. 
# We split the data out by the 2017 season 
train_split = seeding_train[(seeding_train['Season'] < 2017)]
target = train_split[(train_split['Season'] < 2017)]['seed_int']
train_split = train_split.iloc[:,3:]
pred_split = seeding_train[(seeding_train['Season'] >= 2017)]
pred_split_join = pred_split.iloc[:,[0,1,2]].reset_index(drop=True)
pred_split = pred_split.iloc[:,3:]            

# We create the lightgbm dataset, model and predict the results.   
lgb_df = lgb.Dataset(train_split, label=target)
clf_mod_2 = lgb.train(param, lgb_df, num_boost_round = 514)
pred_df = pd.DataFrame(clf_mod_2.predict(pred_split))  

# We build a usable data frame 
pred_df = pd.merge(pred_split_join,pred_df,left_index=True, right_index=True) 
pred_df['pred'] = pred_df.idxmax(axis=1)
pred_df.rename(columns={0:'pred_regression'}, inplace= True)

# We rank our predictions by season. This gives us rankings between 1-68. Simply dividing by 4 doesn't address the issue of needing six 11 seeds and six 16 seeds.
# It would instead return four 17 seeds. If the rank is >= 45, we subtract 2 from the rank and do it again if it's over 65. By doing this, when we divide the rank by 4, 
# we'll get six 11 and 16 seeds matching what has historically happened over the last three seasons. 
pred_df['pred_rank'] = pred_df.groupby('Season')['pred_regression'].rank()
pred_df['pred_rank'] = np.where (pred_df['pred_rank'] >=  45, pred_df['pred_rank'] -2, pred_df['pred_rank'])
pred_df['pred_rank'] = np.where (pred_df['pred_rank'] >= 65, pred_df['pred_rank'] -2, pred_df['pred_rank'])        
pred_df['pred_rank'] = np.ceil((pred_df['pred_rank']/4))                   
pred_df = pred_df.iloc[:,[0,1,2,5]]

# We'll create variables to measure performance for the validation set and break it out by year. 
# The stated goal is to predict the correct seed. Closer is better especially if this approach is viewed as a guide for human decision support. 
# We'll create a metric and print it showing predictions within one seed.

pred_df['correct_seeds'] = np.where(pred_df.seed_int == pred_df.pred_rank,1,0) 
pred_df['seed_diff'] = abs(pred_df['pred_rank']-pred_df['seed_int'])
pred_df['within_one'] = np.where(pred_df.seed_diff <= 1,1,0) 
accuracy = pred_df['correct_seeds'].mean()
within_one = pred_df['within_one'].mean()
by_season_splits = pred_df.groupby(['Season'])['correct_seeds'].mean().reset_index(drop=False) 

print('Total average accuracy between 2017-2019: ' + str(round(accuracy,3)))
print('Prediction Percent within one seed: ' + str(round(within_one,4)))
print(by_season_splits)
# We'll print the feature importance for the model which uses nearly all inputs. Massey Ordinals represent six of the eight most important variables with min_ord being a derivative. 
# For both models, RTH is the most important feature. Win percentage is the only non-ordinal in both models - winning matters. 

feature_imp = pd.DataFrame(sorted(zip(clf_mod_2.feature_importance(),train_split.columns)), columns=['Value','Feature'])
feature_imp = feature_imp[(feature_imp.Value>0)].sort_values('Value', ascending = False)

plt.figure(figsize=(25, 5))
sns.set(style='darkgrid')
ax = sns.barplot(x='Feature', y='Value', data=feature_imp, palette = 'summer') #     color='teal')
plt.title('LightGBM Variable Importance',fontsize=18)
plt.ylabel('LGB Value')
plt.xlabel('Input Variables',fontsize=14) 
plt.show()

# This code creates a correlation plot between all variables. 
corr_2017 = seeding_train[(seeding_train['Season'] < 2017)].iloc[:,2:].corr()
names = ['Seed','WP%','PY_Overage','MOV_avg','COL','DOL','MOR','POM','RTH','SAG','WLK','WOL','Min_Ord','Max_Ord','STD_DEV','Ord_Trnd','SOS','Minor','Won_Last']
fig = plt.figure(figsize=(25, 25))
ax = fig.add_subplot(111)
cax = ax.matshow(corr_2017, vmin=-1, vmax=1,  cmap = 'plasma', interpolation = 'nearest',filternorm = False ) 
fig.colorbar(cax)
ticks = np.arange(0,19,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
# Data wrangling to calculate the 2020 results. This process is similar to the one above for the validation dataset. 
# For any questions about the code, reference the above. 
seeding_pred = pd.merge(NCAA_2020_tourn_teams, season_grp, how='left', left_on = ['Season','Team'], right_on = ['Season','Team'])
seeding_pred = pd.merge(seeding_pred,     won_final, how='left', left_on = ['Season','Team'], right_on = ['Season','Team'])
seeding_pred = pd.merge(seeding_pred,     massey_ord, how='left',   left_on = ['Season','Team'], right_on = ['Season','Team'])
seeding_pred = seeding_pred[(seeding_pred.Season == 2020)]
seeding_pred.drop(labels=['avg_off','avg_def'], inplace=True, axis=1)

seeding_pred = seeding_pred[['Season','Team','wp','pythag_overage','mov_diff_avg','COL','DOL','MOR','POM','RTH','SAG',
                        'WLK','WOL','min_ord','max_ord','std_dev','trend_ratio','schd_strngth','Minors','auto_bid']]
seeding_pred.rename(columns={'auto_bid':'won_final_game'}, inplace = True)

seeding_pred_a = seeding_pred.iloc[:,2:]
seeding_pred_join = seeding_pred.iloc[:,:2] 
pred_df_2020 = pd.DataFrame(clf_mod_2.predict(seeding_pred_a))
pred_df_2020 =  pd.merge(seeding_pred_join,pred_df_2020, left_index = True, right_index = True)

pred_df_2020.rename(columns={0:'pred_regression'}, inplace= True)
pred_df_2020['pred_rank'] = pred_df_2020['pred_regression'].rank()
pred_df_2020['pred_rank'] = np.where (pred_df_2020['pred_rank'] >=  45, pred_df_2020['pred_rank'] -2, pred_df_2020['pred_rank'])
pred_df_2020['pred_rank'] = np.where (pred_df_2020['pred_rank'] >= 65, pred_df_2020['pred_rank'] -2, pred_df_2020['pred_rank'])        
pred_df_2020['seed'] = np.ceil((pred_df_2020['pred_rank']/4))   
# We'll bring in the conferences and team names and print the results by seed. 
# Worth noting, over the last three years there have been six 11 seeds and six 16 seeds so that's what you'll see here. 

pred_df_2020 = pd.merge(pred_df_2020, conference, how='left', left_on = ['Season','Team'], right_on = ['Season','Team'])
pred_df_2020 = pd.merge(pred_df_2020, teams, how='left', left_on = ['Team'], right_on = ['Team'])
pred_df_2020 = pred_df_2020.iloc[:,[0,6,5,4,2]].sort_values('pred_regression')

# Raw data for those that prefer it not in chart form. 
for i in range(0,17,1):
    print(pred_df_2020[(pred_df_2020.seed == i)].iloc[:,:4])
    print()     
# Plotting he seeding for all 68 teams
f, ax = plt.subplots(figsize=(15,25))
plt.title('Team''s Predicted Seeding for 2020' ,fontsize=20)
sns.barplot(x="seed", y="TeamName", data=pred_df_2020,
            label="Total", palette ="plasma")
ax.set( ylabel="Team Names")
plt.show()
# We create a grouped data frame to chart our results for both bid count and average seeding
test_df = pred_df_2020.groupby('ConfAbbrev')['seed'].agg(['count','mean']).sort_values(by = 'count', ascending = False).reset_index(drop=False)

plt.figure(figsize=(20, 5))
sns.set(style='darkgrid')
ax = sns.barplot(x='ConfAbbrev', y='count', data=test_df.head(10), palette='autumn')
plt.title('Bids per conference',fontsize=20)
plt.ylabel('Selections')
plt.xlabel('Conference',fontsize=14)
plt.show()

plt.figure(figsize=(20, 5))
sns.set(style='darkgrid')
ax = sns.barplot(x='ConfAbbrev', y='mean', data=test_df.head(10).sort_values(by = 'mean', ascending = True).reset_index(drop=False), palette='autumn')
plt.title('Average seed per conference',fontsize=20)
plt.ylabel('avg seed')
plt.xlabel('Conference',fontsize=14)
plt.show()
# We'll do some grouping of data to create bar charts. 
pred_df = pd.merge(pred_df, conference, how='left', left_on = ['Season','Team'], right_on = ['Season','Team'])
pred_df = pd.merge(pred_df, teams, how='left', left_on = ['Team'], right_on = ['Team'])
pred_df = pred_df.iloc[:,[0,8,7,3,2,5,4,6]].sort_values('Season')
# Seeding accuracy 
pred_df_seed = pred_df.groupby('seed_int')['correct_seeds','within_one','seed_diff'].mean().reset_index(drop = False) #.sort_values(by = 'mean', ascending = False).reset_index(drop=False)
# Conference level accuracy and under and overseeding
pred_df['overseeded'] = pred_df['seed_int'] - pred_df['pred_rank'] 
pred_df_conf_mean = pred_df.groupby('ConfAbbrev')['correct_seeds','within_one','seed_diff','overseeded'].mean().reset_index(drop = False)
pred_df_conf_count = pred_df.groupby('ConfAbbrev')['correct_seeds'].count().reset_index(drop = False)
pred_df_conf_mean_overseed = pred_df_conf_mean[(pred_df_conf_mean.overseeded !=0)]
# Team level under and over seeding
pred_df_filter = pred_df[(pred_df.seed_diff > 1)]
pred_df_filter['overseeded'] =  pred_df_filter['seed_int'] - pred_df_filter['pred_rank'] 
pred_df_filter = pred_df_filter.iloc[:,[0,1,2,3,4,8]].sort_values(by = 'overseeded', ascending = False).reset_index(drop=True)    
# Plotting the outputs of our model's accuracy for each seed.
plt.figure(figsize=(30, 5))
sns.set(style='darkgrid')
ax = sns.barplot(x='seed_int', y='correct_seeds', data=pred_df_seed.head(16), palette='cool')
plt.title('Average Accuracy per Seed: 2017-2019',fontsize=18)
plt.ylabel('Accuracy')
plt.xlabel('Seeding',fontsize=14) 
plt.show()

# Plotting the outputs of our model's accuracy for each conference.
plt.figure(figsize=(30, 5))
sns.set(style='darkgrid')
ax = sns.barplot(x='ConfAbbrev', y='correct_seeds', data=pred_df_conf_mean.sort_values('correct_seeds'), palette='cool')
plt.title('Average Accuracy per Conference: 2017-2019',fontsize=18)
plt.ylabel('Accuracy')
plt.xlabel('Seeding',fontsize=14) 
plt.show()

# Plotting the outputs of our model's prediction differnce between avg conference seeding 
plt.figure(figsize=(30, 5))
sns.set(style='darkgrid')
ax = sns.barplot(x='ConfAbbrev', y='overseeded', data=pred_df_conf_mean_overseed.sort_values('overseeded', ascending = False), palette = 'plasma') #     color='teal')
plt.title('Average Underseeding per Conference: 2017-2019',fontsize=18)
plt.ylabel('Underseeding')
plt.xlabel('Conference',fontsize=14) 
plt.show()

# Plotting the outputs of our model's prediction differnce at a team level
plt.figure(figsize=(30, 5))
sns.set(style='darkgrid')
ax = sns.barplot(x='TeamName', y='overseeded', data=pred_df_filter, palette = 'plasma') #     color='teal')
plt.title('Underseeding: Teams 2017-2019',fontsize=18)
plt.ylabel('Underseeding')
plt.xlabel('Team',fontsize=14) 
plt.show()