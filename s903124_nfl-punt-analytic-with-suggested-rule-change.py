import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import HTML

import math

plt.style.use('seaborn')
video_review = pd.read_csv('../input/video_review.csv')

video_review.head()
player_role_data = pd.read_csv('../input/play_player_role_data.csv')

player_role_data.head()
play_information_data = pd.read_csv('../input/play_information.csv')

play_information_data.head()
video_footage_injury_data = pd.read_csv('../input/video_footage-injury.csv')

video_footage_injury_data.head()
len(video_review)
pd.value_counts(video_footage_injury_data['Type']).plot.bar()

plt.title('Concussed play season type')
pd.value_counts(video_footage_injury_data['season']).plot.bar()

plt.title('Concussed play season year')

plt.yticks(np.arange(25, step = 5))
pd.value_counts(video_review['Player_Activity_Derived']).plot.bar()

plt.title('Concussed player action')
pd.value_counts(video_review['Primary_Partner_Activity_Derived']).plot.bar()

plt.title('Concussion partner action')
pd.value_counts(video_review['Primary_Impact_Type']).plot.bar()

plt.title('Impact position')
pd.value_counts(video_review['Friendly_Fire']).plot.bar()

plt.title('Friendly fire?')
merged_data = pd.merge(video_review,play_information_data)

merged_data = pd.merge(merged_data,player_role_data)

merged_data.head()
receiving_position = ['PR', 'PFB', 'VR', 'PDR1', 'PDL2']



receive_data = merged_data[merged_data.Role.isin(receiving_position)]

pd.value_counts(receive_data['Role']).plot.bar()

plt.title('Concussed data by receiving team position')
punt_data = merged_data[~merged_data.Role.isin(receiving_position)]

pd.value_counts(punt_data['Role']).plot.bar()

plt.yticks(np.arange(5))

plt.title('Concussed data by punting team position')
punt_data['Concussed_position'] = 'kicking'

receive_data['Concussed_position'] = 'receiving'

pd.value_counts(pd.merge(receive_data,punt_data,how='outer')['Concussed_position']).plot.bar()

plt.title('Concussed count by team split')
def return_yardline_from_string(pos_team,yardline_string):

    

    if (pos_team in yardline_string): #opponent yardline

        yardline = 100 - float(yardline_string.split(' ')[1])

    else:

        yardline = float(yardline_string.split(' ')[1])

        

    return yardline    
#merging all data using pivot table

table = pd.pivot_table(player_role_data,index=['GameKey', 'PlayID'],columns=['Role'], aggfunc=lambda x: len(x.unique()))['GSISID'].fillna(0) 



table.reset_index(inplace=True)



all_punt_data = pd.merge(table,play_information_data)

all_punt_data = pd.merge(all_punt_data,video_review,how='outer')



punt_yards_list = []

adjust_yardline_list = []

returned_list = []

return_yards_list = [] 

fair_catch_list = [] 

for i,yards in enumerate(all_punt_data.PlayDescription.str.split('punts ').str[1].str[:2]):

    try:

        if('No Play' in all_punt_data['PlayDescription'][i] or 'Direct snap' in all_punt_data['PlayDescription'][i] or 'pass' in all_punt_data['PlayDescription'][i] or 'FUMBLE' in all_punt_data['PlayDescription'][i]):

            punt_yards_list.append(np.nan)

        else:

            punt_yards_list.append(float(yards))

    except ValueError:

        punt_yards_list.append(np.nan)



    if(all_punt_data['Poss_Team'][i] in all_punt_data['YardLine'][i]):

        adjust_yardline_list.append(float(all_punt_data['YardLine'][i].split(' ')[1]))

    else:

        adjust_yardline_list.append(100-float(all_punt_data['YardLine'][i].split(' ')[1]))

    if(' for ' in all_punt_data['PlayDescription'][i]): #returned kick

        returned_list.append(1)

    else:

        returned_list.append(0)

    if('fair catch' in all_punt_data['PlayDescription'][i]):

        fair_catch_list.append(1)

    else:

        fair_catch_list.append(0)

        

for i,yards in enumerate(all_punt_data.PlayDescription.str.split(' for ').str[1].str[:2]):        

    try:

        if('No Play' in all_punt_data['PlayDescription'][i] or 'fair catch' in all_punt_data['PlayDescription'][i] or 'Direct snap' in all_punt_data['PlayDescription'][i] or 'pass' in all_punt_data['PlayDescription'][i]):

            return_yards_list.append(np.nan)

        elif('MUFFS' in all_punt_data['PlayDescription'][i]):

            return_yards_list.append(0)

        elif(str(yards) == 'no'):

            return_yards_list.append(0)

        elif('PENALTY' in all_punt_data['PlayDescription'][i]):

            return_yards_list.append(float(yards))

#             if(all_punt_data['PlayDescription'][i].split('PENALTY on ')[1][:3] == all_punt_data['Poss_Team'][i]): #kicking team penalty

#                 return_yards_list.append(float(yards))

#             elif('TOUCHDOWN.' in all_punt_data['PlayDescription'][i]): #penalty after touchdown    

#                 return_yards_list.append(float(yards))

#             else:

#                 return_end_yardline = return_yardline_from_string(all_punt_data['Poss_Team'][i],all_punt_data['PlayDescription'][i].split('to ')[1][:6])           

#                 if(float(all_punt_data['PlayDescription'][i].split('enforced at ')[1][:2]) != 50):

#                     penalty_yardline = return_yardline_from_string(all_punt_data['Poss_Team'][i],all_punt_data['PlayDescription'][i].split('enforced at ')[1][:6])

#                 else:

#                     penalty_yardline = 50

#                 return_yards_list.append(return_end_yardline - penalty_yardline)

                

        else:    

            return_yards_list.append(float(yards))

    except ValueError:    

        return_yards_list.append(np.nan)       

all_punt_data['punt_yards'] = punt_yards_list

all_punt_data['adjust_yardline'] = adjust_yardline_list

all_punt_data['returned'] = returned_list

all_punt_data['concussed'] = all_punt_data.GSISID.notnull().astype(int)

all_punt_data['return_yards'] = return_yards_list

all_punt_data['fair_catch'] = fair_catch_list

all_punt_data['blocked'] = (all_punt_data.PlayDescription.str.contains('BLOCKED') == True | (all_punt_data.punt_yards <= 10)).astype(int)
all_punt_data.to_csv('punt_play_data.csv')
concussed_split = all_punt_data.groupby(['returned'])['concussed'].mean()



print("Non-return punt concussed percentage = %1.1f%%" % (100*concussed_split[0]))

print("Returned punt concussed percentage = %1.1f%%" % (100*concussed_split[1]))
pd.value_counts(all_punt_data[all_punt_data['concussed'] == 1]['returned']).plot.bar()

plt.xticks([0,1], ('Punt return', 'Others'))

plt.title('Concussion number by punt outcome')
single_coverage = all_punt_data[(all_punt_data['VR'] == 1) & (all_punt_data['VL'] == 1) ]



double_coverage = all_punt_data[(all_punt_data['VR'] == 0) & (all_punt_data['VL'] == 0) ]



hybrid_coverage = all_punt_data[(all_punt_data['VR'] == 0) ^ (all_punt_data['VL'] == 0) ]
print("Number of single coverage punt: %d" % len(single_coverage))

print("Concussions from single coverage punt: %d\n" % single_coverage.Primary_Impact_Type.notna().sum())

print("Number of hybrid coverage punt: %d" % len(hybrid_coverage))

print("Concussions from hybrid coverage punt: %d\n" % hybrid_coverage.Primary_Impact_Type.notna().sum())

print("Number of double coverage punt: %d" % len(double_coverage))

print("Concussions from double coverage punt: %d" % double_coverage.Primary_Impact_Type.notna().sum())
print("single coverage average starting yardline = %1.1f" % np.nanmean(np.array(single_coverage.adjust_yardline).astype(float)))

print("hybrid coverage average starting yardline = %1.1f" % np.nanmean(np.array(hybrid_coverage.adjust_yardline).astype(float)))

print("double coverage average starting yardline = %1.1f" % np.nanmean(np.array(double_coverage.adjust_yardline).astype(float)))
hybrid_coverage['coverage_type'] = 0

single_coverage['coverage_type'] = 1

double_coverage['coverage_type'] = 2



regression_df = pd.concat([single_coverage,double_coverage,hybrid_coverage])
import statsmodels

import statsmodels.api as sm



import statsmodels.formula.api as smf





results = smf.logit(formula='returned ~ C(coverage_type) + adjust_yardline', data=regression_df).fit()
results.summary()
print("Odd ratio of return for single coverage compare to hybrid coverage = %f" % np.exp(results.params)[1])

print("Odd ratio of return for single coverage compare to double coverage = %f" % (np.exp(results.params)[1] * np.exp(results.params)[2]))
regression_df_punt = regression_df.dropna(subset=['punt_yards'])

regression_df_return = regression_df.dropna(subset=['return_yards'])
results = smf.ols(formula='punt_yards ~ adjust_yardline +  C(coverage_type)', data=regression_df_punt).fit()
results.summary()
print("Average punting line of scrimmage = %1.3f\n" % np.mean(all_punt_data['adjust_yardline']))



print("Probability of return a punt from 34 yardlime for single coverage: %f" % (math.exp(-0.679+2.133-0.0623*34) / (1+math.exp(-0.679+2.133-0.0623*34))))

print("Probability of return a punt from 34 yardlime for hybrid coverage: %f" % (math.exp(0+2.133-0.0623*34) / (1+math.exp(0+2.133-0.0623*34))))

print("Probability of return a punt from 34 yardlime for double coverage: %f" % (math.exp(0.208+2.133-0.0623*34) / (1+math.exp(0.208+2.133-0.0623*34))))

print("Number of return of all punt = %d\n"% np.nansum(np.array(all_punt_data.returned).astype(float)) )



print("Single coverage number of return = %d" % np.nansum(np.array(single_coverage.returned).astype(float)))

print("Hybrid coverage number of return = %d" % np.nansum(np.array(hybrid_coverage.returned).astype(float)))

print("Double coverage number of return = %d\n" % np.nansum(np.array(double_coverage.returned).astype(float)))







print("Average yards per return of all punt = %1.2f\n"% np.nanmean(np.array(all_punt_data.return_yards).astype(float)) )



print("Single coverage average yards per return = %1.2f" % np.nanmean(np.array(single_coverage.return_yards).astype(float)))

print("Hybrid coverage average yards per return = %1.2f" % np.nanmean(np.array(hybrid_coverage.return_yards).astype(float)))

print("Double coverage average yards per return = %1.2f" % np.nanmean(np.array(double_coverage.return_yards).astype(float)))
results = smf.ols(formula='return_yards ~ adjust_yardline +  C(coverage_type)', data=regression_df_return).fit()
results.summary()
print("Decrease in effective punt yards at own 34 = %1.2f" % ((1865/(1865+1423+3379))*(1.03+(0.503-0.34)*(11-0.0695*34-1.03))+(1423/(1865+1423+3379))*((1.03-0.12)+(0.55-0.34)*(11-0.0695*34-1.03+0.12))))
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153233/Kadeem_Carey_punt_return-Vwgfn5k9-20181119_152809972_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153274/Haack_punts_41_yards-SRJMeOc3-20181119_165546590_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153511/Lechler_58_yard_punt-8LArhoQg-20181121_123420599_5000k.mp4" type="video/mp4"></video>')
HTML('<video width="800" height="600" controls> <source src="http://a.video.nfl.com//films/vodzilla/153517/Schmidt_57_yard_punt-EMXj28Mw-20181121_124742503_5000k.mp4" type="video/mp4"></video>')
file_location = ['NGS-2016-pre.csv', 'NGS-2016-reg-wk1-6.csv', 'NGS-2016-reg-wk7-12.csv', 'NGS-2016-reg-wk13-17.csv','NGS-2016-post.csv',

                 'NGS-2017-pre.csv', 'NGS-2017-reg-wk1-6.csv', 'NGS-2017-reg-wk7-12.csv', 'NGS-2017-reg-wk13-17.csv','NGS-2017-post.csv']



tracking_df = pd.DataFrame()



for i in range(10):

    data = pd.read_csv('../input/' + file_location[i])

    data = pd.merge(data,play_information_data,how='outer')

    data = pd.merge(data,player_role_data,how='outer')

    punter_data = data[(data.Event=='punt') & (data.Role == 'P')][['x','y','Season_Year' ,'GameKey','PlayID','Time']]

    punter_data.columns = ['punter_x', 'punter_y','Season_Year','GameKey','PlayID','Time']

    receiving_position = ['PDL1', 'PDL2','PDL3', 'PDL4', 'PDL5', 'PDL6', 'PDM', 'PDR1', 'PDR2', 'PDR3','PDR4', 'PDR5', 'PDR6', 'PFB','PLL', 'PLL1', 'PLL2',

       'PLL3', 'PLM', 'PLM1', 'PLR', 'PLR1', 'PLR2', 'PLR3','VL', 'VLi', 'VLo', 'VR', 'VRi', 'VRo', 'PR']



    punt_time = np.array(punter_data.Time)

    playid = np.array(punter_data.PlayID)

    gamekey = np.array(punter_data.GameKey)



    receiving_team_data = data[(data.Event=='punt') & (data.Role.isin(receiving_position))] #All member of receiving team at time of punt

    

    merged_data = pd.merge(receiving_team_data,punter_data,how='left').dropna()

    merged_data['distance_to_P'] = ((merged_data['x'] - merged_data['punter_x'])**2-(merged_data['y'] - merged_data['punter_y']))**0.5

    output_df = merged_data[merged_data['distance_to_P'] < 10].groupby(['Season_Year','GameKey','PlayID'])['Role'].count().reset_index() #group all receiving team player around 10 yards of punter

    output_df.columns = ['Season_Year', 'GameKey', 'PlayID', 'no_blocker']



    

    tracking_df = tracking_df.append(output_df)      
all_punt_data = pd.merge(all_punt_data,tracking_df)
pd.value_counts(all_punt_data['no_blocker']).plot.bar()

plt.title('Number of blockers in punt')
results = smf.logit(formula='concussed ~ adjust_yardline  + no_blocker', data=all_punt_data).fit()
results.summary()
results = smf.logit(formula='returned ~ adjust_yardline  + no_blocker', data=all_punt_data).fit()

results.summary()
all_punt_data['overload'] =  ((all_punt_data['PDL1'] + all_punt_data['PDL2'] + all_punt_data['PDL3'] + all_punt_data['PDL4'] + all_punt_data['PDL5'] + all_punt_data['PDL6']) - \

(all_punt_data['PDR1'] + all_punt_data['PDR2'] + all_punt_data['PDR3'] + all_punt_data['PDR4'] + all_punt_data['PDR5'] + all_punt_data['PDR6']) + \

(all_punt_data['PLL1'] + all_punt_data['PLL2'] + all_punt_data['PLL3']) - \

(all_punt_data['PLR1'] + all_punt_data['PLR2'] + all_punt_data['PLR3'])).abs()
results = smf.logit(formula='concussed ~ adjust_yardline  + overload', data=all_punt_data).fit()

results.summary()
results = smf.logit(formula='returned ~ adjust_yardline  + overload', data=all_punt_data).fit()

results.summary()
file_location = ['NGS-2016-pre.csv', 'NGS-2016-reg-wk1-6.csv', 'NGS-2016-reg-wk7-12.csv', 'NGS-2016-reg-wk13-17.csv','NGS-2016-post.csv',

                 'NGS-2017-pre.csv', 'NGS-2017-reg-wk1-6.csv', 'NGS-2017-reg-wk7-12.csv', 'NGS-2017-reg-wk13-17.csv','NGS-2017-post.csv']



tracking_df = pd.DataFrame()



for i in range(10):

    data = pd.read_csv('../input/' + file_location[i])

    data = pd.merge(data,play_information_data,how='outer')

    data = pd.merge(data,player_role_data,how='outer')

    returner_data = data[((data.Event=='fair_catch') | (data.Event=='punt_received')) & (data.Role == 'PR')][['x','y','Season_Year' ,'GameKey','PlayID','Time']]

    returner_data.columns = ['punter_x', 'punter_y','Season_Year','GameKey','PlayID','Time']

    receiving_position = ['PDL1', 'PDL2','PDL3', 'PDL4', 'PDL5', 'PDL6', 'PDM', 'PDR1', 'PDR2', 'PDR3','PDR4', 'PDR5', 'PDR6', 'PFB','PLL', 'PLL1', 'PLL2',

       'PLL3', 'PLM', 'PLM1', 'PLR', 'PLR1', 'PLR2', 'PLR3','VL', 'VLi', 'VLo', 'VR', 'VRi', 'VRo', 'PR']



    return_time = np.array(returner_data.Time)

    playid = np.array(returner_data.PlayID)

    gamekey = np.array(returner_data.GameKey)



    kicking_team_data = data[((data.Event=='fair_catch') | (data.Event=='punt_received')) & ~(data.Role.isin(receiving_position))] #All member of receiving team at time of punt

    

    merged_data = pd.merge(kicking_team_data,returner_data,how='left').dropna()

    merged_data['distance_to_PR'] = ((merged_data['x'] - merged_data['punter_x'])**2-(merged_data['y'] - merged_data['punter_y']))**0.5

    output_df = merged_data[merged_data['distance_to_PR'] < 5].groupby(['Season_Year','GameKey','PlayID'])['Role'].count().reset_index() #group all receiving team player around 10 yards of punter

    output_df.columns = ['Season_Year', 'GameKey', 'PlayID', 'close_defender']



    

    tracking_df = tracking_df.append(output_df)      
close_defender_data = pd.merge(all_punt_data,tracking_df)
pd.value_counts(close_defender_data['close_defender']).plot.bar()
results = smf.logit(formula='returned ~ close_defender + adjust_yardline', data=close_defender_data).fit()

results.summary()
results = smf.logit(formula='concussed ~ close_defender + adjust_yardline', data=close_defender_data).fit()

results.summary()
results = smf.logit(formula='concussed ~ fair_catch + adjust_yardline', data=all_punt_data).fit()

results.summary()
all_punt_data[['punt_yards','adjust_yardline']].groupby('adjust_yardline').mean().plot(kind='bar')

plt.xticks([])

plt.show()