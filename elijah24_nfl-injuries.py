from IPython.display import Image

Image("../input/document-images/banner.png")
Image("../input/documentation-images/table1.png")
Image("../input/documentation-images/table2.png")
Image("../input/documentation-images/table3.png")
Image("../input/documentation-images/table4.png")
Image("../input/documentation-images/coordinate_diagram.png")
#import tensorflow as tf

import os

import matplotlib.pyplot as plt

from matplotlib import animation

import numpy as np

import pandas as pd

import seaborn as sns

from IPython.display import HTML

from scipy.stats import ttest_ind

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score,roc_curve,mean_squared_error,mean_absolute_error

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from statsmodels.genmod.generalized_linear_model import GLM

from statsmodels.genmod import families

import shap

import math

import sys

import urllib.request as urllib2
def cleanStadiumType(df):

    df["StadiumType"] = df["StadiumType"].str.lower()

    df.loc[df["StadiumType"].isna(),"StadiumType"] = "unk"

    df["StadiumType"]=df["StadiumType"].str.replace('domed','dome')

    df.loc[df["StadiumType"].str.contains('out'),"StadiumType"] = "outdoor"

    df.loc[df["StadiumType"].str.contains('our'),"StadiumType"] = "outdoor"

    df.loc[df["StadiumType"].str.contains('open'),"StadiumType"] = "outdoor"

    df.loc[df["StadiumType"].str.contains('bowl'),"StadiumType"] = "outdoor"

    df.loc[df["StadiumType"].str.contains('heinz'),"StadiumType"] = "outdoor"

    df.loc[df["StadiumType"].str.contains('oudoor'),"StadiumType"] = "outdoor"

    df.loc[df["StadiumType"].str.contains('closed'),"StadiumType"] = "indoor"

    df.loc[df["StadiumType"].str.contains('dome'),"StadiumType"] = "indoor"

    df.loc[df["StadiumType"].str.contains('roof'),"StadiumType"] = "indoor"

    df.loc[df["StadiumType"].str.contains('indoor'),"StadiumType"] = "indoor"

    df.loc[df["StadiumType"].str.contains('cloudy'),"StadiumType"] = "unk"

    return(df)

    

def cleanWeather(df):

    df["Weather"] = df["Weather"].str.lower()

    df["Weather"] = df["Weather"].fillna("unk")

    df.loc[df["Weather"].str.contains('ear'),"Weather"] = "clear"

    df.loc[df["Weather"].str.contains('fair'),"Weather"] = "clear"

    df.loc[df["Weather"].str.contains('oud'),"Weather"] = "cloudy"

    df.loc[df["Weather"].str.contains('clou'),"Weather"] = "cloudy"

    df.loc[df["Weather"].str.contains('overcast'),"Weather"] = "cloudy"

    df.loc[df["Weather"].str.contains('hazy'),"Weather"] = "cloudy"

    df.loc[df["Weather"].str.contains('snow'),"Weather"] = "snow"

    df.loc[df["Weather"].str.contains('rain'),"Weather"] = "rain"

    df.loc[df["Weather"].str.contains('show'),"Weather"] = "rain"

    df.loc[df["Weather"].str.contains('sun'),"Weather"] = "sunny"

    df.loc[df["Weather"].str.contains('indoor'),"Weather"] = "indoor"

    df.loc[df["Weather"].str.contains('controlled climate'),"Weather"] = "indoor"

    df.loc[df["Weather"].str.contains('cold'),"Weather"] = "unk"

    df.loc[df["Weather"].str.contains('heat'),"Weather"] = "unk"

    return df



def get_velocity(df):

    vel = [math.sqrt((df.x.values[i+1] - df.x.values[i])**2 + (df.y.values[i+1] - df.y.values[i])**2)/.1 for i in range(0,(len(df.x)-1))]

    df['velocity'] = np.concatenate((np.array([0]), np.array(vel))).flat

    return df



def orientation_diff(df):

    diff_o = [0]

    for i in range(0,(len(df.o)-1)):

        difference = df.o.values[i]-df.o.values[i+1]

        while (difference < -180): difference += 360

        while (difference > 180): difference -= 360

        diff_o.append(difference)

    df['diff_o'] = diff_o

    return df



def get_diff_os(df,col = 'velocity'):

    noise = (np.random.uniform(-1, 1,len(df)))*0.01

    if 'diff_o' not in df.columns:

        df = orientation_diff(df)

    if 'velocity' not in df.columns:

        df = get_velocity(df)

    diff_os = np.multiply(np.array(df.diff_o.values + noise),np.array(df[col]))

    df['diff_os'] = diff_os

    return df



def ecdf(data):

    """Compute ECDF for a one-dimensional array of measurements."""

    # Number of data points: n

    n = len(data)



    # x-data for the ECDF: x

    x = np.sort(data)



    # y-data for the ECDF: y

    y = np.arange(1, n+1) / n



    return x, y



def diff_of_means(data_1, data_2):

    """Difference in means of two arrays."""



    # The difference of means of data_1, data_2: diff

    diff = np.mean(data_1)-np.mean(data_2)



    return diff



def permutation_sample(data1, data2):

    """Generate a permutation sample from two data sets."""



    # Concatenate the data sets: data

    data = np.concatenate((data1,data2))



    # Permute the concatenated array: permuted_data

    permuted_data = np.random.permutation(data)



    # Split the permuted array into two: perm_sample_1, perm_sample_2

    perm_sample_1 = permuted_data[:len(data1)]

    perm_sample_2 = permuted_data[len(data1):]



    return perm_sample_1, perm_sample_2



def draw_perm_reps(data_1, data_2, func, size=1):

    """Generate multiple permutation replicates."""



    # Initialize array of replicates: perm_replicates

    perm_replicates = np.empty(size)



    for i in range(size):

        # Generate permutation sample

        perm_sample_1, perm_sample_2 = permutation_sample(data_1,data_2)



        # Compute the test statistic

        perm_replicates[i] = func(perm_sample_1,perm_sample_2)



    return perm_replicates



def bootstrap_replicate_1d(data, func):

    return func(np.random.choice(data, size=len(data)))



def draw_bs_reps(data, func, size=1):

    """Draw bootstrap replicates."""



    # Initialize array of replicates: bs_replicates

    bs_replicates = np.empty(size)



    # Generate replicates

    for i in range(size):

        bs_replicates[i] = bootstrap_replicate_1d(data,func)



    return bs_replicates



def perm_test(list1,list2, size):

    # Compute difference of mean impact force from experiment: empirical_diff_means

    empirical_diff_means = diff_of_means(list1,list2)



    # Draw 10,000 permutation replicates: perm_replicates

    perm_replicates = draw_perm_reps(list1,list2,

                                     diff_of_means, size=size)



    # Compute p-value: p

    p = np.sum(perm_replicates >= empirical_diff_means) / len(perm_replicates)

    

    if p > 0.5:

        p_val = (1-p)

        print('two-tail p-value =', (1-p)*2)

    else:

        p_val = p

        print('two-tail p-value =', p*2)



    return(empirical_diff_means, perm_replicates, p)

    

def myFE(myList):

    my_met_list = []

    velocity_list = []

    diff_os_list = []

    os_met_list = []

    for i in np.arange(0,len(myList)):

        temp = playertrack[playertrack["PlayKey"] == myList[i]]

        crnt_events = temp.event.drop_duplicates().fillna("")

        strt_event = crnt_events[[i in init_event for i in crnt_events]]

        if len(strt_event) > 0:

            start = temp[temp["event"] == np.array(strt_event)[0]]

        else:

            my_met_list.append(999.0)

            os_met_list.append(999.0)

        if len(strt_event) == 0:

          continue

        temp = temp.loc[start.index.values[0]: ,]

        dist = np.sqrt(abs(temp.tail(1).x.values[0] - temp.head(1).x.values[0])*abs(temp.tail(1).y-temp.head(1).y.values[0]))

        my_met = math.log(dist/temp.dis.sum())

        temp = get_diff_os(temp,'velocity')

        os_met = np.sum(np.sum(temp.diff_os > 34.22) + np.sum(temp.diff_os < -29.88)) # upper and lower limits selected to align with 2.5 and 97.5 percentiles of diff_os for all plays 

        velocity_list.append(np.array(temp.velocity.values))

        diff_os_list.append(np.array(temp.diff_os.values))

        my_met_list.append(my_met)

        os_met_list.append(os_met)

    player_df = pd.DataFrame({'PlayKey':myList,'my_met': my_met_list,'os_met': os_met_list, 'diff_os': diff_os_list, 'velocity': velocity_list})

    return(player_df)



def bootTest(sample1, sample2):

    # Compute difference of mean impact force from experiment: empirical_diff_means

    empirical_diff_means = diff_of_means(sample1,sample2)



    # Compute mean of all forces: mean_force

    mean_all = np.mean(np.concatenate([sample1,sample2]))



    # Generate shifted arrays

    sample1_shifted = sample1 - np.mean(sample1) + mean_all

    sample2_shifted = sample2 - np.mean(sample2) + mean_all 



    # Compute 10,000 bootstrap replicates from shifted arrays

    bs_replicates_sample1 = draw_bs_reps(sample1_shifted, np.mean, 10000)

    bs_replicates_sample2 = draw_bs_reps(sample2_shifted, np.mean, 10000)



    # Get replicates of difference of means: bs_replicates

    bs_replicates =  bs_replicates_sample1 - bs_replicates_sample2



    # Compute and print p-value: p

    p = np.sum(bs_replicates > empirical_diff_means) / 10000

    print('p-value =', p)

    

    # Compute the 95% confidence interval: conf_int

    conf_int = np.percentile(bs_replicates,[2.5,97.5])



    # Print the confidence interval

    print('95% confidence interval =', conf_int, 'games')

    

    return(bs_replicates, empirical_diff_means, conf_int, p)



def plotFieldTypeWeather(df):

    syn = df[df["FieldType"] == 'Synthetic'].groupby("Weather").count()

    syn = syn["PlayerKey"]

    nat = df[df["FieldType"] == 'Natural'].groupby("Weather").count()

    nat = nat["PlayerKey"]

    nat = nat[syn.index.values]

    N = len(nat)

    ind = syn.index.values    # the x locations for the groups

    width = 0.35       # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(x=ind, height = nat, width = width)

    p2 = plt.bar(x=ind, height = syn, width = width, bottom=nat)

    plt.ylabel('Count')

    plt.title('Counts by FieldType and Weather')

    plt.legend((p1[0], p2[0]), ('Natural', 'Synthetic'))

    plt.show()
injury = pd.read_csv("../input/nfl-playing-surface-analytics/InjuryRecord.csv")

playlist = pd.read_csv("../input/nfl-playing-surface-analytics/PlayList.csv")

playertrack = pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv")
img = plt.imread("../input/field-background/field_background.png")
## Injury PreProcessing

#The DM_M# variables can be converted to discrete ordinal variables representing an at least time window. 0 being less than a week, 1 being at least 1 week but not 2, etc...

injury["DM_Sum"] = injury["DM_M1"] + injury["DM_M7"]  + injury["DM_M28"] + injury["DM_M42"] 

injury.loc[injury["DM_Sum"] >2 ,"DM_Sum"] = injury.loc[injury["DM_Sum"] >2 ,"DM_Sum"] + 2

injury.loc[injury["DM_Sum"] >5 ,"DM_Sum"] = injury.loc[injury["DM_Sum"] >5 ,"DM_Sum"] + 1

injury["DM_Sum"] = injury["DM_Sum"] -1

injury["BodyPart_num"] = injury["BodyPart"] + injury["DM_Sum"].astype(str)

injury["DM_Sum_adj"] = injury.DM_Sum + np.random.uniform(-0.5, 0.5,105)

injury["BodyPart"] = injury["BodyPart"].astype('category')

injury["BodyPart_cat"] = injury["BodyPart"].cat.codes

injury["BodyPart_cat"] = injury.BodyPart_cat + np.random.uniform(-0.2, 0.2,105) + 1



found_PlayKey = ['33337-2-25', '45099-5-1', '36591-9-4', '45950-6-81', '39653-4-68', '38253-10-13', '38214-12-36', '43119-12-66', '35648-12-40', '40051-13-30', '39671-12-24', '43229-15-42', '46021-19-68', '38259-2-37', '45158-32-17', '36572-4-30', '43490-9-30', '36573-14-52', '46134-18-19', '47196-7-45', '45975-23-52', '47273-10-50', '40405-29-14', '44423-13-27', '31933-20-26', '47285-4-16', '37068-19-20', '36696-24-22']

injury.loc[injury.PlayKey.isna(),'PlayKey'] = found_PlayKey



playlist = cleanWeather(playlist)

playlist = cleanStadiumType(playlist)

playlist.loc[playlist.PlayType.isna(),'PlayType'] = 'unk'

playlist.loc[playlist.PlayType == '0','PlayType'] = 'unk'

playlist.loc[(playlist["Weather"] == 'unk') & (playlist["StadiumType"] == 'unk'), "StadiumType" ] = 'indoor'

playlist.loc[(playlist["Weather"] == 'unk') & (playlist["StadiumType"] == 'indoor'), "Weather" ] = 'indoor'

playlist.loc[(playlist["Weather"] == 'indoor') & (playlist["StadiumType"] == 'unk'), "StadiumType" ] = 'indoor'

playlist.loc[(playlist["Weather"] == 'rain') & (playlist["StadiumType"] == 'unk'), "Weather" ] = 'outdoor'

playlist.loc[(playlist["Weather"] == 'cloudy') & (playlist["StadiumType"] == 'unk'), "StadiumType" ] = 'outdoor'

playlist.loc[(playlist["Weather"] != 'indoor') & (playlist["StadiumType"] == 'unk'), "StadiumType" ] = 'outdoor'



injury = pd.merge(injury.drop_duplicates(),playlist.drop_duplicates(),'left',['PlayerKey','GameID','PlayKey'])

### Get sample of non injured players

non_inj_sample = ['38274-11-30', '33337-3-17', '44424-11-43', '36554-15-2', '45061-1-10', '42398-15-18', '39715-1-8', '36579-10-2', '47784-13-25', '43489-11-60', '36572-30-27', '39794-29-50', '47220-4-9', '47273-12-12', '42370-16-13', '42344-21-62', '42413-10-15', '42448-16-2', '39836-22-38', '42432-3-50', '42413-11-8', '38325-25-44', '43523-17-3', '34243-8-28', '47282-12-45', '44506-4-6', '31266-8-7', '43229-7-12', '44546-22-58', '44527-16-13', '45962-16-53', '47282-1-11', '43505-14-46', '36555-8-50', '46430-11-4', '42549-9-4', '43672-11-6', '45962-6-15', '43483-24-16', '47278-13-9', '47287-4-54', '35648-10-22', '43535-8-58', '45061-20-7', '40335-31-38', '47307-5-21', '39956-9-21', '45099-1-1', '39731-9-6', '43490-20-43', '43490-30-26', '44418-17-15', '44480-12-6', '44451-14-40', '43119-11-3', '40474-21-25', '42588-2-49', '46038-3-52', '45953-17-43', '44511-20-47', '42346-6-47', '39583-16-31', '40345-11-2', '39038-17-74', '43672-3-6', '42352-21-36', '36757-7-14', '47287-3-54', '43483-20-23', '39680-31-11', '47282-14-40', '44506-9-12', '39653-16-48', '42399-19-44', '44158-8-40', '41558-28-52', '40345-3-3', '38259-5-49', '42344-21-14', '45158-30-13', '42359-10-19', '42406-10-42', '38274-7-42', '44527-1-55', '43050-23-30', '46098-15-30', '46066-6-2', '43656-27-50', '44158-16-41', '43505-2-37', '43050-15-4', '44037-26-66', '43050-23-26', '36572-32-22', '44629-30-42', '42549-24-55', '43540-2-1', '36555-18-27', '46038-9-17', '45927-21-53', '34230-22-15', '47287-15-12', '42404-4-54', '44449-29-10', '44511-21-7']

inj_player = injury['PlayKey']

init_event = ["ball_snap", "kickoff","kickoff_play", "onside_kick","free_kick","free_kick_play", "snap_direct"]
players_df = pd.DataFrame({'PlayKey': np.concatenate((np.array(non_inj_sample),np.array(inj_player))),'injured': np.concatenate((np.zeros(len(non_inj_sample)), np.ones(len(inj_player))))})

players_df2 = pd.merge(players_df.drop_duplicates(),playlist.drop_duplicates(),'left',['PlayKey']).drop(['PlayerKey'],axis=1)

players_df2_features = myFE(players_df2.PlayKey)
players_df3 = pd.merge(players_df2,players_df2_features,'left',['PlayKey'])
Image("../input/document-images/diff_os_def.png", width='500')
Image("../input/document-images/os_met_def.png", width='300')
all_diff_os = np.hstack(np.array([np.array(players_df3.diff_os.values[i]) for i in range(0,len(players_df3.diff_os.values))]))

diff_os_limits = np.percentile(all_diff_os, [2.5,97.5])

all_diff_os_inj = np.hstack(np.array([np.array(players_df3[players_df3.injured == 1].diff_os.values[i]) for i in range(0,len(players_df3[players_df3.injured == 1]))]))

diff_os_limits_inj = np.percentile(all_diff_os_inj, [2.5,97.5])

all_diff_os_non_inj = np.hstack(np.array([np.array(players_df3[players_df3.injured == 0].diff_os.values[i]) for i in range(0,len(players_df3[players_df3.injured == 0]))]))

diff_os_limits_non_inj = np.percentile(all_diff_os_non_inj, [2.5,97.5])



print('diff_os limit for All:' + str(diff_os_limits))

print('diff_os limit for Injured:' + str(diff_os_limits_inj))

print('diff_os limit for Non Injured:' + str(diff_os_limits_non_inj))
all_velocity = np.hstack(np.array([np.array(players_df3.velocity.values[i]) for i in range(0,len(players_df3))]))

all_velocity_inj = np.hstack(np.array([np.array(players_df3[players_df3.injured == 1].velocity.values[i]) for i in range(0,len(players_df3[players_df3.injured == 1]))]))

all_velocity_non_inj = np.hstack(np.array([np.array(players_df3[players_df3.injured == 0].velocity.values[i]) for i in range(0,len(players_df3[players_df3.injured == 0]))]))

all_velocity_syn = np.hstack(np.array([np.array(players_df3[players_df3.FieldType == 'Synthetic'].velocity.values[i]) for i in range(0,len(players_df3[players_df3.FieldType == 'Synthetic']))]))

all_velocity_nat = np.hstack(np.array([np.array(players_df3[players_df3.FieldType == 'Natural'].velocity.values[i]) for i in range(0,len(players_df3[players_df3.FieldType == 'Natural']))]))
plt.hist(all_velocity_syn, density=True,alpha=0.3)

plt.xlim(0,15)

plt.hist(all_velocity_nat[all_velocity_nat <15], density=True,alpha=0.3)

plt.title('Velocity by Field Type')

plt.legend(['Synthetic','Natural'])
perm_test(all_velocity_syn[all_velocity_syn <15],all_velocity_nat[all_velocity_nat <15],10000)
print('Synthetic mean velocity = ' + str(round(np.mean(all_velocity_syn),2)))

print('Natural mean velocity = ' + str(round(np.mean(all_velocity_nat[all_velocity_nat <100]),2)))
all_temp_inj = players_df3[players_df3.injured == 1].Temperature

all_temp_inj = all_temp_inj[all_temp_inj >-30]

all_temp_non_inj = players_df3[players_df3.injured == 0].Temperature

all_temp_non_inj = all_temp_non_inj[all_temp_non_inj >-30]
plt.hist(all_temp_inj,alpha=.2)

plt.hist(all_temp_non_inj,alpha=.2)

plt.title('Temperature by Injury')

plt.legend(['Injured','Non Injured'])
stat, p = ttest_ind(all_temp_inj, all_temp_non_inj)

print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret

alpha = 0.05

if p > alpha:

    print('Temperature Injured & Non-injured Same distributions (fail to reject H0)')

else:

    print('Temperature Injured & Non-injured Different distributions (reject H0)')
all_temp_inj_syn = players_df3[(players_df3.injured == 1)&(players_df3.FieldType == 'Synthetic')].Temperature

all_temp_inj_syn = all_temp_inj_syn[all_temp_inj_syn >-30]

all_temp_inj_nat = players_df3[(players_df3.injured == 1)&(players_df3.FieldType == 'Natural')].Temperature

all_temp_inj_nat = all_temp_inj_nat[all_temp_inj_nat >-30]
plt.hist(all_temp_inj_syn,density=True, alpha=.2)

plt.hist(all_temp_inj_nat,density=True,alpha=.2)

plt.title('Temperature by Field Type')

plt.legend(['Synthetic','Natural'])
stat, p = ttest_ind(all_temp_inj_syn, all_temp_inj_nat)

print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret

alpha = 0.05

if p > alpha:

    print('Temperature Synthetic & Natural Same distributions (fail to reject H0)')

else:

    print('Temperature Synthetic & Natural Different distributions (reject H0)')
last_play_inj = [PlayKey[-2:] for PlayKey in injury.PlayKey]

last_play_inj = [abs(int(i)) for i in last_play_inj]

injury['last_play'] = last_play_inj

last_play_inj = np.array([i for i in last_play_inj if (i > 0) ]) 
last_play = playlist.groupby('GameID').tail(1).PlayerGamePlay
fig = plt.figure(figsize=(20,8))

ax = plt.subplot(121)

ax.hist(injury[injury.Surface == 'Synthetic'].last_play,density=True,alpha = .5)

ax.axvline(np.median(injury[injury.Surface == 'Synthetic'].last_play), color='blue', linestyle='dashed', linewidth=2)

ax.axvline(np.median(injury[injury.Surface == 'Natural'].last_play), color='orange', linestyle='dashed', linewidth=2)

ax.hist(injury[injury.Surface == 'Natural'].last_play,density=True,alpha = .5)

ax.set_title(r'last play Synthetic v Natural', fontsize=18)

ax.set_xlabel(r'last play', fontsize=16)

ax.set_ylabel('PDF', fontsize=16)

ax.legend(('Synthetic Median','Natural Median','Synthetic','Natural'), fontsize=14)



ax2 = plt.subplot(122)

ax2.axvline(np.median(last_play), color='blue', linestyle='dashed', linewidth=2)

ax2.axvline(last_play.mean(), color='lightblue', linestyle='dashed', linewidth=2)

ax2.hist(last_play, density=True,alpha = 0.5)

ax2.axvline(np.median(last_play_inj), color='orange', linestyle='dashed', linewidth=2)

ax2.axvline(last_play_inj.mean(), color='yellow', linestyle='dashed', linewidth=2)

ax2.hist(last_play_inj, density=True,alpha = 0.5)

ax2.set_title(r'last play for all vs inj', fontsize=18)

ax2.set_xlabel(r'last play', fontsize=16)

ax2.set_ylabel('PDF', fontsize=16)

ax2.legend(('all median','all mean','injured median','injured mean','all','injured'), fontsize=14)



print("Synthetic median last play = " +str(np.median(injury[injury.Surface == 'Synthetic'].last_play)))

print("Natural median last play = " +str(np.median(injury[injury.Surface == 'Natural'].last_play)))

test = perm_test(last_play,last_play_inj,10000)
test = perm_test(injury[injury.Surface == 'Synthetic'].last_play,injury[injury.Surface == 'Natural'].last_play,10000)
# All Games PlayType by Natural v Synthetic Stacked Bar Chart

df = playlist[["FieldType","PlayType"]]

syn = df[df["FieldType"] == 'Synthetic'].groupby("PlayType").count()

#syn = syn["PlayerKey"].append(pd.Series([0], index=['snow']))

nat = df[df["FieldType"] == 'Natural'].groupby("PlayType").count()

#nat = nat["PlayerKey"].append(pd.Series([0], index=['snow'])).append(pd.Series([0], index=['indoor']))

stackplot_df = pd.DataFrame(nat)

stackplot_df['synthetic'] = syn

stackplot_df.columns = ['natural','synthetic']

#stackplot_df=stackplot_df.loc[['clear', 'cloudy', 'indoor', 'rain', 'sunny', 'unk', 'snow']]

stackplot_df
# Injured Games PlayType by Natural v Synthetic Stacked Bar Chart

df = injury[["FieldType","PlayType"]]

syn = df[df["FieldType"] == 'Synthetic'].groupby("PlayType").count()

#syn = syn["PlayerKey"].append(pd.Series([0], index=['snow']))

nat = df[df["FieldType"] == 'Natural'].groupby("PlayType").count()

#nat = nat["PlayerKey"].append(pd.Series([0], index=['snow'])).append(pd.Series([0], index=['indoor']))

stackplot_df = pd.DataFrame(nat)

stackplot_df['synthetic'] = syn

stackplot_df.columns = ['natural','synthetic']

#stackplot_df=stackplot_df.loc[['clear', 'cloudy', 'indoor', 'rain', 'sunny', 'unk', 'snow']]

stackplot_df
#plot percentage stacked bars

f, ax = plt.subplots(1, figsize=(15,8))

bar_width = 1

bar_l = [i for i in range(len(stackplot_df['natural']))] 

tick_pos = [i+(bar_width/2) for i in bar_l] 

totals = [i+j for i,j in zip(stackplot_df['natural'], stackplot_df['synthetic'])]

natural_rel = [i / j * 100 if j > 0 else 0 for  i,j in zip(stackplot_df['natural'], totals)]

synthetic_rel = [i / j * 100 if j > 0 else 0 for  i,j in zip(stackplot_df['synthetic'], totals)]

ax.bar(bar_l, natural_rel, label='Natural', alpha=0.8, color='blue',width=bar_width, edgecolor='white')

ax.bar(bar_l, synthetic_rel, bottom=natural_rel,label='Synthetic', alpha=0.8,  color='red', width=bar_width, edgecolor='white')

plt.xticks(tick_pos, stackplot_df.index.values)

ax.set_ylabel("Percentage")

ax.set_xlabel("PlayType")

plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])

plt.ylim(-10, 110)

plt.setp(plt.gca().get_xticklabels(),rotation = 45, horizontalalignment='right')

plt.legend(('Natural', 'Synthetic'))

plt.title('PlayType by Natural v Syntehtic for Injuries')

plt.show()
fig = plt.figure(figsize=(20,10))

ax = plt.subplot(121)

ax.imshow(plt.imread("../input/document-images/playtypeBYfieldtype.png"))

ax.axis('off')

ax2 = plt.subplot(122)

ax2.imshow(plt.imread("../input/document-images/playtypeBYfieldtype_injuries.png"))

ax2.axis('off')
# Injured Games PlayType by Natural v Synthetic Stacked Bar Chart

df = playlist

syn = df[df["FieldType"] == 'Synthetic'].groupby("Weather").count()

syn = syn["PlayerKey"]

nat = df[df["FieldType"] == 'Natural'].groupby("Weather").count()

nat = nat["PlayerKey"]

stackplot_df = pd.DataFrame(nat)

stackplot_df['synthetic'] = syn

stackplot_df.columns = ['natural','synthetic']

stackplot_df=stackplot_df.loc[['clear', 'cloudy', 'indoor', 'rain', 'sunny', 'unk', 'snow']]

stackplot_df
## Injured Games Percentage Stacked Bar Plot 

df = injury

syn = df[df["FieldType"] == 'Synthetic'].groupby("Weather").count()

syn = syn["PlayerKey"].append(pd.Series([0], index=['snow']))

nat = df[df["FieldType"] == 'Natural'].groupby("Weather").count()

nat = nat["PlayerKey"].append(pd.Series([0], index=['snow'])) #.append(pd.Series([0], index=['indoor']))

stackplot_df = pd.DataFrame(nat)

stackplot_df['synthetic'] = syn

stackplot_df.columns = ['natural','synthetic']

stackplot_df=stackplot_df.loc[['clear', 'cloudy', 'indoor', 'rain', 'sunny', 'unk', 'snow']]

stackplot_df
f, ax = plt.subplots(1, figsize=(15,8))

bar_width = 1

bar_l = [i for i in range(len(stackplot_df['natural']))] 

tick_pos = [i+(bar_width/2) for i in bar_l] 

totals = [i+j for i,j in zip(stackplot_df['natural'], stackplot_df['synthetic'])]

natural_rel = [i / j * 100 if j > 0 else 0 for  i,j in zip(stackplot_df['natural'], totals)]

synthetic_rel = [i / j * 100 if j > 0 else 0 for  i,j in zip(stackplot_df['synthetic'], totals)]

ax.bar(bar_l, natural_rel, label='Natural', alpha=0.8, color='blue',width=bar_width, edgecolor='white')

ax.bar(bar_l, synthetic_rel, bottom=natural_rel,label='Synthetic', alpha=0.8,  color='red', width=bar_width, edgecolor='white')

plt.xticks(tick_pos, stackplot_df.index.values)

ax.set_ylabel("Percentage")

ax.set_xlabel("Weather")

plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])

plt.ylim(-10, 110)

plt.setp(plt.gca().get_xticklabels(), horizontalalignment='right')

plt.legend(('Natural', 'Synthetic'))

plt.title('Weather by Natural v Syntehtic for Injured Games')

plt.show()
fig = plt.figure(figsize=(20,10))

ax = plt.subplot(121)

ax.imshow(plt.imread("../input/document-images/Weather by Natural v Synthetic for all.png"))

ax.axis('off')

ax2 = plt.subplot(122)

ax2.imshow(plt.imread("../input/document-images/Weather by Natural v Synthetic for inj.png"))

ax2.axis('off')
part_ct = injury[["BodyPart","DM_Sum"]].groupby("BodyPart").count()

part_ct["BodyPart"]=part_ct.index.values

part_ct.index = [1,2,3,4,5]

part_mn = injury[["BodyPart","DM_Sum"]].groupby("BodyPart").mean()

part_mn["BodyPart"]=part_mn.index.values

part_mn.index = [1,2,3,4,5]

temp = injury[["BodyPart_num","PlayerKey"]].groupby(["BodyPart_num"]).count()

temp["BodyPart"] = [i[:-1] for i in temp.index.values]

temp
temp1 = pd.merge(part_mn, part_ct, 'left', 'BodyPart')

temp1.columns = ["Mean","BodyPart","Count"]

temp1['Mode'] = [0,6,1,1,1]

temp1
inj_by_part = pd.merge(temp,temp1,'left','BodyPart')

inj_by_part['BP_DM_Sum'] = temp.index.values

inj_by_part

inj_by_part.columns = ['Count', 'BodyPart', 'Mean', 'BP_TTL','Mode', 'BP_DM_Sum']

inj_by_part[[ 'BP_DM_Sum','BodyPart', 'Mean','Mode','BP_TTL','Count'  ]]
ax = sns.lmplot( x="BodyPart_cat", y="DM_Sum_adj", data=injury, fit_reg=False, hue='FieldType', legend=False,palette="Set1")

ax.set(xlabel='Body Part', ylabel='Injury Severity')

plt.legend(loc='upper center')

plt.xticks([1,2,3,4,5], np.sort(np.array(injury["BodyPart"].drop_duplicates())))

plt.title('Injury Severity by Body Part and Field Type')
Image("../input/document-images/severity_likelihood.png", width='500')
num = injury[["Weather","FieldType"]].groupby("FieldType").count()

num
pl = playlist[["PlayerKey","GameID","FieldType"]] 



syn_d = len(pl[pl["FieldType"] == 'Synthetic'].drop_duplicates())

nat_d = len(pl[pl["FieldType"] == 'Natural'].drop_duplicates())

nat_num = num["Weather"].Natural

syn_num = num["Weather"].Synthetic



#print("Natrual Injury Base Rate = " + str(round(nat_num/nat_d*100,2)) +"%")

#print("Synthetic Injury Base Rate = " + str(round(syn_num/syn_d*100,2)) +"%")
pl.drop_duplicates().groupby("FieldType").count()
std_field = playlist[["PlayerKey","StadiumType","GameID","Weather","FieldType"]].drop_duplicates()

std_field.groupby(["StadiumType","FieldType"]).count()[["PlayerKey"]]
indoor = injury[injury["StadiumType"] == 'indoor'].groupby("Weather").count()

indoor = indoor["PlayerKey"].append(pd.Series([0], index=['rain']))

outdoor = injury[injury["StadiumType"] == 'outdoor'].groupby("Weather").count()

outdoor = outdoor["PlayerKey"].append(pd.Series([0], index=['indoor']))

outdoor = outdoor[indoor.index.values]

N = len(indoor)

ind = indoor.index.values    # the x locations for the groups

width = 0.35       # the width of the bars: can also be len(x) sequence



p1 = plt.bar(x=ind, height = indoor, width = width)

p2 = plt.bar(x=ind, height = outdoor, width = width)

plt.ylabel('Count')

plt.title('Counts by StadiumType and Weather')

plt.legend((p1[0], p2[0]), ('indoor', 'outdoor'))



plt.show()
num = injury[["Weather","StadiumType"]].groupby("StadiumType").count()

num
in_d = len(std_field[std_field["StadiumType"] == 'indoor'].drop_duplicates())

out_d = len(std_field[std_field["StadiumType"] == 'outdoor'].drop_duplicates())

in_num = num["Weather"].indoor

out_num = num["Weather"].outdoor



#print("Indoor Injury Base Rate = " + str(round(in_num/in_d*100,2)) +"%")

#print("Outdoor Injury Base Rate = " + str(round(out_num/out_d*100,2)) +"%")
std_field.drop_duplicates().groupby("StadiumType").count()[["PlayerKey"]]
inj_player = injury['PlayKey']

colors = ["red" if ft=="Synthetic" else "blue" for ft in injury['FieldType']]

init_event = ["ball_snap", "kickoff","kickoff_play", "onside_kick","free_kick","free_kick_play", "snap_direct"]

temp = playertrack[playertrack["PlayKey"] == inj_player[0]]

crnt_events = temp.event.drop_duplicates().fillna("")

strt_event = crnt_events[[i in init_event for i in crnt_events]].values[0]

start = temp[temp["event"] == strt_event]

temp = temp.loc[start.index.values[0]: , ]

temp = get_diff_os(temp,'velocity')

diff_os_example = temp
fig = plt.figure(figsize=(30,20))

ax = plt.subplot(221)

ax.hist((temp.s-temp.velocity), density=True)

ax.set_xlabel('diff', fontsize = 16)

ax.set_ylabel('PDF', fontsize = 16)

ax.set_title('Difference between Recorded Speed(S) and Calculated Velocity', fontsize = 20)

ax2 = plt.subplot(222)

ax2.plot(temp.time,temp.s,label='Speed')

ax2.plot(temp.time,temp.velocity,label='Velocity')

ax2.set_xlabel('time', fontsize = 16)

ax2.set_ylabel('value', fontsize = 16)

ax2.set_title('Velocity v Speed', fontsize = 20)

ax2.legend(loc="upper right", fontsize = 20)

ax3 = plt.subplot(212)

ax3.plot(temp.time,get_diff_os(temp,'s').diff_os,label='diff_os w/ Speed')

ax3.plot(temp.time,get_diff_os(temp,'velocity').diff_os,label='diff_os w/ Velocity')

ax3.set_xlabel('time', fontsize = 16)

ax3.set_ylabel('value', fontsize = 16)

ax3.set_title('diff_os for Velocity v Speed', fontsize = 20)

ax3.legend(loc="upper right", fontsize = 20)
### Create GIF of player movements

plt.cla

fig = plt.figure(figsize=(30,10))

ax2 = fig.add_subplot(111)



ax2.imshow(img, zorder=0, extent=[0.1, 120, 0, 53.3])

line, = ax2.plot([], [], color=colors[0], alpha = 0.6, lw=2)



def init():

    line.set_data([], [])

    return line,



def animate(t):

    line.set_data(np.array(temp.x)[:t], np.array(temp.y)[:t])

    return line,



ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(temp.x), interval=1, blit=True)

#ani.save('inj_plays.gif', writer='pillow')
### Create GIF of player diff_os

plt.cla

fig = plt.figure(figsize=(30,10))

ax = fig.add_subplot(121)

ax.set_ylim([-50, 50])

ax.set_xlim([0, len(temp.time)])

ax.set(xlabel='time', ylabel='diff_os')



line = ax.plot([], [], color=colors[0], alpha = 0.2)[0]



def init():

    line.set_data([], [])

    return line,



def animate(t):

    line.set_data(np.arange(0,len(temp.time))[:t], temp.diff_os[:t])

    return line,



ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(temp.time), interval=1, blit=True)

#ani.save('inj_plays2.gif', writer='pillow')
plt.cla

fig = plt.figure(figsize=(30,10))

ax = fig.add_subplot(121)

ax.set_title('Injury Paths on Natural')

ax2 = fig.add_subplot(122)

ax2.set_title('Injury Paths on Synthetic')

ax.imshow(img,zorder=0,  extent=[0.1, 120, 0, 53.3])

ax.set_ylim(-10,60)

ax2.imshow(img,zorder=0,  extent=[0.1, 120, 0, 53.3])

ax2.set_ylim(-10,60)



for i in np.arange(0,len(inj_player)):

    temp = playertrack[playertrack["PlayKey"] == inj_player[i]]

    crnt_events = temp.event.drop_duplicates().fillna("")

    strt_event = crnt_events[[i in init_event for i in crnt_events]].values[0]

    start = temp[temp["event"] == strt_event]

    if len(temp[temp["event"] == strt_event]) != 1:

      continue

    temp = temp.loc[start.index.values[0]: , ["x","y"]]

    if injury.FieldType[i] == 'Natural':

        ax.plot(temp.x,temp.y, color='blue', alpha = 0.6)

    else:

        ax2.plot(temp.x,temp.y, color='red', alpha = 0.6)

plt.show()
Image("../input/document-images/nfl_injury_paths.png")
custom_bucket_x = np.linspace(0, 120, 121)

custom_bucket_y = np.linspace(-10, 60, 71)

playertrack2 = playertrack

playertrack2['x_buk'] = pd.cut(playertrack['x'], custom_bucket_x)

playertrack2['y_buk'] = pd.cut(playertrack['y'], custom_bucket_y)

heat_df = playertrack2[['x_buk','x','y_buk','y']].groupby(['y_buk','x_buk']).count()

x_axis_labels = range(0,120)

y_axis_labels = range(-10,60)



plt.figure(figsize=(25,10))

ax = sns.heatmap(np.split(np.array(heat_df.x.values), 70),cmap="YlGnBu",xticklabels=x_axis_labels, yticklabels=y_axis_labels,alpha = 0.5)

ax.imshow(img,zorder=0,  extent=[0.0,120.0, 63.0, 10.0],

          aspect = ax.get_aspect())

          #extent = ax.get_xlim() + ax.get_ylim())

ax.set_title('Field Traveled')
playertrack3 = pd.merge(injury[['PlayKey']].drop_duplicates(),playertrack2[['PlayKey', 'x', 'y', 'x_buk','y_buk']].drop_duplicates(),'left',['PlayKey'])
heat_df = playertrack3[['x_buk','x','y_buk','y']].groupby(['y_buk','x_buk']).count().fillna(0)

plt.figure(figsize=(25,10))

ax = sns.heatmap(np.split(np.sqrt(np.array(heat_df.x.values)), 70),cmap=sns.light_palette("red"),xticklabels=x_axis_labels, yticklabels=y_axis_labels,alpha = 0.3)

ax.imshow(img,zorder=0,  extent=[0.0,120.0, 63.0, 10.0],

          aspect = ax.get_aspect())

ax.set_title('Field Traveled Injuries')
Image("../input/document-images/my_met_def.png", width = '500')
plt.cla

fig = plt.figure(figsize=(30,8))

ax = fig.add_subplot(121)

ax.set_ylim([-50, 50])

ax.set_xlim([0, len(diff_os_example.time)])

ax.set(xlabel='time', ylabel='diff_os')

ax.plot(np.arange(0,len(diff_os_example.time)), diff_os_example.diff_os, color=colors[0], alpha = 0.2)

ax.text(ax.get_xlim()[1]/2,ax.get_ylim()[1]-10,'Left Turn')

ax.text(ax.get_xlim()[1]/2,ax.get_ylim()[0]+10,'Right Turn')

ax.text(ax.get_xlim()[1]-40,diff_os_limits[1]+1,'Upper OS Risk Boundary')

ax.text(ax.get_xlim()[1]-40,diff_os_limits[0]-3,'Lower OS Lower Boundary')

plt.axhline(diff_os_limits[1], color='black', linestyle='dashed', linewidth=1)

plt.axhline(diff_os_limits[0], color='black', linestyle='dashed', linewidth=1)

plt.title('diff_os plot example', Fontsize = 18)
plt.cla

plt.figure(figsize=(20,8))

for i in np.arange(0,len(players_df3)):

    if players_df3.injured[i] == 1 :

        plt.plot(np.array(players_df3.diff_os[i]), color='red',alpha=.2,label='Injured')

    else:

         plt.plot(np.array(players_df3.diff_os[i]), color='blue',alpha=.2,label='Not Injured')

plt.legend(['Injured','Non Injured'], fontsize=16)

plt.title('diff_os injured v non-injured', fontsize=36) 

plt.xlabel('time', fontsize=16)

plt.ylabel('diff_os', fontsize=16)

plt.xlim([0, 400])

plt.ylim([-400, 400])

plt.axhline(diff_os_limits[1], color='black', linestyle='dashed', linewidth=1)

plt.axhline(diff_os_limits[0], color='black', linestyle='dashed', linewidth=1)
my_met_syn =  np.array(players_df3.loc[(players_df3['FieldType'] == 'Synthetic') & (players_df3['injured'] == 1.0), 'my_met'])

os_met_syn =  np.array(players_df3.loc[(players_df3['FieldType'] == 'Synthetic') & (players_df3['injured'] == 1.0), 'os_met'])

my_met_nat =  np.array(players_df3.loc[(players_df3['FieldType'] == 'Natural') & (players_df3['injured'] == 1.0), 'my_met'])

os_met_nat =  np.array(players_df3.loc[(players_df3['FieldType'] == 'Natural') & (players_df3['injured'] == 1.0), 'os_met'])
fig = plt.figure(figsize=(30,10))

ax = plt.subplot(121)

ax.hist(os_met_syn,alpha=.3,density=True)

ax.axvline(np.mean(os_met_syn), color='blue', linestyle='dashed', linewidth=2)

ax.hist(os_met_nat,alpha=.3,density=True)

ax.axvline(np.mean(os_met_nat), color='orange', linestyle='dashed', linewidth=2)

ax.set_xlabel(r'value',fontsize = 16)

ax.set_ylabel('PDF',fontsize = 16)

ax.set_title('os_met Synthetic v Natural',fontsize = 18)

ax.legend(['Synthetic Mean', 'Natural Mean','Synthetic', 'Natural'],fontsize = 16)



ax2 = plt.subplot(122)

ax2.hist(my_met_syn,alpha=.3,density=True)

ax2.axvline(np.mean(my_met_syn), color='blue', linestyle='dashed', linewidth=2)

ax2.hist(my_met_nat,alpha=.3,density=True)

ax2.axvline(np.mean(my_met_nat), color='orange', linestyle='dashed', linewidth=2)

ax2.set_xlabel(r'value',fontsize = 16)

ax2.set_ylabel('PDF',fontsize = 16)

ax2.set_title('my_met Synthetic v Natural',fontsize = 18)

ax2.legend(['Synthetic Mean', 'Natural Mean','Synthetic', 'Natural'],fontsize = 16)
print('os_met_nat: mean= %.3f stdv= %.3f size= %.3f' % (np.mean(os_met_nat), np.std(os_met_nat), len(os_met_nat)))

print('os_met_syn: mean= %.3f stdv= %.3f size= %.3f' % (np.mean(os_met_syn), np.std(os_met_syn), len(os_met_syn)))

print('my_met_nat: mean= %.3f stdv= %.3f size= %.3f' % (np.mean(my_met_nat), np.std(my_met_nat), len(my_met_nat)))

print('my_met_syn: mean= %.3f stdv= %.3f size= %.3f' % (np.mean(my_met_syn), np.std(my_met_syn), len(my_met_syn)))
# compare samples

stat, p = ttest_ind(os_met_nat,os_met_syn)

print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret

alpha = 0.05

if p > alpha:

    print('os_met Same distributions (fail to reject H0)')

else:

    print('os_met Different distributions (reject H0)')



stat, p = ttest_ind(my_met_nat,my_met_syn)

print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret

alpha = 0.05

if p > alpha:

    print('my_met Same distributions (fail to reject H0)')

else:

    print('my_met Different distributions (reject H0)')
perm_test(os_met_nat,os_met_syn, 10000)
perm_test(my_met_nat,my_met_syn, 10000)
boot_os_met = bootTest(os_met_nat,os_met_syn)

boot_my_met = bootTest(my_met_nat,my_met_syn)
fig = plt.figure(figsize=(30,10))

ax = plt.subplot(121)

ax.axvline(boot_os_met[1], color='red', linestyle='dashed', linewidth=1)

ax.axvline(boot_os_met[2][0], color='black', linestyle='dashed', linewidth=1)

ax.axvline(boot_os_met[2][1], color='black', linestyle='dashed', linewidth=1)

ax.hist(boot_os_met[0], bins=50, density=True)

ax.text(boot_os_met[2][0]-2,ax.get_ylim()[1]*.5,'2.5%',fontsize=16)

ax.text(boot_os_met[2][1]+2,ax.get_ylim()[1]*.5,'2.5%',fontsize=16)

ax.text(boot_os_met[1]+0.2,ax.get_ylim()[1]*.95,'Empirical Diff: p-value= '+str(boot_os_met[3]),fontsize=12)

ax.set_title('Bootstrap Confidence Intervals and Empirical Difference of Means os_met',fontsize=18)

ax.set_xlabel(r'os_met diff of means',fontsize=16)

ax.set_ylabel('PDF',fontsize=16)



ax2 = plt.subplot(122)

ax2.axvline(boot_my_met[1], color='red', linestyle='dashed', linewidth=1)

ax2.axvline(boot_my_met[2][0], color='black', linestyle='dashed', linewidth=1)

ax2.axvline(boot_my_met[2][1], color='black', linestyle='dashed', linewidth=1)

ax2.hist(boot_my_met[0], bins=50, density=True)

ax2.text(boot_my_met[2][0]-.1,ax2.get_ylim()[1]*.5,'2.5%',fontsize=16)

ax2.text(boot_my_met[2][1]+.1,ax2.get_ylim()[1]*.5,'2.5%',fontsize=16)

ax2.text(boot_my_met[1]+0.002,ax2.get_ylim()[1]*.95,'Empirical Diff: p-value= '+str(boot_my_met[3]),fontsize=12)

ax2.set_title('Bootstrap Confidence Intervals and Empirical Difference of Means my_met',fontsize=18)

ax2.set_xlabel(r'my_met diff of means',fontsize=16)

ax2.set_ylabel('PDF',fontsize=16)
os_met_list_non_inj = np.array(players_df3.loc[(players_df3['injured'] == 0.0), 'os_met'])

os_met_list_inj = np.array(players_df3.loc[(players_df3['injured'] == 1.0), 'os_met'])

my_met_list_non_inj = np.array(players_df3.loc[(players_df3['injured'] == 0.0), 'my_met'])

my_met_list_inj = np.array(players_df3.loc[(players_df3['injured'] == 1.0), 'my_met'])
fig = plt.figure(figsize=(30,10))

ax = plt.subplot(121)

ax.hist(os_met_list_non_inj,alpha=.3,density=True)

ax.axvline(np.mean(os_met_list_non_inj), color='blue', linestyle='dashed', linewidth=2)

ax.hist(os_met_list_inj,alpha=.3,density=True)

ax.axvline(np.mean(os_met_list_inj), color='orange', linestyle='dashed', linewidth=2)

ax.set_xlabel(r'value',fontsize = 16)

ax.set_ylabel('PDF',fontsize = 16)

ax.set_title('os_met Injured v Not Injured',fontsize = 18)

ax.legend(['Not Injured Mean','Injured Mean','Not Injured','Injured'],fontsize = 16)



ax2 = plt.subplot(122)

ax2.hist(my_met_list_non_inj,alpha=.3)

ax2.axvline(np.mean(my_met_list_non_inj), color='blue', linestyle='dashed', linewidth=2)

ax2.hist(my_met_list_inj,alpha=.3)

ax2.axvline(np.mean(my_met_list_inj), color='orange', linestyle='dashed', linewidth=2)

ax2.set_xlabel(r'value',fontsize = 16)

ax2.set_ylabel('PDF',fontsize = 16)

ax2.set_title('my_met Injured v Not Injured',fontsize = 18)

ax2.legend(['Not Injured Mean','Injured Mean','Not Injured','Injured'],fontsize = 16)
stat, p = ttest_ind(os_met_list_inj,os_met_list_non_inj)

print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret

alpha = 0.05

if p > alpha:

    print('os_met Same distributions (fail to reject H0)')

else:

    print('os_met Different distributions (reject H0)')

    

stat, p = ttest_ind(my_met_list_inj,my_met_list_non_inj)

print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret

alpha = 0.05

if p > alpha:

    print('my_met Same distributions (fail to reject H0)')

else:

    print('my_met Different distributions (reject H0)')
os_met_inj_perm_results = perm_test(os_met_list_inj,os_met_list_non_inj, 10000)
my_met_inj_perm_results = perm_test(my_met_list_inj,my_met_list_non_inj, 10000)
boot_os_met = bootTest(os_met_list_inj,os_met_list_non_inj)

boot_my_met = bootTest(my_met_list_inj,my_met_list_non_inj)
fig = plt.figure(figsize=(30,10))

ax = plt.subplot(121)

ax.axvline(boot_os_met[1], color='red', linestyle='dashed', linewidth=1)

ax.axvline(boot_os_met[2][0], color='black', linestyle='dashed', linewidth=1)

ax.axvline(boot_os_met[2][1], color='black', linestyle='dashed', linewidth=1)

ax.hist(boot_os_met[0], bins=50, density=True)

ax.text(boot_os_met[2][0]-2,ax.get_ylim()[1]*.5,'2.5%',fontsize=16)

ax.text(boot_os_met[2][1]+2,ax.get_ylim()[1]*.5,'2.5%',fontsize=16)

ax.text(boot_os_met[1]+0.2,ax.get_ylim()[1]*.95,'Empirical Diff: p-value= '+str(boot_os_met[3]),fontsize=12)

ax.set_title('Bootstrap Confidence Intervals and Empirical Difference of Means os_met',fontsize=18)

ax.set_xlabel(r'os_met diff of means',fontsize=16)

ax.set_ylabel('PDF',fontsize=16)



ax2 = plt.subplot(122)

ax2.axvline(boot_my_met[1], color='red', linestyle='dashed', linewidth=1)

ax2.axvline(boot_my_met[2][0], color='black', linestyle='dashed', linewidth=1)

ax2.axvline(boot_my_met[2][1], color='black', linestyle='dashed', linewidth=1)

ax2.hist(boot_my_met[0], bins=50, density=True)

ax2.text(boot_my_met[2][0]-.1,ax2.get_ylim()[1]*.5,'2.5%',fontsize=16)

ax2.text(boot_my_met[2][1]+.1,ax2.get_ylim()[1]*.5,'2.5%',fontsize=16)

ax2.text(boot_my_met[1]+0.002,ax2.get_ylim()[1]*.95,'Empirical Diff: p-value= '+str(boot_my_met[3]),fontsize=12)

ax2.set_title('Bootstrap Confidence Intervals and Empirical Difference of Means my_met',fontsize=18)

ax2.set_xlabel(r'my_met diff of means',fontsize=16)

ax2.set_ylabel('PDF',fontsize=16)
# Data preprocessing, convert categorical variabels to OHE, scale continuous variables 

players_df = players_df3

cols = ['PlayKey','GameID', 'injured','StadiumType', 'FieldType', 'Temperature', 'Weather','PlayType', 'Position', 'PositionGroup', 'my_met', 'os_met']



df = pd.concat([players_df[cols] ,pd.get_dummies(players_df['Weather'], prefix='Weather',drop_first=True)],axis=1).drop(['Weather'],axis=1)

for col in ['StadiumType', 'FieldType', 'PlayType', 'Position', 'PositionGroup']:

    df = pd.concat([df,pd.get_dummies(df[col], prefix=col,drop_first=True)],axis=1).drop([col],axis=1)

    

features = ['Temperature', 'my_met', 'os_met']

# Separating out the continuous features

df_scale= df.loc[:, features].values

# Standardizing the features

df_scale = StandardScaler().fit_transform(df_scale)

df_scale = pd.DataFrame(df_scale)

df_scale.columns = np.array(['Temperature', 'my_met', 'os_met'])



df_scale = pd.concat([df.drop(['Temperature', 'my_met', 'os_met'],axis=1), df_scale], axis=1)



df_scale_bs = df_scale.sample(10000,replace=True) #boostrap sample for stable interpretatino of feature importance



x_cols = ['Weather_cloudy', 'StadiumType_outdoor', 'FieldType_Synthetic', 'PlayType_Field Goal','PlayType_Kickoff','PlayType_Punt',

          'PlayType_Punt Not Returned', 'PlayType_Punt Returned', 'PlayType_Rush','Position_DT','Position_OLB','Position_RB', 'Position_SS', 

          'Position_T', 'Position_TE', 'PositionGroup_DL', 'PositionGroup_LB','PositionGroup_OL', 'PositionGroup_TE', 'Temperature', 'my_met',

          'os_met'] #select only relevent variables
corr = df_scale_bs[x_cols].corr()

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(10,10))

ax = sns.heatmap( corr,mask=mask, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True, cbar_kws={"shrink": 0.8} )

ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')

ax.set_title('Coorelation Heatmap for select Variables')
X=df_scale_bs[x_cols]

y=df_scale_bs['injured']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



model = LogisticRegression(penalty="l2", solver = 'liblinear')

model.fit(X_train, y_train)



y_pred = model.predict(X_test)

LR_conf_mat = confusion_matrix(y_test, y_pred)

print(LR_conf_mat)

print('Accuracy of LR logistic regression classifier on test set: ' + str(round((LR_conf_mat[1][1] +LR_conf_mat[0][0]) / LR_conf_mat.sum(),4) ))
explainer = shap.LinearExplainer(model, X_train, feature_dependence = "independent")

shap_values = explainer.shap_values(X_train)

shap.summary_plot(shap_values, X_train, plot_type="bar")
shap.summary_plot(shap_values, X_train)
logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))

fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
# Get the predictions and put them with the test data.

X_output = X_test.copy()

#X_output.loc[:,'predict'] = np.round(logreg.predict(X_output),2)



# Randomly pick some observations

S = X_output.sample(6)

S



# Initialize your Jupyter notebook with initjs(), otherwise you will get an error message.

shap.initjs()



# Write in a function

def shap_plot(j):

    explainerModel = shap.LinearExplainer(model,X_train)

    shap_values_Model = explainerModel.shap_values(S)

    p = shap.force_plot(explainerModel.expected_value, shap_values_Model[j], S.iloc[[j]])

    return(p)
shap_plot(0)
shap_plot(3)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
xgb_clf = XGBClassifier()

eval_set = [(X_val, y_val)]

xgb_clf.fit(X_train, y_train, eval_metric="auc", eval_set=eval_set, verbose=False)

#xgb_clf.fit(X_train, y_train)

score = xgb_clf.score(X_val, y_val)

print('Accuracy of XGBoost classifier on test set: ' + str(round(score,2) ))
shap_values = shap.TreeExplainer(xgb_clf).shap_values(X_train)

shap.summary_plot(shap_values, X_train, plot_type="bar")
shap.summary_plot(shap_values, X_train)
shap.dependence_plot('Temperature', shap_values, X_train)

shap.dependence_plot('my_met', shap_values, X_train)

shap.dependence_plot('os_met', shap_values, X_train)
# Get the predictions and put them with the test data.

X_output = X_test.copy()

#X_output.loc[:,'predict'] = np.round(xgb_clf.predict(X_output),2)



# Randomly pick some observations

S = X_output.sample(6)



# Initialize your Jupyter notebook with initjs(), otherwise you will get an error message.

shap.initjs()



# Write in a function

def shap_plot(j):

    explainerModel = shap.TreeExplainer(xgb_clf)

    shap_values_Model = explainerModel.shap_values(S)

    p = shap.force_plot(explainerModel.expected_value, shap_values_Model[j], S.iloc[[j]])

    return(p)
shap_plot(0)
shap_plot(1)