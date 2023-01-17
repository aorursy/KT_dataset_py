# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_pop = pd.read_csv('/kaggle/input/cpd-police-beat-demographics/beatpop.txt', sep=" ", skiprows = [0], header=None)

df_pop.columns = ["beat", "population", "square_mileage"]

df_pop.set_index('beat', inplace= True)
racef = open("/kaggle/input/cpd-police-beat-demographics/beatrace.txt",'r')

size = int(racef.readline())

beat_race = {}

for i in range (0, size):

    beat = int(racef.readline())

    info = racef.readline().split(' ')

    info[5] = info[5][:-2]

    for j in range(0,6):

        info[j] = float(info[j])

    beat_race[beat] = info

#print(beat_race)

df_race = pd.DataFrame.from_dict(beat_race, orient='index')

df_race.rename(columns={0: "num_white", 1: "num_hispanic", 2: "num_black", 3: "num_asian", 4: "num_mixed", 5: "num_other"}, inplace = True)
df_beat = pd.concat([df_pop, df_race], axis = 1)
df_beat['percent_white'] = df_beat.apply(lambda row: row['num_white']/row['population']*100, axis = 1)

df_beat['percent_hispanic'] = df_beat.apply(lambda row: row['num_hispanic']/row['population']*100, axis = 1)

df_beat['percent_black'] = df_beat.apply(lambda row: row['num_black']/row['population']*100, axis = 1)

df_beat['percent_asian'] = df_beat.apply(lambda row: row['num_asian']/row['population']*100, axis = 1)

df_beat['percent_mixed'] = df_beat.apply(lambda row: row['num_mixed']/row['population']*100, axis = 1)

df_beat['percent_other'] = df_beat.apply(lambda row: row['num_other']/row['population']*100, axis = 1)

df_beat.head()
df_hi = pd.read_csv('/kaggle/input/cpd-police-beat-demographics/beathi.txt', sep=" ", skiprows = [0], header=None)

df_hi.columns = ["beat", "med_income"]

df_hi.set_index('beat', inplace= True)

df_beat = pd.concat([df_beat, df_hi], axis = 1)

df_beat.head()
df_fs = pd.read_csv('/kaggle/input/cpd-police-beat-demographics/beatfs.txt', sep=" ", skiprows = [0], header=None)

df_fs.drop(2, axis = 1, inplace = True)

df_fs.columns = ["beat", "pop_food_stamps"]

df_fs.set_index('beat', inplace= True)

#df_fs.head()

df_beat = pd.concat([df_beat, df_fs], axis = 1)

df_beat.index.name = 'beat'

#df_beat.head()
df_beat['percent_on_fs'] = df_beat.apply(lambda row: row['pop_food_stamps']/row['population']*100, axis = 1)

df_beat.head()
df_ea = pd.read_csv('/kaggle/input/cpd-police-beat-demographics/beatea.txt', sep=" ", skiprows = [0], header=None)

df_ea.columns = ["beat", "bachelors", "high_school", "no_high_school"]

df_ea.set_index('beat', inplace= True)

df_beat = pd.concat([df_beat, df_ea], axis = 1)

df_beat.index.name = 'beat'

#df_beat.head()
df_beat['percent_bachelors'] = df_beat.apply(lambda row: row['bachelors']/row['population']*100, axis = 1)

df_beat['percent_high_school'] = df_beat.apply(lambda row: row['high_school']/row['population']*100, axis = 1)

df_beat['percent_no_high_school'] = df_beat.apply(lambda row: row['no_high_school']/row['population']*100, axis = 1)
df_age = pd.read_csv('/kaggle/input/cpd-police-beat-demographics/beatage.txt', sep=" ", skiprows = [0], header=None)

df_age.columns = ["beat", '85+', '80-84', '75-79', '70-74', '67-69', '65-66', '62-64', '60-61', '55-59', '50-54', '45-49', '40-44', '35-39', '30-34', '25-29', '22-24', '21', '20', '18-19', '15-17', '10-14', '5-9', '0-4']

df_age.set_index('beat', inplace= True)

df_beat = pd.concat([df_beat, df_age], axis = 1)

df_beat.index.name = 'beat'

df_beat.head()
df_beat['<=21'] = df_beat.apply(lambda row: row['21'] + row['20']+row['18-19']+row['15-17']+row['10-14']+row['5-9']+row['0-4'], axis = 1)

df_beat['22-29'] = df_beat.apply(lambda row: row['22-24'] + row['25-29'], axis = 1)

df_beat['30-39'] = df_beat.apply(lambda row: row['30-34'] + row['35-39'], axis = 1)

df_beat['40-49'] = df_beat.apply(lambda row: row['40-44'] + row['45-49'], axis = 1)

df_beat['50-59'] = df_beat.apply(lambda row: row['50-54'] + row['55-59'], axis = 1)

df_beat['60-64'] = df_beat.apply(lambda row: row['60-61'] + row['62-64'], axis = 1)

df_beat['65+'] = df_beat.apply(lambda row: row['65-66'] + row['67-69']+row['70-74']+row['75-79']+row['80-84']+row['85+'], axis = 1)

df_beat.head()
df_ea = pd.read_csv('/kaggle/input/cpd-police-beat-demographics/beatse.txt', sep=" ", skiprows = [0], header=None)

df_ea.columns = ["beat", "se_35+", "se_25-34", "se_20-24", "se_18-19", "se_15-17", "se_10-14", "se_5-9", "se_0-4"]

df_ea.set_index('beat', inplace= True)

df_beat = pd.concat([df_beat, df_ea], axis = 1)

df_beat.index.name = 'beat'

#df_beat.head()
df_beat['total_se'] = df_beat.apply(lambda row: row["se_35+"]+ row["se_25-34"]+ row["se_20-24"]+ row["se_18-19"]+ row["se_15-17"]+ row["se_10-14"]+row["se_5-9"]+row["se_0-4"], axis = 1)
df_beat['percent_se'] = df_beat.apply(lambda row: row['total_se']/row['population']*100, axis = 1)

df_beat['percent_se_0-4'] = df_beat.apply(lambda row: row['se_0-4']/row['0-4']*100, axis = 1)

df_beat['percent_se_5-9'] = df_beat.apply(lambda row: row['se_5-9']/row['5-9']*100, axis = 1)

df_beat['percent_se_10-14'] = df_beat.apply(lambda row: row['se_10-14']/row['10-14']*100, axis = 1)

df_beat['percent_se_15-17'] = df_beat.apply(lambda row: row['se_15-17']/row['15-17']*100, axis = 1)

df_beat['percent_se_18-19'] = df_beat.apply(lambda row: row['se_18-19']/row['18-19']*100, axis = 1)

df_beat['percent_se_20-24'] = df_beat.apply(lambda row: row['se_20-24']/(row['20']+row['21']+row['22-24'])*100, axis = 1)

df_beat['percent_se_25-34'] = df_beat.apply(lambda row: row['se_25-34']/(row['25-29']+row['30-34'])*100, axis = 1)

df_beat['youngPop'] = df_beat.apply(lambda row: row["population"]- (row['25-29']+row['30-34'])- (row['20']+row['21']+row['22-24'])- row["18-19"]- row["15-17"]- row["10-14"]-row["5-9"]-row["0-4"], axis = 1)

df_beat['percent_se_35+'] = df_beat.apply(lambda row: row['se_35+']/(row['population']-row['youngPop'])*100, axis = 1)
df_beat.head()
df_beat.reset_index(inplace = True)
#helper func to properly format beats

def pad0(beat):

    if(len(str(beat)) == 3):

        return '0'+ str(beat)

    else:

        return str(beat)
df_beat['beat'] = df_beat['beat'].apply(lambda x: pad0(x))
df_beat.head()
totPop = df_beat['population'].sum()

wPop = df_beat['num_white'].sum()

bPop = df_beat['num_black'].sum()

hPop = df_beat['num_hispanic'].sum()

mPop = df_beat['num_mixed'].sum()

aPop = df_beat['num_asian'].sum()

oPop = df_beat['num_other'].sum()

print('CHICAGO POLICE BEAT RACE BREAKDOWN:\n')

print('Percentage of Population that is White: '+str(wPop/totPop*100)+'%')

print('Percentage of Population that is Black: '+str(bPop/totPop*100)+'%')

print('Percentage of Population that is Hispanic: '+str(hPop/totPop*100)+'%')

print('Percentage of Population that is Asian: '+str(aPop/totPop*100)+'%')

print('Percentage of Population that is Mixed: '+str(mPop/totPop*100)+'%')

print('Percentage of Population that is Other: '+str(oPop/totPop*100)+'%')
print('CHICAGO POLICE BEAT POPULATION AND SQ. MILEAGE BREAKDOWN:\n')

print('Population of Chicago Police Beats: '+str(df_beat['population'].sum()))

print('Square Mileage of Chicago Police Beats: '+str(df_beat['square_mileage'].sum()))
tot_MI = df_beat['med_income'].sum()

tot_FS = df_beat['pop_food_stamps'].sum()

print('CHICAGO POLICE BEATS INCOME AND FOOD STAMPS BREAKDOWN:\n')

print('Average Median Income: '+str(tot_MI/len(df_beat)))

print('Percentage of Population on Food Stamps: '+str(tot_FS/totPop*100)+'%')
bachelorsTot = df_beat['bachelors'].sum()

hsTot = df_beat['high_school'].sum()

no_hsTot = df_beat['no_high_school'].sum()

print("CHICAGO POLICE BEATS EDUCATIONAL ATTAINMENT BREAKDOWN:\n")

print('Percentage of Population with at least a Bachelor\'s Degree: '+ str(bachelorsTot/totPop*100)+'%')

print('Percentage of Population with at most a High School Diploma: '+ str(hsTot/totPop*100)+'%')

print('Percentage of Population without a High School Diploma: '+ str(no_hsTot/totPop*100)+'%')
minors = df_beat['<=21'].sum()

twenties = df_beat['22-29'].sum()

thirties = df_beat['30-39'].sum()

forties = df_beat['40-49'].sum()

fifties = df_beat['50-59'].sum()

sixties = df_beat['60-64'].sum()

seniors = df_beat['65+'].sum()

print('CHICAGO POLICE BEAT AGE BREAKDOWN:\n')

print('Percentage of Residents that are Minors (<=21): '+str(minors/totPop*100)+'%')

print('Percentage of Residents that are in their Twenties: '+str(twenties/totPop*100)+'%')

print('Percentage of Residents that are in their Thirties: '+str(thirties/totPop*100)+'%')

print('Percentage of Residents that are in their Forties: '+str(forties/totPop*100)+'%')

print('Percentage of Residents that are in their Fifties: '+str(fifties/totPop*100)+'%')

print('Percentage of Residents that are in between 60-64: '+str(sixties/totPop*100)+'%')

print('Percentage of Residents that are Seniors (65+): '+str(seniors/totPop*100)+'%')
total_se = df_beat['total_se'].sum()

total_se_0to4 = df_beat['se_0-4'].sum()

total_se_5to9 = df_beat['se_5-9'].sum()

total_se_10to14 = df_beat['se_10-14'].sum()

total_se_15to17 = df_beat['se_15-17'].sum()

total_se_18to19 = df_beat['se_18-19'].sum()

total_se_20to24 = df_beat['se_20-24'].sum()

total_se_25to34 = df_beat['se_25-34'].sum()

total_se_35plus = df_beat['se_35+'].sum()

total_0to4 = df_beat['0-4'].sum()

total_5to9 = df_beat['5-9'].sum()

total_10to14 = df_beat['10-14'].sum()

total_15to17 = df_beat['15-17'].sum()

total_18to19 = df_beat['18-19'].sum()

total_20to24 = df_beat['20'].sum() + df_beat['21'].sum() + df_beat['22-24'].sum()

total_25to34 = df_beat['25-29'].sum() + df_beat['30-34'].sum() 

total_youngPop = totPop - (df_beat['25-29'].sum() + df_beat['30-34'].sum()) - (df_beat['20'].sum() + df_beat['21'].sum() + df_beat['22-24'].sum()) - df_beat['18-19'].sum() - df_beat['15-17'].sum() -df_beat['10-14'].sum() -df_beat['5-9'].sum()-df_beat['0-4'].sum()



print('CHICAGO POLICE BEAT SCHOOL ENROLLMENT BREAKDOWN:\n')

print('Percentage of Residents that are Enrolled Students (All): '+str(total_se/totPop*100)+'%')

print('Percentage of Residents that are Enrolled Students (0-4): '+str(total_se_0to4/total_0to4*100)+'%')

print('Percentage of Residents that are Enrolled Students (5-9): '+str(total_se_5to9/total_5to9*100)+'%')

print('Percentage of Residents that are Enrolled Students (10-14): '+str(total_se_10to14/total_10to14*100)+'%')

print('Percentage of Residents that are Enrolled Students (15-17): '+str(total_se_15to17/total_15to17*100)+'%')

print('Percentage of Residents that are Enrolled Students (18-19): '+str(total_se_18to19/total_18to19*100)+'%')

print('Percentage of Residents that are Enrolled Students (20-24): '+str(total_se_20to24/total_20to24*100)+'%')

print('Percentage of Residents that are Enrolled Students (25-34): '+str(total_se_25to34/total_25to34*100)+'%')

print('Percentage of Residents that are Enrolled Students (35+): '+str(total_se_35plus/total_youngPop*100)+'%')
df_beat.sort_values(by = 'percent_white', ascending = False, inplace = True)

pW_beats = list(df_beat.beat)

df_beat['rank_%w'] = df_beat.beat.apply(lambda x: pW_beats.index(x)+1)

#df_beat.head()
df_beat.sort_values(by = 'percent_black', ascending = False, inplace = True)

pB_beats = list(df_beat.beat)

df_beat['rank_%b'] = df_beat.beat.apply(lambda x: pB_beats.index(x)+1)

#df_beat.head()
df_beat.sort_values(by = 'percent_hispanic', ascending = False, inplace = True)

pH_beats = list(df_beat.beat)

df_beat['rank_%h'] = df_beat.beat.apply(lambda x: pH_beats.index(x)+1)

#df_beat.head()
df_beat.sort_values(by = 'percent_asian', ascending = False, inplace = True)

pA_beats = list(df_beat.beat)

df_beat['rank_%a'] = df_beat.beat.apply(lambda x: pA_beats.index(x)+1)

#df_beat.head()
df_beat.sort_values(by = 'percent_mixed', ascending = False, inplace = True)

pM_beats = list(df_beat.beat)

df_beat['rank_%m'] = df_beat.beat.apply(lambda x: pM_beats.index(x)+1)

#df_beat.head()
df_beat.sort_values(by = 'percent_other', ascending = False, inplace = True)

pO_beats = list(df_beat.beat)

df_beat['rank_%o'] = df_beat.beat.apply(lambda x: pO_beats.index(x)+1)

#df_beat.head()
df_beat.drop(['percent_white', 'percent_hispanic', 'percent_black', 'percent_mixed', 'percent_other', 'percent_asian'], axis = 1, inplace = True)

df_beat.head()
df_beat.sort_values(by = 'med_income', ascending = False, inplace = True)

mI_beats = list(df_beat.beat)

df_beat['rank_income'] = df_beat.beat.apply(lambda x: mI_beats.index(x)+1)

#df_beat.head()
df_beat.sort_values(by = 'percent_on_fs', ascending = False, inplace = True)

fS_beats = list(df_beat.beat)

df_beat['rank_fs'] = df_beat.beat.apply(lambda x: fS_beats.index(x)+1)

#df_beat.head()
df_beat.sort_values(by = 'percent_bachelors', ascending = False, inplace = True)

bachelors_beats = list(df_beat.beat)

df_beat['rank_bachelors'] = df_beat.beat.apply(lambda x: bachelors_beats.index(x)+1)

df_beat.sort_values(by = 'percent_high_school', ascending = False, inplace = True)

HS_beats = list(df_beat.beat)

df_beat['rank_high_school'] = df_beat.beat.apply(lambda x: HS_beats.index(x)+1)

df_beat.sort_values(by = 'percent_no_high_school', ascending = False, inplace = True)

no_HS_beats = list(df_beat.beat)

df_beat['rank_no_high_school'] = df_beat.beat.apply(lambda x: no_HS_beats.index(x)+1)

df_beat.sort_values(by = 'beat', ascending = True, inplace = True)

#df_beat.head()
df_beat.sort_values(by = 'percent_se', ascending = False, inplace = True)

se_beats = list(df_beat.beat)

df_beat['rank_total_se'] = df_beat.beat.apply(lambda x: se_beats.index(x)+1)

df_beat.sort_values(by = 'percent_se_0-4', ascending = False, inplace = True)

se04_beats = list(df_beat.beat)

df_beat['rank_total_se_0-4'] = df_beat.beat.apply(lambda x: se04_beats.index(x)+1)

df_beat.sort_values(by = 'percent_se_5-9', ascending = False, inplace = True)

se59_beats = list(df_beat.beat)

df_beat['rank_total_se_5-9'] = df_beat.beat.apply(lambda x: se59_beats.index(x)+1)

df_beat.sort_values(by = 'percent_se_10-14', ascending = False, inplace = True)

se1014_beats = list(df_beat.beat)

df_beat['rank_total_se_10-14'] = df_beat.beat.apply(lambda x: se1014_beats.index(x)+1)

df_beat.sort_values(by = 'percent_se_15-17', ascending = False, inplace = True)

se1517_beats = list(df_beat.beat)

df_beat['rank_total_se_15-17'] = df_beat.beat.apply(lambda x: se1517_beats.index(x)+1)

df_beat.sort_values(by = 'percent_se_18-19', ascending = False, inplace = True)

se1819_beats = list(df_beat.beat)

df_beat['rank_total_se_18-19'] = df_beat.beat.apply(lambda x: se1819_beats.index(x)+1)

df_beat.sort_values(by = 'percent_se_20-24', ascending = False, inplace = True)

se2024_beats = list(df_beat.beat)

df_beat['rank_total_se_20-24'] = df_beat.beat.apply(lambda x: se2024_beats.index(x)+1)

df_beat.sort_values(by = 'percent_se_25-34', ascending = False, inplace = True)

se2534_beats = list(df_beat.beat)

df_beat['rank_total_se_25-34'] = df_beat.beat.apply(lambda x: se2534_beats.index(x)+1)

df_beat.sort_values(by = 'percent_se_35+', ascending = False, inplace = True)

se35_beats = list(df_beat.beat)

df_beat['rank_total_se_35+'] = df_beat.beat.apply(lambda x: se35_beats.index(x)+1)

#df_beat.head()
topFile = open("/kaggle/input/cpd-police-beat-demographics/outputBeats_EDA_TOP.txt","r") 
topFile.readline()

t20_crimes = topFile.readline().split(' ')

t20_crimes = t20_crimes[:-1]

for i in range (0,20):

    t20_crimes[i] = pad0(t20_crimes[i])

print(t20_crimes)
df_topCrimes = df_beat[df_beat['beat'].isin(t20_crimes)]

df_topCrimes.set_index('beat', inplace = True)

df_topCrimes = df_topCrimes.reindex(t20_crimes)
df_topCrimes.reset_index(inplace = True)

df_topCrimes
totPop = 0

wPop = 0

bPop = 0

hPop = 0

mPop = 0

aPop = 0

oPop = 0

for ind in df_topCrimes.index:

    totPop += df_topCrimes['population'][ind]

    wPop += df_topCrimes['num_white'][ind]

    bPop += df_topCrimes['num_black'][ind]

    hPop += df_topCrimes['num_hispanic'][ind]

    mPop += df_topCrimes['num_mixed'][ind]

    aPop += df_topCrimes['num_asian'][ind]

    oPop += df_topCrimes['num_other'][ind]

print('TOP 20 CRIME BEATS RACE BREAKDOWN:\n')

print('Percentage of Population that is White: '+str(wPop/totPop*100)+'%')

print('Percentage of Population that is Black: '+str(bPop/totPop*100)+'%')

print('Percentage of Population that is Hispanic: '+str(hPop/totPop*100)+'%')

print('Percentage of Population that is Asian: '+str(aPop/totPop*100)+'%')

print('Percentage of Population that is Mixed: '+str(mPop/totPop*100)+'%')

print('Percentage of Population that is Other: '+str(oPop/totPop*100)+'%')
tot_MI = 0

tot_FS = 0

for ind in df_topCrimes.index:

    tot_MI += df_topCrimes['med_income'][ind]

    tot_FS += df_topCrimes['pop_food_stamps'][ind]

print('TOP 20 CRIME BEATS INCOME AND FOOD STAMPS BREAKDOWN:\n')

print('Average Median Income: '+str(tot_MI/len(df_topCrimes)))

print('Percentage of Population on Food Stamps: '+str(tot_FS/totPop*100)+'%')
bachelorsPop = df_topCrimes['bachelors'].sum()

HSPop = df_topCrimes['high_school'].sum()

no_HSPop = df_topCrimes['no_high_school'].sum()

print('TOP 20 CRIME BEATS EDUCATIONAL ATTAINMENT BREAKDOWN:\n')

print('Percentage of Population with at least a Bachelor\'s Degree: '+str(bachelorsPop/totPop*100)+'%')

print('Percentage of Population with at most a High School Diploma '+str(HSPop/totPop*100)+'%')

print('Percentage of Population without a High School Diploma '+str(no_HSPop/totPop*100)+'%')
total_se = df_topCrimes['total_se'].sum()

total_se_0to4 = df_topCrimes['se_0-4'].sum()

total_se_5to9 = df_topCrimes['se_5-9'].sum()

total_se_10to14 = df_topCrimes['se_10-14'].sum()

total_se_15to17 = df_topCrimes['se_15-17'].sum()

total_se_18to19 = df_topCrimes['se_18-19'].sum()

total_se_20to24 = df_topCrimes['se_20-24'].sum()

total_se_25to34 = df_topCrimes['se_25-34'].sum()

total_se_35plus = df_topCrimes['se_35+'].sum()

total_0to4 = df_topCrimes['0-4'].sum()

total_5to9 = df_topCrimes['5-9'].sum()

total_10to14 = df_topCrimes['10-14'].sum()

total_15to17 = df_topCrimes['15-17'].sum()

total_18to19 = df_topCrimes['18-19'].sum()

total_20to24 = df_topCrimes['20'].sum() + df_topCrimes['21'].sum() + df_topCrimes['22-24'].sum()

total_25to34 = df_topCrimes['25-29'].sum() + df_topCrimes['30-34'].sum() 

total_youngPop = totPop - (df_topCrimes['25-29'].sum() + df_topCrimes['30-34'].sum()) - (df_topCrimes['20'].sum() + df_topCrimes['21'].sum() + df_topCrimes['22-24'].sum()) - df_topCrimes['18-19'].sum() - df_topCrimes['15-17'].sum() -df_topCrimes['10-14'].sum() -df_topCrimes['5-9'].sum()-df_topCrimes['0-4'].sum()



print('TOP 20 CRIME BEATS SCHOOL ENROLLMENT BREAKDOWN:\n')

print('Percentage of Residents that are Enrolled Students (All): '+str(total_se/totPop*100)+'%')

print('Percentage of Residents that are Enrolled Students (0-4): '+str(total_se_0to4/total_0to4*100)+'%')

print('Percentage of Residents that are Enrolled Students (5-9): '+str(total_se_5to9/total_5to9*100)+'%')

print('Percentage of Residents that are Enrolled Students (10-14): '+str(total_se_10to14/total_10to14*100)+'%')

print('Percentage of Residents that are Enrolled Students (15-17): '+str(total_se_15to17/total_15to17*100)+'%')

print('Percentage of Residents that are Enrolled Students (18-19): '+str(total_se_18to19/total_18to19*100)+'%')

print('Percentage of Residents that are Enrolled Students (20-24): '+str(total_se_20to24/total_20to24*100)+'%')

print('Percentage of Residents that are Enrolled Students (25-34): '+str(total_se_25to34/total_25to34*100)+'%')

print('Percentage of Residents that are Enrolled Students (35+): '+str(total_se_35plus/total_youngPop*100)+'%')
minors = df_topCrimes['<=21'].sum()

twenties = df_topCrimes['22-29'].sum()

thirties = df_topCrimes['30-39'].sum()

forties = df_topCrimes['40-49'].sum()

fifties = df_topCrimes['50-59'].sum()

sixties = df_topCrimes['60-64'].sum()

seniors = df_topCrimes['65+'].sum()

print('TOP 20 CRIME BEATS AGE BREAKDOWN:\n')

print('Percentage of Residents that are Minors (<=21): '+str(minors/totPop*100)+'%')

print('Percentage of Residents that are in their Twenties: '+str(twenties/totPop*100)+'%')

print('Percentage of Residents that are in their Thirties: '+str(thirties/totPop*100)+'%')

print('Percentage of Residents that are in their Forties: '+str(forties/totPop*100)+'%')

print('Percentage of Residents that are in their Fifties: '+str(fifties/totPop*100)+'%')

print('Percentage of Residents that are in between 60-64: '+str(sixties/totPop*100)+'%')

print('Percentage of Residents that are Seniors (65+): '+str(seniors/totPop*100)+'%')
df_topCrime_ranks = df_topCrimes[['beat','rank_%w', 'rank_%b', 'rank_%h', 'rank_%a', 'rank_%m',

       'rank_%o', 'rank_income', 'rank_fs', 'rank_bachelors',

       'rank_high_school', 'rank_no_high_school', 'rank_total_se',

       'rank_total_se_0-4', 'rank_total_se_5-9', 'rank_total_se_10-14',

       'rank_total_se_15-17', 'rank_total_se_18-19', 'rank_total_se_20-24',

       'rank_total_se_25-34', 'rank_total_se_35+']]
beatList = list(df_topCrime_ranks.beat)

df_topCrime_ranks['rank_beat'] = df_topCrime_ranks.beat.apply(lambda x: beatList.index(x)+1)

df_topCrime_ranks.set_index('beat',inplace = True)

print('TOP 20 CRIME BEATS MEDIAN RANKINGS: ')

print(df_topCrime_ranks.median(axis = 0))

print()

print('TOP 20 CRIME BEATS AVERAGE (MEAN) RANKINGS: ')

print(df_topCrime_ranks.mean(axis = 0))

df_topCrime_ranks
topFile.readline()

t20_arrests = topFile.readline().split(' ')

t20_arrests = t20_arrests[:-1]

for i in range (0,20):

    t20_arrests[i] = pad0(t20_arrests[i])

print(t20_arrests)
df_topArrests = df_beat[df_beat['beat'].isin(t20_arrests)]

df_topArrests.set_index('beat', inplace = True)

df_topArrests = df_topArrests.reindex(t20_arrests)
df_topArrests.reset_index(inplace = True)

df_topArrests
totPop = 0

wPop = 0

bPop = 0

hPop = 0

mPop = 0

aPop = 0

oPop = 0

for ind in df_topArrests.index:

    totPop += df_topArrests['population'][ind]

    wPop += df_topArrests['num_white'][ind]

    bPop += df_topArrests['num_black'][ind]

    hPop += df_topArrests['num_hispanic'][ind]

    mPop += df_topArrests['num_mixed'][ind]

    aPop += df_topArrests['num_asian'][ind]

    oPop += df_topArrests['num_other'][ind]

print('TOP 20 ARRESTS BEATS RACE BREAKDOWN:\n')

print('Percentage of Population that is White: '+str(wPop/totPop*100)+'%')

print('Percentage of Population that is Black: '+str(bPop/totPop*100)+'%')

print('Percentage of Population that is Hispanic: '+str(hPop/totPop*100)+'%')

print('Percentage of Population that is Asian: '+str(aPop/totPop*100)+'%')

print('Percentage of Population that is Mixed: '+str(mPop/totPop*100)+'%')

print('Percentage of Population that is Other: '+str(oPop/totPop*100)+'%')
tot_MI = 0

tot_FS = 0

for ind in df_topArrests.index:

    tot_MI += df_topArrests['med_income'][ind]

    tot_FS += df_topArrests['pop_food_stamps'][ind]

print('TOP 20 ARRESTS BEATS INCOME AND FOOD STAMPS BREAKDOWN:\n')

print('Average Median Income: '+str(tot_MI/len(df_topArrests)))

print('Percentage of Population on Food Stamps: '+str(tot_FS/totPop*100)+'%')
bachelorsPop = df_topArrests['bachelors'].sum()

HSPop = df_topArrests['high_school'].sum()

no_HSPop = df_topArrests['no_high_school'].sum()

print('TOP 20 ARRESTS BEATS EDUCATIONAL ATTAINMENT BREAKDOWN:\n')

print('Percentage of Population with at least a Bachelor\'s Degree: '+str(bachelorsPop/totPop*100)+'%')

print('Percentage of Population with at most a High School Diploma '+str(HSPop/totPop*100)+'%')

print('Percentage of Population without a High School Diploma '+str(no_HSPop/totPop*100)+'%')
total_se = df_topArrests['total_se'].sum()

total_se_0to4 = df_topArrests['se_0-4'].sum()

total_se_5to9 = df_topArrests['se_5-9'].sum()

total_se_10to14 = df_topArrests['se_10-14'].sum()

total_se_15to17 = df_topArrests['se_15-17'].sum()

total_se_18to19 = df_topArrests['se_18-19'].sum()

total_se_20to24 = df_topArrests['se_20-24'].sum()

total_se_25to34 = df_topArrests['se_25-34'].sum()

total_se_35plus = df_topArrests['se_35+'].sum()

total_0to4 = df_topArrests['0-4'].sum()

total_5to9 = df_topArrests['5-9'].sum()

total_10to14 = df_topArrests['10-14'].sum()

total_15to17 = df_topArrests['15-17'].sum()

total_18to19 = df_topArrests['18-19'].sum()

total_20to24 = df_topArrests['20'].sum() + df_topArrests['21'].sum() + df_topArrests['22-24'].sum()

total_25to34 = df_topArrests['25-29'].sum() + df_topArrests['30-34'].sum() 

total_youngPop = totPop - (df_topArrests['25-29'].sum() + df_topArrests['30-34'].sum()) - (df_topArrests['20'].sum() + df_topArrests['21'].sum() + df_topArrests['22-24'].sum()) - df_topArrests['18-19'].sum() - df_topArrests['15-17'].sum() -df_topArrests['10-14'].sum() -df_topArrests['5-9'].sum()-df_topArrests['0-4'].sum()



print('TOP 20 ARRESTS BEATS SCHOOL ENROLLMENT BREAKDOWN:\n')

print('Percentage of Residents that are Enrolled Students (All): '+str(total_se/totPop*100)+'%')

print('Percentage of Residents that are Enrolled Students (0-4): '+str(total_se_0to4/total_0to4*100)+'%')

print('Percentage of Residents that are Enrolled Students (5-9): '+str(total_se_5to9/total_5to9*100)+'%')

print('Percentage of Residents that are Enrolled Students (10-14): '+str(total_se_10to14/total_10to14*100)+'%')

print('Percentage of Residents that are Enrolled Students (15-17): '+str(total_se_15to17/total_15to17*100)+'%')

print('Percentage of Residents that are Enrolled Students (18-19): '+str(total_se_18to19/total_18to19*100)+'%')

print('Percentage of Residents that are Enrolled Students (20-24): '+str(total_se_20to24/total_20to24*100)+'%')

print('Percentage of Residents that are Enrolled Students (25-34): '+str(total_se_25to34/total_25to34*100)+'%')

print('Percentage of Residents that are Enrolled Students (35+): '+str(total_se_35plus/total_youngPop*100)+'%')
minors = df_topArrests['<=21'].sum()

twenties = df_topArrests['22-29'].sum()

thirties = df_topArrests['30-39'].sum()

forties = df_topArrests['40-49'].sum()

fifties = df_topArrests['50-59'].sum()

sixties = df_topArrests['60-64'].sum()

seniors = df_topArrests['65+'].sum()

print('TOP 20 ARRESTS BEATS AGE BREAKDOWN:\n')

print('Percentage of Residents that are Minors (<=21): '+str(minors/totPop*100)+'%')

print('Percentage of Residents that are in their Twenties: '+str(twenties/totPop*100)+'%')

print('Percentage of Residents that are in their Thirties: '+str(thirties/totPop*100)+'%')

print('Percentage of Residents that are in their Forties: '+str(forties/totPop*100)+'%')

print('Percentage of Residents that are in their Fifties: '+str(fifties/totPop*100)+'%')

print('Percentage of Residents that are in between 60-64: '+str(sixties/totPop*100)+'%')

print('Percentage of Residents that are Seniors (65+): '+str(seniors/totPop*100)+'%')
df_topArrest_ranks = df_topArrests[['beat','rank_%w', 'rank_%b', 'rank_%h', 'rank_%a', 'rank_%m',

       'rank_%o', 'rank_income', 'rank_fs', 'rank_bachelors',

       'rank_high_school', 'rank_no_high_school', 'rank_total_se',

       'rank_total_se_0-4', 'rank_total_se_5-9', 'rank_total_se_10-14',

       'rank_total_se_15-17', 'rank_total_se_18-19', 'rank_total_se_20-24',

       'rank_total_se_25-34', 'rank_total_se_35+']]
beatList = list(df_topArrest_ranks.beat)

df_topArrest_ranks['rank_beat'] = df_topArrest_ranks.beat.apply(lambda x: beatList.index(x)+1)

df_topArrest_ranks.set_index('beat',inplace = True)

print('TOP 20 ARRESTS BEATS MEDIAN RANKINGS: ')

print(df_topArrest_ranks.median(axis = 0))

print()

print('TOP 20 ARRESTS BEATS AVERAGE (MEAN) RANKINGS: ')

print(df_topArrest_ranks.mean(axis = 0))

df_topArrest_ranks
topFile.readline()

t20_ratio = topFile.readline().split(' ')

t20_ratio = t20_ratio[:-1]

for i in range (0,20):

    t20_ratio[i] = pad0(t20_ratio[i])

print(t20_ratio)
df_topRatio = df_beat[df_beat['beat'].isin(t20_ratio)]

df_topRatio.set_index('beat', inplace = True)

df_topRatio = df_topRatio.reindex(t20_ratio)
df_topRatio.reset_index(inplace = True)

df_topRatio
df_topRatio = df_topRatio.loc[(df_topRatio.beat != '0134') & (df_topRatio.beat != '2113')]
totPop = 0

wPop = 0

bPop = 0

hPop = 0

mPop = 0

aPop = 0

oPop = 0

for ind in df_topRatio.index:

    totPop += df_topRatio['population'][ind]

    wPop += df_topRatio['num_white'][ind]

    bPop += df_topRatio['num_black'][ind]

    hPop += df_topRatio['num_hispanic'][ind]

    mPop += df_topRatio['num_mixed'][ind]

    aPop += df_topRatio['num_asian'][ind]

    oPop += df_topRatio['num_other'][ind]

print('TOP 20 ARRESTS:CRIME BEATS RACE BREAKDOWN:\n')

print('Percentage of Population that is White: '+str(wPop/totPop*100)+'%')

print('Percentage of Population that is Black: '+str(bPop/totPop*100)+'%')

print('Percentage of Population that is Hispanic: '+str(hPop/totPop*100)+'%')

print('Percentage of Population that is Asian: '+str(aPop/totPop*100)+'%')

print('Percentage of Population that is Mixed: '+str(mPop/totPop*100)+'%')

print('Percentage of Population that is Other: '+str(oPop/totPop*100)+'%')
tot_MI = 0

tot_FS = 0

for ind in df_topRatio.index:

    tot_MI += df_topRatio['med_income'][ind]

    tot_FS += df_topRatio['pop_food_stamps'][ind]

print('TOP 20 ARRESTS:CRIME BEATS INCOME AND FOOD STAMPS BREAKDOWN:\n')

print('Average Median Income: '+str(tot_MI/len(df_topRatio)))

print('Percentage of Population on Food Stamps: '+str(tot_FS/totPop*100)+'%')
bachelorsPop = df_topRatio['bachelors'].sum()

HSPop = df_topRatio['high_school'].sum()

no_HSPop = df_topRatio['no_high_school'].sum()

print('TOP 20 ARRESTS:CRIME BEATS EDUCATIONAL ATTAINMENT BREAKDOWN:\n')

print('Percentage of Population with at least a Bachelor\'s Degree: '+str(bachelorsPop/totPop*100)+'%')

print('Percentage of Population with at most a High School Diploma '+str(HSPop/totPop*100)+'%')

print('Percentage of Population without a High School Diploma '+str(no_HSPop/totPop*100)+'%')
total_se = df_topRatio['total_se'].sum()

total_se_0to4 = df_topRatio['se_0-4'].sum()

total_se_5to9 = df_topRatio['se_5-9'].sum()

total_se_10to14 = df_topRatio['se_10-14'].sum()

total_se_15to17 = df_topRatio['se_15-17'].sum()

total_se_18to19 = df_topRatio['se_18-19'].sum()

total_se_20to24 = df_topRatio['se_20-24'].sum()

total_se_25to34 = df_topRatio['se_25-34'].sum()

total_se_35plus = df_topRatio['se_35+'].sum()

total_0to4 = df_topRatio['0-4'].sum()

total_5to9 = df_topRatio['5-9'].sum()

total_10to14 = df_topRatio['10-14'].sum()

total_15to17 = df_topRatio['15-17'].sum()

total_18to19 = df_topRatio['18-19'].sum()

total_20to24 = df_topRatio['20'].sum() + df_topRatio['21'].sum() + df_topRatio['22-24'].sum()

total_25to34 = df_topRatio['25-29'].sum() + df_topRatio['30-34'].sum() 

total_youngPop = totPop - (df_topRatio['25-29'].sum() + df_topRatio['30-34'].sum()) - (df_topRatio['20'].sum() + df_topRatio['21'].sum() + df_topRatio['22-24'].sum()) - df_topRatio['18-19'].sum() - df_topRatio['15-17'].sum() -df_topRatio['10-14'].sum() -df_topRatio['5-9'].sum()-df_topRatio['0-4'].sum()



print('TOP 20 ARRESTS:CRIME BEATS SCHOOL ENROLLMENT BREAKDOWN:\n')

print('Percentage of Residents that are Enrolled Students (All): '+str(total_se/totPop*100)+'%')

print('Percentage of Residents that are Enrolled Students (0-4): '+str(total_se_0to4/total_0to4*100)+'%')

print('Percentage of Residents that are Enrolled Students (5-9): '+str(total_se_5to9/total_5to9*100)+'%')

print('Percentage of Residents that are Enrolled Students (10-14): '+str(total_se_10to14/total_10to14*100)+'%')

print('Percentage of Residents that are Enrolled Students (15-17): '+str(total_se_15to17/total_15to17*100)+'%')

print('Percentage of Residents that are Enrolled Students (18-19): '+str(total_se_18to19/total_18to19*100)+'%')

print('Percentage of Residents that are Enrolled Students (20-24): '+str(total_se_20to24/total_20to24*100)+'%')

print('Percentage of Residents that are Enrolled Students (25-34): '+str(total_se_25to34/total_25to34*100)+'%')

print('Percentage of Residents that are Enrolled Students (35+): '+str(total_se_35plus/total_youngPop*100)+'%')
minors = df_topRatio['<=21'].sum()

twenties = df_topRatio['22-29'].sum()

thirties = df_topRatio['30-39'].sum()

forties = df_topRatio['40-49'].sum()

fifties = df_topRatio['50-59'].sum()

sixties = df_topRatio['60-64'].sum()

seniors = df_topRatio['65+'].sum()

print('TOP 20 ARRREST:CRIME BEATS AGE BREAKDOWN:\n')

print('Percentage of Residents that are Minors (<=21): '+str(minors/totPop*100)+'%')

print('Percentage of Residents that are in their Twenties: '+str(twenties/totPop*100)+'%')

print('Percentage of Residents that are in their Thirties: '+str(thirties/totPop*100)+'%')

print('Percentage of Residents that are in their Forties: '+str(forties/totPop*100)+'%')

print('Percentage of Residents that are in their Fifties: '+str(fifties/totPop*100)+'%')

print('Percentage of Residents that are in between 60-64: '+str(sixties/totPop*100)+'%')

print('Percentage of Residents that are Seniors (65+): '+str(seniors/totPop*100)+'%')
df_topRatio_ranks = df_topRatio[['beat','rank_%w', 'rank_%b', 'rank_%h', 'rank_%a', 'rank_%m',

       'rank_%o', 'rank_income', 'rank_fs', 'rank_bachelors',

       'rank_high_school', 'rank_no_high_school', 'rank_total_se',

       'rank_total_se_0-4', 'rank_total_se_5-9', 'rank_total_se_10-14',

       'rank_total_se_15-17', 'rank_total_se_18-19', 'rank_total_se_20-24',

       'rank_total_se_25-34', 'rank_total_se_35+']]
beatList = list(df_topRatio_ranks.beat)

df_topRatio_ranks['rank_beat'] = df_topRatio_ranks.beat.apply(lambda x: beatList.index(x)+1)

df_topRatio_ranks.set_index('beat',inplace = True)

print('TOP 20 ARRESTS:CRIME BEATS MEDIAN RANKINGS: ')

print(df_topRatio_ranks.median(axis = 0))

print()

print('TOP 20 ARRESTS:CRIME BEATS AVERAGE (MEAN) RANKINGS: ')

print(df_topRatio_ranks.mean(axis = 0))

df_topRatio_ranks
topFile.readline()

t20_injuries = topFile.readline().split(' ')

t20_injuries = t20_injuries[:-1]

for i in range (0,20):

    t20_injuries[i] = pad0(t20_injuries[i])

print(t20_injuries)
df_topInjuries = df_beat[df_beat['beat'].isin(t20_injuries)]

df_topInjuries.set_index('beat', inplace = True)

df_topInjuries = df_topInjuries.reindex(t20_injuries)
df_topInjuries.reset_index(inplace = True)

df_topInjuries
totPop = 0

wPop = 0

bPop = 0

hPop = 0

mPop = 0

aPop = 0

oPop = 0

for ind in df_topInjuries.index:

    totPop += df_topInjuries['population'][ind]

    wPop += df_topInjuries['num_white'][ind]

    bPop += df_topInjuries['num_black'][ind]

    hPop += df_topInjuries['num_hispanic'][ind]

    mPop += df_topInjuries['num_mixed'][ind]

    aPop += df_topInjuries['num_asian'][ind]

    oPop += df_topInjuries['num_other'][ind]

print('TOP 20 INJURY BEATS RACE BREAKDOWN:\n')

print('Percentage of Population that is White: '+str(wPop/totPop*100)+'%')

print('Percentage of Population that is Black: '+str(bPop/totPop*100)+'%')

print('Percentage of Population that is Hispanic: '+str(hPop/totPop*100)+'%')

print('Percentage of Population that is Asian: '+str(aPop/totPop*100)+'%')

print('Percentage of Population that is Mixed: '+str(mPop/totPop*100)+'%')

print('Percentage of Population that is Other: '+str(oPop/totPop*100)+'%')
tot_MI = 0

tot_FS = 0

for ind in df_topInjuries.index:

    tot_MI += df_topInjuries['med_income'][ind]

    tot_FS += df_topInjuries['pop_food_stamps'][ind]

print('TOP 20 INJURY BEATS INCOME AND FOOD STAMPS BREAKDOWN:\n')

print('Average Median Income: '+str(tot_MI/20.0))

print('Percentage of Population on Food Stamps: '+str(tot_FS/totPop*100)+'%')
bachelorsPop = df_topInjuries['bachelors'].sum()

HSPop = df_topInjuries['high_school'].sum()

no_HSPop = df_topInjuries['no_high_school'].sum()

print('TOP 20 INJURY BEATS EDUCATIONAL ATTAINMENT BREAKDOWN:\n')

print('Percentage of Population with at least a Bachelor\'s Degree: '+str(bachelorsPop/totPop*100)+'%')

print('Percentage of Population with at most a High School Diploma '+str(HSPop/totPop*100)+'%')

print('Percentage of Population without a High School Diploma '+str(no_HSPop/totPop*100)+'%')
total_se = df_topInjuries['total_se'].sum()

total_se_0to4 = df_topInjuries['se_0-4'].sum()

total_se_5to9 = df_topInjuries['se_5-9'].sum()

total_se_10to14 = df_topInjuries['se_10-14'].sum()

total_se_15to17 = df_topInjuries['se_15-17'].sum()

total_se_18to19 = df_topInjuries['se_18-19'].sum()

total_se_20to24 = df_topInjuries['se_20-24'].sum()

total_se_25to34 = df_topInjuries['se_25-34'].sum()

total_se_35plus = df_topInjuries['se_35+'].sum()

total_0to4 = df_topInjuries['0-4'].sum()

total_5to9 = df_topInjuries['5-9'].sum()

total_10to14 = df_topInjuries['10-14'].sum()

total_15to17 = df_topInjuries['15-17'].sum()

total_18to19 = df_topInjuries['18-19'].sum()

total_20to24 = df_topInjuries['20'].sum() + df_topInjuries['21'].sum() + df_topInjuries['22-24'].sum()

total_25to34 = df_topInjuries['25-29'].sum() + df_topInjuries['30-34'].sum() 

total_youngPop = totPop - (df_topInjuries['25-29'].sum() + df_topInjuries['30-34'].sum()) - (df_topInjuries['20'].sum() + df_topInjuries['21'].sum() + df_topInjuries['22-24'].sum()) - df_topInjuries['18-19'].sum() - df_topInjuries['15-17'].sum() -df_topInjuries['10-14'].sum() -df_topInjuries['5-9'].sum()-df_topInjuries['0-4'].sum()



print('TOP 20 INJURY BEATS SCHOOL ENROLLMENT BREAKDOWN:\n')

print('Percentage of Residents that are Enrolled Students (All): '+str(total_se/totPop*100)+'%')

print('Percentage of Residents that are Enrolled Students (0-4): '+str(total_se_0to4/total_0to4*100)+'%')

print('Percentage of Residents that are Enrolled Students (5-9): '+str(total_se_5to9/total_5to9*100)+'%')

print('Percentage of Residents that are Enrolled Students (10-14): '+str(total_se_10to14/total_10to14*100)+'%')

print('Percentage of Residents that are Enrolled Students (15-17): '+str(total_se_15to17/total_15to17*100)+'%')

print('Percentage of Residents that are Enrolled Students (18-19): '+str(total_se_18to19/total_18to19*100)+'%')

print('Percentage of Residents that are Enrolled Students (20-24): '+str(total_se_20to24/total_20to24*100)+'%')

print('Percentage of Residents that are Enrolled Students (25-34): '+str(total_se_25to34/total_25to34*100)+'%')

print('Percentage of Residents that are Enrolled Students (35+): '+str(total_se_35plus/total_youngPop*100)+'%')
minors = df_topInjuries['<=21'].sum()

twenties = df_topInjuries['22-29'].sum()

thirties = df_topInjuries['30-39'].sum()

forties = df_topInjuries['40-49'].sum()

fifties = df_topInjuries['50-59'].sum()

sixties = df_topInjuries['60-64'].sum()

seniors = df_topInjuries['65+'].sum()

print('TOP 20 INJURY BEATS AGE BREAKDOWN:\n')

print('Percentage of Residents that are Minors (<=21): '+str(minors/totPop*100)+'%')

print('Percentage of Residents that are in their Twenties: '+str(twenties/totPop*100)+'%')

print('Percentage of Residents that are in their Thirties: '+str(thirties/totPop*100)+'%')

print('Percentage of Residents that are in their Forties: '+str(forties/totPop*100)+'%')

print('Percentage of Residents that are in their Fifties: '+str(fifties/totPop*100)+'%')

print('Percentage of Residents that are in between 60-64: '+str(sixties/totPop*100)+'%')

print('Percentage of Residents that are Seniors (65+): '+str(seniors/totPop*100)+'%')
df_topInjuries_ranks = df_topInjuries[['beat','rank_%w', 'rank_%b', 'rank_%h', 'rank_%a', 'rank_%m',

       'rank_%o', 'rank_income', 'rank_fs', 'rank_bachelors',

       'rank_high_school', 'rank_no_high_school', 'rank_total_se',

       'rank_total_se_0-4', 'rank_total_se_5-9', 'rank_total_se_10-14',

       'rank_total_se_15-17', 'rank_total_se_18-19', 'rank_total_se_20-24',

       'rank_total_se_25-34', 'rank_total_se_35+']]
beatList = list(df_topInjuries_ranks.beat)

df_topInjuries_ranks['rank_beat'] = df_topInjuries_ranks.beat.apply(lambda x: beatList.index(x)+1)

df_topInjuries_ranks.set_index('beat',inplace = True)

print('TOP 20 INJURY BEATS MEDIAN RANKINGS: ')

print(df_topInjuries_ranks.median(axis = 0))

print()

print('TOP 20 INJURY BEATS AVERAGE (MEAN) RANKINGS: ')

print(df_topInjuries_ranks.mean(axis = 0))

df_topInjuries.head()
topFile.readline()

t20_complaints = topFile.readline().split(' ')

t20_complaints = t20_complaints[:-1]

for i in range (0,20):

    t20_complaints[i] = pad0(t20_complaints[i])

print(t20_complaints)

topFile.close()
df_topComplaints = df_beat[df_beat['beat'].isin(t20_complaints)]

df_topComplaints.set_index('beat', inplace = True)

df_topComplaints = df_topComplaints.reindex(t20_complaints)
df_topComplaints.reset_index(inplace = True)

df_topComplaints
totPop = 0

wPop = 0

bPop = 0

hPop = 0

mPop = 0

aPop = 0

oPop = 0

for ind in df_topComplaints.index:

    totPop += df_topComplaints['population'][ind]

    wPop += df_topComplaints['num_white'][ind]

    bPop += df_topComplaints['num_black'][ind]

    hPop += df_topComplaints['num_hispanic'][ind]

    mPop += df_topComplaints['num_mixed'][ind]

    aPop += df_topComplaints['num_asian'][ind]

    oPop += df_topComplaints['num_other'][ind]

print('TOP 20 COMPLAINTS BEATS RACE BREAKDOWN:\n')

print('Percentage of Population that is White: '+str(wPop/totPop*100)+'%')

print('Percentage of Population that is Black: '+str(bPop/totPop*100)+'%')

print('Percentage of Population that is Hispanic: '+str(hPop/totPop*100)+'%')

print('Percentage of Population that is Asian: '+str(aPop/totPop*100)+'%')

print('Percentage of Population that is Mixed: '+str(mPop/totPop*100)+'%')

print('Percentage of Population that is Other: '+str(oPop/totPop*100)+'%')
tot_MI = 0

tot_FS = 0

for ind in df_topComplaints.index:

    tot_MI += df_topComplaints['med_income'][ind]

    tot_FS += df_topComplaints['pop_food_stamps'][ind]

print('TOP 20 COMPLAINTS BEATS INCOME AND FOOD STAMPS BREAKDOWN:\n')

print('Average Median Income: '+str(tot_MI/20.0))

print('Percentage of Population on Food Stamps: '+str(tot_FS/totPop*100)+'%')
bachelorsPop = df_topComplaints['bachelors'].sum()

HSPop = df_topComplaints['high_school'].sum()

no_HSPop = df_topComplaints['no_high_school'].sum()

print('TOP 20 COMPLAINTS BEATS EDUCATIONAL ATTAINMENT BREAKDOWN:\n')

print('Percentage of Population with at least a Bachelor\'s Degree: '+str(bachelorsPop/totPop*100)+'%')

print('Percentage of Population with at most a High School Diploma '+str(HSPop/totPop*100)+'%')

print('Percentage of Population without a High School Diploma '+str(no_HSPop/totPop*100)+'%')
total_se = df_topComplaints['total_se'].sum()

total_se_0to4 = df_topComplaints['se_0-4'].sum()

total_se_5to9 = df_topComplaints['se_5-9'].sum()

total_se_10to14 = df_topComplaints['se_10-14'].sum()

total_se_15to17 = df_topComplaints['se_15-17'].sum()

total_se_18to19 = df_topComplaints['se_18-19'].sum()

total_se_20to24 = df_topComplaints['se_20-24'].sum()

total_se_25to34 = df_topComplaints['se_25-34'].sum()

total_se_35plus = df_topComplaints['se_35+'].sum()

total_0to4 = df_topComplaints['0-4'].sum()

total_5to9 = df_topComplaints['5-9'].sum()

total_10to14 = df_topComplaints['10-14'].sum()

total_15to17 = df_topComplaints['15-17'].sum()

total_18to19 = df_topComplaints['18-19'].sum()

total_20to24 = df_topComplaints['20'].sum() + df_topComplaints['21'].sum() + df_topComplaints['22-24'].sum()

total_25to34 = df_topComplaints['25-29'].sum() + df_topComplaints['30-34'].sum() 

total_youngPop = totPop - (df_topComplaints['25-29'].sum() + df_topComplaints['30-34'].sum()) - (df_topComplaints['20'].sum() + df_topComplaints['21'].sum() + df_topComplaints['22-24'].sum()) - df_topComplaints['18-19'].sum() - df_topComplaints['15-17'].sum() -df_topComplaints['10-14'].sum() -df_topComplaints['5-9'].sum()-df_topComplaints['0-4'].sum()



print('TOP 20 COMPLAINTS BEATS SCHOOL ENROLLMENT BREAKDOWN:\n')

print('Percentage of Residents that are Enrolled Students (All): '+str(total_se/totPop*100)+'%')

print('Percentage of Residents that are Enrolled Students (0-4): '+str(total_se_0to4/total_0to4*100)+'%')

print('Percentage of Residents that are Enrolled Students (5-9): '+str(total_se_5to9/total_5to9*100)+'%')

print('Percentage of Residents that are Enrolled Students (10-14): '+str(total_se_10to14/total_10to14*100)+'%')

print('Percentage of Residents that are Enrolled Students (15-17): '+str(total_se_15to17/total_15to17*100)+'%')

print('Percentage of Residents that are Enrolled Students (18-19): '+str(total_se_18to19/total_18to19*100)+'%')

print('Percentage of Residents that are Enrolled Students (20-24): '+str(total_se_20to24/total_20to24*100)+'%')

print('Percentage of Residents that are Enrolled Students (25-34): '+str(total_se_25to34/total_25to34*100)+'%')

print('Percentage of Residents that are Enrolled Students (35+): '+str(total_se_35plus/total_youngPop*100)+'%')
minors = df_topComplaints['<=21'].sum()

twenties = df_topComplaints['22-29'].sum()

thirties = df_topComplaints['30-39'].sum()

forties = df_topComplaints['40-49'].sum()

fifties = df_topComplaints['50-59'].sum()

sixties = df_topComplaints['60-64'].sum()

seniors = df_topComplaints['65+'].sum()

print('TOP 20 COMPLAINTS BEATS AGE BREAKDOWN:\n')

print('Percentage of Residents that are Minors (<=21): '+str(minors/totPop*100)+'%')

print('Percentage of Residents that are in their Twenties: '+str(twenties/totPop*100)+'%')

print('Percentage of Residents that are in their Thirties: '+str(thirties/totPop*100)+'%')

print('Percentage of Residents that are in their Forties: '+str(forties/totPop*100)+'%')

print('Percentage of Residents that are in their Fifties: '+str(fifties/totPop*100)+'%')

print('Percentage of Residents that are in between 60-64: '+str(sixties/totPop*100)+'%')

print('Percentage of Residents that are Seniors (65+): '+str(seniors/totPop*100)+'%')
df_topComplaints_ranks = df_topComplaints[['beat','rank_%w', 'rank_%b', 'rank_%h', 'rank_%a', 'rank_%m',

       'rank_%o', 'rank_income', 'rank_fs', 'rank_bachelors',

       'rank_high_school', 'rank_no_high_school', 'rank_total_se',

       'rank_total_se_0-4', 'rank_total_se_5-9', 'rank_total_se_10-14',

       'rank_total_se_15-17', 'rank_total_se_18-19', 'rank_total_se_20-24',

       'rank_total_se_25-34', 'rank_total_se_35+']]
beatList = list(df_topComplaints_ranks.beat)

df_topComplaints_ranks['rank_beat'] = df_topComplaints_ranks.beat.apply(lambda x: beatList.index(x)+1)

df_topComplaints_ranks.set_index('beat',inplace = True)

print('TOP 20 COMPLAINTS BEATS MEDIAN RANKINGS: ')

print(df_topComplaints_ranks.median(axis = 0))

print()

print('TOP 20 COMPLAINTS BEATS AVERAGE (MEAN) RANKINGS: ')

print(df_topComplaints_ranks.mean(axis = 0))
lastFile = open("/kaggle/input/cpd-police-beat-demographics/outputBeats_EDA_LAST.txt","r") 
lastFile.readline()

l20_crime = lastFile.readline().split(' ')

l20_crime = l20_crime[:-1]

for i in range (0,20):

    l20_crime[i] = pad0(l20_crime[i])

print(l20_crime)
df_lastCrime = df_beat[df_beat['beat'].isin(l20_crime)]

df_lastCrime.set_index('beat', inplace = True)

df_lastCrime = df_lastCrime.reindex(l20_crime)
df_lastCrime.reset_index(inplace = True)

df_lastCrime
df_lastCrime = df_lastCrime.loc[(df_lastCrime.beat != '1653') & (df_lastCrime.beat != '1652') & (df_lastCrime.beat != '1655') & (df_lastCrime.beat != '0430')]
totPop = 0

wPop = 0

bPop = 0

hPop = 0

mPop = 0

aPop = 0

oPop = 0

for ind in df_lastCrime.index:

    totPop += df_lastCrime['population'][ind]

    wPop += df_lastCrime['num_white'][ind]

    bPop += df_lastCrime['num_black'][ind]

    hPop += df_lastCrime['num_hispanic'][ind]

    mPop += df_lastCrime['num_mixed'][ind]

    aPop += df_lastCrime['num_asian'][ind]

    oPop += df_lastCrime['num_other'][ind]

print('LAST 20 CRIME BEATS RACE BREAKDOWN:\n')

print('Percentage of Population that is White: '+str(wPop/totPop*100)+'%')

print('Percentage of Population that is Black: '+str(bPop/totPop*100)+'%')

print('Percentage of Population that is Hispanic: '+str(hPop/totPop*100)+'%')

print('Percentage of Population that is Asian: '+str(aPop/totPop*100)+'%')

print('Percentage of Population that is Mixed: '+str(mPop/totPop*100)+'%')

print('Percentage of Population that is Other: '+str(oPop/totPop*100)+'%')
tot_MI = 0

tot_FS = 0

for ind in df_lastCrime.index:

    tot_MI += df_lastCrime['med_income'][ind]

    tot_FS += df_lastCrime['pop_food_stamps'][ind]

print('LAST 20 CRIME BEATS INCOME AND FOOD STAMPS BREAKDOWN:\n')

print('Average Median Income: '+str(tot_MI/len(df_lastCrime)))

print('Percentage of Population on Food Stamps: '+str(tot_FS/totPop*100)+'%')
bachelorsPop = df_lastCrime['bachelors'].sum()

HSPop = df_lastCrime['high_school'].sum()

no_HSPop = df_lastCrime['no_high_school'].sum()

print('LAST 20 CRIME BEATS EDUCATIONAL ATTAINMENT BREAKDOWN:\n')

print('Percentage of Population with at least a Bachelor\'s Degree: '+str(bachelorsPop/totPop*100)+'%')

print('Percentage of Population with at most a High School Diploma '+str(HSPop/totPop*100)+'%')

print('Percentage of Population without a High School Diploma '+str(no_HSPop/totPop*100)+'%')
total_se = df_lastCrime['total_se'].sum()

total_se_0to4 = df_lastCrime['se_0-4'].sum()

total_se_5to9 = df_lastCrime['se_5-9'].sum()

total_se_10to14 = df_lastCrime['se_10-14'].sum()

total_se_15to17 = df_lastCrime['se_15-17'].sum()

total_se_18to19 = df_lastCrime['se_18-19'].sum()

total_se_20to24 = df_lastCrime['se_20-24'].sum()

total_se_25to34 = df_lastCrime['se_25-34'].sum()

total_se_35plus = df_lastCrime['se_35+'].sum()

total_0to4 = df_lastCrime['0-4'].sum()

total_5to9 = df_lastCrime['5-9'].sum()

total_10to14 = df_lastCrime['10-14'].sum()

total_15to17 = df_lastCrime['15-17'].sum()

total_18to19 = df_lastCrime['18-19'].sum()

total_20to24 = df_lastCrime['20'].sum() + df_lastCrime['21'].sum() + df_lastCrime['22-24'].sum()

total_25to34 = df_lastCrime['25-29'].sum() + df_lastCrime['30-34'].sum() 

total_youngPop = totPop - (df_lastCrime['25-29'].sum() + df_lastCrime['30-34'].sum()) - (df_lastCrime['20'].sum() + df_lastCrime['21'].sum() + df_lastCrime['22-24'].sum()) - df_lastCrime['18-19'].sum() - df_lastCrime['15-17'].sum() -df_lastCrime['10-14'].sum() -df_lastCrime['5-9'].sum()-df_lastCrime['0-4'].sum()



print('LAST 20 CRIME BEATS SCHOOL ENROLLMENT BREAKDOWN:\n')

print('Percentage of Residents that are Enrolled Students (All): '+str(total_se/totPop*100)+'%')

print('Percentage of Residents that are Enrolled Students (0-4): '+str(total_se_0to4/total_0to4*100)+'%')

print('Percentage of Residents that are Enrolled Students (5-9): '+str(total_se_5to9/total_5to9*100)+'%')

print('Percentage of Residents that are Enrolled Students (10-14): '+str(total_se_10to14/total_10to14*100)+'%')

print('Percentage of Residents that are Enrolled Students (15-17): '+str(total_se_15to17/total_15to17*100)+'%')

print('Percentage of Residents that are Enrolled Students (18-19): '+str(total_se_18to19/total_18to19*100)+'%')

print('Percentage of Residents that are Enrolled Students (20-24): '+str(total_se_20to24/total_20to24*100)+'%')

print('Percentage of Residents that are Enrolled Students (25-34): '+str(total_se_25to34/total_25to34*100)+'%')

print('Percentage of Residents that are Enrolled Students (35+): '+str(total_se_35plus/total_youngPop*100)+'%')
minors = df_lastCrime['<=21'].sum()

twenties = df_lastCrime['22-29'].sum()

thirties = df_lastCrime['30-39'].sum()

forties = df_lastCrime['40-49'].sum()

fifties = df_lastCrime['50-59'].sum()

sixties = df_lastCrime['60-64'].sum()

seniors = df_lastCrime['65+'].sum()

print('LAST 20 CRIME BEATS AGE BREAKDOWN:\n')

print('Percentage of Residents that are Minors (<=21): '+str(minors/totPop*100)+'%')

print('Percentage of Residents that are in their Twenties: '+str(twenties/totPop*100)+'%')

print('Percentage of Residents that are in their Thirties: '+str(thirties/totPop*100)+'%')

print('Percentage of Residents that are in their Forties: '+str(forties/totPop*100)+'%')

print('Percentage of Residents that are in their Fifties: '+str(fifties/totPop*100)+'%')

print('Percentage of Residents that are in between 60-64: '+str(sixties/totPop*100)+'%')

print('Percentage of Residents that are Seniors (65+): '+str(seniors/totPop*100)+'%')
df_lastCrime_ranks = df_lastCrime[['beat','rank_%w', 'rank_%b', 'rank_%h', 'rank_%a', 'rank_%m',

       'rank_%o', 'rank_income', 'rank_fs', 'rank_bachelors',

       'rank_high_school', 'rank_no_high_school', 'rank_total_se',

       'rank_total_se_0-4', 'rank_total_se_5-9', 'rank_total_se_10-14',

       'rank_total_se_15-17', 'rank_total_se_18-19', 'rank_total_se_20-24',

       'rank_total_se_25-34', 'rank_total_se_35+']]
beatList = list(df_lastCrime_ranks.beat)

df_lastCrime_ranks['rank_beat'] = df_lastCrime_ranks.beat.apply(lambda x: beatList.index(x)+1)

df_lastCrime_ranks.set_index('beat',inplace = True)

print('LAST 20 CRIME BEATS MEDIAN RANKINGS: ')

print(df_lastCrime_ranks.median(axis = 0))

print()

print('LAST 20 CRIME BEATS AVERAGE (MEAN) RANKINGS: ')

print(df_lastCrime_ranks.mean(axis = 0))

df_lastCrime_ranks
lastFile.readline()

l20_arrests = lastFile.readline().split(' ')

l20_arrests = l20_arrests[:-1]

for i in range (0,20):

    l20_arrests[i] = pad0(l20_arrests[i])

print(l20_arrests)
df_lastArrest = df_beat[df_beat['beat'].isin(l20_arrests)]

df_lastArrest.set_index('beat', inplace = True)

df_lastArrest = df_lastArrest.reindex(l20_arrests)
df_lastArrest.reset_index(inplace = True)
df_lastArrest = df_lastArrest.loc[(df_lastArrest.beat != '1653') & (df_lastArrest.beat != '1652') & (df_lastArrest.beat != '1655') & (df_lastArrest.beat != '0430')]

df_lastArrest
totPop = 0

wPop = 0

bPop = 0

hPop = 0

mPop = 0

aPop = 0

oPop = 0

for ind in df_lastArrest.index:

    totPop += df_lastArrest['population'][ind]

    wPop += df_lastArrest['num_white'][ind]

    bPop += df_lastArrest['num_black'][ind]

    hPop += df_lastArrest['num_hispanic'][ind]

    mPop += df_lastArrest['num_mixed'][ind]

    aPop += df_lastArrest['num_asian'][ind]

    oPop += df_lastArrest['num_other'][ind]

print('LAST 20 ARREST BEATS RACE BREAKDOWN:\n')

print('Percentage of Population that is White: '+str(wPop/totPop*100)+'%')

print('Percentage of Population that is Black: '+str(bPop/totPop*100)+'%')

print('Percentage of Population that is Hispanic: '+str(hPop/totPop*100)+'%')

print('Percentage of Population that is Asian: '+str(aPop/totPop*100)+'%')

print('Percentage of Population that is Mixed: '+str(mPop/totPop*100)+'%')

print('Percentage of Population that is Other: '+str(oPop/totPop*100)+'%')

tot_MI = 0

tot_FS = 0

for ind in df_lastArrest.index:

    tot_MI += df_lastArrest['med_income'][ind]

    tot_FS += df_lastArrest['pop_food_stamps'][ind]

print('LAST 20 ARREST BEATS INCOME AND FOOD STAMPS BREAKDOWN:\n')

print('Average Median Income: '+str(tot_MI/20.0))

print('Percentage of Population on Food Stamps: '+str(tot_FS/totPop*100)+'%')
bachelorsPop = df_lastArrest['bachelors'].sum()

HSPop = df_lastArrest['high_school'].sum()

no_HSPop = df_lastArrest['no_high_school'].sum()

print('LAST 20 ARRESTS BEATS EDUCATIONAL ATTAINMENT BREAKDOWN:\n')

print('Percentage of Population with at least a Bachelor\'s Degree: '+str(bachelorsPop/totPop*100)+'%')

print('Percentage of Population with at most a High School Diploma '+str(HSPop/totPop*100)+'%')

print('Percentage of Population without a High School Diploma '+str(no_HSPop/totPop*100)+'%')
total_se = df_lastArrest['total_se'].sum()

total_se_0to4 = df_lastArrest['se_0-4'].sum()

total_se_5to9 = df_lastArrest['se_5-9'].sum()

total_se_10to14 = df_lastArrest['se_10-14'].sum()

total_se_15to17 = df_lastArrest['se_15-17'].sum()

total_se_18to19 = df_lastArrest['se_18-19'].sum()

total_se_20to24 = df_lastArrest['se_20-24'].sum()

total_se_25to34 = df_lastArrest['se_25-34'].sum()

total_se_35plus = df_lastArrest['se_35+'].sum()

total_0to4 = df_lastArrest['0-4'].sum()

total_5to9 = df_lastArrest['5-9'].sum()

total_10to14 = df_lastArrest['10-14'].sum()

total_15to17 = df_lastArrest['15-17'].sum()

total_18to19 = df_lastArrest['18-19'].sum()

total_20to24 = df_lastArrest['20'].sum() + df_lastArrest['21'].sum() + df_lastArrest['22-24'].sum()

total_25to34 = df_lastArrest['25-29'].sum() + df_lastArrest['30-34'].sum() 

total_youngPop = totPop - (df_lastArrest['25-29'].sum() + df_lastArrest['30-34'].sum()) - (df_lastArrest['20'].sum() + df_lastArrest['21'].sum() + df_lastArrest['22-24'].sum()) - df_lastArrest['18-19'].sum() - df_lastArrest['15-17'].sum() -df_lastArrest['10-14'].sum() -df_lastArrest['5-9'].sum()-df_lastArrest['0-4'].sum()



print('LAST 20 ARRESTS BEATS SCHOOL ENROLLMENT BREAKDOWN:\n')

print('Percentage of Residents that are Enrolled Students (All): '+str(total_se/totPop*100)+'%')

print('Percentage of Residents that are Enrolled Students (0-4): '+str(total_se_0to4/total_0to4*100)+'%')

print('Percentage of Residents that are Enrolled Students (5-9): '+str(total_se_5to9/total_5to9*100)+'%')

print('Percentage of Residents that are Enrolled Students (10-14): '+str(total_se_10to14/total_10to14*100)+'%')

print('Percentage of Residents that are Enrolled Students (15-17): '+str(total_se_15to17/total_15to17*100)+'%')

print('Percentage of Residents that are Enrolled Students (18-19): '+str(total_se_18to19/total_18to19*100)+'%')

print('Percentage of Residents that are Enrolled Students (20-24): '+str(total_se_20to24/total_20to24*100)+'%')

print('Percentage of Residents that are Enrolled Students (25-34): '+str(total_se_25to34/total_25to34*100)+'%')

print('Percentage of Residents that are Enrolled Students (35+): '+str(total_se_35plus/total_youngPop*100)+'%')
minors = df_lastArrest['<=21'].sum()

twenties = df_lastArrest['22-29'].sum()

thirties = df_lastArrest['30-39'].sum()

forties = df_lastArrest['40-49'].sum()

fifties = df_lastArrest['50-59'].sum()

sixties = df_lastArrest['60-64'].sum()

seniors = df_lastArrest['65+'].sum()

print('LAST 20 ARRESTS BEATS AGE BREAKDOWN:\n')

print('Percentage of Residents that are Minors (<=21): '+str(minors/totPop*100)+'%')

print('Percentage of Residents that are in their Twenties: '+str(twenties/totPop*100)+'%')

print('Percentage of Residents that are in their Thirties: '+str(thirties/totPop*100)+'%')

print('Percentage of Residents that are in their Forties: '+str(forties/totPop*100)+'%')

print('Percentage of Residents that are in their Fifties: '+str(fifties/totPop*100)+'%')

print('Percentage of Residents that are in between 60-64: '+str(sixties/totPop*100)+'%')

print('Percentage of Residents that are Seniors (65+): '+str(seniors/totPop*100)+'%')
df_lastArrest_ranks = df_lastArrest[['beat','rank_%w', 'rank_%b', 'rank_%h', 'rank_%a', 'rank_%m',

       'rank_%o', 'rank_income', 'rank_fs', 'rank_bachelors',

       'rank_high_school', 'rank_no_high_school', 'rank_total_se',

       'rank_total_se_0-4', 'rank_total_se_5-9', 'rank_total_se_10-14',

       'rank_total_se_15-17', 'rank_total_se_18-19', 'rank_total_se_20-24',

       'rank_total_se_25-34', 'rank_total_se_35+']]
beatList = list(df_lastArrest_ranks.beat)

df_lastArrest_ranks['rank_beat'] = df_lastArrest_ranks.beat.apply(lambda x: beatList.index(x)+1)

df_lastArrest_ranks.set_index('beat',inplace = True)

print('LAST 20 ARREST BEATS MEDIAN RANKINGS: ')

print(df_lastArrest_ranks.median(axis = 0))

print()

print('LAST 20 ARREST BEATS AVERAGE (MEAN) RANKINGS: ')

print(df_lastArrest_ranks.mean(axis = 0))

df_lastArrest_ranks
lastFile.readline()

l20_ratio = lastFile.readline().split(' ')

l20_ratio = l20_ratio[:-1]

for i in range (0,20):

    l20_ratio[i] = pad0(l20_ratio[i])

print(l20_ratio)
df_lastRatio = df_beat[df_beat['beat'].isin(l20_ratio)]

df_lastRatio.set_index('beat', inplace = True)

df_lastRatio = df_lastRatio.reindex(l20_ratio)
df_lastRatio.reset_index(inplace = True)

df_lastRatio
df_lastRatio = df_lastRatio.loc[(df_lastRatio.beat != '0430') & (df_lastRatio.beat != '2333') & (df_lastRatio.beat != '1323') ]

df_lastRatio
totPop = 0

wPop = 0

bPop = 0

hPop = 0

mPop = 0

aPop = 0

oPop = 0

for ind in df_lastRatio.index:

    totPop += df_lastRatio['population'][ind]

    wPop += df_lastRatio['num_white'][ind]

    bPop += df_lastRatio['num_black'][ind]

    hPop += df_lastRatio['num_hispanic'][ind]

    mPop += df_lastRatio['num_mixed'][ind]

    aPop += df_lastRatio['num_asian'][ind]

    oPop += df_lastRatio['num_other'][ind]

print('LAST 20 ARRESTS:CRIME BEATS RACE BREAKDOWN:\n')

print('Percentage of Population that is White: '+str(wPop/totPop*100)+'%')

print('Percentage of Population that is Black: '+str(bPop/totPop*100)+'%')

print('Percentage of Population that is Hispanic: '+str(hPop/totPop*100)+'%')

print('Percentage of Population that is Asian: '+str(aPop/totPop*100)+'%')

print('Percentage of Population that is Mixed: '+str(mPop/totPop*100)+'%')

print('Percentage of Population that is Other: '+str(oPop/totPop*100)+'%')
tot_MI = 0

tot_FS = 0

for ind in df_lastRatio.index:

    tot_MI += df_lastRatio['med_income'][ind]

    tot_FS += df_lastRatio['pop_food_stamps'][ind]

print('LAST 20 ARREST:CRIME BEATS INCOME AND FOOD STAMPS BREAKDOWN:\n')

print('Average Median Income: '+str(tot_MI/20.0))

print('Percentage of Population on Food Stamps: '+str(tot_FS/totPop*100)+'%')
bachelorsPop = df_lastRatio['bachelors'].sum()

HSPop = df_lastRatio['high_school'].sum()

no_HSPop = df_lastRatio['no_high_school'].sum()

print('LAST 20 ARRESTS:CRIME BEATS EDUCATIONAL ATTAINMENT BREAKDOWN:\n')

print('Percentage of Population with at least a Bachelor\'s Degree: '+str(bachelorsPop/totPop*100)+'%')

print('Percentage of Population with at most a High School Diploma '+str(HSPop/totPop*100)+'%')

print('Percentage of Population without a High School Diploma '+str(no_HSPop/totPop*100)+'%')
total_se = df_lastRatio['total_se'].sum()

total_se_0to4 = df_lastRatio['se_0-4'].sum()

total_se_5to9 = df_lastRatio['se_5-9'].sum()

total_se_10to14 = df_lastRatio['se_10-14'].sum()

total_se_15to17 = df_lastRatio['se_15-17'].sum()

total_se_18to19 = df_lastRatio['se_18-19'].sum()

total_se_20to24 = df_lastRatio['se_20-24'].sum()

total_se_25to34 = df_lastRatio['se_25-34'].sum()

total_se_35plus = df_lastRatio['se_35+'].sum()

total_0to4 = df_lastRatio['0-4'].sum()

total_5to9 = df_lastRatio['5-9'].sum()

total_10to14 = df_lastRatio['10-14'].sum()

total_15to17 = df_lastRatio['15-17'].sum()

total_18to19 = df_lastRatio['18-19'].sum()

total_20to24 = df_lastRatio['20'].sum() + df_lastRatio['21'].sum() + df_lastRatio['22-24'].sum()

total_25to34 = df_lastRatio['25-29'].sum() + df_lastRatio['30-34'].sum() 

total_youngPop = totPop - (df_lastRatio['25-29'].sum() + df_lastRatio['30-34'].sum()) - (df_lastRatio['20'].sum() + df_lastRatio['21'].sum() + df_lastRatio['22-24'].sum()) - df_lastRatio['18-19'].sum() - df_lastRatio['15-17'].sum() -df_lastRatio['10-14'].sum() -df_lastRatio['5-9'].sum()-df_lastRatio['0-4'].sum()



print('LAST 20 ARRESTS:CRIME BEATS SCHOOL ENROLLMENT BREAKDOWN:\n')

print('Percentage of Residents that are Enrolled Students (All): '+str(total_se/totPop*100)+'%')

print('Percentage of Residents that are Enrolled Students (0-4): '+str(total_se_0to4/total_0to4*100)+'%')

print('Percentage of Residents that are Enrolled Students (5-9): '+str(total_se_5to9/total_5to9*100)+'%')

print('Percentage of Residents that are Enrolled Students (10-14): '+str(total_se_10to14/total_10to14*100)+'%')

print('Percentage of Residents that are Enrolled Students (15-17): '+str(total_se_15to17/total_15to17*100)+'%')

print('Percentage of Residents that are Enrolled Students (18-19): '+str(total_se_18to19/total_18to19*100)+'%')

print('Percentage of Residents that are Enrolled Students (20-24): '+str(total_se_20to24/total_20to24*100)+'%')

print('Percentage of Residents that are Enrolled Students (25-34): '+str(total_se_25to34/total_25to34*100)+'%')

print('Percentage of Residents that are Enrolled Students (35+): '+str(total_se_35plus/total_youngPop*100)+'%')
minors = df_lastRatio['<=21'].sum()

twenties = df_lastRatio['22-29'].sum()

thirties = df_lastRatio['30-39'].sum()

forties = df_lastRatio['40-49'].sum()

fifties = df_lastRatio['50-59'].sum()

sixties = df_lastRatio['60-64'].sum()

seniors = df_lastRatio['65+'].sum()

print('LAST 20 ARRESTS:CRIME BEATS AGE BREAKDOWN:\n')

print('Percentage of Residents that are Minors (<=21): '+str(minors/totPop*100)+'%')

print('Percentage of Residents that are in their Twenties: '+str(twenties/totPop*100)+'%')

print('Percentage of Residents that are in their Thirties: '+str(thirties/totPop*100)+'%')

print('Percentage of Residents that are in their Forties: '+str(forties/totPop*100)+'%')

print('Percentage of Residents that are in their Fifties: '+str(fifties/totPop*100)+'%')

print('Percentage of Residents that are in between 60-64: '+str(sixties/totPop*100)+'%')

print('Percentage of Residents that are Seniors (65+): '+str(seniors/totPop*100)+'%')
df_lastRatio_ranks = df_lastRatio[['beat','rank_%w', 'rank_%b', 'rank_%h', 'rank_%a', 'rank_%m',

       'rank_%o', 'rank_income', 'rank_fs', 'rank_bachelors',

       'rank_high_school', 'rank_no_high_school', 'rank_total_se',

       'rank_total_se_0-4', 'rank_total_se_5-9', 'rank_total_se_10-14',

       'rank_total_se_15-17', 'rank_total_se_18-19', 'rank_total_se_20-24',

       'rank_total_se_25-34', 'rank_total_se_35+']]
beatList = list(df_lastRatio_ranks.beat)

df_lastRatio_ranks['rank_beat'] = df_lastRatio_ranks.beat.apply(lambda x: beatList.index(x)+1)

df_lastRatio_ranks.set_index('beat',inplace = True)

print('LAST 20 CRIME BEATS MEDIAN RANKINGS: ')

print(df_lastRatio_ranks.median(axis = 0))

print()

print('LAST 20 CRIME BEATS AVERAGE (MEAN) RANKINGS: ')

print(df_lastRatio_ranks.mean(axis = 0))
lastFile.readline()

l20_injuries = lastFile.readline().split(' ')

l20_injuries = l20_injuries[:-1]

for i in range (0,20):

    l20_injuries[i] = pad0(l20_injuries[i])

print(l20_injuries)
df_lastInjuries = df_beat[df_beat['beat'].isin(l20_injuries)]

df_lastInjuries.set_index('beat', inplace = True)

df_lastInjuries = df_lastInjuries.reindex(l20_injuries)
df_lastInjuries.reset_index(inplace = True)

df_lastInjuries
df_lastInjuries = df_lastInjuries.loc[(df_lastInjuries.beat != '2133') & (df_lastInjuries.beat != '4100') & (df_lastInjuries.beat != '1653') & (df_lastInjuries.beat != '2131') ]

df_lastInjuries
totPop = 0

wPop = 0

bPop = 0

hPop = 0

mPop = 0

aPop = 0

oPop = 0

for ind in df_lastInjuries.index:

    totPop += df_lastInjuries['population'][ind]

    wPop += df_lastInjuries['num_white'][ind]

    bPop += df_lastInjuries['num_black'][ind]

    hPop += df_lastInjuries['num_hispanic'][ind]

    mPop += df_lastInjuries['num_mixed'][ind]

    aPop += df_lastInjuries['num_asian'][ind]

    oPop += df_lastInjuries['num_other'][ind]

print('LAST 20 INJURIES BEATS RACE BREAKDOWN:\n')

print('Percentage of Population that is White: '+str(wPop/totPop*100)+'%')

print('Percentage of Population that is Black: '+str(bPop/totPop*100)+'%')

print('Percentage of Population that is Hispanic: '+str(hPop/totPop*100)+'%')

print('Percentage of Population that is Asian: '+str(aPop/totPop*100)+'%')

print('Percentage of Population that is Mixed: '+str(mPop/totPop*100)+'%')

print('Percentage of Population that is Other: '+str(oPop/totPop*100)+'%')
tot_MI = 0

tot_FS = 0

for ind in df_lastInjuries.index:

    tot_MI += df_lastInjuries['med_income'][ind]

    tot_FS += df_lastInjuries['pop_food_stamps'][ind]

print('LAST 20 INJURY BEATS INCOME AND FOOD STAMPS BREAKDOWN:\n')

print('Average Median Income: '+str(tot_MI/20.0))

print('Percentage of Population on Food Stamps: '+str(tot_FS/totPop*100)+'%')
bachelorsPop = df_lastInjuries['bachelors'].sum()

HSPop = df_lastInjuries['high_school'].sum()

no_HSPop = df_lastInjuries['no_high_school'].sum()

print('LAST 20 INJURY BEATS EDUCATIONAL ATTAINMENT BREAKDOWN:\n')

print('Percentage of Population with at least a Bachelor\'s Degree: '+str(bachelorsPop/totPop*100)+'%')

print('Percentage of Population with at most a High School Diploma '+str(HSPop/totPop*100)+'%')

print('Percentage of Population without a High School Diploma '+str(no_HSPop/totPop*100)+'%')
total_se = df_lastInjuries['total_se'].sum()

total_se_0to4 = df_lastInjuries['se_0-4'].sum()

total_se_5to9 = df_lastInjuries['se_5-9'].sum()

total_se_10to14 = df_lastInjuries['se_10-14'].sum()

total_se_15to17 = df_lastInjuries['se_15-17'].sum()

total_se_18to19 = df_lastInjuries['se_18-19'].sum()

total_se_20to24 = df_lastInjuries['se_20-24'].sum()

total_se_25to34 = df_lastInjuries['se_25-34'].sum()

total_se_35plus = df_lastInjuries['se_35+'].sum()

total_0to4 = df_lastInjuries['0-4'].sum()

total_5to9 = df_lastInjuries['5-9'].sum()

total_10to14 = df_lastInjuries['10-14'].sum()

total_15to17 = df_lastInjuries['15-17'].sum()

total_18to19 = df_lastInjuries['18-19'].sum()

total_20to24 = df_lastInjuries['20'].sum() + df_lastInjuries['21'].sum() + df_lastInjuries['22-24'].sum()

total_25to34 = df_lastInjuries['25-29'].sum() + df_lastInjuries['30-34'].sum() 

total_youngPop = totPop - (df_lastInjuries['25-29'].sum() + df_lastInjuries['30-34'].sum()) - (df_lastInjuries['20'].sum() + df_lastInjuries['21'].sum() + df_lastInjuries['22-24'].sum()) - df_lastInjuries['18-19'].sum() - df_lastInjuries['15-17'].sum() -df_lastInjuries['10-14'].sum() -df_lastInjuries['5-9'].sum()-df_lastInjuries['0-4'].sum()



print('LAST 20 INJURY BEATS SCHOOL ENROLLMENT BREAKDOWN:\n')

print('Percentage of Residents that are Enrolled Students (All): '+str(total_se/totPop*100)+'%')

print('Percentage of Residents that are Enrolled Students (0-4): '+str(total_se_0to4/total_0to4*100)+'%')

print('Percentage of Residents that are Enrolled Students (5-9): '+str(total_se_5to9/total_5to9*100)+'%')

print('Percentage of Residents that are Enrolled Students (10-14): '+str(total_se_10to14/total_10to14*100)+'%')

print('Percentage of Residents that are Enrolled Students (15-17): '+str(total_se_15to17/total_15to17*100)+'%')

print('Percentage of Residents that are Enrolled Students (18-19): '+str(total_se_18to19/total_18to19*100)+'%')

print('Percentage of Residents that are Enrolled Students (20-24): '+str(total_se_20to24/total_20to24*100)+'%')

print('Percentage of Residents that are Enrolled Students (25-34): '+str(total_se_25to34/total_25to34*100)+'%')

print('Percentage of Residents that are Enrolled Students (35+): '+str(total_se_35plus/total_youngPop*100)+'%')
minors = df_lastInjuries['<=21'].sum()

twenties = df_lastInjuries['22-29'].sum()

thirties = df_lastInjuries['30-39'].sum()

forties = df_lastInjuries['40-49'].sum()

fifties = df_lastInjuries['50-59'].sum()

sixties = df_lastInjuries['60-64'].sum()

seniors = df_lastInjuries['65+'].sum()

print('LAST 20 INJURIES BEATS AGE BREAKDOWN:\n')

print('Percentage of Residents that are Minors (<=21): '+str(minors/totPop*100)+'%')

print('Percentage of Residents that are in their Twenties: '+str(twenties/totPop*100)+'%')

print('Percentage of Residents that are in their Thirties: '+str(thirties/totPop*100)+'%')

print('Percentage of Residents that are in their Forties: '+str(forties/totPop*100)+'%')

print('Percentage of Residents that are in their Fifties: '+str(fifties/totPop*100)+'%')

print('Percentage of Residents that are in between 60-64: '+str(sixties/totPop*100)+'%')

print('Percentage of Residents that are Seniors (65+): '+str(seniors/totPop*100)+'%')
df_lastInjuries_ranks = df_lastInjuries[['beat','rank_%w', 'rank_%b', 'rank_%h', 'rank_%a', 'rank_%m',

       'rank_%o', 'rank_income', 'rank_fs', 'rank_bachelors',

       'rank_high_school', 'rank_no_high_school', 'rank_total_se',

       'rank_total_se_0-4', 'rank_total_se_5-9', 'rank_total_se_10-14',

       'rank_total_se_15-17', 'rank_total_se_18-19', 'rank_total_se_20-24',

       'rank_total_se_25-34', 'rank_total_se_35+']]
beatList = list(df_lastInjuries_ranks.beat)

df_lastInjuries_ranks['rank_beat'] = df_lastInjuries_ranks.beat.apply(lambda x: beatList.index(x)+1)

df_lastInjuries_ranks.set_index('beat',inplace = True)

print('LAST 20 INJURY BEATS MEDIAN RANKINGS: ')

print(df_lastInjuries_ranks.median(axis = 0))

print()

print('LAST 20 INJURY BEATS AVERAGE (MEAN) RANKINGS: ')

print(df_lastInjuries_ranks.mean(axis = 0))
lastFile.readline()

l20_complaints  = lastFile.readline().split(' ')

l20_complaints  = l20_complaints[:-1]

for i in range (0,20):

    l20_complaints[i] = pad0(l20_complaints[i])

print(l20_complaints)

lastFile.close()
df_lastComplaints = df_beat[df_beat['beat'].isin(l20_complaints)]

df_lastComplaints.set_index('beat', inplace = True)

df_lastComplaints = df_lastComplaints.reindex(l20_complaints)

df_lastComplaints.reset_index(inplace = True)

df_lastComplaints
df_lastComplaints = df_lastComplaints.loc[(df_lastComplaints['beat'] != '1653') & (df_lastComplaints['beat'] != '1652') & (df_lastComplaints['beat'] != '1655')]

df_lastComplaints
totPop = 0

wPop = 0

bPop = 0

hPop = 0

mPop = 0

aPop = 0

oPop = 0

for ind in df_lastComplaints.index:

    totPop += df_lastComplaints['population'][ind]

    wPop += df_lastComplaints['num_white'][ind]

    bPop += df_lastComplaints['num_black'][ind]

    hPop += df_lastComplaints['num_hispanic'][ind]

    mPop += df_lastComplaints['num_mixed'][ind]

    aPop += df_lastComplaints['num_asian'][ind]

    oPop += df_lastComplaints['num_other'][ind]

print('LAST 20 COMPLAINTS BEATS RACE BREAKDOWN:\n')

print('Percentage of Population that is White: '+str(wPop/totPop*100)+'%')

print('Percentage of Population that is Black: '+str(bPop/totPop*100)+'%')

print('Percentage of Population that is Hispanic: '+str(hPop/totPop*100)+'%')

print('Percentage of Population that is Asian: '+str(aPop/totPop*100)+'%')

print('Percentage of Population that is Mixed: '+str(mPop/totPop*100)+'%')

print('Percentage of Population that is Other: '+str(oPop/totPop*100)+'%')
tot_MI = 0

tot_FS = 0

for ind in df_lastComplaints.index:

    tot_MI += df_lastComplaints['med_income'][ind]

    tot_FS += df_lastComplaints['pop_food_stamps'][ind]

print('LAST 20 COMPLAINTS BEATS INCOME AND FOOD STAMPS BREAKDOWN:\n')

print('Average Median Income: '+str(tot_MI/20.0))

print('Percentage of Population on Food Stamps: '+str(tot_FS/totPop*100)+'%')
bachelorsPop = df_lastComplaints['bachelors'].sum()

HSPop = df_lastComplaints['high_school'].sum()

no_HSPop = df_lastComplaints['no_high_school'].sum()

print('LAST 20 COMPLAINTS BEATS EDUCATIONAL ATTAINMENT BREAKDOWN:\n')

print('Percentage of Population with at least a Bachelor\'s Degree: '+str(bachelorsPop/totPop*100)+'%')

print('Percentage of Population with at most a High School Diploma '+str(HSPop/totPop*100)+'%')

print('Percentage of Population without a High School Diploma '+str(no_HSPop/totPop*100)+'%')
total_se = df_lastComplaints['total_se'].sum()

total_se_0to4 = df_lastComplaints['se_0-4'].sum()

total_se_5to9 = df_lastComplaints['se_5-9'].sum()

total_se_10to14 = df_lastComplaints['se_10-14'].sum()

total_se_15to17 = df_lastComplaints['se_15-17'].sum()

total_se_18to19 = df_lastComplaints['se_18-19'].sum()

total_se_20to24 = df_lastComplaints['se_20-24'].sum()

total_se_25to34 = df_lastComplaints['se_25-34'].sum()

total_se_35plus = df_lastComplaints['se_35+'].sum()

total_0to4 = df_lastComplaints['0-4'].sum()

total_5to9 = df_lastComplaints['5-9'].sum()

total_10to14 = df_lastComplaints['10-14'].sum()

total_15to17 = df_lastComplaints['15-17'].sum()

total_18to19 = df_lastComplaints['18-19'].sum()

total_20to24 = df_lastComplaints['20'].sum() + df_lastComplaints['21'].sum() + df_lastComplaints['22-24'].sum()

total_25to34 = df_lastComplaints['25-29'].sum() + df_lastComplaints['30-34'].sum() 

total_youngPop = totPop - (df_lastComplaints['25-29'].sum() + df_lastComplaints['30-34'].sum()) - (df_lastComplaints['20'].sum() + df_lastComplaints['21'].sum() + df_lastComplaints['22-24'].sum()) - df_lastComplaints['18-19'].sum() - df_lastComplaints['15-17'].sum() -df_lastComplaints['10-14'].sum() -df_lastComplaints['5-9'].sum()-df_lastComplaints['0-4'].sum()



print('LAST 20 COMPLAINTS BEATS SCHOOL ENROLLMENT BREAKDOWN:\n')

print('Percentage of Residents that are Enrolled Students (All): '+str(total_se/totPop*100)+'%')

print('Percentage of Residents that are Enrolled Students (0-4): '+str(total_se_0to4/total_0to4*100)+'%')

print('Percentage of Residents that are Enrolled Students (5-9): '+str(total_se_5to9/total_5to9*100)+'%')

print('Percentage of Residents that are Enrolled Students (10-14): '+str(total_se_10to14/total_10to14*100)+'%')

print('Percentage of Residents that are Enrolled Students (15-17): '+str(total_se_15to17/total_15to17*100)+'%')

print('Percentage of Residents that are Enrolled Students (18-19): '+str(total_se_18to19/total_18to19*100)+'%')

print('Percentage of Residents that are Enrolled Students (20-24): '+str(total_se_20to24/total_20to24*100)+'%')

print('Percentage of Residents that are Enrolled Students (25-34): '+str(total_se_25to34/total_25to34*100)+'%')

print('Percentage of Residents that are Enrolled Students (35+): '+str(total_se_35plus/total_youngPop*100)+'%')
minors = df_lastComplaints['<=21'].sum()

twenties = df_lastComplaints['22-29'].sum()

thirties = df_lastComplaints['30-39'].sum()

forties = df_lastComplaints['40-49'].sum()

fifties = df_lastComplaints['50-59'].sum()

sixties = df_lastComplaints['60-64'].sum()

seniors = df_lastComplaints['65+'].sum()

print('LAST 20 COMPLAINTS BEATS AGE BREAKDOWN:\n')

print('Percentage of Residents that are Minors (<=21): '+str(minors/totPop*100)+'%')

print('Percentage of Residents that are in their Twenties: '+str(twenties/totPop*100)+'%')

print('Percentage of Residents that are in their Thirties: '+str(thirties/totPop*100)+'%')

print('Percentage of Residents that are in their Forties: '+str(forties/totPop*100)+'%')

print('Percentage of Residents that are in their Fifties: '+str(fifties/totPop*100)+'%')

print('Percentage of Residents that are in between 60-64: '+str(sixties/totPop*100)+'%')

print('Percentage of Residents that are Seniors (65+): '+str(seniors/totPop*100)+'%')
df_lastComplaints_ranks = df_lastComplaints[['beat','rank_%w', 'rank_%b', 'rank_%h', 'rank_%a', 'rank_%m',

       'rank_%o', 'rank_income', 'rank_fs', 'rank_bachelors',

       'rank_high_school', 'rank_no_high_school', 'rank_total_se',

       'rank_total_se_0-4', 'rank_total_se_5-9', 'rank_total_se_10-14',

       'rank_total_se_15-17', 'rank_total_se_18-19', 'rank_total_se_20-24',

       'rank_total_se_25-34', 'rank_total_se_35+']]
beatList = list(df_lastComplaints_ranks.beat)

df_lastComplaints_ranks['rank_beat'] = df_lastComplaints_ranks.beat.apply(lambda x: beatList.index(x)+1)

df_lastComplaints_ranks.set_index('beat',inplace = True)

print('LAST 20 COMPLAINTS BEATS MEDIAN RANKINGS: ')

print(df_lastComplaints_ranks.median(axis = 0))

print()

print('LAST 20 COMPLAINTS BEATS AVERAGE (MEAN) RANKINGS: ')

print(df_lastComplaints_ranks.mean(axis = 0))