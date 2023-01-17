# import packages
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
# import data
df = pd.read_csv('../input/play_information.csv')
df1 = pd.read_csv('../input/video_review.csv')
df=df.merge(df1, on=['Season_Year','GameKey', 'PlayID'], how='left')
df['concussion'] = df['Turnover_Related'].apply(lambda x: x=='No')
df['PlayDescription'] = df['PlayDescription'].astype(str)

# punt length 
import re 
punt_length = []
for row in df['PlayDescription']:
    match = re.search('punts (\d+)', row)
    if match:
        punt_length.append(match.group(1))
    elif match is None:
        punt_length.append(0)
        
# return length
# to allow for negative or zero return yards , if the ball is not returned we set a default value of -100
return_length = []
for row in df['PlayDescription']:
    match = re.search('for ([-\d]+)', row)
    if match:
        return_length.append(match.group(1))
    elif match is None:
        return_length.append(-100)
        
# fair catch
fair_catch = []
for row in df['PlayDescription']:
    match = re.search('fair catch', row)
    if match:
        fair_catch.append(1)
    elif match is None:
        fair_catch.append(0)

# injury
injury = []
for row in df['PlayDescription']:
    match = re.search('injured', row)
    if match:
        injury.append(1)
    elif match is None:
            injury.append(0)
            
# penalty         
penalty = []
for row in df['PlayDescription']:
    if 'Penalty' in row.split():
        penalty.append(1)
    elif 'PENALTY' in row.split():
        penalty.append(1)
    elif 'Penalty' not in row.split():
        penalty.append(0)
    elif 'PENALTY' not in row.split():
        penalty.append(0)
        

# downed
downed = []
for row in df['PlayDescription']:
    match = re.search('downed', row)
    if match:
        downed.append(1)
    elif match is None:
        downed.append(0)
        
# fumble
fumble = []
for row in df['PlayDescription']:
    match = re.search('FUMBLES', row)
    if match:
        fumble.append(1)
    elif match is None:
        fumble.append(0)

# muff
muff = []
for row in df['PlayDescription']:
    match = re.search('MUFFS', row)
    if match:
        muff.append(1)
    elif match is None:
        muff.append(0)
        
# Touchback
touchback = []
for row in df['PlayDescription']:
    match = re.search('Touchback', row)
    if match:
        touchback.append(1)
    elif match is None:
        touchback.append(0)

# Touchdown
touchdown = []
for row in df['PlayDescription']:
    match = re.search('TOUCHDOWN', row)
    if match:
        touchdown.append(1)
    elif match is None:
        touchdown.append(0)
        
df["punt_length"] = punt_length
df["return_length"] = return_length
df["fair_catch"] = fair_catch
df["injury"] = injury
df["penalty"] = penalty
df["downed"] = downed
df["fumble"] = fumble
df['muff'] = muff
df['touchback'] = touchback
df['touchdown'] = touchdown

df[["punt_length", "return_length"]] = df[["punt_length", "return_length"]].apply(pd.to_numeric)
# check if punt begins on possession team's side of field and what yardline the play starts
df['Side_of_Field'] = df['YardLine'].apply(lambda x: re.sub(r'[0-9]+', '', x))
df['Side_of_Field'] = df['Side_of_Field'].apply(lambda x: x.strip())
df['Own_Side']= (df['Side_of_Field']==df['Poss_Team'])
df['start_yardline'] = df['YardLine'].apply(lambda x: [int(s) for s in x.split() if s.isdigit()][0])
df['deep'] = df['start_yardline']+df['punt_length']
# check if punt goes out of bounds
df['OOB'] = df['PlayDescription'].str.contains('out of bounds', case=False)
punt_risk = df[df['concussion']].shape[0]/float(df.shape[0])

print("Punt plays have a "+ str(round(punt_risk*100,2))+" percent chance of concussion")
df[df['Own_Side']][df['start_yardline']<=35][df['punt_length']>0]['deep'].hist(bins=16, grid=False, figsize=(8,8))
plt.title("Distribution of Starting Field Position of Non-Punting Team After a Deep Punt")
plt.xlabel("Yards Past Punting Team\'s endzone")
df[df['Own_Side']][df['start_yardline']<=35][df['punt_length']>0]['start_yardline'].hist(grid=False, figsize=(6,6))
plt.title('Distribution of Yardline for Deep Punt Plays')
plt.xlabel("Yards from Punting Team\'s Endzone")
punt_length_mean = df[df['Own_Side']][df['start_yardline']<=35][df['punt_length']>0]['punt_length'].mean()
yard_line_mean = df[df['Own_Side']][df['start_yardline']<=35][df['punt_length']>0]['start_yardline'].mean()

print("Average punt length on deep punts: "+ str(round(punt_length_mean))+ ' yards')
print("Average starting yardline on deep punt plays: "+ str(round(yard_line_mean))+' yardline')
non_return_risk = df[(df['fair_catch']==True) | (df['OOB']==True) | (df['touchback']==True)][df['concussion']].shape[0]/float(df[(df['fair_catch']==True) | (df['OOB']==True) | (df['touchback']==True)].shape[0])

print("Punt plays where the punt is not returned (i.e a touchback, fair catch, or out of bounds occurs) have a "+str(round(non_return_risk*100,2))+ " percent risk of concussion")
return_risk = df[df['return_length']>-90][df['concussion']].shape[0]/float(df[df['return_length']>-90].shape[0])

print("Punt plays where the punt is returned have a "+str(round(return_risk*100,2))+" percent risk of concussion")
deep_punt_risk = df[(df['Own_Side']==True) & (df['start_yardline']<=35) & (df['punt_length']>0)][df['concussion']].shape[0]/float(df[(df['Own_Side']==True) & (df['start_yardline']<=35) & (df['punt_length']>0)].shape[0])

deep_punt_risk = round(deep_punt_risk*100,4)

total_deep_punt_returns = df[(df['Own_Side']==True) & (df['start_yardline']<=35) & (df['punt_length']>0) & (df['deep']>50) & (df['return_length']>-90)].shape[0]
deep_punt_return_risk = df[(df['Own_Side']==True) & (df['start_yardline']<=35) & (df['punt_length']>0) & (df['deep']>50) & (df['return_length']>-90)][df['concussion']].shape[0]*100/float(total_deep_punt_returns)



print("Deep punt plays have a "+str(deep_punt_risk)+" percent chance of concussion")
print("Deep punt plays where the punt is returned have a "+str(round(deep_punt_return_risk,2))+" percent chance of concussion")
deep_punt_length= df[(df['Own_Side']==True) & (df['start_yardline']<=35) & (df['punt_length']>0)]['punt_length'].mean()
non_deep_length = df[(df['Own_Side']==True & (df['start_yardline']>35)) | (df['Own_Side']==False)][df['punt_length']>0]['punt_length'].mean()

print("Average punt length for deep punts: "+str(round(deep_punt_length,2))+ " yards")
print("Average punt length for non-deep punts: "+str(round(non_deep_length,2))+ " yards")
deep_punt_return= df[(df['Own_Side']==True) & (df['start_yardline']<=35) & (df['punt_length']>0) &(df['return_length']>-90)].shape[0]/float(df[(df['Own_Side']==True) & (df['start_yardline']<=35) & (df['punt_length']>0)].shape[0])
non_deep_return = df[(df['Own_Side']==True & (df['start_yardline']>35)) | (df['Own_Side']==False)][df['punt_length']>0][df['return_length']>-90].shape[0]/float(df[(df['Own_Side']==True & (df['start_yardline']>35)) | (df['Own_Side']==False)][df['punt_length']>0].shape[0])

print("Deep punts are returned "+str(round(deep_punt_return*100,2))+" percent of the time")
print("Non-deep punts are returned "+ str(round(non_deep_return*100,2))+" percent of the time")

non_deep_risk = df[(df['Own_Side']==True & (df['start_yardline']>35)) | (df['Own_Side']==False)][df['punt_length']>0][df['concussion']].shape[0]/float(df[(df['Own_Side']==True & (df['start_yardline']>35)) | (df['Own_Side']==False)][df['punt_length']>0].shape[0])

non_deep_risk = round(non_deep_risk*100,4)

print("Non-deep punt plays have a "+str(non_deep_risk)+" percent chance of concussion")
super_short_returns = df[(df['Own_Side']==True) & (df['start_yardline']<=35) & (df['punt_length']>0) & (df['deep']>50) & (df['return_length']>-90) & (df['return_length']<=1)].shape[0]
short_returns = df[(df['Own_Side']==True) & (df['start_yardline']<=35) & (df['punt_length']>0) & (df['deep']>50) & (df['return_length']>-90) & (df['return_length']<=5)].shape[0]


print("Number of deep punts returned: "+ str(total_deep_punt_returns))
print("Number of deep punts returned for very short ( <=1 yard) or negative yardage: "+ str(super_short_returns))
print("Number of deep punts returned for short (<= 5 yards) yardage: "+ str(short_returns))
low_est = deep_punt_return_risk*(total_deep_punt_returns-super_short_returns)/100
high_est = deep_punt_return_risk*(total_deep_punt_returns-short_returns)/100

print("Total number of concussions occuring on punt plays: "+str(df[df['concussion']].shape[0])+" concussions")
print("Number of concussions from deep punt plays where the punt is returned: 22 concussions")
print("Estimated number of concussions from deep punt returns with proposed rule change: "+str(int(round(high_est)))+"-"+str(int(round(low_est)))+ " concussions")

print("Estimated percent reduction in concussions on deep punt returns: "+str(round(2*100./22,2))+" to "+str(round(7*100./22,2))+" percent")
print("Estimated overall percent reduction in concussions on all punt plays: "+str(round(2*100./37,2))+" to "+str(round(7*100./37,2))+" percent")
short_punt_rate = df[(df['Own_Side']==True) & (df['start_yardline']<=35) & (df['punt_length']>0) & (df['deep']>50) & (df['return_length']>-90) & (df['return_length']<=5)].shape[0]/float(df[(df['Own_Side']==True) & (df['start_yardline']<=35) & (df['punt_length']>0) & (df['deep']>50) & (df['return_length']>-90)].shape[0])

print("Percentage of deep punts returned no more than 5 yards: "+str(round(short_punt_rate*100,2) )+ "%")
penalty_rate = df[df['penalty']==1].shape[0]/float(df.shape[0])

print("A penalty occurs on "+str(round(penalty_rate*100))+" percent of all punt plays")
print("A penalty occurs on "+str(round(3484*100./45840))+" percent of all plays (punt and non-punt)")