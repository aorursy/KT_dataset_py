import pandas as pd
import numpy as np

import scipy.stats as stats

#visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

pd.set_option("display.max_columns",None)
# importing dataset
shot = pd.read_csv('../input/PoliceKillingsUS.csv', encoding = "ISO-8859-1")
income = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding = "ISO-8859-1")
poverty = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding = "ISO-8859-1")
education = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding = "ISO-8859-1")
racedist = pd.read_csv('../input/ShareRaceByCity.csv', encoding = "ISO-8859-1")
#cleaning formats and similar

#Shootings
shot.date = pd.to_datetime(shot.date)
shot.insert(3, 'year', pd.DatetimeIndex(shot.date).year)
shot.insert(4, 'month', pd.DatetimeIndex(shot.date).month)
shot.insert(5, 'day', pd.DatetimeIndex(shot.date).day)
shot.insert(6, 'dayofweek', pd.DatetimeIndex(shot.date).weekday_name)

#Race shares
racedist.rename(columns = {'Geographic area':'Geographic Area'}, inplace = True)
racedist.share_asian = racedist.share_asian.replace("(X)", np.nan)
racedist.share_black = racedist.share_black.replace("(X)", np.nan)
racedist.share_hispanic = racedist.share_hispanic.replace("(X)", np.nan)
racedist.share_native_american = racedist.share_native_american.replace("(X)", np.nan)
racedist.share_white = racedist.share_white.replace("(X)", np.nan)
racedist.share_asian = pd.to_numeric(racedist.share_asian)
racedist.share_black = pd.to_numeric(racedist.share_black)
racedist.share_hispanic = pd.to_numeric(racedist.share_hispanic)
racedist.share_native_american = pd.to_numeric(racedist.share_native_american)
racedist.share_white = pd.to_numeric(racedist.share_white)

#Incomes
income['Median Income'] = income['Median Income'].replace("(X)", np.nan)
income['Median Income'] = income['Median Income'].replace("-", np.nan)
income['Median Income'] = income['Median Income'].replace("2,500-", "2500")
income['Median Income'] = income['Median Income'].replace("250,000+", "250000")
income['Median Income'] = pd.to_numeric(income['Median Income'])

#Poverty rate
poverty.poverty_rate = poverty.poverty_rate.replace("-", np.nan)
poverty['poverty_rate'] = pd.to_numeric(poverty['poverty_rate'])

#Education rate
education.percent_completed_hs = education.percent_completed_hs.replace("-", np.nan)
education['percent_completed_hs']  = pd.to_numeric(education['percent_completed_hs'])
short_range = ['hammer','pick-axe','glass shard','box cutter','sharp object', 'meat cleaver',
              'stapler', 'chain saw', 'metal object', 'bayonet', 'baton', 'tire iron', 
               'baseball bat and fireplace poker','machete', 'knife','garden tool','pipe',
              'straight edge razor','blunt object','ax','scissors','hatchet and gun','pole and knife',
              'hatchet','carjack','lawn mower blade','metal hand tool','beer bottle','metal stick',
              'piece of wood','screwdriver']

long_range = ['Taser','gun and knife', 'crossbow', 'gun','bean-bag gun','guns and explosives',
              'machete and gun','fireworks']

mid_range = ['shovel', 'pitchfork', 'metal pipe',"contractor's level", 'pole', 'crowbar', 'flagpole',
            'rock', 'oar', 'metal pole', 'chain','brick', 'metal rake',  'sword','spear','baseball bat']

vehicle = ['motorcycle', 'vehicle'] 

unknown = ['unknown weapon', 'hand torch', 'toy weapon', 'cordless drill', 'undetermined', 'flashlight',
           'nail gun']

unarmed = ['unarmed']

shot.loc[shot.armed.isin(short_range), 'Armed_class'] = 'Short range weapon'
shot.loc[shot.armed.isin(long_range), 'Armed_class'] = 'Long range weapon'
shot.loc[shot.armed.isin(mid_range), 'Armed_class'] = 'Mid range weapon'
shot.loc[shot.armed.isin(vehicle), 'Armed_class'] = 'Unknown'
shot.loc[shot.armed.isin(unknown), 'Armed_class'] = 'Unknown'
shot.loc[shot.armed.isin(unarmed), 'Armed_class'] = 'Unarmed'

shot.Armed_class = shot.Armed_class.fillna('Unknown')

shot.loc[shot.flee == 'Not fleeing', 'Fleeing_class'] = 'Not fleeing'
shot.loc[shot.flee != 'Not fleeing', 'Fleeing_class'] = 'Fleeing'

shot.loc[shot.threat_level == 'attack', 'Threat_class'] = 'Attack'
shot.loc[shot.threat_level != 'attack', 'Threat_class'] = 'Not attack'
shot.info()
shot[shot.race.isnull()].year.value_counts()
shot[(shot.race.isnull()) & (shot.year == 2017)].month.value_counts()
shot = shot[shot.date < '2017-06-01']
shot.info()
# Number of deadly shootings per month over time
shot[['year', 'month', 'name']].groupby(['year', 'month']).size().plot(figsize=(15,5))
plt.title("Number of deadly shootings over time")
shot[shot.race == "W"][['year', 
                        'month', 
                        'name']].groupby(['year', 'month']).size().plot(figsize=(15,5), 
                                                                                         label="W")
shot[shot.race != "W"][['year', 
                        'month', 
                        'name']].groupby(['year', 'month']).size().plot(figsize=(15,5),
                                                                                        label="Non-W")
plt.title("Deadly shootings segmented by race over time")
plt.legend()
shot[shot.gender == "M"][['year', 
                          'month', 
                          'name']].groupby(['year', 'month']).size().plot(figsize=(15,5), label="M")
shot[shot.gender == "F"][['year', 
                          'month', 
                          'name']].groupby(['year', 'month']).size().plot(figsize=(15,5), label="F")
plt.title("Deadly shootings segmented by gender over time")
plt.legend()
shot[shot.body_camera == True][['year', 
                                'month', 
                                'name']].groupby(['year', 'month']).size().plot(figsize=(15,5), label="bodyCam")
shot[shot.body_camera != True][['year', 
                                'month', 
                                'name']].groupby(['year', 'month']).size().plot(figsize=(15,5), label="NoCam")
plt.title("Deadly shootings segmented by body cam over time")
plt.legend()
shot[shot.Threat_class == "Attack"][['year', 
                                     'month', 
                                     'name']].groupby(['year', 
                                                       'month']).size().plot(figsize=(15,5), label="Attack")
shot[shot.Threat_class != 'Attack'][['year', 
                                     'month', 
                                     'name']].groupby(['year', 
                                                       'month']).size().plot(figsize=(15,5), label="NoAttack")
plt.title("Deadly shootings segmented by threat over time")

plt.legend()
fil = (shot.Armed_class != "Unknown") & (shot.Armed_class != "Unarmed")
shot[fil][['year', 
           'month', 
           'name']].groupby(['year', 
                             'month']).size().plot(figsize=(15,5), label="Armed")
shot[shot.Armed_class == "Unknown"][['year', 
                                     'month', 
                                     'name']].groupby(['year', 
                                                       'month']).size().plot(figsize=(15,5), label="Unknown")
shot[shot.Armed_class == "Unarmed"][['year', 
                                     'month', 
                                     'name']].groupby(['year', 
                                                       'month']).size().plot(figsize=(15,5), label="Unarmed")
plt.title("Deadly shootings segmented by weapon over time")

plt.legend()
shot[shot.signs_of_mental_illness == True][['year', 
                                            'month', 
                                            'name']].groupby(['year',
                                                              'month']).size().plot(figsize=(15,5), label="Signs")
shot[shot.signs_of_mental_illness != True][['year', 
                                            'month', 
                                            'name']].groupby(['year',
                                                              'month']).size().plot(figsize=(15,5), label="NoSigns")
plt.title("Deadly shootings segmented by signs of mental illness over time")
plt.legend()
def segm_target(var, target):
    count = shot[[var, target]].groupby([var], as_index=True).count()
    count.columns = ['Count']
    mean = shot[[var, target]].groupby([var], as_index=True).mean()
    mean.columns = ['Mean']
    ma = shot[[var, target]].groupby([var], as_index=True).max()
    ma.columns = ['Max']
    mi = shot[[var, target]].groupby([var], as_index=True).min()
    mi.columns = ['Min']
    median = shot[[var, target]].groupby([var], as_index=True).median()
    median.columns = ['Median']
    std = shot[[var, target]].groupby([var], as_index=True).std()
    std.columns = ['Std']
    df = pd.concat([count, mean, ma, mi, median, std], axis=1)
    return df

def corr_2_cols(Col1, Col2):
    res = shot.groupby([Col1, Col2]).size().unstack()
    res['perc'] = (res[res.columns[1]]/(res[res.columns[0]] + res[res.columns[1]])) * 100
    return res
shot.age.hist(bins=20)
plt.tight_layout()
plt.title("Age distribution")
segm_target('race', 'age')
g = sns.FacetGrid(shot, hue='race', size = 7)
g.map(plt.hist, 'age', alpha = 0.3, bins= 20)
g.add_legend()
segm_target('signs_of_mental_illness', 'age')
g = sns.FacetGrid(shot, hue='signs_of_mental_illness', size = 7)
g.map(plt.hist, 'age', alpha = 0.3, bins= 20)
g.add_legend()
corr_2_cols('race', 'gender')
corr_2_cols('race', 'signs_of_mental_illness')
corr_2_cols('gender', 'signs_of_mental_illness')
plt.figure(figsize=(15,5))
sns.countplot(data=shot, x="race")

plt.title("Total number of people killed, by race", fontsize=12)
#shot.race.dropna(inplace = True)
labels = shot.race.value_counts().index
colors = ['orange','red','green','blue','brown','purple']
explode = [0,0,0,0,0,0]
sizes = shot.race.value_counts().values
plt.figure(figsize = (10,10))
plt.pie(sizes, explode=explode, labels=labels,  colors=colors, autopct='%1.1f%%')
plt.title('Percentage of people killed, by race',fontsize = 12)
tot = shot.shape[0]
USA_demo = pd.DataFrame({'race' : ["W", "B", "H", "A", "N", "O"],
                        'Population' : [int(0.613*tot),
                                       int(0.133*tot),
                                       int(0.178*tot),
                                       int(0.057*tot),
                                       int(0.015*tot),
                                       int((1-0.613-0.178-0.133-0.057-0.015)*tot)]})

USA_demo = USA_demo.sort_values(by='race')
USA_demo = USA_demo.set_index('race')
expected = USA_demo.Population
expected
shotbyrace = pd.crosstab(index=shot.race, columns="count")
observed = shotbyrace['count']
observed
stats.chisquare(observed, expected)
shot[['Armed_class', 'race', 'name']].groupby(['race', 'Armed_class']).count()
fleeing = shot[(shot.Fleeing_class == "Fleeing")]
fleeing.Armed_class.value_counts()
fleeing[['Armed_class', 'race', 'name']].groupby(['race', 'Armed_class']).count()
fleeing.Threat_class.value_counts()
labels = fleeing.race.value_counts().index
colors = ['orange','red','green','blue','brown','purple']
explode = [0,0,0,0,0,0]
sizes = fleeing.race.value_counts().values
plt.figure(figsize = (10,10))
plt.pie(sizes, explode=explode, labels=labels,  colors=colors, autopct='%1.1f%%')
plt.title('Percentage of people killed while fleeing, by race',fontsize = 12)
fleeunarmed = fleeing[fleeing.Armed_class == 'Unarmed']
fleeunarmed.Threat_class.value_counts()
labels = fleeunarmed.race.value_counts().index
colors = ['red','orange','green','brown','purple']
explode = [0,0,0,0,0]
sizes = fleeunarmed.race.value_counts().values
plt.figure(figsize = (10,10))
plt.pie(sizes, explode=explode, labels=labels,  colors=colors, autopct='%1.1f%%')
plt.title('Percentage of people killed while fleeing unarmed, by race',fontsize = 12)
fleearmed = fleeing[fleeing.Armed_class != 'Unarmed']
labels = fleearmed.race.value_counts().index
colors = ['orange','red','green','blue','brown','purple']
explode = [0,0,0,0,0,0]
sizes = fleearmed.race.value_counts().values
plt.figure(figsize = (10,10))
plt.pie(sizes, explode=explode, labels=labels,  colors=colors, autopct='%1.1f%%')
plt.title('Percentage of people killed while fleeing unarmed, by race',fontsize = 12)
name = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN',
       'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH',
       'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT',
       'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
population = [4830620,733375,6641928,2958208,38421464,5278906,3596222,926454,647484,19645772,10006693,
              1406299,1616547,12873761,6568645,309526,2892987,4397353,4625253,1329100,5930538,6705586,
              9900571,5419171,2988081,6045448,1014699,1869365,2798636,1324201,8904413,2084117,19673174,
              9845333,721640,11575977,3849733,3939233,12779559,1053661,4777576,843190,6499615,26538614,
              2903379,626604,8256630,6985464,1851420,5742117,579679]
demoW = [68.8,66.0,78.4,78.0,61.8,84.2,77.3,69.4,40.2,76.0,60.2,25.4,91.7,72.3,84.2,91.2,85.2,87.6,62.8,95.0,57.6,
        79.6,79.0,84.8,59.2,82.6,89.2,88.1,69.0,93.7,68.3,73.2,64.6,69.5,88.7,82.4,73.1,85.1,81.6,81.1,67.2,
        85.0,77.8,74.9,87.6,94.9,69.0,77.8,93.6,86.5,91.0]
demoB = [26.4,3.4,4.2,15.5,5.9,4.0,10.3,21.6,48.9,16.1,30.9,2.0,0.6,14.3,9.2,3.2,5.8,7.9,32.1,1.1,29.5,7.1,14.0,
        5.5,37.4,11.5,0.5,4.7,8.4,1.3,13.5,2.1,15.6,21.5,1.6,12.2,7.2,1.8,11.0,6.5,27.5,1.6,16.8,11.9,1.1,1.1,
         19.2,3.6,3.3,6.3,1.1]
demoN = [0.5,13.8,4.4,0.6,0.7,0.9,0.2,0.3,0.3,0.3,0.3,0.2,1.3,0.2,0.2,0.3,0.8,0.2,0.6,0.6,0.3,0.2,0.5,1.0,0.4,
         0.4,6.5,0.9,1.1,0.2,0.2,9.1,0.4,1.2,5.3,0.2,7.3,1.2,0.2,0.5,0.3,8.6,0.3,0.5,1.1,0.3,0.3,1.3,0.2,0.9,
         2.2]
demoA = [1.3,7.1,3.2,1.6,14.1,3.0,4.2,3.6,3.7,2.7,3.6,47.8,1.4,5.0,1.9,2.1,2.7,1.3,1.7,1.1,6.0,6.0,2.7,4.4,1.0,
         1.8,0.8,2.1,8.3,2.4,9.0,1.5,8.0,2.6,1.2,1.9,2.0,4.4,3.1,3.2,1.5,1.2,1.7,4.3,3.1,1.4,6.1,8.3,0.7,2.5,1.0]
demoO = [3.0,9.7,9.7,4.2,17.4,7.8,7.9,5.0,6.9,4.9,4.9,24.8,5.0,8.0,4.5,3.3,5.5,3.0,2.8,2.2,6.6,7.1,3.7,4.2,
         2.1,3.5,3.0,4.1,13.2,2.3,8.9,14.2,11.5,5.4,5.4,3.0,3.3,10.4,7.5,4.1,8.6,3.5,3.5,8.5,7.1,2.2,
        5.4,9.0,2.2,3.8,4.8]

states = pd.DataFrame({'state' : name, 'Population': population, 'State_share_W': demoW,
                   'State_share_B': demoB, 'State_share_N': demoN, 'State_share_A': demoA,
                    'State_share_O': demoO})
names = states.state
states.drop(labels=['state'], axis = 1, inplace=True)
states.insert(0, 'state', names)

states.sample(10)
kills = shot[['name', 'state']].groupby('state', as_index = True).count()
kills.rename(columns = {'name' : 'N_kills'}, inplace=True)

temp = shot[['state', 'race', 'name']].groupby(['state', 'race']).count().unstack().fillna(0)
kills['N_kills_A'] = temp['name']['A']
kills['N_kills_B'] = temp['name']['B']
kills['N_kills_H'] = temp['name']['H']
kills['N_kills_N'] = temp['name']['N']
kills['N_kills_W'] = temp['name']['W']
kills['N_kills_O'] = kills.N_kills - (kills.N_kills_A + kills.N_kills_B +
                                                     kills.N_kills_H + kills.N_kills_N +
                                                     kills.N_kills_W) #because some values are missing

temp = shot[['state', 'gender', 'name']].groupby(['state', 'gender']).count().unstack().fillna(0)
kills['N_kills_Fem'] = temp['name']['F']
kills['N_kills_Mal'] = temp['name']['M']

temp = shot[['state', 
             'signs_of_mental_illness', 
             'name']].groupby(['state', 'signs_of_mental_illness']).count().unstack().fillna(0)
kills['N_mental_illness'] = temp['name'][True]

temp = shot[['state', 
             'body_camera', 
             'name']].groupby(['state', 'body_camera']).count().unstack().fillna(0)
kills['N_body_camera'] = temp['name'][True]

temp = shot[['state', 
             'manner_of_death', 
             'name']].groupby(['state', 'manner_of_death']).count().unstack().fillna(0)
kills['N_Tasered'] = temp['name']['shot and Tasered']

kills['state'] = kills.index
kills.sample(10)
states = pd.merge(states, kills, on ='state', how = 'left')
states['Kills_pp'] = states['N_kills'] / states.Population
states['Perc_Kills_A'] = states.N_kills_A / states.N_kills * 100
states['Perc_Kills_B'] = states.N_kills_B / states.N_kills * 100
states['Perc_Kills_H'] = states.N_kills_H / states.N_kills * 100
states['Perc_Kills_N'] = states.N_kills_N / states.N_kills * 100
states['Perc_Kills_W'] = states.N_kills_W / states.N_kills * 100
states['Perc_Male'] = states.N_kills_Mal / states.N_kills * 100
states['Perc_Female'] = states.N_kills_Fem / states.N_kills * 100
states['Perc_Mental'] = states.N_mental_illness / states.N_kills * 100
states['Perc_BodyCam'] = states.N_body_camera / states.N_kills * 100
states['Perc_Tasered'] = states.N_Tasered / states.N_kills * 100
states.sample(10)
states.describe()
plt.figure(figsize=(15,5))
sns.barplot(data=states, x="state", y="N_kills")

plt.title("Total number of people killed, by state", fontsize=12)
plt.figure(figsize=(15,5))
sns.barplot(x="state", y="Kills_pp", data=states)

plt.title("Number of killed pro capita, by state", fontsize=12)
states.sort_values(by='Kills_pp', ascending=False).head()
states.sort_values(by='Kills_pp').head()
plt.figure(figsize=(15,5))
sns.barplot(x="state", y="Perc_Female", data=states)

plt.title("Percentage of females killed, by state", fontsize=12)
states.sort_values(by='Perc_Female', ascending=False).head()
plt.figure(figsize=(15,5))
sns.barplot(x="state", y="Perc_Mental", data=states)

plt.title("Percentage of killed with signs of mental illness, by state", fontsize=12)
states.sort_values(by='Perc_Mental', ascending=False).head()
plt.figure(figsize=(15,5))
sns.barplot(x="state", y="Perc_BodyCam", data=states)

plt.title("Percentage of killings with a body cam, by state", fontsize=12)
states.sort_values(by='Perc_BodyCam', ascending=False).head()
plt.figure(figsize=(15,5))
sns.barplot(x="state", y="Perc_Tasered", data=states)

plt.title("Percentage of killed and tasered, by state", fontsize=12)
states.sort_values(by='Perc_Tasered', ascending=False).head()
white = states.Perc_Kills_W
black = states.Perc_Kills_B
hispanic = states.Perc_Kills_H
native = states.Perc_Kills_N
asian = states.Perc_Kills_A

ind = states.state    
width = 0.75    
plt.figure(figsize=(16,5))

p1 = plt.bar(ind, white, width, color='orange', align='edge')
p2 = plt.bar(ind, black, width, bottom=white, color ='red', align='edge')
p3 = plt.bar(ind, hispanic, width, bottom=black+white, color ='green', align='edge')
p4 = plt.bar(ind, native, width, bottom = black+white+hispanic, color = 'brown', align='edge')
p5 = plt.bar(ind, asian, width, bottom = black+white+hispanic+native, color = 'blue', align='edge')

plt.ylabel('Percentage')
plt.title('Deadly killings segmented by race, per state')
plt.xticks(ind)
plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0]), ('W', 'B', 'H', 'N', 'A'))

plt.show()
white = states.State_share_W
black = states.State_share_B
hispanic = states.State_share_O
native = states.State_share_N
asian = states.State_share_A

ind = states.state    
width = 0.75    
plt.figure(figsize=(16,5))

p1 = plt.bar(ind, white, width, color='orange', align='edge')
p2 = plt.bar(ind, black, width, bottom=white, color ='red', align='edge')
p3 = plt.bar(ind, hispanic, width, bottom=black+white, color ='green', align='edge')
p4 = plt.bar(ind, native, width, bottom = black+white+hispanic, color = 'brown', align='edge')
p5 = plt.bar(ind, asian, width, bottom = black+white+hispanic+native, color = 'blue', align='edge')

plt.ylabel('Percentage')
plt.title('Population segmented by race, per state')
plt.xticks(ind)
plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0]), ('W', 'B', 'H', 'N', 'A'))

plt.show()
fil = ['state', 'State_share_A', 'Perc_Kills_A', 'Population', 'N_kills']
states.sort_values(by='Perc_Kills_A', ascending=False).head()[fil]
fil = ['state', 'State_share_B', 'Perc_Kills_B', 'Population', 'N_kills']
states.sort_values(by='Perc_Kills_B', ascending=False).head()[fil]
fil = ['state', 'State_share_W', 'Perc_Kills_H', 'Population', 'N_kills']
states.sort_values(by='Perc_Kills_H', ascending=False).head()[fil]
fil = ['state', 'State_share_N', 'Perc_Kills_N', 'Population', 'N_kills']
states.sort_values(by='Perc_Kills_N', ascending=False).head()[fil]
fil = ['state', 'State_share_W', 'Perc_Kills_W', 'Population', 'N_kills']
states.sort_values(by='Perc_Kills_W', ascending=False).head()[fil]
bodycam = shot[shot.body_camera == True]
bodycam.shape
res = bodycam.groupby(['gender', 'signs_of_mental_illness']).size().unstack()
res['perc'] = (res[res.columns[1]]/(res[res.columns[0]] + res[res.columns[1]])) * 100
res
res = bodycam.groupby(['race', 'signs_of_mental_illness']).size().unstack()
res['perc'] = (res[res.columns[1]]/(res[res.columns[0]] + res[res.columns[1]])) * 100
res
labels = bodycam.race.value_counts().index
colors = ['orange','red','green','blue','brown','purple']
explode = [0,0,0,0,0,0]
sizes = bodycam.race.value_counts().values
plt.figure(figsize = (10,10))
plt.pie(sizes, explode=explode, labels=labels,  colors=colors, autopct='%1.1f%%')
plt.title('Percentage of people killed with body camera, by race',fontsize = 12)
bodycam.Armed_class.value_counts()
bodycam[bodycam.Fleeing_class == 'Fleeing'].Threat_class.value_counts()
bodycam[bodycam.Armed_class == 'Unarmed'].Threat_class.value_counts()
bodycam[bodycam.Armed_class == 'Unknown'].Threat_class.value_counts()