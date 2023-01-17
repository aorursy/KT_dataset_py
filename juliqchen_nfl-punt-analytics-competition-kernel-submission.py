
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# Loading video_review.csv data
vr = pd.read_csv('../input/video_review.csv')
vr.head(10)
primary_impact_type_hist = vr['Primary_Impact_Type'].value_counts().plot(kind='bar')
player_activity_derived_hist = vr['Player_Activity_Derived'].value_counts().plot(kind='bar')
primary_partner_activity_derived = vr['Primary_Partner_Activity_Derived'].value_counts().plot(kind='bar')
# Create play_player_role_data dataframe.
pprd = pd.read_csv('../input/play_player_role_data.csv')
# Create play_information dataframe.
play_inf = pd.read_csv('../input/play_information.csv')
# Create game_inf dataframe.
game_inf = pd.read_csv('../input/game_data.csv')
# Create video_review dataframe
vr = pd.read_csv('../input/video_review.csv')

# Extract roles column from play_player_role_data.csv and check if player is on punting team.
punting_pos = {"PLS","PLG", "PLT", "PLW","PRG","PRT","PRW","PC","PPR", "PPL","P","GL","GR"}
role_col = pprd['Role']
punting_team_col = [x in punting_pos for x in role_col.tolist()]

# Adding punting_team_col to pprd dataframe.
pprd['punting_team'] = punting_team_col

# Function that merges GameKey and PlayID given a row.
def merge(row):
    return str(row['GameKey']) + '_' + str(row['PlayID'])

# Combine GameKey and PlayID to make unique Key that organizes formations.
pprd_merged_col = pprd.apply(lambda row: merge(row), axis=1)
play_inf_merged_col = play_inf.apply(lambda row: merge(row), axis=1)
vr_merged_col = vr.apply(lambda row: merge(row), axis=1)

# Adds merged_col to pprd dataframe.
pprd['game_play_key'] = pprd_merged_col
play_inf['game_play_key'] = play_inf_merged_col
vr['game_play_key'] = vr_merged_col

# Groupby game_play_key and punting_team.
pprd_formations = pprd.groupby(['game_play_key','punting_team']).apply(lambda x: sorted(x['Role'].values.tolist()))
pprd_formations = pprd_formations.rename("form")
punting_team_boolean = [x[0] in punting_pos for x in pprd_formations.tolist()]
pprd_formations = pprd_formations.apply(lambda x: ', '.join(x)).to_frame()
pprd_formations['punting_team'] = punting_team_boolean
# Sorted formations
#print(pprd_formations.keys())
pprd_and_playinf = play_inf.merge(pprd_formations, on=['game_play_key'], how='outer')
pprd_and_playinf = pprd_and_playinf.drop(columns=['Season_Year', 'GameKey', 'PlayID'])

# Merge play_information.csv with play_player_role_data.csv by game_play_key.
merged = pprd_and_playinf.merge(vr, on=['game_play_key'], how='outer')
returnTeamForms = merged[merged['punting_team'] == False]['form']

concuss = merged[merged['Turnover_Related'].notnull()]
concussReturnTeamsForms = concuss[concuss['punting_team'] == False]['form']

count_formations_punting = concuss.loc[merged['punting_team'] == False, 'form']
count_formations_recv = concuss.loc[merged['punting_team'] == True, 'form']
# print(count_formations_punting)
a = count_formations_punting.value_counts()
b = count_formations_recv.value_counts()

print("Types of formations for punting team, counts")
a.head(7)

print("Types of formations for receiving team, counts")
b.head(7)

import time
start = time.time()
NGSDICT = {
    "pre-2016" :  pd.read_csv('../input/NGS-2016-pre.csv'),
    "pre-2017" :  pd.read_csv('../input/NGS-2017-pre.csv'),
    "post-2016" :  pd.read_csv('../input/NGS-2016-post.csv'),
    "post-2017" :  pd.read_csv('../input/NGS-2017-post.csv'),
    
    "reg-2016-wk1-6" :  pd.read_csv('../input/NGS-2016-reg-wk1-6.csv'),
    "reg-2016-wk7-12" :  pd.read_csv('../input/NGS-2016-reg-wk7-12.csv'),
    "reg-2016-wk13-17" :  pd.read_csv('../input/NGS-2016-reg-wk13-17.csv'),
    
    "reg-2017-wk1-6" :  pd.read_csv('../input/NGS-2017-reg-wk1-6.csv'),
    "reg-2017-wk7-12" :  pd.read_csv('../input/NGS-2017-reg-wk7-12.csv'),
    "reg-2017-wk13-17" :  pd.read_csv('../input/NGS-2017-reg-wk13-17.csv')
}
end = time.time()
print("Time to load:")
print(end- start)


def getNGSName(season,week,season_type):
    if season_type == 'Pre':
        return 'pre-' + str(season)
    elif season_type == 'Post':
        return 'post-' + str(season) 
    else:
        if(week <= 6):
            return 'reg-' + str(season)+ '-wk1-6'
        if(week >= 7 and week <=12):
            return 'reg-' + str(season)+ '-wk7-12'
        else:
            return 'reg-' + str(season)+ '-wk13-17'
#print(pprd.keys())
PUNTPOS = {"PLS","PLG", "PLT", "PLW","PRG","PRT","PRW","PC","PPR","P","GL","GR"}
RETURNPOS = {"PDR3","PDR2","PDR1","PDL1","PDL2","PDL3","PLM","PLL","PLR","PFB","PR","VRi","VRo","VLi","VLo","VR","VL"}

cc = vr['game_play_key'].tolist()
gsi = vr['GSISID'].tolist()
#print(cc)
#print(gsi)
# print(pprd['GSISID'] == gsi[0])
c = pd.DataFrame()
for x in range(len(cc)):
    curr = (pprd[(pprd['game_play_key'] == cc[x]) & (pprd['GSISID'] == gsi[x])])
    c = c.append(curr)
#print(c)
c['punting_team'] = c['punting_team'].replace(True,'Punting Team')
c['punting_team'] = c['punting_team'].replace(False, 'Recv Team')
count = c['punting_team'].value_counts().plot(title ="Concussions per Team",kind='bar')
from multiprocessing import Queue, Process
from threading import Thread
featureVectors = Queue()
fvs = []
puntplays = play_inf.query("Play_Type == 'Punt'")
def makeFeatureVector(lo,hi,qq):
    #print(lo,hi)
    localcount = 0
    for rr in range(lo,hi):
        s = time.time()
        currow = puntplays.iloc[rr]
        nam = getNGSName(currow["Season_Year"],currow["Week"],currow["Season_Type"])
        
        #if(localcount > 0):
            #break
        ngsTotal = NGSDICT[nam]
        #print(currow["GameKey"],currow["PlayID"])
        thisplay = ngsTotal.query('Season_Year == %d and GameKey == %d and PlayID == %d' %(currow["Season_Year"], currow['GameKey'],currow['PlayID']))
        if(len(thisplay) > 0):
            puntinfo = thisplay.query("Event == 'punt'")
            prinfo = thisplay.query("Event == 'punt_received' or Event == 'fair_catch'")
            puntinfo = pd.to_datetime(puntinfo['Time'],format='%Y-%m-%d %H:%M:%S.%f')
            prinfo = pd.to_datetime(prinfo['Time'],format='%Y-%m-%d %H:%M:%S.%f')
            ht = 0
            if(len(puntinfo) > 0 and len(prinfo) > 0):
                kicked = min(puntinfo)
                gotten = min(prinfo)
                ht = (gotten-kicked).total_seconds()
            gamedata = game_inf.query("Season_Year == %d and GameKey == %d" % (currow["Season_Year"],currow["GameKey"]))
            smallp = pprd.query('Season_Year == %d and GameKey == %d and PlayID == %d' %(currow["Season_Year"], currow['GameKey'],currow['PlayID']))
            gamedata = gamedata.iloc[0]
            featureVector = {}
            featureVector['HangTime'] = ht
            featureVector['Week'] = gamedata["Week"]
            featureVector['Season_Type'] = gamedata["Season_Type"]
            featureVector['Game_Day'] = gamedata["Game_Day"]
            featureVector['Game_Site'] = gamedata["Game_Site"]
            featureVector['HomeTeamCode'] = gamedata["HomeTeamCode"]
            featureVector['VisitTeamCode'] = gamedata["VisitTeamCode"]
            featureVector['Turf'] = gamedata["Turf"]
            featureVector['GameWeather'] = gamedata["GameWeather"]
            featureVector['Temperature'] = gamedata["Temperature"]
            featureVector['OutdoorWeather'] = gamedata["OutdoorWeather"]
            featureVector['Stadium'] = gamedata["Stadium"]
            featureVector['StadiumType'] = gamedata["StadiumType"]
            featureVector['Start_Time'] = gamedata["Start_Time"]
            featureVector['Game_Clock'] = currow["Game_Clock"]
            featureVector['YardLine'] = currow["YardLine"]
            featureVector['Quarter'] = currow["Quarter"]
            featureVector['GameKey'] = currow["GameKey"]
            featureVector['PlayID'] = currow["PlayID"]
            featureVector['Season_Year'] = currow["Season_Year"]
            featureVector['Play_Type'] = currow["Play_Type"]
            featureVector['Poss_Team'] = currow["Poss_Team"]
            featureVector['HomeScore'] = currow["Score_Home_Visiting"].split(' - ')[0]
            featureVector['AwayScore'] = currow["Score_Home_Visiting"].split(' - ')[1]
            
            for x in PUNTPOS:
                featureVector[x + '_' + 'AVG SPEED'] = "NA"
                featureVector[x + '_' + 'AVG DIR'] = "NA"
                featureVector[x + '_' + 'AVG O'] = "NA"
            
            for x in RETURNPOS:
                featureVector[x + '_' + 'AVG SPEED'] = "NA"
                featureVector[x + '_' + 'AVG DIR'] = "NA"
                featureVector[x + '_' + 'AVG O'] = "NA"
            found = 0
            for ttt in set(thisplay['GSISID']):
                try:
                    k = smallp.query('GSISID == %d' %(ttt))
                except:
                    continue
                if(len(k) > 0):
                    localcount += 1
                    legalplayermovement = thisplay[thisplay['GSISID']== ttt]
                    #print(legalplayermovement['Event'].unique())
                    rl = k.iloc[0]["Role"]
                    split = len(legalplayermovement)//3
                    featureVector[rl + '_' + 'AVG DIR'] = legalplayermovement['dir'].mean()
                    featureVector[rl + '_' + 'AVG O'] = legalplayermovement['o'].mean()
                    wind = 0
                    '''
                    for x in range(0,len(legalplayermovement),split):
                        wind+= 1
                        tempdf = legalplayermovement[x:] if x+split >= len(legalplayermovement) else legalplayermovement[x:x+split]
                        featureVector[rl + '_' + 'AVG DIR_' + str(wind) +'-WINDOW' ] = legalplayermovement['dir'].mean()
                        featureVector[rl + '_' + 'AVG O' + str(wind) +'-WINDOW'] = legalplayermovement['o'].mean()
                        speeds = legalplayermovement["dis"].apply(lambda x: x * 20.5)
                        featureVector[rl + '_' + 'AVG SPEED'+ str(wind) +'-WINDOW'] = speeds.mean()
                    '''
                    speeds = legalplayermovement["dis"].apply(lambda x: x * 20.5)

                    featureVector[rl + '_' + 'AVG SPEED'] = speeds.mean() 
                    found += 1

            qq.put(featureVector)
        #print(thisplay)
        e = time.time()
        #print(e-s)
    print("done with the process")
    return

def empty_queue(q):
    while True:
        k = q.get()
        fvs.append(k)

def makeFeatures(threads):
    ss = time.time()
    ts = []
    numthreads = threads
    workperthread = int(len(puntplays)/numthreads)
    for rr in range(0,len(puntplays),workperthread):
        #if rr >= 100: break
        if(rr + workperthread >= len(puntplays)):
            curt = Process(target = makeFeatureVector, args = (rr,len(puntplays),featureVectors,))
        else:
            curt = Process(target = makeFeatureVector, args = (rr,rr+workperthread,featureVectors,))
        ts.append(curt)
    for t in ts:
        t.start()
    cc = 0
    monit = Thread(target=empty_queue, args=(featureVectors,))
    monit.start()
    for t in ts:
        print("waiting for %d" % cc)
        cc+= 1
        t.join()
        
    
    ee = time.time()
    print("Time Taken")
    print(ee - ss)
    print("Made this many feature vectors:")
    print(len(fvs))
    return 
# Make the features in parallel using 10 processes. 
makeFeatures(10)
print(len(fvs))
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt 
from imblearn.over_sampling import SMOTE
import copy
import matplotlib.pyplot as plt

vf = pd.read_csv('../input/video_footage-injury.csv')
concussionset = set()
for x in range(len(vr)):
    cr = vr.iloc[x]
    idenstring = str(cr['Season_Year'])+ '_' + str(cr['GameKey']) + '_' + str(cr['PlayID'])
    concussionset.add(idenstring)

fvscopy = copy.deepcopy(fvs)
fvscopy2 = copy.deepcopy(fvs)
confvs = []
regularfvs = []

concussionFEATS= []
regularFEATS = []
for x in fvscopy2:
    fviden = str(x['Season_Year'])+ '_' + str(x['GameKey']) + '_' + str(x['PlayID'])
    if(fviden in concussionset):
        concussionFEATS.append(x)
    else:
        regularFEATS.append(x)
        
for x in fvscopy:
    fviden = str(x['Season_Year'])+ '_' + str(x['GameKey']) + '_' + str(x['PlayID'])
    jon_check1, jon_check2 = x['GameKey'],x['PlayID']
    del x['GameKey']
    del x['Season_Year']
    del x['PlayID']
    del x['Play_Type']
    #print(x["HangTime"])
    if(fviden in concussionset and not (jon_check1 == 274 and jon_check2 == 3609)):
        confvs.append(x)
    else:
        regularfvs.append(x)

consdf = pd.DataFrame(confvs)
regsdf = pd.DataFrame(regularfvs)
concussionFEATSdf = pd.DataFrame(concussionFEATS)
regularFEATSdf = pd.DataFrame(regularFEATS)
def subsetMean(c,n):
    comps = {}
    for k in c.keys():
        comps[k] = {}
        #print(k)
        ccworked = False
        ncworked = False
        try:
            if('O' in k or 'DIR' in k):
                comps[k]['Concussion'] = c[k].replace('NA',0).fillna(0).mean() - 180
                #cc.append(consdf[k].replace('NA',0).fillna(0).mean() - 180)
                ccworked = True
            else:
                comps[k]['Concussion'] = c[k].replace('NA',0).fillna(0).mean()
                #cc.append(consdf[k].replace('NA',0).fillna(0).mean())
                ccworked = True
            #print(k, consdf[k].mean(),'c')
        except: pass
        try:
            if('O' in k or 'DIR' in k):
                comps[k]['No Concussion'] = n[k].replace('NA',0).fillna(0).mean() -180
                #nc.append(regsdf[k].replace('NA',0).fillna(0).mean())
                ncworked = True
            else:
                comps[k]['No Concussion'] = n[k].replace('NA',0).fillna(0).mean()
                #nc.append(regsdf[k].replace('NA',0).fillna(0).mean())
                ncworked = True
        except: pass
        #print(k, regsdf[k].mean(),'nc')
    return comps

comps = subsetMean(consdf,regsdf)
compsdf = pd.DataFrame(comps)
def plot_comparisons(word,word2):
    subsetDIR = []
    subsetO = []
    subsetSPEED = []

    subsetDIRR = []
    subsetOR = []
    subsetSPEEDR = []

    for x in PUNTPOS:
        if(x == "PC" or x == "PPR"): continue
        subsetDIR.append(x+'_AVG' +' DIR')
        subsetO.append(x+'_AVG' +' O')
        subsetSPEED.append(x+'_AVG' +' SPEED')

    for x in RETURNPOS:
        if(x == "PC" or x == "PPR"): continue
        subsetDIRR.append(x+'_AVG' +' DIR')
        subsetOR.append(x+'_AVG' +' O')
        subsetSPEEDR.append(x+'_AVG' +' SPEED')
    smoldfDir = compsdf[subsetDIR]
    smoldfO = compsdf[subsetO]
    smoldfSPEED = compsdf[subsetSPEED]

    smoldfDirR = compsdf[subsetDIRR]
    smoldfOR = compsdf[subsetOR]
    smoldfSPEEDR = compsdf[subsetSPEEDR]
    if(word == "punt"):
        if(word2 == 'directions'):
            plt.figure()
            smoldfDir.T.plot.bar(title='Punt Team Directions (degrees)',figsize=(12,8))
        if(word2 == 'o'):
            plt.figure()
            smoldfO.T.plot.bar( title='Punt Team Orientations (degrees)',figsize=(12,8))
        if(word2 == 'speed'):
            plt.figure()
            smoldfSPEED.T.plot.bar( title='Punt Team Speeds (mph)',figsize=(12,8))
    else:
        if(word2 == 'directions'):
            plt.figure()
            smoldfDirR.T.plot.bar(title='Return Team Directions (degrees)',figsize=(12,8))
        if(word2 == 'o'):
            plt.figure()
            smoldfOR.T.plot.bar( title='Return Team Orientations (degrees)',figsize=(12,8))
        if(word2 == 'speed'):
            plt.figure()
            smoldfSPEEDR.T.plot.bar( title='Return Team Speeds (mph)',figsize=(12,8))
print(len(fvs))
plot_comparisons("punt",'speed')
plot_comparisons("recv",'speed')
plot_comparisons("punt",'directions')
plot_comparisons("recv",'directions')
plot_comparisons("punt",'o')
plot_comparisons("recv",'o')
def randomSet(concs, notconcus):
    import random
    testset = []
    results = []
    cs,ncs = 0,0
    nex = 0
    for x in range(500):
        rando = random.randint(0,500)
        if(rando % 4 == 1):
            ind = random.randint(0,len(concs)-1)
            testset.append(concs[nex % len(concs)-1])
            nex += 1
            results.append(1)
            cs += 1
        else:
            ind = random.randint(0,len(notconcus)-1)
            testset.append(notconcus[ind])
            results.append(0)
            ncs += 1
    
    print("set contains %d concussions and %d regular plays" % (cs,ncs))
    return testset,results

def doSelect():
    ts,rs = randomSet(confvs,regularfvs)
    setdf = pd.DataFrame(ts)
    setdf = setdf.fillna(0)
    for s in setdf:
        setdf[s]=setdf[s].astype('str')
    #print(setdf)
    sm = SMOTE(random_state=2)
    fixed = setdf.apply(LabelEncoder().fit_transform)
    X_train_res, y_train_res = sm.fit_sample(fixed, rs)
    print(len(X_train_res))
    feats = SelectKBest(mutual_info_classif,k=50).fit(X_train_res,y_train_res)
    new_features = [] # The list of your K best features
    feature_names = list(setdf.columns.values)
    mask = feats.get_support()#indices=True)
    for bool, feature in zip(mask, feature_names):
        if bool:
            new_features.append(feature)
    cols = feats.get_support(indices=True)
    mutualInfoVal = dict(sorted(zip(fixed.columns.values,feats.scores_),key=lambda x: x[1]))
    ordered = reversed(list(mutualInfoVal))
    #print(new_features)
    for m in ordered:
        try:
            selectResults[m] += mutualInfoVal[m]
        except:
            selectResults[m] = mutualInfoVal[m]
            
selectResults = {}
for x in range(10):
    doSelect()
print('done selections')

featuresCount = 0
#print(selectResults)
for sR in reversed(sorted(selectResults,key=selectResults.get)):
    if(featuresCount > 50):break
    print(sR, selectResults[sR]/10)
    featuresCount += 1
count = 0
jammerIDS = {}
def countJammerFormations(df,s):
    jammercounts = {}
    for x in range(len(df)):
        forms = df.iloc[x].split(', ')
        j = 0
        for y in forms:
            if y.startswith("V"):
                j += 1
        if(j == 5):
            j = 4
            #print(df.iloc[x].split(', '),j)
        try: jammercounts[j] +=1
        except: jammercounts[j] = 1
    for n in sorted(jammercounts):
        print("The formation with %d jammers appears %d times %s" % (n,jammercounts[n],s))
    #count += 1


countJammerFormations(returnTeamForms,'overall')
print()
countJammerFormations(concussReturnTeamsForms,'in the concussion set')
def getFormationInjury(df):
    formationsInjury = {}
    for x in range(len(df)):
        rr = df.iloc[x]
        #print(rr.keys())
        forms = rr["form"].split(', ')
        j=0
        for y in forms:
            if y.startswith("V"):
                j += 1

        primary = rr['GSISID']
        g,p = rr['game_play_key'].split("_")
        prole = pprd.query("GameKey == %d and PlayID == %d and GSISID == %d" %(int(g),int(p),int(primary))).iloc[0]['Role']

        try:formationsInjury[j].append(prole)
        except:
            formationsInjury[j] = []
            formationsInjury[j].append(prole)
        
        #print(prole)
    
    for x in sorted(formationsInjury):
        injuredOffense = 0
        print("In the formations with %d jammers, the following positions received concussions:" % x)
        for y in formationsInjury[x]:
            if(y in PUNTPOS):
                injuredOffense += 1
                
            
        print("\t ", formationsInjury[x])
        print('\t Percent of injuries to the punting team: %f' % (injuredOffense*100/len(formationsInjury[x]) ))
    

    
getFormationInjury(concuss[concuss['punting_team'] == False])
count = 0
jammerIDSBackers = {}
def countBackerFormations(df,s):
    jammercounts = {}
    for x in range(len(df)):
        forms = df.iloc[x].split(', ')
        j = 0
        backer = 0
        dl = 0
        for y in forms:
            if(y == "PR" or y == "PFB"): continue
            if y.startswith("V"):
                j += 1
                continue
            if("DL" in y or "DR" in y):
                dl += 1
            else:
                backer += 1
        if(j == 5):
            j = 4
            #print(df.iloc[x].split(', '),j)
        if(j == 4): continue
        
        try: jammercounts[(backer,dl)] +=1
        except: jammercounts[(backer,dl)] = 1
    #for n in sorted(jammercounts):
        #print("The formation with %d linebackers and %d linemen appears %d times %s" % (n[0],n[1],jammercounts[n],s))
    #print()
    return jammercounts
            
a = countBackerFormations(returnTeamForms,'overall')
b = countBackerFormations(concussReturnTeamsForms,'in the concussion set')
for x in b:
     print("The formation with %d linebackers and %d linemen has concussion percentage %f" % (x[0],x[1],(b[x]*100/a[x])))
