import os
import glob
import datetime as dt
import csv
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import os
from scipy import ndimage, misc
from skimage.transform import resize
import pickle
%matplotlib inline
base_dir = os.path.join("../input")
all_files = glob.glob(os.path.join(base_dir, "*.csv"))
for a in list(all_files):
    print(a)
def consol_pos(x):    
    if x in ['RB','FB']:
        return 'HB' #halfback
    elif x in ['SS','FS']:
        return 'S' #safety
    elif x in ['MLB','ILB','OLB']:
        return 'LB' #linebacker
    elif x in ['PLT','PRT','PLG','PRG']:
        return 'POL' #Punt O-line
    elif x in ['LS','PLS']:
        return 'LS' #long snapper
    elif x in ['PLW','PRW']:
        return 'POW' #Punting offessive-wing (could call them tight ends really)
    elif x in ['GL','GR','GLi','GRo','GLo','GRi']:
        return 'Gn' #Gunner on return team
    elif x in ['VR','VRo','VRi','VL','VLi','VLo']:
        return 'V' #Gunner on kicking team
    elif x in ['PDR1','PDR2','PDR3','PDL1','PDL2','PDL3','PDR4','PDL5','PDR6','PDL6','PDR5','PDL4','PDM']:
        return 'PDL' #PuntD-line
    elif x in ['PLR','PLM','PLL','PLR2','PLL2','PLM1','PLR3','PLL3','PLL1','PLLi','PLR1']:
        return 'PLB' #Punt LB
    elif x in ['PPL','PPR','PPRo','PPRi','PPLo','PPLi','PC']:
        return 'PP' #Punting backfield 
    else:
        return x
    
# Defines whether on kicking or return team
def kick_ret(x):
    if x in ['P','PP','LS','POW','POL','Gn']:
        return 'KICK'
    elif x in ['PDL','PLB','V','PR','PFB']:
        return 'RET'
    else:
        return 'UNKN'

inj = pd.read_csv(os.path.join(base_dir, "video_review.csv"))
inj['concussion'] = 1

ppd = pd.read_csv(os.path.join(base_dir, "player_punt_data.csv"))
ppd = ppd[ppd.GSISID.isin(list(inj.GSISID.unique()))][['GSISID','Position']]\
        .sort_values('GSISID').drop_duplicates().reset_index(drop=True)
ppd['Position_consol'] = ppd.Position.apply(lambda x: consol_pos(x))

pprd = pd.read_csv(os.path.join(base_dir, "play_player_role_data.csv"))
pprd['Role_consol'] = pprd.Role.apply(lambda x: consol_pos(x))
pprd['Kick_ret'] = pprd.Role_consol.apply(lambda x: kick_ret(x))

inj = inj.merge(ppd[['GSISID','Position','Position_consol']],on='GSISID',how='inner')
inj = inj.merge(pprd,on=['Season_Year','GameKey','PlayID','GSISID'])

display(inj.head())
def intl_cities(x):
    if x in ['Wembley','Twickenham','London']:
        return 'UK'
    elif x in ['Mexico','Mexico City']:
        return 'MEX'
    else:
        return 'USA'

def stadium_type(x):
    if x in ['Outdoor','Outdoors','outdoor','Outside','Outdoors ','Ourdoor','Outddors','Oudoor','Outdor',
             'Heinz Field','Turf']:
        return 'outdoor'
    elif x in ['Dome','Indoor','non-retractable roof','Retr. Roof - Closed','Indoors','Indoor',
               'Indoor, Non-Retractable Dome','Retr. Roof-Closed','Retr. roof - closed','Indoor, fixed roof',
              'Indoor, Fixed Roof','Indoors (Domed)','Domed, closed','Indoor, Roof Closed','Retr. Roof Closed',
              'Closed Dome','Dome, closed','Indoor, non-retractable roof']:
        return 'indoor_closed'
    elif x in ['Retractable Roof','Open','Retr. Roof-Open','Retr. Roof - Open','Indoor, Open Roof',
               'Outdoor Retr Roof-Open']:
        return 'indoor_open'
    elif np.isnan(x):
        return 'outdoor'
    else:
        return x
    
def turf_type(x):
    if x in ['Turf','Artificial','Synthetic','Artifical']:
        return 'Generic_turf'
    elif x in ['Grass','Natural Grass', 'Natural grass','Natural Grass ','Natural','Natrual Grass','Naturall Grass',]:
        return 'Grass'
    elif x in ['DD GrassMaster']:
        return 'Grassmaster'
    elif x in ['A-Turf Titan']:
        return 'A-Turf_titan'
    elif x in ['FieldTurf','Field Turf','FieldTurf 360','FieldTurf360','Field turf']:
        return 'Fieldturf'
    elif x in ['UBU Speed Series-S5-M','UBU Sports Speed S5-M','UBU Speed Series S5-M']:
        return 'UBU_speed_series_S5-M'
    elif x in ['AstroTurf GameDay Grass 3D']:
        return 'Astroturf'
    elif pd.isna(x):
        return 'Grass'
    else:
        return x
        
gi = pd.read_csv(os.path.join(base_dir, "game_data.csv"))

gi['Start_time_hour'] = gi.Start_Time.apply(lambda x: x[:2])
gi['StadiumType_consol'] = gi.StadiumType.apply(lambda x: stadium_type(x))
gi['intl_cities'] = gi.Game_Site.apply(lambda x: intl_cities(x))
gi['Turf_consol'] = gi.Turf.apply(lambda x: turf_type(x))
gi['is_sunny'] = gi.GameWeather.apply(lambda x: 1 if any(ss in str(x).lower() for ss in ['sun','part']) else 0)
gi['is_cloudy'] = gi.GameWeather.apply(lambda x: 1 if any(ss in str(x).lower() for ss in ['cloud','part']) else 0)
gi['is_clear'] = gi.GameWeather.apply(lambda x: 1 if any(ss in str(x).lower() for ss in ['clear']) else 0)
gi['is_rain'] = gi.GameWeather.apply(lambda x: 1 if any(ss in str(x).lower() for ss in ['rain','storm']) else 0)
gi['is_snow'] = gi.GameWeather.apply(lambda x: 1 if any(ss in str(x).lower() for ss in ['snow']) else 0)

gi_merge = gi[['GameKey','Season_Year','Season_Type','Week','Game_Day','Game_Site','Start_time_hour',
              'StadiumType_consol','intl_cities','Turf_consol','is_sunny','is_cloudy','is_clear','is_rain','is_snow']]

display(gi_merge.head())
def get_half(x):
    if x in [1,2]:
        return 1
    elif x in [3,4]:
        return 2
    elif x == 5:
        return 3
    
def get_penalty_type(PlayDescription):
    '''Extract Penalty Types From Text'''
    PlayDescription = PlayDescription.upper()
    try:
        enum_object = list(enumerate(PlayDescription.split(',')))
        penalty_obj = [x for x,y in enum_object if 'PENALTY' in y]
        return enum_object[penalty_obj[0]+1][1]
    except:
        return 'None'

pi = pd.read_csv(os.path.join(base_dir, "play_information.csv"))
pi.head()

pi['half'] = pi.Quarter.apply(lambda x: get_half(x))
pi['late_quart'] = [0 if x in [1,3,5] else 1 for x in pi.Quarter]
gc = pi.Game_Clock.str.split(":").apply(pd.Series)
pi['sec_elapsed_quart'] = (60*(14 - gc[0].astype(int)) + (60 - gc[1].astype(int)))
pi['sec_elapsed_half'] = (900*pi.late_quart) + pi.sec_elapsed_quart

pi['punt_in_own_terr'] = pi.apply(lambda x: x.Poss_Team != x.YardLine.split(' ')[0],axis=1).astype(int)
pi['yards_from_own_endzone'] = pi.apply(lambda x: x.punt_in_own_terr*(100-int(x.YardLine.split(' ')[1])) + 
         (1-x.punt_in_own_terr)*(int(x.YardLine.split(' ')[1])),axis=1)

pi['is_muff'] = pi.PlayDescription.str.contains('MUFF',case=False).astype(int)
pi['is_penalty'] = pi.PlayDescription.str.contains('PENALTY',case=False).astype(int)
pi['is_faircatch'] = pi.PlayDescription.str.contains('FAIR',case=False).astype(int)
pi['penalty_type'] = pi.PlayDescription.apply(get_penalty_type)
pi['is_touchback'] = pi.PlayDescription.str.contains('TOUCHBACK',case=False).astype(int)
pi['is_oob'] = pi.PlayDescription.str.contains('OUT OF BOUNDS',case=False).astype(int)
pi['is_downed'] = pi.PlayDescription.str.contains('DOWNED',case=False).astype(int)
pi['is_returned'] = pi.apply(lambda x: x[['is_faircatch','is_downed','is_oob','is_touchback']].any() != 1,axis=1).astype(int)

score_diff = pi.Score_Home_Visiting.str.split(' - ').apply(pd.Series).astype(int)
pi['sd'] = score_diff[0] - score_diff[1]
pi['home_team'] = (pi.Home_Team_Visit_Team.str.split('-').apply(pd.Series))[0]
pi['is_home_team_punting'] = (pi.home_team == pi.Poss_Team).astype(int).replace(0,-1)
pi['score_diff'] = pi.sd * pi.is_home_team_punting


pi_merge = pi[['Season_Year','Season_Type','GameKey','Week','PlayID','sec_elapsed_half','half',
               'yards_from_own_endzone','is_muff','is_penalty','is_oob','is_returned','is_faircatch','penalty_type',
              'score_diff']]

display(pi_merge.head())
merge_data = pd.merge(pi_merge, inj[['GameKey','PlayID','concussion']],on=['GameKey','PlayID'],how='outer').fillna(0)
merge_data = gi_merge.merge(merge_data,left_on=['Season_Year','GameKey'],right_on=['Season_Year','GameKey']).set_index(['Season_Year','GameKey','PlayID'])
merge_data = merge_data.drop(['Week_y','Season_Type_y'],axis=1).rename(mapper = {'Week_x':'Week','Season_Type_x':'Season_Type'},axis=1)

display(merge_data.head())
display(merge_data.shape)
display(merge_data.columns)

game_file_dict = dict()

def make_game_dicts():
     for file in ['NGS-2016-pre.csv','NGS-2016-reg-wk7-12.csv','NGS-2017-reg-wk7-12.csv','NGS-2017-pre.csv',
              'NGS-2016-reg-wk1-6.csv','NGS-2016-post.csv','NGS-2017-post.csv','NGS-2016-reg-wk13-17.csv',
              'NGS-2017-reg-wk1-6.csv','NGS-2017-reg-wk13-17.csv']:
        game_data = pd.read_csv(os.path.join(base_dir, file))
        game_data['season'] = file
        season_dict = game_data.set_index(['Season_Year','GameKey'])['season'].to_dict()
        game_file_dict.update(season_dict)
        
make_game_dicts()

def find_intersection_angle(row):
    finder = FindTheBoom(int_data,NGSTable=pre_2016,gamekey=row.GameKey,playid=row.PlayID)
    p1,p2 = finder.find_partners()
    coords = finder.find_coords(py1 = p1, py2 = p2)
    moment_of_intersection = finder.find_moment_of_intersection(coords)
    angle = finder.find_blindness(coords=coords,moment=moment_of_intersection)
    return angle
#Find the blindside hit
class FindTheBoom:
    
    def __init__(self, InteractionTable,gamekey,playid,seasonyear):
        self.itable = InteractionTable
        self.gk = gamekey
        self.pid = playid
        self.season = seasonyear
    
    def find_game_file(self,seasonyear, gamekey):
        file = game_file_dict[(seasonyear,gamekey)]
        df = pd.read_csv('../input/'+file)
        return df
        
    def find_partners(self):
        temp_table = self.itable[(self.itable.GameKey == self.gk) & (self.itable.PlayID == self.pid)]
        return temp_table.GSISID.values.astype(int), temp_table.Primary_Partner_GSISID.values.astype(int)
    
    def find_coords(self,py1,py2):
        ngs_table = self.find_game_file(self.season,self.gk)
        print(ngs_table)
        p1 = ngs_table[(ngs_table.GameKey == self.gk) & (ngs_table.PlayID == self.pid) & (ngs_table.GSISID == py1[0])].sort_values('Time')[['Time','x','y','o','dir']].set_index('Time')
        p2 = ngs_table[(ngs_table.GameKey == self.gk) & (ngs_table.PlayID == self.pid) & (ngs_table.GSISID == py2[0])].sort_values('Time')[['Time','x','y','o','dir']].set_index('Time')
        return pd.merge(p1,p2,left_index=True,right_index=True,suffixes=['p1','p2'])
    
    def find_moment_of_intersection(self,df):
        df['distance'] = df.apply(lambda x: np.sqrt(abs(x.xp1-x.xp2) + abs(x.yp1 - x.yp2)),axis=1)
        return df.distance.idxmin()
    
    ## Looking at the injuring players body vs injured players head
    def find_blindness(self, coords,moment):
        impact = coords.loc[moment]
        return (impact.op1 - impact.dirp2)
result_frame = pd.DataFrame(columns=['Season_Year','GameKey','PlayID','Angle'])
for i in inj.iterrows():
    try:
        finder = FindTheBoom(inj,gamekey=i[1].GameKey,playid=i[1].PlayID,seasonyear=i[1].Season_Year)
        p1,p2 = finder.find_partners()
        coords = finder.find_coords(py1 = p1, py2 = p2)
        print(coords)
        moment_of_intersection = finder.find_moment_of_intersection(coords)
        angle = finder.find_blindness(coords=coords,moment=moment_of_intersection)
        result_frame = result_frame.append({'Season_Year':i[1].Season_Year,'GameKey':i[1].GameKey,
                                            'PlayID':i[1].PlayID,'Angle':angle},ignore_index=True)
    except:
        result_frame = result_frame.append({'Season_Year':i[1].Season_Year,'GameKey':i[1].GameKey,
                                            'PlayID':i[1].PlayID,'Angle':None},ignore_index=True)
result_frame
injb = inj.copy()
injb = injb.merge(result_frame, on =['Season_Year','GameKey','PlayID'])
injb['is_blindside'] = injb.Angle.apply(lambda x: 1 if abs(x) < 120 else 0)
injb.is_blindside.value_counts()
plt.figure(figsize=(16, 16))
sn.set(context='paper')
sn.catplot(x='is_returned',y='concussion',data=merge_data, kind='point')
plt.ylabel('Concussion Rate')
plt.xlabel('Is Returned')
plt.xticks([0,1],['False','True'])
plt.show()
display(pd.crosstab(merge_data.concussion,merge_data.is_returned))
display(pd.crosstab(merge_data.concussion,merge_data.is_returned,normalize='columns'))
plt.figure(figsize=(16, 16))
sn.set(context='paper')
sn.catplot(x='is_muff',y='concussion',data=merge_data, kind='point')
plt.ylabel('Concussion Rate')
plt.xlabel('Is Muff')
plt.xticks([0,1],['False','True'])
plt.show()
inj.head()
pd.DataFrame(inj.Role_consol.value_counts(normalize=True))
pd.DataFrame(inj.Kick_ret.value_counts(normalize=True))
pd.DataFrame(inj.Position_consol.value_counts(normalize=True))
pd.DataFrame(inj.Primary_Impact_Type.value_counts(normalize=True))
pd.crosstab(inj.Player_Activity_Derived,inj.Primary_Partner_Activity_Derived)
OHE_cols = ['Season_Type','Game_Day','Game_Site','StadiumType_consol','intl_cities','Turf_consol','penalty_type']
OHE = pd.get_dummies(merge_data[OHE_cols])
other_cols = [col for col in merge_data.columns if col not in OHE_cols]
not_OHE = merge_data[other_cols]

coded_data = not_OHE.merge(OHE,left_index=True,right_index=True)
display(coded_data.columns)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

y = coded_data.concussion
X = coded_data.drop('concussion',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,stratify=y)
rfc = RandomForestClassifier(n_estimators=3000,class_weight='balanced')
rfc.fit(X=X_train,y=y_train)
feat_imp = pd.DataFrame(list(zip(X_train.columns,rfc.feature_importances_)),columns=['feat','imp'])
feat_imp = feat_imp.set_index('feat')
display(feat_imp.sort_values('imp',ascending=False))
coded_data[coded_data.concussion==1].score_diff.plot(kind='hist',density=True,label='concussion',alpha=.5)
coded_data[coded_data.concussion==0].score_diff.plot(kind='hist',density=True,label='no_concussion',alpha=.5)
plt.xlabel('score differential')
plt.legend()
coded_data[coded_data.concussion==1].yards_from_own_endzone.plot(kind='hist',density=True,label='concussion',alpha=.5)
coded_data[coded_data.concussion==0].yards_from_own_endzone.plot(kind='hist',density=True,label='no_concussion',alpha=.5)
plt.xlabel('yards from own endzone')
plt.legend()
plt.rcParams['figure.figsize'] = (20,20)
sn.factorplot(y='yards_from_own_endzone',x='is_returned',data=coded_data,ci=95)
plt.xlabel('Is Returned')
plt.ylabel('Yards from own endzone')
plt.xticks([0,1],['False','True'])
plt.show()
plt.rcParams['figure.figsize'] = (20,20)
sn.factorplot(y='score_diff',x='concussion',data=coded_data,ci=95)
plt.xlabel('Is Concussion')
plt.ylabel('Score differntial')
plt.xticks([0,1],['False','True'])
plt.show()
plt.rcParams['figure.figsize'] = (20,20)
sn.factorplot(y='sec_elapsed_half',x='concussion',data=coded_data,ci=95)
plt.xlabel('Is Concussion')
plt.ylabel('Seconds elapsed in the half')
plt.xticks([0,1],['False','True'])
plt.show()
