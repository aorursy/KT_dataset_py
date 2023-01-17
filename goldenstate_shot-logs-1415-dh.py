from collections import defaultdict



import pandas as pd

import numpy as np



from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier

from sklearn.preprocessing import LabelEncoder



import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
df = pd.read_csv('../input/shot_logs.csv')
# check for null values



df.isnull().any()
# drop null values for SHOT_CLOCK

df = df.dropna()
def ReverseName(s):

    if ',' not in s:

        return s

    

    last,first = [x.strip() for x in s.split(',')]

    result = first + ' ' + last

    

    return result



# make player names easier to read

df['player_name'] = df['player_name'].apply(lambda x: x.title())

    

# fix defender names

df.loc[:,'CLOSEST_DEFENDER'] = df['CLOSEST_DEFENDER'].apply(ReverseName)
def ClockToSeconds(row):

    t,p = row

    

    minutes,seconds = [int(x) for x in t.split(':')]

    

    QTR_LEN = 60*12

    OT_PER_LEN = 60*5

    

    # regulation length game

    if p <= 4:

        prev_qtrs = QTR_LEN*(p-1)

        cur_qtr = QTR_LEN - (minutes*60) - (seconds)

        

        return prev_qtrs + cur_qtr

    

    # overtime game (5 minute periods)

    else:

        regulation = 4*QTR_LEN

        prev_ot_pers = OT_PER_LEN*(p-5)

        cur_ot_per = OT_PER_LEN - (minutes*60) - (seconds)

        

        return regulation + prev_ot_pers + cur_ot_per



def IsContested(def_dist):

    '''returns true if defender is within four feet'''

    if def_dist <= 4:

        return 1

    return 0

    

def IsLongTwo(row):

    shot_dst, points_type = row

    

    if (points_type == 2) and (shot_dst > 16):

        return 1

    return 0    





# convert game clock to seconds    

df['game_clock_seconds'] = df[['GAME_CLOCK','PERIOD']].apply(ClockToSeconds,axis=1)



# mark contested shots

df['contested_shot'] = df['CLOSE_DEF_DIST'].apply(IsContested)



# dribbles per second of touch time

df['dribble_rate'] = df['DRIBBLES']/df['TOUCH_TIME']



# SHOT_RESULT is equivalent to FGM

df.drop('SHOT_RESULT',inplace=True,axis=1)
# separate columns by data type

d = df.columns.to_series().groupby(df.dtypes).groups

d = {key.name:value for key, value in d.items()}



cat_cols = list(d['object'])

num_cols = list(d['float64']) + list(d['int64'])



def ShiftItem(list1,list2,item):

    list1.remove(item)

    list2.append(item)



cats = ['PERIOD','PTS_TYPE','FGM','PTS','contested_shot']



# move categorical columns

for item in cats:

    ShiftItem(num_cols,cat_cols,item)
# disregard ID columns and name cols

exclude_num = ['player_id','CLOSEST_DEFENDER_PLAYER_ID','GAME_ID']

exclude_cat = ['CLOSEST_DEFENDER', 'player_name','MATCHUP','GAME_CLOCK']



num_cols = [x for x in num_cols if x not in exclude_num]

cat_cols = [x for x in cat_cols if x not in exclude_cat]



print(len(num_cols),num_cols)

print(len(cat_cols),cat_cols)
sns.set_style('white')

sns.set_context('notebook')



f,a = plt.subplots(len(num_cols),1,figsize=(6,8))

f.suptitle('Numerical Variables',fontsize=20)

f.subplots_adjust(hspace=0.75)

a = a.ravel()



for i,ax in enumerate(a):

    df[num_cols[i]].plot(vert=False,kind='box',ax=ax)
# drop negative touch time values TOUCH_TIME

df = df[~(df.TOUCH_TIME <= 0)]



# exclude incorrectly classified two's and three's

df = df[~((df.PTS_TYPE==2) & (df.SHOT_DIST>23.75))]

df = df[~((df.PTS_TYPE==3) & (df.SHOT_DIST<=22))]
def ShotType(row):    

    names = {2:[(4,'0-4 (point blank)'),

                (12,'4-12 (close range)'),

                (16,'12-16 (mid-range)'),

                (23.75,'16-23.75 (long two)')],

             3:[(28,'23.75-28 (three)'),

                (35,'28-35 (deep three)'),

                (94,'35+ (extreme)')]}



    pts, dist = row

    

    bins = [i[0] for i in names[pts]]

    ind = np.searchsorted(bins,dist)

    

    return names[pts][ind][1]



# classify shot type

df['shot_type'] = df[['PTS_TYPE','SHOT_DIST']].apply(ShotType,axis=1)



cat_cols.append('shot_type')
# plot histograms to identify incorrect values

sns.set_context('notebook')

sns.set_style('white')



f,a = plt.subplots(2,4,figsize=(8,6))

f.suptitle('Categorical Variable Value Counts (in thousands)',fontsize=18)

f.subplots_adjust(wspace=0.6)

a = a.ravel()



for i,ax in enumerate(a):

    (df[cat_cols[i]].value_counts(dropna=False)/1000).plot(kind='barh',ax=ax)

    ax.set_title(cat_cols[i])
from pandas.tools.plotting import scatter_matrix



plotcols = [col for col in num_cols if col not in ['long_two','FINAL_MARGIN']]



scatter_matrix(df[plotcols],

               alpha=0.2,

               figsize=(10, 10),

               diagonal='kde');
df[plotcols].corr()
# distributions by made and missed shots

pcols = ['SHOT_CLOCK','TOUCH_TIME',

         'SHOT_DIST','CLOSE_DEF_DIST',

         'DRIBBLES','game_clock_seconds']



colors = ['r','g']



f, a = plt.subplots(3,2,figsize=(9,7))

f.suptitle('Distributions by Shot Result',fontsize=18)

f.subplots_adjust(hspace=0.25)



a = a.ravel()



df_ = df.groupby('FGM')



for i,ax in enumerate(a):

    for name,group in df_:        

        group[pcols[i]].plot(kind='hist',bins=25,histtype='step',

                             color=colors[name],

                             legend=True,

                             title=pcols[i],

                             ax=ax,lw=1)

        ax.legend(['missed','made'])

        ax.set_ylabel('')
# distributions for 2 and 3 pointers

colors = {2:'b',3:'g'}



f, a = plt.subplots(3,2,figsize=(9,7))

f.suptitle('Distribution by Points Type',fontsize=18)

f.subplots_adjust(hspace=0.25)



a = a.ravel()



df_ = df.groupby('PTS_TYPE')



for i,ax in enumerate(a):

    for name,group in df_:        

        group[pcols[i]].plot(kind='hist',bins=20,histtype='step',

                             color=colors[name],

                             legend=True,

                             title=pcols[i],

                             ax=ax,lw=1)

        ax.legend(['two','three'])

        ax.set_ylabel('')
sns.set_context('notebook')



df_ = df[df.DRIBBLES<=10]



pt = df_.pivot_table(index=['DRIBBLES','PTS_TYPE'],

                     aggfunc={'FGM':{'attempts':np.size,

                                     'made':np.sum,

                                     'fg_pct':np.mean}})



pt['FGM']['fg_pct'].unstack('PTS_TYPE').plot(title='Average FG Percentage and Dribbles',

                                             kind='line',

                                             rot=0);
fn = {'FGM':{'attempts':'size','made':np.sum,'fg_pct':np.mean},

      'SHOT_DIST':{'avg_shot_dist':np.mean},

      'CLOSE_DEF_DIST':{'avg_def_dist':np.mean},

      'DRIBBLES':{'avg_dribbles':np.mean}}



pt = df.pivot_table(index=['player_name','PTS_TYPE'],

                    aggfunc=fn)



pt.columns = pt.columns.droplevel(0)



#pt.head(4)
f,a = plt.subplots(1,2,figsize=(8,6))

f.suptitle('Distributions by Shot Result',fontsize=16)



a = a.ravel()



shots = [(3,82),(2,200)]



for i,ax in enumerate(a):

    shot,thresh = shots[i]

    

    df_ = pt.xs(shot,level=1)

    df_ = df_[df_.made >= thresh].sort_values('fg_pct',ascending=False)[:20]

    

    scatter = df_.plot(x='avg_dribbles',

                      y='fg_pct',

                      kind='scatter',s=50,ax=ax,

                      title='{} Point FG Percentage'.format(str(shot)))

    

    for txt in df_.index:

        scatter.annotate(txt,xy=(df_.avg_dribbles[txt],df_.fg_pct[txt]))

    
df_ = df[df.contested_shot==1]



fn = {'FGM':{'attempts':np.size,

             'made':np.sum,

             'fg_pct':np.mean},

      'CLOSE_DEF_DIST':{'avg_def_dist':np.mean},

      'DRIBBLES':{'avg_dribbles':np.mean},

      'TOUCH_TIME':{'avg_ttime':np.mean}}



by_stype = df_.groupby(['shot_type']).agg(fn)

by_stype.columns = by_stype.columns.droplevel(0)



by_stype = by_stype[by_stype.index != '35+ (extreme)'].sort_values('fg_pct')
sns.set_context('notebook')

cols = ['fg_pct', 'avg_def_dist', 'avg_ttime','avg_dribbles']



by_stype[cols].plot.barh(title='Contested FGs by Shot Type');
fn = {'FGM':{'attempts':'size',

             'made':np.sum,

             'fg_pct':np.mean},

     'CLOSE_DEF_DIST':{'avg_def_dist':np.mean},

     'DRIBBLES':{'avg_dribbles':np.mean}}



by_stype = df_.groupby(['shot_type','CLOSEST_DEFENDER']).agg(fn)

by_stype.columns = by_stype.columns.droplevel(0)
sns.set_context('notebook')



stypes = [stype for stype in by_stype.index.get_level_values(0).unique() if stype != '35+ (extreme)']



f,a = plt.subplots(len(stypes),1,figsize=(5,16))

f.suptitle('Players with Contested FG percentage by Shot Type',fontsize=18)

a = a.ravel()



for i,ax in enumerate(a):

    shots = by_stype.loc[stypes[i]]

    shots = shots[shots.attempts > shots.attempts.quantile(0.50)]

    

    shots['fg_pct'].sort_values(ascending=True).head(10)[::-1].plot(kind='barh',ax=ax,title=stypes[i])

    
cat_cols = [col for col in cat_cols if col not in ['W','FGM','PTS']]

num_cols = [col for col in num_cols if col not in ['FINAL_MARGIN']]
y = df['FGM']



# encode categorical columns

labels = defaultdict(LabelEncoder)



X_cat = df[cat_cols]

X_cat = X_cat.apply(lambda x: labels[x.name].fit_transform(x))



# combine with numerical data



X = df[num_cols].join(X_cat)
forest = ExtraTreesClassifier(n_estimators=250,random_state=0)



forest.fit(X, y)

importances = forest.feature_importances_
imps = pd.DataFrame({'feature':X.columns,'importance':importances})



imps.sort_values('importance',ascending=True).plot(x='feature',

                                                   kind='barh',

                                                   legend=False,

                                                   title='Feature Importances');