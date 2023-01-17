import pandas as pd

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.tree import export_graphviz

from sklearn.externals.six import StringIO

from IPython.display import Image

import seaborn as sns 

import numpy as np
# Import Data Files and display length 

ir = pd.read_csv("../input/nfl-playing-surface-analytics/InjuryRecord.csv")

pl = pd.read_csv("../input/nfl-playing-surface-analytics/PlayList.csv").set_index("PlayKey")

ptd = pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv").set_index("PlayKey")

l1 = len(ir)

l2 = len(pl)

l3 = len(ptd)

print("Length of InjuryRecord is ", l1)

print("Length of PlayList is ", l2)

print("Length of PlayerTrackData is ", l3)
# Modifying InjuryRecord: Adding PlayKey for missing rows and Merging DM columns (indicate severity) into one column. 

def fdm(d):

    DM=[1,7,28,42]

    return DM[d-1] if d else 0



ir = ir.fillna("*")



ir['dmi'] = ir['DM_M1'] + ir['DM_M7'] + ir['DM_M28'] + ir['DM_M42']

ir['DM'] = ir['dmi'].apply(fdm)



gidpk = {}

for gid in ir.query('PlayKey == "*"').GameID:

    gidpk[gid] = pl.query('GameID=="%s"' % gid).index[-1]





def fpku(a,b):

    return gidpk[b] if a == "*" else a



ir['PlayKey'] = ir.apply(lambda x: fpku(x.PlayKey, x.GameID), axis=1)



InjuryRecord = ir.set_index("PlayKey")[['BodyPart','DM']]

InjuryRecord.head()
# joining InjuryRecord and PlayList 

IRPL = pl.join(InjuryRecord).fillna(0)

IRPL.head()
# does a check to ensure that the # of injuries in the dataset is 105. 

IRPL.query('DM > 0').count()
fig = plt.figure(figsize=(20,16))

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(223)

ax4 = fig.add_subplot(224)

ax1.set_title('# Injuries by Body Part')

ax1.set_ylabel('Count')

ax1.set_xlabel('Body Part')

ax2.set_title('# Injuries by Field Type')

ax2.set_ylabel('Count')

ax2.set_xlabel('Body Part')

ax3.set_title('# Natural Grass: Injuries by Body Part by Severity')

ax3.set_ylabel('Count')

ax3.set_xlabel('Body Part')

ax3.set_ylim(0,12)

ax4.set_title('# Synthetic Turf: Injuries by Body Part by Severity')

ax4.set_ylabel('Count')

ax4.set_xlabel('Body Part')

ax4.set_ylim(0,12)

plot1 = IRPL.query('DM>0').pivot_table(index=['BodyPart'], columns=[], values='DM', aggfunc='count').fillna(0).plot(kind='bar',ax=ax1,legend=False)

plot2 = IRPL.query('DM>0').pivot_table(index=['BodyPart'], columns=['FieldType'], values='DM', aggfunc='count').fillna(0).plot(kind='bar',ax=ax2)

plot3 = IRPL.query('DM>0 and FieldType == "Natural"').pivot_table(index=['BodyPart'], columns=['DM'], values='PlayerKey', aggfunc='count').fillna(0).plot(kind='bar',ax=ax3)

plot4 = IRPL.query('DM>0 and FieldType == "Synthetic"').pivot_table(index=['BodyPart'], columns=['DM'], values='PlayerKey', aggfunc='count').fillna(0).plot(kind='bar',ax=ax4)
fig = plt.figure(figsize=(12,4))

ax1 = fig.add_subplot(111)

ax1.set_title('# Injuries by Position and Body Part')

ax1.set_ylabel('Count')

ax1.set_xlabel('Position Group, Position')

IRPL.query('DM>0').pivot_table(index=['PositionGroup','Position'], columns=['BodyPart'], values='DM', aggfunc='count').fillna(0).plot(kind='bar', ax = ax1)

plt.show()
IRPL.query('DM>0').pivot_table(index=['PositionGroup', 'Position','FieldType'], columns=['BodyPart'], values='DM',aggfunc=['count', 'sum']).fillna(0)
IRPL.query('DM>0').pivot_table(index=['PlayType', 'FieldType'], columns=['BodyPart'], values='DM',aggfunc=['count', 'sum']).fillna(0)
fig = plt.figure(figsize=(12,4))

ax1 = fig.add_subplot(111)

ax1.set_title('# Injuries by Play Type and Body Part')

ax1.set_ylabel('Count')

ax1.set_xlabel('Play Type')

IRPL.query('DM>0').pivot_table(index=['PlayType'], columns=['BodyPart'], values='DM', aggfunc='count').fillna(0).plot(kind='bar', ax = ax1)

plt.show()
def ftoint(d):

    l = list(set(d))

    ld = {l[k]:k for k in range(len(l))}



    def fmap(x):

        return ld[x]

    

    return fmap



def fint(s, k):

    f =  ftoint(IRPL[s])

    IRPL[k] = IRPL[s].apply(f)
fint('RosterPosition', 'irp')

fint('StadiumType', 'istadium')

fint('FieldType', 'ifield')

fint('Weather', 'iweather')

fint('PlayType', 'iplay')

fint('Position', 'iposition')

fint('PositionGroup', 'igroup')

fint('BodyPart', 'ibody')



fig = plt.figure(figsize=(20,20))

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222)

ax1.set_title('Heat Map of Correlations of Features to Injury')

ax2.set_title('Heat Map of Correlations of Features')

sns.heatmap(IRPL.query('DM > 0')[['PlayerDay','PlayerGame','PlayerGamePlay','Temperature','iweather','ifield','istadium','iposition','igroup','ibody']].corr(),ax=ax1)

sns.heatmap(IRPL[['PlayerDay','PlayerGame','PlayerGamePlay','Temperature','iweather','ifield','istadium','iposition','igroup','ibody']].corr(),ax=ax2)

plt.show()
# Change this to change the player and play being looked at: 

PLAYER='43518'

PK='43518-6-25'
PK0 = IRPL.query('PlayerKey==%s' % PLAYER).index[0]

PKN = IRPL.query('PlayerKey==%s' % PLAYER).index[-1]

(PK0, PKN)



df1 = ptd[PK0:PKN]

df2 = IRPL[PK0:PKN]



df = df1.join(df2).fillna(0)



df['rdir'] = df['dir'] * np.pi / 180.



PK = '43518-1-12'

dfx = df.query('PlayKey == "%s"' % PK).reset_index().set_index('time')



funwrap = lambda c : np.unwrap(dfx[c], discont=180.)

fdiff = lambda c : abs(dfx[[c]].diff(axis=0).fillna(0))



dfx['udir'] = funwrap('dir')

dfx['vudir'] = fdiff('udir')



dfx['uo'] = funwrap('o')

dfx['vuo'] = fdiff('uo')

fbins = lambda c, b : pd.cut(x=dfx[c], bins=b, labels=b[1:])

sbins = [0., 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 5.0 ]

scols = ['cs_%d' % int(i*100) for i in sbins[1:] ]

dfx['s_bin'] = fbins('s', sbins)



bins = [0,30,60,90,120,150,180,360]

cols1 = ['cdir_%d' % i for i in bins[1:]]

dfx['vudir_bin'] = fbins('vudir', bins)



cols2 = ['co_%d' % i for i in bins[1:]]

dfx['vuo_bin'] = fbins('vuo', bins)



fzip = lambda c, h : dict(zip(h, list(dfx.groupby(c)[c].count())))



x = fzip('vudir_bin', cols1)

x.update(fzip('vuo_bin', cols2))

x.update(fzip('s_bin', scols))



#dfx.groupby('s_bin')['s_bin'].count()
x = dfx.groupby('vudir_bin')['vudir_bin'].count()

cols = ['cdir_%d' % i for i in bins[1:]]

(cols,list(x))



dict(zip(cols,list(x)))



fig = plt.figure(figsize=(20,12))

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(223)

ax4 = fig.add_subplot(224)



ax1.set_title('Player Direction Over Time')

ax1.set_ylabel('Degrees')

ax1.set_xlabel('time (sec)')

ax2.set_title('Change in Direction Over Time')

ax2.set_ylabel('Degrees')

ax2.set_xlabel('time (sec)')

ax3.set_title('Player Velocity Over Time')

ax3.set_ylabel('yards/sec')

ax3.set_xlabel('time (sec)')

ax4.set_title('Change in Orientation Over Time')

ax4.set_ylabel('degrees')

ax4.set_xlabel('time (sec)')

ax4.set_ylim(-90,90)





dfx['udir'].plot(ax=ax1)

dfx['vudir'].plot(ax=ax2)

dfx['s'].plot(ax=ax3)

dfx['vuo'].plot(ax=ax4)

dfx.query('s_bin <= 0.05')['s'].plot(style='.', ax=ax3)

plt.show()
# Flattening Player Tracking Data in order to decrease the amount of data points to one data point per play

# Classified changes in direction and orientation by binning  in 30 degree buckets

# Characterized velocity in the x and y direction by binning in buckets of 0.01 yards/second to 5 yards/second. 



class Play(object):

    

    def __init__(self):

        self.header=list(pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv",nrows=2).keys())

        self.current_play_df = None

        self.next_row = 1

    

    def get_play1(self):

        dft_prev = None

        n = 0

        for df in pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv",names=self.header,chunksize=1000,skiprows=1):

            n = n + 1

            dfc = df[['PlayKey']]

            pkeys_j = dfc.ne(dfc.shift()).apply(lambda x: x.index[x].tolist())

            pkeys = [dfc.PlayKey[j] for j in pkeys_j[0]]

            for playkey in pkeys[0:-1]:

                if dft_prev is None:

                    dft = df.query('PlayKey == "%s"' % playkey)

                else:

                    dft = dft_prev.append(df.query('PlayKey == "%s"' % playkey))

                    dft_prev = None

                

                yield dft.reset_index()

                

            dft_prev = df.query('PlayKey == "%s"' % pkeys[-1])

            

    

    def next_play(self):

        dft = pd.read_csv("../input/nfl-playing-surface-analytics/PlayerTrackData.csv",names=header,skiprows=self.next_row,nrows=1000)

        playkey = dft.PlayKey[0]

        self.current_play_df = dft.query('PlayKey == "%s"' % playkey)

        l = len(self.current_play_df.PlayKey)

        self.next_row = self.next_row + l

        return self.current_play_df

    

    def get_playkey(self):

        return self.current_play_df.PlayKey[0]

    

    def get_events(self):

        return set(self.current_play_df.event)

    

    def get_plays(self, n):

        

        sbins = [0., 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 5.0 ]

        scols = ['cs_%d' % int(i*100) for i in sbins[1:] ]

        scolsx = ['cvx_%d' % int(i*100) for i in sbins[1:] ]

        scolsy = ['cvy_%d' % int(i*100) for i in sbins[1:] ]

        

        bins = [0,30,60,90,120,150,180,360]

        cols1 = ['cdir_%d' % i for i in bins[1:]] 

        cols2 = ['co_%d' % i for i in bins[1:]] 



        funwrap = lambda dfx, c : np.unwrap(dfx[c], discont=180.)

        fdiff = lambda dfx, c : abs(dfx[[c]].diff(axis=0).fillna(0))

        fbins = lambda dfx, c, b : pd.cut(x=dfx[c], bins=b, labels=b[1:])

        fzip = lambda dfx, c, h : dict(zip(h, list(dfx.groupby(c)[c].count())))

        

        data = []

        t0 = datetime.now()

        i = 0

        for df in self.get_play1():

            d = {'PlayKey': df.PlayKey[0]}

            last = len(df) - 1

            d['Events'] = list(set(df.event))

            d['Duration'] = df.time[last]

            d['TotalDis'] = df.dis.sum()

            

            df1 = df.set_index('time')

            df1['udir'] = funwrap(df1, 'dir')

            df1['vudir'] = fdiff(df1, 'udir')           

            df1['uo'] = funwrap(df1, 'o')

            df1['vuo'] = fdiff(df1, 'uo')

            

            df1['vx'] = fdiff(df1, 'x')

            df1['vy'] = fdiff(df1, 'y')

            

            df1['vx_bin'] = fbins(df1, 'vx', sbins)

            df1['vy_bin'] = fbins(df1, 'vy', sbins)



            df1['vudir_bin'] = fbins(df1, 'vudir', bins)

            df1['vuo_bin'] = fbins(df1, 'vuo', bins)

            

            d.update(fzip(df1, 'vudir_bin', cols1))

            d.update(fzip(df1, 'vuo_bin', cols2))

            d.update(fzip(df1, 'vx_bin', scolsx))

            d.update(fzip(df1, 'vy_bin', scolsy))

            

            data.append(d)

            i = i + 1

            if i%10000 == 0:

                t1 = datetime.now()

                print(i,t1-t0)

                t0=t1

            if i >= n: break

        return data

    



print("Start:", datetime.now())

play = Play()

data = play.get_plays(300000)

dfplays = pd.DataFrame(data).set_index('PlayKey')

print("End:", datetime.now())



dfplays.to_parquet('plays4.parquet', compression='GZIP')

dfplays.to_csv('plays4.csv')

df = dfpi.join(dfplays).fillna(0)

df.to_csv("df1.csv")
ptd2 = pd.read_csv("../input/joined2/df2.csv")

#ptd2.keys()
ptd2['cdir_total'] = ptd2['cdir_60']  + ptd2['cdir_90'] + ptd2['cdir_120'] + ptd2['cdir_150'] + ptd2['cdir_180']

ptd2['co_total'] = ptd2['co_60'] + ptd2['co_90'] + ptd2['co_120'] + ptd2['co_150'] + ptd2['co_180']

ptd2['cdir_rate'] = ptd2['cdir_total'] / ptd2['Duration']

ptd2['co_rate'] = ptd2['co_total'] / ptd2['Duration']

ptd2['sum_vx'] = ptd2['cvx_1'] + ptd2['cvx_5'] + ptd2['cvx_10'] + ptd2['cvx_20'] + ptd2['cvx_30'] + ptd2['cvx_40'] + ptd2['cvx_50'] + ptd2['cvx_100'] + ptd2['cvx_500']    

ptd2['sum_vy'] = ptd2['cvy_1'] + ptd2['cvy_5'] + ptd2['cvy_10'] + ptd2['cvy_20'] + ptd2['cvy_30'] + ptd2['cvy_40'] + ptd2['cvy_50'] + ptd2['cvy_100'] + ptd2['cvy_500'] 

ptd2['cvx_rate'] = ptd2['cvx_1'] / ptd2['Duration']

ptd2['cvy_rate'] = ptd2['cvy_1'] / ptd2['Duration']

ptd2['avg_velocity'] = ptd2['TotalDis'] / ptd2['Duration']

ptd2.head()
fig = plt.figure(figsize=(20,18))

ax1 = fig.add_subplot(321)

ax2 = fig.add_subplot(322)

ax3 = fig.add_subplot(323)

ax4 = fig.add_subplot(324)

ax5 = fig.add_subplot(325)

ax6 = fig.add_subplot(326)



ax1.set_title('Injury: Average # Slows/Stops per Second')

ax1.set_ylabel('Average')

ax1.set_xlabel('Field Type')

ax1.set_ylim(0,1)



ax2.set_title('No Injury: Average # Slows/Stops per Second')

ax2.set_ylabel('Average')

ax2.set_xlabel('Field Type')

ax2.set_ylim(0,1)



ax3.set_title('Injury: Average # Changes in Direction per Second')

ax3.set_ylabel('Average')

ax3.set_xlabel('Field Type')

ax3.set_ylim(0,0.25)



ax4.set_title('No Injury: Average # Changes in Direction per Second')

ax4.set_ylabel('Average')

ax4.set_xlabel('Field Type')

ax4.set_ylim(0,0.25)



ax5.set_title('Injury: Average # Changes in Orientation per Second')

ax5.set_ylabel('Average')

ax5.set_xlabel('Field Type')

ax5.set_ylim(0,0.040)



ax6.set_title('No Injury: Average # Changes in Orientation per Second')

ax6.set_ylabel('Average')

ax6.set_xlabel('Field Type')

ax6.set_ylim(0,0.040)



ptd2.query('DM > 0').groupby(['FieldType']).mean()[['cvx_rate', 'cvy_rate']].plot(kind='bar', ax = ax1)

ptd2.query('DM == 0').groupby(['FieldType']).mean()[['cvx_rate', 'cvy_rate']].plot(kind='bar', ax = ax2)

ptd2.query('DM > 0').groupby(['FieldType']).mean()['cdir_rate'].plot(kind='bar', ax = ax3)

ptd2.query('DM == 0').groupby(['FieldType']).mean()['cdir_rate'].plot(kind='bar', ax = ax4)

ptd2.query('DM > 0').groupby(['FieldType']).mean()['co_rate'].plot(kind='bar', ax = ax5)

ptd2.query('DM == 0').groupby(['FieldType']).mean()['co_rate'].plot(kind='bar', ax = ax6)

plt.show()
fig = plt.figure(figsize=(20,18))

ax1 = fig.add_subplot(221)

ax2 = fig.add_subplot(222)

ax3 = fig.add_subplot(223)

ax4 = fig.add_subplot(224)



ax1.set_title('Injury: Average DM on Synthetic VS Natural')

ax1.set_ylabel('Average')

ax1.set_xlabel('Field Type')



ax2.set_title('Injury: Average # Slows/Stops per Sec on Synthetic VS Natural')

ax2.set_ylabel('Average')

ax2.set_xlabel('Field Type')



ax3.set_title('Injury: Average # Changes in Direction per Sec on Synthetic VS Natural')

ax3.set_ylabel('Average')

ax3.set_xlabel('Field Type')



ax4.set_title('Injury: Average # Changes in Orientation per Second')

ax4.set_ylabel('Average')

ax4.set_xlabel('Field Type')



ptd2.query('DM > 0 and StadiumType == "Outdoor"').groupby(['FieldType']).mean()[['DM']].plot(kind='bar', ax = ax1)

ptd2.query('DM > 0 and StadiumType == "Outdoor"').groupby(['FieldType']).mean()[['cvx_rate', 'cvy_rate']].plot(kind='bar', ax = ax2)

ptd2.query('DM > 0 and StadiumType == "Outdoor"').groupby(['FieldType']).mean()[['cdir_rate']].plot(kind='bar', ax = ax3)

ptd2.query('DM > 0 and StadiumType == "Outdoor"').groupby(['FieldType']).mean()[['co_rate']].plot(kind='bar', ax = ax4)

ptd2.query('DM > 0 and StadiumType == "Outdoor" and Position == "WR"').groupby(['FieldType']).mean()[['DM']].plot(kind='bar', ax = ax1)

plt.show()
# columns considered as inputs to the Decision Tree 

feature_cols = ['Temperature','irp', 'istadium', 'iweather', 'iplay', 'iposition', 'ifield',

       'igroup', 'ibody', 'Duration', 'TotalDis', 'cdir_30',

       'cdir_60', 'cdir_90', 'cdir_120', 'cdir_150', 'cdir_180', 'cdir_360',

       'co_30', 'co_60', 'co_90', 'co_120', 'co_150', 'co_180', 'co_360',

       'cvx_1', 'cvx_5', 'cvx_10', 'cvx_20', 'cvx_30', 'cvx_40', 'cvx_50',

       'cvx_100', 'cvx_500', 'cvy_1', 'cvy_5', 'cvy_10', 'cvy_20', 'cvy_30',

       'cvy_40', 'cvy_50', 'cvy_100', 'cvy_500', 'cdir_total', 'co_total',

       'cdir_rate', 'co_rate', 'sum_vx', 'sum_vy', 'cvx_rate', 'cvy_rate',

       'avg_velocity']

# OUTPUT change this for varying what is predicted 

output = ['DM']
# training on 50% of the data 

data_train = ptd2.sample(frac=0.50).fillna(0)



# testing on 50% of injury data (hoping to avoid cross-referencing with training data)

data_test = ptd2.query('DM > 0').fillna(0)

data_test = data_test.sample(frac=0.50)

data_test.head()
# *** BUILD DECISION TREE USING SCI-KIT LEARN ***

# Create Decision Tree classifer object

clf = DecisionTreeClassifier()



# Train Decision Tree Classifer

clf = clf.fit(data_train[feature_cols], data_train[output])
#Predict the response for test dataset

output_pred = clf.predict(data_test[feature_cols])

data_test['dt_output'] = output_pred

print('Decision Tree Result:')

print(data_test.head(100))
# Model Accuracy, how often is the classifier correct?

print("Accuracy: ",metrics.accuracy_score(data_test[output], output_pred))