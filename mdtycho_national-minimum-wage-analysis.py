import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from os import listdir

%matplotlib inline

sns.set()
pd.set_option('display.max_rows', 142)
pd.set_option('display.max_columns', 142)
listdir()
df1 = pd.read_csv('QLFS-2017-01_F1.csv')

df2 = pd.read_csv('QLFS-2017-02_F1.csv')

df3 = pd.read_csv('QLFS-2017-03_F1.csv')

df4 = pd.read_csv('QLFS-2017-04_F1.csv')

df5 = pd.read_csv('QLFS-2018-01_F1.csv')
to_keep = [
         'UQNO','PERSONNO','Province','Q14AGE','Q15POPULATION','Q17EDUCATION','Q20SELFRESPOND','Q24APDWRK','Q24BOWNBUSNS',
         'Q24CUNPDWRK','Q25APDWRK','Q25BOWNBUSNS','Q25CUNPDWRK','Q26AFARMWRK','Q27RSNABSENT','Q31ALOOKWRK','Q31BSTARTBUSNS',
         'Q3201REGISTER','Q3202ENQUIRE','Q3203JOBADS','Q3204JOBSEARCH','Q3205ASSISTANCE','Q3206STARTBUSNS','Q3207CASUAL',
         'Q3208FINASSIST','Q3210NOTHING','Q33HAVEJOB','Q34WANTTOWRK','Q35YNOTWRK','Q39JOBOFFER','Q311RSNNOTAVAILABLE',
         'Q312EVERWRK','Q41MULTIPLEJOBS','Q416NRWORKERS','Q418HRSWRK','Q419TOTALHRS','Q422MOREHRS','Q423ADDHRS','Unempl_status',
         'Status','Education_Status','age_grp1','Status_Exp'
    ]
# df1 = df1[to_keep]

# df2 = df2[to_keep]

# df3 = df3[to_keep]

# df4 = df4[to_keep]

# df5 = df5[to_keep]
list(df1.columns)
q1_to_keep = [
               'UQNO','PERSONNO','Q15POPULATION','Q14AGE','Q17EDUCATION','Education_Status','Province','age_grp1',
               'Status','Status_exp','Unempl_Status', 'Q416NRWORKERS'
            ]
df1 = df1[q1_to_keep]
df1.info()
df1['Unempl_Status'].unique()
df1['Status'].unique()
df1.drop('Unempl_Status', inplace = True, axis = 1)
df1['Q416NRWORKERS'].unique()
df1_nrworkers = df1.copy()
df1_nrworkers.info()
df1.drop('Q416NRWORKERS', inplace = True, axis = 1)
df1[df1.isnull().any(axis=1)]['Q14AGE'].unique()
df1.dropna(inplace = True)
df1.info()
df1['UQNO'].nunique()
df1['PERSONNO'].unique()
employed = df1[df1['Status_exp'] == 1]

unemployed = df1[df1['Status_exp'] == 2]
employed.shape
unemployed.shape
10969/(10969+18634)
employed_ = df1[df1['Status'] == 1]

unemployed_ = df1[df1['Status'] == 2]
unemployed_.shape[0]/(employed_.shape[0] + unemployed_.shape[0])
y_employed = employed[(employed['age_grp1'] >= 4) & (employed['age_grp1'] <= 5)]

y_unemployed = unemployed[(unemployed['age_grp1'] >= 4) & (unemployed['age_grp1'] <= 5)]
get_rate = lambda x,y: y.shape[0]/(x.shape[0] + y.shape[0])
get_rate(y_employed, y_unemployed)
list(df2.columns)
df2 = df2[q1_to_keep]

df3 = df3[q1_to_keep]

df4 = df4[q1_to_keep]

df5 = df5[q1_to_keep]
df2.drop('Unempl_Status', inplace = True, axis = 1)

df3.drop('Unempl_Status', inplace = True, axis = 1)

df4.drop('Unempl_Status', inplace = True, axis = 1)

df5.drop('Unempl_Status', inplace = True, axis = 1)
df2_nrworkers = df2.copy()

df3_nrworkers = df3.copy()

df4_nrworkers = df4.copy()

df5_nrworkers = df5.copy()
df2.drop('Q416NRWORKERS', inplace = True, axis = 1)

df3.drop('Q416NRWORKERS', inplace = True, axis = 1)

df4.drop('Q416NRWORKERS', inplace = True, axis = 1)

df5.drop('Q416NRWORKERS', inplace = True, axis = 1)
df2.dropna(inplace = True)

df3.dropna(inplace = True)

df4.dropna(inplace = True)

df5.dropna(inplace = True)
employed1 = df2[df2['Status_exp'] == 1]

unemployed1 = df2[df2['Status_exp'] == 2]
get_rate(employed1, unemployed1)
y_employed1 = employed1[(employed1['age_grp1'] >= 4) & (employed1['age_grp1'] <= 5)]

y_unemployed1 = unemployed1[(unemployed1['age_grp1'] >= 4) & (unemployed1['age_grp1'] <= 5)]
get_rate(y_employed1, y_unemployed1)
employed2 = df3[df3['Status_exp'] == 1]

unemployed2 = df3[df3['Status_exp'] == 2]
get_rate(employed2, unemployed2)
y_employed2 = employed2[(employed2['age_grp1'] >= 4) & (employed2['age_grp1'] <= 5)]

y_unemployed2 = unemployed2[(unemployed2['age_grp1'] >= 4) & (unemployed2['age_grp1'] <= 5)]
get_rate(y_employed2, y_unemployed2)
employed3 = df4[df4['Status_exp'] == 1]

unemployed3 = df4[df4['Status_exp'] == 2]
get_rate(employed3, unemployed3)
y_employed3 = employed3[(employed3['age_grp1'] >= 4) & (employed3['age_grp1'] <= 5)]

y_unemployed3 = unemployed3[(unemployed3['age_grp1'] >= 4) & (unemployed3['age_grp1'] <= 5)]
get_rate(y_employed3, y_unemployed3)
employed4 = df5[df5['Status_exp'] == 1]

unemployed4 = df5[df5['Status_exp'] == 2]
get_rate(employed4, unemployed4)
y_employed4 = employed4[(employed4['age_grp1'] >= 4) & (employed4['age_grp1'] <= 5)]

y_unemployed4 = unemployed4[(unemployed4['age_grp1'] >= 4) & (unemployed4['age_grp1'] <= 5)]
get_rate(y_employed4, y_unemployed4)
y_changes = pd.Series([
           0,
           get_rate(y_employed1, y_unemployed1) - get_rate(y_employed, y_unemployed), 
           get_rate(y_employed2, y_unemployed2) - get_rate(y_employed1, y_unemployed1),
           get_rate(y_employed3, y_unemployed3) - get_rate(y_employed2, y_unemployed2),
           get_rate(y_employed4, y_unemployed4) - get_rate(y_employed3, y_unemployed3)
          ])*100

changes = pd.Series([
           0,
           get_rate(employed1, unemployed1) - get_rate(employed, unemployed), 
           get_rate(employed2, unemployed2) - get_rate(employed1, unemployed1),
           get_rate(employed3, unemployed3) - get_rate(employed2, unemployed2),
           get_rate(employed4, unemployed4) - get_rate(employed3, unemployed3)
          ])*100

x_axis = pd.Series([0,1, 2, 3, 4])
sns.set()

fig, ax = plt.subplots(figsize = (15,8))

fdict = {'fontsize': 20,
 'fontweight' : 'bold',
 'verticalalignment': 'baseline',
 'horizontalalignment': 'center'}

fdict_label = {'fontsize': 13,
 'fontweight' : 'bold',
 'verticalalignment': 'top',
 'horizontalalignment': 'center'
    
}

ax.plot(x_axis, changes, color = 'blue', marker = 'o')

for xy in zip(x_axis, changes):                                       # <--
    ax.annotate('{}%'.format(round(xy[1], 2)), xy=xy, textcoords='data') # <--

ax.plot(x_axis, y_changes, color = 'orange', marker = 'o')

for xy in zip(x_axis, y_changes):                                       # <--
    ax.annotate('{}%'.format(round(xy[1], 2)), xy=xy, textcoords='data') # 

ax.set_title('Changes In Unemployment Rate Between Q1 2017 and Q1 2018', fontdict = fdict)

ax.legend(['All Age Groups', 'Youth (15-24 years)'], loc = 2)

ax.set_xticks([1, 2, 3, 4, 5])

ax.set_xticklabels(['q2q1', 'q3q2', 'q4q3', 'q1q4'])

ax.set_xlabel('Quarters Being Compared', fontdict = fdict_label)

ax.set_ylabel('Unemployment Change Between Quarters (%)', fontdict = fdict_label)

# ax.set_ylim(bottom = -0.05, top = 0.05)

# ax.set_yticks([-0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04])

fig.savefig('unemployment_changes.png')

plt.show()
listdir()
df1_16 = pd.read_csv('QLFS!2016!01_F1.csv')

df2_16 = pd.read_csv('QLFS-2016-02_F1.csv')

df3_16 = pd.read_csv('QLFS-2016-03_F1.csv')

df4_16 = pd.read_csv('QLFS-2016-04_F1.csv')
df1_16 = df1_16[q1_to_keep]

df2_16 = df2_16[q1_to_keep]

df3_16 = df3_16[q1_to_keep]

df4_16 = df4_16[q1_to_keep]

df1_16_nrworkers = df1_16.copy()

df2_16_nrworkers = df2_16.copy()

df3_16_nrworkers = df3_16.copy()

df4_16_nrworkers = df4_16.copy()
df1_16.drop('Unempl_Status', inplace = True, axis = 1)

df2_16.drop('Unempl_Status', inplace = True, axis = 1)

df3_16.drop('Unempl_Status', inplace = True, axis = 1)

df4_16.drop('Unempl_Status', inplace = True, axis = 1)


df1_16.drop('Q416NRWORKERS', inplace = True, axis = 1)

df2_16.drop('Q416NRWORKERS', inplace = True, axis = 1)

df3_16.drop('Q416NRWORKERS', inplace = True, axis = 1)

df4_16.drop('Q416NRWORKERS', inplace = True, axis = 1)
df1_16.dropna(inplace = True)

df2_16.dropna(inplace = True)

df3_16.dropna(inplace = True)

df4_16.dropna(inplace = True)
employed1_2016 = df1_16[df1_16['Status_exp'] == 1]
unemployed1_2016 = df1_16[df1_16['Status_exp'] == 2]
y_employed1_2016 = employed1_2016[(employed1_2016['age_grp1'] >= 4) & (employed1_2016['age_grp1'] <= 5)]

y_unemployed1_2016 = unemployed1_2016[(unemployed1_2016['age_grp1'] >= 4) & (unemployed1_2016['age_grp1'] <= 5)]
employed2_2016 = df2_16[df2_16['Status_exp'] == 1]
unemployed2_2016 = df2_16[df2_16['Status_exp'] == 2]
y_employed2_2016 = employed2_2016[(employed2_2016['age_grp1'] >= 4) & (employed2_2016['age_grp1'] <= 5)]

y_unemployed2_2016 = unemployed2_2016[(unemployed2_2016['age_grp1'] >= 4) & (unemployed2_2016['age_grp1'] <= 5)]
employed3_2016 = df3_16[df3_16['Status_exp'] == 1]
unemployed3_2016 = df3_16[df3_16['Status_exp'] == 2]
y_employed3_2016 = employed3_2016[(employed3_2016['age_grp1'] >= 4) & (employed3_2016['age_grp1'] <= 5)]

y_unemployed3_2016 = unemployed3_2016[(unemployed3_2016['age_grp1'] >= 4) & (unemployed3_2016['age_grp1'] <= 5)]
employed4_2016 = df4_16[df4_16['Status_exp'] == 1]
unemployed4_2016 = df4_16[df4_16['Status_exp'] == 2]
y_employed4_2016 = employed4_2016[(employed4_2016['age_grp1'] >= 4) & (employed4_2016['age_grp1'] <= 5)]

y_unemployed4_2016 = unemployed4_2016[(unemployed4_2016['age_grp1'] >= 4) & (unemployed4_2016['age_grp1'] <= 5)]
y_changes_2016 = pd.Series([
           0,
           get_rate(y_employed2_2016, y_unemployed2_2016) - get_rate(y_employed1_2016, y_unemployed1_2016), 
           get_rate(y_employed3_2016, y_unemployed3_2016) - get_rate(y_employed2_2016, y_unemployed2_2016),
           get_rate(y_employed4_2016, y_unemployed4_2016) - get_rate(y_employed3_2016, y_unemployed3_2016),
           get_rate(y_employed, y_unemployed) - get_rate(y_employed4_2016, y_unemployed4_2016)
          ])*100

changes_2016 = pd.Series([
           0,
           get_rate(employed2_2016, unemployed2_2016) - get_rate(employed1_2016, unemployed1_2016), 
           get_rate(employed3_2016, unemployed3_2016) - get_rate(employed2_2016, unemployed2_2016),
           get_rate(employed4_2016, unemployed4_2016) - get_rate(employed3_2016, unemployed3_2016),
           get_rate(employed, unemployed) - get_rate(employed4_2016, unemployed4_2016)
          ])*100

x_axis = pd.Series([0,1, 2, 3, 4])
sns.set()

fig, ax = plt.subplots(figsize = (15,8))

fdict = {'fontsize': 20,
 'fontweight' : 'bold',
 'verticalalignment': 'baseline',
 'horizontalalignment': 'center'}

ax.plot(x_axis, changes_2016, color = 'blue', marker = 'o')

for xy in zip(x_axis, changes_2016):                                       # <--
    ax.annotate('{}%'.format(round(xy[1], 3)), xy=xy, textcoords='data') # 

ax.plot(x_axis, y_changes_2016, color = 'orange', marker = 'o')

for xy in zip(x_axis, y_changes_2016):                                       # <--
    ax.annotate('{}%'.format(round(xy[1], 3)), xy=xy, textcoords='data') # 

ax.set_title('Changes In Unemployment Rate Between Q1 2016 and Q1 2017', fontdict = fdict)

ax.legend(['All Age Groups', 'Youth (15-24 years)'], loc = 2)

ax.set_xticks([1, 2, 3, 4, 5])

ax.set_xticklabels(['q2q1', 'q3q2', 'q4q3', 'q1q4'])

ax.set_xlabel('Quarters Being Compared', fontdict = fdict_label)

ax.set_ylabel('Unemployment Change Between Quarters (%)', fontdict = fdict_label)


# ax.set_ylim(bottom = -0.05, top = 0.05)

# ax.set_yticks([-0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03, 0.04])

fig.savefig('unemployment_changes_2016.png')

plt.show()
df1[df1['Education_Status'] == 3].shape
df1.shape
500/47840
df1.columns
def create_education_var(level):
    if level >= 1 and level < 4:
        return 'lower'
    elif level == 4:
        return 'secondary'
    elif level == 5:
        return 'matric'
    else:
        return 'tertiary'
frames = {'2016': [df1_16, df2_16, df3_16, df4_16], '2017':[df1, df2, df3, df4], '2018':[df5]}

frames_nrworkers = {'2016': [df1_16_nrworkers, df2_16_nrworkers, df3_16_nrworkers, df4_16_nrworkers], 
                    '2017':[df1_nrworkers, df2_nrworkers, df3_nrworkers, df4_nrworkers], 
                    '2018':[df5_nrworkers]}
frames.keys()
for k in frames.keys():
    for df in frames[k]:
        df['education'] = df['Education_Status'].apply(create_education_var)
df1.head()
def create_employment_var(level):
    if level == 1:
        return 1
    elif level == 2:
        return 0
    else:
        return -1
for k in frames.keys():
    for df in frames[k]:
        df['employed'] = df['Status_exp'].apply(create_employment_var)
for k in frames.keys():
    for df in frames[k]:
        df.drop(df.loc[df['employed']==-1].index, inplace=True)
for k in frames.keys():
    for df in frames[k]:
        print(df.shape)
# sns.set_style(style = "white")
sns.set()

fig, ax = plt.subplots(figsize = (15, 8))

fdict_label1 = {'fontsize': 13,
 'fontweight' : 'bold',
 'verticalalignment': 'baseline',
 'horizontalalignment': 'center'
    
}


sns.barplot(x = 'education', y = 'employed', data = df1[(df1['education'] == 'matric') | (df1['education'] == 'tertiary')], order=['matric', 'tertiary'], ax = ax, color="#ff66b3", label = "quarter 1 2017")

sns.barplot(x = 'education', y = 'employed', data = df5[(df5['education'] == 'matric') | (df5['education'] == 'tertiary')], order=['matric', 'tertiary'], ax = ax, color="#80ffaa", label = "quarter 1 2018")

ax.set_title('Change In Quarter-On-Quarter Employment', fontdict = fdict)

ax.set_ylabel('Proportion Employed', fontdict = fdict_label1)

ax.set_xlabel('Education Level', fontdict = fdict_label1)

ax.legend()

fig.savefig('change_unemployment_2017_2018_QonQ.png')

print('matric', 100*(df5[df5['education'] == 'matric']['employed'].mean() - df1[df1['education'] == 'matric']['employed'].mean()))
print()
print('tertiary', 100*(df5[df5['education'] == 'tertiary']['employed'].mean() - df1[df1['education'] == 'tertiary']['employed'].mean()))
sns.set()

fig, ax = plt.subplots(figsize = (15, 8))


sns.barplot(x = 'education', y = 'employed', data = df1_16[(df1_16['education'] == 'matric') | (df1_16['education'] == 'tertiary')], order=['matric', 'tertiary'], ax = ax, color ="#ff66b3", label = "quarter 1 2016")

sns.barplot(x = 'education', y = 'employed', data = df1[(df1['education'] == 'matric') | (df1['education'] == 'tertiary')], order=['matric', 'tertiary'], ax = ax, color="#80ffaa", label = "quarter 1 2017")

ax.set_title('Change In Quarter-On-Quarter Employment', fontdict = fdict)

ax.set_ylabel('Proportion Employed', fontdict = fdict_label1)

ax.set_xlabel('Education Level', fontdict = fdict_label1)

ax.legend()

fig.savefig('change_unemployment_2016_2017_QonQ.png')

print('matric', 100*(df1[df1['education'] == 'matric']['employed'].mean() - df1_16[df1_16['education'] == 'matric']['employed'].mean()))
print()
print('tertiary', 100*(df1[df1['education'] == 'tertiary']['employed'].mean() - df1_16[df1_16['education'] == 'tertiary']['employed'].mean()))
sns.set()

fig, ax = plt.subplots(figsize = (15, 8))


sns.barplot(x = 'education', y = 'employed', data = df1_16[(df1_16['education'] == 'matric') | (df1_16['education'] == 'tertiary')], order=['matric', 'tertiary'], ax = ax, color = "#ff66b3", label = "quarter 1 2016")

sns.barplot(x = 'education', y = 'employed', data = df5[(df5['education'] == 'matric') | (df5['education'] == 'tertiary')], order=['matric', 'tertiary'], ax = ax, color = '#80ffaa', label = "quarter 1 2018")

ax.legend()

print('matric', 100*(df5[df5['education'] == 'matric']['employed'].mean() - df1_16[df1_16['education'] == 'matric']['employed'].mean()))
print()
print('tertiary', 100*(df5[df5['education'] == 'tertiary']['employed'].mean() - df1_16[df1_16['education'] == 'tertiary']['employed'].mean()))
q1_to_keep
df1['Q15POPULATION'].unique()
def get_race(val):
    if val == 1:
        return 'African'
    elif val == 2:
        return 'Coloured'
    elif val == 3:
        return 'Indian'
    elif val == 4:
        return 'White'
    else:
        return 'Unknown'
for k in frames.keys():
    for df in frames[k]:
        df['race'] = df['Q15POPULATION'].apply(get_race)
df1['race'].unique()
sns.set()

fig, ax = plt.subplots(figsize = (15, 8))


sns.barplot(x = 'race', y = 'employed', data = df1, order=['African', 'Coloured', 'Indian', 'White'], ax = ax, color="#666699", label = "quarter 1 2017")

sns.barplot(x = 'race', y = 'employed', data = df5, order=['African', 'Coloured', 'Indian', 'White'], ax = ax, color="#cc3300", label = "quarter 1 2018")

ax.legend()

print('African', 100*(df5[df5['race'] == 'African']['employed'].mean() - df1[df1['race'] == 'African']['employed'].mean()))
print()
print('Coloured', 100*(df5[df5['race'] == 'Coloured']['employed'].mean() - df1[df1['race'] == 'Coloured']['employed'].mean()))
print()
print('Indian', 100*(df5[df5['race'] == 'Indian']['employed'].mean() - df1[df1['race'] == 'Indian']['employed'].mean()))
print()
print('White', 100*(df5[df5['race'] == 'White']['employed'].mean() - df1[df1['race'] == 'White']['employed'].mean()))
df1_nrworkers.info()
df1_nrworkers.dropna().info()
for k in frames_nrworkers.keys():
    for df in frames_nrworkers[k]:
        df.dropna(inplace = True)
        df.shape
for k in frames_nrworkers.keys():
    for df in frames_nrworkers[k]:
        print(df.shape)
frames_nrworkers.keys()
frames_nrworkers.pop('2016', None)
for k in frames_nrworkers.keys():
    for df in frames_nrworkers[k]:
        print(df.shape)
df1_nrworkers.columns
for k in frames_nrworkers.keys():
    for df in frames_nrworkers[k]:
        df.drop(df.loc[(df['Q416NRWORKERS']==1) & (df['Q416NRWORKERS']==8)].index, inplace=True)
df1_nrworkers['Q416NRWORKERS'].unique()
for k in frames_nrworkers.keys():
    for df in frames_nrworkers[k]:
        print(df.shape)
df1_nrworkers.drop(df1_nrworkers.loc[(df1_nrworkers['Q416NRWORKERS']==1) & (df1_nrworkers['Q416NRWORKERS']==8)].index, inplace=True)
df1_nrworkers.shape
df1_nrworkers = df1_nrworkers[df1_nrworkers['Q416NRWORKERS'] > 1]
df1_nrworkers['Q416NRWORKERS'].unique()
df1_nrworkers = df1_nrworkers[df1_nrworkers['Q416NRWORKERS'] < 8]
df1_nrworkers['Q416NRWORKERS'].unique()
for k in frames_nrworkers.keys():
    for df in frames_nrworkers[k]:
        print(df.shape)
df1_nrworkers.shape
df2_nrworkers = df2_nrworkers[df2_nrworkers['Q416NRWORKERS'] > 1]
df2_nrworkers = df2_nrworkers[df2_nrworkers['Q416NRWORKERS'] < 8]

df3_nrworkers = df3_nrworkers[df3_nrworkers['Q416NRWORKERS'] > 1]
df3_nrworkers = df3_nrworkers[df3_nrworkers['Q416NRWORKERS'] < 8]

df4_nrworkers = df4_nrworkers[df4_nrworkers['Q416NRWORKERS'] > 1]
df4_nrworkers = df4_nrworkers[df4_nrworkers['Q416NRWORKERS'] < 8]

df5_nrworkers = df5_nrworkers[df5_nrworkers['Q416NRWORKERS'] > 1]
df5_nrworkers = df5_nrworkers[df5_nrworkers['Q416NRWORKERS'] < 8]
df5_nrworkers.shape
def make_business_size(val):
    
    if val >=2 and val <= 4:
        return 'micro'
    elif val == 5:
        return 'small'
    elif val == 6:
        return 'medium'
    elif val == 7:
        return 'big'
temp_dfs = [df1_nrworkers, df2_nrworkers, df3_nrworkers, df4_nrworkers, df5_nrworkers]

for df in temp_dfs:
    df['business size'] = df['Q416NRWORKERS'].apply(make_business_size)
df1_nrworkers.head()
for df in temp_dfs:
        df['employed'] = df['Status_exp'].apply(create_employment_var)
df1_nrworkers.info()
sns.set()

q1_total_17 = df1.shape[0]

q1_total_18 = df5.shape[0]

q1_micro_17 = (df1_nrworkers[df1_nrworkers['business size'] == 'micro'].shape[0])/q1_total_17

q1_small_17 = (df1_nrworkers[df1_nrworkers['business size'] == 'small'].shape[0])/q1_total_17

q1_medium_17 = (df1_nrworkers[df1_nrworkers['business size'] == 'medium'].shape[0])/q1_total_17

q1_big_17 = (df1_nrworkers[df1_nrworkers['business size'] == 'big'].shape[0])/q1_total_17

q1_micro_18 = (df5_nrworkers[df5_nrworkers['business size'] == 'micro'].shape[0])/q1_total_18

q1_small_18 = (df5_nrworkers[df5_nrworkers['business size'] == 'small'].shape[0])/q1_total_18

q1_medium_18 = (df5_nrworkers[df5_nrworkers['business size'] == 'medium'].shape[0])/q1_total_18

q1_big_18 = (df5_nrworkers[df5_nrworkers['business size'] == 'big'].shape[0])/q1_total_18


# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = ['micro', 'small', 'medium', 'big']
sizes1 = [q1_micro_17*100, q1_small_17*100, q1_medium_17*100, q1_big_18*100]
explode = (0, 0, 0,0)  # only "explode" the 2nd slice (i.e. 'Hogs')

sizes2 = [q1_micro_18*100, q1_small_18*100, q1_medium_18*100, q1_big_18*100]

fig1, ax1 = plt.subplots(figsize = (15, 8))
sns.barplot(x = labels, y = sizes1,order=['micro', 'small', 'medium', 'big'], ax = ax1, color="#666699", label = "quarter 1 2017")

sns.barplot(x = labels, y = sizes2, order=['micro', 'small', 'medium', 'big'], ax = ax1, color= "#cc3300", label = "quarter 1 2018")

plt.show()

print('micro', q1_micro_18 - q1_micro_17)
print()

print('small', q1_small_18 - q1_small_17)
print()

print('medium', q1_medium_18-q1_medium_17)
print()

print('big', q1_big_18 - q1_big_17)
print()
print(-0.0015916721040566684*21000000)
print()
print(-0.0004224181814054401*21000000)
print()
print(-0.0011710774580936667*21000000)
print()
print(-0.0057561440828053645*21000000)
print()
listdir()
earnings = pd.read_csv('LMD!2016_F1.csv')
for_later = 'Q58SALARYCATEGORY'

temp = 'Q57a_monthly'
list(earnings.columns)
earnings.info()
q_to_keep = ['UQNO',
 'PERSONNO',
 'Q15POPULATION',
 'Q14AGE',
 'Q17EDUCATION',
 'Education_Status',
 'Province',
 'age_grp1',
 'Status',
 'status_Exp',
 'Unempl_Status',
 'Q416NRWORKERS',
  temp,
  for_later
  ]
earnings = earnings[q_to_keep]
earnings.info()
earnings['Q58SALARYCATEGORY'].unique()
def create_new_monthly1(val):
    if val <= 3500 and val is not None:
        return 1
    elif val > 3500 and val is not None:
        return 0
    else:
        return val
earnings['new_monthly1'] = earnings['Q57a_monthly'].apply(create_new_monthly1)
earnings['new_monthly1'].unique()
def create_new_monthly2(val):
    if val == 1:
        return None
    elif val >=2 and val <= 7:
        return 1
    elif val > 7 and val is not None:
        return 0
    else:
        return val
earnings['new_monthly2'] = earnings['Q58SALARYCATEGORY'].apply(create_new_monthly2)
earnings['new_monthly2'].unique()
def create_new_monthly(row):
    if row['new_monthly2'] is None:
        return row['new_monthly1']
    else:
        return row['new_monthly2']
earnings['new_monthly'] = earnings.apply(create_new_monthly, axis = 1)
earnings.info()
earnings = earnings[(earnings['status_Exp'] == 1) | ((earnings['status_Exp'] == 2))]
earnings.info()
youth_earnings = earnings[(earnings['age_grp1'] == 4) | (earnings['age_grp1'] == 5)]
youth_earnings.info()
youth_earnings.columns
youth_earnings = youth_earnings[['status_Exp','new_monthly']]
youth_earnings.dropna(inplace = True)
youth_earnings = youth_earnings[youth_earnings['status_Exp'] == 1]
youth_earnings1 = youth_earnings[youth_earnings['new_monthly'] == 1]

youth_earnings2 = youth_earnings[youth_earnings['new_monthly'] == 0]
youth_earnings1.shape
youth_earnings2.shape
# Pie chart, where the slices will be ordered and plotted counter-clockwise:

fdicty = {'fontsize': 15,
 'fontweight' : 'bold',
 'verticalalignment': 'center',
 'horizontalalignment': 'center'}
sns.set()
labels = 'less than 3500', 'more than 3500'
sizes = [3137, 2498]
explode = (0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

colors = ['#ff9999','#ffcc99']

fig1, ax1 = plt.subplots(figsize = (15, 8))
patches, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors = colors)

for text in texts:
    text.set_color('black')
for autotext in autotexts:
    autotext.set_color('black')
texts[0].set_fontsize(15)

texts[1].set_fontsize(15)
plt.tight_layout()
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax1.set_title('Proportion Employed Youth (15-24 years) At Both Income Categories', fontdict = fdicty)

fig1.savefig('pie_chart_youth_earnings.png')

plt.show()
earnings.columns
earn = earnings[['status_Exp', 'new_monthly']]

earn1 = earn[earn['new_monthly'] == 1]

earn2 = earn[earn['new_monthly'] == 0]
earn1.shape
earn2.shape
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'less than 3500', 'more than 3500'
sizes = [33584, 38911]
explode = (0.1, 0)  # only "explode" the 2nd slice 

colors = ['#ff9999','#ffcc99']

fig1, ax1 = plt.subplots(figsize = (14, 6))
patches, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors = colors)


for text in texts:
    text.set_color('black')
    
    text.set_fontsize(15)
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(15)
plt.tight_layout()
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax1.set_title('Proportion Of All Employees At Both Categories', fontdict = fdict)

fig1.savefig('pie_chart_all_earnings.png')

plt.show()
earnings_ed = earnings[(earnings['Education_Status'] == 5) | (earnings['Education_Status'] == 6)]
earnings_ed = earnings_ed[earnings_ed['status_Exp'] == 1]
earnings_ed_matric = earnings_ed[earnings_ed['Education_Status'] == 5]

earnings_ed_tertiary = earnings_ed[earnings_ed['Education_Status'] == 6]
earnings_ed_matric[earnings_ed_matric['new_monthly'] == 1].shape
earnings_ed_matric[earnings_ed_matric['new_monthly'] == 0].shape
earnings_ed_tertiary[earnings_ed_tertiary['new_monthly'] == 1].shape
earnings_ed_tertiary[earnings_ed_tertiary['new_monthly'] == 0].shape
# Pie chart, where the slices will be ordered and plotted counter-clockwise:

fdict = {'fontsize': 15,
 'fontweight' : 'bold',
 'verticalalignment': 'center',
 'horizontalalignment': 'center'}

fdict1 = {'fontsize': 15,
 'fontweight' : 'bold',
 'verticalalignment': 'center',
 'horizontalalignment': 'center'}

colors = ['#66b3ff','#99ff99']

labels = 'less than 3500', 'more than 3500'
sizes1 = [8267, 13985]

sizes2 = [2354, 11420]
explode = (0.050, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots(ncols = 2,figsize = (15, 8))
patches1, texts1, autotexts1 = ax1[0].pie(sizes1, explode=explode, autopct='%1.1f%%',
        shadow=True, startangle=90, colors = colors, labeldistance = 1.1)
ax1[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax1[0].set_title('Proportion Employed Matriculants In Both Income Categories', fontdict = fdict)


patches2, texts2, autotexts2 = ax1[1].pie(sizes2, explode=explode, autopct='%1.1f%%',
        shadow=True, startangle=90, colors = colors)
ax1[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax1[1].set_title('Proportion Employed Post-Matrics In Both Income Categories', fontdict = fdict1)

pos1 = ax1[0].get_position()

pos2 = [pos1.x0 + 0.2, pos1.y0 + 0,  pos1.width-0.3, pos1.height-0.3] 
ax1[0].set_position(pos2) # set a new position

ax1[0].legend(patches1, labels,loc = 2, prop = {'size': '15'})

ax1[1].legend(patches2, labels,loc = 2, prop = {'size': '15'})

for text in texts1:
    text.set_color('black')
    text.set_fontsize(15)
for autotext in autotexts1:
    autotext.set_color('black')
    autotext.set_fontsize(15)

    
for text in texts2:
    text.set_color('black')
    text.set_fontsize(15)
for autotext in autotexts2:
    autotext.set_color('black')
    autotext.set_fontsize(15)
    
plt.tight_layout()

fig1.savefig('education_pie_chart.png')

plt.show()
