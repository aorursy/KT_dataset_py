import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
def pad0(string):
    string = str(string)
    if len(string) == 3:
        return '0'+string
    else:
        return string
df = pd.read_csv("/kaggle/input/cpd-police-beat-demographics/master.csv")
df['beat'] = df['beat'].apply(lambda x: pad0(x))
rclist = ['percent_white','percent_hispanic','percent_black']
for rc in rclist:
    ax = df[rc].plot.hist(bins=20,title=rc[8:])
    ax.set_xlabel('percent')
    plt.show()
cmap = {'percent_white':'blue','percent_hispanic':'orange','percent_black':'green'}

def prd(dtf,order,sort,cmap):
    st = dtf.sort_values(by=[sort])
    ax = st[order[0]].plot.bar(stacked=True,color=cmap[order[0]],figsize=(20,5),title='sorted by '+sort)
    sm = st[order[0]]
    for a in order[1:]:
        ax = st[a].plot.bar(color=cmap[a],bottom=sm)
        sm += st[a]
    ax.set_xlabel('beat')
    ax.set_ylabel('percent')
    plt.xticks([])
    plt.show()
    
prd(df,['percent_black','percent_white','percent_hispanic'], 'percent_black', cmap)
prd(df,['percent_white','percent_black','percent_hispanic'], 'percent_white',cmap)
prd(df,['percent_hispanic','percent_black','percent_white'], 'percent_hispanic',cmap)
to_graph = ['med_income','percent_on_fs','percent_se_18-19']
for tg in to_graph:
    prd(df,['percent_black','percent_hispanic','percent_white'],tg,cmap)
    prd(df,[tg],tg,{tg:'red'})
df_crime = pd.read_csv('/kaggle/input/clean-chicago-crime-data/crime_new.csv', dtype={'Primary Type': object, 'Description': object,'Location Description': object, 'Arrest': bool, 'Domestic': bool,'Beat': object,'Year': object,'Violent':bool, 'Petty': bool,'Property': bool})
print(df_crime['Primary Type'].unique())
violent = {'BATTERY','ASSAULT','KIDNAPPING', 'ROBBERY',
       'CRIM SEXUAL ASSAULT', 'HOMICIDE','CRIMINAL SEXUAL ASSAULT'}
petty = {'WEAPONS VIOLATION','PROSTITUTION','GAMBLING','LIQUOR LAW VIOLATION','OBSCENITY','RITUALISM',
         'CONCEALED CARRY LICENSE VIOLATION','PUBLIC INDECENCY','PUBLIC PEACE VIOLATION'}
prop = {'CRIMINAL DAMAGE', 'ARSON', 'THEFT','ROBBERY','MOTOR VEHICLE THEFT','BURGLARY'}
index = {'BATTERY','ASSAULT','KIDNAPPING', 'ROBBERY',
       'CRIM SEXUAL ASSAULT', 'ARSON','HOMICIDE','CRIMINAL SEXUAL ASSAULT',
        'MOTOR VEHICLE THEFT','THEFT','BURGLARY'}
# interesting: 'INTERFERENCE WITH PUBLIC OFFICER'
df_crime.drop(df_crime[df_crime['Year'] == '2020'].index,inplace=True)
print("     Total:",len(df_crime))
print("%  Violent:",df_crime['Violent'].sum()/len(df_crime)*100)
print("%    Petty:",df_crime['Petty'].sum()/len(df_crime)*100)
print("% Property:",df_crime['Property'].sum()/len(df_crime)*100)
def f(x):
    return x['Violent'] and x['Arrest']
def f1(x):
    return x['Petty'] and x['Arrest']
def f2(x):
    return x['Property'] and x['Arrest']
df_crime['V_A'] = df_crime[['Violent','Arrest']].apply(f,axis=1)
df_crime['Pt_A'] = df_crime[['Petty','Arrest']].apply(f1,axis=1)
df_crime['Pp_A'] = df_crime[['Property','Arrest']].apply(f2,axis=1)
beat_dfs = {a:0 for a in df_crime['Beat'].unique()}
g = df_crime.groupby('Beat')
for b in df_crime['Beat'].unique():
    if (b not in df['beat'].unique()):
        continue
    print("at:",b)
    beat_dfs[b] = g.get_group(b)
print("     Total:",len(df_crime))
print("%  Arrest per violent:",df_crime['V_A'].sum()/df_crime['Violent'].sum()*100)
print("%    Arrest per petty:",df_crime['Pt_A'].sum()/df_crime['Petty'].sum()*100)
print("% Arrest per property:",df_crime['Pp_A'].sum()/df_crime['Property'].sum()*100)
for b in df['beat'].unique():
    if b not in beat_dfs:
        print (b)

# drop part of O'Hare
df.drop(df[df['beat'] == '3100'].index,inplace=True)
df['Crimes_2001-2019'] = df['beat'].apply(lambda x: len(beat_dfs[x]))
df.head()
df_crime['Year'].value_counts()
