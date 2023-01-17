# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')



def determine_rep(row):

    if row['gname'] in republican_groups:

        R = 1

    else:

        R = 0

    return R



def determine_loy(row):

    if row['gname'] in loyalist_groups:

        L = 1

    else:

        L = 0

    return L



def determine_oth(row):

    if row['gname'] in other:

        O = 1

    else:

        O = 0

    return O



#import data

df = pd.read_csv('../input/globalterrorismdb_0616dist.csv',encoding='ISO-8859-1')
df = df[df['provstate']=='Northern Ireland']
keep_cols_list = [1,13,14,18,29,35,37,38,39,58,84]

df = df[keep_cols_list]
df_1970s_one = df[df['iyear'] <= 1974]

df_1970s_two = df[(df['iyear'] > 1974) & (df['iyear'] <= 1979)]
df_1970s_one_attack_count = df_1970s_one.groupby(['iyear','attacktype1_txt']).size()

df_1970s_two_attack_count = df_1970s_two.groupby(['iyear','attacktype1_txt']).size()
df_1970s_one_attack_count.plot.barh()
df_1970s_two_attack_count.plot.barh()
df_1980s_one = df[(df['iyear'] >= 1980) &(df['iyear'] <= 1984)]

df_1980s_two = df[(df['iyear'] > 1984) & (df['iyear'] <= 1989)]



df_1980s_one_attack_count = df_1980s_one.groupby(['iyear','attacktype1_txt']).size()

df_1980s_two_attack_count = df_1980s_two.groupby(['iyear','attacktype1_txt']).size()
df_1980s_one_attack_count.plot.barh()
df_1980s_two_attack_count.plot.barh()
df_1990s_one = df[(df['iyear'] >= 1990) &(df['iyear'] <= 1994)]

df_1990s_two = df[(df['iyear'] > 1994) & (df['iyear'] <= 1999)]



df_1990s_one_attack_count = df_1990s_one.groupby(['iyear','attacktype1_txt']).size()

df_1990s_two_attack_count = df_1990s_two.groupby(['iyear','attacktype1_txt']).size()
df_1990s_one_attack_count.plot.barh()
df_1990s_two_attack_count.plot.barh()
df_post_agreement = df[df['iyear'] > 1998]

df_target = df['targtype1_txt']

df_target_post_agreement = df_post_agreement['targtype1_txt']

print(df_target_post_agreement.value_counts())
print(df_target.value_counts())
df_summary = df[df['iyear'] >= 1998]
pattern = 'Protestant|Catholic'

print(df_summary.summary.str.contains(pattern).value_counts())
#classify groups ino 3 broad catagories: Republican, Loyalist and Other

republican_groups = ['Irish Republican Army (IRA)',

 'Irish Republican Extremists','Official Irish Republican Army (OIRA)',

 'Irish National Liberation Army (INLA)',

 "People's Liberation Army (Northern Ireland)", 'Republican Action Force',

 'Catholic Reaction Force',"Irish People's Liberation Organization (IPLO)", 

 'Direct Action Against Drugs (DADD)',

 'Continuity Irish Republican Army (CIRA)',

 'Real Irish Republican Army (RIRA)',

 'Dissident', 

 'Oglaigh na hEireann', 

 'Dissident Republicans', 

 'Republican Action Against Drugs (RAAD)', 'The New Irish Republican Army',

 'The Irish Volunteers']



loyalist_groups = ['Ulster Volunteer Force (UVF)',

 'Protestant Extremists',

 'Red Hand Commandos',

 'Ulster Freedom Fighters (UFF)', 'Protestant Action Group',

 "Prisoner's Action Force", 

 'Red Commandos', 

 'Loyalist Volunteer Forces (LVF)', 

 'Orange Volunteers (OV)', 'Red Hand Defenders (RHD)', 

 'South Londonderry Volunteers (SLV)', 'Loyalist Action Force',

 'Real Ulster Freedom Fighters (UFF) - Northern Ireland',

 'Loyalists']



other = ['Unknown','Guerrillas','Terrorists','Other',

 'Unaffiliated Individual(s)', 'Paramilitaries','Youths',

 'Paramilitary members']



df['R'] = df.apply(determine_rep, axis=1)

df['L'] = df.apply(determine_loy, axis=1)

df['O'] = df.apply(determine_oth, axis=1)
#plot total number of attacks per year - separating between three broad catagories



keep_cols_list = [0,11,12,13]

df_groups = df[keep_cols_list]

df_group_totals = df_groups.groupby('iyear').sum()



years=['1970','1971','1972','1973','1974','1975','1976','1977','1978','1979',

       '1980','1981','1982','1983','1984','1985','1986','1987','1988','1989',

       '1990','1991','1992','1994','1995','1996','1997','1998','1999',

       '2000','2001','2002','2003','2004','2005','2006','2007','2008','2009',

       '2010','2011','2012','2013','2014','2015']

df_group_totals['iyear'] = years



f,ax1 =plt.subplots()

bar_width=0.75

bar_l = [i+1 for i in range(len(df_group_totals['iyear']))]

tick_pos=[i+(bar_width/2) for i in bar_l]

ax1.bar(bar_l,df_group_totals['R'],width=bar_width,label='Republican Attacks',alpha=0.5,color='#43AD0A')

ax1.bar(bar_l,df_group_totals['L'],width=bar_width,bottom=df_group_totals['R'],label='Loyalist Attacks',alpha=0.5,color='#F08A2A')

ax1.bar(bar_l,df_group_totals['O'],width=bar_width,bottom=[i+j for i,j in zip(df_group_totals['R'],df_group_totals['L'])],label='Other/unknown',alpha=0.5,color='#2A6FF0')

plt.xticks(tick_pos, df_group_totals['iyear'],rotation='vertical')

ax1.set_ylabel("Number of Attacks")

ax1.set_xlabel("Year")

plt.legend(loc='upper right')

plt.xlim([min(tick_pos)-bar_width, max(tick_pos)+bar_width])