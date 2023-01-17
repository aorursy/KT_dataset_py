import numpy as np

import pandas as pd

from pandas import Series,DataFrame

import matplotlib.pyplot as plt

import seaborn.apionly as sns
df_basic = pd.read_csv('../input/database.csv',index_col= 'Record ID')

df_stranger = df_basic[df_basic['Relationship'] == 'Stranger']

df_others = df_basic[df_basic['Relationship'] != 'Stranger']

df_others_without_unknown = df_basic[(df_basic['Relationship'] != 'Stranger') & (df_basic['Relationship'] != 'Unknown')]
case_stranger = df_stranger['Year'].value_counts().sort_index()

case_others = df_others['Year'].value_counts().sort_index()

fig, ax = plt.subplots()

x = case_stranger.index[5:]

y1 = case_stranger.values[5:]

y2 = case_others.values[5:]+case_stranger.values[5:]

plot = ax.stackplot(x,y1,y2,colors = ['darkseagreen','green'],labels = ['Stranger murder','All cases'])

ax.set_xticks(x[1::4])

ax.set_title('Case number from 1980 to 2014',fontsize = 14)

ax.set_ylabel('Case Number',fontsize=12)

ax.legend(loc='upper right', shadow=True, fontsize=12)

ax.grid()

def autolabel_plot(df):

    for i in range(6,len(df.index),4):

        plot_x = df.index[i]

        plot_y = df.values[i]

        ax.text(plot_x, plot_y+500, '%d'%plot_y,fontsize=10)

autolabel_plot(case_stranger)

autolabel_plot((case_stranger+case_others))

fig.set_size_inches(12.5, 6.5)

plt.show()
width = 2.5

case_stranger_percent = 100*case_stranger/(case_stranger+case_others)

case_others_percent = 100*case_others/(case_stranger+case_others)

ind = case_others_percent.index[1::3]

rects1_value = case_others_percent[1::3]

rects2_value = case_stranger_percent[1::3]



fig, ax = plt.subplots()



rects1  = ax.bar(ind,rects1_value,width = width,color = 'green',label = 'Others')

rects2 = ax.bar(ind,rects2_value,width = width, color = 'darkseagreen',bottom=case_others_percent[1::3],label = 'Stranger')

ax.legend (loc='lower right', shadow=True, fontsize='10')

ax.set_xticks(ind)

ax.set_yticks(np.arange(0,110,10))

ax.set_ylabel('Percent of Stranger Murder %',fontsize = 12)

ax.set_title('Case number from 1980 to 2014',fontsize = 14)



def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        bottom = rect.get_y()

        ax.text(rect.get_x()+rect.get_width()/2., 0.5*height+bottom, '%.1f'%(height),

                ha='center', va='bottom',color = 'white',fontsize =10,fontweight='bold')



autolabel(rects1)

autolabel(rects2)

fig.set_size_inches(12, 6)

plt.show()
case_stranger_solved= df_stranger['Crime Solved'].value_counts()

case_others_solved= df_others_without_unknown['Crime Solved'].value_counts()

stranger_solved_probability = Series([case_stranger_solved['Yes']/float(case_stranger_solved.sum()),case_stranger_solved['No']/float(case_stranger_solved.sum())],index=['Yes','No'])

others_solved_probability = Series([case_others_solved['Yes']/float(case_others_solved.sum()),case_others_solved['No']/float(case_others_solved.sum())],index=['Yes','No'])



fig,(ax0,ax1) = plt.subplots(ncols=2)

labels = 'Yes', 'No'

sizes0 = [100*stranger_solved_probability['Yes'],100*stranger_solved_probability['No']]

ax0.pie(sizes0, labels=labels, autopct='%1.1f%%', startangle=90,colors = ['green','darkseagreen'])

ax0.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax0.set_title('Solving Ratio of Stranger Murder',fontsize = 14,y=1.1,color = 'black')



sizes1 = [100*others_solved_probability['Yes'],100*others_solved_probability['No']]

ax1.pie(sizes1, labels=labels, autopct='%1.1f%%', startangle=90,colors = ['green','darkseagreen'])

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax1.set_title('Solving Ratio of Other Murder',fontsize = 14,y=1.1,color = 'black')

plt.rcParams['font.size'] = 12

fig.set_size_inches(12.5, 6)

plt.show()
df_stranger_age = df_stranger.ix[:,['Victim Age','Perpetrator Age']].astype(float)

df_stranger_age = df_stranger_age[(df_stranger_age['Victim Age']<99) & (df_stranger_age['Victim Age']>10) & (df_stranger_age['Perpetrator Age']>10)&(df_stranger_age['Perpetrator Age']<99)]

x1 = df_stranger_age['Victim Age']

y1 = df_stranger_age['Perpetrator Age']

#fit the data

z1 = np.polyfit(x1.values, y1.values, 1)

x2 = np.arange(1.0,100.0,0.5)

y2 = x2*z1[0]+z1[1]

#plot

fig, ax = plt.subplots()

ax.set_ylim([10,100])

ax.set_xlim([10,100])

ax.set_title('Victim Age VS Perpetrator Age in Stranger Murder',fontsize=14)

ax.set_ylabel('Perpetrator Age',fontsize = '12')

ax.set_xlabel('Victim Age',fontsize = '12')

plot1 = ax.plot(x1,y1,'g.')

plot2 = ax.plot(x2,y2,'dimgrey',linewidth=3)

ax.text(88,34,'y=%.1fx+%.1f'%(z1[0],z1[1]),fontsize=12,fontweight='bold',color ='dimgrey')

fig.set_size_inches(12, 6)

plt.show()
case_state = df_stranger[['Year','State']]

case_state_count = case_state.groupby([case_state['State'],case_state['Year']]).size().sort_index().unstack(fill_value=0)

sns.set(rc={"figure.figsize": (12, 12)})

fig, ax = plt.subplots()

ax = sns.heatmap(case_state_count,annot=False, fmt="d", linewidths=.1,cmap="Greens",annot_kws={"size": 20})

sns.plt.title('Stranger Murder Counts along State and Year',fontsize = 14)

plt.show()

sns.reset_orig()
case_stranger_v_sex = df_stranger['Victim Sex'].value_counts()

case_others_v_sex = df_others_without_unknown['Victim Sex'].value_counts()

stranger_v_sex_victim_ratio = Series([case_stranger_v_sex['Female']/float(case_stranger_v_sex.sum()),case_stranger_v_sex['Male']/float(case_stranger_v_sex.sum()),case_stranger_v_sex['Unknown']/float(case_stranger_v_sex.sum())],index=['Female','Male','Unknown'])

others_v_sex_victim_ratio = Series([case_others_v_sex['Female']/float(case_others_v_sex.sum()),case_others_v_sex['Male']/float(case_others_v_sex.sum()),case_others_v_sex['Unknown']/float(case_others_v_sex.sum())],index=['Female','Male','Unknown'])



fig,(ax0,ax1) = plt.subplots(ncols=2)

labels = 'Male', 'Female', 'Unknown'

sizes0 = [100*stranger_v_sex_victim_ratio['Male'],100*stranger_v_sex_victim_ratio['Female'],100*stranger_v_sex_victim_ratio['Unknown']]

ax0.pie(sizes0, labels=labels, autopct='%1.1f%%', startangle=90,colors = ['green','darkseagreen','seagreen'])

ax0.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax0.set_title('Victim Sex in Stranger Murder',y=1.1,fontsize = '14')



sizes1 = [100*others_v_sex_victim_ratio['Male'],100*others_v_sex_victim_ratio['Female'],100*others_v_sex_victim_ratio['Unknown']]

ax1.pie(sizes1, labels=labels, autopct='%1.1f%%', startangle=90,colors = ['green','darkseagreen','seagreen'])

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax1.set_title('Victim Sex in Other Murder',y=1.1,fontsize = '14')

#fig.tight_layout()

fig.set_size_inches(12.5, 6)

plt.show()   
case_stranger_p_sex = df_stranger['Perpetrator Sex'].value_counts()

case_others_p_sex = df_others_without_unknown['Perpetrator Sex'].value_counts()

stranger_p_sex_victim_ratio = Series([case_stranger_p_sex['Female']/float(case_stranger_p_sex.sum()),case_stranger_p_sex['Male']/float(case_stranger_p_sex.sum()),case_stranger_p_sex['Unknown']/float(case_stranger_p_sex.sum())],index=['Female','Male','Unknown'])

others_p_sex_victim_ratio = Series([case_others_p_sex['Female']/float(case_others_p_sex.sum()),case_others_p_sex['Male']/float(case_others_p_sex.sum()),case_others_p_sex['Unknown']/float(case_others_p_sex.sum())],index=['Female','Male','Unknown'])



fig,(ax0,ax1) = plt.subplots(ncols=2)

labels = 'Male', 'Female', 'Unknown'

sizes0 = [100*stranger_p_sex_victim_ratio['Male'],100*stranger_p_sex_victim_ratio['Female'],100*stranger_p_sex_victim_ratio['Unknown']]

ax0.pie(sizes0, labels=labels, autopct='%1.1f%%', startangle=90,colors = ['green','darkseagreen','lightgrey'])

ax0.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax0.set_title('Perpetrator Sex in Stranger Murder',y=1.1,fontsize = '14')



sizes1 = [100*others_p_sex_victim_ratio['Male'],100*others_p_sex_victim_ratio['Female'],100*others_p_sex_victim_ratio['Unknown']]

ax1.pie(sizes1, labels=labels, autopct='%1.1f%%', startangle=90,colors = ['green','darkseagreen','lightgrey'])

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax1.set_title('Perpetrator Sex in Other Murder',y=1.1,fontsize = '14')

#fig.tight_layout()

fig.set_size_inches(12.5, 6)

plt.show()
case_stranger_weapon = df_stranger['Weapon'].value_counts()

stranger_weapon_ratio = 100*(case_stranger_weapon/float(case_stranger_weapon.sum()))

case_others_weapon = df_others_without_unknown['Weapon'].value_counts()

others_weapon_ratio = 100*(case_others_weapon/float(case_others_weapon.sum()))

weapon_ratio = pd.concat([stranger_weapon_ratio,others_weapon_ratio], axis=1,join='outer')

weapon_ratio.columns = ['Stranger Murder %', 'Other Murder %']

sns.set(rc={"figure.figsize": (8, 8)})

fig, ax = plt.subplots()

ax = sns.heatmap(weapon_ratio,annot=True,linewidths=.1,cmap="Greens",annot_kws={"size": 10})

sns.plt.title('Weapons of Stranger VS Others Murder',fontsize = 12)

sns.reset_orig()

plt.show()