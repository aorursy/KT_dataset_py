#Basic Imports

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from matplotlib import rcParams

from math import pi 

import warnings

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.datasets import make_classification

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

import statsmodels.api as sm

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

sns.set_style('whitegrid')

%matplotlib inline



from IPython.display import HTML

from IPython.core.display import display#, HTML
#dataset imports

df_all = pd.read_csv('..//input/nfl-final-datasets/df_all.csv')

inj = pd.read_csv('..//input/nfl-final-datasets/inj.csv')

season = pd.read_csv('..//input/nfl-final-datasets/season.csv')

playnew_all = pd.read_csv('..//input/nfl-final-datasets/playnew_all_excl.csv')
playmodel = pd.read_csv('..//input/nfl-final-datasets/play_model.csv')

timedistplt=playmodel[playmodel['pr_fatigue_sec_played']<80]

timedistplt=timedistplt[timedistplt['pr_fatigue_dist_run']<150]

timedistplt=timedistplt[timedistplt['Injury']==1]

timedistplt=timedistplt[['pr_fatigue_dist_run','pr_fatigue_sec_played']].dropna()

# timedistplt_inj = timedistplt.loc[timedistplt['Injury']==1]



timedistplt_health=playmodel.loc[(playmodel['pr_fatigue_sec_played']<80) & (playmodel['pr_fatigue_dist_run']>150) & (playmodel['Injury']==0)]

timedistplt_health=timedistplt_health[['pr_fatigue_dist_run','pr_fatigue_sec_played']].dropna()

#Injured VS Healthy db creation

sns.set_style('white')

j1 = sns.jointplot(x='pr_fatigue_sec_played',y='pr_fatigue_dist_run',data=timedistplt,kind='kde', color="red", joint_kws={'shade_lowest':False})

j1.set_axis_labels('$\it{Accumulated\,\,seconds\,\,played}$', '$\it{Accumulated\,\,distance\,\,run\,\,(Yards)}$', fontsize=16)

j1.ax_marg_x.set_xlim(0, 100)

j1.ax_marg_y.set_ylim(0, 200)
#Injured VS Healthy db creation

j2 = sns.jointplot(x='pr_fatigue_sec_played',y='pr_fatigue_dist_run',data=timedistplt_health,kind='kde', color="darkblue", joint_kws={'shade_lowest':False})



j2.set_axis_labels('$\it{Accumulated\,\,seconds\,\,played}$', '$\it{Accumulated\,\,distance\,\,run\,\,(Yards)}$', fontsize=16)

j2.ax_marg_x.set_xlim(0, 100)

j2.ax_marg_y.set_ylim(0, 200)
inj_last=pd.merge(season,inj,how='inner',on=['PlayerKey','GameID','PlayKey'])

inj_cont = pd.merge(season,inj[inj['PlayerKey'].isin(inj_last['PlayerKey'])==False].dropna(),

                    how='inner',on=['PlayerKey','GameID'])

df_inj=df_all[df_all['Injury']==1]

df_health=df_all[df_all['Injury']==0]

inj_miss = pd.merge(season,inj[inj['PlayKey'].isnull()],how='inner',on=['PlayerKey','GameID'])

playnew_all['Injury']= np.where(playnew_all['BodyPart'].isnull(), 0, 1)

playnew_inj=playnew_all[playnew_all['Injury']==1]

playnew_health=playnew_all[playnew_all['Injury']==0]

misskeylist = [val for sublist in inj_miss.groupby('GameID')['GameID'].unique() for val in sublist]

playnew_health=(playnew_health[~playnew_health.GameID.isin(misskeylist)])
# This code creates a dataframe which will be used for comparing various segments

def cr_stats(y,z):

    data= {'DB Flag':z,'AccMean': round(y['AccMean'].mean(),4),

                         'AccStd':round(y['AccStd'].mean(),4),

            'AccMax': round(y['AccMax'].mean(),4),'DecMax': round(y['DecMax'].mean(),4),

            'DecAvg': round(y['DecAvg'].mean(),4),'AccAvg': round(y['AccAvg'].mean(),4),

            'AvgS': round(y['AvgS'].mean(),4), 'MaxS': round(y['MaxS'].mean(),4),

            'MaxTorque': round(y['MaxTorque'].mean(),4),'TorqueAvg': round(y['TorqueAvg'].mean(),4),

            'TorquePosAvg': round(y['TorquePosAvg'].mean(),4)}

    df = pd.DataFrame(data,index=[0])

    return df

inj_stats = cr_stats(playnew_inj,'Injured')

health_stats=cr_stats(playnew_health,'Healthy')

synth_stats=cr_stats(playnew_all[playnew_all['FieldType'] =='Synthetic'],'Synthetic')

nat_stats=cr_stats(playnew_all[playnew_all['FieldType'] =='Natural'],'Natural')

injkn_stats=cr_stats(playnew_inj[playnew_inj['BodyPart'] =='Knee'],'Knee inj')

injft_stats=cr_stats(playnew_inj[playnew_inj['BodyPart'] =='Foot'],'Foot inj')

injan_stats=cr_stats(playnew_inj[playnew_inj['BodyPart'] =='Ankle'],'Ankle inj')

noprec_stats=cr_stats(playnew_all[(playnew_all['Weather_4'] == 'Clear') |(playnew_all['Weather_4'] == 'Cloudy')|

                       (playnew_all['Weather_4'] == 'Hazy')|(playnew_all['Weather_4'] == 'Partly Cloudy')

                      |(playnew_all['Weather_4'] == 'Mostly Cloudy')],'No Precipitation')

prec_stats=cr_stats(playnew_all[(playnew_all['Weather_4'] == 'Rain') |(playnew_all['Weather_4'] == 'Showers')|

                       (playnew_all['Weather_4'] == 'Snow')|(playnew_all['Weather_4'] == 'Light Rain')

                      |(playnew_all['Weather_4'] == 'Heavy Snow')|(playnew_all['Weather_4'] == 'Light Snow')],'Precipitation')

pass_stats=cr_stats(playnew_all[(playnew_all['PlayType'] == 'Pass')],'Pass')

rush_stats=cr_stats(playnew_all[(playnew_all['PlayType'] == 'Rush')],'Rush')

kick_stats=cr_stats(playnew_all[(playnew_all['PlayType'] == 'Kickoff')],'Kickoff')

field_stats=cr_stats(playnew_all[(playnew_all['PlayType'] == 'Field Goal')],'Field Goal')

frames = [inj_stats,health_stats,synth_stats,nat_stats,injkn_stats,injft_stats,injan_stats,noprec_stats, prec_stats,

         pass_stats, rush_stats, kick_stats, field_stats]

result = pd.concat(frames)
#Injured VS Healthy db creation

sns.set(rc={'figure.figsize':(15,5)})

rcParams['figure.figsize'] = 15,5

sns.set_style('white')

fig, ax =plt.subplots(1,3)



sns.distplot(playnew_inj['AccAvg'],hist=False,color='red',ax=ax[0], label='Injuries').set_title("Distribution of\nAverage Acceleration")

sns.distplot(playnew_health['AccAvg'],hist=False,color='darkblue',ax=ax[0], label='No injuries').set_ylabel('Distribution')

sns.distplot(playnew_inj['DecAvg'],hist=False,color='red',ax=ax[1], label='Injuries').set_title("Distribution of\nAverage Deceleration")

sns.distplot(playnew_health['DecAvg'],hist=False,color='darkblue',ax=ax[1], label='No injuries')

sns.distplot(playnew_inj['MaxTorque'],hist=False,color='red',ax=ax[2], label='Injuries').set_title("Distribution of\nMax Torque")

sns.distplot(playnew_health['MaxTorque'],hist=False,color='darkblue',ax=ax[2], label='No injuries')



ax[0].set(xlabel='$\it{Acceleration\,\,(Yards\,\,per\,\,second-decimal^2)}$', ylabel='$\it{Distribution}$')

ax[1].set(xlabel='$\it{Deceleration\,\,(Yards\,\,per\,\,second-decimal^2)}$')

ax[2].set(xlabel='$\it{Torque\,\,(N-mt)}$')
result1=result[(result['DB Flag']=='Knee inj') | (result['DB Flag']=='Foot inj') | (result['DB Flag']=='Ankle inj')]

result1['AccMaxW']=result1['AccMax']/result1['AccMax'].max()

result1['DecMaxW']=result1['DecMax']/result1['DecMax'].min()

result1['AccAvgW']=result1['AccAvg']/result1['AccAvg'].max()

result1['MaxSW']=result1['MaxS']/result1['MaxS'].max()

result1['MaxTorqueW']=result1['MaxTorque']/result1['MaxTorque'].max()

result1.drop(['AccMean','AccStd','AccMax','DecMax','DecAvg','AccAvg','AvgS','MaxS','MaxTorque','TorqueAvg','TorquePosAvg'],axis=1,inplace=True)
# ------- PART 1: Create background

# number of variable

categories=result1.columns.drop('DB Flag').tolist()

sns.set_style('whitegrid')

N = len(categories)

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)

angles = [n / float(N) * 2 * pi for n in range(N)]

angles += angles[:1]

 # Initialise the spider plot

ax = plt.subplot(111, polar=True)

ax.set_theta_offset(pi / 2)

ax.set_theta_direction(-1)

 

# Draw one axe per variable + add labels labels yet

categories2 = ['Max\nAcceleration', 'Max\nDeceleration', 'Average\nAcceleration', 'Max\nSpeed', 'Max\nTorque']

plt.xticks(angles[:-1], categories2)

 

# Draw ylabels

ax.set_rlabel_position(0)

plt.yticks([0,0.5,1], ["0","0.5","1"], color="grey", size=7)

plt.ylim(0,1)

# ------- PART 2: Add plots

# Ind1

values=result1.iloc[0].drop('DB Flag').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', color ='orange',label="Knee Injuries")

#ax.fill(angles, values, 'y', alpha=0.1)

 

# Ind2

values=result1.iloc[1].drop('DB Flag').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', color ='blue', label="Foot Injuries")

#ax.fill(angles, values, 'blue',alpha=0.1)



# Ind3

values=result1.iloc[2].drop('DB Flag').values.flatten().tolist()

values += values[:1]

ax.plot(angles, values, linewidth=1, linestyle='solid', color ='r', label="Ankle Injuries")

#ax.fill(angles, values, 'g',alpha=0.1)



plt.legend(loc='upper right', bbox_to_anchor=(1.5, 1.1))

plt.show()


a1=pd.DataFrame({'Synthetic Field': round(inj[(inj['Surface']=='Synthetic')]

                                          .groupby(('BodyPart')).size() / len(inj[(inj['Surface']=='Synthetic')]),2),

                'Natural Field': round(inj[(inj['Surface']=='Natural')]

                                          .groupby(('BodyPart')).size() / len(inj[(inj['Surface']=='Natural')]),2)}).fillna(0)



a1.style.format({

    'Synthetic Field': '{:,.2f%}'.format,

    'Natural Field': '{:,.2f%}'.format

})



#a1



plt.rcParams["axes.labelsize"] = 14

plt.rcParams["xtick.labelsize"] = 14

plt.rcParams["axes.titlesize"] = 18



fig = plt.figure()

sns.set_style('white')

ax1 = a1.plot(kind="bar", color = ['darkgray', 'darkgreen'])

#plt.xticks(bar_x_positions, bar_labels)

ax1.set(ylabel='$\it{Distribution\,\,of\,\,injuries}$'+"\n"+'$\it{across\,\,field\,\,types}$', title="Percentage occurrence\nacross field types")

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.set(rc={'figure.figsize':(15,5)})

sns.set_style('white')

fig, ax =plt.subplots(1,3)



plt.rcParams["axes.labelsize"] = 14

plt.rcParams["xtick.labelsize"] = 14

plt.rcParams["axes.titlesize"] = 18



sns.distplot(playnew_all[playnew_all['FieldType'] =='Natural']['AvgS'],hist=False,color='darkgreen',ax=ax[0], label='Natural field').set_title("Distribution of\nAverage Speed")

sns.distplot(playnew_all[playnew_all['FieldType'] =='Synthetic']['AvgS'],hist=False,color='darkgray',ax=ax[0], label='Synthetic field').set_ylabel('Distribution')

sns.distplot(playnew_all[playnew_all['FieldType'] =='Natural']['DecAvg'],hist=False,color='darkgreen',ax=ax[1], label='Natural field').set_title("Distribution of\nAverage Deceleration")

sns.distplot(playnew_all[playnew_all['FieldType'] =='Synthetic']['DecAvg'],hist=False,color='darkgray',ax=ax[1], label='Synthetic field')

sns.distplot(playnew_all[playnew_all['FieldType'] =='Natural']['MaxTorque'],hist=False,color='darkgreen',ax=ax[2], label='Natural field').set_title("Distribution of\nMax Torque")

sns.distplot(playnew_all[playnew_all['FieldType'] =='Synthetic']['MaxTorque'],hist=False,color='darkgray',ax=ax[2], label='Synthetic field')



ax[0].set(xlabel='$\it{Speed\,\,(Yards\,\,per\,\,second-decimal)}$', ylabel='$\it{Distribution}$')

ax[1].set(xlabel='$\it{Deceleration\,\,(Yards\,\,per\,\,second-decimal^2)}$')

ax[2].set(xlabel='$\it{Torque\,\,(N-mt)}$')
result2 = result.copy()

#pd.reset_option('^display.float_format')



result2.rename(columns = {'DB Flag':'Indicators'}, inplace = True)

HTML(result2.to_html(index=False))#Removed indices
#result[(result['DB Flag']=='Synthetic') | (result['DB Flag']=='Natural')]

field_t_summary = result[(result['DB Flag']=='Synthetic') | (result['DB Flag']=='Natural')]

field_t_summary.rename(columns = {'DB Flag':'Field Type'}, inplace = True)

HTML(field_t_summary.to_html(index=False))#Removed indices
plt.rcParams["axes.labelsize"] = 14

plt.rcParams["xtick.labelsize"] = 14

plt.rcParams["axes.titlesize"] = 18



a1=pd.DataFrame({'Percentage': round(df_all.groupby(('FieldType')).size() / len(df_all),2), 

                 'Count':df_all.groupby(('FieldType')).size()})

fig, ax =plt.subplots(1,2)

sns.set_style("white")

#fig = plt.figure(figsize=(3,4))

ax1=sns.countplot(x='FieldType',data=df_all,ax=ax[0], palette=['darkgray', 'darkgreen'])

figure_title = "FieldType - Total Sample"

# plt.text(0.5, 1, figure_title,

#          horizontalalignment='center',

#          fontsize=20,

#          transform = ax1.transAxes)

a2=pd.DataFrame({'Percentage': round(df_inj.groupby(('FieldType')).size() / len(df_inj),2), 

                 'Count':df_inj.groupby(('FieldType')).size()})

ax2=sns.countplot(x='FieldType',data=df_inj,order=['Synthetic','Natural'], palette=['darkgray', 'darkgreen'])#.grid(False)



figure_title = "FieldType - Injury Sample"

# plt.text(0.6, 1, figure_title,

#          horizontalalignment='center',

#          fontsize=20,

#          transform = ax2.transAxes)



ax1.set(xlabel='$\it{Field\,\,Type}$', ylabel='$\it{Count}$', title='Total Sample')

ax2.set(xlabel='$\it{Field\,\,Type}$', ylabel='$\it{Count}$', title='Injury Sample')
temp=playnew_all[playnew_all['Temperature']!=-999]



plt.rcParams["axes.labelsize"] = 14

plt.rcParams["xtick.labelsize"] = 14

plt.rcParams["axes.titlesize"] = 18



sns.set_style('white')

#sns.violinplot(x='Injury', y='Temperature', data=temp,palette='rainbow')

ax=sns.boxplot(x='FieldType', y='Temperature', data=temp, palette=['darkblue', 'red'],hue='Injury')



ax.set(xlabel='$\it{Field\,\,Type}$', ylabel='$\it{Temperature\,\,(F)}$', title='Field Types Distributions by\nTemperature and Injury Occurrence')

plt.show()
a1=pd.DataFrame({'Count injured':df_inj.groupby(('Weather_4')).size(),

                 'Count not injured':df_health.groupby(('Weather_4')).size(),

                 'Percentage injured': round(df_inj.groupby(('Weather_4')).size() / len(df_inj),4),

                 'Percentage not injured': round(df_health.groupby(('Weather_4')).size() / len(df_health),4)}).fillna(0)



a1 = a1.astype({"Count injured": int})# Convert count injured to integer, thus removing decimals

a1 = a1.to_html(formatters={

    'Count injured': '{:,}'.format,

    'Count not injured': '{:,}'.format,

    'Percentage injured': '{:,.2%}'.format,

    'Percentage not injured': '{:,.2%}'.format

})

HTML(a1)#Removed indices
synth=df_all[df_all['Weather_4']=='N/A (Indoors)']

weather_summary_1 = pd.DataFrame({'count' : synth.groupby(['Weather_4','FieldType']).size()}).reset_index()#Prettified groupby

weather_summary_1.rename(columns = {'Weather_4':'Weather'}, inplace = True)#Renamed Weather_4 column

HTML(weather_summary_1.to_html(index=False))#Removed indices
nosyinj=df_inj[df_inj['Weather_4']!='N/A (Indoors)']

nosyhea=df_health[df_health['Weather_4']!='N/A (Indoors)']

a1=pd.DataFrame({'Count injured':nosyinj.groupby(('Weather_4')).size(),

                 'Count not injured':nosyhea.groupby(('Weather_4')).size(),

                 'Percentage injured': round(nosyinj.groupby(('Weather_4')).size() / len(nosyinj),4),

                 'Percentage not injured': round(nosyhea.groupby(('Weather_4')).size() / len(nosyhea),4)}).fillna(0)

a2 = a1.astype({"Count injured": int})# Convert count injured to integer, thus removing decimals

a2 = a2.to_html(formatters={

    'Count injured': '{:,}'.format,

    'Count not injured': '{:,}'.format,

    'Percentage injured': '{:,.2%}'.format,

    'Percentage not injured': '{:,.2%}'.format

})

HTML(a2)#Removed indices
injcount = a1['Count injured']

healthcount = a1['Count not injured']

chisq = np.array([injcount,healthcount])

chi2_stat, p_val, dof, ex = stats.chi2_contingency(chisq)



weather_v_inj_ChiSq_stats = pd.DataFrame({#'Analysis':'Weather v. Injury occurrence',

                 'Chi2 Stat Result':chi2_stat,

                 'Degrees of Freedom': dof,

                 'P-Value': p_val}, index=['Weather v. Injury occurrence'])



HTML(weather_v_inj_ChiSq_stats.to_html(index=True))#Prettified
outinj=df_inj[(df_inj['Weather_4']!='N/A (Indoors)')&(df_inj['FieldType']=='Synthetic')]

outhealth=df_health[(df_health['Weather_4']!='N/A (Indoors)')&(df_health['FieldType']=='Synthetic')]

a1=pd.DataFrame({'Count injured':outinj.groupby(('Weather_4')).size(),

                 'Count not injured':outhealth.groupby(('Weather_4')).size(),

                 'Percentage injured': round(outinj.groupby(('Weather_4')).size() / len(outinj),4),

                 'Percentage not injured': round(outhealth.groupby(('Weather_4')).size() / len(outhealth),4)}).fillna(0)



a2 = a1.astype({"Count injured": int})# Convert count injured to integer, thus removing decimals

a2 = a2.to_html(formatters={

    'Count injured': '{:,}'.format,

    'Count not injured': '{:,}'.format,

    'Percentage injured': '{:,.2%}'.format,

    'Percentage not injured': '{:,.2%}'.format

})

HTML(a2)#Removed indices
injcount = a1['Count injured']

healthcount = a1['Count not injured']

chisq = np.array([injcount,healthcount])

chi2_stat, p_val, dof, ex = stats.chi2_contingency(chisq)



weather_v_inj_ChiSq_stats2 = pd.DataFrame({#'Analysis':'Weather v. Injury occurrence',

                 'Chi2 Stat Result':chi2_stat,

                 'Degrees of Freedom': dof,

                 'P-Value': p_val}, index=['Weather v. Injury occurrence'])



HTML(weather_v_inj_ChiSq_stats2.to_html(index=True))#Prettified
precipitation_summary = result[(result['DB Flag']=='No Precipitation') | (result['DB Flag']=='Precipitation')]

precipitation_summary.rename(columns = {'DB Flag':'Precipitation occurrence'}, inplace = True)

HTML(precipitation_summary.to_html(index=False))#Removed indices, prettified
plt.rcParams["axes.labelsize"] = 14

plt.rcParams["xtick.labelsize"] = 12

plt.rcParams["axes.titlesize"] = 18

plt.rcParams["xtick.alignment"] = 'center'

sns.set_style('white')

ax=sns.countplot(x='PlayType',data= playnew_inj, palette=sns.color_palette("husl", 8))



ax.set(xlabel='$\it{Play\,\,Type}$', ylabel='$\it{Count}$', title='Play Types Counts for\nInjured Players')

# plt.xticks(rotation=45)

plt.show()
pt_summary = result[(result['DB Flag']=='Pass') | (result['DB Flag']=='Rush')|(result['DB Flag']=='Kickoff')|

       (result['DB Flag']=='Field Goal')]

pt_summary.rename(columns = {'DB Flag':'Play Type'}, inplace = True)

HTML(pt_summary.to_html(index=False))#Removed indices, prettified
data = pd.read_csv('..//input/nfl-final-datasets/play_model.csv', header=0)

data=data[data['Temperature'] != -999]

data['synthetic_temperature'] = data['st_ft_synthetic'].astype('str') + '_' + data['Temperature'].astype('str')

data = pd.get_dummies(data, columns=['synthetic_temperature'])

cols=['Injury','AccStd', 'DecAvg', 'MaxS', #'TorqueAvg', 

      'st_ft_synthetic',

      'Temperature', 'pr_fatigue_dist_run', 'm_a_dt_dir_180',

      'synthetic_temperature_0.0_33','synthetic_temperature_0.0_38',

      'synthetic_temperature_1.0_39','synthetic_temperature_1.0_42','synthetic_temperature_0.0_44', 

      'synthetic_temperature_1.0_44','synthetic_temperature_0.0_45','synthetic_temperature_1.0_45',

      'synthetic_temperature_1.0_46','synthetic_temperature_1.0_47','synthetic_temperature_0.0_48',

      'synthetic_temperature_0.0_52', 'synthetic_temperature_0.0_53',   'synthetic_temperature_1.0_55', 

      'synthetic_temperature_1.0_57','synthetic_temperature_0.0_58','synthetic_temperature_1.0_60',

      'synthetic_temperature_0.0_61','synthetic_temperature_1.0_61','synthetic_temperature_1.0_62',

      'synthetic_temperature_1.0_63','synthetic_temperature_1.0_65','synthetic_temperature_0.0_67',

      'synthetic_temperature_0.0_68','synthetic_temperature_1.0_68','synthetic_temperature_0.0_70',

      'synthetic_temperature_1.0_70','synthetic_temperature_0.0_71','synthetic_temperature_0.0_72', 

      'synthetic_temperature_1.0_73','synthetic_temperature_1.0_74','synthetic_temperature_0.0_75',

      'synthetic_temperature_1.0_75','synthetic_temperature_0.0_76','synthetic_temperature_1.0_76',

      'synthetic_temperature_1.0_78', 'synthetic_temperature_0.0_79','synthetic_temperature_0.0_80',

      'synthetic_temperature_0.0_81','synthetic_temperature_1.0_84']

data=data[cols].dropna()



data['Intercept'] = 1



X=data.drop('Injury',axis=1)

y=data['Injury']

logit_model=sm.Logit(y,X)

result=logit_model.fit()

sm_lr_probs = result.predict(X)

sm_lr_probs = np.asarray(sm_lr_probs.to_list())

print(result.summary2())
ns_probs = [0 for _ in range(len(y))]

lr_probs = sm_lr_probs #[:, 1]

# calculate scores

ns_auc = roc_auc_score(y, ns_probs)

lr_auc = roc_auc_score(y, lr_probs)

# summarize scores

print('No Skill: ROC AUC=%.3f' % (ns_auc))

print('Logistic: ROC AUC=%.3f' % (lr_auc))

# calculate roc curves

ns_fpr, ns_tpr, _ = roc_curve(y, ns_probs)

lr_fpr, lr_tpr, _ = roc_curve(y, lr_probs)

# plot the roc curve for the model

plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')

plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')

# axis labels

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

# show the legend

plt.legend()

# show the plot

plt.show()
plt.rcParams["axes.labelsize"] = 14

plt.rcParams["xtick.labelsize"] = 12

plt.rcParams["axes.titlesize"] = 18

plt.rcParams["xtick.alignment"] = 'center'



fig = plt.figure(figsize=(15,5))

sns.set_style('white')

ax1=sns.countplot(x='RosterPosition',data=df_inj, palette=sns.color_palette("husl", 8))



ax1.set(xlabel='$\it{Roster\,\,Position}$', ylabel='$\it{Count}$', title='Roster Position Counts for\nInjured Players')

# plt.xticks(rotation=45)

plt.show()





rp_summary = pd.DataFrame({'Percentage': round(df_inj.groupby(('RosterPosition')).size() / len(df_inj),4), 

              'Count':df_inj.groupby(('RosterPosition')).size()})



a1 = rp_summary.to_html(formatters={

    'Count': '{:,}'.format,

    'Percentage': '{:,.2%}'.format

})

HTML(a1)#Removed indices
plt.rcParams["axes.labelsize"] = 14

plt.rcParams["xtick.labelsize"] = 14

plt.rcParams["axes.titlesize"] = 18

plt.rcParams["xtick.alignment"] = 'center'



fig = plt.figure(figsize=(8,4))

sns.set_style('white')

temp=df_all[df_all['Temperature']!=-999]

#sns.violinplot(x='Injury', y='Temperature', data=temp,palette='rainbow')

ax1=sns.boxplot(x='Injury', y='Temperature', data=temp, palette=['red', 'darkblue'])







ax1.set(xlabel='$\it{Injury\,\,Event}$', ylabel='$\it{Temperature (F)}$', title='Temperature distribution\nacross Injured Players')

# plt.xticks(rotation=45)

plt.show()



plt.rcParams["axes.labelsize"] = 14

plt.rcParams["xtick.labelsize"] = 14

plt.rcParams["axes.titlesize"] = 18

plt.rcParams["xtick.alignment"] = 'center'



fig = plt.figure(figsize=(15,5))

ax1=sns.distplot(df_all['DaysRest'],bins=20,kde=False)



ax1.set(xlabel='$\it{Number\,\,of\,\,Days\,\,of\,\,Rest\,\,Between\,\,Games}$', ylabel='$\it{Count}$', title='Days Between Games\nDistribution')

# plt.xticks(rotation=45)

plt.show()
plt.rcParams["axes.labelsize"] = 14

plt.rcParams["xtick.labelsize"] = 14

plt.rcParams["axes.titlesize"] = 18

plt.rcParams["xtick.alignment"] = 'center'



fig = plt.figure(figsize=(12,5))

rest = df_all[df_all['DaysRest']<30]

ax1=sns.distplot(rest['DaysRest'],bins=10,kde=False)



ax1.set(xlabel='$\it{Number\,\,of\,\,Days\,\,of\,\,Rest\,\,Between\,\,Games}$', ylabel='$\it{Count}$', title='Days Between Games\nDistribution (without outliers)')

# plt.xticks(rotation=45)

plt.show()
restinj = df_inj[df_inj['DaysRest']<15]

resthealth = df_health[df_health['DaysRest']<15]

a1=pd.DataFrame({'Count injured':restinj.groupby(('DaysRest')).size(),

                 'Count not injured':resthealth.groupby(('DaysRest')).size(),

                 'Percentage injured': round(restinj.groupby(('DaysRest')).size() / len(restinj),4),

                 'Percentage not injured': round(resthealth.groupby(('DaysRest')).size() / len(resthealth),4)}).fillna(0).reset_index()



a2 = a1.astype({"DaysRest": int, "Count injured": int})# Convert to integers, thus removing decimals



a2 = a2.astype({"Count injured": int})# Convert count injured to integer, thus removing decimals

a2 = a2.to_html(formatters={

    'Count injured': '{:,}'.format,

    'Count not injured': '{:,}'.format,

    'Percentage injured': '{:,.2%}'.format,

    'Percentage not injured': '{:,.2%}'.format

}, index=False)

HTML(a2)#Removed indices
injcount = a1['Count injured']

healthcount = a1['Count not injured']

chisq = np.array([injcount,healthcount])

chi2_stat, p_val, dof, ex = stats.chi2_contingency(chisq)



DaysRest_v_inj_ChiSq_stats2 = pd.DataFrame({#'Analysis':'Weather v. Injury occurrence',

                 'Chi2 Stat Result':chi2_stat,

                 'Degrees of Freedom': dof,

                 'P-Value': p_val}, index=['Days of rest v. Injury occurrence'])



HTML(DaysRest_v_inj_ChiSq_stats2.to_html(index=True))#Prettified
plt.rcParams["axes.labelsize"] = 14

plt.rcParams["xtick.labelsize"] = 14

plt.rcParams["axes.titlesize"] = 18

plt.rcParams["xtick.alignment"] = 'center'



ax1=sns.violinplot(x='DaysRest', y='InjurySeverity', data=restinj,palette=sns.color_palette("husl", 4),order=['1-7','7-28','28-42','42+'])



ax1.set(xlabel='$\it{Number\,\,of\,\,Days\,\,of\,\,Rest\,\,Between\,\,Games}$', ylabel='$\it{Injury\,\,Severity\,\,(Days\,\,recovery)}$', title='Distribution of Days of Rest Between Games\nand Injury Severity')

# plt.xticks(rotation=45)

plt.show()
multi_inj_players = df_all[df_all['DoubleInj']==True]



HTML(multi_inj_players.to_html(index=False))#Prettified, index removed
dates=df_all[(df_all['PlayerKey']==45950)&(df_all['PlayerDay']<300)]['PlayerDay']

names=df_all[(df_all['PlayerKey']==45950)&(df_all['PlayerDay']<300)]['BodyPart'].fillna(0)

# levels

levels = np.tile([-5, 5, -3, 3, -1, 1],

                 int(np.ceil(len(dates)/6)))[:len(dates)]

#Season 1

# Create figure and plot a stem plot with timeline

fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)

ax.set(xlabel='$\it{Days\,\,Since\,\,t0}$', title="Player 45950 Season 1")

markerline, stemline, baseline = ax.stem(dates, levels,

                                         linefmt="C3-", basefmt="k-")#,

#                                         use_line_collection=True) #--> This does not work for me

plt.setp(markerline, mec="k", mfc="w", zorder=3)

# Shift the markers to the baseline by replacing the y-data by zeros.

markerline.set_ydata(np.zeros(len(dates)))

# annotate lines

vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]

for d, l, r, va in zip(dates, levels, names, vert):

    ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*3),

                textcoords="offset points", va=va, ha="right")

# remove y axis and spines

ax.get_yaxis().set_visible(False)

for spine in ["left", "top", "right"]:

    ax.spines[spine].set_visible(False)

print("\n")# 2

print("\n")

# 2

dates=df_all[(df_all['PlayerKey']==45950)&(df_all['PlayerDay']>300)]['PlayerDay']

names=df_all[(df_all['PlayerKey']==45950)&(df_all['PlayerDay']>300)]['BodyPart'].fillna(0)



levels = np.tile([-5, 5, -3, 3, -1, 1],

                 int(np.ceil(len(dates)/6)))[:len(dates)]

# 

fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)

ax.set(xlabel='$\it{Days\,\,Since\,\,t0}$', title="Player 45950 Season 2")

markerline, stemline, baseline = ax.stem(dates, levels,

                                         linefmt="C3-", basefmt="k-")#,

#                                         use_line_collection=True) #--> This does not work for me

plt.setp(markerline, mec="k", mfc="w", zorder=3)

# 

markerline.set_ydata(np.zeros(len(dates)))

# annotate lines

vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]

for d, l, r, va in zip(dates, levels, names, vert):

    ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*3),

                textcoords="offset points", va=va, ha="right")

# remove y axis and spines

ax.get_yaxis().set_visible(False)

for spine in ["left", "top", "right"]:

    ax.spines[spine].set_visible(False)

ax.margins(y=0.1)

plt.show()
dates=df_all[(df_all['PlayerKey']==44449)&(df_all['PlayerDay']<300)]['PlayerDay']

names=df_all[(df_all['PlayerKey']==44449)&(df_all['PlayerDay']<300)]['BodyPart'].fillna(0)

# Choose some nice levels

levels = np.tile([-5, 5, -3, 3, -1, 1],

                 int(np.ceil(len(dates)/6)))[:len(dates)]

# Create figure and plot a stem plot with the date

fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)

ax.set(xlabel='$\it{Days\,\,Since\,\,t0}$', title="Player 44449 Season 1")

markerline, stemline, baseline = ax.stem(dates, levels,

                                         linefmt="C3-", basefmt="k-")#,

#                                         use_line_collection=True) #--> This does not work for me

plt.setp(markerline, mec="k", mfc="w", zorder=3)

# Shift the markers to the baseline by replacing the y-data by zeros.

markerline.set_ydata(np.zeros(len(dates)))

# annotate lines

vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]

for d, l, r, va in zip(dates, levels, names, vert):

    ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*3),

                textcoords="offset points", va=va, ha="right")

# remove y axis and spines

ax.get_yaxis().set_visible(False)

for spine in ["left", "top", "right"]:

    ax.spines[spine].set_visible(False)

print("\n")

print("\n")

dates=df_all[(df_all['PlayerKey']==44449)&(df_all['PlayerDay']>300)]['PlayerDay']

names=df_all[(df_all['PlayerKey']==44449)&(df_all['PlayerDay']>300)]['BodyPart'].fillna(0)

# Choose some nice levels

levels = np.tile([-5, 5, -3, 3, -1, 1],

                 int(np.ceil(len(dates)/6)))[:len(dates)]

# Create figure and plot a stem plot with the date

fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)

ax.set(xlabel='$\it{Days\,\,Since\,\,t0}$', title="Player 44449 Season 2")

markerline, stemline, baseline = ax.stem(dates, levels,

                                         linefmt="C3-", basefmt="k-")#,

#                                          use_line_collection=True)

plt.setp(markerline, mec="k", mfc="w", zorder=3)

# Shift the markers to the baseline by replacing the y-data by zeros.

markerline.set_ydata(np.zeros(len(dates)))

# annotate lines

vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]

for d, l, r, va in zip(dates, levels, names, vert):

    ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*3),

                textcoords="offset points", va=va, ha="right")

# remove y axis and spines

ax.get_yaxis().set_visible(False)

for spine in ["left", "top", "right"]:

    ax.spines[spine].set_visible(False)

ax.margins(y=0.1)

plt.show()
dates=df_all[(df_all['PlayerKey']==43540)&(df_all['PlayerDay']<300)]['PlayerDay']

names=df_all[(df_all['PlayerKey']==43540)&(df_all['PlayerDay']<300)]['BodyPart'].fillna(0)

# Choose some nice levels

levels = np.tile([-5, 5, -3, 3, -1, 1],

                 int(np.ceil(len(dates)/6)))[:len(dates)]

# Create figure and plot a stem plot with the date

fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)

ax.set(xlabel='$\it{Days\,\,Since\,\,t0}$', title="Player 43540 Season 1")

markerline, stemline, baseline = ax.stem(dates, levels,

                                         linefmt="C3-", basefmt="k-")#,

#                                         use_line_collection=True) #--> This does not work for me

plt.setp(markerline, mec="k", mfc="w", zorder=3)

# Shift the markers to the baseline by replacing the y-data by zeros.

markerline.set_ydata(np.zeros(len(dates)))

# annotate lines

vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]

for d, l, r, va in zip(dates, levels, names, vert):

    ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*3),

                textcoords="offset points", va=va, ha="right")

# remove y axis and spines

ax.get_yaxis().set_visible(False)

for spine in ["left", "top", "right"]:

    ax.spines[spine].set_visible(False)

print("\n")

print("\n")

dates=df_all[(df_all['PlayerKey']==43540)&(df_all['PlayerDay']>300)]['PlayerDay']

names=df_all[(df_all['PlayerKey']==43540)&(df_all['PlayerDay']>300)]['BodyPart'].fillna(0)

# Choose some nice levels

levels = np.tile([-5, 5, -3, 3, -1, 1],

                 int(np.ceil(len(dates)/6)))[:len(dates)]

# Create figure and plot a stem plot with the date

fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)

ax.set(xlabel='$\it{Days\,\,Since\,\,t0}$', title="Player 43540 Season 2")

markerline, stemline, baseline = ax.stem(dates, levels,

                                         linefmt="C3-", basefmt="k-")#,

#                                         use_line_collection=True) #--> This does not work for me

plt.setp(markerline, mec="k", mfc="w", zorder=3)

# Shift the markers to the baseline by replacing the y-data by zeros.

markerline.set_ydata(np.zeros(len(dates)))

# annotate lines

vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]

for d, l, r, va in zip(dates, levels, names, vert):

    ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*3),

                textcoords="offset points", va=va, ha="right")

# remove y axis and spines

ax.get_yaxis().set_visible(False)

for spine in ["left", "top", "right"]:

    ax.spines[spine].set_visible(False)

ax.margins(y=0.1)

plt.show()
dates=df_all[(df_all['PlayerKey']==33337)&(df_all['PlayerDay']<300)]['PlayerDay']

names=df_all[(df_all['PlayerKey']==33337)&(df_all['PlayerDay']<300)]['BodyPart'].fillna(0)

# Choose some nice levels

levels = np.tile([-5, 5, -3, 3, -1, 1],

                 int(np.ceil(len(dates)/6)))[:len(dates)]

# Create figure and plot a stem plot with the date

fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)

ax.set(xlabel='$\it{Days\,\,Since\,\,t0}$', title="Player 33337 Season 1")

markerline, stemline, baseline = ax.stem(dates, levels,

                                         linefmt="C3-", basefmt="k-")#,

#                                         use_line_collection=True) #--> This does not work for me

plt.setp(markerline, mec="k", mfc="w", zorder=3)

# Shift the markers to the baseline by replacing the y-data by zeros.

markerline.set_ydata(np.zeros(len(dates)))

# annotate lines

vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]

for d, l, r, va in zip(dates, levels, names, vert):

    ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*3),

                textcoords="offset points", va=va, ha="right")

# remove y axis and spines

ax.get_yaxis().set_visible(False)

for spine in ["left", "top", "right"]:

    ax.spines[spine].set_visible(False)

print("\n")

print("\n")

dates=df_all[(df_all['PlayerKey']==33337)&(df_all['PlayerDay']>300)]['PlayerDay']

names=df_all[(df_all['PlayerKey']==33337)&(df_all['PlayerDay']>300)]['BodyPart'].fillna(0)

# Choose some nice levels

levels = np.tile([-5, 5, -3, 3, -1, 1],

                 int(np.ceil(len(dates)/6)))[:len(dates)]

# Create figure and plot a stem plot with the date

fig, ax = plt.subplots(figsize=(8.8, 4), constrained_layout=True)

ax.set(xlabel='$\it{Days\,\,Since\,\,t0}$', title="Player 33337 Season 2")

markerline, stemline, baseline = ax.stem(dates, levels,

                                         linefmt="C3-", basefmt="k-")#,

#                                         use_line_collection=True) #--> This does not work for me

plt.setp(markerline, mec="k", mfc="w", zorder=3)

# Shift the markers to the baseline by replacing the y-data by zeros.

markerline.set_ydata(np.zeros(len(dates)))

# annotate lines

vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]

for d, l, r, va in zip(dates, levels, names, vert):

    ax.annotate(r, xy=(d, l), xytext=(-3, np.sign(l)*3),

                textcoords="offset points", va=va, ha="right")

# remove y axis and spines

ax.get_yaxis().set_visible(False)

for spine in ["left", "top", "right"]:

    ax.spines[spine].set_visible(False)

ax.margins(y=0.1)

plt.show()
plt.rcParams["axes.labelsize"] = 14

plt.rcParams["xtick.labelsize"] = 14

plt.rcParams["axes.titlesize"] = 18

plt.rcParams["xtick.alignment"] = 'center'



ax1=sns.boxplot(x='PlayerGamePlay', y='InjurySeverity', data=df_inj,palette=sns.color_palette("husl", 4),order=['1-7','7-28','28-42','42+'])



ax1.set(ylabel='$\it{Injury\,\,Severity\,\,(Days\,\,recovery)}$', xlabel='$\it{Count\,\,of\,\,Plays}$', title='Distribution of Number of Plays Involvement\nand Injury Severity')

# plt.xticks(rotation=45)

plt.show()



inj_last=pd.merge(season,inj,how='inner',on=['PlayerKey','GameID','PlayKey'])

inj_cont = pd.merge(season,inj[inj['PlayerKey'].isin(inj_last['PlayerKey'])==False].dropna(),

                    how='inner',on=['PlayerKey','GameID'])

inj_miss = pd.merge(season,inj[inj['PlayKey'].isnull()],how='inner',on=['PlayerKey','GameID'])

#its ok to do the miss and cont merges on the inj table because 

#the player with a double entry is already captured by the first merge and there is no duplication
a1=pd.DataFrame({'Count Left Game':inj_last.groupby(('InjurySeverity')).size(),

                 'Count Cont. Game':inj_cont.groupby(('InjurySeverity')).size(),

                 'Count Miss.':inj_miss.groupby(('InjurySeverity')).size(),

                 'Percentage Left': round(inj_last.groupby(('InjurySeverity')).size() / len(inj_last),2),

                 'Percentage Cont.': round(inj_cont.groupby(('InjurySeverity')).size() / len(inj_cont),2),

                 'Percentage Miss.': round(inj_miss.groupby(('InjurySeverity')).size() / len(inj_miss),2)}).fillna(0)



plt.rcParams["axes.labelsize"] = 14

plt.rcParams["xtick.labelsize"] = 14

plt.rcParams["axes.titlesize"] = 18

plt.rcParams["xtick.alignment"] = 'center'



fig = plt.figure()

order=['1-7','7-28','28-42','42+']

ax1 = a1[['Percentage Left','Percentage Cont.','Percentage Miss.']].loc[order].plot(kind="bar", color=sns.color_palette("husl", 3))

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



ax1.set(ylabel='$\it{Percentage}$', xlabel='$\it{Injury\,\,Severity\,\,(Days\,\,recovery)}$', title="Distribution of Player's decision Upon Injury Event\nand Injury Severity")

plt.xticks(rotation=0)

plt.show()



a1=pd.DataFrame({'Synthetic field': round(inj[(inj['Surface']=='Synthetic')]

                                          .groupby(('BodyPart')).size() / len(inj[(inj['Surface']=='Synthetic')]),2),

                'Natural field': round(inj[(inj['Surface']=='Natural')]

                                          .groupby(('BodyPart')).size() / len(inj[(inj['Surface']=='Natural')]),2)}).fillna(0)



plt.rcParams["axes.labelsize"] = 14

plt.rcParams["xtick.labelsize"] = 14

plt.rcParams["axes.titlesize"] = 18

plt.rcParams["xtick.alignment"] = 'center'



fig = plt.figure()

ax1 = a1.plot(kind="bar", color=['darkgray', 'darkgreen'])

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



ax1.set(ylabel='$\it{Distribution\,\,of\,\,injuries}$'+"\n"+'$\it{across\,\,field\,\,types}$', xlabel='$\it{Body\,\,Part\,\,Injured}$', title="Percentage occurrence\nacross field types")

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

plt.show()



a1=pd.DataFrame({'Synthetic field': round(inj[(inj['Surface']=='Synthetic')]

                                          .groupby(('InjurySeverity')).size() / len(inj[(inj['Surface']=='Synthetic')]),2),

                'Natural field': round(inj[(inj['Surface']=='Natural')]

                                          .groupby(('InjurySeverity')).size() / len(inj[(inj['Surface']=='Natural')]),2)}).fillna(0)



plt.rcParams["axes.labelsize"] = 14

plt.rcParams["xtick.labelsize"] = 14

plt.rcParams["axes.titlesize"] = 18

plt.rcParams["xtick.alignment"] = 'center'



fig = plt.figure(figsize=(15,1))

order=['1-7','7-28','28-42','42+']

ax1 = a1.loc[order].plot(kind="bar", color=['darkgray', 'darkgreen'])

plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)



ax1.set(ylabel='$\it{Distribution\,\,of\,\,injuries}$'+"\n"+'$\it{across\,\,field\,\,types}$', xlabel='$\it{Injury\,\,Severity\,\,(Days\,\,recovery)}$', title="Percentage occurrence\nacross field types")

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

plt.show()





plt.rcParams["axes.labelsize"] = 14

plt.rcParams["xtick.labelsize"] = 14

plt.rcParams["axes.titlesize"] = 18

plt.rcParams["xtick.alignment"] = 'center'



ax1 = sns.countplot(x='InjurySeverity', data=inj, palette=sns.color_palette("husl", 5), order=['1-7','7-28','28-42','42+'],hue= 'BodyPart')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



ax1.set(ylabel='$\it{Distribution\,\,of\,\,injuries}$'+"\n"+'$\it{across\,\,field\,\,types}$', xlabel='$\it{Body\,\,Part\,\,Injured}$', title="Percentage occurrence\nacross field types")

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

plt.show()


