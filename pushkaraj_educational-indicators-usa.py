import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

import matplotlib.patches as mpatches
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Series(indicator) dataframe

df_ser=pd.read_csv('/kaggle/input/global-education-statistics/EdStatsSeries.csv')

# Attainment series

att_ser,att_iname=zip(*df_ser[df_ser['Topic']=='Attainment'][['Series Code','Indicator Name']].values)
# Stats dataframe (with only United States and European Union)

df_stats=pd.read_csv('/kaggle/input/global-education-statistics/EdStatsData.csv')

df_usa_stats=df_stats[df_stats['Country Name']=='United States']

df_stats_cols=df_stats.columns
female_age_no_edu_ser=[val for ind,val in enumerate(att_ser[:30]) if ind%2==0]

female_age_no_edu_df=df_usa_stats[df_usa_stats['Indicator Code'].apply(lambda val:val in female_age_no_edu_ser)].copy()

perc_female_no_edu=female_age_no_edu_df[female_age_no_edu_df['Country Code']=='USA'][df_stats_cols[4:]].dropna(axis=1).sum(axis=0)



pop_age_no_edu_ser=[val for ind,val in enumerate(att_ser[:30]) if ind%2!=0]

pop_age_no_edu_df=df_stats[df_stats['Indicator Code'].apply(lambda val:val in pop_age_no_edu_ser)].copy()

perc_pop_no_edu=pop_age_no_edu_df[pop_age_no_edu_df['Country Code']=='USA'][df_stats_cols[4:]].dropna(axis=1).sum(axis=0)
fig,[ax1,ax2]=plt.subplots(figsize=(17,5),nrows=1,ncols=2)



colors=['#89fe05' if val<0 else '#ff474c' for val in perc_female_no_edu.diff().values]

ax1.bar(perc_female_no_edu.index,perc_female_no_edu.values,color=colors)



ax1.set_ylabel('Percentage of female population with no education',fontsize=12)

ax1.set_xlabel('Years',fontsize=14)



colors=['#89fe05' if val<0 else '#ff474c' for val in perc_pop_no_edu.diff().values]

ax2.bar(perc_pop_no_edu.index,perc_pop_no_edu.values,color=colors)

ax2.set_ylabel('Percentage of population with no education',fontsize=13)

ax2.set_xlabel('Years',fontsize=14)



red_patch = mpatches.Patch(color='#ff474c', label='Higher than previous year')

green_patch = mpatches.Patch(color='#89fe05', label='Lower than previous year')

fig.legend(handles=[red_patch,green_patch]);
fig,[ax1,ax2]=plt.subplots(figsize=(17,3),ncols=2,nrows=1)



pop_avg_year_prim=[val for ind,val in enumerate(att_ser[120:150]) if ind%2==0]

pop_avg_year_prim_df=df_usa_stats[df_usa_stats['Indicator Code'].apply(lambda val:val in pop_avg_year_prim)][df_stats_cols[4:]].dropna(axis=1).sum(axis=0)

colors=['#ff474c' if val<0 else '#89fe05' for val in pop_avg_year_prim_df.diff().values]

ax1.bar(pop_avg_year_prim_df.index,pop_avg_year_prim_df.values,color=colors)

ax1.set_xlabel('Years')

ax1.set_ylabel('Average years of\nprimary schooling completed\nby total population',fontsize=13)

ax1.set_title('Total population')



fem_avg_year_prim=[val for ind,val in enumerate(att_ser[120:150]) if ind%2!=0]

fem_avg_year_prim_df=df_usa_stats[df_usa_stats['Indicator Code'].apply(lambda val:val in fem_avg_year_prim)][df_stats_cols[4:]].dropna(axis=1).sum(axis=0)

colors=['#ff474c' if val<0 else '#89fe05' for val in fem_avg_year_prim_df.diff().values]

ax2.bar(fem_avg_year_prim_df.index,fem_avg_year_prim_df.values,color=colors)

ax2.set_xlabel('Years')

ax2.set_ylabel('Average years of\nprimary schooling completed\nby female population',fontsize=13)

ax2.set_title('Female')



red_patch = mpatches.Patch(color='#ff474c', label='Lower than previous year')

green_patch = mpatches.Patch(color='#89fe05', label='Higher than previous year')

fig.legend(handles=[red_patch,green_patch]);
infra_ser,infra_iname=zip(*df_ser[df_ser['Topic']=='Infrastructure: Communications'][['Series Code','Indicator Name']].values)
fig,[ax1,ax2]=plt.subplots(figsize=(18,6),nrows=1,ncols=2)



pc_df=df_usa_stats[df_usa_stats['Indicator Code'].apply(lambda val:val == infra_ser[0])][df_stats_cols[4:]].dropna(axis=1)

cmap = plt.get_cmap('GnBu')

colors=[cmap(i) for i in np.linspace(0, 1, len(pc_df.columns))]

pc_df.iloc[0].plot.bar(ax=ax1,color=colors)

ax1.set_ylabel(infra_iname[0],fontsize=14)



inter_df=df_usa_stats[df_usa_stats['Indicator Code'].apply(lambda val:val == infra_ser[1])][df_stats_cols[4:]].dropna(axis=1)

cmap = plt.get_cmap('GnBu')

colors=[cmap(i) for i in np.linspace(0, 1, len(inter_df.columns))]

inter_df.iloc[0].plot.bar(ax=ax2,color=colors)

ax2.set_ylabel(infra_iname[1],fontsize=14);
lrout_ser,lrout_iname=zip(*df_ser[df_ser['Topic']=='Learning Outcomes'][['Series Code','Indicator Name']].values)
indi_name='PISA: Mean performance on the mathematics scale'

mean_per_math_scale=df_usa_stats[df_usa_stats['Indicator Code']==[val[0] for val in list(zip(lrout_ser,lrout_iname)) if indi_name in val[1]][0]][df_stats_cols[4:]].dropna(axis=1).iloc[0]



colors=['#ff474c' if val<0 else '#89fe05' for val in mean_per_math_scale.to_frame().diff().values]



fig,ax=plt.subplots(figsize=(15,3))

ax.bar(mean_per_math_scale.index,mean_per_math_scale.values,color=colors)

ax.set_ylabel('Mean performance on\nthe mathematics scale',fontsize=13)



red_patch = mpatches.Patch(color='#ff474c', label='Lower than previous year')

green_patch = mpatches.Patch(color='#89fe05', label='Higher than previous year')

fig.legend(handles=[red_patch,green_patch]);
indi_name='PISA: Mean performance on the reading scale'

mean_per_read_scale=df_usa_stats[df_usa_stats['Indicator Code']==[val[0] for val in list(zip(lrout_ser,lrout_iname)) if indi_name in val[1]][0]][df_stats_cols[4:]].dropna(axis=1).iloc[0]



colors=['#ff474c' if val<0 else '#89fe05' for val in mean_per_read_scale.to_frame().diff().values]



fig,ax=plt.subplots(figsize=(15,3))

ax.bar(mean_per_read_scale.index,mean_per_read_scale.values,color=colors)

ax.set_ylabel('Mean performance on\nthe reading scale',fontsize=13)



red_patch = mpatches.Patch(color='#ff474c', label='Lower than previous year')

green_patch = mpatches.Patch(color='#89fe05', label='Higher than previous year')

fig.legend(handles=[red_patch,green_patch]);
indi_name='PISA: Mean performance on the science scale'

mean_per_sci_scale=df_usa_stats[df_usa_stats['Indicator Code']==[val[0] for val in list(zip(lrout_ser,lrout_iname)) if indi_name in val[1]][0]][df_stats_cols[4:]].dropna(axis=1).iloc[0]



colors=['#ff474c' if val<0 else '#89fe05' for val in mean_per_sci_scale.to_frame().diff().values]



fig,ax=plt.subplots(figsize=(15,3))

ax.bar(mean_per_sci_scale.index,mean_per_sci_scale.values,color=colors)

ax.set_ylabel('Mean performance on\nthe science scale',fontsize=13)



red_patch = mpatches.Patch(color='#ff474c', label='Lower than previous year')

green_patch = mpatches.Patch(color='#89fe05', label='Higher than previous year')

fig.legend(handles=[red_patch,green_patch]);
indi_name='PIAAC: Mean Adult Literacy Proficiency. Total'

df_stats_lit=df_stats.iloc[df_stats[df_stats['Indicator Name']==indi_name]['2012'].dropna(axis=0,how='all').index][['Country Name','2012']]



usa_val=df_stats_lit[df_stats_lit['Country Name']=='United States']['2012'].values[0]

df_stats_lit=df_stats_lit[df_stats_lit['Country Name']!='United States']



colors=['#ff474c' if val<usa_val else '#89fe05' for val in df_stats_lit['2012'].values]



fig,ax=plt.subplots(figsize=(17,3))



red_patch = mpatches.Patch(color='#ff474c', label='Lower than USA')

green_patch = mpatches.Patch(color='#89fe05', label='Higher than USA')

ax.legend(handles=[red_patch,green_patch])





ax.bar(df_stats_lit['Country Name'],df_stats_lit['2012'],color=colors)

ax.bar(['United States'],usa_val)

ax.set_xticklabels(df_stats_lit['Country Name'].tolist()+['United States'],rotation=90)

ax.set_ylabel('Mean Adult Literacy\nProficiency, 2012',fontsize=12);
df_stats_num=df_stats.iloc[df_stats[df_stats['Indicator Name']=='PIAAC: Mean Adult Numeracy Proficiency. Total']['2012'].dropna(axis=0,how='all').index][['Country Name','2012']]



usa_val=df_stats_num[df_stats_num['Country Name']=='United States']['2012'].values[0]

df_stats_num=df_stats_num[df_stats_num['Country Name']!='United States']



colors=['#ff474c' if val<usa_val else '#89fe05' for val in df_stats_num['2012'].values]



fig,ax=plt.subplots(figsize=(17,3))



red_patch = mpatches.Patch(color='#ff474c', label='Lower than USA')

green_patch = mpatches.Patch(color='#89fe05', label='Higher than USA')

ax.legend(handles=[red_patch,green_patch])





ax.bar(df_stats_num['Country Name'],df_stats_num['2012'],color=colors)

ax.bar(['United States'],usa_val)

ax.set_xticklabels(df_stats_num['Country Name'].tolist()+['United States'],rotation=90)

ax.set_ylabel('Mean Adult Numeracy\nProficiency, 2012',fontsize=12);
df_stats_prob_no_cs=df_stats.iloc[df_stats[df_stats['Indicator Name']=='PIAAC: Adults by proficiency level in problem solving in technology-rich environments (%). No computer experience']['2012'].dropna(axis=0,how='all').index][['Country Name','2012']]



usa_val=df_stats_prob_no_cs[df_stats_prob_no_cs['Country Name']=='United States']['2012'].values[0]

df_stats_prob_no_cs=df_stats_prob_no_cs[df_stats_prob_no_cs['Country Name']!='United States']



colors=['#ff474c' if val<usa_val else '#89fe05' for val in df_stats_prob_no_cs['2012'].values]



fig,ax=plt.subplots(figsize=(17,3))



red_patch = mpatches.Patch(color='#ff474c', label='Lower than USA')

green_patch = mpatches.Patch(color='#89fe05', label='Higher than USA')

ax.legend(handles=[red_patch,green_patch])





ax.bar(df_stats_prob_no_cs['Country Name'],df_stats_prob_no_cs['2012'],color=colors)

ax.bar(['United States'],usa_val)

ax.set_xticklabels(df_stats_prob_no_cs['Country Name'].tolist()+['United States'],rotation=90)

ax.set_ylabel('Adults by proficiency level in problem\nsolving in technology-rich environments (%).\nNo computer experience, 2012',fontsize=12);
df_stats_prob_with_cs=df_stats.iloc[df_stats[df_stats['Indicator Name']=='PIAAC: Adults by proficiency level in problem solving in technology-rich environments (%). Opted out of computer-based assessment']['2012'].dropna(axis=0,how='all').index][['Country Name','2012']]



usa_val=df_stats_prob_with_cs[df_stats_prob_with_cs['Country Name']=='United States']['2012'].values[0]

df_stats_prob_with_cs=df_stats_prob_with_cs[df_stats_prob_with_cs['Country Name']!='United States']



colors=['#ff474c' if val<usa_val else '#89fe05' for val in df_stats_prob_with_cs['2012'].values]



fig,ax=plt.subplots(figsize=(17,3))



red_patch = mpatches.Patch(color='#ff474c', label='Lower than USA')

green_patch = mpatches.Patch(color='#89fe05', label='Higher than USA')

ax.legend(handles=[red_patch,green_patch])





ax.bar(df_stats_prob_with_cs['Country Name'],df_stats_prob_with_cs['2012'],color=colors)

ax.bar(['United States'],usa_val)

ax.set_xticklabels(df_stats_prob_with_cs['Country Name'].tolist()+['United States'],rotation=90)

ax.set_ylabel('Adults by proficiency level in problem solving\nin technology-rich environments (%). Opted out of\ncomputer-based assessment, 2012',fontsize=12);
df_gdp_edu_exp=df_stats.iloc[df_stats[df_stats['Indicator Name']=='Expenditure on education as % of total government expenditure (%)']['2012'].dropna(axis=0).index][['Country Name','2012']].sort_values(by='2012')

usa_val=df_gdp_edu_exp[df_gdp_edu_exp['Country Name']=='United States']['2012'].iloc[0]

colors=['#ff474c' if val<usa_val else '#89fe05' for val in df_gdp_edu_exp['2012'].values]

fig,ax=plt.subplots(figsize=(20,6))

ax.bar(df_gdp_edu_exp['Country Name'].values,df_gdp_edu_exp['2012'].values,color=colors)

ax.set_xticklabels(df_gdp_edu_exp['Country Name'].values,rotation=90)

ax.set_ylabel('Expenditure on education as %\nof total government expenditure (%)',fontsize=13)



red_patch = mpatches.Patch(color='#ff474c', label='Lower than USA')

green_patch = mpatches.Patch(color='#89fe05', label='Higher than USA')

fig.legend(handles=[red_patch,green_patch]);