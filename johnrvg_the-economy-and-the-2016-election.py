import numpy as np
import pandas as pd

df = pd.read_excel('../input/election1216.xlsx', sheet_name='data', index_col=0)
print(df.columns)
print('2016:\n','GOP votes, DEM votes, OTH votes')
print([df['gop_2016'].sum(),
       df['dem_2016'].sum(),
       df['oth_2016'].sum()
       ])
print('\n2012:\n','GOP votes, DEM votes, OTH votes')
print([df['gop_2012'].sum(),
       df['dem_2012'].sum(),
       df['oth_2012'].sum()
       ])
# Calculate vote margin by which R won/lost in each county
df['rv_margin_2012'] = (df['gop_2012'] - df['dem_2012'])*100
df['rv_margin_2016'] = (df['gop_2016'] - df['dem_2016'])*100

# Calc states won by each party 2012
r_states12 = []
d_states12 = []
for state in df['st'].dropna().unique():
    stv_margin = df.where(df['st']==state)['rv_margin_2012'].sum()
    if stv_margin == None:
        continue
    if stv_margin <0:
        d_states12.append(state)
    else:
        r_states12.append(state)
print('2012 GOP states: ', len(r_states12))
print(r_states12)
print('2012 DEM states: ', len(d_states12))
print(d_states12)

# Calc states won by each party 2016
r_states16 = []
d_states16 = []
for state in df['st'].dropna().unique():
    stv_margin = df.where(df['st']==state)['rv_margin_2016'].sum()
    if stv_margin == None:
        continue
    if stv_margin <0:
        d_states16.append(state)
    else:
        r_states16.append(state)
print('2016 GOP states: ', len(r_states16))
print(r_states16)
print('2016 DEM states: ', len(d_states16))
print(d_states16)
GOPflip = []
for state in d_states12:
    if state in r_states16:
        GOPflip.append(state)
print('GOP flipped states: ', len(GOPflip), GOPflip)

DEMflip = []
for state in r_states12:
    if state in d_states16:
        DEMflip.append(state)
print('DEM flipped states: ', len(DEMflip), DEMflip)
# Calculate the absolute vote and percent of the vote won/lost by GOP in each county 2012 & 2016
df['rp_margin_2012'] = ((df['gop_2012'] - df['dem_2012'])/df['total_2012'])*100
df['rp_margin_2016'] = ((df['gop_2016'] - df['dem_2016'])/df['total_2016'])*100
df['change_rv_1216'] = df['rv_margin_2016'] - df['rv_margin_2012']
df['change_tv_1216'] = df['total_2016'] - df['total_2012']
df['change_rp_1216'] = (df['rp_margin_2016'] - df['rp_margin_2012'])

# Calculate the absolute and percent change in average total and private manufacturing
#  wages by county for the period 2013-2016 (Obama's 2nd term; 2012 data not available)
df['change_tw_1316'] = df['avg_total_wages_16'] - df['avg_total_wages_13']
df['pct_change_tw_1316'] = (df['change_tw_1316']/df['avg_total_wages_13'])*100
df['change_pmw_1316'] = (df['avg_pm_wages_16'] - df['avg_pm_wages_13'])
df['pct_change_pmw_1316'] = (df['change_pmw_1316']/df['avg_pm_wages_13'])*100

# Calculate the absolute and percent change in exports per capita by county for the period 2012-2016
df['exports_pc_2016'] = (df['exports_2016']/df['pop_2016'])
df['exports_pc_2012'] = (df['exports_2012']/df['pop_2012'])
df['change_epc_1216'] = df['exports_pc_2016'] - df['exports_pc_2012']
df['pct_change_epc_1216'] = (df['change_epc_1216']/df['exports_pc_2012'])*100

# Calculate the absolute and percent change in unemplyment rate (% points) by county 
#  for the period 2013-2016 (Obama's 2nd term; 2012 data not available)
df['change_u_1316'] = df['u_rate_16'] - df['u_rate_13']
df['pct_change_u_1316'] = (df['change_u_1316']/df['u_rate_13'])*100

# Calculate the absolute and percent change in labor force by county 
#  for the period 2013-2016 (Obama's 2nd term; 2012 data not available)
df['change_lf_1316'] = df['labor_force_16'] - df['labor_force_13']
df['pct_change_lf_1316'] = (df['change_lf_1316']/df['labor_force_13'])*100

# Calculate the absolute and percent change in population**
df['change_pop_1216'] = df['pop_2016'] - df['pop_2012']
df['pct_change_pop_1216'] = df['change_pop_1216']/df['pop_2012']*100
# **Population data comes from the trade data set, which comprises of just over 300 (not 3000)
#   counties' data on exports.  These represent the top 50 exporting metropolitan areas
#   as compiled by the ITA.
import seaborn as sns
import matplotlib.pyplot as plt

# Descriptive statistics for GOP margin of victory by county
print('\nDescriptive statistics for GOP 2016 margin of victory by county'
      '(% of total county vote):\n')
print(df['rp_margin_2016'].describe(),'\n')

print(sns.distplot(df['rp_margin_2016'].dropna()))
print('Skewness: ',round(df['rp_margin_2016'].skew(),3))
print('Kurtosis: ',round(df['rp_margin_2016'].kurt(),3))
# Descriptive statistics for county GOP margin of victory CHANGE
print('\nDescriptive statistics for change in GOP margin of victory by county from 2012 to 2016)'
      '(% of total county vote):\n')
print(df['change_rp_1216'].describe(),'\n')

print(sns.distplot(df['change_rp_1216'].dropna()))
print('Skewness: ',round(df['change_rp_1216'].skew(),3))
print('Kurtosis: ',round(df['change_rp_1216'].kurt(),3))
print('Biggest GOP 2016 win:\n', df.loc[df['rp_margin_2016'].idxmax()]['county'],
      round(df['rp_margin_2016'].max(),2),"% of the vote\n")
print('Biggest GOP 2016 loss:\n', df.loc[df['rp_margin_2016'].idxmin()]['county'],
      round(df['rp_margin_2016'].min(),2),"% of the vote\n")
print('Biggest increase in GOP margin from 2012 to 2016:\n', df.loc[df['change_rp_1216'].idxmax()]['county'],
      round(df['change_rp_1216'].max(),2),"% of the vote\n")
print('Biggest decrease in GOP margin from 2012 to 2016::\n', df.loc[df['change_rp_1216'].idxmin()]['county'],
      round(df['change_rp_1216'].min(),2),"% of the vote\n")
print('Correlations:','\n',
    round((df['pop_2016'].corr(df['rp_margin_2016'])),3),
    '(2016 population : 2016 margins)','\n',
    round((df['pop_2016'].corr(df['change_rp_1216'])),3),
    '(2016 population : The change in margins from 2012)','\n',
    round((df['pct_change_pop_1216'].corr(df['rp_margin_2016'])),3),
    '(Change in 2016 population : 2016 margins)','\n',
    round((df['pct_change_pop_1216'].corr(df['change_rp_1216'])),3),
    '(Change in 2016 population : Change in 2016 margins)','\n',
)
plt.scatter(df['pop_2016']/1000000,df['rp_margin_2016'],marker='.')
plt.xlabel('2016 Population by County (millions)')
plt.ylabel('2016 GOP Margin (% of vote)')
plt.show()

plt.scatter(df['pct_change_pop_1216'],df['change_rp_1216'],marker='.')
plt.xlabel('Growth in Population from 2012 to 2016 by County (%)')
plt.ylabel('Change in GOP Margin from 2012 (% of vote)')
plt.show()   
corr_cols_16 = ['rp_margin_2016','avg_total_wages_16','avg_pm_wages_16',
              'exports_pc_2016','u_rate_16','labor_force_16']
df1 = df[corr_cols_16]
print(df1.corr())
corr_cols_1216 = ['change_rp_1216','pct_change_tw_1316','pct_change_pmw_1316',
              'pct_change_epc_1216','pct_change_u_1316','pct_change_lf_1316']
df2 = df[corr_cols_1216]
print(df2.corr())
corr_cols_16oc = ['rp_margin_2016','pct_change_tw_1316','pct_change_pmw_1316',
              'pct_change_epc_1216','pct_change_u_1316','pct_change_lf_1316']
df3 = df[corr_cols_16oc]
print(df3.corr())
corr_cols_c016 = ['change_rp_1216','avg_total_wages_16','avg_pm_wages_16',
              'exports_pc_2016','u_rate_16','labor_force_16']
df4 = df[corr_cols_c016]
print(df4.corr())
plt.scatter(df['avg_total_wages_16']/1000,df['rp_margin_2016'],marker='.')
plt.xlabel('2016 Avg Total Wages ($1000s)')
plt.ylabel('2016 GOP Margin (% of vote)')
plt.show()

plt.scatter(df['avg_total_wages_16']/1000,df['change_rp_1216'],marker='.')
plt.xlabel('2016 Avg Total Wages ($1000s)')
plt.ylabel('Change in GOP Margin from 2012 (% of vote)')
plt.show()
import statsmodels.api as sm

model = sm.OLS(endog=df['rp_margin_2016'],
               exog=df[['avg_total_wages_16','pop_2016']],
               missing='drop')
result = model.fit()
print(result.params)
print(result.summary())
print(
    '1) ',round((df['avg_total_wages_16'].corr(df['pop_2016'])),3),'    Wages : Population','\n',
    '2) ',round((df['pop_2016'].corr(df['rp_margin_2016'])),3),'  Population : GOP vote margins'
)
print(sns.distplot(df['avg_total_wages_16'].dropna()))
print('Skewness: ',round(df['avg_total_wages_16'].skew(),3))
print('Kurtosis: ',round(df['avg_total_wages_16'].kurt(),3))

print(df['avg_total_wages_16'].describe())
w = 60000
total_wages = df['avg_total_wages_16'].count()
sub80_wages = df['avg_total_wages_16'].where(df['avg_total_wages_16']<=w).count()
pcts = round((sub80_wages/total_wages)*100,1)
print(pcts,'percent of counties have an average wage at or below $',w,'per year.')
df5 = df.where(df['avg_total_wages_16']<=w)

model = sm.OLS(endog=df5['rp_margin_2016'],
               exog=df5[['avg_total_wages_16','pop_2016']],
               missing='drop')
result = model.fit()
print(result.params)
print(result.summary())