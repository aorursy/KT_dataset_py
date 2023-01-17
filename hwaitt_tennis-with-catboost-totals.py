import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline

pd.options.mode.chained_assignment = None

sns.set_style("whitegrid")
bins = [0, 13.1, 14.1, 15.1, 16.1, 17.1, 18.1, 19.1, 20.1, 21.1, 22.1, 99999]

binlabels = ['<=13', '(13-14]', '(14-15]', '(15-16]', '(16-17]', '(17-18]', '(18-19]', '(19-20]', '(20-21]', '(21-22]', '>22']



bins_regr = [0, 13.1, 14.1, 15.1, 16.1, 17.1, 18.1, 19.1, 20.1, 21.1, 22.1, 23.1, 24.1, 25.1, 26.1, 27.1, 28.1, 99999]

binlabels_regr = ['<=13','(13-14]','(14-15]','(15-16]','(16-17]','(17-18]','(18-19]','(19-20]', '(20-21]', '(21-22]', 

                    '(22-23]', '(23-24]', '(24-25]', '(25-26]', '(26-27]', '(27-29]', '>28']



bins_diff = [-10, -2, -1, -0.5, -0.25, 0, 1, 2, 3, 4, 5, 6, 7, 99999]

binlabels_diff = ['<-2','<-1','<-0.5','<-0.25','<0','<1','<2','<3','<4','<5', '<6', '<7', '>7']



odds=1.9
def calc_res(df):

    df=df[['TourRank', 'RID', 'GameD', 'Year', 'TTL', 'TPoints', 'REGR']]

    # Should remove low ranked tours

    df=df[df['TourRank']>0]

    df['C']=1

    df['GameD']=pd.to_datetime(df['GameD'])

    df['Month']=df['GameD'].dt.month

    df['Weekday']=df['GameD'].dt.weekday_name



    df['TPointsR']=pd.cut(df['TPoints'], bins=bins, labels=binlabels)

    df['RegrR']=pd.cut(df['REGR'], bins=bins_regr, labels=binlabels_regr)



    # Wager on bookies line: Under

    df['ResU']=np.where(df['TTL'] < df['TPoints'],1,np.NaN)

    df['ResU']=np.where(df['TTL'] > df['TPoints'],0,df['ResU'])

    df['PrfU']=np.where(df['ResU']==1,odds-1,np.where(df['ResU']==0,-1,0))



    # Wager on bookies line: Over

    df['ResO']=np.where(df['TTL'] > df['TPoints'],1,np.NaN)

    df['ResO']=np.where(df['TTL'] < df['TPoints'],0,df['ResO'])

    df['PrfO']=np.where(df['ResO']==1,odds-1,np.where(df['ResO']==0,-1,0))



    # Diffs

    df['Diff']=df['REGR']-df['TPoints']

    df['Wag']=np.where(df['Diff'] > 0,'Ov','Un')

    df['DiffR']=pd.cut(df['Diff'], bins=bins_diff, labels=binlabels_diff)

    return df 
def calc_total(target, df):

    total=df.groupby(['Year',f'{target}R']).sum()[['PrfU','PrfO','C']]

    total.reset_index(inplace=True)

    total['RoiU']=total['PrfU']/total['C']*100

    total['RoiO']=total['PrfO']/total['C']*100



    ytotal=total.groupby(['Year']).sum()[['PrfU','PrfO','C']]

    ytotal['RoiU']=ytotal['PrfU']/ytotal['C']*100

    ytotal['RoiO']=ytotal['PrfO']/ytotal['C']*100

    return (total, ytotal)
dfw=calc_res(pd.read_csv('../input/tennis-20112019/wta_picks.csv'))

dfa=calc_res(pd.read_csv('../input/tennis-20112019/atp_picks.csv'))

# Should remove any 5-sets ATP tours

dfa=dfa[dfa['TourRank']<4]



cw=len(dfw)

ca=len(dfa)

dfw.dropna(subset=['TPoints'], inplace=True)

dfa.dropna(subset=['TPoints'], inplace=True)

print('Removed NaNs about {:.0%} in WTA and {:.0%} in ATP matches.'.format((cw-len(dfw))/cw,(ca-len(dfa))/ca))
target='TPoints'

tw,ytw=calc_total(target, dfw)

ytw
ta,yta=calc_total(target, dfa)

yta
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfU', hue='TPointsR', data=tw);

gr.set(xlabel=None, ylabel='Profit, units', title='WTA. Profit of under wagers by bookies lines')

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfU', hue='TPointsR', data=ta);

gr.set(xlabel=None, ylabel='Profit, units', title='ATP. Profit of under wagers by bookies lines')

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfO', hue='TPointsR', data=tw);

gr.set(xlabel=None, ylabel='Profit, units', title='WTA. Profit of over wagers by bookies lines')

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfO', hue='TPointsR', data=ta);

gr.set(xlabel=None, ylabel='Profit, units', title='ATP. Profit of over wagers by bookies lines')

plt.show()
target='Regr'

tw,ytw=calc_total(target, dfw)

ta,yta=calc_total(target, dfa)
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfU', hue='RegrR', data=tw);

gr.set(xlabel=None, ylabel='Profit, units', title='WTA. Profit of under wagers by regressor')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfU', hue='RegrR', data=ta);

gr.set(xlabel=None, ylabel='Profit, units', title='ATP. Profit of under wagers by regressor')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfO', hue='RegrR', data=tw);

gr.set(xlabel=None, ylabel='Profit, units', title='WTA. Profit of over wagers by regressor')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfO', hue='RegrR', data=ta);

gr.set(xlabel=None, ylabel='Profit, units', title='ATP. Profit of over wagers by regressor')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
target='Diff'

tw,ytw=calc_total(target, dfw)

ta,yta=calc_total(target, dfa)
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfU', hue='DiffR', data=tw);

gr.set(xlabel=None, ylabel='Profit, units', title='WTA. Profit of under wagers by diffs')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfU', hue='DiffR', data=ta);

gr.set(xlabel=None, ylabel='Profit, units', title='ATP. Profit of under wagers by diffs')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfO', hue='DiffR', data=tw);

gr.set(xlabel=None, ylabel='Profit, units', title='WTA. Profit of over wagers by diffs')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfO', hue='DiffR', data=ta);

gr.set(xlabel=None, ylabel='Profit, units', title='ATP. Profit of over wagers by diffs')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
dfu=pd.concat([dfw[dfw['Diff']<0],dfa[dfa['Diff']<0]], axis=0, ignore_index=True)

tu,ytu=calc_total(target, dfu)

ytu
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfU', hue='DiffR', data=tu);

gr.set(xlabel=None, ylabel='Profit, units', title='Profit of under wagers by diffs')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='RoiU', hue='DiffR', data=tu);

gr.set(xlabel=None, ylabel='ROI, %', title='ROI of under wagers by diffs')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
dfu=pd.concat([dfw[dfw['Diff']<-0.5],dfa[dfa['Diff']<-0.5]], axis=0, ignore_index=True)

tu,ytu=calc_total(target, dfu)

ytu
dfu=pd.concat([dfw[dfw['Diff']<=-1],dfa[dfa['Diff']<=-1]], axis=0, ignore_index=True)

tu,ytu=calc_total(target, dfu)

ytu
dfu=pd.concat([dfw[dfw['Diff']<-0.5],dfa[dfa['Diff']<-0.5]], axis=0, ignore_index=True)

dfu.sort_values(by=['GameD'], inplace=True, ascending=True)

total=dfu.groupby(['GameD']).sum()[['PrfU','C']]

total.reset_index(inplace=True)

total['mod']=total['GameD'].dt.strftime('%m-%d')

total['wn']=total['GameD'].dt.strftime('%W')

total['Year']=total['GameD'].dt.year

dft=pd.DataFrame(np.sort(total['mod'].unique()), columns=['mod'])

dft=pd.merge(dft, total[total['Year']==2017][['mod','PrfU','C']], how='left', on=['mod'])

dft=pd.merge(dft, total[total['Year']==2018][['mod','PrfU','C']], how='left', on=['mod'], suffixes=('_2017','_2018'))

dft=pd.merge(dft, total[total['Year']==2019][['mod','PrfU','C']], how='left', on=['mod'])

dft['SUM_2017']=dft['PrfU_2017'].cumsum()

dft['SUM_2018']=dft['PrfU_2018'].cumsum()

dft['SUM_2019']=dft['PrfU'].cumsum()

dft['m']=dft['mod'].str[:2]
dfplot=dft[['m','SUM_2017','SUM_2018','SUM_2019']]

fig, ax = plt.subplots(figsize=(18,6))

gr=sns.lineplot(x='m', y='value', hue='Year', data=pd.melt(dfplot, ['m'], var_name='Year'), linewidth=2)

gr.set(xlabel=None, ylabel='Profit, units', title='Total Under. Cumulative profits for three years')

plt.show()
fig, ax = plt.subplots(figsize=(18,6))

ax=sns.barplot(x='mod', y='PrfU_2017', data=dft,color='b')

ax.set(xlabel=None, ylabel='Profit, units', title='Daily profit in 2017')

plt.show()
fig, ax = plt.subplots(figsize=(18,6))

ax=sns.barplot(x='mod', y='C_2017', data=dft,color='b')

ax.set(xlabel=None, ylabel='# of picks', title='Picks amount per day in 2017')

plt.show()
fig, ax = plt.subplots(figsize=(18,6))

ax=sns.barplot(x='mod', y='PrfU_2018', data=dft,color='b')

ax.set(xlabel=None, ylabel='Profit, units', title='Daily profit in 2018')

plt.show()
fig, ax = plt.subplots(figsize=(18,6))

ax=sns.barplot(x='mod', y='C_2018', data=dft,color='b')

ax.set(xlabel=None, ylabel='# of picks', title='Picks amount per day in 2018')

plt.show()
fig, ax = plt.subplots(figsize=(18,6))

ax=sns.barplot(x='mod', y='PrfU', data=dft,color='b')

ax.set(xlabel=None, ylabel='Profit, units', title='Daily profit in 2019')

plt.show()
fig, ax = plt.subplots(figsize=(18,6))

ax=sns.barplot(x='mod', y='C', data=dft,color='b')

ax.set(xlabel=None, ylabel='# of picks', title='Picks amount per day in 2019')

plt.show()