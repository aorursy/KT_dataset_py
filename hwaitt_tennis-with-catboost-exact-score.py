import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.dates import drange

from datetime import date,timedelta

%matplotlib inline

pd.options.mode.chained_assignment = None

sns.set(font_scale = 1.2)

sns.set_style("whitegrid")
# to2002 - 2002 threshold

# to2112 - 2112 threshold

# tod2112 - 2112 threshold on non-classified diffs

def cpk_filter(df, to2002=0,to2112=0,tod2112=0):

    

    df2002=df[(df['ClfBin']=='2002') & (df['DiffClf']>=to2002)]

    df2112=df[(df['ClfBin']=='2112') & (df['DiffClf']>=to2112)]

    dfr=df[df['PKDN']!=df['ClfSC']]

    dfr2112=dfr[(dfr['PKBin']=='2112') & (dfr['DiffPK']>=tod2112)]

    dfr2112['PrfClf']=dfr2112['PrfPK']

    dfr2112['OddsClfR']=dfr2112['OddsPKR']

    return pd.concat([df2002, df2112,dfr2112], axis=0, ignore_index=True)
bins = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,1.8,1.9,2,2.1,2.2,2.3,2.5,3,4,5,7,10,99999]

binlabels = ['<1.1', '<1.2', '<1.3', '<1.4', '<1.5', '<1.6', '<1.7','<1.8','<1.9','<2.0','<2.1','<2.2','<2.3','<2.5','<3.0','<4.0','<5.0','<7.0','<10','>10']



bins_diff = [-10, -0.4, -0.2, 0, 0.01,0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 99999]

binlabels_diff = ['<-0.4', '<-0.2', '<0', '<0.01', '<0.02', '<0.03', '<0.05', '<0.1', '<0.2', '<0.3', '<0.4', '<0.5', '<0.6', '>0.6']
def calc_res(df):

    di = {0:'2-0', 1:'0-2',2: '2-1', 3:'1-2'}

    cols_orig=['Tour','Name_1','Name_2','TourRank','RID','GameD','Year', 'SETS', 'K20', 'K02', 'K21', 'K12', 'ClfSC', 'P_2-0', 'P_0-2', 'P_2-1', 'P_1-2']

    cols=['Tour','Name_1','Name_2','TourRank','RID','GameD','Year', 'SETS', 'K2-0', 'K0-2', 'K2-1', 'K1-2', 'ClfSC', 'P_2-0', 'P_0-2', 'P_2-1', 'P_1-2']



    df=df[cols_orig]

    df.columns=cols

    df['GameD']=pd.to_datetime(df['GameD'])

    df['Month']=df['GameD'].dt.month

    df['Weekday']=df['GameD'].dt.weekday_name

    df['Week']=df['GameD'].dt.week

    df['S']=df['SETS']

    df['C']=df['ClfSC']

    df=pd.get_dummies(df, columns=['S'])

    df=pd.get_dummies(df, columns=['C'])

    df['C']=1

    df['ResClf']=np.where(df['SETS']==df['ClfSC'],1,0)

    df['ClfBin']=np.where(df['ClfSC']<2,'2002','2112')

    df['SETS']=df['SETS'].map(di)

    df['ClfSC']=df['ClfSC'].map(di)

    # Should remove low ranked tours

    df=df[df['TourRank']>0]



    # Wager on regressor prediction

    df['OddsClf']=df.apply(lambda r: 1/r[f'K{r.ClfSC}'], axis = 1)

    df['OddsClfR']=pd.cut(df['OddsClf'], bins=bins, labels=binlabels)

    df['PrfClf']=np.where(df['ResClf']==1,df['OddsClf']-1,-1)

    df['DiffClf']=df.apply(lambda r: r[f'P_{r.ClfSC}']-r[f'K{r.ClfSC}'], axis = 1)

    df['DiffClfR']=pd.cut(df['DiffClf'], bins=bins_diff, labels=binlabels_diff)



    # Wager on P-K diffs

    df['PKD2-0']=df['P_2-0']-df['K2-0']

    df['PKD0-2']=df['P_0-2']-df['K0-2']

    df['PKD2-1']=df['P_2-1']-df['K2-1']

    df['PKD1-2']=df['P_1-2']-df['K1-2']

    df['PKDN']=df.apply(lambda r: ['2-0','0-2','2-1','1-2'][np.argmax([r['PKD2-0'],r['PKD0-2'],r['PKD2-1'],r['PKD1-2']])], axis = 1)

    # Filter out already classified picks

    # df=df[df['PKDN']!=df['ClfSC']]

    df['ResPK']=np.where(df['SETS']==df['PKDN'],1,0)

    df['OddsPK']=df.apply(lambda r: 1/r[f'K{r.PKDN}'], axis = 1)

    df['OddsPKR']=pd.cut(df['OddsPK'], bins=bins, labels=binlabels)

    df['PrfPK']=np.where(df['ResPK']==1,df['OddsPK']-1,-1)

    df['DiffPK']=df.apply(lambda r: r[f'PKD{r.PKDN}'], axis = 1)

    df['DiffPKR']=pd.cut(df['DiffPK'], bins=bins_diff, labels=binlabels_diff)

    df['PKBin']=np.where((df['PKDN']=='2-0')|(df['PKDN']=='0-2'),'2002','2112')

    return df 
def calc_total(target, hue, df):

    total=df.groupby(['Year',f'{target}Bin',f'{hue}{target}R']).sum()[[f'Prf{target}',f'Res{target}','C']]

    total.reset_index(inplace=True)

    total[f'Roi{target}']=total[f'Prf{target}']/total['C']*100

    total[f'Acc{target}']=total[f'Res{target}']/total['C']*100



    ytotal=total.groupby(['Year',f'{target}Bin']).sum()[[f'Prf{target}',f'Res{target}','C']]

    ytotal[f'Roi{target}']=ytotal[f'Prf{target}']/ytotal['C']*100

    ytotal[f'Acc{target}']=ytotal[f'Res{target}']/ytotal['C']*100

    return (total, ytotal)
dfw=calc_res(pd.read_csv('../input/tennis-20112019/wta_picks.csv'))

dfa=calc_res(pd.read_csv('../input/tennis-20112019/atp_picks.csv'))

# Should remove Grand Slams in ATP

dfa=dfa[dfa['TourRank']<4]
total=dfw.groupby(['ClfBin']).sum()[['C','ResClf']]

total['Acc']=total['ResClf']/total['C']*100

total
total=dfa.groupby(['ClfBin']).sum()[['C','ResClf']]

total['Acc']=total['ResClf']/total['C']*100

total
cw=len(dfw)

dfw.dropna(subset=['K2-0','K0-2','K2-1','K1-2'], inplace=True)

ca=len(dfa)

dfa.dropna(subset=['K2-0','K0-2','K2-1','K1-2'], inplace=True)

print('Removed about {:.0%} WTA and {:.0%} ATP matches.'.format((cw-len(dfw))/cw,(ca-len(dfa))/ca))
total=dfw.groupby(['ClfBin']).sum()[['C','ResClf']]

total['Acc']=total['ResClf']/total['C']*100

total
total=dfa.groupby(['ClfBin']).sum()[['C','ResClf']]

total['Acc']=total['ResClf']/total['C']*100

total
target='Clf'

tw,ytw=calc_total(target, 'Odds', dfw)

ytw
ta,yta=calc_total(target, 'Odds', dfa)

yta
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfClf', hue='OddsClfR', data=tw[tw['ClfBin']=='2002']);

gr.set(xlabel=None, ylabel='Profit, units', title='WTA. Profit of 2002 classifier')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfClf', hue='OddsClfR', data=ta[ta['ClfBin']=='2002']);

gr.set(xlabel=None, ylabel='Profit, units', title='ATP. Profit of 2002 classifier')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfClf', hue='OddsClfR', data=tw[tw['ClfBin']=='2112']);

gr.set(xlabel=None, ylabel='Profit, units', title='WTA. Profit of 2112 classifier')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfClf', hue='OddsClfR', data=ta[ta['ClfBin']=='2112']);

gr.set(xlabel=None, ylabel='Profit, units', title='ATP. Profit of 2112 classifier')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
tw,ytw=calc_total(target, 'Diff', dfw)

ta,yta=calc_total(target, 'Diff', dfa)
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfClf', hue='DiffClfR', data=tw[tw['ClfBin']=='2002']);

gr.set(xlabel=None, ylabel='Profit, units', title='WTA. Profit of 2002 classifier')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='RoiClf', hue='DiffClfR', data=tw[tw['ClfBin']=='2002']);

gr.set(xlabel=None, ylabel='ROI, %', title='WTA. ROI of 2002 classifier')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfClf', hue='DiffClfR', data=ta[ta['ClfBin']=='2002']);

gr.set(xlabel=None, ylabel='Profit, units', title='ATP. Profit of 2002 classifier')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='RoiClf', hue='DiffClfR', data=ta[ta['ClfBin']=='2002']);

gr.set(xlabel=None, ylabel='ROI, %', title='ATP. ROI of 2002 classifier')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
print('There are {} WTA and {} ATP picks with Diff>30%, {} WTA / {} ATP - with Diff>40%'.format(len(dfw[(dfw['ClfBin']=='2002') & (dfw['DiffClf']>=0.3)]),len(dfa[(dfa['ClfBin']=='2002') & (dfa['DiffClf']>=0.3)]),len(dfw[(dfw['ClfBin']=='2002') & (dfw['DiffClf']>=0.4)]),len(dfa[(dfa['ClfBin']=='2002') & (dfa['DiffClf']>=0.4)])))
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfClf', hue='DiffClfR', data=tw[tw['ClfBin']=='2112']);

gr.set(xlabel=None, ylabel='Profit, units', title='WTA. Profit of 2112 classifier')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='RoiClf', hue='DiffClfR', data=tw[tw['ClfBin']=='2112']);

gr.set(xlabel=None, ylabel='ROI, %', title='WTA. ROI of 2112 classifier')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfClf', hue='DiffClfR', data=ta[ta['ClfBin']=='2112']);

gr.set(xlabel=None, ylabel='Profit, units', title='ATP. Profit of 2112 classifier')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='RoiClf', hue='DiffClfR', data=ta[ta['ClfBin']=='2112']);

gr.set(xlabel=None, ylabel='ROI, %', title='ATP. ROI of 2112 classifier')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
print('There are {} WTA with Diff>=5%, and {} ATP - with Diff>=20%'.format(len(dfw[(dfw['ClfBin']=='2112') & (dfw['DiffClf']>=0.05)]),len(dfa[(dfa['ClfBin']=='2112') & (dfa['DiffClf']>=0.2)])))
# Filter out already classified picks

dfrw=dfw[dfw['PKDN']!=dfw['ClfSC']]

dfra=dfa[dfa['PKDN']!=dfa['ClfSC']]
target='PK'

tw,ytw=calc_total(target, 'Odds', dfrw)

ytw
ta,yta=calc_total(target, 'Odds', dfra)

yta
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfPK', hue='OddsPKR', data=tw[tw['PKBin']=='2002']);

gr.set(xlabel=None, ylabel='Profit, units', title='WTA. Profit of 2002 on non-classified diffs')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfPK', hue='OddsPKR', data=ta[ta['PKBin']=='2002']);

gr.set(xlabel=None, ylabel='Profit, units', title='ATP. Profit of 2002 on non-classified diffs')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfPK', hue='OddsPKR', data=tw[tw['PKBin']=='2112']);

gr.set(xlabel=None, ylabel='Profit, units', title='WTA. Profit of 2112 on non-classified diffs')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfPK', hue='OddsPKR', data=ta[ta['PKBin']=='2112']);

gr.set(xlabel=None, ylabel='Profit, units', title='ATP. Profit of 2112 on non-classified diffs')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
tw,ytw=calc_total(target, 'Diff', dfrw)

ta,yta=calc_total(target, 'Diff', dfra)
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfPK', hue='DiffPKR', data=tw[tw['PKBin']=='2002']);

gr.set(xlabel=None, ylabel='Profit, units', title='WTA. Profit of 2002 on non-classified diffs')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfPK', hue='DiffPKR', data=ta[ta['PKBin']=='2002']);

gr.set(xlabel=None, ylabel='Profit, units', title='ATP. Profit of 2002 on non-classified diffs')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfPK', hue='DiffPKR', data=tw[tw['PKBin']=='2112']);

gr.set(xlabel=None, ylabel='Profit, units', title='WTA. Profit of 2112 on non-classified diffs')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='RoiPK', hue='DiffPKR', data=tw[tw['PKBin']=='2112']);

gr.set(xlabel=None, ylabel='ROI, %', title='WTA. ROI of 2112 on non-classified diffs')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfPK', hue='DiffPKR', data=ta[ta['PKBin']=='2112']);

gr.set(xlabel=None, ylabel='Profit, units', title='ATP. Profit of 2112 on non-classified diffs')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='RoiPK', hue='DiffPKR', data=ta[ta['PKBin']=='2112']);

gr.set(xlabel=None, ylabel='ROI, %', title='ATP. ROI of 2112 on non-classified diffs')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
print('Huh, we can grab some profit even here, just set threshold to 10%. There are {} such picks for WTA and {} ATP.'.format(len(dfrw[(dfrw['PKBin']=='2112') & (dfrw['DiffPK']>=0.1)]),len(dfra[(dfra['PKBin']=='2112') & (dfra['DiffPK']>=0.1)])))
def ytotal(df):

    ytotal=df.groupby(['Year']).sum()[['PrfClf','ResClf','C']]

    ytotal['RoiClf']=ytotal['PrfClf']/ytotal['C']*100

    ytotal['AccClf']=ytotal['ResClf']/ytotal['C']*100

    return ytotal
df31=pd.concat([cpk_filter(dfw, to2002=0.3,to2112=0.05,tod2112=0.1), cpk_filter(dfa, to2002=0.3,to2112=0.2,tod2112=0.1)], axis=0, ignore_index=True)

ytotal(df31)
df32=pd.concat([cpk_filter(dfw, to2002=0.3,to2112=0.05,tod2112=0.2), cpk_filter(dfa, to2002=0.3,to2112=0.2,tod2112=0.2)], axis=0, ignore_index=True)

ytotal(df32)
df41=pd.concat([cpk_filter(dfw, to2002=0.4,to2112=0.05,tod2112=0.1), cpk_filter(dfa, to2002=0.4,to2112=0.2,tod2112=0.1)], axis=0, ignore_index=True)

ytotal(df41)
df42=pd.concat([cpk_filter(dfw, to2002=0.4,to2112=0.05,tod2112=0.2), cpk_filter(dfa, to2002=0.4,to2112=0.2,tod2112=0.2)], axis=0, ignore_index=True)

ytotal(df42)
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y='PrfClf', hue='OddsClfR', data=df32);

gr.set(xlabel=None, ylabel='Profit, units', title='Profit of filtered picks')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
df32.sort_values(by=['GameD'], inplace=True, ascending=True)

total=df32.groupby(['GameD']).sum()[['PrfClf','ResClf','C']]

total.reset_index(inplace=True)

total['mod']=total['GameD'].dt.strftime('%m-%d')

total['wn']=total['GameD'].dt.strftime('%W')

total['Year']=total['GameD'].dt.year

dft=pd.DataFrame(np.sort(total['mod'].unique()), columns=['mod'])

dft=pd.merge(dft, total[total['Year']==2017][['mod','PrfClf','ResClf','C']], how='left', on=['mod'])

dft=pd.merge(dft, total[total['Year']==2018][['mod','PrfClf','ResClf','C']], how='left', on=['mod'], suffixes=('_2017','_2018'))

dft=pd.merge(dft, total[total['Year']==2019][['mod','PrfClf','ResClf','C']], how='left', on=['mod'])

dft['SUM_2017']=dft['PrfClf_2017'].cumsum()

dft['SUM_2018']=dft['PrfClf_2018'].cumsum()

dft['SUM_2019']=dft['PrfClf'].cumsum()

dft['m']=dft['mod'].str[:2]
dfplot=dft[['m','SUM_2017','SUM_2018','SUM_2019']]

fig, ax = plt.subplots(figsize=(18,6))

gr=sns.lineplot(x='m', y='value', hue='Year', data=pd.melt(dfplot, ['m'], var_name='Year'), linewidth=2)

gr.set(xlabel=None, ylabel='Profit, units', title='Cumulative profits for three years')

plt.show()
fig, ax = plt.subplots(figsize=(18,6))

ax=sns.barplot(x='mod', y='PrfClf_2017', data=dft,color='b')

ax.set(xlabel=None, ylabel='Profit, units', title='Daily profit in 2017')

plt.show()
fig, ax = plt.subplots(figsize=(18,6))

ax=sns.barplot(x='mod', y='C_2017', data=dft,color='b')

ax.set(xlabel=None, ylabel='# of picks', title='Picks amount per day in 2017')

plt.show()
fig, ax = plt.subplots(figsize=(18,6))

ax=sns.barplot(x='mod', y='PrfClf_2018', data=dft,color='b')

ax.set(xlabel=None, ylabel='Profit, units', title='Daily profit in 2018')

plt.show()
fig, ax = plt.subplots(figsize=(18,6))

ax=sns.barplot(x='mod', y='C_2018', data=dft,color='b')

ax.set(xlabel=None, ylabel='# of picks', title='Picks amount per day in 2018')

plt.show()
fig, ax = plt.subplots(figsize=(18,6))

ax=sns.barplot(x='mod', y='PrfClf', data=dft,color='b')

ax.set(xlabel=None, ylabel='Profit, units', title='Daily profit in 2019')

plt.show()
fig, ax = plt.subplots(figsize=(18,6))

ax=sns.barplot(x='mod', y='C', data=dft,color='b')

ax.set(xlabel=None, ylabel='# of picks', title='Picks amount per day in 2019')

plt.show()