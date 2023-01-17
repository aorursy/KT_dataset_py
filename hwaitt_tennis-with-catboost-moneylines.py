import pandas as pd

import numpy as np

from sklearn.preprocessing import KBinsDiscretizer

import seaborn as sns

import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None



%matplotlib inline

sns.set(style="whitegrid")
def cpk_filter(df, threshold=0,oddsmin=1,oddsmax=9999, trank=[], rid=[]):

    rdf=df[(df['PKDiff']>threshold) & (df['OddsClf']>=oddsmin) & (df['OddsClf']<=oddsmax)].copy()

    if trank:

        rdf=rdf[~rdf['TourRank'].isin(trank)]

    if rid:

        rdf=rdf[~rdf['RID'].isin(rid)]

    return rdf
def calc_res(df):

    df['C']=1



    # Wager on Favs

    df['OddsFav']=np.where(df['K1']>0.5,1/df['K1'],1/df['K2'])

    df=df[df['OddsFav']<=2]

    df['OddsFavR']=pd.cut(df['OddsFav'], bins=bins, labels=binlabels)

    df['WagFav']=np.where(df['K1']>0.5,1,0)

    df['ResFav']=np.where(df['WagFav']==df['GRes'],1,0)

    df['PrfFav']=np.where(df['ResFav']==1,df['OddsFav']-1,-1)



    # Wager on binary classifier

    df['OddsClf']=np.where(df['P1']>0.5,1/df['K1'],1/df['K2'])

    df['OddsClfR']=pd.cut(df['OddsClf'], bins=bins, labels=binlabels)

    df['WagClf']=np.where(df['P1']>0.5,1,0)

    df['ResClf']=np.where(df['WagClf']==df['GRes'],1,0)

    df['PrfClf']=np.where(df['ResClf']==1,df['OddsClf']-1,-1)



    # Wager on P-K difference 

    df['PKDiff1']=(df['P1']-df['K1'])*100

    df['PKDiff2']=(df['P2']-df['K2'])*100

    df['PKDiff']=np.where(df['PKDiff1']>df['PKDiff2'],df['PKDiff1'],df['PKDiff2'])



    df['OddsDiff']=np.where(df['PKDiff1']>df['PKDiff2'],1/df['K1'],1/df['K2'])

    df['OddsDiffR']=pd.cut(df['OddsDiff'], bins=bins, labels=binlabels)

    df['WagDiff']=np.where(df['PKDiff1']>df['PKDiff2'],1,0)

    df['ResDiff']=np.where(df['WagDiff']==df['GRes'],1,0)

    df['PrfDiff']=np.where(df['ResDiff']==1,df['OddsDiff']-1,-1)



    kbd = KBinsDiscretizer(n_bins=25, encode='ordinal', strategy='quantile')

    df['PKDiffKBD'] = kbd.fit_transform(df[['PKDiff']])

    labels = { k: round(v,2) for k, v in enumerate(kbd.bin_edges_[0][1:]) }

    df['PKDiffR']=df['PKDiffKBD'].map(labels)

    return df 
def calc_total(target, hue, df):

    total=df.groupby(['Year',f'{hue}{target}R']).sum()[[f'Prf{target}',f'Res{target}','C']]

    total.reset_index(inplace=True)

    total[f'Roi{target}']=total[f'Prf{target}']/total['C']*100

    total[f'Acc{target}']=total[f'Res{target}']/total['C']*100



    ytotal=total.groupby(['Year']).sum()[[f'Prf{target}',f'Res{target}','C']]

    ytotal[f'Roi{target}']=ytotal[f'Prf{target}']/ytotal['C']*100

    ytotal[f'Acc{target}']=ytotal[f'Res{target}']/ytotal['C']*100

    return (total, ytotal)
bins = [1, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,1.8,1.9,2,2.1,2.2,2.3,2.5,3,5,10,99999]

binlabels = ['<1.1', '<1.15', '<1.2', '<1.3', '<1.4', '<1.5', '<1.6', '<1.7','<1.8','<1.9','<2.0','<2.1','<2.2','<2.3','<2.5','<3.0','<5.0','<10','>10']
dfa=pd.read_csv('../input/tennis-20112019/atp_picks.csv')[['TourRank', 'RID', 'GameD', 'Year', 'GRes', 'ClfML', 'K1', 'K2', 'P2', 'P1']]

dfa['C']=1

dfa['ResClf']=np.where(dfa['GRes']==dfa['ClfML'],1,0)

dfa['GameD']=pd.to_datetime(dfa['GameD'])

dfa['Month']=dfa['GameD'].dt.month

dfa['Weekday']=dfa['GameD'].dt.weekday_name

dfa['Week']=dfa['GameD'].dt.week

dfa.sort_values(by=['GameD'], inplace=True, ascending=True)



dfw=pd.read_csv('../input/tennis-20112019/wta_picks.csv')[['TourRank', 'RID', 'GameD', 'Year', 'GRes', 'ClfML', 'K1', 'K2', 'P2', 'P1']]

dfw['C']=1

dfw['ResClf']=np.where(dfw['GRes']==dfw['ClfML'],1,0)

dfw['GameD']=pd.to_datetime(dfw['GameD'])

dfw['Month']=dfw['GameD'].dt.month

dfw['Weekday']=dfw['GameD'].dt.weekday_name

dfw['Week']=dfw['GameD'].dt.week

dfw.sort_values(by=['GameD'], inplace=True, ascending=True)



print('Out clasifier makes {:.1%} accuracy for wta and {:.1%} for atp matches. Let\'s remove matches without odds.'.format(dfw['ResClf'].mean(),dfa['ResClf'].mean()))
cw=len(dfw)

dfw.dropna(subset=['K1', 'K2'], inplace=True)

ca=len(dfa)

dfa.dropna(subset=['K1', 'K2'], inplace=True)

print('Removed about {:.0%} WTA and {:.0%} ATP matches. Accuracy now is {:.1%} WTA and {:.1%} ATP'.format((cw-len(dfw))/cw,(ca-len(dfa))/ca,dfw['ResClf'].mean(),dfa['ResClf'].mean()))
dfrw=calc_res(dfw.copy())

dfra=calc_res(dfa.copy())
target='Fav'

tw,ytw=calc_total(target, 'Odds', dfrw)

ytw
ta,yta=calc_total(target, 'Odds', dfra)

yta
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y=f'Acc{target}', hue=f'Odds{target}R', data=tw)

gr.set(xlabel=None, ylabel='Accuracy, %', title='WTA. Accuracy of bookies favs ')

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y=f'Acc{target}', hue=f'Odds{target}R', data=ta)

gr.set(xlabel=None, ylabel='Accuracy, %', title='ATP. Accuracy of bookies favs')

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y=f'Roi{target}', hue=f'Odds{target}R', data=tw);

gr.set(xlabel=None, ylabel='ROI, %', title='WTA. ROI of bookies favs wagers')

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y=f'Roi{target}', hue=f'Odds{target}R', data=ta);

gr.set(xlabel=None, ylabel='ROI, %', title='ATP. ROI of bookies favs wagers')

plt.show()
target='Clf'

tw,ytw=calc_total(target,'Odds', dfrw)

ytw
ta,yta=calc_total(target,'Odds', dfra)

yta
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y=f'Acc{target}', hue=f'Odds{target}R', data=tw)

gr.set(xlabel=None, ylabel='Accuracy, %', title='WTA. Classification accuracy')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y=f'Acc{target}', hue=f'Odds{target}R', data=ta)

gr.set(xlabel=None, ylabel='Accuracy, %', title='ATP. Classification accuracy')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y=f'Prf{target}', hue=f'Odds{target}R', data=tw);

gr.set(xlabel=None, ylabel='Profit, units', title='WTA. Classification profit')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y=f'Prf{target}', hue=f'Odds{target}R', data=ta);

gr.set(xlabel=None, ylabel='Profit, units', title='ATP. Classification profit')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
target='Diff'

tw,ytw=calc_total(target,'PK', dfrw)

ytw
ta,yta=calc_total(target,'PK', dfra)

yta
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y=f'Acc{target}', hue='PKDiffR', data=tw)

gr.set(xlabel=None, ylabel='Accuracy, %', title='WTA. P-K difference accuracy')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y=f'Acc{target}', hue=f'PKDiffR', data=ta)

gr.set(xlabel=None, ylabel='Accuracy, %', title='ATP. P-K difference accuracy')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y=f'Prf{target}', hue=f'PKDiffR', data=tw);

gr.set(xlabel=None, ylabel='Profit, units', title='WTA. P-K difference profit')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y=f'Prf{target}', hue=f'PKDiffR', data=ta);

gr.set(xlabel=None, ylabel='Profit, units', title='ATP. P-K difference profit')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
target='Clf'

tw,ytw=calc_total(target,'Odds', cpk_filter(dfrw,threshold=30))

ytw
ta,yta=calc_total(target,'Odds', cpk_filter(dfra,threshold=30))

yta
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y=f'Prf{target}', hue=f'Odds{target}R', data=tw);

gr.set(xlabel=None, ylabel='Profit, units', title='WTA. Profit of classied picks with confidence>=30%')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y=f'Roi{target}', hue=f'Odds{target}R', data=tw);

gr.set(xlabel=None, ylabel='ROI, units', title='WTA. ROI of classied picks with confidence>=30%')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y=f'Prf{target}', hue=f'Odds{target}R', data=ta);

gr.set(xlabel=None, ylabel='Profit, units', title='ATP. Profit of classied picks with confidence>=30%')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
fig, ax = plt.subplots(figsize=(15,6))

gr=sns.barplot(x='Year', y=f'Roi{target}', hue=f'Odds{target}R', data=ta);

gr.set(xlabel=None, ylabel='ROI, units', title='ATP. ROI of classied picks with confidence>=30%')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
tw,ytw=calc_total(target,'Odds', cpk_filter(dfrw,threshold=30,oddsmin=1.2))

ytw
tw,ytw=calc_total(target,'Odds', cpk_filter(dfra,threshold=30,oddsmin=1.2))

ytw
tw,ytw=calc_total(target,'Odds', cpk_filter(dfrw,threshold=40,oddsmin=1.2))

ytw
ta,yta=calc_total(target,'Odds', cpk_filter(dfra,threshold=40,oddsmin=1.2))

yta
total=cpk_filter(dfrw,threshold=40,oddsmin=1.2).groupby(['Year','Week']).sum()[[f'Prf{target}',f'Res{target}','C']]

total.reset_index(inplace=True)

total[f'Roi{target}']=total[f'Prf{target}']/total['C']*100

total[f'Acc{target}']=total[f'Res{target}']/total['C']*100



fig, ax = plt.subplots(figsize=(20,10))

gr=sns.barplot(x='Year', y=f'Prf{target}', hue='Week', data=total)

gr.set(xlabel=None, ylabel='Profit, units', title='WTA. Classification profit by Week Number')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
total=cpk_filter(dfra,threshold=40,oddsmin=1.2).groupby(['Year','Week']).sum()[[f'Prf{target}',f'Res{target}','C']]

total.reset_index(inplace=True)

total[f'Roi{target}']=total[f'Prf{target}']/total['C']*100

total[f'Acc{target}']=total[f'Res{target}']/total['C']*100



fig, ax = plt.subplots(figsize=(20,10))

gr=sns.barplot(x='Year', y=f'Prf{target}', hue='Week', data=total)

gr.set(xlabel=None, ylabel='Profit, units', title='ATP. Classification profit by Week Number')

plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)

plt.show()
labels = ['ITF up to $15K','ITF > $15K','International','Premier','Grand Slams','Fed Cup','Others']



total=cpk_filter(dfrw,threshold=40,oddsmin=1.3).groupby(['Year','TourRank']).sum()[[f'Prf{target}',f'Res{target}','C']]

total.reset_index(inplace=True)

total[f'Roi{target}']=total[f'Prf{target}']/total['C']*100

total[f'Acc{target}']=total[f'Res{target}']/total['C']*100



fig, ax = plt.subplots(figsize=(15,6))

ax=sns.barplot(x='Year', y=f'Roi{target}', hue='TourRank', data=total)

ax.set(xlabel=None, ylabel='ROI, %', title='WTA. ROI by Tour Rank')

h, l = ax.get_legend_handles_labels()

ax.legend(h, labels)

plt.show()
labels = ['Futures','Challengers','World Tour','Masters','Grand Slams','Others']



total=cpk_filter(dfra,threshold=40,oddsmin=1.3).groupby(['Year','TourRank']).sum()[[f'Prf{target}',f'Res{target}','C']]

total.reset_index(inplace=True)

total[f'Roi{target}']=total[f'Prf{target}']/total['C']*100

total[f'Acc{target}']=total[f'Res{target}']/total['C']*100



fig, ax = plt.subplots(figsize=(15,6))

ax=sns.barplot(x='Year', y=f'Roi{target}', hue='TourRank', data=total)

ax.set(xlabel=None, ylabel='ROI, %', title='ATP. ROI by Tour Rank')

h, l = ax.get_legend_handles_labels()

ax.legend(h, labels)

plt.show()
tw,ytw=calc_total(target,'Odds', cpk_filter(dfrw,threshold=30,oddsmin=1.2, trank=[4,5]))

ytw
tw,ytw=calc_total(target,'Odds', cpk_filter(dfra,threshold=30,oddsmin=1.2, trank=[4,6]))

ytw
tw,ytw=calc_total(target,'Odds', cpk_filter(dfrw,threshold=40,oddsmin=1.2, trank=[4,5]))

ytw
ta,yta=calc_total(target,'Odds', cpk_filter(dfra,threshold=40,oddsmin=1.2, trank=[4,6]))

yta
df=pd.concat([cpk_filter(dfrw,threshold=40,oddsmin=1.2, trank=[4,5]),cpk_filter(dfra,threshold=40,oddsmin=1.2, trank=[4,6])])

df.sort_values(by=['GameD'], inplace=True, ascending=True)

ta,yta=calc_total(target,'Odds', df)

yta
total=df.groupby(['GameD']).sum()[[f'Prf{target}',f'Res{target}','C']]

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