import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)
por = pd.read_csv('../input/por-for-kiva-partners/por_for_kiva_partners.csv',usecols=['partner_id','name','portfolio_yield','total_amount_raised','countries','score','average_mpi','POR_norm'])
por = por[por.portfolio_yield>0]
por.index=range(1,por.shape[0]+1)
por[['name','countries','POR_norm']][:10]
por.index=range(1,por.shape[0]+1)
por[['name','countries','POR_norm']][-10:]
set(por.countries.unique())
sns.distplot(por.POR_norm, bins=15,kde=False, rug=True);
plt.ylabel('Number of MFIs with speciffic PoR');
plt.xlabel('Normalized PoR score');
tmp = por.copy()
tmp.columns = ['partner_id', 'name', 'Portfolio Yield', 'Total Amount Raised in USD million',
       'countries', 'score', "Average MPI in MFI's country", 'Normalized PoR']
sns.jointplot(x=tmp["Average MPI in MFI's country"],y=tmp['Normalized PoR']);
gmos = pd.read_csv('../input/gmos-data/gmos.csv')
gmos_score = []
tmp['GMOS score for country'] = 0.
for i in tmp.index:
    tmp.loc[i,'GMOS score for country'] = gmos[gmos['Overall score']==tmp.loc[i,'countries']]['2016'].mean()
    gmos_score.append(gmos[gmos['Overall score']==tmp.loc[i,'countries']]['2016'].mean())
sns.jointplot(x=tmp['GMOS score for country'], y=tmp['Normalized PoR']);
print(tmp['GMOS score for country'].mean())
print(tmp['GMOS score for country'].median())
tmp.columns
sns.jointplot(x=tmp['Portfolio Yield'], y=tmp['Normalized PoR']);
sns.jointplot(x=tmp['Total Amount Raised in USD million']/1000000., y=tmp['Normalized PoR']);
por.POR_norm.mean()
chart = sns.barplot(x=por.POR_norm,y=por.countries);
plt.plot([por.POR_norm.mean(),por.POR_norm.mean()],[-2,30]);
plt.ylabel('Countries');
plt.xlabel('Normalized PoR');