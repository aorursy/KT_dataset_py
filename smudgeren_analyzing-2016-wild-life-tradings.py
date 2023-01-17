%matplotlib inline
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import matplotlib as mlp
import matplotlib.pyplot as plt
from matplotlib import cm
mlp.rcParams['font.size'] = 12
raw_data = pd.read_csv('../input/comptab_2018-01-29 16_00_comma_separated.csv')
raw_data.drop(raw_data[raw_data['Year']==2017].index,inplace=True)
raw_data.head(3)
def plot_labels_values(series):
    labels = [idx for idx in series.index]
    values = [value for value in series.values]
    index = [x for x in range(len(series))]
    return labels,values,index
# total kinds of taxons traded by CITES Appendix(in which Appendix I is the most endangered)
taxon_trade_records = raw_data.groupby(['App.','Taxon'])['Taxon'].value_counts()
app_count = taxon_trade_records.groupby(level=0).count()
app_count
fig,ax = plt.subplots(figsize=[6,6])
labels,values,index = plot_labels_values(app_count)
comments = '6400 kinds of Taxons were engaged in trading,\n of which 8.2% are Appendix I (threatened with extinction)'
explode = [0.2,0,0,0]
ax.pie(values, labels = labels,explode=explode,labeldistance=1.05)
ax.axis('equal')
ax.set_title('Total kinds of animals traded')
ax.text(1.5,0.5,comments,wrap = True)
plt.show()
# total trades by taxons(import or export records)
top_traded_taxon = taxon_trade_records.groupby(level=1).sum().sort_values(ascending = False).head(10)
top_traded_taxon
fig,ax = plt.subplots()
labels,values,index = plot_labels_values(top_traded_taxon.sort_values())
ax.barh(index,values)
plt.yticks(index,labels)
ax.set_title('Top traded Taxons')
comments = 'Top 3 traded species are:\n'\
            '\n'\
            'Nile crocdile, 2410 trades recorded,\n'\
            'Python reticulatus, 2304 trades recorded,\n'\
            'American alligator, 2081 trades recorded.'
ax.text(3000,5,comments)
plt.show()
countries = raw_data.groupby(['Exporter','Importer'])['Taxon'].value_counts()
exporter = countries.groupby(level=0).sum()
importer = countries.groupby(level=1).sum()

country_records = pd.concat([exporter,importer],axis=1).fillna(0)
country_records.columns = ['Import','Export']
country_records['Total_trades'] = country_records['Export'] + country_records['Import']
country_records.sort_values(by='Total_trades',ascending=False).head(10)
purpose_code = {'B':'Breeding in captivity or artificial propagation',
                 'E':'Educational',
                 'G':'Botanical garden',
                 'H':'Hunting trophy',
                 'L':'Law enforcement / judicial / forensic',
                 'M':'Medical (including biomedical research)',
                 'N':'Reintroduction or introduction into the wild',
                 'P':'Personal',
                 'Q':'Circus or travelling exhibition',
                 'S':'Scientific',
                 'T':'Commercial',
                 'Z':'Zoo'}
trading_purposes = raw_data['Purpose'].value_counts()
trading_purposes.rename(index=purpose_code,inplace=True)
fig,ax = plt.subplots(figsize=[10,6])
labels,values,index = plot_labels_values(trading_purposes.sort_values())
colors = cm.rainbow(np.arange(len(index))/len(index))
explode = [0]*(len(index)-3) + [0.1]*3
comments = 'Approximately 3/4 of tradings are for commercial use.'
patches,text = ax.pie(values,colors=colors,explode=explode,startangle=300)
ax.set_title('Purpose of trading')
ax.axis('equal')
ax.legend(patches[-3:],labels[-3:])
ax.text(1.5,0,comments)
plt.show
ae_trades = raw_data[raw_data['Taxon'] == 'Loxodonta africana']
terms_of_trading = ae_trades.groupby('Term')['Importer reported quantity'].count().sort_values(ascending = False)
terms_of_trading.sum()
terms_of_trading[:3]
fig,ax = plt.subplots(figsize=[8,10])
labels,values,index = plot_labels_values(terms_of_trading.sort_values())
ax.barh(index,values)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
comments = 'Most African elephant are traded in terms of ivory carvings, tusks, and trophies'
ax.text(60,15,comments,wrap = True)
plt.ylim(min(index)-0.5, max(index)+0.5)
plt.yticks(index,labels)
plt.show()
importer_terms = ae_trades.loc[ae_trades['Term'].isin(['ivory carvings', 'tusks', 'trophies'])]\
                .groupby(['Importer','Term'])['Importer reported quantity'].sum().unstack()
top_ivory_importer = importer_terms.sort_values(by='ivory carvings',ascending = False).head(5).fillna(0)
top_ivory_importer