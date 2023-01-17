import pandas as pd

import matplotlib.pyplot as plt



GDP = pd.read_csv(

    "../input/LV_real_GDP.csv",

    index_col = 0,

    header = 1

    )

GDP = GDP.transpose()

GDP.index = pd.to_datetime(GDP.index)



g = GDP.plot.bar()

g.set_xticklabels(GDP.index.strftime('%Y'));

g.set_yticks([0,GDP.loc[:,'GDP'].min(), GDP.loc[:,'GDP'].max()]);

g.set_yticklabels([0,round(GDP.loc[:,'GDP'].min()/1000000, 2), round(GDP.loc[:,'GDP'].max()/1000000,2)]);

g.set_ylabel('Real GDP, bEUR');
TRSM = pd.read_csv(

    "../input/LV_tourism_value.csv",

    index_col = 0,

    header = 1

    )



TRSM = TRSM.transpose()

TRSM.index = pd.to_datetime(TRSM.index)



t = TRSM.plot.bar()

t.set_xticklabels(TRSM.index.strftime('%Y'));

t.set_yticks([0,TRSM.loc[:,'Tourism'].min(), TRSM.loc[:,'Tourism'].max()])

t.set_yticklabels([0,round(TRSM.loc[:,'Tourism'].min()/1000000,2), round(TRSM.loc[:,'Tourism'].max()/1000000,2)]);

t.set_ylabel('Tourism contribution to GDP, bEUR');
TRSMvsGDP = GDP.join(TRSM, how = 'left')

TRSMvsGDP['GDP2'] = TRSMvsGDP['GDP'] - TRSMvsGDP['Tourism']



tg = TRSMvsGDP.iloc[:,[2,1]].plot.bar(stacked = True)

tg.set_xticklabels(TRSMvsGDP.index.strftime('%Y'));

tg.set_yticks([0,TRSMvsGDP.loc[:,'GDP'].min(), TRSMvsGDP.loc[:,'GDP'].max()])

tg.set_yticklabels([0,round(TRSMvsGDP.loc[:,'GDP'].min()/1000000, 2), round(TRSMvsGDP.loc[:,'GDP'].max()/1000000, 2)])

tg.set_ylabel('Tourism contribution to GDP, bEUR');

tg.legend(['Total GDP','Tourism']);
TRSMvsGDP['TShare'] = TRSMvsGDP['Tourism'] * 100 / TRSMvsGDP['GDP']

tgp = TRSMvsGDP.plot(y=3)

tgp.set_xticks(TRSMvsGDP.index)

tgp.set_xticklabels(TRSMvsGDP.index.strftime('%Y'), rotation=90);

tgp.set_yticks([0,TRSMvsGDP.loc[:,'TShare'].min(), TRSMvsGDP.loc[:,'TShare'].max()])

tgp.set_yticklabels([0,round(TRSMvsGDP.loc[:,'TShare'].min(), 2), round(TRSMvsGDP.loc[:,'TShare'].max(),2)])

tgp.set_ylim(0, 2)

tgp.set_ylabel('Tourism contribution to GDP, % of total GDP');

tgp.legend(['Tourism share in GDP']);
VISIT = pd.read_csv(

    "../input/LV_visitors_by_country.csv",

    index_col = 0,

    header = 1,

    skiprows = (3,5)

    )



VISIT = VISIT.transpose()

VISIT.index = pd.to_datetime(VISIT.index)

#VISIT.sort_values(by = pd.to_datetime('2017'), axis = 1, ascending = False, inplace = True)

TOP_VISIT = VISIT.loc[:,VISIT.loc[pd.to_datetime('2017'), :] > 50000]



v = TOP_VISIT.plot()

v.set_xticks(VISIT.index)

v.set_xticklabels(VISIT.index.strftime('%Y'), rotation=90);

v.set_yticklabels(map(lambda l: round(int(l)/1000, 2), v.get_yticks()))

v.set_ylabel('Number of visitors, k');

v.legend(bbox_to_anchor=(1.05, 1), loc=2);

SPEND = pd.read_csv(

    "../input/LV_visitor_spending.csv",

    index_col = 0,

    header = 1

    )

SPEND.index = pd.to_datetime(SPEND.index, format = '%Y')

SPEND_FILT = SPEND.loc[SPEND.iloc[:,0] != 'Total',:]



LV = pd.read_csv(

    "../input/LV_domestic_travelers.csv",

    header = 1

    )



LV_NEW = pd.DataFrame(columns = ['Year','Country','Total expenditure in Latvia (mln euro)','Avg daily expenditure per traveller (euro)','Avg stay (nights)'])

LV_NEW.set_index('Year', inplace = True)

for i in range (1, len(LV.iloc[0,:])):

    idx, col = LV.columns[i].split(maxsplit = 1)

    val = LV.iloc[0,i]

    LV_NEW.loc[pd.to_datetime(idx, format = '%Y'),col] = val



LV_NEW.loc[:,'Country'] = 'Latvian residents'

SPEND_ALL = SPEND_FILT.append(LV_NEW)



_, top_countries  = v.get_legend_handles_labels()

T_EXP = SPEND_ALL.iloc[:,[0,1]].pivot(columns = SPEND_ALL.columns[0], values = SPEND_ALL.columns[1]);



t_exp = T_EXP.loc[:,top_countries].plot()

t_exp.legend(bbox_to_anchor=(1.05, 1), loc=2);

t_exp.set_ylabel('Total expenditures, mEUR');
t_exp_2 = T_EXP.loc[:,T_EXP.loc[pd.to_datetime('2017'),:] > 10].plot()

t_exp_2.legend(bbox_to_anchor=(1.05, 1), loc=2);

t_exp_2.set_ylabel('Total expenditures, mEUR');
D_EXP = SPEND_ALL.iloc[:,[0,2]].pivot(columns = SPEND_ALL.columns[0], values = SPEND_ALL.columns[2]);

d_exp = D_EXP.loc[:,T_EXP.loc[pd.to_datetime('2017'),:] > 10].plot()

d_exp.legend(bbox_to_anchor=(1.05, 1), loc=2);

d_exp.set_ylabel('Avg daily expenditures, EUR');

S_EXP = SPEND_ALL.iloc[:,[0,3]].pivot(columns = SPEND_ALL.columns[0], values = SPEND_ALL.columns[3]);

s_exp = S_EXP.loc[:,S_EXP.loc[pd.to_datetime('2011'),:] > 4.2].plot()

s_exp.legend(bbox_to_anchor=(1.05, 1), loc=2);

s_exp.set_ylabel('Avg length of stay, nights');
S_EXP = SPEND_ALL.iloc[:,[0,3]].pivot(columns = SPEND_ALL.columns[0], values = SPEND_ALL.columns[3]);

s_exp = S_EXP.loc[:,top_countries].plot()

s_exp.legend(bbox_to_anchor=(1.05, 1), loc=2);

s_exp.set_ylabel('Avg length of stay, nights');
#eur = plt.bar(TRSMvsGDP.iloc[:,[0,1]], bottom = TRSMvsGDP.iloc[:,0])

#gdp_eur = plt.gca()

gdp_eur = TRSMvsGDP.iloc[:,[2,1]].plot.bar(stacked = True)

gdp_eur.set_ylabel('GDP, bEUR');

gdp_eur.set_ylim(0, 25000000);

gdp_eur.set_yticks([0,TRSMvsGDP.loc[:,'GDP'].min(), TRSMvsGDP.loc[:,'GDP'].max()])

gdp_eur.set_yticklabels([0,round(TRSMvsGDP.loc[:,'GDP'].min()/1000000, 2), round(TRSMvsGDP.loc[:,'GDP'].max()/1000000, 2)])

gdp_eur.set_title('Tourism contribution to GDP');

gdp_eur.set_xticklabels(TRSMvsGDP.index.strftime('%Y'));



gdp_prc = gdp_eur.twinx()

prc = gdp_prc.plot(TRSMvsGDP['TShare'].values, color = 'darkorange')

gdp_prc.set_yticks([0,TRSMvsGDP.loc[:,'TShare'].min(), TRSMvsGDP.loc[:,'TShare'].max()]);

gdp_prc.set_yticklabels([0,round(TRSMvsGDP.loc[:,'TShare'].min(), 2), round(TRSMvsGDP.loc[:,'TShare'].max(),2)]);

gdp_prc.set_ylim(0, 2);

gdp_prc.set_ylabel('% of total GDP');



h1, l1 = gdp_eur.get_legend_handles_labels()

gdp_eur.legend(h1+prc, 

               ['Total GDP','Tourism share in GDP',

                'Tourism share in GDP (prc)'], 

               bbox_to_anchor=(0.8, 1.1), 

               loc=4);

p1 = plt.bar(TRSMvsGDP.index, TRSMvsGDP.iloc[:,2], width = 200, label = 'Bar1')

p2 = plt.bar(TRSMvsGDP.index, TRSMvsGDP.iloc[:,1], bottom=0, width = 200, label = 'Bar2')



plt.gca().set_xticks(TRSMvsGDP.index);

plt.gca().set_xticklabels(TRSMvsGDP.index.strftime('%Y'), rotation = 90);

plt.gca().set_ylim(0, 25000000);

plt.gca().set_yticks([0,TRSMvsGDP.loc[:,'GDP'].min(), TRSMvsGDP.loc[:,'GDP'].max()]);

plt.gca().set_yticklabels([0,round(TRSMvsGDP.loc[:,'GDP'].min()/1000000, 2), round(TRSMvsGDP.loc[:,'GDP'].max()/1000000, 2)]);

plt.gca().set_ylabel('GDP, bEUR')



h1, l1 = plt.gca().get_legend_handles_labels()



ax2 = plt.gca().twinx()

ax2.set_ylim(0, 2);

ax2.set_yticks([0,TRSMvsGDP.loc[:,'TShare'].min(), TRSMvsGDP.loc[:,'TShare'].max()]);

ax2.set_yticklabels([0,round(TRSMvsGDP.loc[:,'TShare'].min(), 2), round(TRSMvsGDP.loc[:,'TShare'].max(),2)]);



prc = plt.plot(TRSMvsGDP['TShare'], color = 'darkorange')

ax2.set_xticks(TRSMvsGDP.index);

ax2.set_ylabel('% of total GDP');

ax2.set_title('Tourism contribution to GDP');



plt.legend(h1+prc, 

           ['Total GDP','Tourism share in GDP', 'Tourism share in GDP (prc)'], 

           bbox_to_anchor=(0.8, 1.1), 

           loc=4);



p1 = plt.plot(TRSMvsGDP.index, TRSMvsGDP.iloc[:,2], label = 'Bar1')

p2 = plt.plot(TRSMvsGDP.index, TRSMvsGDP.iloc[:,1], label = 'Bar2')



plt.gca().set_xticks(TRSMvsGDP.index);

plt.gca().set_xticklabels(TRSMvsGDP.index.strftime('%Y'), rotation = 90);

plt.gca().set_ylim(0, 25000000);

plt.gca().set_yticks([0,TRSMvsGDP.loc[:,'GDP'].min(), TRSMvsGDP.loc[:,'GDP'].max()]);

plt.gca().set_yticklabels([0,round(TRSMvsGDP.loc[:,'GDP'].min()/1000000, 2), round(TRSMvsGDP.loc[:,'GDP'].max()/1000000, 2)]);

plt.gca().set_ylabel('GDP, bEUR')



h1, l1 = plt.gca().get_legend_handles_labels()



ax2 = plt.gca().twinx()

ax2.set_ylim(0, 2);

ax2.set_yticks([0,TRSMvsGDP.loc[:,'TShare'].min(), TRSMvsGDP.loc[:,'TShare'].max()]);

ax2.set_yticklabels([0,round(TRSMvsGDP.loc[:,'TShare'].min(), 2), round(TRSMvsGDP.loc[:,'TShare'].max(),2)]);



prc = plt.plot(TRSMvsGDP['TShare'], color = 'red')

ax2.set_xticks(TRSMvsGDP.index);

ax2.set_ylabel('% of total GDP');

ax2.set_title('Tourism contribution to GDP');



plt.legend(h1+prc, 

           ['Total GDP','Tourism share in GDP', 'Tourism share in GDP (prc)'], 

           bbox_to_anchor=(0.8, 1.1), 

           loc=4);
import seaborn as sns

sns.set()

#sns.set_style("whitegrid")



ax1 = plt.subplot(311)

h1 = ax1.plot(TRSMvsGDP.index, TRSMvsGDP.loc[:,'GDP'])

ax1.set_ylim(0, 25000000);

ax1.set_yticks([0,TRSMvsGDP.loc[:,'GDP'].min(), TRSMvsGDP.loc[:,'GDP'].max()]);

ax1.set_yticklabels(['0',

                     str(round(TRSMvsGDP.loc[:,'GDP'].min()/1000000, 2)) + ' bEUR', 

                     str(round(TRSMvsGDP.loc[:,'GDP'].max()/1000000, 2)) + ' bEUR']);

ax1.tick_params(

    axis='x',          

    labelbottom=False)

ax1.legend(h1, ['Total GDP'], loc=4);

ax1.set_title('Tourism added value to Latvian GDP')





ax2 = plt.subplot(312, sharex=ax1)

h2 = ax2.plot(TRSMvsGDP.index, TRSMvsGDP.iloc[:,1], label = 'Bar2')

ax2.set_ylim(0, 450000);

ax2.set_yticks([0,TRSMvsGDP.loc[:,'Tourism'].min(), TRSMvsGDP.loc[:,'Tourism'].max()]);

ax2.set_yticklabels(['0',

                     str(round(TRSM.loc[:,'Tourism'].min()/1000000,2))+ ' bEUR', 

                     str(round(TRSM.loc[:,'Tourism'].max()/1000000,2)) + ' bEUR']);

ax2.tick_params(

    axis='x',          

    labelbottom=False)

ax2.legend(h2, ['Tourism contribution to GDP'], loc=4);



ax3 = plt.subplot(313, sharex=ax1)

h3 = ax3.plot(TRSMvsGDP['TShare'])

ax3.set_xticks(TRSMvsGDP.index)

ax3.set_xticklabels(TRSMvsGDP.index.strftime('%Y'), rotation=90);

ax3.set_yticks([0,TRSMvsGDP.loc[:,'TShare'].min(), TRSMvsGDP.loc[:,'TShare'].max()])

ax3.set_yticklabels(['0',

                     str(round(TRSMvsGDP.loc[:,'TShare'].min(), 2))+ ' %', 

                     str(round(TRSMvsGDP.loc[:,'TShare'].max(), 2))+ ' %'])

ax3.set_ylim(0, 2)

ax3.legend(h3, ['Tourism share of GDP'], loc=4);



plt.subplots_adjust(top = 1.5, hspace = 0.1, right = 1.1)



sns.set_style("darkgrid", {"axes.facecolor": ".94"})



ax1 = plt.subplot()

v1 = ax1.plot(TOP_VISIT)

ax1.set_xticks(TOP_VISIT.index)

ax1.set_xticklabels(TOP_VISIT.index.strftime('%Y'), rotation=90)

ax1.set_ylim(0, 900000)

ax1.set_yticklabels(map(lambda l: str(round(int(l)/1000, 2)) + ' k', ax1.get_yticks()));

ax1.legend(v1,TOP_VISIT.columns, bbox_to_anchor=(1.05, 1), loc=2);

ax1.set_title('Number of vistors (travelers) to Latvia');



plt.subplots_adjust(top = 1.5, right = 1.1);



import numpy as np

sns.set_style("darkgrid", {"axes.facecolor": ".94"})

#sns.set_style("whitegrid", {'grid.color': '.9'})



ax1 = plt.subplot(411)

v1 = ax1.plot(TOP_VISIT.loc[T_EXP.index,:])

ax1.set_ylim(0, 900000)

ax1.set_yticks(np.arange(0,1000000,150000))

ax1.set_yticklabels(map(lambda l: str(round(int(l)/1000, 2)) + ' k', ax1.get_yticks()));

ax1.tick_params(

    axis='x',

    labeltop=True,

    labelbottom=False)

ax1.set_title('Number of vistors (travelers) to Latvia (thousands)', y=1.1);

ax1.legend(v1,top_countries, bbox_to_anchor=(1.05, 1), loc=2);



ax2 = plt.subplot(412, sharex=ax1)

v2 = ax2.plot(T_EXP.loc[:,top_countries])

ax2.set_ylim(0,140)

ax2.set_yticks(np.arange(0, 150, 20))

ax2.set_yticklabels(map(lambda l: str(l) + ' m', ax2.get_yticks()))

ax2.tick_params(

    axis='x',          

    labelbottom=False)

ax2.set_title('Total expenditures (mEUR)');

ax2.legend(v2,top_countries, bbox_to_anchor=(1.05, 1), loc=2);



ax3 = plt.subplot(413, sharex=ax1)

v3 = ax3.plot(D_EXP.loc[:,top_countries])

ax2.set_ylim(0,140)

ax3.set_yticks(np.arange(0, 150, 20))

#ax3.set_yticklabels(map(lambda l: str(l) + ' EUR', ax3.get_yticks()))

ax3.tick_params(

    axis='x',          

    labelbottom=False)

ax3.set_title('Avg daily expenditures (EUR)');

ax3.legend(v3,top_countries, bbox_to_anchor=(1.05, 1), loc=2);



ax4 = plt.subplot(414, sharex=ax1)

v4 = ax4.plot(S_EXP.loc[:,top_countries])

ax4.set_title('Avg length of stay (nights)');

ax4.legend(v4, top_countries, bbox_to_anchor=(1.05, 1), loc=2);



plt.subplots_adjust(top = 3, hspace = 0.15, right = 1);

plt.suptitle('Visitor (traveler) stays and spending in Latvia \nTOP 10 countries by visitor numbers',

            y = 3.3, x=0.55);



from IPython.display import IFrame

IFrame('https://public.tableau.com/views/TourismInLatvia/Story?:embed=y&:display_count=yes&publish=yes&:showVizHome=no', width=900, height=550)