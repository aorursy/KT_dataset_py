import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns







pd.set_option('display.max_rows', 50)

pd.set_option('display.max_columns', 50)

pd.set_option('display.width', 1000)



document = pd.read_csv('../input/Indicators.csv')



#want to see all the countries listed in the document  

document['CountryName'].unique()



#get rid of indicators that aren't countries 

list = ['Arab World', 'Caribbean small states', 'Central Europe and the Baltics',

 'East Asia & Pacific (all income levels)',

 'East Asia & Pacific (developing only)', 'Euro area',

 'Europe & Central Asia (all income levels)',

 'Europe & Central Asia (developing only)', 'European Union',

 'Fragile and conflict affected situations',

 'Heavily indebted poor countries (HIPC)', 'High income',

 'High income: nonOECD', 'High income: OECD',

 'Latin America & Caribbean (all income levels)',

 'Latin America & Caribbean (developing only)',

 'Least developed countries: UN classification', 'Low & middle income',

 'Low income', 'Lower middle income',

 'Middle East & North Africa (all income levels)',

 'Middle East & North Africa (developing only)', 'Middle income',

 'North America' 'OECD members' ,'Other small states',

 'Pacific island small states', 'Small states', 'South Asia',

 'Sub-Saharan Africa (all income levels)',

 'Sub-Saharan Africa (developing only)' ,'Upper middle income' ,'World', 'North America', 'OECD members']









lowestGNI_2014 = document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName != list & Year == 2014").sort_values(by = 'Value', ascending = True)[:15]

lowestGNI_1960 = document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName != list & Year == 1962").sort_values(by = 'Value', ascending = True)[:15]



fig = plt.subplots()



graph1 = sns.barplot(x = "Value", y = "CountryName", palette = "PuBu", data = lowestGNI_1960)

plt.xlabel('Average Income ($)', fontsize = 14)

plt.ylabel('Country',  fontsize=14)

plt.title('The 15 Countries with Lowest Average Income in 1962', fontsize = 14)









fig2 = plt.subplots()



graph2 = sns.barplot(x = "Value", y = "CountryName", palette = "PuBu", data = lowestGNI_2014)

plt.xlabel('Average Income($)', fontsize = 14)

plt.ylabel('Country', fontsize = 14)

plt.title('The 15 Countries with Lowest Average Income in 2014', fontsize = 14)



for key,group in lowestGNI_1960.groupby(['CountryName']):

    for key2, group2 in lowestGNI_2014.groupby(['CountryName']):

        if key == key2:

            print (key)
rich_1960 = document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName != list & Year == 1962").sort_values(by = 'Value')[-15:]

rich_2014 = document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName != list & Year == 2014").sort_values(by= 'Value')[-15:]
fig = plt.subplots()



graph_rich = sns.barplot(x = "Value", y = "CountryName", palette = "BuGn", data = rich_1960)

plt.xlabel('Average Income ($)', fontsize = 14)

plt.ylabel('Country',  fontsize=14)

plt.title('The 15 Countries with Highest Average Income in 1960', fontsize = 14)





fig = plt.subplots()



graph_rich2 = sns.barplot(x = "Value", y = "CountryName", palette = "BuGn", data = rich_2014)

plt.xlabel('Average Income ($)', fontsize = 14)

plt.ylabel('Country',  fontsize=14)

plt.title('The 15 Countries with Highest Average Income in 2014', fontsize = 14)



for key, group in rich_1960.groupby(['CountryName']):

    for key2, group2 in rich_2014.groupby(['CountryName']):

        if key == key2:

            print (key)
fig8, ax8 = plt.subplots(figsize = [15,8], ncols = 2)

ax6, ax7 = ax8



labels = []

GNP_revised = document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName == ['Australia','Austria','Canada', 'Luxembourg', 'Netherlands','Norway','United States']").groupby(['CountryName'])

for key, group in GNP_revised:

    ax6 = group.plot(ax = ax6, kind = "line", x = "Year", y = "Value", title = "Average Income from 1960-2014 in 'Rich' Countries")

    labels.append(key)



lines, _ = ax6.get_legend_handles_labels()

ax6.legend(lines, labels, loc='best')



labels2 = []

GNP_revised = document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName == ['Burundi', 'Togo', 'Malawi', 'Central African Republic']").groupby(['CountryName'])

for key, group in GNP_revised:

    ax7 = group.plot(ax = ax7, kind = "line", x = "Year", y = "Value", title = "Average Income from 1960-2014 in 'Poor' Countries")

    labels2.append(key)



lines, _ = ax7.get_legend_handles_labels()

ax7.legend(lines, labels2, loc='best')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = [15,8], sharex = True)

income_query = document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & Year == 1962 & CountryName == ['Malawi', 'China', 'Luxembourg', 'United States']")

income_query_graph = sns.barplot(x = 'CountryName', y = 'Value', order = ['Malawi', 'China', 'Luxembourg', 'United States'], ax = ax1, data = income_query)

ax1.set_title("Average Income in 1962", fontsize = 14)

ax1.set_xlabel('Country', fontsize = 14)

ax1.set_ylabel('Average Income ($)', fontsize = 14)



for p in income_query_graph.patches:

    height = p.get_height()

    income_query_graph.text(p.get_x() + p.get_width()/2., 1.05*height,

                '%d' % int(height), ha='center', va='bottom')

    

income_query_now=document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & Year == 2014 & CountryName == ['Malawi', 'China', 'Luxembourg', 'United States']")

income_query_now_graph = sns.barplot(x = 'CountryName', y = 'Value', order = ['Malawi', 'China', 'Luxembourg', 'United States'], ax = ax2, data = income_query_now)

ax2.set_title("Average Income in 2014", fontsize = 14)

ax2.set_xlabel('Country', fontsize = 14)

ax2.set_ylabel('Average Income ($)', fontsize = 14)

plt.ylim([0,90000])



for p in income_query_now_graph.patches:

    height = p.get_height()

    income_query_now_graph.text(p.get_x() + p.get_width()/2., 1.05*height,

                '%d' % int(height), ha='center', va='bottom')
a = pd.Series(income_query_now['Value'].reset_index(drop = True))

b = pd.Series(income_query['Value'].reset_index(drop = True))

ratio = a/b



income_ratio = sns.barplot(x = ['China', 'Luxembourg', 'Malawi', 'United States'], y = ratio, order = ['China', 'Luxembourg', 'United States', 'Malawi'])

plt.title('Measuring Income Growth- Which countries have seen the most change in incomes?', fontsize = 11)

plt.xlabel('Country', fontsize = 10)

plt.ylabel('Income Ratio (2014 Income/1962 Income)', fontsize = 10)



for p in income_ratio.patches:

    height = p.get_height()

    income_ratio.text(p.get_x() + p.get_width()/2., 1.05*height,

                '%d' % int(height), ha='center', va='bottom')
fig11, ax21 = plt.subplots(figsize = [15,8])

labels_cGNP = []

for key, group in document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName == ['China', 'Malawi']").groupby(['CountryName']):

    ax21 = group.plot(ax=ax21, kind = "line", x = "Year", y = "Value", title = "Comparing average incomes- China vs Malawi")

    labels_cGNP.append(key)



lines, _ = ax21.get_legend_handles_labels()

ax21.legend(lines, labels_cGNP, loc = 'best')
fig12, axs12 = plt.subplots(figsize = [15,8])

labels_cross3pt2 = []

for key, group in document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & CountryName == ['China', 'Malawi', 'Luxembourg', 'United States']").groupby(['CountryName']):

    axs12 = group.plot(ax = axs12, kind = "line", x = "Year", y = "Value", title = "Comparing average income- China vs Malawi vs Luxembourg vs US")

    labels_cross3pt2.append(key)



lines,_ = axs12.get_legend_handles_labels()

axs12.legend(lines, labels_cross3pt2, loc = 'best')
income_share = document.query("IndicatorCode == ['SI.DST.FRST.20','SI.DST.02ND.20','SI.DST.03RD.20','SI.DST.04TH.20','SI.DST.05TH.20'] & CountryName == ['Malawi', 'China', 'Luxembourg', 'United States'] & Year == 2010 ").groupby("IndicatorCode")

N = 4

i1 = income_share.get_group('SI.DST.FRST.20')['Value']

i2 = income_share.get_group('SI.DST.02ND.20')['Value']

i3 = income_share.get_group('SI.DST.03RD.20')['Value']

i4 = income_share.get_group('SI.DST.04TH.20')['Value']

i5 = income_share.get_group('SI.DST.05TH.20')['Value']



f, ax_1 = plt.subplots(1, figsize = (15,8))

ind = np.arange(N)

width = 0.35

p1 = ax_1.bar(ind, i1, width, color = '#404040')

p2 = ax_1.bar(ind, i2, width, color = '#bababa', bottom = i1)

p3 = ax_1.bar(ind, i3, width, color = '#ffffff', bottom = [i+j for i,j in zip(i1,i2)])

p4 = ax_1.bar(ind, i4, width, color = '#f4a582', bottom = [i+j+k for i,j,k in zip(i1,i2,i3)])

p5 = ax_1.bar(ind, i5, width, color = '#ca0020', bottom = [i+j+k+l for i,j,k,l in zip(i1,i2,i3,i4)])

plt.ylabel('Percent', fontsize = 14)

plt.xlabel('Country Name', fontsize = 14)

plt.xticks(ind + (width/2), ('China', 'Luxembourg', 'Malawi', 'United States'))

plt.title('Examining wealth distributions- China, Luxembourg, Malawi, and US', fontsize = 14)

plt.legend((p1[0],p2[0],p3[0],p4[0],p5[0]),('Income Share Lowest 20%', 'Income Share Second 20%', 'Income Share Third 20%', 'Income Share Fourth 20%', 'Income Share Highest 20%'), loc = 'upper right', bbox_to_anchor=(1.3, 0.9))

axes = plt.gca()

axes.set_ylim([0,100])