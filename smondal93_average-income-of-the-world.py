import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

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

income_average = document.query("IndicatorCode == 'NY.GNP.PCAP.CD'").groupby(['Year']).mean()
plt.figure()
plt.plot(income_average)
plt.xlabel('Year')
plt.ylabel('Average Income ($)')
plt.title('Average Income of the World by Year', fontsize = 14)
plt.figure()
income_average_2013 = document.query("IndicatorCode == 'NY.GNP.PCAP.CD' & Year == 2013")
boxplot = sns.boxplot(x = income_average_2013['Value'])
plt.xlabel('Average Income($) in 2013')
plt.title('Boxplot of Average Income ($) across Various Countries in 2013')

print ("Some income stats:\n" )
print ("Min: " + str(np.min(income_average_2013['Value'])))
print ("10th Percentile: " + str(np.percentile(income_average_2013['Value'],10)))
print ("25th Percentile: " + str(np.percentile(income_average_2013['Value'], 25)))
print ("Median: "+str(np.median(income_average_2013['Value'])))
print ("75th Percentile: " + str(np.percentile(income_average_2013['Value'], 75)))
print ("90th Percentile: " + str(np.percentile(income_average_2013['Value'], 90)))
print ("Max: " + str(np.max(income_average_2013['Value'])))
