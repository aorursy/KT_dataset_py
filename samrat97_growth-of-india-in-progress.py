import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sqlite3

import os

import warnings

warnings.filterwarnings('ignore')



conn = sqlite3.connect('../input/database.sqlite') #SQLite connection
ind = pd.read_sql(

        """ 

            SELECT   IndicatorName, IndicatorCode 

            FROM     Indicators 

            WHERE    CountryCode = 'IND' AND Year = "1960"

        """, con=conn)

		

indicators = ind.drop_duplicates().values

#print(indicators) Uncomment this line to explicitly see all the indicator name and codes
modified_indicators = []



for ele in indicators:

	indicator_name = ele[0].strip().lower()

	modified_indicators.append([indicator_name,ele[1]])

	

Indicators = pd.DataFrame(modified_indicators,columns=['IndicatorName','IndicatorCode'])



Indicators = Indicators.drop_duplicates()

print(Indicators.shape)
key_word_dict = {}

key_word_dict['Demography'] = ['population','birth','death','fertility','mortality','expectancy']

key_word_dict['Food'] = ['food','grain','nutrition','calories']

key_word_dict['Trade'] = ['trade','import','export','good','agriculture','inventories','price','value','capital']

key_word_dict['Health'] = ['health','desease','hospital','mortality','doctor']

key_word_dict['Economy'] = ['income','gdp','gni','deficit','budget','market','stock','bond','infrastructure']

key_word_dict['Energy'] = ['fuel','energy','power','emission','electric','electricity']

key_word_dict['Education'] = ['education','literacy']

key_word_dict['Employment'] =['employed','employment','umemployed','unemployment']

key_word_dict['Rural'] = ['rural','village']

key_word_dict['Urban'] = ['urban','city']
final_indicator = []



feature = 'Trade'

for indicator_ele in Indicators.values:

    for ele in key_word_dict[feature]:

        word_list = indicator_ele[0].split()

        if ele in word_list or ele+'s' in word_list:

            final_indicator.append([indicator_ele[0],indicator_ele[1]])

            break

			

feature = 'Economy'

for indicator_ele in Indicators.values:

    for ele in key_word_dict[feature]:

        word_list = indicator_ele[0].split()

        if ele in word_list or ele+'s' in word_list:

            final_indicator.append([indicator_ele[0],indicator_ele[1]])

            break						



FinalIndicators = pd.DataFrame(final_indicator,columns=['IndicatorName','IndicatorCode'])	

print(FinalIndicators.shape)

#print(FinalIndicators) Uncomment this to see the 110 indicators 
chi = pd.read_sql(

        """ 

            SELECT   IndicatorName, IndicatorCode 

            FROM     Indicators 

            WHERE    CountryCode = 'CHN' AND Year = "1960"

        """, con=conn)



indicators = chi.drop_duplicates().values



for ele in indicators:

	indicator_name = ele[0].strip().lower()

	modified_indicators.append([indicator_name,ele[1]])

	

Indicators = pd.DataFrame(modified_indicators,columns=['IndicatorName','IndicatorCode'])



Indicators = Indicators.drop_duplicates()

#print(Indicators.shape)



final_indicator = []



feature = 'Trade'

for indicator_ele in Indicators.values:

    for ele in key_word_dict[feature]:

        word_list = indicator_ele[0].split()

        if ele in word_list or ele+'s' in word_list:

            final_indicator.append([indicator_ele[0],indicator_ele[1]])

            break

			

feature = 'Economy'

for indicator_ele in Indicators.values:

    for ele in key_word_dict[feature]:

        word_list = indicator_ele[0].split()

        if ele in word_list or ele+'s' in word_list:

            final_indicator.append([indicator_ele[0],indicator_ele[1]])

            break

			

FinalIndicatorschi = pd.DataFrame(final_indicator,columns=['IndicatorName','IndicatorCode'])



print(FinalIndicatorschi.shape)

#print(FinalIndicatorschi.values)
commonindicators = []



for keys in FinalIndicators.values:

	if keys in FinalIndicatorschi.values:

		commonindicators.append(keys)

		

commonindicatorsdf = pd.DataFrame(commonindicators,columns=['IndicatorName','IndicatorCode'])



print(commonindicatorsdf.shape)

#print(commonindicatorsdf.values) Uncomment this line to view the common Indicators of India and China over trade and economy
indicatorcodesofwork = ["'NV.AGR.TOTL.ZS'","'NE.GDI.STKB.CD'","'NE.EXP.GNFS.ZS'","'NE.RSB.GNFS.ZS'","'NY.GDP.MKTP.KD'","'NE.GDI.TOTL.ZS'","'NE.GDI.FTOT.ZS'","'NE.IMP.GNFS.ZS'","'NV.IND.TOTL.ZS'","'NV.IND.MANF.ZS'","'TG.VAL.TOTL.GD.ZS'","'NV.SRV.TETC.ZS'","'NE.TRD.GNFS.ZS'","'NY.GDP.MKTP.KN'","'NY.GDP.PCAP.KD'","'NY.GNP.PCAP.CN'","'SM.POP.TOTL.ZS'","'NY.GSR.NFCY.CD'","'NY.GDS.TOTL.ZS'","'NE.DAB.TOTL.ZS'","'NE.CON.PETC.ZS'"]
def plot_indicator(code):

	indquer = "SELECT Year, Value, IndicatorName FROM  Indicators  WHERE  CountryCode = 'IND' AND IndicatorCode = " + code

	chnquer = "SELECT Year, Value, IndicatorName FROM  Indicators  WHERE  CountryCode = 'CHN' AND IndicatorCode = " + code



	ind = pd.read_sql(indquer, con=conn)

	chn = pd.read_sql(chnquer, con=conn)



	plt.plot(ind['Year'].values,ind['Value'].values,label="India")

	plt.plot(chn['Year'].values,chn['Value'].values,label="China")



	plt.title(ind['IndicatorName'].iloc[0])

	plt.legend(loc=1)

	plt.show()
plot_indicator(indicatorcodesofwork[0])
plot_indicator(indicatorcodesofwork[1])
plot_indicator(indicatorcodesofwork[2])
plot_indicator(indicatorcodesofwork[3])
plot_indicator(indicatorcodesofwork[4])
plot_indicator(indicatorcodesofwork[5])
plot_indicator(indicatorcodesofwork[6])
plot_indicator(indicatorcodesofwork[7])
plot_indicator(indicatorcodesofwork[8])
plot_indicator(indicatorcodesofwork[9])
plot_indicator(indicatorcodesofwork[10])
plot_indicator(indicatorcodesofwork[11])
plot_indicator(indicatorcodesofwork[12])
plot_indicator(indicatorcodesofwork[13])
plot_indicator(indicatorcodesofwork[14])
plot_indicator(indicatorcodesofwork[15])
plot_indicator(indicatorcodesofwork[17])
plot_indicator(indicatorcodesofwork[18])
plot_indicator(indicatorcodesofwork[20])
plot_indicator(indicatorcodesofwork[19])