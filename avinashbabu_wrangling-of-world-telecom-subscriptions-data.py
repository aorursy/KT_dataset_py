# importing libraries
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

# reading data from csv
bb_meta = pd.read_csv("../input/Fixed broadband subscriptions Definition and Source.csv")
bb = pd.read_csv("../input/Fixed broadband subscriptions.csv")
tel_meta = pd.read_csv("../input/Fixed telephone subscriptions Definition and Source.csv")
tel = pd.read_csv("../input/Fixed telephone subscriptions.csv")
c_codes = pd.read_csv("../input/Metadata_Country_API_IT.CEL.SETS.P2_DS2_en_csv_v2.csv")
mob_meta = pd.read_csv("../input/Metadata_Indicator_API_IT.CEL.SETS.P2_DS2_en_csv_v2.csv")
mob = pd.read_csv("../input/Mobile cellular subscriptions.csv")
# First we collect all the country codes from c_codes 
country_Codes = c_codes['Country Code'][c_codes['Region'].notnull()]

##
# We start with wrangling of Fixed broadband subscriptions Data
# 1. Filter rows in "bb" using country_Codes collected from "c_codes"
# 2. Rename the columns
# 3. Drop column 1990, 2000 and 2017 as it has many column values empty
# 4. Convert the datatypes
# 5. Check for Outliers in columns [2008, 2016]
# 6. Convert from wide to long
# 7. Check for NaN values in bb_sub and impute using linear regression of order 3
# 8. round the subscriptions to 2 decimal places
##

bbOnlyC = bb[bb['Country Code'].isin(country_Codes)]
names = bbOnlyC.columns.tolist()
names[4:] = ['1990', '2000', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
names[:2] = ['bb_desc', 'bb_code']
bbOnlyC.columns = names
bbOnlyC = bbOnlyC.drop(['1990', '2000', '2017'], axis=1)
bbOnlyC = bbOnlyC.convert_objects(convert_numeric = True)
bbOnlyC
bbOnlyC.iloc[:,4:].boxplot()
# Examining outlier records and necessary corrections
bbOnlyC[bbOnlyC['2008']>30]
bbOnlyC[bbOnlyC['2009']>40]
bbOnlyC[bbOnlyC['2010']>40]
bbOnlyC[bbOnlyC['2013']>40]
bbOnlyC = pd.melt(bbOnlyC, id_vars=['bb_desc', 'bb_code', 'Country Name', 'Country Code'], value_vars=['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016'], var_name='year', value_name='bb_sub')
bbOnlyC = bbOnlyC.convert_objects(convert_numeric = True)
order = 2
for c_code in bbOnlyC['Country Code'].unique():
    if ((bbOnlyC.year[bbOnlyC['bb_sub'].notnull() & (bbOnlyC['Country Code'] == c_code)].size != 0) & (bbOnlyC.year[bbOnlyC['bb_sub'].isnull() & (bbOnlyC['Country Code'] == c_code)].size != 0)):
        predict = np.poly1d(np.polyfit(bbOnlyC.year[bbOnlyC['bb_sub'].notnull() & (bbOnlyC['Country Code'] == c_code)], bbOnlyC.bb_sub[bbOnlyC['bb_sub'].notnull() & (bbOnlyC['Country Code'] == c_code)], order))
        bbOnlyC.bb_sub[bbOnlyC['bb_sub'].isnull() & (bbOnlyC['Country Code'] == c_code)] = abs(predict(bbOnlyC.year[bbOnlyC['bb_sub'].isnull() & (bbOnlyC['Country Code'] == c_code)]))
bbOnlyC = bbOnlyC.dropna(0, how = 'any')
bbOnlyC.bb_sub = bbOnlyC.bb_sub.round(4)
bbOnlyC
# Wrangling of Fixed telecom subscriptions Data
telOnlyC = tel[tel['Country Code'].isin(country_Codes)]
names = telOnlyC.columns.tolist()
names[4:] = ['1990', '2000', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
names[:2] = ['tel_desc', 'tel_code']
telOnlyC.columns = names
telOnlyC = telOnlyC.drop(['1990', '2000', '2017'], axis=1)
telOnlyC = telOnlyC.convert_objects(convert_numeric = True)
telOnlyC.iloc[:,4:].boxplot()
# Examining outlier records and necessary corrections
telOnlyC[telOnlyC['2008']>80]
telOnlyC[telOnlyC['2009']>80]
telOnlyC[telOnlyC['2010']>80]
telOnlyC[telOnlyC['2011']>80]
telOnlyC[telOnlyC['2012']>70]
telOnlyC[telOnlyC['2013']>100]
telOnlyC[telOnlyC['2014']>70]
telOnlyC[telOnlyC['2015']>60]
telOnlyC[telOnlyC['2016']>50]
telOnlyC = pd.melt(telOnlyC, id_vars=['tel_desc', 'tel_code', 'Country Name', 'Country Code'], value_vars=['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016'], var_name='year', value_name='tel_sub')
telOnlyC = telOnlyC.convert_objects(convert_numeric = True)
order = 2
for c_code in telOnlyC['Country Code'].unique():
    if ((telOnlyC.year[telOnlyC['tel_sub'].notnull() & (telOnlyC['Country Code'] == c_code)].size != 0) & (telOnlyC.year[telOnlyC['tel_sub'].isnull() & (telOnlyC['Country Code'] == c_code)].size != 0)):
        predict = np.poly1d(np.polyfit(telOnlyC.year[telOnlyC['tel_sub'].notnull() & (telOnlyC['Country Code'] == c_code)], telOnlyC.tel_sub[telOnlyC['tel_sub'].notnull() & (telOnlyC['Country Code'] == c_code)], order))
        telOnlyC.tel_sub[telOnlyC['tel_sub'].isnull() & (telOnlyC['Country Code'] == c_code)] = abs(predict(telOnlyC.year[telOnlyC['tel_sub'].isnull() & (telOnlyC['Country Code'] == c_code)]))
telOnlyC = telOnlyC.dropna(0, how = 'any')
telOnlyC.tel_sub = telOnlyC.tel_sub.round(4)
telOnlyC
# Wrangling of Mobile cellular subscriptions Data
mobOnlyC = mob[mob['Country Code'].isin(country_Codes)]
names = mobOnlyC.columns.tolist()
names[2:4] = ['mob_desc', 'mob_code']
mobOnlyC.columns = names
mobOnlyC = mobOnlyC.drop(map(str, range(1960, 2008)), axis=1)
mobOnlyC = mobOnlyC.drop(['2017'], axis=1)
mobOnlyC = mobOnlyC.convert_objects(convert_numeric = True)
mobOnlyC.iloc[:,4:].boxplot()
# Examining outlier records and necessary corrections
mobOnlyC[mobOnlyC['2011']>200]
mobOnlyC[mobOnlyC['2012']>200]
mobOnlyC[mobOnlyC['2013']>200]
mobOnlyC[mobOnlyC['2014']>200]
mobOnlyC[mobOnlyC['2015']>200]
mobOnlyC[mobOnlyC['2016']>200]
mobOnlyC[mobOnlyC['2016']<30]
mobOnlyC = pd.melt(mobOnlyC, id_vars=['Country Name', 'Country Code', 'mob_desc', 'mob_code'], value_vars=['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016'], var_name='year', value_name='mob_sub')
mobOnlyC = mobOnlyC.convert_objects(convert_numeric = True)
order = 2
for c_code in mobOnlyC['Country Code'].unique():
    if ((mobOnlyC.year[mobOnlyC['mob_sub'].notnull() & (mobOnlyC['Country Code'] == c_code)].size != 0) & (mobOnlyC.year[mobOnlyC['mob_sub'].isnull() & (mobOnlyC['Country Code'] == c_code)].size != 0)):
        predict = np.poly1d(np.polyfit(mobOnlyC.year[mobOnlyC['mob_sub'].notnull() & (mobOnlyC['Country Code'] == c_code)], mobOnlyC.mob_sub[mobOnlyC['mob_sub'].notnull() & (mobOnlyC['Country Code'] == c_code)], order))
        mobOnlyC.mob_sub[mobOnlyC['mob_sub'].isnull() & (mobOnlyC['Country Code'] == c_code)] = abs(predict(mobOnlyC.year[mobOnlyC['mob_sub'].isnull() & (mobOnlyC['Country Code'] == c_code)]))
mobOnlyC = mobOnlyC.dropna(0, how = 'any')
mobOnlyC.mob_sub = mobOnlyC.mob_sub.round(4)
mobOnlyC
bb_telOnlyC = pd.merge(bbOnlyC, telOnlyC, how='inner', on=['year', 'Country Name', 'Country Code'])
bb_tel_mobOnlyC = pd.merge(bb_telOnlyC, mobOnlyC, how='inner', on=['year', 'Country Name', 'Country Code'])
subpsData = pd.merge(bb_tel_mobOnlyC, c_codes[["Country Code", "Region", "IncomeGroup"]], how='inner', on=['Country Code'])
subpsData = subpsData.reindex_axis(['Country Name', 'Country Code', 'year', 'bb_code', 'bb_desc', 'bb_sub', 'tel_code', 'tel_desc', 'tel_sub', 'mob_code', 'mob_desc', 'mob_sub', 'Region', 'IncomeGroup'], axis=1)
# Applying log transformation on mob_sub to make its values in liase with bb_sub and tel_sub
#for i in range(0, len(subpsData)):
#    subpsData["mob_sub"][i] = math.log(subpsData["mob_sub"][i])
subpsData[subpsData.year == 2008]

subpsData.to_csv('subscriptions.csv')