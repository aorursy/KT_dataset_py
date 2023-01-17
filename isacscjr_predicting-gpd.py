# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from matplotlib import pyplot as plt
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

df = pd.read_csv('../input/undata-country-profiles/country_profile_variables.csv')
df.head()
pd.options.display.max_rows = 1000
df[['country','GDP per capita (current US$)']]
#Wikipedia Economy of country name
#NationMaster
#CIA The World FactBook
#The World Bank
#IndexMundi
#World Health Organization

#algeria
df.at[2,'Health: Physicians (per 1000 pop.)']                         = 1.19
df.at[2,'Education: Government expenditure (% of GDP)']               = 4.34

#American Samoa
df.at[3,'GDP: Gross domestic product (million current US$)']          = 658
df.at[3,'GDP growth rate (annual %, const. 2005 prices)']             = -2.5
df.at[3,'GDP per capita (current US$)']                               = 11200
df.at[3,'Economy: Agriculture (% of GVA)']                            = 27.4
df.at[3,'Economy: Industry (% of GVA)']                               = 12.4
df.at[3,'Economy: Services and other activity (% of GVA)']            = 60.2
df.at[3,'Labour force participation (female/male pop. %)']            = '23.1/58.0'
df.at[3,'Unemployment (% of labour force)']                           = 23.8
df.at[3,'Employment: Agriculture (% of employed)']                    = 33.0
df.at[3,'Employment: Industry (% of employed)']                       = 34.0
df.at[3,'Employment: Services (% of employed)']                       = 33.0
df.at[3,'International trade: Exports (million US$)']                 = 428
df.at[3,'International trade: Imports (million US$)']                 = 615
df.at[3,'International trade: Balance (million US$)']                 = 187
df.at[3,'Population growth rate (average annual %)']                  = 0.0
df.at[3,'Health: Total expenditure (% of GDP)']                       = 7.2
#http://www.wpro.who.int/countries/asm/1FAMSpro2011_finaldraft.pdf
df.at[3,'Health: Physicians (per 1000 pop.)']                         = 0.88
#http://americansamoa.prism.spc.int/images/downloads/2011_Statistical_Yearbook.pdf
df.at[3,'Education: Government expenditure (% of GDP)']               = 12.249
df.at[3,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   = '00.0/00.0' 
df.at[3,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] = '00.00/00.00'
df.at[3,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '00.00/00.00'
df.at[3,'Seats held by women in national parliaments %']              = 0
df.at[3,'Mobile-cellular subscriptions (per 100 inhabitants)']        = 3.8
df.at[3,'Mobile-cellular subscriptions (per 100 inhabitants).1']      = 0
df.at[3,'CO2 emission estimates (million tons/tons per capita)']      = 0 
df.at[3,'Energy production, primary (Petajoules)']                    = 0.6
df.at[3,'Pop. using improved sanitation facilities (urban/rural, %)'] = 0
df.at[3,'Refugees and others of concern to UNHCR (in thousands)']     = 0
df.at[3,'Balance of payments, current account (million US$)']         = -33


#Andorra
df.at[4,'Unemployment (% of labour force)']                           = 03.7
df.at[4,'Employment: Agriculture (% of employed)']                    = 00.5
df.at[4,'Employment: Industry (% of employed)']                       = 04.4
df.at[4,'Employment: Services (% of employed)']                       = 95.1
df.at[4,'Labour force participation (female/male pop. %)']            = '38.2/61.8'
df.at[4,'Life expectancy at birth (females/males, years)']            = '80.7/85.2'
df.at[4,'Infant mortality rate (per 1000 live births']                = 3.6
df.at[4,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   = '116.6/112.0'
df.at[4,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] = '52.07/46.94'
df.at[4,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[4,'Agricultural production index (2004-2006=100)']              = 0
df.at[4,'Food production index (2004-2006=100)']                      = 0
df.at[4,'Refugees and others of concern to UNHCR (in thousands)']     = 0
df.at[4,'Balance of payments, current account (million US$)']         = 0

#Angola
df.at[5,'Health: Physicians (per 1000 pop.)']                         = 0.14
df.at[5,'Education: Government expenditure (% of GDP)']               = 3.476

#Anguilla
df.at[6,'Unemployment (% of labour force)']                           =  08.0
df.at[6,'Employment: Agriculture (% of employed)']                    =  04.0
df.at[6,'Employment: Industry (% of employed)']                       = 21.0
df.at[6,'Employment: Services (% of employed)']                       = 75.0
df.at[6,'Labour force participation (female/male pop. %)']            = '62.8/81.2'
df.at[6,'Fertility rate, total (live births per woman)']              = 1.74
df.at[6,'Life expectancy at birth (females/males, years)']            = '78.9/84.2'
df.at[6,'Infant mortality rate (per 1000 live births']                = 3.3
#http://www.heart-resources.org/wp-content/uploads/2012/05/293435-Anguilla_Health_Sector_Review_2011_Final.pdf
df.at[6,'Health: Total expenditure (% of GDP)']                       = 6.9
df.at[6,'Health: Physicians (per 1000 pop.)']                         = 1.25
df.at[6,'Education: Government expenditure (% of GDP)']               = 2.8
df.at[6,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   = '0.00/0.00'
df.at[6,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] = '0.00/0.00'
df.at[6,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[6,'Seats held by women in national parliaments %']              = 0
df.at[6,'Energy supply per capita (Gigajoules)']                      = '94.6/00.0'
df.at[6,'Pop. using improved drinking water (urban/rural, %)']        = '97.9/0.00'
df.at[6,'Agricultural production index (2004-2006=100)']              = 0
df.at[6,'Food production index (2004-2006=100)']                      = 0
df.at[6,'Refugees and others of concern to UNHCR (in thousands)']     = 0

#antigua
df.at[7,'Unemployment (% of labour force)']                            = 11.0
df.at[7,'Employment: Agriculture (% of employed)']                     = 07.0
df.at[7,'Employment: Industry (% of employed)']                        = 11.0
df.at[7,'Employment: Services (% of employed)']                        = 82.0
df.at[7,'Labour force participation (female/male pop. %)']             = '60.5/72.0'
df.at[7,'Health: Physicians (per 1000 pop.)']                          = 0.17
df.at[7,'Education: Government expenditure (% of GDP)']                = 2.52
df.at[7,'CO2 emission estimates (million tons/tons per capita)']       = 0 
df.at[7,'Refugees and others of concern to UNHCR (in thousands)']     = 0.015

#Aruba
df.at[10,'Unemployment (% of labour force)']                           = 06.9
#https://www.paho.org/hq/dmdocuments/2013/FactsheetHEFJan31.pdf
df.at[10,'Health: Total expenditure (% of GDP)']                       = 11.7
df.at[10,'Health: Physicians (per 1000 pop.)']                         = 1.82
df.at[10,'Seats held by women in national parliaments %']              = 0
df.at[10,'Pop. using improved sanitation facilities (urban/rural, %)'] = 0
df.at[10,'Agricultural production index (2004-2006=100)']              = 0
df.at[10,'Food production index (2004-2006=100)']                      = 0
df.at[10,'Refugees and others of concern to UNHCR (in thousands)']     = 0.002

#Azerbaijan
df.at[13,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] = '76.68/71.08'

# Bahamas
df.at[14,'Health: Physicians (per 1000 pop.)']                         = 2.26
df.at[14,'Education: Government expenditure (% of GDP)']               = 3.475
df.at[14,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   = '97.91/92.89'
df.at[14,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] = '95.07/90.23'
df.at[14,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '21.12/8.17'

#Barbados 
df.at[17,'Health: Physicians (per 1000 pop.)']                         = 1.21
df.at[17,'Refugees and others of concern to UNHCR (in thousands)']     = 0.001

# Belarus
df.at[18,'Population growth rate (average annual %)']                  = 0.0
df.at[18,'Urban population growth rate (average annual %)']            = 0.0

#Belize
df.at[20,'Health: Physicians (per 1000 pop.)']                         = 1.05

#Bermuda
df.at[22,'Infant mortality rate (per 1000 live births']                = 2.5
df.at[22,'Health: Total expenditure (% of GDP)']                       = 11.8
df.at[22,'Health: Physicians (per 1000 pop.)']                         = 1.77
df.at[22,'Seats held by women in national parliaments %']              = 0
df.at[22,'Energy supply per capita (Gigajoules)']                      = '100/00.0'
df.at[22,'Pop. using improved drinking water (urban/rural, %)']        = '97.9/0.00'
df.at[22,'Refugees and others of concern to UNHCR (in thousands)']     = 0

#Bhutan
df.at[23,'Refugees and others of concern to UNHCR (in thousands)']     = 0


#Bolivia
df.at[24,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'

#Bosnia and Herzegovinia
#https://pefa.org/sites/default/files/assements/comments/BA-May14-PFMPR-Public.pdf
df.at[26,'Education: Government expenditure (% of GDP)']               = 1.765
df.at[26,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[26,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] = '0.00/0.00'
df.at[26,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'

#Bostwana
df.at[27,'Education: Government expenditure (% of GDP)']               = 7.8
df.at[27,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] = '81.85/75.01'

#British Virgin Islands
df.at[29,'Unemployment (% of labour force)']                           = 08.7
df.at[29,'Employment: Agriculture (% of employed)']                    = 00.6
df.at[29,'Employment: Industry (% of employed)']                       = 40.0
df.at[29,'Employment: Services (% of employed)']                       = 59.4
df.at[29,'Labour force participation (female/male pop. %)']            = '66.5/81.4'
df.at[29,'International trade: Exports (million US$)']                 = 23
df.at[29,'Fertility rate, total (live births per woman)']              = 1.29
df.at[29,'Life expectancy at birth (females/males, years)']            = '77.4/80.3'
df.at[29,'Population age distribution (0-14 / 60+ years, %)']          = '16.7/08.9'
df.at[29,'Infant mortality rate (per 1000 live births']                = 2.5
df.at[29,'Health: Total expenditure (% of GDP)']                       = 17
df.at[29,'Health: Physicians (per 1000 pop.)']                         = 0.367
df.at[29,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[29,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] = '00.00/00.00'
df.at[29,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[29,'Seats held by women in national parliaments %']              = 0
df.at[29,'Energy supply per capita (Gigajoules)']                      = '0.00/00.0'
df.at[29,'Refugees and others of concern to UNHCR (in thousands)']     = 0.002
df.at[29,'Balance of payments, current account (million US$)']         = 362.6 

#Brunei Darasalam
df.at[30,'Energy supply per capita (Gigajoules)']                      = '100/100'
df.at[30,'Pop. using improved drinking water (urban/rural, %)']        = '0.00/0.00'

#Bonaire, Sint Eustatius and Saba
df.drop(25,inplace=True)

#Burkina faso
df.at[32,'Health: Physicians (per 1000 pop.)']                         = 0.0

#Burundi
df.at[33,'Health: Physicians (per 1000 pop.)']                         = 0.03

#Cambodia
df.at[35,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] = '41.61/48.68'

#Cameroon
df.at[36,'Health: Physicians (per 1000 pop.)']                         = 0.29

#Canada
df.at[37,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '96.50/81.12'

#Cayman Island
df.at[38,'Fertility rate, total (live births per woman)']              = 1.84
df.at[38,'Life expectancy at birth (females/males, years)']            = '78.6/84.1'
df.at[38,'Infant mortality rate (per 1000 live births']                = 5.9
#https://cnslibrary.com/wp-content/uploads/OAG-Report-Cayman-Islands-Health-System-January-2017.pdf
df.at[38,'Health: Total expenditure (% of GDP)']                       = 9.74
df.at[38,'Health: Physicians (per 1000 pop.)']                         = 1.94
#https://amithandshawn4ever.weebly.com/literacy-rate-and-education-expenditures.html
df.at[38,'Education: Government expenditure (% of GDP)']               = 6
df.at[38,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[38,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] = '00.00/00.00'
df.at[38,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[38,'Seats held by women in national parliaments %']              = 0
df.at[38,'CO2 emission estimates (million tons/tons per capita)']      = 0 
df.at[38,'Energy supply per capita (Gigajoules)']                      = '100/00.0'
df.at[38,'Pop. using improved drinking water (urban/rural, %)']        = '0.00/0.00'

# Central Africa Republic - 1994 DATA
df.at[39,'Balance of payments, current account (million US$)']         =-24.675
df.at[39,'Health: Physicians (per 1000 pop.)']                         =0.08

#Chad - 1994 DATA
df.at[40,'Balance of payments, current account (million US$)']         =-37.744
df.at[40,'Health: Physicians (per 1000 pop.)']                         = 0.0

#Channel Island (Jersey)
df.at[41,'GDP: Gross domestic product (million current US$)']          = 5569
df.at[41,'GDP growth rate (annual %, const. 2005 prices)']             = 1.0
df.at[41,'GDP per capita (current US$)']                               = 56600
df.at[41,'Economy: Agriculture (% of GVA)']                            = 02.0
df.at[41,'Economy: Industry (% of GVA)']                               = 02.0
df.at[41,'Economy: Services and other activity (% of GVA)']            = 96.0
df.at[41,'International trade: Exports (million US$)']                 = 0
df.at[41,'International trade: Imports (million US$)']                 = 0
df.at[41,'International trade: Balance (million US$)']                 = 0
#https://www.gov.je/Government/JerseyInFigures/GovernmentAccounts/pages/statesincomeexpenditure.aspx
df.at[41,'Health: Total expenditure (% of GDP)']                       = 5.03
df.at[41,'Health: Physicians (per 1000 pop.)']                         = 1.0
df.at[41,'Education: Government expenditure (% of GDP)']               = 2.463
df.at[41,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[41,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] = '0.00/0.00'
df.at[41,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[41,'Seats held by women in national parliaments %']              = 0
df.at[41,'Mobile-cellular subscriptions (per 100 inhabitants)']        = 65.63
df.at[41,'Mobile-cellular subscriptions (per 100 inhabitants).1']      = 0
df.at[41,'Individuals using the Internet (per 100 inhabitants)']       = 38.15
df.at[41,'CO2 emission estimates (million tons/tons per capita)']      = 0 
df.at[41,'Energy production, primary (Petajoules)']                    = 0
df.at[41,'Energy supply per capita (Gigajoules)']                      = '100/100'
df.at[41,'Pop. using improved drinking water (urban/rural, %)']        = '0.00/0.00'
df.at[41,'Agricultural production index (2004-2006=100)']              = 0
df.at[41,'Food production index (2004-2006=100)']                      = 0
df.at[41,'Refugees and others of concern to UNHCR (in thousands)']     = 0
df.at[41,'Balance of payments, current account (million US$)']         = 0

#Chile
df.at[42,'Health: Physicians (per 1000 pop.)']                         = 0.6

#Hong Kong
#https://www.dh.gov.hk/english/statistics/statistics_hs/files/Health_Statistics_pamphlet_E.pdf
df.at[43,'Health: Total expenditure (% of GDP)']                       = 6.0
df.at[43,'Health: Physicians (per 1000 pop.)']                         = 0.3
df.at[43,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='105.98/108.31'
df.at[43,'Seats held by women in national parliaments %']              = 0
df.at[43,'Threatened species (number)']                                = 43
df.at[43,'CO2 emission estimates (million tons/tons per capita)']      = 7 
df.at[43,'Energy supply per capita (Gigajoules)']                      = '100/100'
df.at[43,'Pop. using improved drinking water (urban/rural, %)']        = '100/0.00'


#Macao
#http://hiip.wpro.who.int/portal/countryprofiles/MacaoChina.aspx
df.at[44,'Economy: Agriculture (% of GVA)']                            = 0.0
df.at[44,'Health: Total expenditure (% of GDP)']                       = 10.3
df.at[44,'Health: Physicians (per 1000 pop.)']                         = 0.4
df.at[44,'Seats held by women in national parliaments %']              = 0
df.at[44,'Threatened species (number)']                                = 9
df.at[44,'Energy supply per capita (Gigajoules)']                      = '100/00.0'
df.at[44,'Pop. using improved drinking water (urban/rural, %)']        = '100/0.00'
df.at[44,'Refugees and others of concern to UNHCR (in thousands)']     = 0.008

#China
df.at[45,'Education: Government expenditure (% of GDP)']               = 4.0

#Colombia
df.at[46,'Health: Physicians (per 1000 pop.)']                         = 1.35

#Comoros
#2004 DATA
df.at[47,'Health: Physicians (per 1000 pop.)']                         = 0.19
df.at[47,'Refugees and others of concern to UNHCR (in thousands)']     = 0.001

#Congo
df.at[48,'Health: Physicians (per 1000 pop.)']                         = 0.1
df.at[48,'Education: Government expenditure (% of GDP)']               = 6.2

#Cook Island
df.at[49,'Health: Physicians (per 1000 pop.)']                         = 1.19
df.at[49,'Seats held by women in national parliaments %']              = 0
df.at[49,'Refugees and others of concern to UNHCR (in thousands)']     = 0
df.at[41,'Balance of payments, current account (million US$)']         = 125

#Cuba
df.at[52,'Education: Government expenditure (% of GDP)']               = 12.837

#North Korea
#https://tradingeconomics.com/north-korea/health-expenditure-public-percent-of-gdp-wb-data.html
df.at[55,'Health: Total expenditure (% of GDP)']                       = 9.06
df.at[55,'Education: Government expenditure (% of GDP)']               = 4.21
df.at[55,'Refugees and others of concern to UNHCR (in thousands)']     = 0

#Comoro - 2012 DATA
df.at[47,'Balance of payments, current account (million US$)']         =-41.140

#Congo - 2016 DATA
df.at[48,'Balance of payments, current account (million US$)']         =-1334

#Cook Island
df.at[49,'Mobile-cellular subscriptions (per 100 inhabitants)']        = 0
df.at[49,'Mobile-cellular subscriptions (per 100 inhabitants).1']      = 0
df.at[49,'CO2 emission estimates (million tons/tons per capita)']      = 0 

#Cuba
df.at[52,'Balance of payments, current account (million US$)']         = 2008

#North Korea
df.at[55,'Balance of payments, current account (million US$)']         = 870.31


#Dem Congo
df.at[56,'Health: Physicians (per 1000 pop.)']                         = 0.09

#Djiboti
df.at[58,'Education: Government expenditure (% of GDP)']               = 4.49

#Dominica
df.at[59,'Unemployment (% of labour force)']                           = 12.5
df.at[59,'Employment: Agriculture (% of employed)']                    = 14.6
df.at[59,'Employment: Industry (% of employed)']                       = 22.3
df.at[59,'Employment: Services (% of employed)']                       = 63.1
df.at[59,'Labour force participation (female/male pop. %)']            = '40.8/61.8'
df.at[59,'Fertility rate, total (live births per woman)']              = 2.04
df.at[59,'Life expectancy at birth (females/males, years)']            = '74.2/80.3'
df.at[59,'Population age distribution (0-14 / 60+ years, %)']          = '21.7/11.1'
df.at[59,'Infant mortality rate (per 1000 live births']                = 10.6
df.at[59,'Health: Physicians (per 1000 pop.)']                         = 0.5
df.at[59,'Education: Government expenditure (% of GDP)']               = 3.39
df.at[59,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '7.24/7.21'
df.at[59,'Energy supply per capita (Gigajoules)']                      = '100/100'
df.at[59,'Pop. using improved drinking water (urban/rural, %)']        = '0.00/0.00'
df.at[59,'Refugees and others of concern to UNHCR (in thousands)']     = 0

#Dominican Republic
df.at[60,'Education: Government expenditure (% of GDP)']               = 2.047

#Egypt 
df.at[62,'Education: Government expenditure (% of GDP)']               = 3.761

#El Salvador
df.at[63,'Health: Physicians (per 1000 pop.)']                         = 1.24

#Equatorial Guine
df.at[64,'Balance of payments, current account (million US$)']         =-344
df.at[64,'Health: Physicians (per 1000 pop.)']                         = 0.3
df.at[64,'Education: Government expenditure (% of GDP)']               = 2.19
df.at[64,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] = '21.95/29.84'
df.at[64,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '1.21/2.37'

df.at[64,'Refugees and others of concern to UNHCR (in thousands)']     = 40

# Eritreia
df.at[65,'Balance of payments, current account (million US$)']         =-104
df.at[65,'Health: Physicians (per 1000 pop.)']                         = 0.05
df.at[65,'Education: Government expenditure (% of GDP)']               = 2.127

#Ethiopia
df.at[67,'Balance of payments, current account (million US$)']         =-8269
df.at[67,'Health: Physicians (per 1000 pop.)']                         = 0.03

#Falkland Islands
df.at[68,'GDP: Gross domestic product (million current US$)']          = 206
df.at[68,'GDP growth rate (annual %, const. 2005 prices)']             = 25.5
df.at[68,'GDP per capita (current US$)']                               = 70800
df.at[68,'Economy: Agriculture (% of GVA)']                            = 95.0
df.at[68,'Economy: Industry (% of GVA)']                               = 02.5
df.at[68,'Economy: Services and other activity (% of GVA)']            = 02.5
df.at[68,'Unemployment (% of labour force)']                           = 04.1
df.at[68,'Employment: Agriculture (% of employed)']                    = 95.0
df.at[68,'Employment: Industry (% of employed)']                       = 02.5
df.at[68,'Employment: Services (% of employed)']                       = 02.5
df.at[68,'Labour force participation (female/male pop. %)']            = '40.3/69.4'
df.at[68,'Fertility rate, total (live births per woman)']              = 0
df.at[68,'Life expectancy at birth (females/males, years)']            = '75.6/79.6'
df.at[68,'Infant mortality rate (per 1000 live births']                = 4.9
#http://en.mercopress.com/2017/06/03/falkland-islands-budget-highlights-for-2017-2018-spending-within-means-and-no-borrowing
df.at[68,'Health: Total expenditure (% of GDP)']                       = 8.23
# https://www.fidc.co.fk/library/other-documents/192-fig-statistics-yearbook-2014/file
df.at[68,'Health: Physicians (per 1000 pop.)']                         = 2.5
df.at[68,'Education: Government expenditure (% of GDP)']               = 4.89
df.at[68,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[68,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='0.00/0.00'
df.at[68,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[68,'Seats held by women in national parliaments %']              = 0
df.at[68,'Energy supply per capita (Gigajoules)']                      = '00.0/00.0'
df.at[68,'Pop. using improved drinking water (urban/rural, %)']        = '0.00/0.00'
df.at[68,'Refugees and others of concern to UNHCR (in thousands)']     = 0
df.at[68,'Balance of payments, current account (million US$)']         = 0

# Faroe Islands
df.at[69,'Economy: Agriculture (% of GVA)']                            = 18
df.at[69,'GDP growth rate (annual %, const. 2005 prices)']             = 7.5
df.at[69,'GDP per capita (current US$)']                               = 45709
df.at[69,'Unemployment (% of labour force)']                           = 05.5
df.at[69,'Employment: Agriculture (% of employed)']                    = 10.7
df.at[69,'Employment: Industry (% of employed)']                       = 18.9
df.at[69,'Employment: Services (% of employed)']                       = 70.3
df.at[69,'Balance of payments, current account (million US$)']         =-194.304
df.at[69,'Infant mortality rate (per 1000 live births']                = 5.4
#http://www.hagstova.fo/sites/default/files/Faroe%20Islands%20in%20figures%202014_0.pdf
df.at[69,'Health: Total expenditure (% of GDP)']                       = 7.7
df.at[69,'Health: Physicians (per 1000 pop.)']                         = 2.63
df.at[69,'Education: Government expenditure (% of GDP)']               = 8.1
df.at[69,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[69,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='0.00/0.00'
df.at[69,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[69,'Seats held by women in national parliaments %']              = 0
df.at[69,'Energy supply per capita (Gigajoules)']                      = '00.0/00.0'
df.at[69,'Pop. using improved drinking water (urban/rural, %)']        = '0.00/0.00'
df.at[69,'Refugees and others of concern to UNHCR (in thousands)']     = 0

#Fiji
df.at[70,'Health: Physicians (per 1000 pop.)']                         = 0.426
df.at[70,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[70,'Refugees and others of concern to UNHCR (in thousands)']     = 0.007

# French Guiana
df.at[73,'GDP: Gross domestic product (million current US$)']          = 4400
df.at[73,'GDP growth rate (annual %, const. 2005 prices)']             = 2.13
df.at[73,'GDP per capita (current US$)']                               = 16200
df.at[73,'Economy: Agriculture (% of GVA)']                            = 0.0
df.at[73,'Economy: Industry (% of GVA)']                               = 0.0
df.at[73,'Economy: Services and other activity (% of GVA)']            = 100.0
df.at[73,'Employment: Agriculture (% of employed)']                    = 26.8
df.at[73,'International trade: Exports (million US$)']                 = 1501
df.at[73,'International trade: Imports (million US$)']                 = 1693
df.at[73,'International trade: Balance (million US$)']                 = -192
df.at[73,'Health: Total expenditure (% of GDP)']                       = 5.5
#https://www.paho.org/hq/dmdocuments/2012/2012-hia-frenchguiana.pdf
df.at[73,'Health: Physicians (per 1000 pop.)']                         = 1.8
#http://rozenbergquarterly.com/extended-statehood-the-french-departements-doutre-mer-guadeloupe-and-martinique/
df.at[73,'Education: Government expenditure (% of GDP)']               = 0.767
df.at[73,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[73,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='0.00/0.00'
df.at[73,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[73,'Seats held by women in national parliaments %']              = 0
df.at[73,'Mobile-cellular subscriptions (per 100 inhabitants)']        = 0
df.at[73,'Mobile-cellular subscriptions (per 100 inhabitants).1']      = 0
df.at[73,'Refugees and others of concern to UNHCR (in thousands)']     = 0.011
df.at[73,'Balance of payments, current account (million US$)']         = 0

#French Polynesia
#http://hiip.wpro.who.int/portal/countryprofiles/FrenchPolynesia.aspx
df.at[74,'Health: Total expenditure (% of GDP)']                       = 13.9
df.at[74,'Health: Physicians (per 1000 pop.)']                         = 3.17
#https://tradingeconomics.com/french-polynesia/trade-percent-of-gdp-wb-data.html
df.at[74,'Education: Government expenditure (% of GDP)']               = 0.49
df.at[74,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[74,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='0.00/0.00'
df.at[74,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  ='0.00/0.00'
df.at[74,'Seats held by women in national parliaments %']              = 0
df.at[74,'Refugees and others of concern to UNHCR (in thousands)']     = 0

#Gabon -2012 DATA
df.at[75,'Balance of payments, current account (million US$)']         =-1978
df.at[75,'Health: Physicians (per 1000 pop.)']                         = 0.41
df.at[75,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='44.75/51.03'
df.at[75,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '6.13/10.75'


# Gambia
df.at[76,'Balance of payments, current account (million US$)']         =-95.561
df.at[76,'Health: Physicians (per 1000 pop.)']                         = 0.11
df.at[76,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='55.67/58.51'

#Ghana
df.at[79,'Health: Physicians (per 1000 pop.)']                         =0.1

# Gibraltar
df.at[80,'GDP: Gross domestic product (million current US$)']          = 2044
df.at[80,'GDP growth rate (annual %, const. 2005 prices)']             = 9.5
df.at[80,'GDP per capita (current US$)']                               = 61700
df.at[80,'Economy: Agriculture (% of GVA)']                            = 0.0
df.at[80,'Economy: Industry (% of GVA)']                               = 0.0
df.at[80,'Economy: Services and other activity (% of GVA)']            = 100.0
df.at[80,'Unemployment (% of labour force)']                           = 03.0
df.at[80,'Employment: Agriculture (% of employed)']                    = 00.0
df.at[80,'Employment: Industry (% of employed)']                       = 40.0
df.at[80,'Employment: Services (% of employed)']                       = 60.0
df.at[80,'Labour force participation (female/male pop. %)']            = '00.0/00.0'
df.at[80,'Fertility rate, total (live births per woman)']              = 1.90
df.at[80,'Life expectancy at birth (females/males, years)']            = '76.7/82.6'
df.at[80,'Infant mortality rate (per 1000 live births']                = 5.9
#https://www.gibraltar.gov.gi/new/sites/default/files/HMGoG_Documents/Budget%202017-18%20WEB.PDF
df.at[80,'Health: Total expenditure (% of GDP)']                       = 18.22
df.at[80,'Health: Physicians (per 1000 pop.)']                         = 0
df.at[80,'Education: Government expenditure (% of GDP)']               = 5.91
df.at[80,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[80,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='0.00/0.00'
df.at[80,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  ='0.00/0.00'
df.at[80,'Seats held by women in national parliaments %']              = 0
df.at[80,'CO2 emission estimates (million tons/tons per capita)']      = 0 
df.at[80,'Energy supply per capita (Gigajoules)']                      = '100/00.0'
df.at[80,'Pop. using improved drinking water (urban/rural, %)']        = '0.00/0.00'
df.at[80,'Agricultural production index (2004-2006=100)']              = 0
df.at[80,'Food production index (2004-2006=100)']                      = 0
df.at[80,'Refugees and others of concern to UNHCR (in thousands)']     = 0
#http://trade.ec.europa.eu/doclib/docs/2011/january/tradoc_147284.pdf
df.at[80,'Balance of payments, current account (million US$)']         = -7700

#Greece
df.at[81,'Education: Government expenditure (% of GDP)']               = 3.964

#Greenland
#http://www.stat.gl/publ/en/GF/2017/pdf/Greenland%20in%20Figures%202017.pdf
df.at[82,'Health: Total expenditure (% of GDP)']                       = 10.61
df.at[82,'Health: Physicians (per 1000 pop.)']                         = 1.73
#http://www.stat.gl/publ/kl/GF/2012/takussutissiat/Greenland%20in%20Figures%202012.pdf
df.at[82,'Education: Government expenditure (% of GDP)']               = 5.525
df.at[82,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[82,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='0.00/0.00'
df.at[82,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[82,'Seats held by women in national parliaments %']              = 0
df.at[82,'Threatened species (number)']                                = 25
df.at[82,'Refugees and others of concern to UNHCR (in thousands)']     = 0
df.at[82,'Labour force participation (female/male pop. %)']            = '71.49/76.31'
#http://trade.ec.europa.eu/doclib/docs/2011/january/tradoc_147286.pdf
df.at[82,'Balance of payments, current account (million US$)']         = -100

# Grenada
df.at[83,'Unemployment (% of labour force)']                           = 33.5
df.at[83,'Employment: Agriculture (% of employed)']                    = 11.0
df.at[83,'Employment: Industry (% of employed)']                       = 20.0
df.at[83,'Employment: Services (% of employed)']                       = 69.0
df.at[83,'Labour force participation (female/male pop. %)']            = '43.3/67.7'
df.at[83,'Health: Physicians (per 1000 pop.)']                         = 0.66
df.at[83,'Education: Government expenditure (% of GDP)']               = 10.29
df.at[83,'Refugees and others of concern to UNHCR (in thousands)']     = 0

#Guadeloupe
df.at[84,'GDP: Gross domestic product (million current US$)']          = 9740
df.at[84,'GDP growth rate (annual %, const. 2005 prices)']             = 0
df.at[84,'GDP per capita (current US$)']                               = 21780
df.at[84,'Economy: Agriculture (% of GVA)']                            = 15.0
df.at[84,'Economy: Industry (% of GVA)']                               = 17.0
df.at[84,'Economy: Services and other activity (% of GVA)']            = 68.0
df.at[84,'Labour force participation (female/male pop. %)']            = '37.0/45.2'
df.at[84,'International trade: Exports (million US$)']                 = 676
df.at[84,'International trade: Imports (million US$)']                 = 3102
df.at[84,'International trade: Balance (million US$)']                 = -2426
df.at[84,'Population growth rate (average annual %)']                  = 0.0
df.at[84,'Health: Total expenditure (% of GDP)']                       = 10.46
#https://www.paho.org/hq/dmdocuments/2012/2012-hia-frenchguiana.pdf
df.at[84,'Health: Physicians (per 1000 pop.)']                         =2.7
#http://rozenbergquarterly.com/extended-statehood-the-french-departements-doutre-mer-guadeloupe-and-martinique/
#http://rozenbergquarterly.com/wp-content/uploads/2015/10/pag75.jpg
df.at[84,'Education: Government expenditure (% of GDP)']               = 0.542
df.at[84,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[84,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='0.00/0.00'
df.at[84,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[84,'Seats held by women in national parliaments %']              = 0
df.at[84,'Mobile-cellular subscriptions (per 100 inhabitants)']        = 0
df.at[84,'Mobile-cellular subscriptions (per 100 inhabitants).1']      = 0
df.at[84,'Refugees and others of concern to UNHCR (in thousands)']     = 0
df.at[84,'Balance of payments, current account (million US$)']         = -100


#Guam
df.at[85,'GDP: Gross domestic product (million current US$)']          = 5793
df.at[85,'GDP growth rate (annual %, const. 2005 prices)']             = 0.4
df.at[85,'GDP per capita (current US$)']                               = 35600
df.at[85,'Economy: Agriculture (% of GVA)']                            = 7.0
df.at[85,'Economy: Industry (% of GVA)']                               = 15.0
df.at[85,'Economy: Services and other activity (% of GVA)']            = 78.0
df.at[85,'International trade: Exports (million US$)']                 = 1124
df.at[85,'International trade: Imports (million US$)']                 = 2964
df.at[85,'International trade: Balance (million US$)']                 = -1840
df.at[85,'Health: Total expenditure (% of GDP)']                       = 9.0
df.at[85,'Health: Physicians (per 1000 pop.)']                         = 0.84
#http://www.opaguam.org/sites/default/files/gdoehighlights17_0.pdf
df.at[85,'Education: Government expenditure (% of GDP)']               = 3.9
df.at[85,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[85,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='0.00/0.00'
df.at[85,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[85,'Seats held by women in national parliaments %']              = 0
df.at[85,'Mobile-cellular subscriptions (per 100 inhabitants)']        = 3.8
df.at[85,'CO2 emission estimates (million tons/tons per capita)']      = 0 
df.at[85,'Energy production, primary (Petajoules)']                    = 0.005
df.at[85,'Balance of payments, current account (million US$)']         = 0

#guatemala
df.at[86,'Health: Physicians (per 1000 pop.)']                         = 0.9

#guine-bissau
df.at[87,'Health: Physicians (per 1000 pop.)']                         = 0.08
df.at[87,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='114.05/122.11'
df.at[87,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='12.34/22.79'
df.at[87,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.05/0.93'

#Guinea
df.at[88,'Health: Physicians (per 1000 pop.)']                         = 0.1

#Guyana
df.at[89,'Health: Physicians (per 1000 pop.)']                         = 0.21
df.at[89,'Refugees and others of concern to UNHCR (in thousands)']     = 0.007

#Haiti
df.at[90,'Health: Physicians (per 1000 pop.)']                         = 0.1
df.at[90,'Education: Government expenditure (% of GDP)']               = 10.29
df.at[90,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='112.96/110.75'
df.at[90,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='12.35/22.79'
df.at[90,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.52/1.54'

#Honduras
df.at[92,'Health: Physicians (per 1000 pop.)']                          = 0.37

#Holy See
df.drop(91,inplace=True)

#Iran
df.at[97,'Balance of payments, current account (million US$)']          = 12481

#Iraq
df.at[98,'Education: Government expenditure (% of GDP)']                = 3.55
df.at[98,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']    ='98.74/116.98'
df.at[98,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)']  ='45.59/60.99'
df.at[98,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']   = '9.37/19.99'

# Isle of Man
df.at[100,'GDP: Gross domestic product (million current US$)']          = 6792
df.at[100,'GDP growth rate (annual %, const. 2005 prices)']             = -8.6
df.at[100,'GDP per capita (current US$)']                               = 84600
df.at[100,'Economy: Agriculture (% of GVA)']                            = 1.0
df.at[100,'Economy: Industry (% of GVA)']                               = 13.0
df.at[100,'Economy: Services and other activity (% of GVA)']            = 86.0
df.at[100,'Unemployment (% of labour force)']                           = 02.0
df.at[100,'Employment: Agriculture (% of employed)']                    = 03.0
df.at[100,'Employment: Industry (% of employed)']                       = 21.0
df.at[100,'Employment: Services (% of employed)']                       = 76.0
df.at[100,'International trade: Exports (million US$)']                 = 0
df.at[100,'International trade: Imports (million US$)']                 = 0
df.at[100,'International trade: Balance (million US$)']                 = 0
df.at[100,'Fertility rate, total (live births per woman)']              = 1.92
df.at[100,'Life expectancy at birth (females/males, years)']            = '77.4/80.3'
df.at[100,'Infant mortality rate (per 1000 live births']                = 4.0
# http://www.tynwald.org.im/business/OPHansardIndex1416/3729.pdf
df.at[100,'Health: Total expenditure (% of GDP)']                       = 4.1
df.at[100,'Health: Physicians (per 1000 pop.)']                         = 0.2
#https://www.gov.im/media/1350838/2016-02-19-isle-of-man-in-numbers-2016-report-final.pdf
df.at[100,'Education: Government expenditure (% of GDP)']               = 1.71
df.at[100,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[100,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='0.00/0.00'
df.at[100,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[100,'Seats held by women in national parliaments %']              = 0
df.at[100,'Mobile-cellular subscriptions (per 100 inhabitants)']        = 0
df.at[100,'Mobile-cellular subscriptions (per 100 inhabitants).1']      = 0
df.at[100,'Energy supply per capita (Gigajoules)']                      = '100/100'
df.at[100,'Pop. using improved drinking water (urban/rural, %)']        = '0.00/0.00'
df.at[100,'Agricultural production index (2004-2006=100)']              = 0
df.at[100,'Food production index (2004-2006=100)']                      = 0
df.at[100,'Refugees and others of concern to UNHCR (in thousands)']     = 0
df.at[100,'Balance of payments, current account (million US$)']         = 0

#Jamaica
df.at[103,'Health: Physicians (per 1000 pop.)']                         = 0.41
df.at[103,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='96.13/96.45'
df.at[103,'Refugees and others of concern to UNHCR (in thousands)']     = 0.008

#Jordan
df.at[105,'Education: Government expenditure (% of GDP)']               = 3.895

#Kenya
df.at[107,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='54.91/60.74'
df.at[107,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '3.33/4.74'

# Kiribati
df.at[108,'Unemployment (% of labour force)']                           = 02.0
df.at[108,'Employment: Agriculture (% of employed)']                    = 02.7
df.at[108,'Employment: Industry (% of employed)']                       = 32.0
df.at[108,'Employment: Services (% of employed)']                       = 65.3
df.at[108,'Labour force participation (female/male pop. %)']            = '79.6/83.2'
df.at[108,'Education: Government expenditure (% of GDP)']               = 11.99
df.at[108,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='91.54/82.94'
df.at[108,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[108,'Refugees and others of concern to UNHCR (in thousands)']     = 0

#Kuwait
df.at[109,'Education: Government expenditure (% of GDP)']               = 3.760

#Lao
df.at[111,'Refugees and others of concern to UNHCR (in thousands)']     = 0

#Lesotho 
df.at[114,'Health: Physicians (per 1000 pop.)']                         = 0.043
df.at[114,'Education: Government expenditure (% of GDP)']               = 11.39
df.at[114,'Refugees and others of concern to UNHCR (in thousands)']     = 0.036

#Liberia
df.at[115,'Health: Physicians (per 1000 pop.)']                         = 0.01

#Libya
df.at[116,'Education: Government expenditure (% of GDP)']               = 2.3
df.at[116,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='106.98/111.98'
df.at[116,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='105.97/90.15'
df.at[116,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '63.64/57.50'
df.at[116,'Energy supply per capita (Gigajoules)']                      = '99.10/96.44'

#Liechtenstein
df.at[117,'Unemployment (% of labour force)']                           = 02.5
df.at[117,'Employment: Agriculture (% of employed)']                    = 00.8
df.at[117,'Employment: Industry (% of employed)']                       = 39.4
df.at[117,'Employment: Services (% of employed)']                       = 59.95
df.at[117,'International trade: Exports (million US$)']                 = 3217
df.at[117,'International trade: Imports (million US$)']                 = 0
df.at[117,'International trade: Balance (million US$)']                 = 3217
df.at[117,'Life expectancy at birth (females/males, years)']            = '79.7/84.6'
df.at[117,'Infant mortality rate (per 1000 live births']                = 4.2
df.at[117,'Health: Total expenditure (% of GDP)']                       = 13.61
df.at[117,'Health: Physicians (per 1000 pop.)']                         = 7.4
df.at[117,'Energy supply per capita (Gigajoules)']                      = '100/100'
df.at[117,'Pop. using improved drinking water (urban/rural, %)']        = '0.00/0.00'
#https://www.ons.gov.uk/economy/nationalaccounts/balanceofpayments/timeseries/bfpg
df.at[117,'Balance of payments, current account (million US$)']         = -61

#Malawi
df.at[121,'Health: Physicians (per 1000 pop.)']                         = 0.02

#Maldives
df.at[123,'Health: Physicians (per 1000 pop.)']                         = 1.58
df.at[123,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='74.63/63.96'
df.at[123,'Refugees and others of concern to UNHCR (in thousands)']     = 0

#Mali
df.at[124,'Health: Physicians (per 1000 pop.)']                         = 0.09

#Marshall Island
df.at[126,'Employment: Agriculture (% of employed)']                    = 22.0
df.at[126,'Employment: Industry (% of employed)']                       = 18.0
df.at[126,'Employment: Services (% of employed)']                       = 60.0
df.at[126,'Education: Government expenditure (% of GDP)']               = 12.24
df.at[126,'Refugees and others of concern to UNHCR (in thousands)']     = 0

# Martinique
df.at[127,'Economy: Agriculture (% of GVA)']                            = 6
df.at[127,'GDP growth rate (annual %, const. 2005 prices)']             = 2.1
df.at[127,'International trade: Exports (million US$)']                 = 957
df.at[127,'International trade: Imports (million US$)']                 = 3098
df.at[127,'International trade: Balance (million US$)']                 =-2141
df.at[127,'Health: Total expenditure (% of GDP)']                       = 10.71
df.at[127,'Health: Physicians (per 1000 pop.)']                         = 2.6
df.at[127,'Education: Government expenditure (% of GDP)']               = 1.047
df.at[127,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[127,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='0.00/0.00'
df.at[127,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[127,'Seats held by women in national parliaments %']              = 0
df.at[127,'Mobile-cellular subscriptions (per 100 inhabitants)']        = 0
df.at[127,'Mobile-cellular subscriptions (per 100 inhabitants).1']      = 0
df.at[127,'Refugees and others of concern to UNHCR (in thousands)']     = 0
df.at[127,'Balance of payments, current account (million US$)']         = 0

#Mauritania
df.at[128,'Health: Physicians (per 1000 pop.)']                         = 0.13

#Mauritius
df.at[129,'Health: Physicians (per 1000 pop.)']                         = 1.93
df.at[129,'Refugees and others of concern to UNHCR (in thousands)']     = 0

#Mayotte
df.drop(130, inplace=True)

#Micronesia
df.at[132,'Unemployment (% of labour force)']                           = 16.2
df.at[132,'Employment: Agriculture (% of employed)']                    = 00.9
df.at[132,'Employment: Industry (% of employed)']                       = 05.2
df.at[132,'Employment: Services (% of employed)']                       = 93.9
df.at[132,'Health: Physicians (per 1000 pop.)']                         = 0.19
df.at[132,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] = '86.7/80.08'
df.at[132,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '20.97/23.68'
df.at[132,'Refugees and others of concern to UNHCR (in thousands)']     = 0.003
df.at[132,'Labour force participation (female/male pop. %)']            = '0.0/0.0'

#Monaco
df.at[133,'Economy: Agriculture (% of GVA)']                            = 0.0
df.at[133,'Unemployment (% of labour force)']                           = 02.0
df.at[133,'Employment: Agriculture (% of employed)']                    = 00.0
df.at[133,'Employment: Industry (% of employed)']                       = 16.1
df.at[133,'Employment: Services (% of employed)']                       = 83.9
df.at[133,'Labour force participation (female/male pop. %)']            = '33.2/55.9'
df.at[133,'International trade: Exports (million US$)']                 = 964.6
df.at[133,'International trade: Imports (million US$)']                 = 1371
df.at[133,'International trade: Balance (million US$)']                 = -352.4
df.at[133,'Fertility rate, total (live births per woman)']              = 1.53
df.at[133,'Life expectancy at birth (females/males, years)']            = '85.6/93.5'
df.at[133,'Population age distribution (0-14 / 60+ years, %)']          = '10.7/32.2'
df.at[133,'Infant mortality rate (per 1000 live births']                = 1.8
df.at[133,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[133,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='0.00/0.00'
df.at[133,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[133,'Threatened species (number)']                                = 18
df.at[133,'Energy supply per capita (Gigajoules)']                      = '100/00.0'
#It's depend of french energy
df.at[133,'Energy production, primary (Petajoules)']                    = 0
df.at[133,'Agricultural production index (2004-2006=100)']              = 0
df.at[133,'Food production index (2004-2006=100)']                      = 0
df.at[133,'Refugees and others of concern to UNHCR (in thousands)']     = 0.032
df.at[133,'Balance of payments, current account (million US$)']         = -194.8

#Mongolia
df.at[134,'Refugees and others of concern to UNHCR (in thousands)']     = 0.009

#Montenegro
df.at[135,'Education: Government expenditure (% of GDP)']               = 9.94
df.at[135,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '63.36/46.47'

#Montserrat
df.at[136,'Unemployment (% of labour force)']                           = 05.8
df.at[136,'Employment: Agriculture (% of employed)']                    = 01.4
df.at[136,'Employment: Industry (% of employed)']                       = 12.7
df.at[136,'Employment: Services (% of employed)']                       = 85.9
df.at[136,'Labour force participation (female/male pop. %)']            = '47.6/68.1'
df.at[136,'Fertility rate, total (live births per woman)']              = 1.33
df.at[136,'Life expectancy at birth (females/males, years)']            = '75.9/73.2'
df.at[136,'Infant mortality rate (per 1000 live births']                = 12.3
#https://www.paho.org/hq/dmdocuments/2013/FactsheetHEFJan31.pdf
df.at[136,'Health: Total expenditure (% of GDP)']                       = 11.2
df.at[136,'Health: Physicians (per 1000 pop.)']                         = 1.2
df.at[136,'Education: Government expenditure (% of GDP)']               = 5.1
df.at[136,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[136,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='0.00/0.00'
df.at[136,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[136,'Seats held by women in national parliaments %']              = 0
df.at[136,'CO2 emission estimates (million tons/tons per capita)']      = 0 
df.at[136,'Pop. using improved drinking water (urban/rural, %)']        = '0.00/0.00'
df.at[136,'Refugees and others of concern to UNHCR (in thousands)']     = 0

#Morocco
df.at[137,'Education: Government expenditure (% of GDP)']               = 5.261

#Myanmar
df.at[139,'Education: Government expenditure (% of GDP)']               = 2.17

# Namibia
df.at[140,'Health: Total expenditure (% of GDP)']                       = 4.0
df.at[140,'Health: Physicians (per 1000 pop.)']                         = 0.37
df.at[140,'Education: Government expenditure (% of GDP)']               = 8.35
df.at[140,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='69.03/60.12'
df.at[140,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '10.40/8.20'

#Nauru (No actual data just says that the most of it work in mining, education e transportation, DROPPING IT
df.drop([141], inplace=True)

#Nepal
df.at[142,'Health: Physicians (per 1000 pop.)']                         = 0.598

#New Caledonia
df.at[144,'Health: Total expenditure (% of GDP)']                       = 9.5
df.at[144,'Health: Physicians (per 1000 pop.)']                         = 2.22
#https://www.indexmundi.com/facts/new-caledonia/education-expenditure
df.at[144,'Education: Government expenditure (% of GDP)']               = 0.54
df.at[144,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[144,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='0.00/0.00'
df.at[144,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[144,'Seats held by women in national parliaments %']              = 0
df.at[144,'Refugees and others of concern to UNHCR (in thousands)']     = 0

#New Zealand
df.at[145,'Pop. using improved drinking water (urban/rural, %)']        = '100/100'

#Nicaragua
df.at[146,'Education: Government expenditure (% of GDP)']               = 4.484
df.at[146,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='122.72/123.76'
df.at[146,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='78.78/69.83'
df.at[146,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '18.66/16.80'

#Niger
df.at[147,'Health: Physicians (per 1000 pop.)']                         = 0.02

#Nigeria
df.at[148,'Health: Physicians (per 1000 pop.)']                         = 0.38
df.at[148,'Education: Government expenditure (% of GDP)']               = 3.064

#Niue (There are no actual paid jobs out of the public sector, more of the work force is in plantation so i will DROP IT
df.drop([149], inplace=True)

#Northern Mariana Islands
df.at[150,'GDP: Gross domestic product (million current US$)']          = 1242
df.at[150,'GDP growth rate (annual %, const. 2005 prices)']             = 28.6
df.at[150,'GDP per capita (current US$)']                               = 24500
df.at[150,'Economy: Agriculture (% of GVA)']                            = 1.7
df.at[150,'Economy: Industry (% of GVA)']                               = 58.1
df.at[150,'Economy: Services and other activity (% of GVA)']            = 40.2
df.at[150,'Unemployment (% of labour force)']                           = 11.2
df.at[150,'Employment: Agriculture (% of employed)']                    = 01.9
df.at[150,'Employment: Industry (% of employed)']                       = 10.0
df.at[150,'Employment: Services (% of employed)']                       = 88.1
df.at[150,'Labour force participation (female/male pop. %)']            = '77.8/78.4'
#https://mchb.tvisdata.hrsa.gov/uploadedfiles/StateSubmittedFiles/2018/stateSnapshots/MP_StateSnapshot.pdf'
df.at[150,'Health: Total expenditure (% of GDP)']                       = 0.06
df.at[150,'Health: Physicians (per 1000 pop.)']                         = 0.36
#https://nces.ed.gov/pubs2018/2018301.pdf
df.at[150,'Education: Government expenditure (% of GDP)']               = 7.41
df.at[150,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   = '0.00/0.00'
df.at[150,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] = '0.00/0.00'
df.at[150,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[150,'Seats held by women in national parliaments %']              = 0
df.at[150,'Mobile-cellular subscriptions (per 100 inhabitants)']        = 0
df.at[150,'CO2 emission estimates (million tons/tons per capita)']      = 0 
df.at[150,'Energy production, primary (Petajoules)']                    = 0.0002
df.at[150,'Agricultural production index (2004-2006=100)']              = 0
df.at[150,'Food production index (2004-2006=100)']                      = 0
df.at[150,'Refugees and others of concern to UNHCR (in thousands)']     = 0
df.at[150,'Balance of payments, current account (million US$)']         = 128

#Oman
df.at[152,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '59.68/32.77'
df.at[152,'Threatened species (number)']                                = 66

# Palau
df.at[154,'Unemployment (% of labour force)']                           = 01.7
df.at[154,'Employment: Agriculture (% of employed)']                    = 01.2
df.at[154,'Employment: Industry (% of employed)']                       = 12.4
df.at[154,'Employment: Services (% of employed)']                       = 86.4
df.at[154,'Health: Physicians (per 1000 pop.)']                         = 1.42
df.at[154,'Education: Government expenditure (% of GDP)']               = 7.6
df.at[154,'Mobile-cellular subscriptions (per 100 inhabitants).1']      = 0
df.at[154,'CO2 emission estimates (million tons/tons per capita)']      = 0 
df.at[154,'Agricultural production index (2004-2006=100)']              = 0
df.at[154,'Food production index (2004-2006=100)']                      = 0
df.at[154,'Refugees and others of concern to UNHCR (in thousands)']     = 0.001
df.at[154,'Labour force participation (female/male pop. %)']            = "58.1/100"

#Papua New Guinea
df.at[156,'Health: Physicians (per 1000 pop.)']                         = 0.06
df.at[156,'Education: Government expenditure (% of GDP)']               = 7.39
df.at[156,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '1.35/2.37'

#Paraguay
df.at[157,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '41.24/29.13'

#Peru
df.at[158,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '35.42/33.25'

#Philipines
df.at[159,'Health: Physicians (per 1000 pop.)']                         = 1.1
df.at[159,'Education: Government expenditure (% of GDP)']               = 2.653

# Poland
df.at[160,'Population growth rate (average annual %)']                  = 6.3

#Puerto rico
df.at[162,'GDP growth rate (annual %, const. 2005 prices)']             = -0.6
df.at[162,'International trade: Exports (million US$)']                 = 73160
df.at[162,'International trade: Imports (million US$)']                 = 49010
df.at[162,'International trade: Balance (million US$)']                 = 24150
#https://www.urban.org/sites/default/files/publication/87016/2001051-environmental-scan-of-puerto-ricos-health-care-infrastructure_1.pdf
df.at[162,'Health: Total expenditure (% of GDP)']                       = 17.1
df.at[162,'Health: Physicians (per 1000 pop.)']                         = 1.75
df.at[162,'Seats held by women in national parliaments %']              = 0
df.at[162,'Energy supply per capita (Gigajoules)']                      = '100/100'
df.at[162,'Refugees and others of concern to UNHCR (in thousands)']     = 0
df.at[162,'Balance of payments, current account (million US$)']         = 10526.6

#Romania
df.at[166,'Urban population growth rate (average annual %)']            = 0.0

#Rwanda
df.at[168,'Health: Physicians (per 1000 pop.)']                         = 0.06

# Saint Helena
df.at[169,'Economy: Agriculture (% of GVA)']                            = 0.0
df.at[169,'GDP growth rate (annual %, const. 2005 prices)']             = 16.5
df.at[169,'Unemployment (% of labour force)']                           = 14.0
df.at[169,'Employment: Agriculture (% of employed)']                    = 06.0
df.at[169,'Employment: Industry (% of employed)']                       = 48.0
df.at[169,'Employment: Services (% of employed)']                       = 46.0
df.at[169,'Labour force participation (female/male pop. %)']            = '52.6/64.0'
df.at[169,'International trade: Exports (million US$)']                 = 19
df.at[169,'Life expectancy at birth (females/males, years)']            = '76.7/82.7'
df.at[169,'Population age distribution (0-14 / 60+ years, %)']          = '18.3/11.4'
# https://en.wikipedia.org/wiki/Healthcare_in_Saint_Helena
df.at[169,'Health: Total expenditure (% of GDP)']                       = 6.13
df.at[169,'Health: Physicians (per 1000 pop.)']                         = 0.58
#http://www.sainthelena.gov.sh/wp-content/uploads/2013/01/St-Helena-Government-Budget-Book-201718.pdf
df.at[169,'Education: Government expenditure (% of GDP)']               = 7.71
df.at[169,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[169,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='0.00/0.00'
df.at[169,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[169,'Seats held by women in national parliaments %']              = 0
df.at[169,'Energy supply per capita (Gigajoules)']                      = '00.0/00.0'
df.at[169,'Pop. using improved drinking water (urban/rural, %)']        = '0.00/0.00'
df.at[169,'Agricultural production index (2004-2006=100)']              = 0
df.at[169,'Food production index (2004-2006=100)']                      = 0
df.at[169,'Refugees and others of concern to UNHCR (in thousands)']     = 0
df.at[169,'Balance of payments, current account (million US$)']         = 0
df.at[169,'Fertility rate, total (live births per woman)']              = 1.56

# Saint Kitts and Nevis - no data DROPPING IT
df.drop([170], inplace=True)

#Santa Lucia
df.at[171,'Health: Physicians (per 1000 pop.)']                           = 0.1
df.at[171,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']     ='0.00/0.00'
df.at[171,'Refugees and others of concern to UNHCR (in thousands)']       = 0

# Saint Pierre and Miquelon
df.at[172,'GDP: Gross domestic product (million current US$)']            = 261
df.at[172,'GDP growth rate (annual %, const. 2005 prices)']               = 0.0
df.at[172,'GDP per capita (current US$)']                                 = 46200
df.at[172,'Economy: Agriculture (% of GVA)']                              = 02.0
df.at[172,'Economy: Industry (% of GVA)']                                 = 15.0
df.at[172,'Economy: Services and other activity (% of GVA)']              = 83.0
df.at[172,'Unemployment (% of labour force)']                             = 08.7
df.at[172,'Employment: Agriculture (% of employed)']                      = 18.0
df.at[172,'Employment: Industry (% of employed)']                         = 41.0
df.at[172,'Employment: Services (% of employed)']                         = 41.0
df.at[172,'Labour force participation (female/male pop. %)']              = '00.0/00.0'
df.at[172,'Population growth rate (average annual %)']                    = 0.0
df.at[172,'Fertility rate, total (live births per woman)']                = 1.57
df.at[172,'Life expectancy at birth (females/males, years)']              = '78.3/83.1'
df.at[172,'Population age distribution (0-14 / 60+ years, %)']            = '15.3/20.3'
df.at[172,'Infant mortality rate (per 1000 live births']                  = 6.5
df.at[172,'Health: Total expenditure (% of GDP)']                         = 1.50
df.at[172,'Health: Physicians (per 1000 pop.)']                           = 3.31
#http://www.iedom.fr/IMG/pdf/ra_st-pierre_2016.pdf
df.at[172,'Education: Government expenditure (% of GDP)']                 = 8.72
df.at[172,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']     ='0.00/0.00'
df.at[172,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[172,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']    = '0.00/0.00'
df.at[172,'Seats held by women in national parliaments %']                = 0
df.at[172,'Mobile-cellular subscriptions (per 100 inhabitants)']          = 0
df.at[172,'Mobile-cellular subscriptions (per 100 inhabitants).1']        = 0
df.at[172,'Energy supply per capita (Gigajoules)']                        = '00.0/00.0'
df.at[172,'Pop. using improved drinking water (urban/rural, %)']          = '0.00/0.00'
df.at[172,'Refugees and others of concern to UNHCR (in thousands)']       = 0
df.at[172,'Balance of payments, current account (million US$)']           = 0

#Saint Vicent and Granadine
df.at[173,'Population growth rate (average annual %)']                    = 0.0
df.at[173,'Health: Physicians (per 1000 pop.)']                           = 0.53
df.at[173,'Education: Government expenditure (% of GDP)']                 = 5.1
df.at[173,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']    = '0.00/0.00'
df.at[173,'Pop. using improved drinking water (urban/rural, %)']          = '0.00/0.00'
df.at[173,'Refugees and others of concern to UNHCR (in thousands)']       = 0

#Samoa
df.at[174,'Health: Physicians (per 1000 pop.)']                           = 0.46
df.at[174,'Education: Government expenditure (% of GDP)']                 = 4.081
df.at[174,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']    = '7.23/7.84'
df.at[174,'Refugees and others of concern to UNHCR (in thousands)']       = 0

#San Marino
df.at[175,'Unemployment (% of labour force)']                           = 08.0
df.at[175,'Employment: Agriculture (% of employed)']                    = 00.2
df.at[175,'Employment: Industry (% of employed)']                       = 33.5
df.at[175,'Employment: Services (% of employed)']                       = 66.3
df.at[175,'Labour force participation (female/male pop. %)']            = '53.4/75.2'
df.at[175,'International trade: Exports (million US$)']                 = 3829
df.at[175,'International trade: Imports (million US$)']                 = 2551
df.at[175,'International trade: Balance (million US$)']                 = 1278
df.at[175,'Fertility rate, total (live births per woman)']              = 1.5
df.at[175,'Infant mortality rate (per 1000 live births']                = 4.3
df.at[175,'CO2 emission estimates (million tons/tons per capita)']      = 0 
df.at[175,'Energy production, primary (Petajoules)']                    = 0
df.at[175,'Energy supply per capita (Gigajoules)']                      = '100/100'
df.at[175,'Pop. using improved drinking water (urban/rural, %)']        = '0.00/0.00'
df.at[175,'Agricultural production index (2004-2006=100)']              = 0
df.at[175,'Food production index (2004-2006=100)']                      = 0
df.at[175,'Refugees and others of concern to UNHCR (in thousands)']     = 0
df.at[175,'Balance of payments, current account (million US$)']         = 0

#Sao Tome and Principe
df.at[176,'Health: Physicians (per 1000 pop.)']                          = 0.49
df.at[176,'Refugees and others of concern to UNHCR (in thousands)']      = 0

#Saudi Arabia
df.at[177,'Education: Government expenditure (% of GDP)']                = 5.138
#Senegal
df.at[178,'Health: Physicians (per 1000 pop.)']                          = 0.06

#Seychelles
df.at[180,'Refugees and others of concern to UNHCR (in thousands)']     = 0

#Sierra Leone
df.at[181,'Health: Physicians (per 1000 pop.)']                          = 0.02
df.at[181,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']   ='1.09/2.73'

#Singapoure
df.at[182,'Economy: Agriculture (% of GVA)']                             = 0.03
df.at[182,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']    ='100.69/100.85'
df.at[182,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)']  ='107.65/108.58'
df.at[182,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']   ='3.98/8.89'
df.at[182,'Refugees and others of concern to UNHCR (in thousands)']      = 0.003
df.at[182,'Energy supply per capita (Gigajoules)']                       = '100/0.0'

#Sint Maarten (Dutch part)
df.at[183,'Unemployment (% of labour force)']                           = 12.0
df.at[183,'Employment: Agriculture (% of employed)']                    = 01.1
df.at[183,'Employment: Industry (% of employed)']                       = 15.2
df.at[183,'Employment: Services (% of employed)']                       = 83.7
df.at[183,'Labour force participation (female/male pop. %)']            = '00.0/00.0'
df.at[183,'International trade: Exports (million US$)']                 = 0
df.at[183,'International trade: Imports (million US$)']                 = 0
df.at[183,'International trade: Balance (million US$)']                 = 0
df.at[183,'Fertility rate, total (live births per woman)']              = 2.05
df.at[183,'Infant mortality rate (per 1000 live births']                = 8.1
df.at[183,'Health: Total expenditure (% of GDP)']                       = 2
#https://bearingpointcaribbean.esimg.net/wp-content/uploads/2015/06/cbs_2014_Statistical_Yearbook_2014_Sint-Maarten.pdf
df.at[183,'Health: Physicians (per 1000 pop.)']                         = 0.95
#http://www.sintmaartengov.org/government/Budget/EDU%20%20CULT%206.pdf
df.at[183,'Education: Government expenditure (% of GDP)']               = 11.26
df.at[183,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[183,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='0.00/0.00'
df.at[183,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[183,'Seats held by women in national parliaments %']              = 0
df.at[183,'Mobile-cellular subscriptions (per 100 inhabitants)']        = 0
df.at[183,'Mobile-cellular subscriptions (per 100 inhabitants).1']      = 0
df.at[183,'Threatened species (number)']                                = 34
df.at[183,'CO2 emission estimates (million tons/tons per capita)']      = 0 
df.at[183,'Energy supply per capita (Gigajoules)']                      = '100/100'
df.at[183,'Pop. using improved drinking water (urban/rural, %)']        = '0.00/0.00'
df.at[183,'Agricultural production index (2004-2006=100)']              = 0
df.at[183,'Food production index (2004-2006=100)']                      = 0
df.at[183,'Refugees and others of concern to UNHCR (in thousands)']     = 0

#Solomon Island
df.at[186,'Education: Government expenditure (% of GDP)']               = 7.3
df.at[186,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[186,'Refugees and others of concern to UNHCR (in thousands)']     = 0.003

#Somalia
#http://www.who.int/whr/2004/annex/country/som/en/
df.at[187,'Health: Total expenditure (% of GDP)']                       = 2.1
df.at[187,'Health: Physicians (per 1000 pop.)']                         = 0.03
df.at[187,'Education: Government expenditure (% of GDP)']               = 1.28
df.at[187,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='16.75/30.29'
df.at[187,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='3.75/8.16'
df.at[187,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.95/3.84'
df.at[187,'Balance of payments, current account (million US$)']         = -497

# South Sudan - no data - DROPPING IT
df.drop([189], inplace=True)

#Sri Lanka
df.at[191,'Health: Physicians (per 1000 pop.)']                         = 0.73

#Palestine
#http://www.pcbs.gov.ps/Portals/_pcbs/PressRelease/NHA09-10E.pdf
df.at[192,'Health: Total expenditure (% of GDP)']                       = 13.7
#http://www.pcbs.gov.ps/post.aspx?lang=en&ItemID=1079
df.at[192,'Health: Physicians (per 1000 pop.)']                         = 2.2
df.at[192,'Seats held by women in national parliaments %']              = 0
df.at[192,'Refugees and others of concern to UNHCR (in thousands)']     = 0

#Sudan
df.at[193,'Surface area (km2)']                                         = 1886068
df.at[193,'Education: Government expenditure (% of GDP)']               = 2.219
df.at[193,'Threatened species (number)']                                = 84
df.at[193,'Energy supply per capita (Gigajoules)']                      = '70.20/22.20'
df.at[193,'Pop. using improved drinking water (urban/rural, %)']        = '0.00/0.00'
df.at[193,'Agricultural production index (2004-2006=100)']              = 108.26
df.at[193,'Food production index (2004-2006=100)']                      = 106.86

#Suriname
df.at[194,'Health: Physicians (per 1000 pop.)']                         = 0.91
#https://www.epdc.org/sites/default/files/documents/EPDC%20NEP_Suriname_2.pdf
df.at[194,'Education: Government expenditure (% of GDP)']               = 16.5
df.at[194,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '15.80/9.55'
df.at[194,'Refugees and others of concern to UNHCR (in thousands)']     = 0.001

#Swaziland
df.at[195,'Health: Physicians (per 1000 pop.)']                         = 0.15

#Syria
df.at[198,'Balance of payments, current account (million US$)']         =-367.388
df.at[198,'Education: Government expenditure (% of GDP)']               =5.13

#Thailand
df.at[200,'Health: Physicians (per 1000 pop.)']                         = 0.39

#Macedonia
df.at[201,'Education: Government expenditure (% of GDP)']               = 3.30

#Timor-Leste
df.at[202,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '15.76/21.88'
df.at[202,'Refugees and others of concern to UNHCR (in thousands)']     = 0.001

#Togo
df.at[203,'Health: Physicians (per 1000 pop.)']                         = 0.06
df.at[203,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='30.70/57.86'

#Tokelau NO DATA - DROPPING IT
df.drop([204], inplace=True)

#Tonga
df.at[205,'Health: Physicians (per 1000 pop.)']                         = 0.56
df.at[205,'Education: Government expenditure (% of GDP)']               = 3.91
df.at[205,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '8.00/4.82'
df.at[205,'Refugees and others of concern to UNHCR (in thousands)']     = 0.003

#Trinad and Tobago
df.at[206,'Balance of payments, current account (million US$)']         = 2325
df.at[206,'Health: Physicians (per 1000 pop.)']                         = 1.18
df.at[206,'Education: Government expenditure (% of GDP)']               = 3.137
df.at[206,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='104.30/107.97'
df.at[206,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='88.48/82.61'
df.at[206,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '13.36/10.56'

#Turkmenistan
df.at[209,'Energy supply per capita (Gigajoules)']                      = '100/100'
df.at[209,'Pop. using improved drinking water (urban/rural, %)']        = '85.51/86.78'
df.at[209,'Balance of payments, current account (million US$)']         = -4369

# Turks and Caico Island - NO DATA - DROPPING IT
df.drop([210], inplace=True)

#Tuvalu
df.drop([211], inplace=True)

#Uganda
df.at[212,'Health: Physicians (per 1000 pop.)']                         = 0.12

#United Arabs Emirates
df.at[214,'Education: Government expenditure (% of GDP)']               = 3.3
df.at[214,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='92.98/74.08'
df.at[214,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  ='53.17/26.68'
df.at[214,'Balance of payments, current account (million US$)']         = 17630

#Tanzania
df.at[216,'Health: Physicians (per 1000 pop.)']                         = 0.03

#United States Virgin Islands
df.at[218,'GDP: Gross domestic product (million current US$)']          = 3872
df.at[218,'GDP growth rate (annual %, const. 2005 prices)']             = 0.9
df.at[218,'GDP per capita (current US$)']                               = 37000
df.at[218,'Economy: Agriculture (% of GVA)']                            = 02.0
df.at[218,'Economy: Industry (% of GVA)']                               = 20.0
df.at[218,'Economy: Services and other activity (% of GVA)']            = 78.0
df.at[218,'Labour force participation (female/male pop. %)']            = '00.0/00.0'
df.at[218,'International trade: Exports (million US$)']                 = 2627
df.at[218,'International trade: Imports (million US$)']                 = 2694
df.at[218,'International trade: Balance (million US$)']                 = -67
#https://www.bea.gov/newsreleases/general/terr/2017/vigdp_120117.pdf
df.at[218,'Health: Total expenditure (% of GDP)']                       = 8.63
#http://www.populstat.info/Americas/virgislg.htm
df.at[218,'Health: Physicians (per 1000 pop.)']                         = 1.42
#https://nces.ed.gov/pubs2018/2018301.pdf
df.at[218,'Education: Government expenditure (% of GDP)']               = 4.225
df.at[218,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[218,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='0.00/0.00'
df.at[218,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[218,'Seats held by women in national parliaments %']              = 0
df.at[218,'Mobile-cellular subscriptions (per 100 inhabitants)']        = 74.49
df.at[218,'CO2 emission estimates (million tons/tons per capita)']      = 0 
df.at[218,'Energy production, primary (Petajoules)']                    = 0.0
df.at[218,'Refugees and others of concern to UNHCR (in thousands)']     = 0
df.at[218,'Balance of payments, current account (million US$)']         = 0

#Uruguay
df.at[219,'Health: Physicians (per 1000 pop.)']                         = 3.94
df.at[219,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '57.95/34.50'

#Uzbekistan
#https://www.indexmundi.com/facts/uzbekistan/education-expenditure
df.at[220,'Education: Government expenditure (% of GDP)']               = 9.523
df.at[220,'Balance of payments, current account (million US$)']         = 1774

#Vanuatu
df.at[221,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '3.49/5.94'
df.at[221,'Refugees and others of concern to UNHCR (in thousands)']     = 0.002

#Venezuela
df.at[222,'Health: Physicians (per 1000 pop.)']                         = 1.94
df.at[222,'Education: Government expenditure (% of GDP)']               = 6.87
df.at[222,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '97.68/57.81'

#Wallis and Futuna Islands
df.at[224,'Health: Total expenditure (% of GDP)']                       = 24
df.at[224,'GDP: Gross domestic product (million current US$)']          = 60
df.at[224,'GDP growth rate (annual %, const. 2005 prices)']             = 0.0
df.at[224,'GDP per capita (current US$)']                               = 3800
df.at[224,'Economy: Agriculture (% of GVA)']                            = 0.0
df.at[224,'Economy: Industry (% of GVA)']                               = 0.0
df.at[224,'Economy: Services and other activity (% of GVA)']            = 0.0
df.at[224,'Unemployment (% of labour force)']                           = 08.8
df.at[224,'Employment: Agriculture (% of employed)']                    = 74.4
df.at[224,'Employment: Industry (% of employed)']                       = 03.0
df.at[224,'Employment: Services (% of employed)']                       = 23.0
df.at[224,'Labour force participation (female/male pop. %)']            = '00.0/00.0'
df.at[224,'Infant mortality rate (per 1000 live births']                = 4.3
df.at[224,'Health: Physicians (per 1000 pop.)']                         = 1.1
#http://www.ieom.fr/IMG/pdf/ra2009_wallis.pdf
df.at[224,'Education: Government expenditure (% of GDP)']               = 33.22
df.at[224,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[224,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='0.00/0.00'
df.at[224,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  = '0.00/0.00'
df.at[224,'Seats held by women in national parliaments %']              = 0
df.at[224,'Mobile-cellular subscriptions (per 100 inhabitants)']        = 0
df.at[224,'CO2 emission estimates (million tons/tons per capita)']      = 0 
df.at[224,'Energy supply per capita (Gigajoules)']                      = '00.0/00.0'
df.at[224,'Pop. using improved drinking water (urban/rural, %)']        = '0.00/0.00'
df.at[224,'Refugees and others of concern to UNHCR (in thousands)']     = 0
df.at[224,'Balance of payments, current account (million US$)']         = 0

#Viet Nam
df.at[223,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='54.67/60.92'

#Western Sahara
df.at[225,'Health: Total expenditure (% of GDP)']                        = 0
df.at[225,'GDP: Gross domestic product (million current US$)']          = 906
df.at[225,'GDP growth rate (annual %, const. 2005 prices)']             = 0.0
df.at[225,'GDP per capita (current US$)']                               = 2500
df.at[225,'Economy: Agriculture (% of GVA)']                            = 0.0
df.at[225,'Economy: Industry (% of GVA)']                               = 0.0
df.at[225,'Economy: Services and other activity (% of GVA)']            = 40.0
df.at[225,'International trade: Imports (million US$)']                 = 0.722
df.at[225,'International trade: Exports (million US$)']                 = 0
df.at[225,'International trade: Balance (million US$)']                 = -0.722
df.at[225,'Health: Physicians (per 1000 pop.)']                         = 0.44
df.at[225,'Education: Government expenditure (% of GDP)']               = 0
df.at[225,'Education: Primary gross enrol. ratio (f/m per 100 pop.)']   ='0.00/0.00'
df.at[225,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='0.00/0.00'
df.at[225,'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']  ='0.00/0.00'
df.at[225,'Seats held by women in national parliaments %']              = 0
df.at[225,'Mobile-cellular subscriptions (per 100 inhabitants)']        = 0 
df.at[225,'Mobile-cellular subscriptions (per 100 inhabitants).1']      = 0
df.at[225,'CO2 emission estimates (million tons/tons per capita)']      = 0 
df.at[225,'Energy production, primary (Petajoules)']                    = 0
df.at[225,'Energy supply per capita (Gigajoules)']                      = '00.0/00.0'
df.at[225,'Pop. using improved drinking water (urban/rural, %)']        = '0.00/0.00'
df.at[225,'Refugees and others of concern to UNHCR (in thousands)']     = 0
df.at[225,'Balance of payments, current account (million US$)']         = 0

#Yemen
df.at[226,'Education: Government expenditure (% of GDP)']               = 5.15

#Zambia
df.at[227,'Education: Government expenditure (% of GDP)']               = 1.1
df.at[227,'Education: Secondary gross enrol. ratio (f/m per 100 pop.)'] ='14.75/24.92'
df.head()
df.drop(df.columns[-3:], axis=1, inplace=True)
df.head()
df.drop(columns=['Forested area (% of land area)'], inplace=True)
df.head()
df.info()
df = df.apply(pd.to_numeric, errors='ignore')
df.info()
region = pd.get_dummies(df['Region'])
region.head()
df.drop(columns=['Region'], inplace=True)
df = pd.concat([df,region], axis=1)
df.head()
df[['country','Life expectancy at birth (females/males, years)']].head()
life_exp = df['Life expectancy at birth (females/males, years)']
life_exp = life_exp.str.split("/", expand=True)
life_exp.columns = ['Life exp. Male','Life exp. Female']
life_exp.head()
labour_force             = df['Labour force participation (female/male pop. %)']
labour_force             = labour_force.str.split("/", expand=True)
labour_force.columns     = ['Labour Force: Male','Labour Force: Female']

age_distribution         = df['Population age distribution (0-14 / 60+ years, %)']
age_distribution         = age_distribution.str.split("/", expand=True)
age_distribution.columns = ['Age dist.: Male','Age dist.: Female']

migrants                 = df['International migrant stock (000/% of total pop.)']
migrants                 = migrants.str.split("/", expand=True)
migrants.columns         = ["Migrants: Total","Migrants: % pop."]

edu_prim                 = df['Education: Primary gross enrol. ratio (f/m per 100 pop.)']
edu_prim                 = edu_prim.str.split("/", expand=True)
edu_prim.columns         = ['Primary Edu.Enroll: Female','Primary Edu. Enroll: Male']

edu_sec                  = df['Education: Secondary gross enrol. ratio (f/m per 100 pop.)']
edu_sec                  = edu_sec.str.split("/", expand=True)
edu_sec.columns          = ['Sec. Edu.Enroll: Female','Sec. Edu. Enroll: Male']

edu_ter                  = df['Education: Tertiary gross enrol. ratio (f/m per 100 pop.)']
edu_ter                  = edu_ter.str.split("/", expand=True)
edu_ter.columns          = ['Ter. Edu.Enroll: Female','Ter. Edu. Enroll: Male']

energy                   = df['Energy supply per capita (Gigajoules)']
energy                   = energy.str.split("/", expand=True)
energy.columns           = ['Energy Supply:Urban','Energy Supply:Rural']


labour_force
df.drop(columns=['Life expectancy at birth (females/males, years)',
                'Labour force participation (female/male pop. %)',
                'Population age distribution (0-14 / 60+ years, %)',
                'International migrant stock (000/% of total pop.)',
                'Education: Primary gross enrol. ratio (f/m per 100 pop.)',
                'Education: Secondary gross enrol. ratio (f/m per 100 pop.)',
                'Education: Tertiary gross enrol. ratio (f/m per 100 pop.)',
                'Energy supply per capita (Gigajoules)'], inplace=True)
df = pd.concat([df,labour_force,age_distribution,migrants,edu_prim,edu_sec,edu_ter,energy], axis=1)
df.head()
df = df.apply(pd.to_numeric, errors='ignore')
df.info()
from sklearn.ensemble import RandomForestRegressor
black_box_model = RandomForestRegressor()
gdp_per_capita = df['GDP per capita (current US$)']
df.drop(columns=['country','GDP: Gross domestic product (million current US$)','GDP per capita (current US$)'], inplace=True)

df.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, gdp_per_capita, test_size=0.30, random_state=42)



black_box_model.fit(X_train,y_train)
black_box_model.feature_importances_
def mape_mspe(model,X_train,X_test,y_train,y_test,n_runs=10):
    mape = 0
    mspe = 0
    for i in range(n_runs):
        model.fit(X_train,y_train)
        pred = model.predict(X_test)
        pred = pd.DataFrame(pred,index=y_test.index)
        pred.columns =['Predictions']
        bench = pd.concat([pred,y_test],axis=1)
        bench['% of error'] = (bench['Predictions'] - bench[bench.columns[1]])/(bench[bench.columns[1]]/100)
        mape += bench['% of error'].abs().mean()
        mspe += (bench['% of error']**2).mean()
    mape /= n_runs
    mspe /= n_runs
    return mape,mspe
random_forest_macro_mae,random_forest_macro_mse = mape_mspe(black_box_model,X_train,X_test,y_train,y_test)
print('Random Forest - Macro Economics')
print('Mean Absolute Error:')
print(random_forest_macro_mae)
print('Mean Squared Error:')
print(random_forest_macro_mse)
freedom = pd.read_excel('../input/heritage-freedom-index/index2018_data.xls')
freedom.head()
freedom.drop(columns=['CountryID',
                 'Country Name',
                 'WEBNAME',
                 'Country',
                 'GDP (Billions, PPP)'], inplace=True)

freedom.head()
freedom.dropna(inplace=True)
GDP_freedom = freedom['GDP per Capita (PPP)']
freedom.drop(columns=['GDP per Capita (PPP)'], inplace=True)

freedom
freedom.info()
freedom = freedom.apply(pd.to_numeric, errors='ignore')
freedom.info()
GDP_freedom = GDP_freedom.apply(pd.to_numeric, errors='ignore')
freedom_region = pd.get_dummies(freedom['Region'])
freedom_region.head()
freedom.drop(columns=['Region'], inplace=True)
freedom = pd.concat([freedom,freedom_region], axis=1)
freedom.head()
from sklearn.ensemble import RandomForestRegressor
freedom_bbm = RandomForestRegressor()
from sklearn.model_selection import train_test_split
free_X_train, free_X_test, free_y_train, free_y_test = train_test_split(freedom, GDP_freedom, test_size=0.30, random_state=42)

freedom_bbm.fit(free_X_train,free_y_train)
importance_free_bbm = freedom_bbm.feature_importances_
importance_free_bbm = pd.DataFrame(importance_free_bbm)
importance_free_bbm.index = freedom.columns
importance_free_bbm.columns = ['Features']
importance_free_bbm

importance_free_bbm.describe()
importance_free_bbm = importance_free_bbm.sort_values(by='Features',ascending=False)
import seaborn as sns
fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)
sns.barplot(x='Features',y=importance_free_bbm.index,data=importance_free_bbm)
gov_size    =   importance_free_bbm.loc["Gov't Expenditure % of GDP "]+ \
                importance_free_bbm.loc['Tax Burden % of GDP']        + \
                importance_free_bbm.loc['Fiscal Health']      
                
reg_efec    =   importance_free_bbm.loc['Business Freedom']           + \
                importance_free_bbm.loc['Labor Freedom']              + \
                importance_free_bbm.loc['Monetary Freedom']          
        
open_market =   importance_free_bbm.loc['Trade Freedom']              + \
                importance_free_bbm.loc['Investment Freedom ']        + \
                importance_free_bbm.loc['Financial Freedom']    
        
rule_of_law =   importance_free_bbm.loc['Property Rights']            + \
                importance_free_bbm.loc['Judical Effectiveness']      + \
                importance_free_bbm.loc['Government Integrity']        
importance_free_bbm.loc['Rule of Law']           = rule_of_law
importance_free_bbm.loc['Government Size']       = gov_size
importance_free_bbm.loc['Regulatory Efficiency'] = reg_efec
importance_free_bbm.loc['Open Markets']          = open_market

importance_free_bbm
liberty = reg_efec + open_market
liberty
importance_free_bbm.loc['Liberty']           = rule_of_law
importance_free_bbm = importance_free_bbm.sort_values(by='Features',ascending=False)
fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)
sns.barplot(x='Features',y=importance_free_bbm.index,data=importance_free_bbm)
plt.title('Feature Importance')
fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)
sns.heatmap(pd.concat([freedom,GDP_freedom],axis=1).corr(),  linewidths=.5)
plt.title('Features Correlation')
random_forest_freedom_mae,random_forest_freedom_mse = mape_mspe(freedom_bbm,free_X_train,free_X_test,free_y_train,free_y_test)
print('Random Forest - Economic Freedom Index')
print('Mean Absolute Error:')
print(random_forest_freedom_mae)
print('Mean Squared Error:')
print(random_forest_freedom_mse)
from sklearn import decomposition
from sklearn.feature_selection import SelectKBest,mutual_info_regression

pca = decomposition.PCA(n_components=6)
pca.fit(X_train)
pca_X = pca.transform(X_train)
pca_X_test = pca.transform(X_test)

freedom_pca = decomposition.PCA(n_components=6)
freedom_pca.fit(free_X_train)
freedom_pca_X = freedom_pca.transform(free_X_train)
freedom_pca_X_test = freedom_pca.transform(free_X_test)
random_forest_macro_pca_mae,random_forest_macro_pca_mse = mape_mspe(black_box_model,pca_X,pca_X_test,y_train,y_test)
print('Random Forest - MacroEconomics Features - PCA')
print('Mean Absolute Error:')    
print(random_forest_macro_pca_mae) 
print('Mean Squared Error:')
print(random_forest_macro_pca_mse)
random_forest_freedom_pca_mae,random_forest_freedom_pca_mse = mape_mspe(freedom_bbm,freedom_pca_X,freedom_pca_X_test,free_y_train,free_y_test)
print('Random Forest - Freedom Index - PCA')
print('Mean Absolute Error')
print(random_forest_freedom_pca_mae)
print('Mean Squared Error')
print(random_forest_freedom_pca_mse)
kbest = SelectKBest(k=10,score_func=mutual_info_regression)
kbest.fit(X_train,y_train)
kbest_X = kbest.transform(X_train)
kbest_X_test = kbest.transform(X_test)

freedom_kbest = SelectKBest(k=10,score_func=mutual_info_regression)
freedom_kbest.fit(free_X_train,free_y_train)
freedom_kbest_X = freedom_kbest.transform(free_X_train)
freedom_kbest_X_test = freedom_kbest.transform(free_X_test)
random_forest_macro_kbest_mae,random_forest_macro_kbest_mse = mape_mspe(black_box_model,kbest_X,kbest_X_test,y_train,y_test)
print('Random Forest - MacroEconomics - KBest')
print('Mean Absolute Error')
print(random_forest_macro_kbest_mae)
print('Mean Squared Error')
print(random_forest_macro_kbest_mse)
random_forest_freedom_kbest_mae,random_forest_freedom_kbest_mse = mape_mspe(freedom_bbm,freedom_kbest_X,freedom_kbest_X_test,free_y_train,free_y_test)
print('Random Forest - Freedom Index - kBest')
print('Mean Absolute Error')
print(random_forest_freedom_kbest_mae)
print('Mean Squared Error')
print(random_forest_freedom_kbest_mse)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
tree = DecisionTreeRegressor()
ada_boost = AdaBoostRegressor()
tree_macro_mae,tree_macro_mse = mape_mspe(tree,X_train,X_test,y_train,y_test)
print('Decision Tree Black Box Model - Macro Economics')
print('Mean Absolute Error')
print(tree_macro_mae)
print('Mean Squared Error')
print(tree_macro_mse)
tree_macro_kbest_mae,tree_macro_kbest_mse = mape_mspe(tree,kbest_X,kbest_X_test,y_train,y_test)
print('Decision Tree k-Best Model')
print('Mean Absolute Error')
print(tree_macro_kbest_mae)
print('Mean Squared Error')
print(tree_macro_kbest_mse)
tree_macro_pca_mae,tree_macro_pca_mse = mape_mspe(tree,pca_X,pca_X_test,y_train,y_test)
print('Decision Tree PCA Model')
print('Mean Absolute Error')
print(tree_macro_pca_mae)
print('Mean Squared Error')
print(tree_macro_pca_mse)  
tree_freedom_mae,tree_freedom_mse = mape_mspe(tree,free_X_train,free_X_test,free_y_train,free_y_test)
print('Decision Tree Black Box Model - Freedom Index')
print('Mean Absolute Error')
print(tree_freedom_mae)
print('Mean Squared Error')
print(tree_freedom_mse)
tree_freedom_kbest_mae,tree_freedom_kbest_mse = mape_mspe(tree,freedom_kbest_X,freedom_kbest_X_test,free_y_train,free_y_test)
print('Decision Tree K-Best Model')
print('Mean Absolute Error')
print(tree_freedom_kbest_mae)
print('Mean Squared Error')
print(tree_freedom_kbest_mse)
tree_freedom_pca_mae,tree_freedom_pca_mse = mape_mspe(tree,freedom_pca_X,freedom_pca_X_test,free_y_train,free_y_test)
print('Decision Tree PCA Model')
print('Mean Absolute Error')
print(tree_freedom_pca_mae)
print('Mean Squared Error')
print(tree_freedom_pca_mse)
ada_macro_mae,ada_macro_mse = mape_mspe(ada_boost,X_train,X_test,y_train,y_test)
print('AdaBoost Black Box Model')
print('Mean Absolute Error')
print(ada_macro_mae)
print('Mean Squared Error')
print(ada_macro_mse)
ada_macro_kbest_mae,ada_macro_kbest_mse = mape_mspe(ada_boost,kbest_X,kbest_X_test,y_train,y_test)
print('AdaBoost k-Best Model')
print('Mean Absolute Error')
print(ada_macro_kbest_mae)
print('Mean Squared Error')
print(ada_macro_kbest_mse)
ada_macro_pca_mae,ada_macro_pca_mse = mape_mspe(ada_boost,pca_X,pca_X_test,y_train,y_test)
print('AdaBoost PCA Model')
print('Mean Absolute Error')
print(ada_macro_pca_mae)
print('Mean Squared Error')
print(ada_macro_pca_mse)
ada_freedom_mae,ada_freedom_mse = mape_mspe(ada_boost,free_X_train,free_X_test,free_y_train,free_y_test)
print('AdaBoost Black Box Model')
print('Mean Absolute Error')
print(ada_freedom_mae)
print('Mean Squared Error')
print(ada_freedom_mse)
ada_freedom_kbest_mae,ada_freedom_kbest_mse = mape_mspe(ada_boost,freedom_kbest_X,freedom_kbest_X_test,free_y_train,free_y_test)
print('AdaBoost k-Best Box Model')
print('Mean Absolute Error')
print(ada_freedom_kbest_mae)
print('Mean Squared Error')
print(ada_freedom_kbest_mse)
ada_freedom_pca_mae,ada_freedom_pca_mse = mape_mspe(ada_boost,freedom_pca_X,freedom_pca_X_test,free_y_train,free_y_test)
print('AdaBoost PCA Box Model')
print('Mean Absolute Error')
print(ada_freedom_pca_mae)
print('Mean Squared Error')
print(ada_freedom_pca_mse)
benchmark_dict = {'Mean Absolute Error':[random_forest_macro_mae,
                                     random_forest_macro_kbest_mae,
                                     random_forest_macro_pca_mae,
                                     random_forest_freedom_mae,
                                     random_forest_freedom_kbest_mae,
                                     random_forest_freedom_pca_mae,
                                     tree_macro_mae,
                                     tree_macro_kbest_mae,
                                     tree_macro_pca_mae,
                                     tree_freedom_mae,
                                     tree_freedom_kbest_mae,
                                     tree_freedom_pca_mae,
                                     ada_macro_mae,
                                     ada_macro_kbest_mae,
                                     ada_macro_pca_mae,
                                     ada_freedom_mae,
                                     ada_freedom_kbest_mae,
                                     ada_freedom_pca_mae],
                  'Mean Squared Error':[random_forest_macro_mse,
                                     random_forest_macro_kbest_mse,
                                     random_forest_macro_pca_mse,
                                     random_forest_freedom_mse,
                                     random_forest_freedom_kbest_mse,
                                     random_forest_freedom_pca_mse,
                                     tree_macro_mse,
                                     tree_macro_kbest_mse,
                                     tree_macro_pca_mse,
                                     tree_freedom_mse,
                                     tree_freedom_kbest_mse,
                                     tree_freedom_pca_mse,
                                     ada_macro_mse,
                                     ada_macro_kbest_mse,
                                     ada_macro_pca_mse,
                                     ada_freedom_mse,
                                     ada_freedom_kbest_mse,
                                     ada_freedom_pca_mse]
                 }
benchmark_data_frame = pd.DataFrame(data=benchmark_dict,
                                    index =['Random Forest - Macro',
                                            'Random Forest - Macro - KBest',
                                            'Random Forest - Macro - PCA',
                                            'Random Forest - Freedom',
                                            'Random Forest - Freedom - KBest',
                                            'Random Forest - Freedom - PCA',
                                            'Decision Tree - Macro',
                                            'Decision Tree - Macro - KBest',
                                            'Decision Tree - Macro - PCA',
                                            'Decision Tree - Freedom',
                                            'Decision Tree - Freedom - KBest',
                                            'Decision Tree - Freedom - PCA',
                                            'AdaBoost - Macro',
                                            'AdaBoost - Macro - KBest',
                                            'AdaBoost - Macro - PCA',
                                            'AdaBoost - Freedom',
                                            'AdaBoost - Freedom - KBest',
                                            'AdaBoost - Freedom - PCA']
                                    )
benchmark_data_frame
fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)
sns.barplot(y= benchmark_data_frame.index,
            x = 'Mean Absolute Error',
            data=benchmark_data_frame)
plt.title('Models Mean Absolute Error')
fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)
sns.barplot(y= benchmark_data_frame.index,
            x = 'Mean Squared Error',
            data=benchmark_data_frame)
plt.title('Models Squared Absolute Error')
from sklearn.model_selection import GridSearchCV
random_forest_parameters = {'n_estimators':[5,10,15,20,25],'max_depth':[None,4,6,8,10],'min_samples_split':[2,3,4,5]}
macro_regressor   = GridSearchCV(black_box_model, random_forest_parameters)
freedom_regressor = GridSearchCV(black_box_model, random_forest_parameters)
#Scikit-learn GridSearchCV do the Cross Validation
#Selecting Kbest Features
macro_kbest = SelectKBest(k=10,score_func=mutual_info_regression)
macro_kbest.fit(df,gdp_per_capita)
macro_kbest_X = macro_kbest.transform(df)
macro_kbest_X_test = macro_kbest.transform(X_test)
#Grid Seach of RandomForest Macro economics Features Model
macro_regressor.fit(macro_kbest_X,gdp_per_capita)
#PCA Transformation
freedom_pca = decomposition.PCA(n_components=6)
freedom_pca.fit(freedom)
freedom_pca_X = freedom_pca.transform(freedom)
freedom_pca_X_test = freedom_pca.transform(free_X_test)
#Grid Seach of RandomForest Freedom Index Model
freedom_regressor.fit(freedom_pca_X,GDP_freedom)

macro_regressor.get_params()
freedom_regressor.get_params()
macro_predictions = macro_regressor.predict(macro_kbest_X_test)
macro_predictions = pd.DataFrame(macro_predictions,index=y_test.index)
macro_predictions.columns = ['Predictions']
macro_benchmark = pd.concat([macro_predictions,y_test],axis=1)
macro_benchmark['% of error'] = (macro_benchmark['Predictions'] - macro_benchmark['GDP per capita (current US$)'])/(macro_benchmark['GDP per capita (current US$)']/100)
print('Random Forest Macro Economics Features Model')
print('Mean Absolute Error')
freedom_mae = macro_benchmark['% of error'].abs().mean()
print(freedom_mae)
print('Mean Squared Error')
freedom_mse = (macro_benchmark['% of error']**2).mean()
print((macro_benchmark['% of error']**2).mean())
freedom_predictions = freedom_regressor.predict(freedom_pca_X_test)
freedom_predictions = pd.DataFrame(macro_predictions,index=free_y_test.index)
freedom_predictions.columns = ['Predictions']
freedom_benchmark = pd.concat([freedom_predictions,free_y_test],axis=1)
freedom_benchmark['% of error'] = (freedom_benchmark['Predictions'] - freedom_benchmark['GDP per Capita (PPP)'])/(freedom_benchmark['GDP per Capita (PPP)']/100)
print('RandomForest Economic Freedom Features Model')
print('Mean Absolute Error')
freedom_mae = freedom_benchmark['% of error'].abs().mean()
print(freedom_mae)
print('Mean Squared Error')
freedom_mse = (freedom_benchmark['% of error']**2).mean()
print((freedom_benchmark['% of error']**2).mean())
#Loading 
freedom_13 = pd.read_excel('../input/heritage-freedom-index/index2013_data.xls')
freedom_14 = pd.read_excel('../input/heritage-freedom-index/index2014_data.xls')
freedom_15 = pd.read_excel('../input/heritage-freedom-index/index2015_data.xls')
freedom_16 = pd.read_excel('../input/heritage-freedom-index/index2016_data.xls')
freedom_17 = pd.read_excel('../input/heritage-freedom-index/index2017_data.xls')

freedom_13.head()
freedom_14.head()
freedom_15.head()
freedom_16.head()
freedom_17.head()
columns = [0,1,2,7,9,11,13,15,17,19,21,23,25,27,33,35]
freedom_13.drop(freedom_13.columns[columns], axis=1, inplace=True)
freedom_14.drop(freedom_14.columns[columns], axis=1, inplace=True)
freedom_15.drop(freedom_15.columns[columns], axis=1, inplace=True)
freedom_16.drop(freedom_16.columns[columns], axis=1, inplace=True)
freedom_17.drop(freedom_17.columns[[0,1,2,24,26]], axis = 1, inplace=True)
freedom_17.drop('Judical Effectiveness', axis=1, inplace=True)
freedom.drop('Judical Effectiveness', axis=1, inplace=True)
freedom_17.drop('Tax Burden', axis=1, inplace=True)
freedom.drop('Tax Burden', axis=1, inplace=True)
freedom.rename(columns={'Government Integrity': 'Freedom from Corruption', 'Fiscal Health': 'Fiscal Freedom '}, inplace=True)
freedom_17.rename(columns={'Government Integrity': 'Freedom from Corruption', 'Fiscal Health': 'Fiscal Freedom '}, inplace=True)
freedom_13.rename(columns={'2013 Score': 'Score'}, inplace=True)
freedom_14.rename(columns={'2014 Score': 'Score'}, inplace=True)
freedom_15.rename(columns={'2015 Score': 'Score'}, inplace=True)
freedom_16.rename(columns={'2016 Score': 'Score'}, inplace=True)
freedom_17.rename(columns={'2017 Score': 'Score'}, inplace=True)
freedom.rename(columns={'2018 Score': 'Score'}, inplace=True)
freedom_index = pd.concat([freedom_13,freedom_14,freedom_15,freedom_16,freedom_17],axis=0)
#GDP_freedom_index = freedom_index['GDP per Capita (PPP)']
#freedom_index.drop('GDP per Capita (PPP)', axis=1, inplace=True)
freedom_index_region = pd.get_dummies(freedom_index['Region'])
freedom_index = pd.concat([freedom_index,freedom_index_region], axis=1)
freedom_index
freedom_index.drop('Region', axis=1, inplace=True)
freedom_index.info()
freedom_index.dropna(inplace=True)
freedom_index = freedom_index.apply(pd.to_numeric, errors='ignore')
freedom_index.info()
GDP_freedom_index = freedom_index['GDP per Capita (PPP)']
freedom_index.drop('GDP per Capita (PPP)', axis=1, inplace=True)
freedom_index_X_train, freedom_index_X_test, freedom_index_y_train, freedom_index_y_test = train_test_split(freedom_index, GDP_freedom_index, test_size=0.33, random_state=42)
freedom_index_mae,freedom_index_mse = mape_mspe(tree,freedom_index_X_train,freedom_index_X_test,freedom_index_y_train,freedom_index_y_test)
print('Decision Tree - Economic Freedom Index')
print('Mean Absolute Error:')
print(freedom_index_mae)
print('Mean Squared Error:')
print(freedom_index_mse)
tree_parameters = {'max_depth':[None,4,6,8,10,15,20,25],'min_samples_split':[2,3,4,5]}
index_kbest_regressor   = GridSearchCV(tree, tree_parameters)
index_pca_regressor = GridSearchCV(tree, tree_parameters)
index_regressor = GridSearchCV(tree, tree_parameters)
#Scikit-learn GridSearchCV do the Cross Validation
#GridSearch
index_regressor.fit(freedom_index,GDP_freedom_index)
#Selecting Kbest Features
index_kbest = SelectKBest(k=10,score_func=mutual_info_regression)
index_kbest.fit(freedom_index,GDP_freedom_index)
index_kbest_X = index_kbest.transform(freedom_index)
index_kbest_X_test = index_kbest.transform(freedom_index_X_test)
#Grid Seach
index_kbest_regressor.fit(index_kbest_X,GDP_freedom_index)
#PCA Transformation
index_pca = decomposition.PCA(n_components=6)
index_pca.fit(freedom_index)
index_pca_X = freedom_pca.transform(freedom_index)
index_pca_X_test = freedom_pca.transform(freedom_index_X_test)
#Grid Seach
index_pca_regressor.fit(index_pca_X,GDP_freedom_index)
index_predictions = index_regressor.predict(freedom_index_X_test)
index_predictions = pd.DataFrame(index_predictions,index=freedom_index_y_test.index)
index_predictions.columns = ['Predictions']
index_benchmark = pd.concat([index_predictions,freedom_index_y_test],axis=1)
index_benchmark['% of error'] = (index_benchmark['Predictions'] - index_benchmark['GDP per Capita (PPP)'])/(index_benchmark['GDP per Capita (PPP)']/100)
print('Decision Tree Freedom Index Model')
print('Mean Absolute Error')
freedom_index_mae = index_benchmark['% of error'].abs().mean()
print(freedom_index_mae)
print('Mean Squared Error')
freedom_index_mse = (index_benchmark['% of error']**2).mean()
print(freedom_index_mse)
index_kbest_predictions = index_kbest_regressor.predict(index_kbest_X_test)
index_kbest_predictions = pd.DataFrame(index_kbest_predictions,index=freedom_index_y_test.index)
index_kbest_predictions.columns = ['Predictions']
index_kbest_benchmark = pd.concat([index_kbest_predictions,freedom_index_y_test],axis=1)
index_kbest_benchmark['% of error'] = (index_kbest_benchmark['Predictions'] - index_kbest_benchmark['GDP per Capita (PPP)'])/(index_kbest_benchmark['GDP per Capita (PPP)']/100)
print('Decision Tree Freedom Index Model')
print('Mean Absolute Error')
freedom_index_kbest_mae = index_kbest_benchmark['% of error'].abs().mean()
print(freedom_index_kbest_mae)
print('Mean Squared Error')
freedom_index_kbest_mse = (index_kbest_benchmark['% of error']**2).mean()
print(freedom_index_kbest_mse)
index_pca_predictions = index_pca_regressor.predict(index_pca_X_test)
index_pca_predictions = pd.DataFrame(index_pca_predictions,index=freedom_index_y_test.index)
index_pca_predictions.columns = ['Predictions']
index_pca_benchmark = pd.concat([index_pca_predictions,freedom_index_y_test],axis=1)
index_pca_benchmark['% of error'] = (index_pca_benchmark['Predictions'] - index_pca_benchmark['GDP per Capita (PPP)'])/(index_pca_benchmark['GDP per Capita (PPP)']/100)
print('Decision Tree Freedom Index Model - PCA')
print('Mean Absolute Error')
freedom_index_pca_mae = index_pca_benchmark['% of error'].abs().mean()
print(freedom_index_pca_mae)
print('Mean Squared Error')
freedom_index_pca_mse = (index_pca_benchmark['% of error']**2).mean()
print(freedom_index_pca_mse)
index_regressor.best_params_

index_regressor.best_estimator_
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
freedom_index_scaled = scaler.fit_transform(freedom_index)
freedom_index_scaled = pd.DataFrame(freedom_index_scaled,columns=freedom_index.columns)
freedom_index_scaled.head()
target_scaler = MinMaxScaler()
scaled_GDP = target_scaler.fit_transform(pd.DataFrame(GDP_freedom_index))
index_regressor_scaled = GridSearchCV(tree, tree_parameters)
index_regressor_scaled.fit(freedom_index_scaled,scaled_GDP)
freedom_index_scaled_X_test = scaler.transform(freedom_index_X_test)
freedom_index_scaled_y_test = pd.DataFrame(target_scaler.transform(pd.DataFrame(freedom_index_y_test)),
                                           index=freedom_index_y_test.index,
                                           columns=['GDP per Capita (PPP)'])
index_scaled_predictions = index_regressor_scaled.predict(freedom_index_scaled_X_test)
index_scaled_predictions = pd.DataFrame(index_scaled_predictions,index=freedom_index_y_test.index)
index_scaled_predictions.columns = ['Predictions']
index_scaled_benchmark = pd.concat([index_scaled_predictions,freedom_index_scaled_y_test],axis=1)
index_scaled_benchmark['% of error'] = (index_scaled_benchmark['Predictions'] - index_scaled_benchmark['GDP per Capita (PPP)'])/(index_scaled_benchmark['GDP per Capita (PPP)']/100)
print('Decision Tree Freedom Index Model')
print('Mean Absolute Error')
freedom_index_scaled_mae = index_scaled_benchmark['% of error'].abs().mean()
print(freedom_index_scaled_mae)
print('Mean Squared Error')
freedom_index_scaled_mse = (index_scaled_benchmark['% of error']**2).mean()
print(freedom_index_scaled_mse)
index_scaled_benchmark
index_scaled_benchmark.describe()
fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)
sns.distplot(index_scaled_benchmark['% of error'], bins=20)
plt.title('Percentage of Error Distribution')
fig, ax = plt.subplots()
# the size of A4 paper
fig.set_size_inches(11.7, 8.27)
ax = sns.boxplot(data=freedom_index_scaled, orient="h")
plt.title('Economic Freedom Index')
