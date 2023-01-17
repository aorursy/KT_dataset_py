import pandas as pd
from pandas import DataFrame 
import numpy as np
mobility=DataFrame(pd.read_csv('../input/globalmobilityreports/GlobalMobilityReports.csv'))
mobility.head()
mobility.drop(['sub_region_1','sub_region_2'],axis=1)
finland=mobility[mobility['country_region_code']=='FI']
finland=finland.reset_index().drop('index',axis=1)
# print(finland)
NewZ=mobility[mobility['country_region']=='New Zealand']
NewZ=NewZ.reset_index().drop('index',axis=1)
# print(NewZ)
Aus=mobility[mobility['country_region']=='Australia']
Aus=Aus.reset_index().drop('index',axis=1)
# print(Aus)
Tai=mobility[mobility['country_region']=='Taiwan']
Tai=Tai.reset_index().drop('index',axis=1)
# print(Tai)
lowest=['Finland','New Zealand','Australia','Taiwan']
highest=['USA','Italy','Spain','China']

us=mobility[mobility['country_region']=='United States']
us=us.reset_index().drop('index',axis=1)
# print(us)

italy=mobility[mobility['country_region']=='Italy']
italy=italy.reset_index().drop('index',axis=1)
# print(italy)

spain=mobility[mobility['country_region']=='Spain']
spain=spain.reset_index().drop('index',axis=1)
# print(spain)

fran=mobility[mobility['country_region']=='France']
fran=fran.reset_index().drop('index',axis=1)
# print(fran)

# print(mobility['date'])


findates=list(finland['date'])

newzdates=list(NewZ['date'])

ausdates=list(Aus['date'])

taidates=list(Tai['date'])

usdates=list(us['date'])

itdates=list(italy['date'])

spdates=list(spain['date'])

frdates=list(fran['date'])


#Retail and Recreation
from matplotlib import pyplot as plt
plt.style.use('seaborn')
plt.plot_date(findates,list(finland['retail_and_recreation_percent_change_from_baseline']),label='Finland')
plt.plot_date(newzdates,list(NewZ['retail_and_recreation_percent_change_from_baseline']),label='NewZealand')
plt.plot_date(ausdates,list(Aus['retail_and_recreation_percent_change_from_baseline']),label='Australia')
plt.plot_date(taidates,list(Tai['retail_and_recreation_percent_change_from_baseline']),label='Taiwan')
plt.xlabel('Date')
plt.ylabel('retail_and_recreation_percent_change_from_baseline')
plt.title('Retail mobility for countries least impacted')
plt.tight_layout()
plt.legend()
plt.show()
plt.plot_date(usdates,list(us['retail_and_recreation_percent_change_from_baseline']),label='USA')
plt.plot_date(itdates,list(italy['retail_and_recreation_percent_change_from_baseline']),label='Italy')
plt.plot_date(spdates,list(spain['retail_and_recreation_percent_change_from_baseline']),label='Spain')
plt.plot_date(frdates,list(fran['retail_and_recreation_percent_change_from_baseline']),label='France')
plt.xlabel('Date')
plt.ylabel('retail_and_recreation_percent_change_from_baseline')
plt.title('Retail and recreation mobility for countries most impacted')
plt.tight_layout()
plt.legend()
plt.show()


# grocery_and_pharmacy_percent_change_from_baseline
plt.style.use('seaborn')
plt.plot_date(findates,list(finland['grocery_and_pharmacy_percent_change_from_baseline']),label='Finland')
plt.plot_date(newzdates,list(NewZ['grocery_and_pharmacy_percent_change_from_baseline']),label='NewZealand')
plt.plot_date(ausdates,list(Aus['grocery_and_pharmacy_percent_change_from_baseline']),label='Australia')
plt.plot_date(taidates,list(Tai['grocery_and_pharmacy_percent_change_from_baseline']),label='Taiwan')
plt.xlabel('Date')
plt.ylabel('grocery_and_pharmacy_percent_change_from_baseline')
plt.title('Grocery and Pharmacy mobility for countries least impacted')
plt.tight_layout()
plt.legend()
plt.show()
plt.plot_date(usdates,list(us['grocery_and_pharmacy_percent_change_from_baseline']),label='USA')
plt.plot_date(itdates,list(italy['grocery_and_pharmacy_percent_change_from_baseline']),label='Italy')
plt.plot_date(spdates,list(spain['grocery_and_pharmacy_percent_change_from_baseline']),label='Spain')
plt.plot_date(frdates,list(fran['grocery_and_pharmacy_percent_change_from_baseline']),label='France')
plt.xlabel('Date')
plt.ylabel('grocery_and_pharmacy_percent_change_from_baseline')
plt.title('Grocery and Pharmacy mobility for countries most impacted')
plt.tight_layout()
plt.legend()
plt.show()
# parks_percent_change_from_baseline
plt.style.use('seaborn')
plt.plot_date(findates,list(finland['parks_percent_change_from_baseline']),label='Finland')
plt.plot_date(newzdates,list(NewZ['parks_percent_change_from_baseline']),label='NewZealand')
plt.plot_date(ausdates,list(Aus['parks_percent_change_from_baseline']),label='Australia')
plt.plot_date(taidates,list(Tai['parks_percent_change_from_baseline']),label='Taiwan')
plt.xlabel('Date')
plt.ylabel('parks_percent_change_from_baseline')
plt.title('Parks mobility for countries least impacted')
plt.tight_layout()
plt.legend()
plt.show()
plt.plot_date(usdates,list(us['parks_percent_change_from_baseline']),label='USA')
plt.plot_date(itdates,list(italy['parks_percent_change_from_baseline']),label='Italy')
plt.plot_date(spdates,list(spain['parks_percent_change_from_baseline']),label='Spain')
plt.plot_date(frdates,list(fran['parks_percent_change_from_baseline']),label='France')
plt.xlabel('Date')
plt.ylabel('parks_percent_change_from_baseline')
plt.title('Park mobility for countries most impacted')
plt.tight_layout()
plt.legend()
plt.show()
# transit_stations_percent_change_from_baseline
plt.style.use('seaborn')
plt.plot_date(findates,list(finland['transit_stations_percent_change_from_baseline']),label='Finland')
plt.plot_date(newzdates,list(NewZ['transit_stations_percent_change_from_baseline']),label='NewZealand')
plt.plot_date(ausdates,list(Aus['transit_stations_percent_change_from_baseline']),label='Australia')
plt.plot_date(taidates,list(Tai['transit_stations_percent_change_from_baseline']),label='Taiwan')
plt.xlabel('Date')
plt.ylabel('transit_stations_percent_change_from_baseline')
plt.title('Transit stations mobility for countries least impacted')
plt.tight_layout()
plt.legend()
plt.show()
plt.plot_date(usdates,list(us['transit_stations_percent_change_from_baseline']),label='USA')
plt.plot_date(itdates,list(italy['transit_stations_percent_change_from_baseline']),label='Italy')
plt.plot_date(spdates,list(spain['transit_stations_percent_change_from_baseline']),label='Spain')
plt.plot_date(frdates,list(fran['transit_stations_percent_change_from_baseline']),label='France')
plt.xlabel('Date')
plt.ylabel('transit_stations_percent_change_from_baseline')
plt.title('Transit Stations mobility for countries most impacted')
plt.tight_layout()
plt.legend()
plt.show()
# workplaces_percent_change_from_baseline
plt.style.use('seaborn')
plt.plot_date(findates,list(finland['workplaces_percent_change_from_baseline']),label='Finland')
plt.plot_date(newzdates,list(NewZ['workplaces_percent_change_from_baseline']),label='NewZealand')
plt.plot_date(ausdates,list(Aus['workplaces_percent_change_from_baseline']),label='Australia')
plt.plot_date(taidates,list(Tai['workplaces_percent_change_from_baseline']),label='Taiwan')
plt.xlabel('Date')
plt.ylabel('workplaces_percent_change_from_baseline')
plt.title('Workplace mobility for countries least impacted')
plt.tight_layout()
plt.legend()
plt.show()
plt.plot_date(usdates,list(us['workplaces_percent_change_from_baseline']),label='USA')
plt.plot_date(itdates,list(italy['workplaces_percent_change_from_baseline']),label='Italy')
plt.plot_date(spdates,list(spain['workplaces_percent_change_from_baseline']),label='Spain')
plt.plot_date(frdates,list(fran['workplaces_percent_change_from_baseline']),label='France')
plt.xlabel('Date')
plt.ylabel('workplaces_percent_change_from_baseline')
plt.title('Workplace mobility for countries most impacted')
plt.tight_layout()
plt.legend()
plt.show()
# residential_percent_change_from_baseline
plt.style.use('seaborn')
plt.plot_date(findates,list(finland['residential_percent_change_from_baseline']),label='Finland')
plt.plot_date(newzdates,list(NewZ['residential_percent_change_from_baseline']),label='NewZealand')
plt.plot_date(ausdates,list(Aus['residential_percent_change_from_baseline']),label='Australia')
plt.plot_date(taidates,list(Tai['residential_percent_change_from_baseline']),label='Taiwan')
plt.xlabel('Date')
plt.ylabel('residential_percent_change_from_baseline')
plt.title('Residential mobility for countries least impacted')
plt.tight_layout()
plt.legend()
plt.show()
plt.plot_date(usdates,list(us['residential_percent_change_from_baseline']),label='USA')
plt.plot_date(itdates,list(italy['residential_percent_change_from_baseline']),label='Italy')
plt.plot_date(spdates,list(spain['residential_percent_change_from_baseline']),label='Spain')
plt.plot_date(frdates,list(fran['residential_percent_change_from_baseline']),label='France')
plt.xlabel('Date')
plt.ylabel('residential_percent_change_from_baseline')
plt.title('Residential mobility for countries most impacted')
plt.tight_layout()
plt.legend()
plt.show()
