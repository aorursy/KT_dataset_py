import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
raw_table = pd.read_excel ('../input/scraped1.xlsx',  header = 1,
                            parse_dates = True)
raw_table.head()
raw_table.dropna(axis = 1, how = 'all', inplace= True) # Drop NA
raw_table.drop(index = 0,columns = [0,1], inplace= True) # Drop Raw and Column
raw_table.columns
columns=['patient_no', 'state_patient_no', 'date_detected', 'age', 'gender', 
         'city', 'district','State', 'State code' ,'current_status', 'notes', 'contracted_from', 
         'nationality','type of transmission', 'last_status', 'source1', 'source2', 'source3','remarks']
raw_table.columns = columns # rename columns name
raw_table = raw_table.infer_objects() # assign data type
raw_table.date_detected = pd.to_datetime(raw_table.date_detected, dayfirst= True ) # to convert Datetime
raw_table.last_status = pd.to_datetime (raw_table.last_status, dayfirst= True)
raw_table.head()
raw_table.info()
covid19_date_grpby = raw_table.groupby(['date_detected']).patient_no.count() # group by date to patient number
covid19_date_grpby[-5:]
covid19_date_grpby_cumsum = covid19_date_grpby.cumsum() # cumsum taking and generate new data frame
covid19_date_grpby_cumsum[-5:]
covid19_date_grpby_cumsum = covid19_date_grpby_cumsum.to_frame() # converting to data frame
covid19_date_grpby_cumsum['daily_pct_chng'] = covid19_date_grpby_cumsum['patient_no'].pct_change() # taking change in percentage
# converting percentage in to double digit
covid19_date_grpby_cumsum['daily_pct_chng'] = round(covid19_date_grpby_cumsum['daily_pct_chng'] * 100) 
covid19_date_grpby_cumsum[-5:]
# plotting Graph of Date Vs No of infected patient

covid19_date_grpby_cumsum.plot(y = 'patient_no', kind = 'bar', figsize=(15,6), legend = True , 
                               title = 'Day wise Infected patient count')
plt.xlabel ('Date')
plt.ylabel ('No of Infected Patient')
plt.savefig('date_wise.png',bbox_inches= 'tight')
# plotting Graph of Date Vs No of infected patient in log scale

ax = covid19_date_grpby_cumsum.plot(y = 'patient_no',kind = 'bar', figsize=(15,7), logy = True, legend = True, 
                               title = 'Day wise logarithmic infected patient count')
ax.set_xlabel('Date')
ax.set_ylabel('log scale')

plt.savefig('date_wise_log.png',bbox_inches= 'tight' )
covid19_date_grpby_cumsum[-5:]
# plotting Graph of Date Vs Daily change in percentage of Infected patient

covid19_date_grpby_cumsum.plot (y= 'daily_pct_chng',kind = 'bar',legend = True  ,ylim = (0,120),
                figsize=(15,7),title = 'Daily change in Percentage',)

plt.xlabel('Date')
plt.ylabel ('Percentage %')
plt.savefig('daily_pct_chng.png', format = 'png', bbox_inches= 'tight' )
# Grouping by state

covid19_state_grpby = raw_table.groupby('State').patient_no.count() # Grouping by state
covid19_state_grpby = covid19_state_grpby.sort_values(ascending = False)
covid19_state_grpby = covid19_state_grpby.to_frame()
covid19_state_grpby[:3]
# plotting Name of State vs Patient Infection in nos

ax = covid19_state_grpby.plot(kind = 'bar', figsize = (15,6), 
                              legend = True, title = 'State wise Infection')
ax.set_ylabel ('Infection In Numbers')
ax.set_xlabel ('Name of State')
plt.savefig('state_wise_barchart',bbox_inches= 'tight')
# plotting state wise infection in pie chart

covid19_state_grpby.plot(kind = 'pie', y = 'patient_no', figsize=(10,10) ,legend = False,autopct = '%0.2f%%' ,
                        labeldistance= 0.7, rotatelabels = 180, pctdistance = 0.55,
                         title= 'State wise Infection status', )

plt.ylabel('')
plt.axis ('equal')
plt.savefig('state_piechart.png', format = 'png', bbox_inches= 'tight' )
# adding column of Current status of Infected patient

covid19_current_status_grpby = (raw_table.groupby(['date_detected','current_status']).patient_no.count()).to_frame()
covid19_current_status_grpby
# plotting graph of Status of Infected patient on Daily basis

covid19_current_status_grpby.unstack().plot(kind = 'bar', figsize = (15,12), 
                                            legend = True , subplots = True) 
                                              
plt.savefig('Patient_status.png', bbox_inches= 'tight')
# importing data

covid_raw_table2 = pd.read_excel ('../input/scraped2.xlsx', header = 1,
                                 parse_dates= True, )
# data cleaning

covid_raw_table2.drop(columns = [0,1], inplace= True)
covid_raw_table2.dropna(axis= 1,how = 'all' ,inplace= True)
covid_raw_table2.dropna(axis=0,how ='all' ,inplace= True)
covid_raw_table2[:5]
# creating data frame of Infected patient wise

covid19_total = pd.DataFrame(data = covid_raw_table2.loc[0])
covid19_total = covid19_total.loc['Recovered':'Active']
covid19_total

# plotting status of infected patient

covid19_total.plot(kind = 'pie', figsize = (10,7), y = 0, 
            legend = False, autopct = '%0.2f%%' ,labeldistance= 1.02, rotatelabels = 0, pctdistance = 0.65,
            title= 'Patient Helth Status INDIA', fontsize=12)


plt.ylabel('')
plt.axis('equal')
plt.savefig("patient_health_status_INDIA.png")
# importing sate wise information

state_wise_information = pd.read_excel ('../input/state wise information.xlsx',
                        index_col= 'State')
# state wise information collected, population, area and density
# Covid19_state_grpby and state_wise_analysis is merged 

state_wise_analysis=pd.merge(covid19_state_grpby, state_wise_information, left_index= True,right_index= True)
state_wise_analysis
# new column addition

state_wise_analysis['patient_infct_area_density'] = (state_wise_analysis.patient_no / 
                                                     state_wise_analysis['Area (km2)'])
state_wise_analysis['patient_infct_population'] = (state_wise_analysis.patient_no / 
                                                  state_wise_analysis.Population) 
state_wise_analysis[:2]
# plotting data of state vs infected patient / area km2

state_wise_analysis.sort_values(by= 'patient_infct_area_density', ascending= False).plot(kind = 'bar',
                        y = 'patient_infct_area_density', figsize= (15,7), 
                        legend = True, title = 'State wise infected patient per km2 area')

plt.ylabel ('infected patient / Area')
plt.savefig ('State wise infected patient per km2 area.png', bbox_inches= 'tight')
# Zoom in in existing plot of state vs infected patient / area km2
 
state_wise_analysis.sort_values(by = 'patient_infct_area_density', ascending= False).plot(kind = 'bar', 
                        y = 'patient_infct_area_density', figsize= (15,7), ylim= (0,0.02),
                         legend = True, title = 'State wise infected patient per km2 area in depth')

plt.ylabel ('infected patient / Area')
plt.savefig ('State wise infected patient per km2 area in depth.png', bbox_inches= 'tight')
# plotting data of state vs (infected patient / population)

state_wise_analysis.sort_values(by = 'patient_infct_population', ascending= False).plot(kind = 'bar', 
                        y = 'patient_infct_population', figsize= (15,7),
                        legend = True, title = ('State wise Infected patient to population'))

plt.ylabel ('Infected patient to population')
plt.savefig ('State wise Infected patient to population.png', bbox_inches= 'tight')
# Making Data Frame For Gujrat State Daily Infected Patient Increasing and CUMSUM 
state_datewise_chng = raw_table.groupby(['State', 'date_detected']).patient_no.count() 
state_datewise_chng = state_datewise_chng.sort_index (ascending = True)
state_datewise_chng = state_datewise_chng.to_frame()
state_datewise_chng_gujrat = state_datewise_chng.loc['Gujarat'].sort_values (by = 'date_detected', ascending = True)
state_datewise_chng_gujrat = state_datewise_chng_gujrat.cumsum()

state_datewise_chng_gujrat['daily_pct_chng'] = round ((state_datewise_chng_gujrat['patient_no'].pct_change()) *100)
state_datewise_chng_gujrat[-5:] 
# Gujrat Daily Change In PCT CUMSUm analysis

ax1 = state_datewise_chng_gujrat.plot(kind = 'bar', 
                        y = 'daily_pct_chng', figsize= (15,7), 
                        legend = True, title = 'Gujrat Daily Change in corona infection In PCT')


for p in ax1.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    plt.annotate('{0}'.format(height), (x, y + height + 1.5))

plt.xlabel ('Date') 
plt.ylabel ('Percentage')
plt.savefig ('Gujrat Daily Change in corona infection In PCT.png', bbox_inches= 'tight')
# NOw same calculating For Delhi

state_datewise_chng_delhi = state_datewise_chng.loc['Delhi'].sort_values (by = 'date_detected', ascending = True)
state_datewise_chng_delhi = state_datewise_chng_delhi.cumsum()
state_datewise_chng_delhi [:5]
state_datewise_chng_delhi['daily_pct_chng'] = round ((state_datewise_chng_delhi['patient_no'].pct_change()) *100)
state_datewise_chng_delhi[-5:] 
# Delhi Daily Change In PCT CUMSUm analysis

ax2 = state_datewise_chng_delhi.plot(kind = 'bar', 
                        y = 'daily_pct_chng', figsize= (15,7), 
                        legend = True, title = 'Delhi Daily Change in corona infection In PCT')

for p in ax2.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    plt.annotate('{0}'.format(height), (x, y + height + 1.5))
    
plt.xlabel ('Date') 
plt.ylabel ('Percentage')
plt.savefig ('Delhi Daily Change in corona infection In PCT.png', bbox_inches= 'tight')
# calculating for Maharastra

state_datewise_chng_maharashtra = state_datewise_chng.loc['Maharashtra'].sort_values (by = 'date_detected', ascending = True)
state_datewise_chng_maharashtra = state_datewise_chng_maharashtra.cumsum()

state_datewise_chng_maharashtra['daily_pct_chng'] = round((state_datewise_chng_maharashtra['patient_no'].pct_change()) *100)
state_datewise_chng_maharashtra[-5:]
# Maharashtra Daily Change In PCT CUMSUm analysis

ax3 = state_datewise_chng_maharashtra.plot(kind = 'bar', 
                        y = 'daily_pct_chng', figsize= (15,7), 
                        legend = True, title = 'Maharashtra Daily Change in corona infection In PCT')


for p in ax3.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    plt.annotate('{0}'.format(height), (x, y + height + 1.5))
    
plt.xlabel ('Date') 
plt.ylabel ('Percentage')
plt.savefig ('Maharashtra Daily Change in corona infection In PCT.png', bbox_inches= 'tight')
# calculating for Uttar Pradesh

state_datewise_chng_up = state_datewise_chng.loc['Uttar Pradesh'].sort_values (by = 'date_detected', ascending = True)
state_datewise_chng_up = state_datewise_chng_up.cumsum()

state_datewise_chng_up['daily_pct_chng'] = round ((state_datewise_chng_up['patient_no'].pct_change()) *100)
state_datewise_chng_up[-5:]
#  Uttar Pradesh Daily Change In PCT CUMSUm analysis

ax4 = state_datewise_chng_up.plot(kind = 'bar', 
                        y = 'daily_pct_chng', figsize= (15,7), 
                        legend = True, title = ' Uttar Pradesh Daily Change in corona infection In PCT')

for p in ax4.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    plt.annotate('{0}'.format(height), (x, y + height + 1.5))
    
plt.xlabel ('Date') 
plt.ylabel ('Percentage')
plt.savefig (' Uttar Pradesh Daily Change in corona infection In PCT.png', bbox_inches= 'tight')
# calculating for Tamil Nadu

state_datewise_chng_tamilnadu = state_datewise_chng.loc['Tamil Nadu'].sort_values (by = 'date_detected', ascending = True)
state_datewise_chng_tamilnadu = state_datewise_chng_tamilnadu.cumsum()

state_datewise_chng_tamilnadu['daily_pct_chng'] = round((state_datewise_chng_tamilnadu['patient_no'].pct_change()) *100)
state_datewise_chng_tamilnadu[-5:]
#  Tamil Nadu Daily Change In PCT CUMSUm analysis

ax5 = state_datewise_chng_tamilnadu.plot(kind = 'bar', 
                        y = 'daily_pct_chng', figsize= (15,7), 
                        legend = True, title = ' Tamil Nadu Daily Change in corona infection In PCT')

for p in ax5.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    plt.annotate('{0}'.format(height), (x, y + height + 1.5))
    
plt.xlabel ('Date') 
plt.ylabel ('Percentage')
plt.savefig (' Tamil Nadu Daily Change in corona infection In PCT.png', bbox_inches= 'tight')
# calculating for Rajasthan

state_datewise_chng_rajasthan = state_datewise_chng.loc['Rajasthan'].sort_values (by = 'date_detected', ascending = True)
state_datewise_chng_rajasthan = state_datewise_chng_rajasthan.cumsum()

state_datewise_chng_rajasthan['daily_pct_chng'] = round ((state_datewise_chng_rajasthan['patient_no'].pct_change()) *100)
state_datewise_chng_rajasthan[-5:]
#  Rajasthan Daily Change In PCT CUMSUm analysis

ax6 = state_datewise_chng_rajasthan.plot(kind = 'bar', 
                        y = 'daily_pct_chng', figsize= (15,7), 
                        legend = True, title = ' Rajasthan Daily Change in corona infection In PCT')

for p in ax6.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    plt.annotate('{0}'.format(height), (x, y + height + 1.5))
    
plt.xlabel ('Date') 
plt.ylabel ('Percentage')
plt.savefig (' Rajasthan Daily Change in corona infection In PCT.png', bbox_inches= 'tight')
# calculating for Madhya Pradesh

state_datewise_chng_mp = state_datewise_chng.loc['Madhya Pradesh'].sort_values (by = 'date_detected', ascending = True)
state_datewise_chng_mp = state_datewise_chng_mp.cumsum()

state_datewise_chng_mp['daily_pct_chng'] = round ((state_datewise_chng_mp['patient_no'].pct_change()) *100)
state_datewise_chng_mp[-5:]
#  Madhya Pradesh Daily Change In PCT CUMSUm analysis

ax7 = state_datewise_chng_mp.plot(kind = 'bar', 
                        y = 'daily_pct_chng', figsize= (15,7), 
                        legend = True, title = ' Madhya Pradesh Daily Change in corona infection In PCT')

for p in ax7.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    plt.annotate('{0}'.format(height), (x, y + height + 1.5))
    
plt.xlabel ('Date') 
plt.ylabel ('Percentage')
plt.savefig (' Madhya Pradesh Daily Change in corona infection In PCT.png', bbox_inches= 'tight')
# Grouping by state

covid19_Gujrat_grpby = raw_table.groupby(['State', 'district']).patient_no.count() 
covid19_Gujrat_grpby = covid19_Gujrat_grpby.sort_index(ascending = True)
covid19_Gujrat_grpby = covid19_Gujrat_grpby.to_frame()
covid19_Gujrat_grpby = covid19_Gujrat_grpby.loc['Gujarat'].sort_values(by = 'patient_no', ascending = False)
covid19_Gujrat_grpby[:5]
ax8 = covid19_Gujrat_grpby.plot (kind = 'bar', y = 'patient_no', figsize= (15,7),
                        legend = True, title = ('Gujarat District wise Infected patient NO'))


for p in ax8.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    plt.annotate('{0}'.format(height), (x, y + height + 1.5))

plt.ylabel ('Infected patient NO')
plt.xlabel ('Name of District')
plt.savefig ('Gujarat District wise Infected patient No.png', bbox_inches= 'tight')
covid19_Gujrat_grpby.plot (kind = 'pie', figsize=(15,10), y = 'patient_no', 
            legend = False, autopct = '%0.2f%%' ,labeldistance= 1.02, rotatelabels = 0, pctdistance = 0.65,
            title = ('Gujarat District wise Infected patient NO'))

plt.axis ('equal')
plt.ylabel ('')
plt.savefig ('Gujarat District wise Infected patient No pie chart.png', bbox_inches= 'tight')
# Gujarat State Status

covid19_gujrat = covid_raw_table2.loc[8]
covid19_gujrat = pd.DataFrame (data = covid19_gujrat)
covid19_gujrat = covid19_gujrat.loc['Recovered':'Active']
covid19_gujrat
# plotting Gujarat State status of infected patient

covid19_gujrat.plot(kind = 'pie', figsize = (10,7), y = 8, 
            legend = False, autopct = '%0.2f%%' ,labeldistance= 1.02, rotatelabels = 0, pctdistance = 0.65,
            title= ' Gujarat State Patient Helth Status', fontsize=12)


plt.ylabel('')
plt.axis('equal')
plt.savefig("Gujarat_state_patient_health_status.png")
# Gujrat Disctict wise information

Gujrat_dist_info = pd.read_excel('../input/gujarat district.xlsx', 
                                index_col= 'District Name')
Gujrat_dist_info[:5]
# Merge two Data frame for population, area , density and infected patient number

Gujrat_dist_analysis = pd.merge(Gujrat_dist_info,covid19_Gujrat_grpby , left_index= True, right_index= True)
Gujrat_dist_analysis = Gujrat_dist_analysis.sort_values (by = 'patient_no', ascending = False)
Gujrat_dist_analysis[:5]
# Adding Two new column in Gujrat_dist_analysis

Gujrat_dist_analysis['patient_infct_area_density'] = ((Gujrat_dist_analysis.patient_no / 
                                                       Gujrat_dist_analysis['Area(km2)']))
Gujrat_dist_analysis['patient_infct_population'] = (Gujrat_dist_analysis.patient_no / 
                                                    Gujrat_dist_analysis.Population) 
Gujrat_dist_analysis[:5]
# plotting data of District vs (infected patient / population)

Gujrat_dist_analysis.sort_values(by = 'patient_infct_population', ascending= False).plot(
    kind = 'bar', y = 'patient_infct_population', figsize= (15,7),
    legend = True, title = ('Gujrat District wise Infected patient to population Ratio'))

plt.ylabel ('Infected patient to population')

plt.savefig ('Gujrat District wise Infected patient to population ratio.png', bbox_inches= 'tight')
# plotting data of Gujrat District vs infected patient / area km2

Gujrat_dist_analysis.sort_values(by = 'patient_infct_area_density', ascending= False).plot(kind = 'bar', 
                        y = 'patient_infct_area_density', figsize= (15,7), 
                        legend = True, title = 'Gujrat District wise infected patient per km2 area')

plt.ylabel ('infected patient / Area')
plt.savefig ('Gujrat District wise infected patient per km2 area.png', bbox_inches= 'tight')
# Making Data Frame For Gujrat State Daily Infected Patient Increasing and CUMSUM 
gj_district = raw_table.groupby(['district', 'date_detected']).patient_no.count() 
gj_district = gj_district.sort_index (ascending = True)
gj_district = gj_district.to_frame()
gj_district_abad= gj_district.loc['Ahmadabad'].sort_values (by = 'date_detected', ascending = True)
gj_district_abad = gj_district_abad.cumsum()

gj_district_abad['daily_pct_chng'] = round ((gj_district_abad['patient_no'].pct_change()) *100)
gj_district_abad[-5:]
# Gujrat - Ahmadabad District Change In PCT In infected Patient analysis

ax9 = gj_district_abad.plot(kind = 'bar', 
                        y = 'daily_pct_chng', figsize= (15,7), 
                        legend = True, title = 'Ahmadabad District Daily Change in corona infection In PCT')


for p in ax9.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    plt.annotate('{0}'.format(height), (x, y + height + 1.2))

plt.xlabel ('Date') 
plt.ylabel ('Percentage')
plt.savefig ('Ahmadabad District Daily Change in corona infection In PCT.png', bbox_inches= 'tight')
# Making Data Frame For Gujrat State Daily Infected Patient Increasing and CUMSUM 

gj_district_surat= gj_district.loc['Surat'].sort_values (by = 'date_detected', ascending = True)
gj_district_surat = gj_district_surat.cumsum()

gj_district_surat['daily_pct_chng'] = round((gj_district_surat['patient_no'].pct_change()) *100)
gj_district_surat[-5:]
# Gujrat - Surat District Change In PCT In infected Patient analysis

ax10 = gj_district_surat.plot(kind = 'bar', 
                        y = 'daily_pct_chng', figsize= (15,7), 
                        legend = True, title = 'Surat District Daily Change in corona infection In PCT')

for p in ax10.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    plt.annotate('{0}'.format(height), (x, y + height + 1.5))

plt.xlabel ('Date') 
plt.ylabel ('Percentage')
plt.savefig ('Surat District Daily Change in corona infection In PCT.png', bbox_inches= 'tight')
# Gujrat - Vadodra District Change In PCT In infected Patient analysis

gj_district_vadodara= gj_district.loc['Vadodara'].sort_values (by = 'date_detected', ascending = True)
gj_district_vadodara = gj_district_vadodara.cumsum()

gj_district_vadodara['daily_pct_chng'] = round((gj_district_vadodara['patient_no'].pct_change()) *100)
gj_district_vadodara[-5:]
# Gujrat - Vadodara District Change In PCT In infected Patient analysis

ax11 = gj_district_vadodara.plot(kind = 'bar', 
                        y = 'daily_pct_chng', figsize= (15,7), 
                        legend = True, title = 'Vadodara District Daily Change in corona infection In PCT')

for p in ax11.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    plt.annotate('{0}'.format(height), (x, y + height + 1.5))

plt.xlabel ('Date') 
plt.ylabel ('Percentage')
plt.savefig ('Vadodara District Daily Change in corona infection In PCT.png', bbox_inches= 'tight')


