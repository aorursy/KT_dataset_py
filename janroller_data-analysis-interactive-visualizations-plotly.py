import pandas as pd

import matplotlib.pyplot as pp

import numpy as np

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()
raw_data = pd.read_csv('../input/uspollution/pollution_us_2000_2016.csv')
raw_data.head(5)
raw_data.columns
data = raw_data.drop(['Unnamed: 0', 'State Code', 'County Code', 'Site Num', 'Address', 'County', 'NO2 Units', 

        'O3 Units', 'SO2 Units', 'CO Units'], axis = 1)
data.columns
data.describe(include='all')
data.isnull().sum()
data_no_mv =  data.dropna(axis=0)
data_no_dupl = data_no_mv.drop_duplicates()

data_no_duplindex_col=0
data_no_dupl.describe(include='all')
import seaborn as sb

from pylab import *

# Show all distribution plots simultaneously in four subplots 



sb.set(rc={"figure.figsize": (15, 10)})



subplot(2,2,1)

ax = sb.kdeplot(data_no_dupl['CO AQI'],shade=True)



subplot(2,2,2)

ax = sb.kdeplot(data_no_dupl['SO2 AQI'], shade=True)



subplot(2,2,3)

ax = sb.kdeplot(data_no_dupl['NO2 AQI'], shade=True)



subplot(2,2,4)

ax = sb.kdeplot(data_no_dupl['O3 AQI'], shade=True)



pp.show()
# Define 99% quantile

q1 = data_no_dupl['CO AQI'].quantile(0.99)

q1
# Dropping observations that are greater than the 99% quantile, which lies above the value 33.

CO_outliers = data_no_dupl[data_no_dupl['CO AQI'] > q1]

data_no_dupl = data_no_dupl.drop(CO_outliers.index, axis= 0)
# Show the new data distribution after outliers were removed. 

sb.distplot(data_no_dupl['CO AQI'])
# Define 99% quantile

q2 = data_no_dupl['SO2 AQI'].quantile(.99)

q2
# Dropping observations that are greater than the 99% quantile, which lies above the value 69.

SO2_outliers = data_no_dupl[data_no_dupl['SO2 AQI']>q2]

data_no_dupl = data_no_dupl.drop(SO2_outliers.index)
# Show the new data distribution after outliers were removed. 

sb.distplot(data_no_dupl['SO2 AQI'])
q3 = data_no_dupl['NO2 AQI'].quantile(.99)

q3
# Dropping observations that are greater than the 99% quantile, which lies above the value 70.

NO2_outliers = data_no_dupl[data_no_dupl['NO2 AQI']>q3]

data_no_dupl = data_no_dupl.drop(NO2_outliers.index)
# Show the new data distribution after outliers were removed. 

sb.distplot(data_no_dupl['NO2 AQI'])
# Define 99% quantile

q4 = data_no_dupl['O3 AQI'].quantile(.99)

q4
# Dropping observations that are greater than the 99% quantile, which lies above the value 119.

O3_outliers = data_no_dupl[data_no_dupl['O3 AQI']>q4]

data_no_dupl = data_no_dupl.drop(O3_outliers.index)
# Show the new data distribution after outliers were removed. 

sb.distplot(data_no_dupl['O3 AQI'])
data_no_dupl['State'].unique()
# Remove all rows that contain string 'Country Of Mexico' in column 'State'

data_var1 = data_no_dupl[~data_no_dupl.State.str.contains("Country Of Mexico")]
data_var1['City'].unique()
import datetime as dt

# make copy of data first to avoid ~SettingWithCopyWarning~

data_var1 = data_var1[data_var1['Date Local'].notnull()].copy()

data_var1['Year_Month'] = pd.to_datetime(data_var1['Date Local']).dt.strftime('%Y-%m') #Year-Month

data_var1['Year'] = pd.to_datetime(data_var1['Date Local']).dt.strftime('%Y') #Year

data_var1['Month'] = pd.to_datetime(data_var1['Date Local']).dt.strftime('%m') #Year
# create sub data set with relevant variables only

pollution_df = data_var1[['Year_Month','Year','Month','State','City','NO2 AQI','O3 AQI','SO2 AQI','CO AQI']]
pollution_CA = pollution_df [pollution_df['State'] == 'California'].reset_index(drop=True)

pollution_CA.head()
CA_10per = pollution_CA.sample(frac=0.5)

CA_10per = CA_10per.sort_values('Year_Month').reset_index(drop=True)
#import chart_studio.tools as tls

import plotly.express as px 

import cufflinks as cf            

import plotly.graph_objs as go
# Create two new pd data frames

AQI_time= CA_10per[['Year_Month','NO2 AQI','O3 AQI','SO2 AQI','CO AQI']] # all four AQIs incl. date

AQI = AQI_time.iloc[:,1:]  # all four AQIs only
#This interactive plot I had to toggle because there is an error when loading the chart_studio package. 

#It works fine in other environments (e.g. Jupyter), but import the package here gives me the error 

# 'ModuleNotFoundError: No module named 'chart_studio''. Uninstalling and reinstalling the package did also not solve it.

#Any help on this issue is appreciated!



#AQI.iplot(kind='histogram', subplots= True, shape = (1,4),

 #               xaxis_title="Value", yaxis_title = 'Count',

  #              color=["red", "goldenrod", "#00D", 'lightgreen'],

   #             title= {'text': "Distribution of AQI values per AQI category in CA"},

    #            filename='US Pollution-CA-AQI-distrib-multiple-histo')
sb.pairplot(AQI)
sb.jointplot(x=AQI_time["NO2 AQI"], y=AQI_time["CO AQI"], kind='kde',color='blue', xlim=(0,50),ylim=(0,15))

plt.show()
sb.jointplot(x=AQI_time["O3 AQI"], y=AQI_time["CO AQI"], kind='kde',color='green', xlim=(0,50),ylim=(0,15))

plt.show()
fig, (ax1, ax2,ax3,ax4) = pp.subplots(4,1, figsize = (20,20)) 

for ax in ax1, ax2,ax3,ax4:

    ax.set(xlabel='Date')

    ax.set(ylabel='Value')



ax1.bar(AQI_time['Year_Month'],AQI_time['CO AQI'], color = 'purple')

ax1.set_title('CO AQI')

ax1.set_xticks(['2000-06','2001-06','2002-06','2003-06','2004-06','2005-06','2006-06','2007-06','2008-06','2009-06','2010-06','2011-06', '2012-06','2013-06','2014-06','2015-06','2016-06']) 



ax2.bar(AQI_time['Year_Month'], AQI_time['SO2 AQI'], color = 'red')

ax2.set_title('SO2 AQI')

ax2.set_xticks(['2000-06','2001-06','2002-06','2003-06','2004-06','2005-06','2006-06','2007-06','2008-06','2009-06','2010-06','2011-06', '2012-06','2013-06','2014-06','2015-06','2016-06']) 



ax3.bar(AQI_time['Year_Month'],AQI_time['NO2 AQI'], color = 'green')

ax3.set_title('NO2 AQI')

ax3.set_xticks(['2000-06','2001-06','2002-06','2003-06','2004-06','2005-06','2006-06','2007-06','2008-06','2009-06','2010-06','2011-06', '2012-06','2013-06','2014-06','2015-06','2016-06']) 



ax4.bar(AQI_time['Year_Month'],AQI_time['O3 AQI'], color = 'blue')

ax4.set_title('O3 AQI')

ax4.set_xticks(['2000-06','2001-06','2002-06','2003-06','2004-06','2005-06','2006-06','2007-06','2008-06','2009-06','2010-06','2011-06', '2012-06','2013-06','2014-06','2015-06','2016-06']) 



pp.show()
AQI_time_grouped =AQI_time.groupby(['Year_Month']).mean().plot()
# All four AQI categories over time

fig = go.Figure()

fig.add_trace(go.Scatter(x=CA_10per['Year_Month'], y=CA_10per['NO2 AQI'],

                    mode='lines', name='NO2 AQI', opacity=0.7))

fig.add_trace(go.Scatter(x=CA_10per['Year_Month'], y=CA_10per['O3 AQI'],

                    mode='lines', name='O3 AQI', opacity=0.7))

fig.add_trace(go.Scatter(x=CA_10per['Year_Month'], y=CA_10per['SO2 AQI'],

                    mode='lines', name='SO2 AQI', opacity=1.0))

fig.add_trace(go.Scatter(x=CA_10per['Year_Month'], y=CA_10per['CO AQI'],

                    mode='markers', name='CO AQI', opacity=0.6))

fig.update_layout(legend_title_text = "AQI categories",

                  title='AQI categories 2000-2010')

fig.update_yaxes(title_text="Value (in PPM/PPB)")

fig.update_xaxes(title_text="Time")

fig.show()
fig = px.scatter(CA_10per,x='O3 AQI', y='NO2 AQI', 

             animation_frame='Year', animation_group='City',color='City',

             range_y=[0, 90], range_x=[0, 100])

fig.show()