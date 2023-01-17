import pandas as pd

import numpy as np
df_hs_exp_2010 = pd.read_csv('../input/all_house_senate_2010.csv', na_values='0', index_col='tra_id', sep=',', low_memory=False)



# Filter rows having null or NaN in the date and amount field

df_hs_exp_2010 = df_hs_exp_2010.dropna(subset = ['dis_dat', 'dis_amo'])



# Filter records relevant to year 2010

df_hs_exp_2010 = df_hs_exp_2010[df_hs_exp_2010['dis_dat'].apply(lambda x: x[0:4]) == '2010']



# Format the datatime column for time series analysis

df_hs_exp_2010['dis_dat'] = pd.to_datetime(df_hs_exp_2010['dis_dat'], format='%Y-%m-%d')



# Format the currency field to convert them to float

df_hs_exp_2010['dis_amo'] = (df_hs_exp_2010['dis_amo'].replace( '[\$,)]','', regex=True ).replace( '[(]','-',   regex=True ).astype(float))
#Applying same modifications to 2012, 2014 and 2016 datasets

df_hs_exp_2012 = pd.read_csv('../input/2012.csv', na_values='0', index_col='tra_id', sep=',', parse_dates=['dis_dat'],low_memory=False)

df_hs_exp_2012 = df_hs_exp_2012.dropna(subset = ['dis_dat', 'dis_amo'])

df_hs_exp_2012 = df_hs_exp_2012[df_hs_exp_2012['dis_dat'].apply(lambda x: x[0:4]) == '2012'] #relevant 2012 records

df_hs_exp_2012['dis_dat'] = pd.to_datetime(df_hs_exp_2012['dis_dat'], format='%Y-%m-%d')

df_hs_exp_2012['dis_amo'] = (df_hs_exp_2012['dis_amo'].replace( '[\$,)]','', regex=True ).replace( '[(]','-',   regex=True ).astype(float))

df_hs_exp_2014 = pd.read_csv('../input/all_house_senate_2014.csv', na_values='0', index_col='tra_id', sep=',',low_memory=False)

df_hs_exp_2014 = df_hs_exp_2014.dropna(subset = ['dis_dat', 'dis_amo'])

df_hs_exp_2014 = df_hs_exp_2014[df_hs_exp_2014['dis_dat'].apply(lambda x: x[0:4]) == '2014'] #relevant 2014 records

df_hs_exp_2014['dis_dat'] = pd.to_datetime(df_hs_exp_2014['dis_dat'], format='%Y-%m-%d')

df_hs_exp_2014['dis_amo'] = (df_hs_exp_2014['dis_amo'].replace( '[\$,)]','', regex=True ).replace( '[(]','-',   regex=True ).astype(float))
df_hs_exp_2016 = pd.read_csv('../input/all_house_senate_2016.csv', na_values='0', index_col='tra_id', sep=',',low_memory=False, encoding = "ISO-8859-1")

df_hs_exp_2016 = df_hs_exp_2016.dropna(subset = ['dis_dat', 'dis_amo'])

df_hs_exp_2016 = df_hs_exp_2016[df_hs_exp_2016['dis_dat'].apply(lambda x: x[0:4]) == '2016'] #relevant 2016 records

df_hs_exp_2016['dis_dat'] = pd.to_datetime(df_hs_exp_2016['dis_dat'], format='%Y-%m-%d')

df_hs_exp_2016['dis_amo'] = (df_hs_exp_2016['dis_amo'].replace( '[\$,)]','', regex=True ).replace( '[(]','-',   regex=True ).astype(float))
import datetime

import matplotlib.pyplot as plt

import numpy as np



import plotly.plotly as py

import plotly.tools as tls
# Setting the figure size for the plo



# Caluclating sumtotal of disbursements by each office in Million Dollars 

disbursements_2010 = pd.DataFrame(df_hs_exp_2010.groupby(['can_off', 'ele_yea'])['dis_amo'].sum()/1000000).reset_index().rename(index=str, columns={"can_off": "Candidate Office"})

disbursements_2012 = pd.DataFrame(df_hs_exp_2012.groupby(['can_off', 'ele_yea'])['dis_amo'].sum()/1000000).reset_index().rename(index=str, columns={"can_off": "Candidate Office"})

disbursements_2014 = pd.DataFrame(df_hs_exp_2014.groupby(['can_off', 'ele_yea'])['dis_amo'].sum()/1000000).reset_index().rename(index=str, columns={"can_off": "Candidate Office"})

disbursements_2016 = pd.DataFrame(df_hs_exp_2016.groupby(['can_off', 'ele_yea'])['dis_amo'].sum()/1000000).reset_index().rename(index=str, columns={"can_off": "Candidate Office"})



# Temporary holder for all disbursments

frames = [

            disbursements_2010[['ele_yea', 'Candidate Office', 'dis_amo']], 

            disbursements_2012[['ele_yea', 'Candidate Office', 'dis_amo']],

            disbursements_2014[['ele_yea', 'Candidate Office', 'dis_amo']],

            disbursements_2016[['ele_yea', 'Candidate Office', 'dis_amo']]

         ]



# Combining dataframes into single source

df = pd.concat(frames)



#Renaming abbrevations for H, S and P - House, President and Senate respectively

df['Candidate Office'] = df['Candidate Office'].replace(['H', 'P', 'S'], ['House', 'President','Senate'])



# Pivot the table so as to get disbursement amounts by years 

df = df.pivot(index='ele_yea', columns='Candidate Office', values='dis_amo')



#Plotting the bar graph

df.plot.bar()



# Configuring the axes for the chart

plt.xlabel("Year")

plt.xticks(rotation = 'horizontal')

plt.ylabel("Expenditure (Million Dollars)")



plt.show()