#Import required libraries

import numpy as np

import pandas as pd
#analyzing data for 2016

accidents_by_transport_type_data_2016 = "../input/india-mode-of-transport-deaths-road-accidents2016/All India Level Mode of Transport-wise Number of Persons Died in Road Accidents during 2016.csv"

accidents_by_type_2016_data = pd.read_csv(accidents_by_transport_type_data_2016)

accidents_by_type_2016_data.head()
def transport_type_header(column_data):

    """We do-not need line level details about the accidents, so slicing the dataframe to have only header rows """

    select_row = []

    for transport_type in column_data:

        if transport_type.find(".")< 0:

            select_row.append(True)

        else:

            select_row.append(False)

    return select_row
#Slicing dataframe to have details about motorised vehicles and not further details 

acc_by_type_16 = accidents_by_type_2016_data[[a and b for a, b in zip(transport_type_header(accidents_by_type_2016_data['Mode of Transport']),

    list(accidents_by_type_2016_data['Sl. No.']<2.0))]]



#Slicing the dataframe to have number of road deaths caused due to different mode of transport

acc_by_type_16 = acc_by_type_16[['Mode of Transport', 'No. of Offending Driver/Pedestrian - Died',

                                 'No. of Victims - Died','Total Persons Died']]

acc_by_type_16
#analyzing data for 2017

accidents_by_transport_type_data_2017 = "../input/india-mode-of-transport-deaths-road-accidents2016/All India Level Mode of Transport-wise Number of Persons Died in Road Accidents during 2017.csv"

accidents_by_type_2017_data = pd.read_csv(accidents_by_transport_type_data_2017)



#Slicing dataframe to have details about motorised vehicles and not further details 

acc_by_type_17 = accidents_by_type_2017_data[[a and b for a, b in zip(transport_type_header(accidents_by_type_2017_data['Mode of Transport']),

    list(accidents_by_type_2017_data['Sl. No.']<2.0))]]



#Slicing the dataframe to have number of road deaths caused due to different mode of transport

acc_by_type_17 = acc_by_type_17[['Mode of Transport', 'No. of Offending Driver/Pedestrian - Died',

                                 'No. of Victims - Died','Total Persons Died']]

acc_by_type_17
#analyzing data for 2018

accidents_by_transport_type_data_2018 = "../input/india-mode-of-transport-deaths-road-accidents2016/All India Level Mode of Transport-wise Number of Persons Died in Road Accidents during 2018.csv"

accidents_by_type_2018_data = pd.read_csv(accidents_by_transport_type_data_2018)



#Slicing dataframe to have details about motorised vehicles and not further details 

acc_by_type_18 = accidents_by_type_2018_data[[a and b for a, b in zip(transport_type_header(accidents_by_type_2018_data['Mode of Transport']),

    list(accidents_by_type_2018_data['Sl. No.']<2.0))]]



#Slicing the dataframe to have number of road deaths caused due to different mode of transport

acc_by_type_18 = acc_by_type_18[['Mode of Transport', 'No. of Offending Driver/Pedestrian - Died',

                                 'No. of Victims - Died','Total Persons Died']]

acc_by_type_18
#Combining Dataset from all 3 years (2016-2018)

deaths_by_mode_last3yrs = acc_by_type_17.merge(acc_by_type_18,on="Mode of Transport",how = "inner",suffixes = ("_2017", "_2018"))

deaths_by_mode_last3yrs = acc_by_type_16.merge(deaths_by_mode_last3yrs,on="Mode of Transport",how = "inner")

deaths_by_mode_last3yrs
import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline



sns.set_style('darkgrid')

matplotlib.rcParams['font.size'] = 11

matplotlib.rcParams['figure.figsize'] = (9, 5)

matplotlib.rcParams['figure.facecolor'] = '#00000000'
#Lets take a quick look on the trend of total deaths in road accidents between 2016-2018

#deaths_by_mode_last3yrs.tail(1).loc[:,['Total Persons Died','Total Persons Died_2017','Total Persons Died_2018']]

plt.plot([2016,2017,2018],deaths_by_mode_last3yrs.tail(1).loc[:,['Total Persons Died','Total Persons Died_2017',

                                                                 'Total Persons Died_2018']].transpose(),

        marker = 'x')



plt.xscale("linear")

plt.xlabel("Years")

plt.ylabel("Total Accidents")

plt.title("Number of Deaths due to Road Accidents (2016-18)")

plt.show()
deaths_by_mode_last3yrs[['Total Persons Died%','Total Persons Died%_2017','Total Persons Died%_2018']] = deaths_by_mode_last3yrs.loc[:7,['Total Persons Died','Total Persons Died_2017','Total Persons Died_2018']]/deaths_by_mode_last3yrs.loc[8,['Total Persons Died','Total Persons Died_2017','Total Persons Died_2018']]*100

deaths_by_mode_last3yrs
data = deaths_by_mode_last3yrs.loc[:7,['Mode of Transport','Total Persons Died%','Total Persons Died%_2017','Total Persons Died%_2018']].transpose()

matplotlib.rcParams['figure.figsize'] = (15, 10)

plt.plot([2016,2017,2018],data.iloc[1:,:],marker = 'X')

plt.legend(labels = data.iloc[0,:])

plt.xlabel("Years")

plt.ylabel("Percentage of total accidents")

plt.show()
plt.pie(data.iloc[3,:],labels = data.iloc[0,:],autopct='%1.1f%%')

plt.show()
deaths_by_mode_last3yrs.loc[:7,:].sort_values(['Total Persons Died','Total Persons Died_2017','Total Persons Died_2018'],ascending=False).head(1)['Mode of Transport']
deaths_by_mode_last3yrs['2016-2017']=(deaths_by_mode_last3yrs['Total Persons Died_2017']-deaths_by_mode_last3yrs['Total Persons Died'])/deaths_by_mode_last3yrs['Total Persons Died']*100

deaths_by_mode_last3yrs['2017-2018']=(deaths_by_mode_last3yrs['Total Persons Died_2018']-deaths_by_mode_last3yrs['Total Persons Died_2017'])/deaths_by_mode_last3yrs['Total Persons Died_2017']*100

deaths_by_mode_last3yrs
deaths_by_mode_last3yrs.sort_values(['2016-2017','2017-2018'],ascending=False).head(1)[['Mode of Transport','2016-2017','2017-2018']]
deaths_by_mode_last3yrs.sort_values(['2016-2017','2017-2018'],ascending=True).head(2)[['Mode of Transport','2016-2017','2017-2018']]
deaths_by_mode_last3yrs.loc[1:7,:].sort_values(['No. of Offending Driver/Pedestrian - Died','No. of Offending Driver/Pedestrian - Died_2017','No. of Offending Driver/Pedestrian - Died_2018'],ascending=False).head(1)[['Mode of Transport','No. of Offending Driver/Pedestrian - Died','No. of Offending Driver/Pedestrian - Died_2017','No. of Offending Driver/Pedestrian - Died_2018']]
deaths_by_mode_last3yrs.loc[1:7,:].sort_values(['No. of Victims - Died','No. of Victims - Died_2017','No. of Victims - Died_2018'],ascending=False).head(1)[['Mode of Transport','No. of Victims - Died','No. of Victims - Died_2017','No. of Victims - Died_2018']]