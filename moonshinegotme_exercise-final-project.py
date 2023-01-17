import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex7 import *

print("Setup Complete")
# Check for a dataset with a CSV file

step_1.check()
# Fill in the line below: Specify the path of the CSV file to read

my_filepath = "../input/met-historic-aberport-weather-data/methistoricaberporth.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

data_aberporth = pd.read_csv(my_filepath)



# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first five rows of the data

data_aberporth.head()
data_aberporth = data_aberporth.iloc[5:-4, :7] #reframe selecting from row 5 up to column 7 in index

"""Actually is better to eliminate 2019 from the data as year incomplete and 

it interferes with splitting array we need later""" #so above was [5:, :7] first val replaced with 5:-4

data_aberporth.columns = (['Year', 'Month', 'Tmax', 'Tmin', 'Airfrost', 'Rainfall', 'Sun']) #rename columns

data_aberporth.reset_index(inplace = True, drop = True) #reset the index to start from 0 again 
data_aberporth.shape
aberporth_temp_rain_sun_data = data_aberporth.loc[:, ['Year', 'Tmax', 'Rainfall', 'Sun']] 

#new df for year sun and rainfall 

aberporth_temp_rain_sun_data[0:5]
aberporth_temp_rain_sun_data['Sun'] = aberporth_temp_rain_sun_data.Sun.str.strip('\((.*)\)#') 

# removing unwanted chars from Sun column

aberporth_temp_rain_sun_data # checking

abr_year_list = aberporth_temp_rain_sun_data['Year'].tolist() 

abr_tmax_list = aberporth_temp_rain_sun_data['Tmax'].tolist() 

abr_rain_list = aberporth_temp_rain_sun_data['Rainfall'].tolist() 

abr_sun_list = aberporth_temp_rain_sun_data['Sun'].tolist() 
[len(abr_year_list), len(abr_tmax_list), len(abr_rain_list), len(abr_sun_list)] #checking items match
def remove_blank(x):

    for n, i in enumerate(x): #changing --- with 0.0

        if i == '---':

             x[n] = 0.0

        

remove_blank(abr_year_list)

remove_blank(abr_tmax_list)

remove_blank(abr_rain_list)

remove_blank(abr_sun_list)
abr_year_uniques = set(abr_year_list) # eliminate duplicate years
abr_year_uniques_list = list(abr_year_uniques) # reverting to list and sorting
abr_year_uniques_list.sort()
len(abr_year_uniques) # checking how many unique years are recorded in the dataset
abr_tmax_flist = [] #convert list str into float type

for item in abr_tmax_list:

    abr_tmax_flist.append(float(item))
abr_rain_flist = [] #convert list str into float type

for item in abr_rain_list:

    abr_rain_flist.append(float(item))



abr_sun_flist = [] #convert list str into float type

for item in abr_sun_list:

    abr_sun_flist.append(float(item))

import numpy as np
tmax_array = np.array_split(abr_tmax_flist, 78) # isolating yearly values for each month in list of arrays

rain_array = np.array_split(abr_rain_flist, 78) # 78 years in the dataset

sun_array = np.array_split(abr_sun_flist, 78)

len(tmax_array) # checking values match the data for years
# getting a list of averages for each of the arrays in the arrays lists

tmax_averages = []

for n in range(0, 78):

    tmax_averages.append(np.average(tmax_array[n]))

    

rain_averages = []

for n in range(0, 78):

    rain_averages.append(np.average(rain_array[n]))

    

sun_averages = []

for n in range(0, 78):

    sun_averages.append(np.average(sun_array[n]))
# creating dictionary for new dataframe cointaining mean values per year

abr_average_data_dict = {'Year':abr_year_uniques_list,'Tmax(Mean)':tmax_averages, 'Rain(Mean)':rain_averages, 'Sun(Mean)':sun_averages}
abr_average_data = pd.DataFrame(abr_average_data_dict, columns=['Year', 'Tmax(Mean)', 'Rain(Mean)', 'Sun(Mean)'])
abr_average_data.head()
# dropping 1941



abr_average_data_adjust = abr_average_data.iloc[1:, :]
abr_average_data_adjust.head()
abr_average_data_adjust.set_index('Year')
abr_average_data_adjust.dtypes
# normalising data from selected columns

from sklearn import preprocessing



min_max_scaler = preprocessing.MinMaxScaler()



column_names_to_normalize = ['Tmax(Mean)', 'Rain(Mean)', 'Sun(Mean)']

x = abr_average_data_adjust[column_names_to_normalize].values

x_scaled = min_max_scaler.fit_transform(x)

df_normalised = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = abr_average_data_adjust['Year'])

abr_average_data_adjust[column_names_to_normalize] = df_normalised
df_normalised
# Create a plot

plt.figure(figsize = (30, 12))

plt.plot('Tmax(Mean)', data=df_normalised, marker='', color='orange', linewidth=2)

plt.plot('Rain(Mean)', data=df_normalised, marker='', color='blue', linewidth=2)

plt.plot('Sun(Mean)', data=df_normalised, marker='', color='red', linewidth=2)

plt.legend()

# Your code here



# Check that a figure appears below

step_4.check()
plt.figure(figsize = (20, 12))

sns.regplot('Rain(Mean)', 'Sun(Mean)', data = df_normalised) # correlation between rainfall and sun days
plt.figure(figsize = (20, 12))

sns.residplot('Rain(Mean)', 'Sun(Mean)', data = df_normalised) # checking residual plot
plt.figure(figsize = (20, 12))

sns.heatmap(data = df_normalised)