# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
fString = "/kaggle/input/tennis_courts.csv" # take the print out from the beginning and re-use it as a variable, rather than retyping the string each time we need it
tennisData = pd.read_csv(fString) # create a pandas dataframe and assign it to the tennisData variable
tennisData.head() # take a look at the first 5
# renames all of the columns to appropriately match their representation in the .csv
# we need to rename longitude at the start, or else we'll end up with 2 columns called longitude
# place it into a new dataframe so we still have the raw data if necessary
col = {'longitude': 'combined', 'name' :'address', 'address' :'city', 'city' :'state', 'state' : 'zip_code', 'zip_code' : 'type', 'type' : 'count', 'count' :'clay', 'clay' : 'wall' ,'wall' : 'grass', 'grass' : 'indoor', 'indoor' : 'lights', 'lights' : 'proshop', 'proshop' : 'latitude', 'latitude' : 'longitude'}
tennis_fix = tennisData.rename(columns=col)
tennis_fix # preview the data
# remove the columns that we're not interested in currently
# put it into a new dataframe so if we need to go back, we don't have issues
tennis_strip = tennis_fix.drop(['zip_code', 'address', 'combined'], axis=1)

# preview data from the last step
tennis_strip
import matplotlib.pyplot as plt # import some plotting libraries if we need them

bins = np.arange(0, tennis_strip['count'].max() + 1.5) - 0.5 # we want to center the bins at the integers; calculations to do so
plt.figure(figsize=(20,10)) # set the x,y size of the figure so it's easier to read
data = plt.hist(tennis_strip['count'], bins) # plot a histogram using the data from our count and the bins we just created

# make some changes to the tick marks on both axes
plt.xticks(np.arange(0, tennis_strip['count'].max(), 2)) # modifying the x axis ticks so that it's easier to see where the columns align
plt.yticks(np.arange(0, data[0].max(), 500)) # do the same for y ticks
# look at some visualizations for type splits
tennis_strip['type'].value_counts().plot(kind='pie', legend=True, autopct='%.2f', fontsize=15, figsize=(10,10)) # create a pie chart using the type column and show percents; adjust font size + figure size
# that was kind of boring, so let's do some more breakdowns of this information using histograms
bins2 = np.arange(0, tennis_strip[tennis_strip['type'] == 'Public']['count'].max() + 1.5) - 0.5 # bin centering
plt.figure(figsize=(20,10)) 

data2 = plt.hist(tennis_strip[tennis_strip['type'] == 'Public']['count'], bins2) # create histogram

# set ticks for axes
plt.xticks(np.arange(0, tennis_strip[tennis_strip['type'] == 'Public']['count'].max(), 1)) # x axis ticks
plt.yticks(np.arange(0, data2[0].max(), 200)) # y axis ticks           
# how about private ones?
bins3 = np.arange(0, tennis_strip[tennis_strip['type'] == 'Homeowners Community']['count'].max() + 1.5) - 0.5 # bin centering
plt.figure(figsize=(20,10))

data3 = plt.hist(tennis_strip[tennis_strip['type'] == 'Homeowners Community']['count'], bins3) # new histogram for private

# set ticks for axes
plt.xticks(np.arange(0, tennis_strip[tennis_strip['type'] == 'Homeowners Community']['count'].max(), 1)) # x axis ticks
plt.yticks(np.arange(0, data3[0].max(), 250)) # y axis ticks
# remove some additional columns that we don't really need/want to work with

tennis_type = tennis_strip.drop(['city', 'state', 'wall', 'proshop'], axis=1)
tennis_type
num_clay = len(tennis_type[tennis_type['clay'] == True]['clay']) # if the data says that it's a hard court, keep it in the data, but then just get the value; if we wanted to do further analysis, we would keep the dataframe that meets this criteria
num_grass = len(tennis_type[tennis_type['grass'] == True]['grass']) # do the same for grass
num_hard = len(tennis_type) - num_clay - num_grass # if it's not grass or clay, it must be a hard court

labels = ['clay', 'grass', 'hard'] # create labels for the bar graph
plt.barh(labels,[num_clay,num_grass,num_hard]) # make a horizontal bar chart