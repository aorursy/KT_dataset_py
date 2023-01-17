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

my_filepath = "../input/crimes-in-boston/crime.csv"



# Fill in the line below: Specify the path of the CSV file to read

my_filepath2 = "../input/crimes-in-boston/offense_codes.csv"



# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data_1 = pd.read_csv(my_filepath, encoding = "ISO-8859-1")

my_data_1.rename(columns={'OFFENSE_CODE':'CODE'}, inplace=True)

my_data_1['CODE'] = my_data_1.CODE.astype('int64')

my_data_2 = pd.read_csv(my_filepath2, encoding = "ISO-8859-1")

my_data_2['CODE'] = my_data_2.CODE.astype('int64')

# Check that a dataset has been uploaded into my_data

step_3.check()
import pandas as pd

my_data = pd.merge(my_data_1, my_data_2, on='CODE')

my_data['OCCURRED_ON_DATE'] = pd.to_datetime(my_data['OCCURRED_ON_DATE'])

my_data = my_data.set_index('OCCURRED_ON_DATE')

my_data['OCCURRED_ON_DATE_NEW'] = my_data.index.date
my_data.tail()
# Print the first five rows of the data

my_data.head()
import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(10,10))



# create aggregation 

temp_df = my_data.groupby(['YEAR']).size().reset_index()

temp_df.columns = ['YEAR','CNT']



# Create a plot

sns.barplot(x="YEAR", y="CNT",data=temp_df) # Your code here



# Check that a figure appears below

step_4.check()
temp_df.size
f, axes  = plt.subplots(2,1,figsize=(20,20))



# create aggregation 

temp_df = my_data.groupby(['OFFENSE_CODE_GROUP']).size().reset_index()

temp_df.columns = ['OFFENSE_CODE_GROUP','CNT']

temp_df = temp_df.sort_values(by=['CNT'], ascending=False)



# Create a plot

sns.barplot(x="OFFENSE_CODE_GROUP", y="CNT", data=temp_df[:5] ,  orient='v' ,ax=axes[0])

sns.barplot(x="OFFENSE_CODE_GROUP", y="CNT", data=temp_df[62:],  orient='v' ,ax=axes[1])





# Check that a figure appears below

step_4.check()
## as the graphs shows that the top crime is larency and the lowest crime is Burgerly 
my_data.groupby(['OFFENSE_CODE_GROUP','OCCURRED_ON_DATE_NEW']).size()
f, axes  = plt.subplots(2,1,figsize=(20,20))

import datetime

# create aggregation 



temp_df = my_data[my_data['OFFENSE_CODE_GROUP'].isin(['Larceny','Medical Assistance','Investigate Person'])].groupby(['OFFENSE_CODE_GROUP','OCCURRED_ON_DATE_NEW','YEAR']).size().reset_index()

temp_df.columns = ['OFFENSE_CODE_GROUP','OCCURRED_ON_DATE','YEAR','CNT']

#temp_df = temp_df.sort_values(by=['CNT'], ascending=False)



# Create a plot

sns.lineplot(x="YEAR", y="CNT",hue='OFFENSE_CODE_GROUP', data=temp_df,ax=axes[0])

sns.swarmplot(x="YEAR", y="CNT",hue='OFFENSE_CODE_GROUP', data=temp_df ,ax=axes[1])





# control x and y limits

#plt.ylim(0, 20)

#plt.xlim(datetime.date(2015,1, 1), datetime.date(2019, 12, 31))









# Check that a figure appears below

step_4.check()