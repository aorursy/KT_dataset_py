# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import os

from matplotlib import pyplot as plt
# Get all the file names from the tree of path provided. Please note that the downloaded zipped Kaggle file is extracted under 

# the following path . i.e. 'Idexcel\Data\''



path = r'../input/'



files = []

files_others = []

# r=root, d=directories, f = files

for r, d, f in os.walk(path):

    for file in f:

        if '.csv' in file:

            files.append(os.path.join(r, file))

        elif '.pdf' in file:

            files_others.append(os.path.join(r, file))



# Print all the file names from it's directory tree along with it's column name

for file_counter in files:

    #print(file_counter)

    with open(file_counter, 'r', encoding="utf8") as f:

        print(file_counter)

        print(f.readline())
#From the above file name, it's path and their column names, it's found that below three files are required to answer Q2 and Q3

# covid-statistics-by-us-states-daily-updates.csv, hospital-capacity-by-state-20-population-contracted.csv and definitive-healthcare-usa-hospital-beds.csv



df_covid_stat = pd.read_csv(r'../input/uncover/UNCOVER/covid_tracking_project/covid-statistics-by-us-states-daily-updates.csv')

df_hospital_capacity = pd.read_csv('../input/uncover/UNCOVER/harvard_global_health_institute/hospital-capacity-by-state-20-population-contracted.csv')
# All the states where patients are hospitalised, it is expected to get the ventilators ready to face the eventuality. 

# Hence ventilators are required where 'No of positive patients are more than 90% of hospital beds' for that particular state.



# Read statistics data for the last information collected date. As, the data against each dates are running sum of the respective column values.



choose_date = df_covid_stat["date"] == df_covid_stat["date"].max()

df_covid_stat_new = df_covid_stat[choose_date]



# Join the datasets which are required for answering Q2 and select the useful columns



df_ventolator_analysis = pd.merge(df_covid_stat_new, df_hospital_capacity, on='state')

df_ventolator_analysis_columns = df_ventolator_analysis[["state", "positive", "hospitalized","death","total_icu_beds"]]



df_ventolator_analysis_columns.describe()
# Above description shows, hospitalized column has less number of available data. So, Find the unique values of hospitalized column



df_ventolator_analysis_columns["hospitalized"].unique()
df_ventolator_analysis_columns = df_ventolator_analysis_columns.fillna(0)

df_ventolator_analysis_columns.info()
threshold = 0.5

state_potions = df_ventolator_analysis_columns["hospitalized"]/df_ventolator_analysis_columns["total_icu_beds"] > threshold

df_ventolator_analysis_columns = df_ventolator_analysis_columns[state_potions]

df_ventolator_analysis_columns["hospitalized_to_icu_beds_percent"] = df_ventolator_analysis_columns["hospitalized"] * 100/df_ventolator_analysis_columns["total_icu_beds"]

df_ventolator_analysis_columns
# Get the visualization for above analysis



labels = df_ventolator_analysis_columns["state"]

x1 = df_ventolator_analysis_columns["hospitalized_to_icu_beds_percent"]

x = np.arange(len(labels))  # the label locations

width = 0.6  # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, x1, width, label='hospitalized_to_icu_beds',  color = 'r')





# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('hospitalized_to_icu_beds')

ax.set_xlabel('States')

ax.set_title('Hospitalized Vs ICU_beds_percent')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend(loc='best', bbox_to_anchor=(1, 0.5),

          fancybox=True, shadow=True, ncol=5)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



df_definitive_health_care = pd.read_csv('../input/uncover/UNCOVER/esri_covid-19/esri_covid-19/definitive-healthcare-usa-hospital-beds.csv')

df_hospital_capacity = pd.read_csv('../input/uncover/UNCOVER/harvard_global_health_institute/hospital-capacity-by-state-20-population-contracted.csv')



#Get the unique values of num_staffe

df_definitive_health_care["num_staffe"].unique
# Replace the **** on num_staffe to zero.



df_definitive_health_care.loc[(df_definitive_health_care.num_staffe == '****'),'num_staffe']=0

#df_definitive_health_care.head(10)
# Group by total number of staff per state

df_definitive_health_care_columns = df_definitive_health_care[["hq_state","num_staffe"]]

df_definitive_health_care_columns.astype({'num_staffe': 'float'}).dtypes



df_definitive_health_care_columns["num_staffe"] = df_definitive_health_care_columns["num_staffe"].astype(str).astype(int)

print(df_definitive_health_care_columns.dtypes)
df_definitive_health_care.info()
# Aggregate the data



df_state_vs_staff = df_definitive_health_care_columns.groupby(["hq_state"])["num_staffe"].aggregate(sum)
df_state_vs_staff_group = pd.DataFrame(df_state_vs_staff).reset_index()

df_state_vs_staff_group.columns = ['hq_state', 'num_staffe']

#df_state_vs_staff_group["num_staffe"]
df_protective_analysis = pd.merge(df_state_vs_staff, df_hospital_capacity, how='inner', left_on='hq_state', right_on = 'state')

df_protective_analysis_columns = df_protective_analysis[["state", "adult_population", "num_staffe"]]



df_protective_analysis_columns.describe()
threshold = 360

state_options = df_protective_analysis_columns["adult_population"]/df_protective_analysis_columns["num_staffe"] > threshold

df_protective_analysis_columns = df_protective_analysis_columns[state_options]

df_protective_analysis_columns["population_to_med_staff_percent"] = df_protective_analysis_columns["adult_population"]/df_protective_analysis_columns["num_staffe"]

df_protective_analysis_columns.sort_values("population_to_med_staff_percent" , ascending=False, inplace=False)
import seaborn as sns

import matplotlib.pyplot as plt

sns.set(color_codes=True)

# Get the visualization



labels = df_protective_analysis_columns["state"]

x1 = df_protective_analysis_columns["population_to_med_staff_percent"]

x = np.arange(len(labels))  # the label locations

width = 0.6  # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, x1, width, label='Population_to_med_staff',  color = 'r')





# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Population_to_med_staff')

ax.set_xlabel('States')

ax.set_title('Population Vs Medical Staff')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend(loc='best', bbox_to_anchor=(1, 0.5),

          fancybox=True, shadow=True, ncol=5)