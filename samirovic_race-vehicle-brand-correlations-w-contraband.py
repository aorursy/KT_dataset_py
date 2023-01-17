# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import itertools

sns.set(style="whitegrid")

dtype={'user_id': int}



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



#    Load the data set : put a limit when writing drafts to avoid loading time

lim = -1

if lim>0:

    data = pd.read_csv("../input/IL.csv").head(lim)

else:

    data = pd.read_csv("../input/IL.csv")
#    Informations about the dataset :

Cols = data.columns

print(Cols)

N_controls = len(data['id']) # Number of Controls

# ------------------------------------------------------------------------------------------------------------



# Informations loading + Preprocessing :

        # Loading of informations

Ages = np.sort(data['driver_age'].unique()).tolist()

Races = np.sort(data['driver_race'].unique()).tolist()

Violations = np.sort(data['violation'].unique()).tolist()

Vehicles = np.sort(data['vehicle_type'].unique()).tolist()

data['stop_time'] = data['stop_time'].astype(str) # Put the hours in string format

data['search_conducted'] = data['search_conducted'].astype(str)

data['contraband_found'] = data['contraband_found'].astype(str)  

# ------------------------------------------------------------------------------------------------------------
data.drop(data[data.stop_time=='nan'].index, inplace=True)

data['hour'] = data['stop_time'].map(lambda x : x[:2])

data['vehicle_brand'] = data['vehicle_type'].map(lambda x : x[:3])

# ------------------------------------------------------------------------------------------------------------
fig1 = plt.subplots(figsize=(10,15))

cat = ['driver_gender','driver_race','violation','stop_outcome']

length=len(cat)

for i,j in itertools.zip_longest(cat,range(length)): 

    plt.subplot(np.ceil(length/2),2,j+1)

    plt.subplots_adjust(hspace=.5)

    df_count = data[i].value_counts()

    l = len(data[i])

    df_perc = df_count/l

    sns.barplot(df_perc.index, df_perc.values, alpha=0.7)

    plt.xticks(rotation=90)

    plt.title("Repartition of " + i.replace('_',' ')+ " (%)")

# ------------------------------------------------------------------------------------------------------------
# Group the results by the hour of day :

    # Arrests by hour

Arrests_per_hour = data.groupby(['hour'])['id'].count()

Arrests = data['id'].count()

Arrests_per_hour_perc = Arrests_per_hour/Arrests*100

fig_ar = plt.figure(figsize=(12,6))

sns.barplot(Arrests_per_hour_perc.index,Arrests_per_hour_perc.values, alpha=1.0)

plt.xlabel('Hour of the day', fontsize=14)

plt.ylabel('Percentage of Arrests (%)', fontsize=14)

plt.title('Ratio of Arrests in the day (%)')

plt.show()
# Group the results by the hour of day :

    # Age Mean by hourMethod = 'A'

Age_by_hour = data.groupby(['hour'])['driver_age'].mean()

fig_ag = plt.figure(figsize=(12,6))

sns.barplot(Age_by_hour.index,Age_by_hour.values)

plt.xlabel('Hour of the day', fontsize=14)

plt.ylabel('Mean of the Age', fontsize=14)

plt.title('Average Age vs Hour of the Day')

plt.show()

# Violin Plot of Arrest by Category

var_name_x = 'driver_race'

var_name_y = 'driver_age'

def Violin_Plot_Num(var_name_x,var_name_y):

    data[var_name_y] = data[var_name_y].astype(np.float64)

    col_order = np.sort(data[var_name_x].unique()).tolist()

    fig3 = plt.figure(figsize=(12,6))

    sns.violinplot(x=var_name_x , y=var_name_y, data=data,palette="Set3", order=col_order)

    plt.xlabel(var_name_x, fontsize=12)

    plt.ylabel('y', fontsize=12)

    plt.title("Distribution of " + var_name_y.replace('_',' ') +" variable with "+var_name_x.replace('_',' ') , fontsize=15)

    plt.show()

Violin_Plot_Num('driver_race','driver_age')
cat_race = Races

length=len(cat_race)

fig6=plt.subplots(figsize=(10,15))

for i,j in itertools.zip_longest(cat_race,range(length)): 

    plt.subplot(np.ceil(length/2),2,j+1)

    plt.subplots_adjust(hspace=1)

    df_outcome = data[data['driver_race']==i]['stop_outcome'].value_counts()

    out_total = len(data[data['driver_race']==i]['stop_outcome'])

    df_outcome_perc = df_outcome/out_total*100

    sns.barplot(df_outcome_perc.index, df_outcome_perc.values,palette="Set3", alpha=0.7)

    plt.xticks(rotation=90)

    plt.title("Repartion of Stop Outcome % for " + i)
cat_race = Races

length=len(cat_race)

fig6=plt.subplots(figsize=(10,15))

for i,j in itertools.zip_longest(cat_race,range(length)): 

    plt.subplot(np.ceil(length/2),2,j+1)

    plt.subplots_adjust(hspace=1)

    df_search = data[data['driver_race']==i]['search_conducted'].value_counts()

    search_total = len(data[data['driver_race']==i]['search_conducted'])

    df_search_perc = df_search/search_total*100

    sns.barplot(df_search_perc.index, df_search_perc.values,palette="Blues_d", alpha=0.7)

    plt.xticks(rotation=90)

    plt.title("Ratio of Search Conducted  % for " + i)
fig10 = plt.figure(figsize=(12,6))

df_sch = data[data['search_conducted']=='True'].groupby(['driver_race'])['contraband_found'].count()

df_contra_found = data[data['contraband_found']=='True'].groupby(['driver_race'])['contraband_found'].count()

df_contra_found_perc = df_contra_found/df_sch*100

sns.barplot(df_contra_found_perc.index, df_contra_found_perc.values,palette="Set2", alpha=0.7)

plt.xticks(rotation=90)

plt.title("Ratio of Contraband found when search is conducted (%) ")
df_total_cars = data['vehicle_brand'].count()

df_10 = data.groupby(['vehicle_brand'])['id'].count().sort_values(axis=0,ascending=False).head(10)

df_10_perc = df_10/df_total_cars*100

df_total_speed = data[data['violation']=='Speeding']['violation'].count()

df_spd = data[data['violation']=='Speeding'].groupby(['vehicle_brand'])['violation'].count().sort_values(axis=0,ascending=False).head(10)

df_spd_perc = df_spd/df_total_speed*100



df_contra_found = data[data['contraband_found']=='True'].groupby(['vehicle_type'])['contraband_found'].count()

df_search = data[data['search_conducted']=='True'].groupby(['vehicle_type'])['contraband_found'].count()

df_search = df_search[(df_search.values>50)]

df_contra_found_perc = df_contra_found/df_search*100

df_contra_found_perc = df_contra_found_perc.sort_values(axis=0,ascending=False).head(10)



def H_barplot(dataframe,namex,namey,title,color):

    plt.figure(figsize=(12,6))

    sns.barplot(dataframe.values,dataframe.index,palette=color,orient="h").set_ylabel("Sequential")

    plt.xlabel(namex , fontsize=14)

    plt.ylabel(namey, fontsize=14)

    plt.title(title)

    plt.show()



H_barplot(df_10_perc,'% of Arrests','Vehicle Brands','Top ten vehicle brands for number of Arrests',"BuGn_d")

H_barplot(df_spd_perc,'% of Speeding Violation','Vehicle Brands','Top ten vehicle brands for number of Speeding Violation',"Set2")
H_barplot(df_contra_found_perc,'% of cases Contraband found', 'Vehicle Brands', 'Top fifteen vehicle brands for % of contraband found when search are conducted',"Set1")