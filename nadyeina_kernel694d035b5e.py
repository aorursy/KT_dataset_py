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
# load data
df = pd.read_excel("/kaggle/input/covid-in-europe-time-peaksgenderage-group-study/11bb715b-9833-4eda-97e1-8ffa5a7810a5.xlsx")
df.head()
#check size of the datatset
df.shape
#clean data: drop rows, containing unknown or empty age group and gender
df_clean= df.drop(df[(df['Age group'] == "-") | (df['Gender'] == "-") | (df['Gender'] == "Unknown")].index)
df_clean.shape
#in order to make dataframe better plotable, I set up datetime format
df_clean["week"] = df_clean["Reporting week"].str.extract('.*(\d{2})', expand = False) 
df_clean.head()
df_clean["year"] = [2020]*47068
df_clean.head()
#create date column in datetime format 
df_clean['date'] = pd.to_datetime(df_clean.week.astype(str)+
                           df_clean.year.astype(str).add('-2') ,format='%W%Y-%w')
df_clean.date.head()
# drop unnessessary columns ['Reporting week',"Onset week", "Source", "week", "year"]
# drop columns "Cases" and "Deaths", as they contain only one "0" value
df_clean.drop(['Reporting week',"Onset week", "Source", "week", "year", "Cases" , "Deaths"], axis=1, inplace=True) 

df_clean.head()
# genaral descriptive statistics
df_clean.describe(include='all')
df_clean["Reporting country"].value_counts().head(10)
df_clean["Age group"].value_counts()
df_clean["Gender"].value_counts()
# available data about hospitalisation
df_clean["Hospitalisation"].value_counts()
df_clean["Intensive care"].value_counts()
df_clean["Outcome"].value_counts()
df_clean["Country of infection"].value_counts().head(10)
# would be interesting to map the migrations for covid infection
# I would like to check development of all possible outcomes for patientes with confirmed Covid-19 in time
# for this I first make a grouping 
time_outcome_count= df_clean.groupby(['date',"Outcome"]).size().reset_index(name='count')
time_outcome_count.head()
#plot possible Outcomes for patientes with confirmed covid-19 in Europe: development over time

import matplotlib.pyplot as plt

time_outcome_count.pivot_table(index="date", columns="Outcome", values="count", fill_value=0).plot\
                                                                    (figsize=(13, 6)).\
                                                                                        legend(loc=2, 
                                                                                     fontsize=  'large',
                                                                                     edgecolor="None", 
                                                                                     borderpad=1.5,
                                                                                    title="Outcomes",
                                                                                     title_fontsize=14)

plt.rc('axes', edgecolor='gray')  # axis color
plt.ylabel('Number of cases', fontsize=12).set_color('gray')
plt.xlabel("Reporting date", fontsize=12).set_color('gray')
plt.tick_params(axis='x', labelsize=9, colors='gray', which='both')
plt.tick_params(axis='y', labelsize=9, colors='gray', which='both')

plt.annotate('The data provided by \n European Centre for Disease Prevention and Control', 
             xy=(0.98, 0.9), xycoords='axes fraction', size=10,
             ha='right', #horisontal alignment
             va='baseline').set_color('gray')# add text about origin of data
plt.title('Weekly development of outcomes for patientes with confirmed Covid-19 in Europe', fontsize=16)
# in order to look at number of confirmed covid-19 patients and their outcomes in European countries
# first step  is grouping:
df_clean.groupby(["Reporting country","Outcome"]).size().reset_index(name='count').head()
# pivot this grouping table for plotting
country_outcomes_pivot = df_clean.groupby(["Reporting country","Outcome"]).size().reset_index(name='count').\
        pivot_table(index="Reporting country", columns='Outcome', values="count", fill_value=0)
country_outcomes_pivot.head()
# sort pivot table adscending by sum of all columns (total nb of registred cases) and 
# plot with horisontal stacked bars
country_outcomes_pivot.\
assign(tmp=country_outcomes_pivot.sum(axis=1)).sort_values('tmp', ascending=True).drop('tmp', 1).\
plot.barh(stacked=True, figsize=(12, 7), width=0.8).\
legend(fontsize='small',#loc=1,
       edgecolor="None", borderpad=1.2)
plt.rcParams["legend.labelspacing"] = 0.02 # vertical space between legend entries
plt.xlabel('Number of cases', fontsize=10)
plt.annotate('The data provided by European Centre for Disease Prevention and Control', 
             xy=(0.03, 0.02), xycoords='axes fraction', size=10).set_color('gray')# add text about origin of data
plt.title('Outcomes for patientes with confirmed Covid-19', fontsize=16)
# reshape df: count cumularive cases number per reporting week
country_date_outcome = df_clean.groupby(["Reporting country","date","Outcome"]).size().reset_index(name='Count').pivot_table(index=["Reporting country","date"], columns="Outcome", values="Count", fill_value=0).reset_index()

country_date_outcome = country_date_outcome.assign(Total_cases=country_date_outcome.sum(axis=1))
country_date_outcome.head()
#subset selected countries and columns
country_date_outcome[(country_date_outcome["Reporting country"]==
                      "Austria")|(country_date_outcome["Reporting country"]== 
                    "Germany")#|(country_date_outcome["Reporting country"]== "Italy")
                    |(country_date_outcome["Reporting country"]== 
                    "Czechia")|(country_date_outcome["Reporting country"]== 
                    "Poland")|(country_date_outcome["Reporting country"]== 
                    "Portugal")|(country_date_outcome["Reporting country"]== 
                    "Finland")#|(country_date_outcome["Reporting country"]==  "Netherlands")
                   |(country_date_outcome["Reporting country"]== 
                    "Norway")|(country_date_outcome["Reporting country"]== "Sweden")][['Reporting country', 
                                'date','Fatal',
                                'Total_cases']].head()
# melt 2 columns (Fatal and Total_cases) in one (Outcome) with value equal to Fatal or Total_cases value
table_to_plot=country_date_outcome[(country_date_outcome["Reporting country"]==
                      "Austria")|(country_date_outcome["Reporting country"]== 
                    "Germany")#|(country_date_outcome["Reporting country"]== "Italy")
                    |(country_date_outcome["Reporting country"]== 
                    "Czechia")|(country_date_outcome["Reporting country"]== 
                    "Poland")|(country_date_outcome["Reporting country"]== 
                    "Portugal")|(country_date_outcome["Reporting country"]== 
                    "Finland")#|(country_date_outcome["Reporting country"]==  "Netherlands")
                   |(country_date_outcome["Reporting country"]== 
                    "Norway")|(country_date_outcome["Reporting country"]== "Sweden")][['Reporting country', 
                                'date','Fatal',
                                'Total_cases']].melt(id_vars=['Reporting country', 'date']).rename(columns={"value": "Confirmed cases"})
table_to_plot.head()
#plot facets with seaborn
import seaborn as sns
import matplotlib.pyplot as plt
#sns.set(style="ticks") #to get rig gray background
#sns.set_context("paper", rc={"axes.labelsize":16}) 
sns.set(rc={"font.size":20,"axes.titlesize":18,"axes.labelsize":14},style="ticks")

g = sns.relplot(x="date", y="Confirmed cases",
                 col="Reporting country", 
                hue="Outcome",#style="Outcome",
                 kind="line", 
                legend=False, 
                data=table_to_plot, col_wrap=4,
                height=4, aspect=1)#.set(title = "Total amount of confirmed covid-19 cases to death cases in the selected European countries")

g.set_ylabels("Number of cases")
g.set_xlabels("Date")

#g.legend(fontsize='x-large', title_fontsize='20')
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle('Total amount of confirmed Covid-19 cases and deaths in the selected European countries')

plt.legend(("Deaths", "Total cases"),loc='upper right',fontsize='medium',
           edgecolor="None", borderpad=1 )

# in order to plot age distribution in two subplots: total cases and fatal (death) cases, I create 2 pivot tables: 

#First, for selected countries:
country_age_total_pivot=df_clean[(df_clean["Reporting country"]==
                      "Austria")|(df_clean["Reporting country"]== 
                    "Germany")|(df_clean["Reporting country"]== 
                    "Czechia")|(df_clean["Reporting country"]== 
                    "Poland")|(df_clean["Reporting country"]== 
                    "Portugal")|(df_clean["Reporting country"]== 
                    "Finland")|(df_clean["Reporting country"]== 
                    "Norway")|(df_clean["Reporting country"]== "Sweden")].\
    groupby(["Reporting country","Age group"]).size().reset_index(name='count').\
    pivot_table(index="Reporting country", columns="Age group", values="count",
               fill_value=0)
country_age_total_pivot.head()
#Second, for selected countries and dead patiensts:
country_age_death_pivot = df_clean[(df_clean.Outcome == "Fatal")& ((df_clean["Reporting country"]==
                      "Austria")|(df_clean["Reporting country"]== 
                    "Germany")|(df_clean["Reporting country"]== 
                    "Czechia")|(df_clean["Reporting country"]== 
                    "Poland")|(df_clean["Reporting country"]== 
                    "Portugal")|(df_clean["Reporting country"]== 
                    "Finland")|(df_clean["Reporting country"]== 
                    "Norway")|(df_clean["Reporting country"]== "Sweden"))].\
groupby(["Reporting country","Age group"]).size().reset_index(name='count').\
pivot_table(index="Reporting country", columns="Age group", values="count",
               fill_value=0)
country_age_death_pivot.head()
#its occur, that one age group is missing in death dataframe. I set it to "0"
country_age_death_pivot['05-14']=0
country_age_death_pivot = country_age_death_pivot[['00-04','05-14' ,'15-24', '25-49', '50-64', '65-79', '80+']]
country_age_death_pivot.head()
#plot two pivots as subplots
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2)#, sharey=True)
fig.set_size_inches(15, 4)
plt.subplots_adjust(top=0.8)
fig.suptitle('Age groups for patients with confirmed Covid-19 in Europe', fontsize=20)
plt.annotate('The data provided by European Centre for Disease Prevention and Control', 
             xy=(0.1, 0.02), xycoords='axes fraction',  fontsize=10).set_color('gray')
 
country_age_total_pivot.assign(tmp=country_age_total_pivot.sum(axis=1)).sort_values('tmp', ascending=True).drop('tmp', 1).\
plot(kind="barh", stacked=True, ax=axes[0], legend=False).\
legend(fontsize=12,
      title="Age group", edgecolor="None")
axes[0].set_title("Total cases", fontsize=14)
axes[0].set_xlabel('Number of cases', fontsize=12)#.set_color('gray')
axes[0].set_ylabel( "Reporting country", fontsize=12)#.set_color('gray')

country_age_death_pivot.assign(tmp=country_age_death_pivot.sum(axis=1)).sort_values('tmp', ascending=True).drop('tmp', 1).\
plot(kind="barh", stacked=True, ax=axes[1]).\
legend(fontsize=12,#loc=3, 
    title="Age group",edgecolor="None", borderpad=1)
plt.rcParams["legend.facecolor"] = "None"
plt.rcParams["legend.framealpha"] = 0.1
axes[1].set_title("Fatal cases", fontsize=14)
axes[1].set_xlabel('Number of cases', fontsize=12)
axes[1].set_ylabel('')
axes[1].tick_params(axis='y', labelleft=False, left=True)

plt.subplots_adjust(wspace=0.05)
# in order to plot age distribution in two subplots: total cases and fatal (death) cases, I create 2 pivot tables: 

#First, for all confirmed covid-19 cases:
ds_total = df_clean[["Age group","Gender"]].groupby(["Age group","Gender"]).size().reset_index(name='count').pivot_table(index="Age group", columns="Gender", values="count",
               fill_value=0)
ds_total.loc['Total',:] = ds_total.sum(axis=0)
ds_total
#Second, for all fatal covid-19 cases:
ds_death=df_clean[df_clean.Outcome == "Fatal"][["Age group","Gender"]].groupby(["Age group","Gender"]).size().reset_index(name='count').pivot_table(index="Age group", columns="Gender", values="count",
               fill_value=0)#.plot(kind="barh", stacked=False)
ds_death.loc['Total',:] = ds_death.sum(axis=0)
ds_death
#plot two pivots as subplots
fig, axes = plt.subplots(nrows=1, ncols=2)#, sharey=True)
fig.set_size_inches(12, 4)
plt.subplots_adjust(top=0.8)
fig.suptitle('Age vs gender groups for patients with confirmed Covid-19 in Europe', fontsize=18)
plt.annotate('The data provided by European Centre for Disease Prevention and Control', 
             xy=(-0.9, 0.02), xycoords='axes fraction',  fontsize=8).set_color('gray')

ds_total.plot(kind="barh", stacked=False, color=['tab:red', "tab:blue"], ax=axes[0]).legend(loc='lower right')#, 
                                                                                            #title="Gender")
axes[0].set_title("Total cases", fontsize=14)
axes[0].set_xlabel('Number of cases', fontsize=12)#.set_color('gray')
axes[0].set_ylabel('Age group', fontsize=12)

ds_death.plot(kind="barh", stacked=False, color=['tab:red', "tab:blue"], ax=axes[1]).legend(loc='lower right')#, title="Gender")
#plt.rcParams["legend.framealpha"] = 0.1
axes[1].set_title("Fatal cases", fontsize=14)
axes[1].set_xlabel('Number of cases', fontsize=12)#.set_color('gray')
axes[1].set_ylabel('')#.set_color('gray')
axes[1].tick_params(axis='y', labelleft=False, left=True)
plt.subplots_adjust(wspace=0.05)
# create table to plot
icu= df_clean[(df_clean["Intensive care"] != "Unknown")][["Intensive care", 
                                                     "Outcome" ]].\
groupby(["Intensive care","Outcome"]).size().reset_index(name='count').\
pivot_table(index="Intensive care", columns="Outcome", values="count",fill_value=0)
icu
ax= icu.plot(figsize=(11,4), width=0.5,
            kind="barh", stacked=True)
ax.legend(fontsize='x-small',
          edgecolor="None", 
          borderpad=1.8)
# find the values in plot-patches, calculate respecive value in % and insert as text 
for i,j in enumerate (ax.patches):
    if j.get_width()>500:
        if i % 2: #even
            ax.text( j.get_x()+200,j.get_y()+0.2, str(round(j.get_width()*100/icu.sum(axis=1)[1], 0))+'%', fontsize=9) 
        else: #odd
            ax.text( j.get_x()+200,j.get_y()+0.2, str(round(j.get_width()*100/icu.sum(axis=1)[0], 0))+'%', fontsize=9)
                
plt.xlabel('Number of cases', fontsize=12)
plt.ylabel('', fontsize=14)
plt.annotate('The data provided by European Centre for Disease Prevention and Control', 
             xy=(0.36, 0.05), xycoords='axes fraction',  fontsize=10).set_color('gray')# add text about origin of data
plt.title("Outcomes by the ICU (Intensive care unit) treatment vs hospitalisation without ICU", fontsize=16)
