import pandas as pd
import numpy as np

import operator

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv('../input/multipleChoiceResponses.csv')
df.drop([0],inplace=True)

######
# Start of time column
df['Time from Start to Finish (seconds)'] = df['Time from Start to Finish (seconds)'].apply(int)
# Rejecting those who answered questions too fast:
df = df[df['Time from Start to Finish (seconds)']>60]
# drop "Time" column
df.drop(['Time from Start to Finish (seconds)'],axis=1,inplace=True)
# End of time column
######

def rename_some_salaries(salary):
    if (salary!=salary): return 'unknown'
    elif (salary=='I do not wish to disclose my approximate yearly compensation'): 
        return 'unknown'
    return salary

df['Q9']=df['Q9'].apply(lambda x: rename_some_salaries(x))

# drop those who didn't disclose disclose their salary
df = df[(df['Q9']!='unknown')]

# all valid salary ranges:
all_salaries = ['0-10,000','10-20,000','20-30,000','30-40,000','40-50,000','50-60,000','60-70,000',
                       '70-80,000','80-90,000','90-100,000','100-125,000','125-150,000','150-200,000',
                       '200-250,000','250-300,000','300-400,000','400-500,000','500,000+']

# remove students
df = df[df['Q6']!='Student']
df = df[df['Q7']!='I am a student']

# def: plot salary distribution
def plot_salary_distribution(df,col='Q9',x_label='yearly compensation, [USD]',order=None):
    fig, ax2 = plt.subplots(figsize=(18,6))
    g2 = sns.countplot(x=col,data=df, order=order, ax=ax2)
    g2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
    g2.set_title('Yearly compensation distribution',fontsize=20)
    g2.set_ylabel('')
    g2.set_xlabel(x_label,fontsize=16)
    #ax2.set(yscale="log")
    for p in ax2.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        ax2.annotate(p.get_height(), (x.mean(), y), ha='center', va='bottom')
        
# plot salary distribution
smth0 = plot_salary_distribution(df,order=all_salaries)
# all valid salary ranges:
all_salaries = ['0-10,000','10-20,000','20-30,000','30-40,000','40-50,000','50-60,000','60-70,000',
                       '70-80,000','80-90,000','90-100,000','100-125,000','125-150,000','150-200,000',
                       '200-250,000','250-300,000','300-400,000','400-500,000','500,000+']
        
# shorten names of some countries
def rename_some_countries(x):
    if (x=='United States of America'): return 'USA'
    if (x=='United Kingdom of Great Britain and Northern Ireland'): return 'United Kingdom'
    if (x=='Iran, Islamic Republic of...'): return 'Iran'
    if (x=='Hong Kong (S.A.R.)'): return 'Hong Kong'
    return x

df['Q3']=df['Q3'].apply(lambda x: rename_some_countries(x))

df['country']=df['Q3']


# distribution over salary ranges
df_for_plot = df[(df['country']=='USA') | (df['country']=='India')]

fig, ax2 = plt.subplots(figsize=(18,6))
g2 = sns.countplot(x='Q9',data=df_for_plot, 
                   order=all_salaries, ax=ax2, hue='country')
smth0 = g2.set_xticklabels(ax2.get_xticklabels(),rotation=90)
smth1 = g2.set_title('Yearly compensation distribution',fontsize=20)
smth2 = g2.set_ylabel('')
smth3 = g2.set_xlabel('yearly compensation [USD]',fontsize=16)
import math

# this dict is for 'quantify_average'
dict_averages = {'0-10,000':5000,'10-20,000':15000,'20-30,000':25000,
                '30-40,000':35000,'40-50,000':45000,'50-60,000':55000,
                '60-70,000':65000,'70-80,000':75000,'80-90,000':85000,
                '90-100,000':95000,'100-125,000':112500,'125-150,000':137500,
                '150-200,000':175000,'250-300,000':275000,'200-250,000':225000,
                '300-400,000':350000,'400-500,000':450000,'500,000+':650000}

def quantify_enumerate(x):
    for i in range(1,len(all_salaries)+1):
        if (x==all_salaries[i-1]): return int(i)
    return -100

def quantify_average(x):
    return dict_averages[x]

def quantify_log_average(x):
    return math.log(dict_averages[x])

def order_subset(subset,whole_set):
    ordered = ['']*len(subset)
    i = 0
    for s in whole_set:
        if s in subset:
            ordered[i]=s
            i = i+1
    return ordered

def ranges_to_numerical(df, col_name='Q9', whole_set = all_salaries):
    subset = df[col_name].unique()
    ordered_subset = order_subset(subset,whole_set)
    dict_ranges = dict(df[col_name].value_counts())
    N_tot = df.shape[0]
    N_values = len(ordered_subset)
    N_values1 = len(dict_ranges)
    if (N_values!=N_values1):
        print('In ranges_to_numerical: (N_values!=N_values1)')
        return
    Ns_given_range = [0]*N_values
    percentile = 0
    ordered_dict = {}
    for key in ordered_subset:
        N_i = dict_ranges[key]
        percentile = percentile+100*N_i/N_tot
        ordered_dict[key] = percentile
    return ordered_dict

def quantify_percentile(x, ordered_dict):
    if (x in ordered_dict.keys()):
        return round(ordered_dict[x],3)
    else:
        return -100
    
# calculate world wide percentile
ordered_dict = ranges_to_numerical(df, col_name='Q9', whole_set = all_salaries)
df['world_wide_percentile'] = df['Q9'].apply(lambda x: 
                                             quantify_percentile(x,ordered_dict))

df['enumerate_salary_ranges'] = df['Q9'].apply(lambda x: quantify_enumerate(x))

df['salary_averages'] = df['Q9'].apply(lambda x: quantify_average(x))

df['salary_log_averages'] = df['Q9'].apply(lambda x: quantify_log_average(x))

df_USA = pd.DataFrame(df[df['country']=='USA'])
df_India = pd.DataFrame(df[df['country']=='India'])

# calculate country-wide (local) percentile
ordered_dict_USA = ranges_to_numerical(df_USA, col_name='Q9', whole_set = all_salaries)
df_USA['country_percentile'] = df_USA['Q9'].apply(lambda x: 
                                             quantify_percentile(x,ordered_dict_USA))
ordered_dict_India = ranges_to_numerical(df_India, col_name='Q9', whole_set = all_salaries)
df_India['country_percentile'] = df_India['Q9'].apply(lambda x: 
                                             quantify_percentile(x,ordered_dict_India))

df_USA_India = pd.concat([df_USA,df_India])
#all_salaries = ['0-10,000','10-20,000','20-30,000','30-40,000','40-50,000','50-60,000','60-70,000',
#                       '70-80,000','80-90,000','90-100,000','100-125,000','125-150,000','150-200,000',
#                       '200-250,000','250-300,000','300-400,000','400-500,000','500,000+']

# enumerate
# the code to enumerate salary ranges is available above
fig = plt.figure(figsize=(18,6))
ax = fig.add_subplot(111)
p1 = plt.hist(df_USA['enumerate_salary_ranges'],bins=18)
p2 = plt.hist(df_India['enumerate_salary_ranges'],bins=18,alpha=0.5)
p3 = plt.title('Distribution by compensation ranges enumerated',fontsize=20)
p4 = plt.xlabel('compensation range number',fontsize=16)
p5 = plt.xlim(1,18)

# salary averages
# the code to compute salary range averages is available above
fig = plt.figure(figsize=(18,6))
ax = fig.add_subplot(111)
p1 = plt.hist(df_USA['salary_averages'],bins=65)
p2 = plt.hist(df_India['salary_averages'],bins=65,alpha=0.5)
p3 = plt.title('Distribution by compensation range averages',fontsize=20)
p4 = plt.xlabel('compensation range average [USD]',fontsize=16)
p5 = plt.xlim(0,650000)
# salary log averages
# the code to compute salary range log-averages is available above
fig = plt.figure(figsize=(18,6))
ax = fig.add_subplot(111)
p1 = plt.hist(df_USA['salary_log_averages'],bins=24)
p2 = plt.hist(df_India['salary_log_averages'],bins=24,alpha=0.5)
p3 = plt.title('Distribution by compensation range log averages',fontsize=20)
p4 = plt.xlabel('compensation range log average [log(USD)]',fontsize=16)
p5 = plt.xlim(8,14)
# world-wide percentile
# the code to compute world-wide percentile is available above
fig = plt.figure(figsize=(18,6))
ax = fig.add_subplot(111)
p1 = plt.hist(df_USA['world_wide_percentile'],bins=50)
p2 = plt.hist(df_India['world_wide_percentile'],bins=50,alpha=0.5)
p3 = plt.title('Distribution by world-wide compensation percentile',fontsize=20)
p4 = plt.xlabel('world-wide compensation percentile',fontsize=16)
p5 = plt.xlim(10,100)
# country percentile
# the code to compute local (country) percentile is available above
fig = plt.figure(figsize=(18,6))
ax = fig.add_subplot(111)
p1 = plt.hist(df_USA['country_percentile'],bins=50)
p2 = plt.hist(df_India['country_percentile'],bins=50,alpha=0.5)
p3 = plt.title('Distribution by local compensation percentile',fontsize=20)
p4 = plt.xlabel('local compensation percentile',fontsize=16)
p5 = plt.xlim(0,100)