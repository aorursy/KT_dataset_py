import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from functools import reduce
import os
from bokeh.io import output_file,show,output_notebook,push_notebook
from bokeh.plotting import figure
from bokeh.models import HoverTool
from bokeh.layouts import gridplot
output_notebook()

# admissions table contains undergraduate GPA and LSAT information for incoming classes
admit = pd.read_csv('../input/ABA_Admissions.csv')

# import bar passage table - contains passage rates by school, state, and year
# we are dropping calendar year because we care about the year students took the bar,
# not the year the school reported the data
# filter out Puerto Rico schools because we only want US state
# filter out Wisconsin because all WI students automatically pass bar by graduating
bar = pd.read_csv('../input/ABA_BarPassage_Jurisdiction.csv') \
            .drop(['schoolid', 'calendaryear'], axis = 1) \
            .query('state != "PR"') \
            .query('state != "WI"') \
            .rename(index=str,columns={'firsttimebaryear' : 'calendaryear'})
            
# import dataset containing school average first-time pass rate
# the previous dataset 'bar' contains pass rate by state, not overall school pass rate
school_rate = pd.read_csv('../input/ABA_BarPassage_School.csv') \
            .loc[:, ['schoolname', 'firsttimebaryear', 'avgschoolpasspct']] \
            .rename(index=str,columns={'firsttimebaryear' : 'calendaryear'})

# create df of total applications and acceptances by year to be used in plot later
apps = admit[['calendaryear', 'numapps']] \
        .groupby('calendaryear').sum()
                  
# only keep needed columns of 'admit' dataframe
admit_keep = ['schoolname', 'calendaryear', 'numapps', 'uggpa50', 'lsat50']
admit = admit[admit_keep]

# make dataframe of standardized variables, and year adjustments, for future use
admit_std_cols = ['numapps', 'uggpa50', 'lsat50']
admit_std = admit.copy(deep=True)
# change column years and filter prior to standardizing, so that standardized terms
# are only based on years we have bar data for
admit_std['calendaryear'] = admit_std['calendaryear'] + 2
# only keep years from 2017 or earlier because these are the only years we have bar data for
admit_std = admit_std.query('calendaryear <= 2017')
# standardize gpa, lsat, numapps columns
admit_std[admit_std_cols] = admit_std[admit_std_cols].transform(lambda x: (x - x.mean()) / x.std())
# create additional dataset with admission metrics that adds two to calendaryear
# this has the effect of making admissions metrics align with the year students take bar

# create list of all gpa / lsat columns that are needed
metric_cols = list(admit.filter(regex = 'uggpa|lsat').columns)

# create list of other columns needed, to be added to metric_cols list
name_cols = ['schoolname', 'calendaryear']

# combine name and metric lists of column names
name_cols.extend(metric_cols)

# create new dataset, where year is to be adjusted, then add two to year so
# that incoming class year aligns with when students took bar exam
admit_adjust = admit[name_cols]
admit_adjust['calendaryear'] = admit_adjust['calendaryear'] + 2

# change variable names to reflect that they are year adjusted
# all year adjusted variable names will start with 'adj_'
for col in admit_adjust.columns[2:]:
    admit_adjust.rename({col : 'adj_' + col}, axis = 1, inplace = True)

# combine bar, admit, and admit_adjusted datasets to create master dataset
# that contains bar passage by school, jurisdiction, and year;
# and also contains lsat / gpa and lsat / gpa of bar year

# column in admit to merge into master dataset
master = bar.merge(admit, how = 'left', on=['schoolname', 'calendaryear']) \
            .merge(admit_adjust, how = 'left', on=['schoolname', 'calendaryear']) \
            .merge(school_rate, how = 'left', on=['schoolname', 'calendaryear'])

# query 2013 so that all GPA / LSAT columns have values in them 
master.query('calendaryear == 2013').head()
# this dataset calculates the estimated lsat / gpa of takers in a given state

# create copy of master to use for calcualting state averages
# deep = True so that changes to the new dataframe do not impact the old dataframe
master_state = master.copy(deep=True)

# for each row in master_state, multiply takers by metric columns

# identify GPA / LSAT columns
metric_cols = master_state.filter(regex = 'uggpa|lsat').columns
# multiply each GPA / LSAT column by the number of test takers
multiples = master_state[metric_cols].transform(lambda x: master_state['takers'] * x)

# add descriptive info to multiples dataset
multiples[['calendaryear', 'state', 'takers', 'passers']] = master_state[['calendaryear', 'state', 'takers', 'passers']]

# group by calendar year and state, add up takers to create total takers in state,
# and add up multiples
# must add up takers for each different metric column individually, because some columns 
# will not have a metric number for the row, but will have takers; 
# these takers must be excluded
metric_sums = []
for col in metric_cols:
    df = multiples[[col, 'calendaryear', 'state', 'takers']].dropna(axis = 0, how = 'any')
    # group by state and year, and add up total takers and metric total
    df = df.groupby(by = ['calendaryear', 'state'], axis=0)[[col, 'takers']] \
            .sum(skipna = True)
    # divide metric total by total takers
    df['avg_' + col] = df[col] / df['takers']
    # add results to list
    metric_sums.append(df.drop([col, 'takers'], axis = 1).reset_index())
    
# each average state metric dataframe is in its own list; merge all dataframes into one list
state_metrics = reduce(lambda left,right: pd.merge(left,right,on=['calendaryear', 'state'], how = 'outer'), metric_sums)   

# add on the state pass rate reported in the data

# create dataframe that is includes state pass rate - to be merged with state_metrics
state_pass = master[['calendaryear', 'state', 'passpct']].drop_duplicates()
# merge state_metrics dataframe with dataframe including state pass rates
state_metrics = state_metrics.merge(state_pass, on=['calendaryear', 'state'], how='left') \
                             .rename(index=str, columns={'passpct': 'state_pass_pct'})

# query for year 2013 so that all columns have values
state_metrics.query('calendaryear == 2013').head()
# create dataset that only contains school's average pass rate and GPA / lsat
school_pass = master \
    .loc[:, ['schoolname', 'calendaryear', 'adj_uggpa50', 'adj_lsat50', 'avgschoolpasspct']] \
    .drop_duplicates() \
    .dropna()

# add school city and state information to dataset
address = pd.read_csv('../input/ABA_SchoolInformation.csv') \
            .loc[:, ['schoolname', 'schoolcity', 'schoolstate']] \
            .drop_duplicates()

school_pass = school_pass.merge(address, on='schoolname', how='left')
    
# start and end years of data, to be used when iterating through years for subplots 
start_year = school_pass['calendaryear'].min()
end_year = school_pass['calendaryear'].max()

# create function that can plot bar passage rates by gpa / lsat
def gpa_lsat_plot(metric, axis_label, circle_color):
    
    # initialize lsit to store each year's plot
    plots = []
    
    # create data points that display when mouse hovers over point
    hover_tips = [('School name', '@schoolname'),
                  ('Bar pass rate', '@avgschoolpasspct'),
                  ('City', '@schoolcity'),
                  ('State', '@schoolstate')]
    
    # iterate through each year of data, creating plot
    for year in range(start_year, end_year+1):
        p = figure(title=str(year) + " Bar Pass Rates" + " and " + axis_label, width=400, height=400, 
                   tools='hover', tooltips=hover_tips)
        p.circle(metric, "avgschoolpasspct", fill_alpha = 0.8, color=circle_color,
                 source=school_pass[school_pass['calendaryear']==year])
        # if year is odd number then it will be on left side of plot
        # in this case, and only this case, plot needs y-axis label
        if year % 2 == 1:
            p.yaxis.axis_label = 'Bar pass %'
        # x axis label is only needed on years 2016 and 2017
        if year in [2016, 2017]:
            p.xaxis.axis_label = axis_label
        plots.append(p)

    full = gridplot(plots, ncols = 2)
    return(show(full))
gpa_lsat_plot('adj_uggpa50', 'Avg. Undergrad GPA', 'DodgerBlue')
gpa_lsat_plot('adj_lsat50', 'Avg. LSAT', 'Salmon')
#create list of state abbreviations, sorted alphabetically
state_abb = state_metrics['state'].unique()
state_abb.sort()

# regions of states based on alphabetical order of states
regions = [6, 3, 3, 6, 6, 5, 1, 2, 1, 3, 3, 
           6, 4, 6, 4, 2, 5, 2, 5, 1, 2, 1, 
           2, 4, 4, 3, 4, 3, 4, 4, 1, 1, 5,
           6, 1, 2, 5, 6, 2, 1, 3, 4, 3, 5,
           5, 2, 1, 6, 2, 5]

# create mapping of region numbers to region names
region_mapping = {1: 'Northeast', 2: 'Mid-Atlantic', 3: 'Southeast', 
                  4: 'Midwest', 5: 'Mountain', 6: 'West'}

# create column of regions
state_metrics['region'] = state_metrics['state'] \
                            .replace(to_replace = state_abb, value = regions) \
                            .replace(region_mapping)
# create list of regions without duplicates; 
# will iterate through list to create subplots of regions
unique_regions = list(set(state_metrics['region']))

# create plot

# create horizontal white lines across each plot
sns.set_style("whitegrid")
sns.despine()

# increase plot size
fig = plt.figure(figsize=(15, 15))

# increase amount of whitespace between subplots
# needed so that legend can be placed outside of subplots
fig.subplots_adjust(hspace = 0.4, wspace = 0.4)

# iterate through each region, creating a subplot
for region, i in zip(unique_regions, range(1, 7)):
    # plot will have 3 rows and 2 columns
    plt.subplot(3, 2, i)
    # create base plot 
    ax = sns.pointplot(x="calendaryear", y='state_pass_pct', hue="state", palette = 'Set2',
                   data=state_metrics[state_metrics['region'] == region], scale = 0.7)
    # convert y-axis to percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    # axis labels and titles
    plt.xlabel("Year")
    plt.ylabel("State bar passage rate")
    plt.title(region)
    # move legend to the outside and right of the plot
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
# group by year and sum total takers and passers in the given year
nation_pass = master_state.groupby(['calendaryear'])[['takers', 'passers']].sum() \
                .reset_index()
# use the sum of takers and passers to calculate yearly nation-wide pass rate
nation_pass['percentage'] = nation_pass['passers'] / nation_pass['takers']

# create plot
ax = sns.pointplot(x="calendaryear", y='percentage',
                   data=nation_pass)
# convert y-axis to percentile
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
plt.xlabel("Year")
plt.ylabel("Nation-wide bar passage rate")
plt.title('Nation-wide bar passage rate by year')
# create nation-wide aggregates for bar passage, average LSAT, and average GPA

# create copy of master with needed variables
master_nation = master[['calendaryear', 'adj_uggpa50', 'adj_lsat50', 'takers', 'passers']] \
                .copy(deep=True) \
                .dropna(how = 'any')

# multiply gpa and lsat by number of takers for each row
master_nation['total_uggpa50'] = master_nation['adj_uggpa50'] * master_nation['takers']
master_nation['total_lsat50'] = master_nation['adj_lsat50'] * master_nation['takers']
master_nation.drop(['adj_uggpa50', 'adj_lsat50'], axis = 1, inplace = True)

master_nation = master_nation.groupby('calendaryear').sum()

# calcualte nation-wide pass rate, LSAT, and GPA
iterate_cols = ['total_uggpa50', 'total_lsat50', 'passers']
final_cols = ['avg_uggpa50', 'avg_lsat50', 'avg_pass']

for col, result in zip(iterate_cols, final_cols):
    master_nation[result] = master_nation[col] / master_nation['takers']

master_nation.drop(['total_uggpa50', 'total_lsat50', 'takers', 'passers'], 
                   axis = 1, inplace = True)

# add yearly number of applications and matriculants to yearly pass rate / LSAT / GPA data
master_nation = master_nation.merge(apps, on = 'calendaryear', how = 'left')

# scale lsat, gpa, and pass rate columns
master_nation_scale = master_nation.transform(lambda x: (x - x.mean()) / x.std())

# convert for wide to long form for easier plotting
# drop number of matriculants and passers because not needed for plotting
master_nation_scale = master_nation_scale.unstack() \
            .reset_index() \
            .rename(index=str, columns={'level_0' :'metric', 0 :'value'})
        
# plot scaled takers, pass_pct, and lsat / gpa
ax = sns.pointplot(x="calendaryear", y="value", hue="metric", scale = 0.7,
                   data=master_nation_scale, palette="Set2")
leg_handles = ax.get_legend_handles_labels()[0]
leg_descriptions = ['Undergraduate GPA', 'LSAT', 'Pass Rate', '# of Apps']
ax.legend(leg_handles, leg_descriptions, title='')
ax.set(xlabel='Year', ylabel='Scaled Metric')
# identify number of total groupings
# save this number as variable, so that it is the only item that needs to be chagned as the number of groupings changes
num_groups = 7

# create copy of master with needed variables
master_school = master[['schoolname', 'calendaryear', 'adj_uggpa50', 'adj_lsat50', 
                        'avgschoolpasspct', 'numapps']] \
                .copy(deep=True) \
                .dropna(how = 'any') \
                .drop_duplicates()
            
# sum total takers for each school and add to master_school
takers = master.groupby(['schoolname', 'calendaryear'])['takers'].sum().reset_index()
master_school = master_school.merge(takers, on=['schoolname', 'calendaryear'], how='left')

# create number that is scaled uggpa + scaled lsat
# formula scales uggpa and lsat, then adds them together   
master_school['scaled'] = ((master_school['adj_uggpa50'] - master_school['adj_uggpa50'].mean()) / 
                           master_school['adj_uggpa50'].std()) + \
                           ((master_school['adj_lsat50'] - master_school['adj_lsat50'].mean()) / 
                            master_school['adj_lsat50'].std())

# group by calendar year and iterate through each year, calculating gpa / lsat group in each year    
groups = []
for name, group in master_school.groupby('calendaryear'):
        group['group_num'] = pd.qcut(group['scaled'], q=num_groups, labels=False)
        groups.append(group)

# combine all years into one dataset
groups = pd.concat(groups).reset_index(drop=True)

# create additional variables so group gpa, lsat, apps, and bar passage can be calculated on grouped data
groups[['gpa_mult', 'lsat_mult', 'pass_mult']] = groups[['adj_uggpa50', 'adj_lsat50','avgschoolpasspct']] \
    .apply(lambda x: x * groups['takers'])

# group by calendar year and group number
# sum groups so that gruop level gpa / lsat / bar passage can be computed
groups = groups.groupby(['calendaryear', 'group_num']) \
    .sum() \
    .loc[ :, ['takers', 'pass_mult', 'gpa_mult', 'lsat_mult', 'numapps']]
    
# calculate aggregate apps, gpa, lsat, and pass rate
groups[['pass_mult', 'gpa_mult', 'lsat_mult']] = groups[['pass_mult', 'gpa_mult', 'lsat_mult']] \
    .transform(lambda x: x / groups['takers'])
    
groups.rename(index=str, columns={'pass_mult': 'bar_pass',
                                  'gpa_mult': 'avg_gpa', 
                                  'lsat_mult': 'avg_lsat'}, inplace=True)

groups = groups.reset_index()

# multiply bar passage times 100 for percentage
groups['bar_pass'] = groups['bar_pass']*100

# change group numbers so that lowest ranking is 0, and highest is 9
# makes plot legend easier
# change group numbers to more descriptive labels, so plots are easier to understand
gp_descriptions = ['Highest', 'High', 'Above avg.', 'Avg.', 'Below avg.', 'Low', 'Lowest']
groups['group_num'] = (groups['group_num'].astype('int')+num_groups)-num_groups
groups['group_num'] = groups['group_num'].astype(str) \
    .replace(to_replace=['6', '5', '4', '3', '2', '1', '0'], value=gp_descriptions)

# columns to plot, needed in list so that list can be iterated through to make plots
plot_cols = ['bar_pass', 'numapps', 'avg_gpa', 'avg_lsat']
# column descriptions for axis labels
col_desc = ['Bar passage (%)', '# of Applicants', 'Avg. GPA', 'Avg. LSAT']

# create plot
plt.figure(figsize=(12, 12))

for r in range(1, 5):
    plt.subplot(2, 2, r)
    ax = sns.pointplot(x='calendaryear', y=plot_cols[r-1], hue='group_num', hue_order = gp_descriptions,
                       palette="Set2", data=groups)
    ax.set(xlabel='', ylabel=col_desc[r-1])
    # only add legend to second plot
    if r != 2:
        ax.legend_.remove()
    else:
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 14}) \
        .set_title('LSAT / GPA Combo', prop={'size': 14})
# calcualte average yearly change of each group
perc_change = []
gp_change = []
# cycle through each group
for gp in gp_descriptions:
    df = groups[groups['group_num'] == gp]
    # remove non-numeric columns, because you cannot subtract with them
    df = df.drop(['calendaryear','group_num', 'takers'], axis=1) \
            .reset_index(drop=True)
    # within each group, calculate the change from one year to the next
    for r in range(1,5):
        yr_change = pd.DataFrame(df.iloc[r, :] - df.iloc[r-1, :]).T
        # append to lis that will contain one group's changes
        perc_change.append(yr_change)
    # convert list to dataframe,find mean of yearly changes, 
    # and add to list that will contain each group's mean change
    gp_change.append(pd.concat(perc_change).mean())
    # must reset list because it only contains one group's change
    perc_change = []

# create dataframe out of changes and add column for group
gp_change = pd.DataFrame(gp_change)
gp_change['group'] = gp_descriptions
# create plot
plt.figure(figsize=(12, 12))

for r in range(1, 5):
    plt.subplot(2, 2, r)
    ax = sns.barplot(y="group", x=plot_cols[r-1], palette="Set2", data=gp_change)
    ax.set(ylabel='', xlabel='', title=col_desc[r-1] + ' Change')
    ax.invert_xaxis()