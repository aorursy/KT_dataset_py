import scipy as sp
import scipy.io
from scipy import stats
import os
import numpy as np
import pandas as pd
import glob
import csv
from collections import Iterable
import matplotlib.pylab as plt
import matplotlib.patches as patch
from datetime import datetime, timedelta
import matplotlib.dates as mdates
plt.style.use('seaborn-white')
plt.close('all')
font = {'family' : 'Arial',
        'weight' : 'normal',
        'size'   : 16}
plt.rcParams['xtick.direction'] = 'out'
plt.rc('font', **font)
plt.rc('xtick', labelsize=16) 
plt.rc('ytick', labelsize=16)
plt.rc('axes', labelsize=16)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# Load relevant DonorsChoose data
projects_df = pd.read_csv('../input/Projects.csv')
donations_df = pd.read_csv('../input/Donations.csv')
display(projects_df.info())
display(donations_df.info())
display(projects_df.head(2))
display(donations_df.head(2))
finished_projects_df = projects_df[projects_df['Project Current Status'].str.contains('Fully Funded|Expired')].copy()

###convert date columns to datetime
rel_date_cols = ['Project Posted Date', 'Project Fully Funded Date', 'Project Expiration Date']
finished_projects_df[rel_date_cols] = finished_projects_df[rel_date_cols].apply(pd.to_datetime)

#### a few 'Project Expiration Date' are empty - fill in with 'Project posted date' + 4 months
nulls = finished_projects_df['Project Expiration Date'].isnull()
finished_projects_df.loc[nulls, 'Project Expiration Date'] = (finished_projects_df.loc[nulls, 'Project Posted Date'] +
                                                              pd.DateOffset(months = 4))

### fill in 'project end date' for projects that expired with 'Project expiration date'
expired = finished_projects_df['Project Fully Funded Date'].isnull()
finished_projects_df.loc[expired, 'Project Fully Funded Date'] = finished_projects_df.loc[expired, 'Project Expiration Date']
finished_projects_df = finished_projects_df.rename(columns = {'Project Fully Funded Date':'Project End'})

### certain projects did not have a valid expiration date - fill those in with posting date + 4 months
nulls = finished_projects_df['Project End'].isnull()
finished_projects_df.loc[nulls, 'Project End'] = finished_projects_df.loc[nulls, 'Project Posted Date'] + pd.DateOffset(months = 4)
finished_projects_df['Days Open'] = ((finished_projects_df['Project End'] - finished_projects_df['Project Posted Date'])/
                                     np.timedelta64(1, 'D')).astype(int)

### remove projects that have expiration dates that precede posting date (data entry error?)
finished_projects_df = finished_projects_df[finished_projects_df['Days Open']>0]
finished_projects_df= finished_projects_df.reset_index(drop = True)
# finished_projects_df = finished_projects_df.sample(frac = 0.2, random_state=42) #subset for faster prototyping
# finished_projects_df= finished_projects_df.reset_index(drop = True)
## Group all finished projects (expired + fully funded) by week
finished_projects_df['Project End'] = pd.to_datetime(finished_projects_df['Project End'])
all_fin_proj_posted_grouped = finished_projects_df.groupby(pd.Grouper(key = 'Project Posted Date', freq = 'W'))

## Group all fully funded project by week of fully funded date
ff_proj_completed_df = finished_projects_df[finished_projects_df['Project Current Status'] == 'Fully Funded']
ff_proj_completed_grouped = ff_proj_completed_df.groupby(pd.Grouper(key = 'Project End', freq = 'W'))

## Group all fully funded project by week of posted date
ff_proj_posted_df = finished_projects_df[finished_projects_df['Project Current Status'] == 'Fully Funded']
ff_proj_posted_grouped = ff_proj_posted_df.groupby(pd.Grouper(key = 'Project Posted Date', freq = 'W'))

## Group all expired projects by week of posted date
exp_proj_df = finished_projects_df[finished_projects_df['Project Current Status'] == 'Expired']
exp_proj_posted_grouped = exp_proj_df.groupby(pd.Grouper(key = 'Project Posted Date', freq = 'W'))
## Find start and end of each school year
weeks = pd.Series(ff_proj_completed_grouped.groups).reset_index()
school_start_inds = weeks['index'].apply(lambda x: x.week == 35)
school_end_inds = weeks['index'].apply(lambda x: x.week == 22)

school_start_inds.iloc[0] = True
school_end_inds.iloc[-1] = True

school_year = list(zip(weeks.loc[school_start_inds, 'index'], weeks.loc[school_end_inds, 'index']))
school_year
## count the number projects funded each week
weekly_funded = ff_proj_completed_grouped.size()

## plot
fig,ax = plt.subplots(1,1, figsize = (15,4), sharex = True)
ax.bar(weekly_funded.index, weekly_funded, 7)

# plot patches to denote school years
ymax = ax.get_ylim()[1]
for start, stop in school_year:
    ax.add_patch(patch.Rectangle((mdates.date2num(start),0), 
                                 mdates.date2num(stop)-mdates.date2num(start), ymax, color = 'C4', alpha = 0.15))

## find outlier weeks where a large number of projects were funded
spikes = weekly_funded.nlargest(3)

## annotate outliers and school years
annotations = ['#BestSchoolDay(2018)', '#BestSchoolDay(2017)','Bill and Melinda Gates match']
for text, (spike_ind, spike) in zip(annotations, spikes.items()):
    ax.annotate(text, xy=(mdates.date2num(spike_ind)-10, 
                spike+500), xytext=(mdates.date2num(spike_ind)-500,spike+3000), arrowprops = {'arrowstyle': '->'})
    
ax.annotate('School\n  year', xy=(mdates.date2num(school_year[4][0])+1, 
                27250), xytext=(mdates.date2num(school_year[4][0])+80,25000), arrowprops = {'arrowstyle': '->'}, color = 'C4')
ax.annotate('', xy=(mdates.date2num(school_year[4][1])-1, 
                27250), xytext=(mdates.date2num(school_year[4][1])-75,27250), arrowprops = {'arrowstyle': '->'}, color = 'C4')

#clean up plot and label axes
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel('Number of projects fully funded')
xrange = ax.get_xlim()

fig.savefig('Funded_proj_timeline.png', format = 'PNG')
## list the days where each project was active
ranges = finished_projects_df[['Project Posted Date','Project End']].applymap(lambda x: int(x.timestamp()))
ranges = ranges.apply(lambda x: range(x[0], x[1], 86400), axis = 1) #86400 is number of seconds in a day
ranges = np.concatenate(ranges.apply(list).values)

## bin and count how many projects were active during each day
active_hist, xvals = np.histogram(ranges, bins = np.arange(ranges.min(), ranges.max()+1, 86400))
to_time_vec = np.vectorize(datetime.utcfromtimestamp)
xvals = to_time_vec(xvals[:-1])

active_proj = pd.DataFrame({'date':xvals, 'Active Projects':active_hist})
## calculate how many projects were funded each day
daily_fund_rate = ff_proj_completed_df.groupby(pd.Grouper(key = 'Project End', freq = 'D')).size()
daily_fund_rate = daily_fund_rate.reset_index()
daily_fund_rate['Project End'] = pd.to_datetime(daily_fund_rate['Project End'])
daily_fund_rate = daily_fund_rate.rename(columns={0: 'Number funded'})

# ## calculate fraction of active projects funded each day 
# active_proj = active_proj.merge(daily_fund_rate, left_on = 'date', 
#                                 right_on = 'Project End', how = 'inner').drop('Project End', axis = 1)
# active_proj['frac_funded'] = active_proj.apply(lambda x: x[2]/x[1], axis = 1)
## count number of projects that were posted each week
weekly_posted_all = all_fin_proj_posted_grouped.size()

## plot
fig,ax = plt.subplots(1,1, figsize = (15,4), sharex = True)
ax.bar(xvals, active_hist, 1, alpha = 0.3, color = 'C7')
ax.bar(weekly_posted_all.index, weekly_posted_all, 7, color = 'C1', alpha = 0.7)
ax.bar(weekly_funded.index, weekly_funded, 7, color = 'C0', alpha = 0.7)

## label things
ax.text(0.3, 0.9, 'Total live projects', color = 'C7', transform=ax.transAxes)
ax.text(0.3, 0.82, 'Total projects posted', color = 'C1', transform=ax.transAxes)
ax.text(0.3, 0.74, 'Total projects fully funded', color = 'C0', transform=ax.transAxes)

## clean up figure
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlim(xrange)
ax.set_ylabel('Number of projects')

fig.savefig('Stacked_proj_timeline.png', format = 'PNG')
from numpy.polynomial.polynomial import polyfit

fig,ax = plt.subplots(1,1, figsize = (6,6))

## plot each week based on number of projects posted vs funded
ax.scatter(weekly_posted_all, weekly_funded.iloc[:-1], alpha = 0.4, s = 20,color = 'C3')

## calculate best fit line and plot
m, b, r_value, p_value, std_err = scipy.stats.linregress(weekly_posted_all, weekly_funded.iloc[:-1])
ax.plot(np.array([500,10000]), np.array([500,10000]) * m + b, 'k', linewidth = 1)
text

## clean up figure and label things
ax.set_ylabel('Number of projects posted')
ax.set_xlabel('Number of projects funded')
plt.xlim(0, 15000)
plt.ylim(0, 15000)
plt.gca().set_aspect('equal', adjustable='box')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout()
fig.savefig('correlation_posted_funded.png', format = 'png')

print('r2:' + str(r_value**2))
print('p: ' + str(p_value))
#count number of projects posted that either were eventually funded or expired
weekly_posted_exp = exp_proj_posted_grouped.size()
weekly_posted = ff_proj_posted_grouped.size()

# plot
fig,ax = plt.subplots(1,1, figsize = (15,4), sharex = True)
ax.bar(weekly_posted.index, weekly_posted, 7, color = 'C1', alpha = 0.5)
ax.bar(weekly_posted_exp.index, weekly_posted_exp, 7, color = 'C1')

# label things
for start, stop in school_year:
    ax.add_patch(patch.Rectangle((mdates.date2num(start),0), 
                                 mdates.date2num(stop)-mdates.date2num(start), ymax, color = 'C4', alpha = 0.15))
ax.set_ylabel('Number of projects posted')
ax.text(0.2, 0.75, 'Fully funded', color = 'C1', transform=ax.transAxes, alpha = 0.7)
ax.text(0.2, 0.65, 'Expired', color = 'C1', transform=ax.transAxes)

# clean up figure
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

fig.savefig('Posted_proj_timeline_exp.png', format = 'PNG')
## caclulate the fraction of projects that are posted each week that ultimately get funded
frac_ffunded_by_post_day = weekly_posted/weekly_posted_all
frac_ffunded_by_post_day = frac_ffunded_by_post_day.reset_index()

## exclude any projects that are posted at during the last school year since it is incomplete
frac_ffunded_by_post_day = frac_ffunded_by_post_day[frac_ffunded_by_post_day['Project Posted Date'] < school_year[-1][0]]

## group projects by week across the years to calculate means ect.
frac_ffunded_by_post_day['week'] = frac_ffunded_by_post_day['Project Posted Date'].apply(lambda x: x.week)
frac_ffunded_by_post_day = frac_ffunded_by_post_day.rename(columns = {0:'Fraction funded'})
frac_ffunded_by_post_day = frac_ffunded_by_post_day[frac_ffunded_by_post_day['week']<53]

## calc means and sem for the fraction of funded projects
mean_frac_funded_by_post_day = frac_ffunded_by_post_day.groupby('week').mean().reset_index()
sem_frac_funded_by_post_day = frac_ffunded_by_post_day.groupby('week')['Fraction funded'].apply(stats.sem).reset_index()
## reorder weeks to fit the academic year
order = np.concatenate([np.where(mean_frac_funded_by_post_day['week']>21)[0], 
                        np.where(mean_frac_funded_by_post_day['week']<22)[0]])

mean_frac_funded_by_post_day = mean_frac_funded_by_post_day.iloc[order].reset_index(drop=True)
sem_frac_funded_by_post_day = sem_frac_funded_by_post_day.iloc[order].reset_index(drop = True)
order
## transform week representation from int to month
mean_frac_funded_by_post_day['week'] = mean_frac_funded_by_post_day['week'].apply(lambda x: datetime.strptime(str(x)+'-0', '%W-%w'))
mean_frac_funded_by_post_day['week'] = mean_frac_funded_by_post_day['week'].apply(lambda x: datetime.strftime(x, '%B'))
## Need to repeat the above but want to group number funded and total number of projects by week before calculating fraction
## want to do this in order to calculate fraction funded during four week periods that we will compare statistically with 
## other four week periods

funded_proj_week = pd.concat([weekly_posted.to_frame(name = 'funded'), weekly_posted_all.to_frame(name = 'total')], axis = 1)
funded_proj_week = funded_proj_week.reset_index()
funded_proj_week['week'] = funded_proj_week['Project Posted Date'].dt.week
funded_proj_week['year'] = funded_proj_week['Project Posted Date'].dt.year
funded_proj_week = funded_proj_week[funded_proj_week['year']<2018]
funded_proj_week = funded_proj_week.groupby('year')

def calc_epoch_ff(week_range, grouped):
    frac_ff = funded_proj_week.apply(lambda x: np.sum(x.loc[(x['week']>week_range[0]) &
                                                            (x['week']<=week_range[1]), ['funded', 'total']], axis = 0))
    frac_ff = (frac_ff['funded']/frac_ff['total']).dropna()
    return frac_ff

## calculate the fraction of projects funded for three planned comparisons
early_frac_ff = calc_epoch_ff([32,36], funded_proj_week)
late_frac_ff = calc_epoch_ff([13,17], funded_proj_week)
middle_frac_ff = calc_epoch_ff([39,43], funded_proj_week)
display(pd.DataFrame([early_frac_ff, late_frac_ff,middle_frac_ff], index = ['start', 'end', 'middle']))

## calculate if means are significantly different for the different epochs
start_end_test = stats.mannwhitneyu(early_frac_ff,late_frac_ff)
start_middle_test = stats.mannwhitneyu(early_frac_ff,middle_frac_ff)

## calculate effect size
def cliffs_d(stat,x1,x2):
    return (2*stat/(len(x1)*len(x2)))-1
    
print('start vs end pvalue: ' + str(start_end_test.pvalue))
print('start vs middle pvalue: ' + str(start_middle_test.pvalue))

print('start vs end effect size: ' + str(cliffs_d(start_end_test.statistic, early_frac_ff,late_frac_ff)))
print('start vs middle effect size: ' + str(cliffs_d(start_middle_test.statistic, early_frac_ff,middle_frac_ff)))

fig, ax = plt.subplots(1,1, figsize = (7,6))

## plot fraction funded by week
ax.plot(np.arange(0,52), mean_frac_funded_by_post_day['Fraction funded'], color = 'C2')
ax.fill_between(np.arange(0,52),
                mean_frac_funded_by_post_day['Fraction funded'] -sem_frac_funded_by_post_day['Fraction funded'],
                mean_frac_funded_by_post_day['Fraction funded'] +sem_frac_funded_by_post_day['Fraction funded'], alpha =0.4,color = 'C2')

## label epochs that we compared
week_order = order +1
start = np.where(order == 33)[0]
end = np.where(order == 14)[0]
middle = np.where(order == 40)[0]

ax.plot([start,start,start+4,start+4], [0.84,0.84,0.84,0.84], 'C7')
ax.plot([end,end,end+4,end+4], [0.84,0.84,0.84,0.84], 'C7')
ax.plot([middle,middle,middle+4,middle+4], [0.84,0.84,0.84,0.84],'C7')
ax.plot([(start+start+4)/2,(start+start+4)/2, (end+end+4)/2,(end+end+4)/2], [0.9,0.92,0.92,0.9], 'C7')
ax.plot([(middle+middle+4)/2,(middle+middle+4)/2, (start+start+4)/2,(start+start+4)/2], [0.85,0.87,0.87,0.85], 'C7')
ax.scatter((end+start+4)/2.1, .94, color = 'k', marker = '*')
ax.text((middle+start+4)/2.1, .88, 'ns')

## label academic year
ax.add_patch(patch.Rectangle((14,0), 37, ymax, color = 'C4', alpha = 0.15))

## clean up figure
xticks = np.sort(np.unique(mean_frac_funded_by_post_day['week'].values, return_index = True)[1])
ax.set_xticks(xticks)
ax.set_xticklabels(mean_frac_funded_by_post_day['week'].values[xticks], rotation = 65)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylim(0,1); ax.set_xlim(0,51)
ax.set_ylabel('Fraction of projects funded')
ax.set_xlabel('Date of project posting')
ax.text(0.5, 0.4, 'School year', transform = ax.transAxes, color = 'C4')
fig.tight_layout()
fig.savefig('fraction of projects funded')

