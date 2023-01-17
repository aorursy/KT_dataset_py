# Loading libraries for python

%matplotlib inline

%pylab inline





import numpy as np # Linear algebra

import pandas as pd # Data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Data visualization

import seaborn as sns # Advanced data visualization

import re # Regular expressions for advanced string selection

import missingno as msno # Advanced missing values handling

import pandas_profiling # Advanced exploratory data analysis

from scipy import stats # Advanced statistical library

sns.set()

from pylab import rcParams

from scipy import stats # statistical operations

import warnings

warnings.filterwarnings("ignore");



# changing graph style following best practices from Storyteling with Data from Cole Knaflic [3]

sns.set(font_scale=2.5);

sns.set_style("whitegrid");
# Reading data parsing dates in correct format for users DataFrame

engagement = pd.read_csv(r"../input/engagement.csv")

users = pd.read_csv(r"../input/users.csv", parse_dates=['registration'], infer_datetime_format=True)
# Initially analyzed each dataset using pandas_profiling

# This is a very insightful library for initial EDA, I left the code as comments due to the large space it takes on screen

# Fork the notebook and run the codes below to try out how powerful pandas_profiling is!



#pandas_profiling.ProfileReport(users)

#pandas_profiling.ProfileReport(engagement[['hrs_per_week', 'browser', 'program']])



# Getting summary statistics for early conclusions excluding user_id

engagement.describe(include="all").drop(['user_id'], axis=1)
# Let's have a macro view of the hours per week by program

fig1 = sns.boxplot(x='program', y='hrs_per_week', data=engagement);



# All visualizations will follow a similar styling following best practices from Storytelling with Data [3][8][9]

fig1 = plt.gcf()

fig1.set_size_inches(10, 8)

plt.title('Hours per week by program', loc='left', y=1.1);

plt.ylabel("Hours per week [hrs]");

plt.xlabel("Program type");

plt.text(-0.5, 18.5, "Note that median values are similar, but the Binge program has larger variance.\nAlso, Binge has more 'extreme users' denoted by the observed outliers and max value.", fontsize=16, color='grey', style='italic', weight='semibold',);

plt.plot();
# Checking mean hours per week per program

engagement.groupby(['program']).mean()[['hrs_per_week']]
# Joining the tables on User_ID for hrs_per_week by country and checking by possible nulls

# Merging on user_is and adding a simple year column for aggregations

df = pd.merge(engagement, users, on='user_id')

df['year'] = df.registration.dt.year



# Checking for null values and exploring further using missingno library

#df.isnull().sum()

#msno.bar(df);
# Checking engagement distribution for the entire sample (both programs)

fig, ax = plt.subplots()

fig = sns.distplot(df.hrs_per_week, kde=False)

fig.grid(False)

fig = plt.gcf()

fig.set_size_inches(14, 8)

plt.title("Distribution of hours per week for the entire sample (both programs)", loc="left", y=1.1, fontsize=20);

plt.text(-0.8, 5350, "Note that the proportion of users that used the platform between 0 and 1 hour is very significant", fontsize=16, color='grey', style='italic', weight='semibold',);

plt.ylabel("User count");

plt.xlabel("Hours per week [h]");

sns.despine(left=False);

ax.annotate('window shoppers', xy=(0.2, 2700), xytext=(0.6, 3700), fontsize=11,

            arrowprops=dict(facecolor='black', shrink=0.05));

plt.plot();
# User distributions per country

# Not much of a difference on the aggregate view;

# On the granular view (by program by country), we have once again the conclusion of extreme users in favor to the Binge program;

# However, still on the granular view, all medians are higher for the Drip program while standard deviations are lower, indicating a more regular community.

df.groupby(['country', 'program']).agg({'hrs_per_week':[np.mean, np.std]})
# Let's check the program distribution for the given period

sns.countplot(x='year', hue='program', data=df);

fig2 = plt.gcf()

fig2.set_size_inches(10, 8)

plt.title('Registrations per year for each program', loc="left", y=1.1);

plt.text(-0.5, 37000, 'Most of the experiment happend during 2016, and data is balanced for the given period', fontsize=16, color='grey', style='italic', weight='semibold',);

plt.legend(loc=0);

plt.ylabel("Count of registrations");

plt.xlabel("Year");

plt.plot();
# Checking the distribution of registered users throughout the period of analysis

pivot_table = pd.pivot_table(df, index=['country', 'program'], columns='year', aggfunc=np.count_nonzero, values=['registration'], margins=True, )

pivot_table



# Distribution of registered users by country throughout the period of analysis

#sns.catplot('country', col="year", col_wrap=2, hue='program', data=df, kind="count", height=11, aspect=.6);
# As a best practice prior to hypothesis testing we evaluate eventual Confidence Interval overlaps at 95%

# So here we're visualizing Z confidence intervals for the sample means by program



fig = sns.pointplot(x="hrs_per_week", y="program", data=engagement, join=False, capsize=0.1)

fig = plt.gcf()

fig.set_size_inches(10, 6)

plt.title("95% Confidence interval for the sample means", loc="left", fontsize=20, y=1.15);

plt.text(4.57307, -0.6, 'The overlap seen between CI suggests that there is no real difference \non the true mean difference between populations ', fontsize=16, color='grey', style='italic', weight='semibold',);

plt.ylabel("");

plt.xlabel("Sample mean");

sns.despine(left=False)

plt.plot();
# Storing necessary statistics for double checking the statistical test using [7]



# Hours per week mean difference between program

mean_difference = engagement.loc[engagement.program == "binge", "hrs_per_week"].mean() - engagement.loc[engagement.program == "drip", "hrs_per_week"].mean()



# Mean and standard deviation from hour per week per program

mean_hpw_binge = df[df.program == 'binge'].hrs_per_week.mean()

std_hpw_binge = df[df.program == 'binge'].hrs_per_week.std()

n_binge = df[df.program == 'binge'].shape[0]



mean_hpw_drip = df[df.program == 'drip'].hrs_per_week.mean()

std_hpw_drip = df[df.program == 'drip'].hrs_per_week.std()

n_drip = df[df.program == 'drip'].shape[0]
# Building the confidence interval for the difference of the sample means 

SE = np.sqrt((np.power(std_hpw_binge, 2)/n_binge) + (np.power(std_hpw_drip, 2)/n_drip))



figure(figsize=(7, 6))

plot(mean_difference, "bo", markersize=9)

plot(mean_difference - 1.96*SE, "_", markersize=15, color='b')

plot(mean_difference + 1.96*SE, "_", markersize=15, color='b')

plt.axvline(x=0, ymin=0.045, ymax=0.95);

plt.title("95% Confidence interval for the sample means", loc="left", fontsize=20, y=1.18);

plt.text(-0.055, 0.046, 'The fact that our CI for the mean difference goes \nbeyond zero reinforces the no statistical significant difference \nbetween hour per week for the different programs', fontsize=16, color='grey', style='italic', weight='semibold',);

plt.ylabel("");

plt.plot();
# T-test using statsmodel for getting our p-value 

# (probability of observing a difference in hours per week as at least or more extreme 

# than the one observed given that the null hypothesis is true)

pval = stats.ttest_ind(df[df.program == 'binge'].hrs_per_week, df[df.program == 'drip'].hrs_per_week, equal_var = False)

print("P-value = {}".format(pval[1].round(4)))
# Histograms for both program types so we can see how many "window shoppers" each one had before eliminating them



bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

fig, ax = plt.subplots(1, 2)

df.loc[(df.program == 'binge') & (df.country == 'Germany'), ['hrs_per_week']].hist(bins=bins, ax=ax[0]);

df.loc[(df.program == 'drip') & (df.country == 'Germany'), ['hrs_per_week']].hist(bins=bins, ax=ax[1]);

ax[0].set_ylim([0, 4200])

ax[0].set_title("Binge distribution")

ax[0].grid(False)

ax[1].set_ylim([0, 4200])

ax[1].set_title("Drip distribution")

ax[1].grid(False)

for i in ax.flat:

    i.set(xlabel='Hours per week [h]', ylabel='Count of users')

fig.suptitle("Distribution of hours per week by program for Germany", fontsize=28, x=0.465, y=1.03);

plt.text(-22, 4550, "Note that the majority of 'window shoppers' were on the Binge program, \nsuch behavior is seens proportionally for France and Italy as well.", fontsize=16, color='grey', style='italic', weight='semibold',);

fig.set_size_inches(16, 10)

plt.plot();
# Checking summary statistics between programs after removing window shoppers

df_eval0 = df.loc[df.hrs_per_week > 1].groupby(['program']).agg({'hrs_per_week':[np.mean, np.std], 'registration':np.count_nonzero})

df_eval0.columns.set_levels(['sample size (n)','mean','std.dev'],level=1,inplace=True)

df_eval0
# Checking summary statistics between programs for every country after removing window shoppers

df_eval1 = df.loc[df.hrs_per_week > 1].groupby(['country', 'program']).agg({'hrs_per_week':[np.mean, np.std], 'registration':np.count_nonzero})

df_eval1.columns.set_levels(['sample size (n)','mean','std.dev'],level=1,inplace=True)

df_eval1
# Testing the difference between means from the entire sample

pval = stats.ttest_ind(df.loc[(df.program == 'binge') & (df.hrs_per_week > 1)].hrs_per_week, df.loc[(df.program == 'drip')  & (df.hrs_per_week > 1)].hrs_per_week, equal_var = False)

print("P-value for the difference between means after removing window shoppers = {}".format(pval[1].round(6)))
# T-test using statsmodel for getting our p-value - Germany

pval_ger = stats.ttest_ind(df[(df.program == "binge") & (df.country == "Germany") & (df.hrs_per_week >= 1)].hrs_per_week, df[(df.program == "drip") & (df.country == "Germany") & (df.hrs_per_week >= 1)].hrs_per_week, equal_var = False)

pval_fra = stats.ttest_ind(df[(df.program == "binge") & (df.country == "France") & (df.hrs_per_week >= 1)].hrs_per_week, df[(df.program == "drip") & (df.country == "France") & (df.hrs_per_week >= 1)].hrs_per_week, equal_var = False)

pval_it = stats.ttest_ind(df[(df.program == "binge") & (df.country == "Italy") & (df.hrs_per_week >= 1)].hrs_per_week, df[(df.program == "drip") & (df.country == "Italy") & (df.hrs_per_week >= 1)].hrs_per_week, equal_var = False)



print("P-values assesment by country after removing window shoppers:\nP-value for Germany = {} \nP-value for France = {} \nP-value for Italy = {}".format(pval_ger[1].round(6), pval_fra[1].round(6), pval_it[1].round(6)))
# Removing outliers and evaluating the new sequence of statistical tests

df_nooutliers = df[pd.Series(np.abs(stats.zscore(df.hrs_per_week)) < 3)]

df_eval2 = df_nooutliers.loc[df_nooutliers.hrs_per_week > 1].groupby(['country', 'program']).agg({'hrs_per_week':[np.mean, np.std], 'registration':np.count_nonzero})

df_eval2.columns.set_levels(['sample size (n)','mean','std.dev'],level=1,inplace=True)

df_eval2
# Testing the difference between means from the entire sample

pval = stats.ttest_ind(df_nooutliers.loc[(df_nooutliers.program == 'binge') & (df_nooutliers.hrs_per_week > 1)].hrs_per_week, df_nooutliers.loc[(df_nooutliers.program == 'drip')  & (df_nooutliers.hrs_per_week > 1)].hrs_per_week, equal_var = False)

print("P-value for the difference between means after removing window shoppers and outliers = {}".format(pval[1].round(6)))



# Checking summary statistics between programs after removing window shoppers and outliers

#df_eval3 = df_nooutliers.loc[df_nooutliers.hrs_per_week > 1].groupby(['program']).agg({'hrs_per_week':[np.mean, np.std], 'registration':np.count_nonzero})

#df_eval3.columns.set_levels(['sample size (n)','mean','std.dev'],level=1,inplace=True)

#df_eval3
# T-test using statsmodel for getting our p-value - Germany

pval_ger = stats.ttest_ind(df_nooutliers[(df_nooutliers.program == "binge") & (df_nooutliers.country == "Germany") & (df_nooutliers.hrs_per_week >= 1)].hrs_per_week, df_nooutliers[(df_nooutliers.program == "drip") & (df_nooutliers.country == "Germany") & (df_nooutliers.hrs_per_week >= 1)].hrs_per_week, equal_var = False)

pval_fra = stats.ttest_ind(df_nooutliers[(df_nooutliers.program == "binge") & (df_nooutliers.country == "France") & (df_nooutliers.hrs_per_week >= 1)].hrs_per_week, df_nooutliers[(df_nooutliers.program == "drip") & (df_nooutliers.country == "France") & (df_nooutliers.hrs_per_week >= 1)].hrs_per_week, equal_var = False)

pval_it = stats.ttest_ind(df_nooutliers[(df_nooutliers.program == "binge") & (df_nooutliers.country == "Italy") & (df_nooutliers.hrs_per_week >= 1)].hrs_per_week, df_nooutliers[(df_nooutliers.program == "drip") & (df_nooutliers.country == "Italy") & (df_nooutliers.hrs_per_week >= 1)].hrs_per_week, equal_var = False)



print("P-values assesment by country after removing window shoppers and outliers:\nP-value for Germany = {} \nP-value for France = {} \nP-value for Italy = {}".format(pval_ger[1].round(6), pval_fra[1].round(6), pval_it[1].round(6)))