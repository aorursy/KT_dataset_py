import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import explained_variance_score as EVS
from sklearn.metrics import mean_squared_log_error
import pydot
from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.model_selection import train_test_split
df_temp = pd.read_csv('/kaggle/input/american-census-2019-person/pppub19/pppub19.csv')
df_temp.shape
df_temp = df_temp[['ERN_VAL', 'A_HRSPAY', 'WSAL_VAL', 'PEARNVAL', 'DIV_VAL',
                   'RNT_VAL', 'DSAB_VAL', 'ED_VAL', 'FIN_VAL', 'INT_VAL',
                   'OI_VAL', # income variables
                   'A_AGE', 'PRDTRACE', 'A_SEX', 'A_HGA', 'A_MARITL',
                   'PEHSPNON', 'MIG_ST', 'MIG_DIV', # demographics
                   'A_LFSR', 'A_USLHRS', 'A_CLSWKR', 'A_MJIND', 
                   'A_MJOCC', # employment variables
                   'MOOP', 'HEA']] # health
# Plotting distribution of the labor force

df_temp['A_LFSR'] = df_temp['A_LFSR'].replace(7, 5)

labels = ['Children or Armed Forces', 'Working' ,'With job, not at work',
          'Unemployed, looking for work', 'Unemployed, on layoff', 
          'Not in labor force']
ax = df_temp['A_LFSR'].plot(kind='hist', figsize=(22,8), fontsize=13, 
                         bins = np.arange(len(labels)+1)-0.5, rwidth = 0.5)
ax.set_title("Employment status histogram", fontsize=22)
ax.set_ylabel("Frequency", fontsize=15);
plt.xlabel('Employment categories', fontsize = 14)
y_pos = np.arange(len(labels))
plt.xticks(y_pos, labels, rotation='horizontal')
fig = plt.show

# Showing normalized distribution of the labor force
df_temp['A_LFSR'].value_counts(normalize = True)

# We can see that only a very small fraction of the sample is in the labor force
# but not under 'working'. So let's continue with this category alone.
df_temp1 = df_temp.loc[(df_temp['A_LFSR'] == 1)]

labels = ['Not in universe', 'Private', 'Federal government',
          'State government', 'Local government', 'Self-employed-incorporated',
          'Self-employed-not incorporated', 'Without pay']
df_temp1['A_CLSWKR'].plot(kind='hist', rwidth = 0.4,  figsize = (27, 7), 
                       bins = np.arange(len(labels)+1)-0.5, fontsize = 12)
plt.xlabel('Worker class categories', fontsize = 14)
y_pos = np.arange(len(labels))
plt.xticks(y_pos, labels)
fig = plt.show

df_temp1['A_CLSWKR'].value_counts(normalize = True)

### df_temp1 = df_temp.loc[(df_temp['ERN_VAL'] > 0) & (df_temp['ERN_SRCE'] == 1)]
# df_temp1.head

# Worker class can be roughly divided into 3 groups - private (76%), 
# government (~14%) & self-employed (~10%). We decided to focus on the private
# sector
df_temp1 = df_temp1.loc[(df_temp1['A_CLSWKR'] == 1)]
df_temp1.shape

# After dropping the other categories we now have about 62,000 records
# Examination of continuous variables
df_continuous = df_temp1[['A_AGE', 'A_USLHRS', # Age & hours 
               'WSAL_VAL', 'ERN_VAL', 'A_HRSPAY', 'PEARNVAL', # Income from work
               'DSAB_VAL', 'ED_VAL', 'FIN_VAL', # Income from grants
               'DIV_VAL', 'RNT_VAL', 'INT_VAL', # additional income sources
               'OI_VAL', # Overall additional income
               'MOOP']] # Total medical out of pocket expenditures
df_continuous.describe(percentiles = [.01, .05, .1, .25, .5, .75, .9, .95,
                                       .98, .99])

# We can see that A_HRSPAY, DIV_VAL, RNT_VAL, DSAB_VAL, ED_VAL, FIN_VAL & OI_VAL
# are mostly '0'. So we decided to drop them.

# Also, ERN_VAL and PEARNVAL have negative values, which should only be possible 
# for self-employed. WSAL_VAL has a '0' value, which is something we don't want.
# So, we crop these observations.

# Another thing to notice is the huge jump in ERN_VAL, WSAL_VAL & PEARNVAL
# from 99th percentile to max. It's a multiplcation by a factor of 4-5, 
# while the increase from 98th to 99th percentile is a multiplcation of about
# a factor of 1.4.
# Dropping useless variables and showing scatterplots
df_continuous1 = df_continuous.drop(columns = ['DIV_VAL', 'RNT_VAL', 'DSAB_VAL',
                                               'ED_VAL', 'FIN_VAL', 'OI_VAL',
                                               'A_HRSPAY', 'INT_VAL'])

df_continuous1 = df_continuous1.loc[df_continuous1['WSAL_VAL'] > 0]
df_continuous1 = df_continuous1.loc[df_continuous1['ERN_VAL'] > 0]
df_continuous1 = df_continuous1.loc[df_continuous1['PEARNVAL'] > 0]

fig = sns.pairplot(df_continuous1)
df_continuous1.describe(percentiles = 
                         [.01, .05, .1, .25, .5, .75, .9, .95, .99])

# We can see that DIV_VAL, RNT_VAL & DSAB_VAL are correlated, though they have 
# outliers, especially in the lower values.
# We should note that this graph is after a log transformation. And we can see 
# in the histograms that the vast majority of the people are in the extreme 
# left side of the graph. 

# We can see that the min for the income variables is 6, while percentile 1 is
# over 1,000. So lets drop the 1st percentile. We can also see that the max
# for the income variables and for MOOP is extremely high compared to the 99th
# percentile. So let's drop the 100th percentile. This is also true for MOOP


# ******************************************************************* #

# ERN_VAL - How much did (you) earn from this (primary) employer before deductions
# WSAL_VAL - total wage and salary earnings
# PEARNVAL - total persons earnings
# MOOP - Total medical out of pocket expenditures
# Dropping outliers of VAL features and of MOOP feature

df_continuous1['to_drop'] = 0

# 1st and 100th percentiles of WSAL_VAL
df_continuous1.loc[df_continuous1['WSAL_VAL'] >= 
                    df_continuous1.WSAL_VAL.quantile(.99), 'to_drop'] = 1
df_continuous1.loc[df_continuous1['WSAL_VAL'] <= 
                    df_continuous1.WSAL_VAL.quantile(.01), 'to_drop'] = 1

# 1st and 100th percentiles of ERN_VAL
df_continuous1.loc[df_continuous1['ERN_VAL'] >= 
                    df_continuous1.ERN_VAL.quantile(.99), 'to_drop'] = 1
df_continuous1.loc[df_continuous1['ERN_VAL'] <= 
                    df_continuous1.ERN_VAL.quantile(.01), 'to_drop'] = 1

# 1st and 100th percentiles of PEARNVAL
df_continuous1.loc[df_continuous1['PEARNVAL'] >= 
                    df_continuous1.PEARNVAL.quantile(.99), 'to_drop'] = 1
df_continuous1.loc[df_continuous1['PEARNVAL'] <= 
                    df_continuous1.PEARNVAL.quantile(.01), 'to_drop'] = 1

# 100th percentiles of MOOP
df_continuous1.loc[df_continuous1['MOOP'] >=
                    df_continuous1.MOOP.quantile(.99), 'to_drop'] = 1

# people over 79 years old 
# df_interval_min.loc[df_interval_min['A_AGE'] >= 80, 'A_AGE'] = np.nan

df_continuous1 = df_continuous1.loc[df_continuous1['to_drop'] == 0]
df_continuous1 = df_continuous1.drop(columns = (['to_drop']))
fig = sns.pairplot(df_continuous1)
df_continuous1.describe(percentiles = 
                         [.01, .05, .1, .25, .5, .75, .9, .95, .99])

# We can see now that the VAL features are less correlated than before, but they
# still represent the same thing, more or less. So we'll only keep ERN_VAL. 
# Also, values higher than 79 in the A_AGE are categorical. So we'll drop them
# too. In addition,we'll drop records in which A_USLHRS is 0 or lower, as these 
# are categorical values. Finally, the distributions of ERN_VAL and MOOP are 
# very skewed. We'll calculate their square root and see if we can get a nicer 
# distribution. 
# Plotting the histogram of ERN_VAL

plt.subplots(figsize=(10, 9))
sns.kdeplot(df_continuous1['ERN_VAL'], shade=True);
# Plotting the histogram of MOOP

plt.subplots(figsize=(10, 9))
sns.kdeplot(df_continuous1['MOOP'], shade=True);
# Dropping categorical data and calculating square roots

#df_continuous1 = df_continuous1.drop(columns = ['WSAL_VAL', 'PEARNVAL'])
df_continuous1 = df_continuous1.loc[df_continuous1['A_AGE'] < 80]
df_continuous1 = df_continuous1.loc[df_continuous1['A_USLHRS'] > 0]
df_continuous1['sqrt_ern_val'] = df_continuous1['ERN_VAL'] ** 0.5
df_continuous1['sqrt_moop'] = df_continuous1['MOOP'] ** 0.5

fig = sns.pairplot(df_continuous1)
df_continuous1.describe(percentiles = 
                         [.01, .05, .1, .25, .5, .75, .9, .95, .99])

# The data looks much nicer now, especially sqrt_ern_val.
# We now have about 55,000 records.
# Plotting the histogram of sqrt_ern_val

plt.subplots(figsize=(10, 9))
sns.kdeplot(df_continuous1['sqrt_ern_val'], shade=True);
# Plotting the histogram of sqrt_ern_val

plt.subplots(figsize=(10, 9))
sns.kdeplot(df_continuous1['sqrt_moop'], shade=True);
### Recreating df 

df_temp = pd.read_csv('/kaggle/input/american-census-2019-person/pppub19/pppub19.csv')
df_temp2 = df_temp[['ERN_VAL','A_AGE', 'PRDTRACE', 'A_SEX', 'A_HGA', 'A_MARITL',
                   'PEHSPNON', 'A_LFSR', 'A_USLHRS', 'A_CLSWKR', 'A_MJOCC', 
                   'MOOP', 'HEA']]

df_temp2 = df_temp2.loc[(df_temp['A_LFSR'] == 1)]
df_temp2 = df_temp2.loc[(df_temp2['A_CLSWKR'] == 1)]

# Dropping unnecessary columns
df_temp2 = df_temp2.drop(columns = ['A_LFSR', 'A_CLSWKR'])

# Cropping '0's and negatives from ERN_VAL
df_temp2 = df_temp2.loc[df_temp2['ERN_VAL'] > 0]

df_temp2['to_drop'] = 0

# Selecting 1st and 100th percentiles of ERN_VAL
df_temp2.loc[df_temp2['ERN_VAL'] > 
                    df_temp2.ERN_VAL.quantile(.99), 'to_drop'] = 1
df_temp2.loc[df_temp2['ERN_VAL'] < 
                    df_temp2.ERN_VAL.quantile(.01), 'to_drop'] = 1

# Selecting 100th percentiles of MOOP
df_temp2.loc[df_temp2['MOOP'] > df_temp2.MOOP.quantile(.99), 'to_drop'] = 1

# Dropping selected rows
df_temp2 = df_temp2.loc[df_temp2['to_drop'] == 0]
df_temp2 = df_temp2.drop(columns = (['to_drop']))

# Dropping ages 80+
df_temp2 = df_temp2.loc[df_temp2['A_AGE'] < 80]

# Dropping less than 1 hours of work
df_temp2 = df_temp2.loc[df_temp2['A_USLHRS'] > 0]

# Calculating square roots
df_temp2['sqrt_ern_val'] = df_temp2['ERN_VAL'] ** 0.5
df_temp2['sqrt_moop'] = df_temp2['MOOP'] ** 0.5

df = df_temp2
df.describe()

# We now have 55,507 records and 13 features
df['A_MJOCC'].plot(kind='hist', rwidth = 0.8, figsize = (10, 5))
plt.show()
df['A_MJOCC'].value_counts(normalize=True)

# There are a lot of occupation types and no obvious way to group them.
# Lets look at the median income of the categories

# ******************************************
# Legend is too long to be shown on plot so it is listed here

# 1 = Management, business, and financial occupations
# 2 = Professional and related occupations
# 3 = Service occupations
# 4 = Sales and related occupations
# 5 = Office and administrative support occupations
# 6 = Farming, fishing, and forestry occupations
# 7 = Construction and extraction occupations
# 8 = Installation, maintenance, and repair occupations
# 9 = Production occupations
# 10 = Transportation and material moving occupations
fig = df.groupby('A_MJOCC')['ERN_VAL'].median().plot(figsize = (10, 6))

# It seems that categories 1 (16.8%) and 2 (20.1%) have a higher median
# income than the rest (63.1%). This finding makes sense. So let's a
# new occupation variable, based on this grouping.
# Creating occupation variable:
# 1 - Management, business, and financial occupations
# 2 - Professional and related occupations
# 3 - Other

occupation_dict = {1: 0, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 2, 9: 2, 10: 2}
df['occupation'] = df['A_MJOCC'].replace(occupation_dict)
df['occupation'].value_counts(normalize = True)
## Set dummy variables for occupation 

df['occ_management'] = np.where(df['occupation'] == 0, 1, 0)
df['occ_professional'] = np.where(df['occupation'] == 1, 1, 0)
df['occ_other'] = np.where(df['occupation'] == 2, 1, 0)
## df['occ_other'].value_counts(normalize = True)
# We already saw that the working hours variables has a major peak at 40.
# So lets divide it into 3 categories. This is done arbitrarily.
df['hours_cat'] = 2
df.loc[df['A_USLHRS'] < 36, 'hours_cat'] = 1
df.loc[df['A_USLHRS'] > 44, 'hours_cat'] = 3
df['hours_cat'].value_counts(normalize = True)
df.groupby('hours_cat')['ERN_VAL'].median()

# Obviously, median income differs by working hours
## Set dummy variables for working hours 
df['less_than_36_hours'] = np.where(df['hours_cat'] == 1, 1, 0) 
df['between_36_and_44_hours'] = np.where(df['hours_cat'] == 2, 1, 0) 
df['more_than_44_hours'] = np.where(df['hours_cat'] == 3, 1, 0)  
df['A_SEX'].value_counts(normalize=True)
# 1 - Male, 2 - Female
# We can see that we have a little bit more males than females
# (Sorry for using the terms males and females and not Men / Women. That's how the dictionary specifies them)
df['PRDTRACE'].value_counts(normalize=True) 

# 1 - White only, 2 - Black only, 4 - Asian only

# From first glance it seems that white greatly predominates everything else
# However, we found out that 91% of latino's are labeled as white
# Are you Spanish, Hispanic, or Latino?
# 1 - Yes
# 2 - No
df['PEHSPNON'].value_counts(normalize=True) 
# Race by latino. We found out that many Latino's are classified as white
pd.crosstab(df.PRDTRACE, df.PEHSPNON, normalize = 'columns')
# We decided to label 'white latinos' as 'latinos'. 
# Creating a new race variable:
# 0 - White, 1 - Latino, 2 - Black, 3 - Asian, 4 - Other

race_dict = {1: 0, 2: 2, 3: 4, 4: 3, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4, 10: 4, 11: 4,
             12: 4, 13: 4, 14: 4, 15: 4, 16: 4, 17: 4, 18: 4, 19: 4, 20: 4, 
             21: 4, 22: 4, 23: 4, 24: 4, 25: 4, 26: 4}

df['race_cat'] = df['PRDTRACE'].replace(race_dict)

# Now assigning white latinos as '1'
df.loc[(df['PEHSPNON'] == 1) & (df['PRDTRACE'] == 1), 'race_cat'] = 1

# Plotting the graph
labels = ['White', 'Latino', 'Black', 'Asian', 'Other']
ax = df['race_cat'].plot(kind='hist', figsize=(14,6), fontsize=13, 
                         bins = np.arange(len(labels)+1)-0.5, rwidth = 0.5)
ax.set_title("Race histogram", fontsize=22)
ax.set_ylabel("Frequency", fontsize=15);
plt.xlabel('Race categories', fontsize = 14)
y_pos = np.arange(len(labels))
plt.xticks(y_pos, labels, rotation='horizontal')
fig = plt.show
df.groupby('race_cat')['ERN_VAL'].median()

# We can see that giving 'white latinos' their own category makes a lot of sense
# as their median income is significantly lower than 'regular' whites. However, 
# we decided to create a binary race variable because the sample simply didn't 
# contain enough people from the other races. 
# Creating a new, binary, variable - white vs not white
df['white'] = np.where(df['race_cat'] == 0, 1, 0)
df['white'].value_counts(normalize=True)

# 0 - Not white
# 1 - White
df.groupby('white')['ERN_VAL'].median()
# 2 - Married - AF spouse present (AF is an airforce program to assist the 
#     spouses of the soldiers)
# 3 - Married - spouse absent (exc.separated)
# 4 - Widowed
# 5 - Divorced
# 6 - Separated
# 7 - Never married
df['A_MARITL'].value_counts(normalize=True)
# We decided to create a new family status variable with 0 - married (53.6%), 
# 1 - never married (31.5%), 2 - divorced (9.6%) & 3 - other (5.3%)
marital_dict = {1: 0, 2: 3, 3: 3, 4: 3, 5: 2, 6: 3, 7: 1}
df['marital_cat'] = df['A_MARITL'].replace(marital_dict)

# Plotting the graph
labels = ['Married', 'Never married', 'Divorced', 'Other']
ax = df['marital_cat'].plot(kind='hist', figsize=(14,6), fontsize=13, 
                            bins = np.arange(len(labels)+1)-0.5, rwidth = 0.4)
ax.set_title("Family status histogram", fontsize=22)
ax.set_ylabel("Frequency", fontsize=15);
plt.xlabel('Family status categories', fontsize = 14)
y_pos = np.arange(len(labels))
plt.xticks(y_pos, labels, rotation='horizontal')
fig = plt.show
# Now lets create 3 dummy variables based on family status
df['married'] = 0
df['never_married'] = 0
df['divorced'] = 0

df.loc[df['marital_cat'] == 0, 'married'] = 1
df.loc[df['marital_cat'] == 1, 'never_married'] = 1
df.loc[df['marital_cat'] == 2, 'divorced'] = 1

# dropping people with other status
df = df.loc[df['marital_cat'] != 3]
df.shape
df['A_HGA'].value_counts(normalize=True) 

# 39 - High school graduate
# 43 - Bachelor's degree
# 40 - Some college but no degree
# 44 - Master's degree
# 42 - Associate degree in college - academic program
# 41 - Associate degree in college - occupation/vocation program

### To be presented in a graph

# By aggregating these results into the categories of 'less than high school
# degree', 'high school degree but less than b.a 'b.a./m.a' we cover 96.4%
# of the population. This leaves out professional schools (1.6%), which varies 
# a lot, and doctorates (2.2%). 
# Lets create a new education variable with 0 - less than high school diploma
# (9.2%), 1 - high_school_no_ba (56.5%), 2 - ba (23.2%) & 3 - ma_doc_pro (9.2%)
education_dict = {39: 1, 43: 2, 40: 1, 44: 3, 42: 1, 41: 1, 46: 3, 37: 0, 
                  45: 3, 36: 0, 35: 0, 38:0, 33: 0, 34: 0, 32: 0, 31: 0}
df['educ_cat'] = df['A_HGA'].replace(education_dict)

# Plotting the graph
labels = ['Less than high school diploma', 'High_school_no_B.A.',
          'B.A.', 'M.A. / Dr. / Pro']
ax = df['educ_cat'].plot(kind='hist', figsize=(14,6), fontsize=13,
                         bins = np.arange(len(labels)+1)-0.5, rwidth = 0.4)
ax.set_title("Education level histogram", fontsize=22)
ax.set_ylabel("Frequency", fontsize=15);
plt.xlabel('Education categories', fontsize = 14)
y_pos = np.arange(len(labels))
plt.xticks(y_pos, labels, rotation='horizontal')
fig = plt.show
# Now lets create 3 dummy variables based on education
df['less_than_high_school'] = 0
df['high_school_no_ba'] = 0
df['ba'] = 0
df['ma_dr_pro'] = 0

df.loc[df['educ_cat'] == 0, 'less_than_high_school'] = 1
df.loc[df['educ_cat'] == 1, 'high_school_no_ba'] = 1
df.loc[df['educ_cat'] == 2, 'ba'] = 1
df.loc[df['educ_cat'] == 3, 'ma_dr_pro'] = 1
df['HEA'].value_counts(normalize=True) 

# 1 - Excellent
# 2 - Very good 
# 3 - Good
# 4 - Fair
# 5 - Poor

# This distribution is not shown in a chart because that would require changing 
# the values of the feature
# Let's create a dummy variable based on health with 'good health' (1+2) and 
# 'bad health' (3+4+5). We realize that 'good' and 'fair' don't logically fit
# into the 'bad' category. However, we believe that whoever chooses 3 in a scale 
# of 5 is not in a very good shape.

df.loc[df['HEA'] <= 2, 'good_health'] = 1
df.loc[df['HEA'] > 2, 'good_health'] = 0
# Income by sex
inc_by_sex = df.groupby(['A_SEX'])['ERN_VAL'].median()
labels = ['Male', 'Female']
ax = inc_by_sex.plot(kind='bar', figsize=(10,6), fontsize=13, width = 0.3)
ax.set_title("Median income by sex", fontsize=22)
ax.set_ylabel("Median income", fontsize=15);
plt.xlabel('Sex categories', fontsize = 14)
plt.ylim(0, 80000)
y_pos = np.arange(len(labels))
plt.xticks(y_pos, labels, rotation='horizontal')
fig = plt.show

# There is a difference in the median income by sex
# Plotting a chart of income by occupation
inc_by_occupation = df.groupby(['occupation'])['ERN_VAL'].median()
labels = ['Management', 'professional', 'Other']
ax = inc_by_occupation.plot(kind='bar', figsize=(10,6), fontsize=13, width = 0.3)
ax.set_title("Median income by occupation", fontsize=22)
ax.set_ylabel("Median income", fontsize=15);
plt.xlabel('Occupation categories', fontsize = 14)
plt.ylim(0, 80000)
y_pos = np.arange(len(labels))
plt.xticks(y_pos, labels, rotation='horizontal')
fig = plt.show

# There is a big difference in the median income by occupation
# Plotting a chart of income by white
inc_by_white = df.groupby(['white'])['ERN_VAL'].median()
labels = ['Not white', 'White']
ax = inc_by_white.plot(kind='bar', figsize=(10,6), fontsize=13, width = 0.3)
ax.set_title("Median income by white", fontsize=22)
ax.set_ylabel("Median income", fontsize=15);
plt.xlabel('White categories', fontsize = 14)
plt.ylim(0, 80000)
y_pos = np.arange(len(labels))
plt.xticks(y_pos, labels, rotation='horizontal')
fig = plt.show

# There is a difference in the median income by white/not white
# Plotting a chart of income by family status
inc_by_family = df.groupby(['marital_cat'])['ERN_VAL'].median()
labels = ['Married', 'Never Married', 'Divorced']
ax = inc_by_family.plot(kind='bar', figsize=(10,6), fontsize=13, width = 0.3)
ax.set_title("Median income by family status", fontsize=22)
ax.set_ylabel("Median income", fontsize=15);
plt.xlabel('family status categories', fontsize = 14)
plt.ylim(0, 80000)
y_pos = np.arange(len(labels))
plt.xticks(y_pos, labels, rotation='horizontal')
fig = plt.show

# There is a difference in median income by family status
# Plotting a chart of income by education level
inc_by_educ = df.groupby(['educ_cat'])['ERN_VAL'].median()
labels = ['No high school diploma', 'High school diploma no B.A.', 'B.A.',
          'M.A. / Dr. / Pro']
ax = inc_by_educ.plot(kind='bar', figsize=(15,8), fontsize=13, width = 0.3)
ax.set_title("Median income by education level", fontsize=22)
ax.set_ylabel("Median income", fontsize=15);
plt.xlabel('Education level categories', fontsize = 14)
plt.ylim(0, 100000)
y_pos = np.arange(len(labels))
plt.xticks(y_pos, labels, rotation='horizontal')
fig = plt.show

# We can see that this aggregation to education levels makes a lot of sense
# income wise
# Plotting a chart of income by health level
inc_by_health = df.groupby(['good_health'])['ERN_VAL'].median()
labels = ['Bad health', 'Good health']
ax = inc_by_health.plot(kind='bar', figsize=(10,6), fontsize=13, width = 0.3)
ax.set_title("Median income by health level", fontsize=22)
ax.set_ylabel("Median income", fontsize=15);
plt.xlabel('Health level categories', fontsize = 14)
plt.ylim(0, 80000)
y_pos = np.arange(len(labels))
plt.xticks(y_pos, labels, rotation='horizontal')
fig = plt.show

# There seems to be a slight difference in the median income by health
# Average income by white and sex
fig = sns.barplot(x="white", y="ERN_VAL", hue="A_SEX", data=df)

# We can see that while white men earn on average a lot, all others earn little
# Average income by married and sex
fig = sns.barplot(x="married", y="ERN_VAL", hue="A_SEX", data=df)

# We can see that married men earn on average more than the rest
# Average income by sex and education
fig = sns.barplot(x="A_SEX", y="ERN_VAL", hue="educ_cat", data=df)

# Notice that women in the M.A. / Dr. / Pro category earn about as much as men 
# in the B.A. category
# Average income by occupation and sex
fig = sns.barplot(x="occupation", y="ERN_VAL", hue="A_SEX", data=df)
df_for_model = df[['sqrt_ern_val', 'A_AGE', 'A_SEX', 'white', 
                   'occ_management', 'occ_professional', 'occ_other', 
                   'less_than_36_hours', 'between_36_and_44_hours',
                   'more_than_44_hours', 'married', 'never_married', 'divorced',
                   'MOOP', 'less_than_high_school', 'high_school_no_ba', 'ba',
                   'ma_dr_pro', 'good_health']]
df_for_model.shape
X = df_for_model.drop(columns = ['sqrt_ern_val'], axis=1)
y = df_for_model['sqrt_ern_val']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=123456,
                                                    shuffle=True)
# We set the min leaf sample to be about 1% of the entire sample
model = DecisionTreeRegressor(min_samples_leaf=500)
model.fit(X_train, y_train)
dot_data = StringIO()  
export_graphviz(model, out_file=dot_data, feature_names=X_test.columns,
                leaves_parallel=True)  
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
Image(graph.create_png(), width=1000) 
for feature, importance in zip(X_test.columns, model.feature_importances_):
    print('{:12}: {}'.format(feature, importance))

# Feature importance can be misleading. Still, it seems that age is very
# important. Occupation, working hours and MOOP also seem important
y_train_pred = model.predict(X_train)
y_train_pred
plt.figure(figsize = (16,9))
ax = sns.scatterplot(x=y_train, y=y_train_pred)
ax.set_title("Prediction figure - training", fontsize=22)
ax.set_ylabel("Actual square root of the income", fontsize=15);
plt.xlabel('Predicted square root of the income', fontsize = 14)
fig = ax.plot(y_train, y_train, 'r')
# Calaculating RMSLE
y_train_squared = y_train ** 2
y_train_pred_squared = y_train_pred ** 2
rmsle = np.sqrt(mean_squared_log_error(y_train_squared, y_train_pred_squared))
print("#### Decision Tree Performance:  ####")
print("Root Mean Squared Logarithmic Error =", round(rmsle, 2))
y_test_pred = model.predict(X_test)
plt.figure(figsize = (16,9))
ax = sns.scatterplot(x=y_test, y=y_test_pred)
ax.set_title("Prediction figure - test", fontsize=22)
ax.set_ylabel("Actual square root of the income", fontsize=15);
plt.xlabel('Predicted square root of the income', fontsize = 14)
fig = ax.plot(y_test, y_test, 'r')
y_test_squared = y_test ** 2
y_test_pred_squared = y_test_pred ** 2
rmsle = np.sqrt(mean_squared_log_error(y_test_squared, y_test_pred_squared))
print("#### Decision Tree Performance:  ####")
print("Root Mean Squared Logarithmic Error =", round(rmsle, 2))
# Plotting the histogram of sqrt_ern_val
plt.subplots(figsize=(15, 10))
sns.kdeplot(y_test, label="Test data");
sns.kdeplot(y_train, label="Train data");
sns.kdeplot(y_test_pred, label="Prediction data");

# The precition data is slightly less skewed but has a much larger kurtosis.
# Still, this is expected.