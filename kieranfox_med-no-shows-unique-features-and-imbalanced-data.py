import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from scipy import stats
from scipy.stats.distributions import chi2
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Columns 3 and 4 have date information
df = pd.read_csv('/kaggle/input/noshowappointments/KaggleV2-May-2016.csv', parse_dates=[3,4])
print(df.info())
print(df.head())
# There are some spelling mistakes and some columns have slightly
# misleading names
df.rename(columns = {'Hipertension': 'Hypertension',
                         'Handcap': 'Handicap',
                        'ScheduledDay': 'AppointMade',
                        'AppointmentDay': 'AppointFor'}, inplace = True)

# Appointment ID is unique for each isntance and so can be used for the index
df.set_index('AppointmentID', inplace=True)
# Replace M and F with 1 and 0 to make it easier to test with statistical models
df['Gender'].replace(('M', 'F'), (1, 0), inplace=True)

print(df.info())

# No show is 'Yes' and 'No'. Making these into dummies gives us flexibility in
# how we use the data for statistics
dummies = pd.get_dummies(df['No-show'])
df = pd.concat((df, dummies), axis = 1)

# The overall probability of no show
noshow_prob = df.Yes.sum() / (df.Yes.sum() + df.No.sum())
print(noshow_prob)
print(df.describe())
# Removing entries where the age is less than 0
df = df.loc[df['Age'] >= 0]
# showing the smallest and largest age categories and how many are in each
print(df.groupby('Age')['PatientId'].count())

# Creating a simple histogram to show distribution
df['Age'].hist(bins = 15)
plt.title('The distribution of ages')
plt.xlabel('Age')
plt.ylabel('Number of appointments')
plt.show()
# GRouping all appointments by age and then calculating the probability of 
# no show for each age
df_age = pd.concat([df.groupby('Age')['No'].sum(), df.groupby('Age')['Yes'].sum()], axis=1)
df_age['Probability of no-show'] = df_age['Yes'] / (df_age['No'] + df_age['Yes'])
df_age.reset_index(inplace=True)
df_age.head()

# Plotting the probabiltiy of no-show by age
df_age.plot(kind='scatter', x='Age', y='Probability of no-show')
# Creating a line at the overall probability
plt.axhline(noshow_prob, c='r', label='Overall probability')
plt.legend()
plt.title('All appointments grouped into age categories')
plt.show()
# reducing all ages over 85 to 85
max_age = 85
df['Age'][df['Age'] >= max_age] = max_age

# Calculating the new probabilities with all ages capped at 85
df_age = pd.concat([df.groupby('Age')['No'].sum(), df.groupby('Age')['Yes'].sum()], axis=1)
df_age['Probability of no-show'] = df_age['Yes'] / (df_age['No'] + df_age['Yes'])
df_age.reset_index(inplace=True)

# PLotting the new probabilities with all ages capped at 85
df_age.plot(kind='scatter', x='Age', y='Probability of no-show')
plt.axhline(noshow_prob, c='r', label='Overall probability')
plt.legend()
plt.title('All ages over 84 grouped into one age group')
plt.show()
print('Correlation for Handicap is', df['Handicap'].corr(df['Yes']))

# splitting the handicap feature into dummies and loking for a correlaion with no-show
# for each of them in turn
dummies_hand = pd.get_dummies(df['Handicap'])
for dum in dummies_hand.columns:
    cor = dummies_hand[dum].corr(df['Yes'])
    print(f'Correlation for handicap - {dum} is {cor}')

# Grouping all appointments by patient handicap and calculating the probability
# of no-show for each of them
df_hand = pd.concat([df.groupby('Handicap')['No'].sum(), df.groupby('Handicap')['Yes'].sum()], axis=1)
df_hand['Probability of no-show'] = df_hand['Yes'] / (df_hand['No'] + df_hand['Yes'])
df_hand.reset_index(inplace=True)

# Plotting the probability of no-show for each of the 5 handicap categories
df_hand.plot(kind='bar', x='Handicap', y='Probability of no-show')
plt.axhline(noshow_prob, c='r', label='Overall probability')
plt.legend()
plt.title('Handicap split into its categories')
plt.ylabel('Probability of no-show')
plt.show()
# This is a list of all the binary features
binary_cats = ['Gender', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism']

# Calculating the probablity of no show for both categories in each binary feature
df_probs = pd.DataFrame()
for cat in binary_cats:
    probs = []
    for unique in df[cat].unique():
        probs.append(df[(df[cat] == unique) & (df['Yes'] == 1)].shape[0] / df[df[cat] == unique].shape[0])
    df_probs[cat] = probs

# Altering the df to make it easy to plot. We need a row for each bar
# We need a column with the probability, a column for the feature and a 
# column for the category
df_probs = df_probs.T
df_probs2 = pd.melt(df_probs.reset_index(), id_vars='index')

g = sns.factorplot(x='index', y="value", hue="variable", data=df_probs2, size=4,
                     aspect=3, kind="bar", legend=False)
plt.legend()
plt.title('How do the categorical features affect probability of no-show')
plt.ylabel('Probability of no-show')
plt.xlabel('')
plt.show()
# Creating a Pearson correlation matrix to visualise correlations between features
corr_cats = ['Age', 'Scholarship', 'Hypertension', 'Diabetes', 'Handicap', 'SMS_received', 'Yes']
df_corr = df.loc[:, corr_cats]
corr = df_corr.corr()
fig, ax = plt.subplots(dpi=200)
sns.heatmap(corr, cmap = 'Wistia', annot= True, ax=ax, annot_kws={"size": 6})
plt.show()
# How many differnet neighbourhoods are there
neigh_size = df['Neighbourhood'].unique().size
neighbourhoods = df['Neighbourhood'].unique()

# Visualising the neighbourhood feature and each of its categories
print(df['Neighbourhood'].unique())
print(f'\nNumber of neighbourhoods - {neigh_size} \n')
print(f'The overall probability of a no show is {noshow_prob}')

# Calculating the probability of no-show for each neighbourhood
df_neigh = pd.concat([df.groupby('Neighbourhood')['No'].sum(), df.groupby('Neighbourhood')['Yes'].sum()], axis=1)
df_neigh['Total'] = df_neigh.sum(axis=1)
df_neigh['Probability of no-show'] = df_neigh['Yes'] / df_neigh['Total']
df_neigh.sort_values('Total', ascending=False, inplace=True)
print(df_neigh)
# Setting our minimum sample size for a category within neighbourhood
small_samp_size = 50

print(f'The number of districts with sample size above our threshold of {small_samp_size} is -')
print(df_neigh[df_neigh['Total'] >= small_samp_size].shape[0])
print(f'The number of districts with sample size below our threshold of {small_samp_size} is -')
print(df_neigh[df_neigh['Total'] < small_samp_size].shape[0])
print('The number of rejected samples is then')
print(df_neigh[df_neigh['Total'] < small_samp_size]['Total'].sum())

# Removing instances that come from categories with not enough samples
df_neigh = df_neigh[df_neigh['Total'] >= small_samp_size]
# Calculating the chi-squared for neighbourhood as a feature in its entirety
# This is extracting the necessary data
neigh_vals = df_neigh.loc[:, ['Yes', 'No']].values
chi2_stat, p_val, dof, ex = stats.chi2_contingency(neigh_vals)
print(f'Chi squared value is {chi2_stat} and the p-value is {p_val}')
# Calculating the expected Yes and No values by the number of instances
# within each category timesed by the overall probability of no-show
df_neigh['Exp_yes'] = df_neigh['Total'] * noshow_prob
df_neigh['Exp_no'] = df_neigh['Total'] - df_neigh['Exp_yes']
# Reordering the column titles so it goes No then Yes and also Exp_no and then Exp_yes
columns_titles = ['No', 'Yes', 'Probability of no-show', 'Exp_no','Exp_yes']
df_neigh=df_neigh.reindex(columns=columns_titles)

print(df_neigh)
def chi_squared(row):
    # Extracting the observed and expected values form each row
    observed = row[['No', 'Yes']].values
    expected = row[['Exp_no', 'Exp_yes']].values
    
    # Chi squared on this 1x2 set of values
    chi = expected - observed
    chi = chi * chi
    chi = chi / expected
    chi = np.sum(chi)
    
    # calculating the p-value with 1 degree of freedom (dof)
    pval = chi2.sf(chi,1)
    
    return pd.Series({'chi': chi, 'pvalue': pval})

# Applying the chi squared function to each row of a matrix
chi_results = df_neigh.apply(chi_squared, axis = 1)
print(chi_results)
# making a list of neighbourhoods with a chi squared value of under 0.05
neigh_keep = chi_results[chi_results['pvalue'] <= 0.05].index.tolist()
print('Number of kept neighbourhoods with p-val under 0.05 is -', len(neigh_keep))
print(neigh_keep)

# making a list of neighbourhoods with a chi squared value of under 0.01
neigh_keep2 = chi_results[chi_results['pvalue'] <= 0.01].index.tolist()
print('Number of kept neighbourhoods with p-val under 0.01 is -', len(neigh_keep2))
# Turning the datetime informaiton into the correct form
# We want the datetime the appointment was made so we can order the data like
# that at some point
# we want the Day datetime so we can calculate number of days between them easily
df['AppointMade'] = df['AppointMade'].values.astype('datetime64[s]')
df['AppointMadeD'] = df['AppointMade'].values.astype('datetime64[D]')
df['AppointFor'] = df['AppointFor'].values.astype('datetime64[D]')

# grouping the appointments into the week of the year it was made and counting
# the number of instances
df.groupby(df["AppointMade"].dt.week)["AppointMade"].count().plot(kind="bar", color='b', alpha=0.3)
plt.title('When the appointment was made')
plt.xlabel('Week of the year')
plt.ylabel('Number of appointments made')
plt.show()

# grouping the appointments into the week of the year it was made for and counting
# the number of instances
df.groupby(df["AppointFor"].dt.week)["AppointFor"].count().plot(kind="bar", color='r', alpha=0.3)
plt.title('When the appointment was made for')
plt.xlabel('Week of the year')
plt.ylabel('Number of appointments')
plt.show()

# Calculating the number of days inbetween the appointment and when it was made
# Have to convert it into an integer for later use
df['days_wait'] = (df['AppointFor'] - df['AppointMadeD'])  / np.timedelta64(1, 'D')
df['days_wait'] = df['days_wait'].astype(int)
print(df['days_wait'].describe())
print('Number of days being cut as the appointment was before it was made... -', 
      df[df['days_wait'].astype(int) < 0].shape[0])
# Removing all instances where the appointment was before the day it was made
# These are obviously typos and should be removed
df = df[df['days_wait'] >= 0]

# GRouping all appointments into how many days the patient had to wait
# and calculating a probability of showing up for each 'number of days wait'
df_days = pd.concat([df.groupby('days_wait')['No'].sum(), df.groupby('days_wait')['Yes'].sum()], axis=1)
df_days['Total'] = df_days['No'] + df_days['Yes']
df_days['Probability of no-show'] = df_days['Yes'] / df_days['Total']
df_days.reset_index(inplace=True)

# Plotting the probabilities of no-show for each days_waited value
df_days.plot(kind='scatter', x='days_wait', y='Probability of no-show')
plt.axhline(noshow_prob, c='r', label='Overall probability')
plt.legend()
plt.title('All appointments grouped into waiting time')
plt.xlabel('Days between appointment and when it was made')
plt.show()

# Showing the distribution of appointment wait times
sns.lineplot(data=df_days, x='days_wait', y='Total')
plt.title('Distribution of waiting times')
plt.xlabel('Days between appointment and when it was made')
plt.show()
# Creating a feature for if the appointment was made on the same day
df['same_day'] = np.NaN
df.loc[df['days_wait'] == 0, 'same_day'] = 1
df['same_day'].fillna(0, inplace=True)

# capping the wait time at 75. above that there aren't enough samples for
# reliable data
max_days = 75
df['days_wait'][df['days_wait'] >= max_days] = max_days

# Grouping all appointments into number of days waiting and caluclaing the 
# probability of no-show
df_days = pd.concat([df.groupby('days_wait')['No'].sum(), df.groupby('days_wait')['Yes'].sum()], axis=1)
df_days['Total'] = df_days['No'] + df_days['Yes']
df_days['Probability of no-show'] = df_days['Yes'] / df_days['Total']
df_days.reset_index(inplace=True)

# Plotting the probabilities
df_days.plot(kind='scatter', x='days_wait', y='Probability of no-show')
plt.axhline(noshow_prob, c='r', label='Overall probability')
plt.legend()
plt.title('All appointments grouped into waiting time')
plt.xlabel('Days between appointment and when it was made')
plt.show()
# Extracting the hour of the day form the appointment made time
df['hour_made'] = df['AppointMade'].dt.hour

# GRouping all appointments into the time of day it was made and 
# calculating probability of no-show for each value
df_hour = pd.concat([df.groupby('hour_made')['No'].sum(), df.groupby('hour_made')['Yes'].sum()], axis=1)
df_hour['Total'] = df_days['No'] + df_hour['Yes']
df_hour['Probability of no-show'] = df_hour['Yes'] / df_hour['Total']
df_hour.reset_index(inplace=True)

df_hour.plot(kind='scatter', x='hour_made', y='Probability of no-show')
plt.axhline(noshow_prob, c='r', label='Overall probability')
plt.legend()
plt.title('All appointments grouped into hour appointment was made')
plt.show()
# To replace 0,2,3,4,5 with the actual days of the week
# there are no appointments on or made on sunday
week_days = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat']

#  GEtting the day of he week the appointment was made form the date
df['dow_made'] = df['AppointMade'].dt.dayofweek
# Grouping all appointments into the day of the week it was made and 
# calculating probability of no-show for each value
df_dow = pd.concat((df.groupby('dow_made')['Yes'].sum(), df.groupby('dow_made')['No'].sum()), axis=1)
df_dow['Probability of no-show'] = df_dow['Yes'] / (df_dow['Yes'] + df_dow['No'])
df_dow.index = week_days

print(df_dow.head(7))
df_dow['Probability of no-show'].plot(kind='bar')
plt.axhline(noshow_prob, c='r', label='Overall probability')
plt.xlabel('Days of the week')
plt.ylabel('Probability of no-show')
plt.title('When the appointment was made')
plt.show()

# Grouping all appointments into the day of the week it was made FOR and 
# calculating probability of no-show for each value
df['dow_for'] = df['AppointFor'].dt.dayofweek
df_dowf = pd.concat((df.groupby('dow_for')['Yes'].sum(), df.groupby('dow_for')['No'].sum()), axis=1)
df_dowf['Probability of no-show'] = df_dowf['Yes'] / (df_dowf['Yes'] + df_dowf['No'])
df_dowf.index = week_days

print(df_dowf.head(7))
df_dowf['Probability of no-show'].plot(kind='bar')
plt.axhline(noshow_prob, c='r', label='Overall probability')
plt.xlabel('Days of the week')
plt.ylabel('Probability of no-show')
plt.title('The day of the appointment')
plt.show()
# This function looks at all appointments made before the current appointFor date
# and where the patient number is the same as at the current row
# and count the number of missed appointments
def count_missed_apts_before_now(row, df):
    subdf = df.query("AppointFor<@row.AppointMade and\
                     `No-show`=='Yes' and PatientId==@row.PatientId")
    return len(subdf)

# Sorting by appointment made date so we can get rolling counts
df.sort_values(by='AppointMade', inplace=True)
# Calculating how many time the patient has made an appointment before
df['book_count'] = df.groupby('PatientId').cumcount()
# Calculating the number of times a patient has missed an appointment before making the current one
t3 = time.time()
df['miss_count'] = df.apply(count_missed_apts_before_now, axis=1, args = (df,))
t4 = time.time()
miss_count_t = t4-t3
print(f'miss count column calculated in {miss_count_t}')
# These additional date categories are of no use after processing the info we need
df.drop(['dow_made', 'dow_for', 'AppointFor', 'AppointMade', 'AppointMadeD'], 
        axis=1, inplace=True)
# These are the results categories but in the wrong form
df.drop(['No', 'No-show'], axis=1, inplace=True)
# These two categories showed no correlation with showing up for an appointment
df.drop(['Gender', 'Alcoholism'], inplace=True, axis=1)
# handicap may have some vlaue and would need to be treated as dummy cols
# but there may be some cross-correlation with age
df.drop('Handicap', axis=1, inplace=True)
# These categories showed a weak correlation so could tell us something, but there
# was considerable cross correlation with Age
df.drop(['Scholarship', 'Hypertension', 'Diabetes'], inplace=True, axis = 1)
# No longer need Patient ID as we have all the information on repeat patients
df.drop('PatientId', inplace=True, axis=1)

df['Age^2'] = df['Age'] ** 2
df['Age^3'] = df['Age'] ** 3
df['hour_made^2'] = df['hour_made'] ** 2
df['hour_made^3'] = df['hour_made'] ** 3
# Only going up to squared for days_wait as the relationship seemed to be more simple
df['days_wait^2'] = df['days_wait'] ** 2

print(df.head())
# These will be added to the dataframe during feature engineering
dum_neigh = pd.get_dummies(df['Neighbourhood'])
# neigh_keep is my list of statistically significant neighbourhoods
dum_neigh = dum_neigh[neigh_keep]
dum_neigh.head()

df = pd.concat((df, dum_neigh), axis = 1)
df.drop('Neighbourhood', axis=1, inplace=True)
print(df.columns.tolist())
# Splitting the dataframe into features and the target
X = df.loc[:, df.columns != 'Yes']
y = df.loc[:, df.columns == 'Yes']

# Splitting the dataframe into a train dataset and a test dataset
# random state used so its the same split each time. this aids with comparisons
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
features = X_train.columns
# substantiating an isntance of the SMOTE object
os = SMOTE(random_state=0)

# Adding a lot of new dummy data for no-show = Yes to even out the Yes and No
os_data_X, os_data_y = os.fit_sample(X_train, y_train)
# Giving the new dataframes the old names
os_data_X = pd.DataFrame(data=os_data_X, columns=features )
os_data_y = pd.DataFrame(data=os_data_y, columns=['Yes'])

print('new number of instances is ', X_train.shape[0])
print('new number of no shows is ', y_train['Yes'].sum())
print('proportions of no shows is ', y_train['Yes'].sum() / X_train.shape[0], '\n\n')

print('new number of instances is ', os_data_X.shape[0])
print('new number of no shows is ', os_data_y['Yes'].sum())
print('proportions of no shows is ', os_data_y['Yes'].sum() / os_data_X.shape[0])
# These are all fetures which go way above 1. They need to be brough back to a 
# similar scale as our binary features
to_normalize = ['Age', 'Age^2', 'Age^3', 'hour_made', 'hour_made^2', 'hour_made^3', 
                'days_wait', 'days_wait^2', 'book_count', 'miss_count']

# Normalizing very simply by just dividing be the range
for cat in to_normalize:
    os_data_X[cat] = os_data_X[cat] / (os_data_X[cat].max() - os_data_X[cat].min())
    # Done separately so there is no cross-contamination between test and train
    X_test[cat] = X_test[cat] / (X_test[cat].max() - X_test[cat].min())

print(os_data_X.describe())
# A logisitc regression object
logreg = LogisticRegression()
# The recursive feature elimination object
rfe = RFE(logreg, 20)

X = os_data_X.values
y = os_data_y.values

# This is our list of features before feature filtering
start_feats = X_train.columns.tolist()

# Using the log regression object to test the importance of each feature in turn
rfe = rfe.fit(X, y)
for ii, cat in enumerate(start_feats):
    print(cat, ' - ', rfe.support_[ii])
# Tried to remove all features which do not have such a strong influence but ended up
# making the model worse. We will check the statistical signigicance of
# each feature now
#feats2 = [x for x, y in zip(start_feats, rfe.support_) if y == True]
feats2 = start_feats
print(feats2)

X = os_data_X.loc[:, os_data_X.columns.isin(feats2)].values
y = os_data_y.values

# Looking at the statistical significance of each of our features in a log model
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())
# Removing the 'hour_made^2' feature
feats2 = [x for x in feats2 if x != 'hour_made^2']

X = os_data_X.loc[:, os_data_X.columns.isin(feats2)].values
y = os_data_y.values

# The object we will use for our final model
logreg = LogisticRegression()
# Making the model
logreg.fit(X, y)

# Extracting only our new filtered feature dataset
X_test = X_test[feats2]
# Making our predictions on the test dataset
y_pred = logreg.predict(X_test)
# This is how well the model predicts the 'Yes' column whether there was a no-show
score = logreg.score(X_test, y_test)
print('Accuracy when tested on the test set - ', score)

# This is to show how often we got true/negative, true/positive, false/negative and false/positive
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print('ratio of predicted noshows to show-ups -',y_pred.sum() / len(y_pred))
print(classification_report(y_test, y_pred))
# Calculating the false positive and true positive rates
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])

plt.figure()
plt.plot(fpr, tpr, label=f'Logistic Regression (area = {logit_roc_auc:.2f})')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()