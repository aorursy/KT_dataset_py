from IPython.display import display

import numpy as np

import scipy.stats as st

import pandas as pd

import matplotlib.pyplot as plt
# Air data

air_store = pd.read_csv('../input/air_store_info.csv')

air_reserve = pd.read_csv('../input/air_reserve.csv')

air_visit = pd.read_csv('../input/air_visit_data.csv')

# HPG data

hpg_store = pd.read_csv('../input/hpg_store_info.csv')

hpg_reserve = pd.read_csv('../input/hpg_reserve.csv')

# Diverse data

date_info = pd.read_csv('../input/date_info.csv')

store_ids = pd.read_csv('../input/store_id_relation.csv')

# Submission file

submission = pd.read_csv('../input/sample_submission.csv')
# Check Air dataframes

display(air_store.head(3))

display(air_reserve.head(3))

display(air_visit.head(3))
# Check HPG dataframes

display(hpg_store.head(3))

display(hpg_reserve.head(3))
# Check diverse dataframes

display(date_info.head(3))

display(store_ids.head(3)) 
# Check submission dataframes

submission.head(3)
# Check for NaN

print(air_store.isnull().sum(),'\n',air_reserve.isnull().sum(),'\n',air_visit.isnull().sum(),

      '\n',hpg_store.isnull().sum(),'\n',hpg_reserve.isnull().sum(),'\n',date_info.isnull().sum(),

      '\n',store_ids.isnull().sum(),'\n',submission.isnull().sum())
# Check 'visitors' data distribution: all positive, skewed...

air_visit.visitors.hist(bins=100)

plt.xlim(0,100)

plt.show()
# Format all dates using datetime

air_visit.visit_date = pd.to_datetime(air_visit.visit_date)

air_reserve.visit_datetime = pd.to_datetime(air_reserve.visit_datetime)

air_reserve.reserve_datetime = pd.to_datetime(air_reserve.reserve_datetime)

hpg_reserve.visit_datetime = pd.to_datetime(hpg_reserve.visit_datetime)

hpg_reserve.reserve_datetime = pd.to_datetime(hpg_reserve.reserve_datetime)

date_info.calendar_date = pd.to_datetime(date_info.calendar_date)
# Apply log transform to 'visitors' so we have a more normal distribution to work with (calculation of means, 

# confidence intervals, etc). 

# Create copy of visit data.

visit_data = air_visit.copy()

visit_data['visitors'] = visit_data.visitors.map(pd.np.log1p)

visit_data.visitors.hist(bins=25)

plt.show()
# Remove holiday_flg from weekend days

wkend_holidays = date_info.apply((lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday') and x.holiday_flg==1), axis=1)

date_info.loc[wkend_holidays, 'holiday_flg'] = 0
# Add weights to date_info before merging

# date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 6  # LB 0.497

date_info['weight'] = (((date_info.index + 1) / len(date_info)) ** 6.5)  # LB 0.496 (but better that 6)

plt.plot(len(date_info)-date_info.index, date_info.weight)

# date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 7  # LB 0.496 (not better than 6)



# Check weights indicative result

plt.plot(len(date_info)-date_info.index, date_info.weight)

plt.xlabel('Lookback days'); plt.ylabel('Weight factor'); plt.ylim(0,); plt.xlim(0,)

plt.show()

# Define the date weighted mean function

wmean = lambda x:( (x.weight * x.visitors).sum() / x.weight.sum() )
# Add day of the week and holiday flag to air_visit dataframe

visit_data = visit_data.merge(date_info,left_on='visit_date',right_on='calendar_date',how='left')

visit_data.drop('calendar_date', axis=1, inplace=True)

visit_data.head(3)
# Add restaurant genre to air_visit dataframe

visit_data = visit_data.merge(air_store[['air_store_id','air_genre_name']],on='air_store_id',how='left')

display(visit_data.head(3))
# Combine both reservation systems. Data structure is the same, only need to match store_ids

hpg_reserve = hpg_reserve.merge(store_ids, on='hpg_store_id')

hpg_reserve.drop('hpg_store_id', axis=1, inplace=True)

all_reserve = pd.concat([air_reserve, hpg_reserve])
# Extract store_id and date from submission. Add day of the week

submission['visit_date'] = pd.to_datetime(submission.id.str[-10:])

submission = submission.merge(date_info,left_on='visit_date',right_on='calendar_date')

submission['air_store_id'] = submission.id.str[:-11]

submission.drop('calendar_date', axis=1, inplace=True)

submission.head(3)
# Check how many restaurants are included in each system (reservations and actual visits)

print ("There are %s different stores in the visits data" %len(visit_data.air_store_id.unique()))

print ("There are %s different stores in the reservations data" %len(all_reserve.air_store_id.unique()))

print ("There are %s different stores in the submission data" %len(submission.air_store_id.unique()))
# Some plots of reservations vs. actual visitors

plt.figure(figsize=(15,4))

plt.plot(all_reserve.groupby('visit_datetime').visit_datetime.first(),

         all_reserve.groupby('visit_datetime').reserve_visitors.sum(), 'o', label='Reservation visitors')

plt.plot(air_visit.groupby('visit_date').visit_date.first(),

         air_visit.groupby('visit_date').visitors.sum(), 'o', label='Actual visitors')

plt.plot(visit_data.visit_date, visit_data.holiday_flg*10000, '*')

plt.ylim(0,)

plt.legend()

plt.show()



# Check data timeline. It's tricky to draw conclusions from data not aligned time-wise...

# Number of stores added to the data files

plt.figure(figsize=(15,4))

visit_data.groupby('air_store_id').visit_date.first().hist(bins=50, grid=False, 

                                                          label='Rest. added to visits system')

air_reserve.groupby('air_store_id').reserve_datetime.first().hist(bins=50, grid=False, 

                                                                  label='Rest. added to AirRes system')

hpg_reserve.groupby('air_store_id').reserve_datetime.first().hist(bins=50, grid=False, 

                                                                  label='Rest. added to HPGRes system')

plt.legend()

plt.show()





# Number of stores in the the data files

plt.figure(figsize=(15,4))

plt.plot(visit_data.groupby('visit_date').visit_date.first(),

         visit_data.groupby('visit_date').air_store_id.count(), 'o', label='Rest. in visits system')

plt.plot(air_reserve.groupby('visit_datetime').visit_datetime.first(),

         air_reserve.groupby('visit_datetime').air_store_id.count(), 'o', label='Rest. in AirRes system')

plt.plot(hpg_reserve.groupby('visit_datetime').visit_datetime.first(),

         hpg_reserve.groupby('visit_datetime').air_store_id.count(), 'o', label='Rest. in HPGRes system')

plt.legend()

plt.show()
# Populate each store with the WEIGHTED average for the same day for the same store

g1 = visit_data.groupby(['air_store_id', 'day_of_week', 'holiday_flg']).apply(wmean)



# g1 = air_visit.groupby(['air_store_id', 'day_of_week', 'holiday_flg']).mean()

display(pd.DataFrame(g1).head())

submission.groupby(['air_store_id', 'day_of_week', 'holiday_flg']).mean().head()
# Combine groupby results with the submission file. Use 'left' to keep the orinal number of rows

submission = submission.merge(g1.reset_index(), on=['air_store_id', 'day_of_week', 'holiday_flg'], how='left')

# Tidy up the resulting submission file

submission.drop('visitors', axis=1, inplace=True)

submission.rename(columns = {0:'visitors'}, inplace=True) # New 'visitors' column comes with a weird naming...

# Check

submission.head(3)
# Check we still have the required 32019 rows for submission

print (len(submission))

# Check if final results have any NaN

submission.isnull().sum()
# Fill NaN using store_id and day_of_week

g2 = visit_data.groupby(['air_store_id','day_of_week']).mean()

# g2 = air_visit.groupby(['day_of_week', 'holiday_flg']).mean()

display(g2.head())

# Merge as new visitors column ('vistors_y'...previous one is renamed as 'visitors_x')

submission = submission.merge(g2.visitors.reset_index(), on=['air_store_id','day_of_week'], how='left')

# Tidy up the resulting submission file

submission.visitors_x.fillna(submission.visitors_y, inplace=True)

submission.head(3)
# Check we still have the required 32019 rows for submission

print (len(submission))

# Check if final results have any NaN

submission.isnull().sum()
# Fill NaN using store_id

g3 = visit_data.groupby('air_store_id').mean()

# g2 = air_visit.groupby(['day_of_week', 'holiday_flg']).mean()

display(g3.head())

# Merge as new visitors column (names as 'visitors')

submission = submission.merge(g3.visitors.reset_index(), on=['air_store_id'], how='left')



# # Fill NaN in original visitors column with values in the recently merged on

submission['visitors_x'].fillna(submission['visitors'], inplace=True)

# # Tidy up

submission.drop(['visitors_y', 'visitors'], axis=1, inplace=True) # Drop the added ones

submission.rename(columns = {'visitors_x':'visitors'}, inplace=True) # Rename the good one

submission.head(3)
# Check we still have the required 32019 rows for submission

print (len(submission))

# Check if final results have any NaN

submission.isnull().sum()
# Tweaks, only for testing

# submission.visitors = submission.visitors *1.025 # From previous run
# Check distribution of results (log-transformed)

visit_data.visitors.hist(bins=25, grid=False, edgecolor='black', normed=True, label='Actual')

submission.visitors.hist(bins=25, grid=False, edgecolor='black', normed=True, alpha=0.5, label='Prediction')

plt.xlim(0,6)

plt.ylim(0, 0.6)

plt.legend()

plt.show()
# Undo log transformatoin to 'visitors'

submission['visitors'] = submission.visitors.map(pd.np.expm1)

visit_data.visitors = visit_data.visitors.map(pd.np.expm1)
# Check distribution of results

visit_data.visitors.hist(bins=105, grid=False, edgecolor='black', normed=True, label='Actual')

submission.visitors.hist(bins=25, grid=False, edgecolor='black', normed=True, alpha=0.5, label='Prediction')

plt.xlim(0,200)

plt.ylim(0, 0.04)

plt.legend()

plt.show()
# Save results file

results = submission[['id', 'visitors']]

results.set_index('id', inplace=True)

results.to_csv('results.csv')
submission.sort_values(by='id').head()