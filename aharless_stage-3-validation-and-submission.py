import numpy as np

import pandas as pd

from sklearn.metrics import mean_squared_error



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
dir_raw = "../input/favorita-grocery-sales-forecasting/"

dir_prep = "../input/preparing-data-for-lgbm-or-something-else/"

dir_1st_result = "../input/stage-2a-the-first-half/"

dir_2nd_result = "../input/stage-2b-the-second-half/"

dirs = [dir_raw, dir_prep, dir_1st_result, dir_2nd_result]
for d in dirs:

    print(d, '\n', check_output(["ls", d]).decode("utf8"), '\n')
val1 = pd.read_csv(dir_1st_result+'nn_val_first_half.csv')

print( val1.shape )

val1.head()
val2 = pd.read_csv(dir_2nd_result+'nn_val_second_half.csv')

print( val2.shape )

val2.head()
val2.columns = pd.Index([str(i) for i in range(8,16)])

val2.head()
val_pred = pd.concat([val1,val2],axis=1)

print(val_pred.shape)

val_pred.head()
y_val = pd.read_csv(dir_prep + 'y_val.csv')

print( y_val.shape )

y_val.head()
items = pd.read_csv( dir_raw + 'items.csv' ).set_index("item_nbr")

stores_items = pd.read_csv(dir_prep + 'stores_items.csv', index_col=['store_nbr','item_nbr'])

items = items.reindex( stores_items.index.get_level_values(1) )

print( items.shape )

items.head()
n_public = 5 # Number of days in public test set

weights=pd.concat([items["perishable"]]) * 0.25 + 1

print("Unweighted validation mse: ", mean_squared_error(

    np.array(y_val), np.array(val_pred)) )

print("Full validation mse:       ", mean_squared_error(

    np.array(y_val), np.array(val_pred), sample_weight=weights) )

print("'Public' validation mse:   ", mean_squared_error(

    np.array(y_val)[:,:n_public], np.array(val_pred)[:,:n_public], sample_weight=weights) )

print("'Private' validation mse:  ", mean_squared_error(

    np.array(y_val)[:,n_public:], np.array(val_pred)[:,n_public:], sample_weight=weights) )
# This is all zeros for the second half

sub1 = pd.read_csv(dir_1st_result+'nn_sub_first_half.csv', index_col='id')

print( sub1.shape )

sub1.head()
# This is all zeros for the first half

sub2 = pd.read_csv(dir_2nd_result+'nn_sub_second_half.csv', index_col='id')

print( sub2.shape )

sub2.head()
submax = sub1.join(sub2, rsuffix='_2').max(axis=1)

submax.head()
submax.name = 'unit_sales'

sub = pd.DataFrame(submax).reset_index()

print( sub.shape )

sub.head()
test = pd.read_csv('../input/favorita-grocery-sales-forecasting/test.csv', 

                   parse_dates=['date'], index_col=0)

test.head()
check_df = test[['date']].join(sub.set_index("id"))

check_df.head()
# Dates of public test set

public_dates = ['2017-08-16', '2017-08-17', '2017-08-18', '2017-08-19', '2017-08-20']

# Dates of private test set exactly one week after those of public test set

private_public_dows = ['2017-08-23', '2017-08-24', '2017-08-25', '2017-08-26', '2017-08-27']



# Dates of public test set that don't correspond to holidays in the subsequent week

public_dates_nohol = ['2017-08-16', '2017-08-18', '2017-08-19', '2017-08-20']

# Non-holdiay dates of private test set exactly one week after those of public test set

private_public_dows_nohol = ['2017-08-23', '2017-08-25', '2017-08-26', '2017-08-27']



# Dates of private test set that occur before end of first 7 days of full test set

early_private_dates = ['2017-08-21', '2017-08-22']

# Dates of private test set exactly one week after those

private_private_dows = ['2017-08-28', '2017-08-29']



# Last days of private test set

last_private_dates = ['2017-08-30', '2017-08-31']

# Days exactly 2 weeks before the last days

public_last_dows = ['2017-08-16', '2017-08-17']
def compare_results(results, earlier_dates, later_dates):

    early = results[results.date.isin(earlier_dates)].describe()

    late = results[results.date.isin(later_dates)].describe()

    print( '\nIf these are dramatically different with no good reason why,')

    print(  '   there might be a problem:\n')

    print( early.join(late, lsuffix='_early', rsuffix='_late') )
def corr_results(results, earlier_dates, later_dates):

    d1 = results[results.date.isin(earlier_dates)].reset_index()[['unit_sales']]

    d2 = results[results.date.isin(later_dates)].reset_index()[['unit_sales']]

    d = pd.concat([d1, d2], axis=1)

    print( d.corr() )
print( '\n\nComparing results for public test data with subsequent week...')

compare_results(check_df, public_dates, private_public_dows)

print( '\n\nComparing results for public test data with non-holidays in subsequent week...')

compare_results(check_df, public_dates_nohol, private_public_dows_nohol)

print( '\n\nComparing results for early private test data with a week later...')

compare_results(check_df, early_private_dates, private_private_dows)

print( '\n\nComparing results for end of private data with corresponding days in public data...')

compare_results(check_df, public_last_dows, last_private_dates)
print( '\n\nCorrelating results for public test data with subsequent week...')

corr_results(check_df, public_dates, private_public_dows)

print( '\n\nCorrelating results for public test data with non-holidays in subsequent week...')

corr_results(check_df, public_dates_nohol, private_public_dows_nohol)

print( '\n\nCorrelating results for early private test data with a week later...')

corr_results(check_df, early_private_dates, private_private_dows)

print( '\n\nCorrelating results for end of private data with corresponding days in public data...')

corr_results(check_df, public_last_dows, last_private_dates)
sub.to_csv('nn_sub_combined.csv', float_format='%.5f', index=None)