import numpy as np

import pandas as pd



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
e00 = check_output(["ls", "../input/ensemble"]).decode("utf8")

e01 = check_output(["ls", "../input/ensemble-grocery-01"]).decode("utf8")

print(e00)

print(e01)
mods = e00.split('\n')

mods.pop()  # Get rid of empty string at end

preds = [ pd.read_csv('../input/ensemble/' + m, index_col=0) for m in mods ]

moremods = e01.split('\n')

moremods.pop()

preds += [ pd.read_csv('../input/ensemble-grocery-01/' + m, index_col=0) for m in moremods ]

mods += moremods
df = pd.DataFrame(index=preds[0].index)

modnames = []

for m,p in zip(mods,preds):

    name = m[:-4]

    if len(name)>20:

        name = name[:13] # Too damn long

    modnames += [name]

    p.columns = [name]

    df = df.join( p )

print( modnames )

df.head()
test = pd.read_csv('../input/favorita-grocery-sales-forecasting/test.csv', 

                   parse_dates=['date'], index_col=0)

test.head()
df = test[['date']].join(df)
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
for m in modnames:

    print( '\n\n\nModel: ' + m )

    results = df[['date', m]]

    print( '\n\nComparing results for public test data with subsequent week...')

    compare_results(results, public_dates, private_public_dows)

    print( '\n\nComparing results for public test data with non-holidays in subsequent week...')

    compare_results(results, public_dates_nohol, private_public_dows_nohol)

    print( '\n\nComparing results for early private test data with a week later...')

    compare_results(results, early_private_dates, private_private_dows)

    print( '\n\nComparing results for end of private data with corresponding days in public data...')

    compare_results(results, public_last_dows, last_private_dates)