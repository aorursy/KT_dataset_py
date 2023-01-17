import pandas as pd
## Explore boolean operations on a pandas Series

test = pd.Series([1,2,3,4,5])

print(test > 3)

print(test != 3)

print(test == ~test)

print(test == test)

print(test <= 4)  



## Use proper boolean comparator operators

boolean_s = pd.Series([True, False, True, False])

boolean_s2 = pd.Series([True, True, False, False])



print(boolean_s & boolean_s2)

print(boolean_s | boolean_s2)

print(boolean_s & (~boolean_s2))

print(boolean_s | ~boolean_s)



## Test 'is' and 'is not'

print(boolean_s is boolean_s2)

print(boolean_s is not boolean_s2)



# but you can't just use and, and you can't use 'not' in place of the '~' operator

#print(boolean_array and boolean_array2)

#print(boolean_array and (not boolean_array2))



## Pass collection of booleans into an array to filter out values

print(test[test > 3])

print(test[test != 3])
## Import data



import pandas as pd

life_expectancy_df = pd.read_csv("../input/life_expectancy.csv", index_col=0)



## Explore data



print("First 3 rows and last 3 columns: \n", life_expectancy_df.iloc[:3,-3:])

print("\nAll columns, first 5 rows: \n", life_expectancy_df.head(n=5)) 
## Use index selection to filter data



print("\nLast five years of United States data:\n", life_expectancy_df.loc['United States'][-5:])

print("\nFirst five rows of data for 2013:\n", life_expectancy_df['2013'][:5]) # or life_expectancy_df['2013'].head()
## Compare the iat function vs the iloc functions in terms of speed



%timeit life_expectancy_df.iloc[0,1]
%timeit life_expectancy_df.iat[0,1]
## Subset the DataFrame using boolean filtering



## Create boolean array

filter80 = life_expectancy_df['2013'] >= 80



## Verify filter80 is a Series that contains boolean values

print('Data structure: {}, containing {} dtype'.format(type(filter80),filter80.dtype)),



## Subset using loc method with an array of bools

eighty_plus_in_2013_df = life_expectancy_df.loc[filter80]

print("\nLast three years L.E. of countries with L.E. of 80+ in 2013:\n", eighty_plus_in_2013_df.iloc[:,-3:])
## Find countries with life expectancy between 80 and 85 in the year 2013 and between 70 and 80 for year 2012



newFilter = (life_expectancy_df['2013'] > 80) & (life_expectancy_df['2013'] < 85) & (life_expectancy_df['2012'] > 70) & (life_expectancy_df['2012'] < 80)

print(life_expectancy_df.loc[newFilter])

print(life_expectancy_df[newFilter].index)
## Convert column names to better names to enable query method to work



life_expectancy_df.columns = ["y" + col for col in life_expectancy_df.columns]
## Use query method to find countries with life expectancy between 70 and 80 in the year 2013 and year 2012

life_expectancy_df.query('80 < y2013 < 85 and 70 < y2012 < 80').index
## Find countries with missing data for the latest 100 years of data

life_expectancy_100_df = life_expectancy_df.iloc[:,-100:].copy()



mask = life_expectancy_100_df.isna().any(axis=1).tolist() # any() function returns whether any element is True over requested axis.



countriesMissing = life_expectancy_100_df[mask].index.tolist()

print(countriesMissing)



## For each country with missing data, calculate how many values are missing for the past 100 years

dfMissing = life_expectancy_100_df.loc[countriesMissing]

print(dfMissing.isna().sum(axis=1))
## Remove countries that have 100 missing values for the past 100 years



mask100 = (dfMissing.isna().sum(axis=1) == 100).tolist()

countriesToDrop = dfMissing.loc[mask100].index



life_expectancy_100_clean_df = life_expectancy_100_df.drop(countriesToDrop)
## For the countries with less than 100 missing values, replace those missing values with the average for all countries for any particular past 100 years

life_expectancy_100_clean_df.fillna(life_expectancy_100_clean_df.mean(),inplace=True)



## Confirm all NaN's have been replaced

life_expectancy_100_clean_df.isna().any(axis=1).sum()



print(life_expectancy_100_clean_df)