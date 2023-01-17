import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('../input/Indicators.csv')
data.shape
health_gdp_indicator_name = 'Health expenditure, total \(% of GDP\)'
health_gdp_indicator_mask = data['IndicatorName'].str.contains(health_gdp_indicator_name)
health_gdp_indicator_df = data[health_gdp_indicator_mask]
health_gdp_indicator_df.head()
print(health_gdp_indicator_df.Year.min(),' to ',health_gdp_indicator_df.Year.max())
death_indicator_name = 'Death rate, crude \(per 1,000 people\)'
death_indicator_mask = data['IndicatorName'].str.contains(death_indicator_name)
death_indicator_df = data[death_indicator_mask]
death_indicator_df.head()
print(death_indicator_df.Year.min(),' to ',death_indicator_df.Year.max())
death_indicator_df_reduced = death_indicator_df[death_indicator_df['Year'] > 1994]
death_indicator_df_reduced.shape
health_gdp_indicator_df.shape
len(death_indicator_df_reduced['CountryName'].unique().tolist())
len(health_gdp_indicator_df['CountryName'].unique().tolist())
countries1 = np.array(death_indicator_df_reduced['CountryName'].unique().tolist())
countries2 = np.array(health_gdp_indicator_df['CountryName'].unique().tolist())
both_countries = np.intersect1d(countries1, countries2)

dt_country_mask = death_indicator_df_reduced['CountryName'].isin(both_countries)
hg_country_mask = health_gdp_indicator_df['CountryName'].isin(both_countries)

death_indicator_df_cropped = death_indicator_df_reduced[dt_country_mask]
health_gdp_indicator_df_cropped = health_gdp_indicator_df[hg_country_mask]
len(death_indicator_df_cropped['CountryName'].unique().tolist())
len(health_gdp_indicator_df_cropped['CountryName'].unique().tolist())
death_indicator_df_cropped.shape
health_gdp_indicator_df_cropped.shape
dt_year_count = death_indicator_df_cropped.groupby('Year').count()
dt_year_count[['CountryName','CountryCode']]
hg_year_count = health_gdp_indicator_df_cropped.groupby('Year').count()
hg_year_count[['CountryName','CountryCode']]
def does_it_has_complete_history(country_name,df):
    mask1 = df['CountryName'].str.contains(country_name) 
    mask2 = df['Year'].isin(df['Year'].unique().tolist())
    # apply our mask
    full = df[mask1 & mask2]
    if len(full['Year']) == len(df['Year'].unique().tolist()):
        return True
    else:
        return False

def does_the_country_increasing(country_name,df):
    values = df[df.CountryName == country_name]['Value']
    values = values[0::5] # only check the difference in 5 years span
    if strictly_increasing(values):
        return True
    else:
        return False

def does_the_country_decreasing(country_name,df):
    values = df[df.CountryName == country_name]['Value']
    values = values[0::5] # only check the difference in 5 years span
    if strictly_decreasing(values):
        return True
    else:
        return False

#referenced from https://stackoverflow.com/questions/4983258/python-how-to-check-list-monotonicity
def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def strictly_decreasing(L):
    return all(x>y for x, y in zip(L, L[1:]))
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups') # need to silence this warning :)

country_that_decreasing = []
country_that_increasing = []
country_that_has_history = []
for country in both_countries:
    dt_has_history = does_it_has_complete_history(country,death_indicator_df_cropped)
    hg_has_history = does_it_has_complete_history(country,health_gdp_indicator_df_cropped)
    if dt_has_history & hg_has_history: # only country that exists on both dataframe
        country_that_has_history.append(country)
        hg_increasing = does_the_country_increasing(country, health_gdp_indicator_df_cropped)
        if hg_increasing:
            country_that_increasing.append(country)
            hg_decreasing = does_the_country_decreasing(country, health_gdp_indicator_df_cropped)
            if hg_decreasing:
                country_that_decreasing.append(country)
        
    


len(country_that_increasing)
len(country_that_decreasing)
len(country_that_has_history)
mask = health_gdp_indicator_df_cropped['CountryName'].isin(country_that_increasing)
health_gdp_indicator_df_cropped[mask].head(60)
values = health_gdp_indicator_df_cropped[health_gdp_indicator_df_cropped.CountryName == 'Indonesia'][['Year','Value']]
values
values = death_indicator_df_cropped[death_indicator_df_cropped.CountryName == 'Indonesia'][['Year','Value']]
values
# those that only increasing their health expenditure
dt_country_mask_2 = death_indicator_df_cropped['CountryName'].isin(country_that_increasing)
hg_country_mask_2 = health_gdp_indicator_df_cropped['CountryName'].isin(country_that_increasing)

death_indicator_clean = death_indicator_df_cropped[dt_country_mask_2]
health_gdp_indicator_clean = health_gdp_indicator_df_cropped[hg_country_mask_2]

# all countries that has complete death rate and health expenditure history from 1995 to 2013
dt_country_mask_3 = death_indicator_df_cropped['CountryName'].isin(country_that_has_history)
hg_country_mask_3 = health_gdp_indicator_df_cropped['CountryName'].isin(country_that_has_history)

death_indicator_all = death_indicator_df_cropped[dt_country_mask_3]
health_gdp_indicator_all = health_gdp_indicator_df_cropped[hg_country_mask_3]
death_indicator_clean.shape
health_gdp_indicator_clean.shape
death_indicator_all.shape
health_gdp_indicator_all.shape
death_indicator_clean_years = death_indicator_clean.groupby('Year').count()
death_indicator_clean_years[['CountryName','CountryCode']]
health_gdp_indicator_clean_years2 = health_gdp_indicator_clean.groupby('Year').count()
health_gdp_indicator_clean_years2[['CountryName','CountryCode']]
death_indicator_all_years = death_indicator_all.groupby('Year').count()
death_indicator_all_years[['CountryName','CountryCode']]
health_gdp_indicator_all_years2 = health_gdp_indicator_all.groupby('Year').count()
health_gdp_indicator_all_years2[['CountryName','CountryCode']]
health_gdp_indicator_all_mean = health_gdp_indicator_all.groupby('Year' , as_index=False).mean()
plt.figure(1, figsize=(12, 5))
# switch to a line plot
plt.plot(health_gdp_indicator_all_mean['Year'].values, health_gdp_indicator_all_mean['Value'].values)
# Label the axes
plt.xlabel('Year')
plt.ylabel(health_gdp_indicator_all.iloc[0].IndicatorName)
plt.grid(color='gray', linewidth=0.2, axis='y', linestyle='solid')

#label the figure
plt.title('World\'s yearly average of ' + health_gdp_indicator_all.iloc[0].IndicatorName )

# to make more honest, start they y axis at 0
plt.axis([1995, 2013,0,9])
plt.show()
death_indicator_all_mean = death_indicator_all.groupby('Year' , as_index=False).mean()
plt.figure(1, figsize=(12, 5))
# switch to a line plot
plt.plot(death_indicator_all_mean['Year'].values, death_indicator_all_mean['Value'].values)
# Label the axes
plt.xlabel('Year')
plt.ylabel(death_indicator_all.iloc[0].IndicatorName)
plt.grid(color='gray', linewidth=0.2, axis='y', linestyle='solid')

#label the figure
plt.title('World\'s yearly average of  ' + death_indicator_all.iloc[0].IndicatorName)

# to make more honest, start they y axis at 0
plt.axis([1995, 2013,0,11])
plt.show()
fig, axis = plt.subplots()
# Grid lines, Xticks, Xlabel, Ylabel

axis.yaxis.grid(True)
axis.xaxis.grid(True)
axis.set_title('Death rate vs. health expenditure (world yearly average)',fontsize=10)
axis.set_xlabel('Yearly average of ' + health_gdp_indicator_all.iloc[0].IndicatorName,fontsize=10)
axis.set_ylabel('Yearly average of ' + death_indicator_all.iloc[0].IndicatorName,fontsize=9)

X = health_gdp_indicator_all_mean['Value']
Y = death_indicator_all_mean['Value']

fig.dpi = 110
fig.figsize = (10,5)
axis.scatter(X, Y)
plt.show()
np.corrcoef(health_gdp_indicator_all_mean['Value'],death_indicator_all_mean['Value'])
death_indicator_clean_mean = death_indicator_clean.groupby('Year' , as_index=False).mean()
plt.figure(1, figsize=(12, 5))
# switch to a line plot
plt.plot(death_indicator_clean_mean['Year'].values, death_indicator_clean_mean['Value'].values)
# Label the axes
plt.xlabel('Year')
plt.ylabel(death_indicator_clean.iloc[0].IndicatorName)
plt.grid(color='gray', linewidth=0.2, axis='y', linestyle='solid')

#label the figure
plt.title('Yearly average of ' + death_indicator_clean.iloc[0].IndicatorName + ' on country that increases its health expenditure')

# to make more honest, start they y axis at 0
plt.axis([1995, 2013,0,11])
plt.show()
health_gdp_indicator_clean_mean = health_gdp_indicator_clean.groupby('Year' , as_index=False).mean()
plt.figure(1, figsize=(12, 5))
# switch to a line plot
plt.plot(health_gdp_indicator_clean_mean['Year'].values, health_gdp_indicator_clean_mean['Value'].values)
# Label the axes
plt.xlabel('Year')
plt.ylabel(health_gdp_indicator_clean.iloc[0].IndicatorName)
plt.grid(color='gray', linewidth=0.2, axis='y', linestyle='solid')

#label the figure
plt.title('Yearly average of ' + health_gdp_indicator_clean.iloc[0].IndicatorName + ' on country that increases its health expenditure')

# to make more honest, start they y axis at 0
plt.axis([1995, 2013,0,9])
plt.show()
fig, axis = plt.subplots()
# Grid lines, Xticks, Xlabel, Ylabel

axis.yaxis.grid(True)
axis.xaxis.grid(True)
axis.set_title('Death rate vs. health expenditure yearly average on 57 countries',fontsize=10)
axis.set_xlabel('Yearly average of ' + health_gdp_indicator_clean.iloc[0].IndicatorName,fontsize=10)
axis.set_ylabel('Yearly average of ' + death_indicator_clean.iloc[0].IndicatorName,fontsize=9)

X = health_gdp_indicator_clean_mean['Value']
Y = death_indicator_clean_mean['Value']

fig.dpi = 110
fig.figsize = (10,5)
axis.scatter(X, Y)
plt.show()
np.corrcoef(health_gdp_indicator_clean_mean['Value'],death_indicator_clean_mean['Value'])