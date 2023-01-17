# importing data libraries
import numpy as np
import pandas as pd

# statistics libraries
from scipy import stats

# importing visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = pd.read_csv('../input/hotel-booking-demand/hotel_bookings.csv')
data.shape
data.head()
data.describe()
data.info()
# nan values

total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)

nan_cols = pd.concat([total, percent], axis=1, keys=['Total', 'nan'])

nan_cols = nan_cols[nan_cols['nan'] > 0]

nan_cols
plt.figure(figsize=(10, 10))

plt.pie(nan_cols.reset_index()['Total'])

plt.title("NaN Values")
plt.legend(labels=nan_cols.reset_index()['index'])
plt.show()
# We will drop columns 'company' and 'agent' because, there are too many null values for that columns

data.drop(['company', 'agent'], axis=1, inplace=True)
# Reviewing rows with null country values 
data[data['country'].isnull()]
# Reviewing rows with null children values 
data[data['children'].isnull()]
# There are 488 rows which have null country values and 4 rows which have null children values
# These rows are not esential in terms of our purpose, therefore we will be dropping them.

data.drop(data[data['country'].isnull()].index, axis=0, inplace=True)
data.drop(data[data['children'].isnull()].index, axis=0, inplace=True)
total = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum() / data.isnull().count()).sort_values(ascending=False)

nan_cols = pd.concat([total, percent], axis=1, keys=['Total', 'nan'])

nan_cols = nan_cols[nan_cols['nan'] > 0]

nan_cols
def plot_canceling_prob(col_name: str, data: pd.DataFrame):
    """
    Displays canceling probabilities for categorical data.
    """
    
    plt.figure(figsize=(16, 8))
    
    x = data.groupby('is_canceled')[col_name].value_counts(sort=True, normalize=True)[1].keys().values
    y = data.groupby('is_canceled')[col_name].value_counts(sort=True, normalize=True)[1].values
    leg = data.groupby('is_canceled')[col_name].value_counts(normalize=True, sort=True)[1].values
    

    g = sns.barplot(x, y)
    g.set(title=f'Canceled Booking Distribution on {col_name}')
    
    plt.legend(leg)
    plt.show(g)
def count_cat_prob_plot(col_name: str, data: pd.DataFrame):
    
    g1 = sns.countplot(x=col_name, data=data)
    plt.title(f"Count Plot for {col_name}")
    plt.show(g1)
    
    g2 = sns.catplot(x=col_name, y='is_canceled', data=data, kind='bar', aspect=3)
    plt.title(f"Canceling Probabilities for each {col_name}")
    plt.show(g2)
    
    plot_canceling_prob(col_name, data) 
# Looking at the overall data, after handling with null data 

data.shape 
data.columns
# seaborn initial settings

sns.set(context='notebook', palette='Set1', style='whitegrid', rc={'figure.figsize':(16, 8)})
columns_to_remove = list()
columns_to_dummy = list()
# keep analysis for each feature

analysis = {}

for col in data.columns:
    analysis[col] = []
data_corr = data.corr()

column = 'is_canceled'
corr_cols = data.shape[1]

cols = data_corr.nlargest(corr_cols, column)[column].index
coef = data_corr.nlargest(corr_cols, column)[cols].values

plt.figure(figsize=(16, 16))

g = sns.heatmap(coef, cbar=True, annot=True, square=True, fmt='.2f', 
                yticklabels=cols.values, xticklabels=cols.values)
data['hotel'].unique()
plt.pie(data['hotel'].value_counts().values, labels=data['hotel'].value_counts().keys())

plt.title("Hotels")
plt.show()
g = sns.countplot(x='hotel', hue='is_canceled', data=data)

g.set_title("Hotels")

plt.show(g)
analysis['hotel'].append('City hotel has more bookings and higher cancellation rates.')
g = sns.catplot(x='hotel', y='is_canceled', data=data, kind='bar', height=7, aspect=2)

g.despine(left=True) # removes axis line. Here removes y axis line.

g.set(xlabel='Hotel Name', ylabel='Canceling Probability', title="Hotel's Canceling Probabilities")

plt.show(g)
analysis['hotel'].append('Customers who booked to City hotel more likely to cancel their bookings.')
plot_canceling_prob('hotel', data)
analysis['hotel'].append("In total, City hotel has more canceled bookings. This because City hotel's higher number of bookings compared to Resort Hotel.")
#Displaying final indidual analysis on hotel

analysis['hotel']
# Total lost money due to canceling booking for each hotel


resort = data[data['hotel'] == 'Resort Hotel'].copy()
city = data[data['hotel'] == 'City Hotel'].copy()

resort['total_stays'] = resort['stays_in_week_nights'] + resort['stays_in_weekend_nights']
city['total_stays'] = city['stays_in_week_nights'] + city['stays_in_weekend_nights']

resort['customer_total_payment'] = resort['adr'].values * resort['total_stays'].values
city['customer_total_payment'] = city['adr'] * city['total_stays']

resort_lost_revenue = resort[resort['is_canceled'] == 1]['customer_total_payment'].sum()
city_lost_revenue = city[city['is_canceled'] == 1]['customer_total_payment'].sum()

resort_total_revenue = resort['customer_total_payment'].sum()
city_total_revenue = city['customer_total_payment'].sum()

sns.set_color_codes("pastel")
g = sns.barplot(x=['Resort', 'City'], y=[resort_total_revenue, city_total_revenue], color='b')
sns.set_color_codes("muted")
g = sns.barplot(x=['Resort', 'City'], y=[resort_lost_revenue, city_lost_revenue], color='b')


plt.legend([f'Resort Total: {round(resort_total_revenue)} - Lost: {round(resort_lost_revenue)}',
            f'City Total: {round(city_total_revenue)} - Lost: {round(city_lost_revenue)}'])
plt.title('Lost money due to Canceling Bookings')
plt.show()

g = sns.countplot(x='arrival_date_month', data=data, hue='hotel')

plt.title("Occupancy Rate")
plt.legend(['resort', 'city'])
plt.show(g)
data['total_stays'] = data['stays_in_week_nights'] + data['stays_in_weekend_nights']
data['customer_total_payment'] = data['adr'] * data['total_stays']


months = ['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
              'October', 'November', 'December']

g = sns.lineplot(x='arrival_date_month', y='adr', data=data, hue='hotel', color='r')

plt.title("Montly ADR")
plt.legend(['resort', 'city'])
plt.show(g) 
columns_to_dummy.append('hotel')
data['lead_time'].describe()
g = sns.distplot(a=data['lead_time'], label='lead_time_distribution')

plt.xlim([0, 750])
plt.show(g)
analysis['lead_time'].append('We see that there is a positive skewness in the lead time.')
# lead_time vs canceled

g = sns.FacetGrid(data, col='is_canceled', height=6)
g = g.map(sns.distplot, 'lead_time')

plt.xlim([0, 750])
plt.show(g)
# lead_time dist vs is_canceled

g = sns.kdeplot(data['lead_time'][data['is_canceled'] == 0],
               color='Red', shade=True)

g = sns.kdeplot(data['lead_time'][data['is_canceled'] == 1],
               color='Blue', shade=True)

g.set_xlabel('lead_time')
g.set_ylabel('Freq')

g = g.legend(['Not Canceled', 'Canceled'])

plt.plot([50, 50], [0.00, 0.0045], ':')
plt.xlim([0, 750])
plt.show(g)
analysis['lead_time'].append('High lead time causes high canceling probability.')
data['lead_time'].min(), data['lead_time'].mean(), data['lead_time'].max()
# we can use binning method to convert lead time in day to months 

data['lead_time_30'] = data['lead_time'] // 30
data['lead_time_60'] = data['lead_time'] // 60
data['lead_time_120'] = data['lead_time'] // 120
data['lead_time_360'] = data['lead_time'] // 360

# respect to 30 days binning.
g = sns.countplot(x='lead_time_30', hue='is_canceled', data=data)

plt.title('lead time 30 day binned')
plt.show(g)
# prices according to lead_time

# we will use monthly lead_time

g = sns.lineplot(x='lead_time_30', y='adr', data=data, hue='hotel', markers=True, dashes=False)

plt.title("Lead time vs ADR")
plt.legend(['resort', 'city'])
plt.show(g) 

analysis['lead_time'].append('Bookings are maded for 7 months later more likely to be canceled.')
countries_lead_time = data.groupby('country')['lead_time'].sum().reset_index(name = 'Total Lead Time')
# Lead time averages by countries

import plotly.express as px

px.choropleth(countries_lead_time,
                    locations = "country",
                    color= "Total Lead Time", 
                    hover_name= "Total Lead Time",
                    color_continuous_scale=px.colors.sequential.Oranges,
                    title="Lead Time by Countries")

columns_to_remove.extend(['lead_time_60', 'lead_time_30', 'lead_time_120', 'lead_time_360'])
#Showcasing final analysis on lead_time 

analysis['lead_time']
data['arrival_date_year'].describe()
data['arrival_date_year'].unique()
data['arrival_date_year'].value_counts()
g1 = sns.countplot(x='arrival_date_year', hue='hotel', data=data)

plt.title('Yearly Occupation Rate')
plt.show(g1)
g2 = sns.countplot(x='arrival_date_year', hue='is_canceled', data=data)

plt.title('Yearly Canceling Counts')
plt.show(g2)
analysis['arrival_date_year'].append('The highest number of booking belongs to 2016 then 2017 and 2015.')
g = sns.catplot(x='arrival_date_year', y='is_canceled', data=data, kind='bar', height=7, aspect=2)

g.despine(left=True)

g.set(title='Canceling Probabilities - Arrival Years')
analysis['arrival_date_year'].append('2015 - 2016 - 2017 have similar canceling probabilities.')
g = sns.catplot(x='arrival_date_year', y='is_canceled', hue='hotel', data=data, kind='bar', height=7, aspect=2)

g.despine(left=True)

g.set(title='Canceling Probabilities - Arrival Years')

plt.show(g)
g = sns.boxplot(x='arrival_date_year', y='adr', data=data, hue='hotel')

plt.title("Yearly ADR")
plt.show(g) 
## Revenue Total and lost by years

yearly_revenue_city = data[(data['hotel'] == 'City Hotel') & (data['is_canceled'] == 0)].groupby('arrival_date_year')['adr'].sum().reset_index(name = 'Total Revenue')
yearly_revenue_resort = data[(data['hotel'] == 'Resort Hotel') & (data['is_canceled'] == 0)].groupby('arrival_date_year')['adr'].sum().reset_index(name = 'Total Revenue')

yearly_revenue_city_cancel = data[(data['hotel'] == 'City Hotel') & (data['is_canceled'] == 1)].groupby('arrival_date_year')['adr'].sum().reset_index(name = 'Total Revenue')
yearly_revenue_resort_cancel = data[(data['hotel'] == 'Resort Hotel') & (data['is_canceled'] == 1)].groupby('arrival_date_year')['adr'].sum().reset_index(name = 'Total Revenue')


g = sns.lineplot(x='arrival_date_year', y='Total Revenue', data=yearly_revenue_resort)
g = sns.lineplot(x='arrival_date_year', y='Total Revenue', data=yearly_revenue_resort_cancel)

g = sns.lineplot(x='arrival_date_year', y='Total Revenue', data=yearly_revenue_city)
g = sns.lineplot(x='arrival_date_year', y='Total Revenue', data=yearly_revenue_city_cancel)


plt.title("Yearly Revenue")
plt.legend(['Resort Revenue', 'Resort Lost Revenue', 'City Revenue', 'City Lost Revenue'])
plt.show(g) 
analysis['arrival_date_year'].append('In each year canceling probability is higher for Ciy Hotel.')
columns_to_dummy.append('arrival_date_year')
#Showcasing final analysis on arrival_date_year

analysis['arrival_date_year']
data['arrival_date_week_number'].describe()
data['arrival_date_week_number'].value_counts()[:10]
# we have info for arrival month that's why we can use week info to detect which week in a month they arrived.

data['arrival_date_weekth_in_month'] = data['arrival_date_week_number'] % 4
data['arrival_date_weekth_in_month'].describe()
data['arrival_date_weekth_in_month'].value_counts(sort=False)
g = sns.catplot(x='arrival_date_weekth_in_month', y='is_canceled', data=data, kind='bar', height=7, aspect=2)

g.despine(left=True)

g.set(title='Canceling Probabilities - Arrival Weekth of Month')

plt.show(g)
plot_canceling_prob('arrival_date_weekth_in_month', data)
analysis['arrival_date_week_number'].append('Canceling probability is high for weeks 1st and 2nd.')
g = sns.boxplot(x='arrival_date_weekth_in_month', y='adr', data=data, hue='hotel')

plt.title("Ith week ADR")
plt.show(g) 
columns_to_dummy.append('arrival_date_weekth_in_month')
#Showcasing final analysis on arrival_date_week_number 
analysis['arrival_date_week_number']
data['arrival_date_month'].describe()
data['arrival_date_month'].unique()
canceled_months = data[data['is_canceled'] == 1].groupby('arrival_date_month').size().reset_index(name='Total')
total_months = data.groupby('arrival_date_month').size().reset_index(name='Total')

g = sns.barplot(x='arrival_date_month', y='Total', data=total_months)
g = sns.barplot(x='arrival_date_month', y='Total', data=canceled_months, color='r')

plt.title("Montly Bookings vs Canceled Bookings")
plt.show()
g = sns.catplot(x='arrival_date_month', y='is_canceled', data=data, kind='bar', height=7, aspect=2)

g.despine(left=True)

g.set(title='Arrival Month - Canceling Probability')

plt.show(g)
analysis['arrival_date_month'].append('We have higher canceling probabilities for summer.')
# we may try to generate season data.

def month_to_season(month):
    
    if month in ['June', 'July', 'August']:
        return "summer"
    elif month in ['March', 'April', 'May']:
        return "spring"
    elif month in ['October', 'November', 'September']:
        return "autumn"
    else:
        return "winter"
data['seasons'] = data['arrival_date_month'].apply(month_to_season)
data['seasons'].value_counts()
canceled_seasons = data[data['is_canceled'] == 1].groupby('seasons').size().reset_index(name='Total')
total_seasons = data.groupby('seasons').size().reset_index(name='Total')

g = sns.barplot(x='seasons', y='Total', data=total_seasons)
g = sns.barplot(x='seasons', y='Total', data=canceled_seasons, color='r')

plt.title("Seasons Bookings vs Canceled Bookings")
plt.show()

g = sns.catplot(x='seasons', y='is_canceled', data=data, kind='bar', height=7, aspect=2)

g.despine(left=True)

g.set(title='Arrival Season - Canceling Probability')

plt.show(g)
analysis['arrival_date_month'].append('Lowest Cancel Probability is for winter season.')
plot_canceling_prob('seasons', data)
g = sns.boxplot(x='seasons', y='adr', data=data, hue='hotel')

plt.title("Seasons ADR")
plt.show(g) 
g = sns.catplot(x='seasons', y='is_canceled', hue='hotel', data=data, kind='bar', height=7, aspect=2)

g.despine(left=True)

g.set(title='Arrival Season - Canceling Probability')

plt.show(g)
columns_to_dummy.extend(['arrival_date_month'])
#Showcasing final analysis on arrival_date_month
analysis['arrival_date_month']
columns_to_remove.append('seasons')
data['arrival_date_day_of_month'].describe()
g = sns.countplot(x='arrival_date_day_of_month', data=data)

g.set(title='Arrival Day of Month - Reservation Count')

plt.show(g)
g = sns.catplot(x='arrival_date_day_of_month', y='is_canceled', data=data, kind='bar', height=7, aspect=2)

g.despine(left=True)

g.set(title='Arrival Day of Month - Canceling Probability')

plt.show(g)
g = sns.catplot(x='arrival_date_day_of_month', y='is_canceled', hue='hotel', data=data, kind='bar', height=7, aspect=2)

g.despine(left=True)

g.set(title='Arrival Day of Month - Canceling Probability')

plt.show(g)
g = sns.lineplot(x='arrival_date_day_of_month', y='adr', data=data, hue='hotel')

plt.xticks(range(0, 31))
plt.title("Arrival Date of Month ADR")
plt.show(g) 
# we will convert day of month to day of week

def date_to_day_of_week(row):
    
    import datetime
    
    months = ['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
              'October', 'November', 'December']
    
    year = row['arrival_date_year']
    month = months.index(row['arrival_date_month'])
    day = row['arrival_date_day_of_month']
    
    arrival_date = datetime.date(year, month, day)
    
    # row['arrival_date_day_of_week'] = arrival_date.strftime("%A")
    
    return arrival_date.strftime("%A")
data['arrival_date_day_of_week'] = np.nan
data['arrival_date_day_of_week'] = data.reset_index().apply(date_to_day_of_week, axis=1)
g = sns.countplot(x='arrival_date_day_of_week', data=data)

g.set(title='Arrival Day of Week')
plt.show(g)
g = sns.lineplot(x='arrival_date_day_of_week', y='adr', data=data, hue='hotel')

plt.title("Arrival Date of Week ADR")
plt.show(g) 
g = sns.catplot(x='arrival_date_day_of_week', y='is_canceled', data=data, kind='bar', height=7, aspect=2)

g.despine(left=True)

g.set(title='Arrival Day of Week - Canceling Probability')

plt.show(g)
g = sns.catplot(x='arrival_date_day_of_week', y='is_canceled', hue='hotel', data=data, kind='bar', height=7, aspect=2)

g.despine(left=True)

g.set(title='Arrival Day of Week - Canceling Probability')

plt.show(g)
analysis['arrival_date_day_of_month'].append('Risky days for hotels differ from each other.')
# we will drop the column of day_of_month

columns_to_remove.append('arrival_date_day_of_month')
columns_to_dummy.append('arrival_date_day_of_week')
#Showcasing final analysis on 'arrival_date_day_of_month'
analysis['arrival_date_day_of_month']
data['stays_in_weekend_nights'].isna().sum()
data['stays_in_weekend_nights'].describe()
sorted(data['stays_in_weekend_nights'].unique())
g = sns.catplot(x='stays_in_weekend_nights', y='is_canceled', data=data, kind='bar', height=8, aspect=2)

g.despine(left=True)

g.set(title='Weekend Nights - Canceling Probabilities')

plt.show(g)
data['stays_in_week_nights'].isna().sum()
data['stays_in_week_nights'].describe()
data['stays_in_week_nights'].value_counts(normalize=True)
g = sns.catplot(x='stays_in_week_nights', y='is_canceled', data=data, kind='bar', height=8, aspect=2)

g.despine(left=True)

g.set(title='Weekend Nights - Canceling Probabilities')

plt.show(g)
# we may check the total stay time.

data['stays_total'] = data['stays_in_week_nights'] + data['stays_in_weekend_nights']
data['stays_total'].describe()
g = sns.catplot(x='stays_total', y='is_canceled', data=data, kind='bar', height=8, aspect=2)

g.despine(left=True)

g.set(title='Weekend Nights - Canceling Probabilities')

plt.show(g)
g = sns.boxplot(x='total_of_special_requests', y='stays_total', data=data, hue='hotel')

plt.title("Total Special Requests vs Stays Total")
plt.show(g)
g = sns.boxplot(x='is_repeated_guest', y='stays_total', data=data, hue='hotel')

plt.title("Repeated Guest vs Stays Total")
plt.show(g)
data['adults'].describe()
data['adults'].unique()
data['adults'].value_counts()
g = sns.countplot(x='adults', data=data)

plt.show(g)
g = sns.catplot(x='adults', y='is_canceled', data=data, kind='bar', height=8, aspect=2)

g.despine(left=True)

plt.show(g)
# we can create a column for all adults count > 4

def adults_large(adults):
    
    if adults > 4:
        return 5
    else:
        return adults
    
data['adults'] = data['adults'].apply(adults_large)
g = sns.catplot(x='adults', y='is_canceled', data=data, kind='bar', height=8, aspect=2)

g.despine(left=True)

plt.show(g)
data['children'].value_counts()
# 10 children seems to be wrong or outliar value.

data[data['children'] > 9]
data.drop(data[data['children'] > 9].index, axis=0, inplace=True)
g = sns.catplot(x='children', y='is_canceled', data=data, kind='bar', height=8, aspect=2)

g.despine(left=True)

plt.show(g)
data['babies'].describe()
data['babies'].unique()
data['babies'].value_counts()
data[data['babies'] == 9]
data[data['babies'] == 10]
data.groupby('babies')['is_canceled'].value_counts(normalize=True)
# we will remove the outliar baby counts

data.drop(data[data['babies'] > 8].index, axis=0, inplace=True)
g = sns.catplot(x='babies', y='is_canceled', data=data, kind='bar', height=8, aspect=2)

g.despine(left=True)

plt.show(g)
data.drop(data[data['babies'] > 8].index, axis=0, inplace=True)
data['babies'].value_counts() # we don't have outliars any more.
data['meal'].unique()
# 'SC' and 'undefined' means same thing no meal
data['meal'].replace(['SC', 'Undefined'], 'NoMeal', inplace=True)
g = sns.countplot(x='meal', hue='is_canceled', data=data)

g.set(title='Meal Type')

plt.show(g)
g = sns.catplot(x='meal', y='is_canceled', data=data, kind='bar', aspect=3)

g.set(title='Each meal type canceling Probs')

plt.show(g)
g = sns.lineplot(x='meal', y='adr', data=data, hue='hotel')

plt.title("Meal vs ADR")
plt.show(g)
columns_to_dummy.append('meal')
countries_bookings = data.groupby(['country']).size().reset_index(name = 'Total')
canceled_countries = data[data['is_canceled'] == 1].groupby(['country']).size().reset_index(name = 'Canceled')
not_canceled_countries = data[data['is_canceled'] == 0].groupby(['country']).size().reset_index(name = 'Not_Canceled')
import pycountry

def country_code_to_name(country_code):
    
    if len(country_code) == 2:
        country = pycountry.countries.get(alpha_2=country_code)
    else:
        country = pycountry.countries.get(alpha_3=country_code)

    if not country:
        return 'Not Found'
    else:
        return country.name
        
countries_bookings['country_name'] = countries_bookings['country'].apply(country_code_to_name)
not_canceled_countries['country_name'] = not_canceled_countries['country'].apply(country_code_to_name)
canceled_countries['country_name'] = canceled_countries['country'].apply(country_code_to_name)
import plotly.express as px

px.choropleth(countries_bookings,
                    locations = "country",
                    color= "Total", 
                    hover_name= "country_name",
                    color_continuous_scale=px.colors.sequential.Oranges,
                    title="Booking Counts by Countries")
px.choropleth(not_canceled_countries,
                    locations = "country",
                    color= "Not_Canceled", 
                    hover_name= "country_name",
                    color_continuous_scale=px.colors.sequential.Oranges,
                    title="Not_Canceled Booking Counts by Countries")
px.choropleth(canceled_countries,
                    locations = "country",
                    color= "Canceled", 
                    hover_name= "country_name",
                    color_continuous_scale=px.colors.sequential.Oranges,
                    title="Canceled Booking Counts by Countries")
columns_to_dummy.append('country')
data['market_segment'].unique()
g = sns.countplot(x='market_segment', data=data)

plt.show(g)
g = sns.catplot(x='market_segment', y='is_canceled', data=data, kind='bar', aspect=3)

plt.show(g)
plot_canceling_prob('market_segment', data)
columns_to_dummy.append('market_segment')
data['distribution_channel'].unique()
count_cat_prob_plot('distribution_channel', data)
columns_to_dummy.append('distribution_channel')
data['is_repeated_guest'].unique()
count_cat_prob_plot('is_repeated_guest', data)
g = sns.countplot(x='is_repeated_guest', data=data, hue='is_canceled')
analysis['is_repeated_guest'].extend(['Most of booking are from new customers.', 'Repeated bookings have less canceling probability than new comers.'])
#Showcasing final analysis on is_repeated_guest
analysis['is_repeated_guest']
data['previous_bookings_not_canceled'].unique()
data['previous_bookings_not_canceled'].describe()
def cancel_ratio(row):
        
    if not row['previous_bookings_not_canceled'] + row['previous_cancellations'] == 0:
        return row['previous_cancellations'] / (row['previous_bookings_not_canceled'] + row['previous_cancellations'])
    else:
        return 0
data['customer_cancel_ratio'] = data.apply(cancel_ratio, axis=1)
data['customer_cancel_ratio'].describe()
data.groupby('is_canceled')['customer_cancel_ratio'].mean()
g = sns.barplot(x='is_canceled', y='customer_cancel_ratio', data=data)


plt.title("Customer Cancel Ratio")
plt.show(g)
columns_to_remove.extend(['previous_bookings_not_canceled', 'previous_cancellations'])
data['previous_cancellations'].unique()
data['previous_cancellations'].value_counts()
count_cat_prob_plot('previous_cancellations', data[data['previous_cancellations'] > 1])
data['reserved_room_type'].unique()
count_cat_prob_plot('reserved_room_type', data)
def is_room_changed(row):
    
    if row['assigned_room_type'] == row['reserved_room_type']:
        return 0
    else:
        return 1
data['room_changed'] = data.apply(is_room_changed, axis=1)
g = sns.catplot(x='room_changed', y='is_canceled', data=data, kind='bar', aspect=2, height=8)
columns_to_dummy.append('reserved_room_type')
data['assigned_room_type'].unique()
count_cat_prob_plot('assigned_room_type', data)
### assigned room and reserved_room looks like similar columns that's why we will remove it.

columns_to_dummy.append('assigned_room_type')
data['booking_changes'].describe()
data['booking_changes'].unique()
data['booking_changes'].value_counts()
count_cat_prob_plot('booking_changes', data)
data['deposit_type'].describe()
data['deposit_type'].unique()
data.groupby('deposit_type')['is_canceled'].value_counts(normalize=True)
count_cat_prob_plot('deposit_type', data)
g = sns.lineplot(x='deposit_type', y='customer_total_payment', data=data, hue='hotel')
columns_to_dummy.append('deposit_type')
data['days_in_waiting_list'].describe()
data['days_in_waiting_list'].unique()
data['days_in_waiting_list_30'] = data['days_in_waiting_list'] // 30
count_cat_prob_plot('days_in_waiting_list_30', data[data['days_in_waiting_list_30'] > 0])
data['customer_type'].unique()
count_cat_prob_plot('customer_type', data)
columns_to_dummy.append('customer_type')
data['adr'].describe()
data[data['adr'] < 0]['adr']
data.drop(data[data['adr'] < 0].index, axis=0, inplace=True)
g = sns.scatterplot(x=data.index, y=data['adr'], hue=data['is_canceled'])

plt.show(g)
# We have an outliar for adr column which greater than 5000
data[data['adr'] > 1000]
data.drop(data[data['adr'] > 1000].index, axis=0, inplace=True)
g = sns.scatterplot(x=data.index, y=data['adr'], hue=data['is_canceled'])

plt.show(g)
data['required_car_parking_spaces'].describe()
data['required_car_parking_spaces'].unique()
data['required_car_parking_spaces'].value_counts()
count_cat_prob_plot('required_car_parking_spaces', data)
def car_parking_space(required_park):
    
    if required_park > 0:
        return 1
    else:
        return 0
data['required_car_parking_spaces'] = data['required_car_parking_spaces'].apply(car_parking_space)
count_cat_prob_plot('required_car_parking_spaces', data)
columns_to_remove.append('required_car_parking_spaces')
data['total_of_special_requests'].unique()
data['total_of_special_requests'].value_counts()
count_cat_prob_plot('total_of_special_requests', data)
g = sns.lineplot(x='total_of_special_requests', y='customer_total_payment', data=data, hue='hotel')

plt.title('Special Requests vs Total Payment')
plt.show(g)
data['reservation_status'].unique()
count_cat_prob_plot('reservation_status', data)
columns_to_remove.append('reservation_status')
columns_to_remove.append('reservation_status_date')
from sklearn.utils import shuffle

data = shuffle(data).reset_index(drop=True)
data.shape
columns_to_remove
columns_to_dummy
cleaned_data = data.drop(columns=columns_to_remove, axis=1)
cleaned_data = pd.get_dummies(cleaned_data, columns=columns_to_dummy)
cleaned_data.shape # because of dummy columns we have too many columns. 
cleaned_data.info() # we removed all object (str) data.
