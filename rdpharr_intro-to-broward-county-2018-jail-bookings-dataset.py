%load_ext autoreload
%autoreload 2
%matplotlib inline

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore",category=DeprecationWarning)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
charges_df = pd.read_csv("../input/anon_broward_bookings/anon_broward_bookings.csv")
charges_df.T
#convert necessary columns to dates
date_fmt = "%m/%d/%Y %I:%M:%S %p"
charges_df.arrest_date = pd.to_datetime(charges_df.arrest_date, format=date_fmt)
charges_df.admitted_date = pd.to_datetime(charges_df.admitted_date, format=date_fmt).dt.date
charges_df.released_date = pd.to_datetime(charges_df.released_date, format=date_fmt).dt.date

charges_df.dob = pd.to_datetime(charges_df.dob, format="%m/%d/%Y").dt.date
#create df with one row per person
charges_df['num_charges'] = charges_df.groupby(['name_anonymized', 'dob','arrest_date'])['charge'].transform('count')
people_cols = ['name_anonymized', 'sex', 'race', 'place_of_birth', 'dob',
       'home_address_anonymized', 'state', 'city', 'zip',
       'arresting_officer_anonymized', 'arrest_agency', 'arrest_date',
       'admitted_date','num_charges']
people_df = charges_df[people_cols].drop_duplicates().reset_index(drop=True)
from fbprophet import Prophet
bookings = pd.DataFrame({'y' : charges_df.groupby(["admitted_date"])['name_anonymized'].nunique()}).reset_index()
bookings.rename(columns={'admitted_date':'ds'}, inplace=True)

m = Prophet(daily_seasonality=False, yearly_seasonality=False)
m.fit(bookings)
future = m.make_future_dataframe(periods=0) #no forecast
prophet_forecast = m.predict(future)

ax = m.plot(prophet_forecast)
ax = m.plot_components(prophet_forecast)
plt.figure(figsize=(15,10))
ax = sns.countplot(y=charges_df.charge, order=charges_df.charge.value_counts().iloc[:25].index
                  ).set_title("Most Common Charges")
race_df = people_df.pivot_table(columns='race', aggfunc={'num_charges':'mean', 'name_anonymized':'count'}
                     ).T.sort_values(by='name_anonymized', ascending = False).reset_index()
race_df.rename(columns={'name_anonymized':"people_booked", 'num_charges':'charges_per_booking'}, inplace=True)
race_df['portion_of_total'] = race_df.people_booked / len(people_df)
race_df
portion_black = len(people_df[people_df.race=='B'])/len(people_df)
print(f"Broward County jail bookings are {portion_black*100:.1f}% black people, even though the county has only 20.5% black people. ")
plt.figure(figsize=(15,5))
plt.suptitle("Number of Charges Per Arrest by Race", fontsize='xx-large')
ax = sns.barplot(x="race", y="num_charges", data=people_df, ci=0.2)
people_df['Home'] = 'Tourist'
people_df['Home'].loc[people_df.state=='FL'] ='Local'
people_df['Home'].loc[people_df.city.isna()]='Homeless'

plt.figure(figsize=(15,5))
plt.suptitle("Where the Inmates Are From", fontsize='xx-large')
ax = sns.countplot(y=people_df.Home)
homeless_charges = charges_df[charges_df.state.isna()].copy()
plt.figure(figsize=(15,10))
ax = sns.countplot(y=homeless_charges.charge, order=homeless_charges.charge.value_counts().iloc[:25].index
                  ).set_title("Most Common Charges for Homeless")
tourist_charges = charges_df.loc[charges_df.state != 'FL'].copy()
tourist_charges = tourist_charges.loc[-tourist_charges.state.isna()]
plt.figure(figsize=(15,10))
ax = sns.countplot(y=tourist_charges.charge, order=tourist_charges.charge.value_counts().iloc[:25].index
                  ).set_title("Most Common Charges for Tourists")
