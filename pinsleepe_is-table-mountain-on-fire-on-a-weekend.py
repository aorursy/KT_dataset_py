import numpy as np 
import pandas as pd 
csv_path = '../input/fire_log_cape_town.csv'
df = pd.read_csv(csv_path, parse_dates=['Datetime'])
df.head()
list(df.columns)
df['Datetime'].min()
df['Datetime'].max()
week_df = df.groupby(df['Datetime'].dt.weekday_name).count()
print(week_df['Incident_number'])
field = "Day"
day_order = ["Monday", "Tuesday", "Wednesday", 
             "Thursday", "Friday", "Saturday", "Sunday"]
ax = week_df['Incident_number'].loc[day_order].plot(kind="bar", 
                                                    legend=False,
                                                   figsize=(8,5));
ax.set_ylabel("Number of fires");
ax.set_xlabel("Day of the week");
fire_type = df['Description_of_Incident'].unique().tolist()
fire_type = [x for x in fire_type if str(x) != 'nan']
veg_fire = [s for s in fire_type if 'Vegetation' in s]

df_fire = df.loc[df['Description_of_Incident'].isin(veg_fire)]
veg_week_df = df_fire.groupby(df_fire['Datetime'].dt.weekday_name).count()
print(veg_week_df['Incident_number'])
ax = veg_week_df['Incident_number'].loc[day_order].plot(kind="bar", 
                                                        legend=False, 
                                                   figsize=(8,5));
ax.set_ylabel("Number of vegetation fires");
ax.set_xlabel("Day of the week");
dff = df_fire.reset_index().set_index('Datetime')
dff = dff[dff['Incident_number'].index.notnull()]
dff.loc[:, 'index'] = 1
fire_month = dff['index'].resample('M').sum().to_period(freq='M')
fire_month.rename(columns={'index': 'sum'}, inplace=True)
fire_month.head()
weekends = fire_month.index.month.isin([12,1])
n = 3

ax = fire_month.plot(figsize=(8,5),  legend=False)
ax.ticklabel_format(style='plain', axis='y')
ax.set_ylabel("Number of vegetation fires")
ax.set_xlabel("Date")

times = fire_month.loc[weekends].index
begins, ends = times[1:-1:2], times[2::2]
for begin, end in zip(begins, ends):
    ax.axvspan(begin, end, color='red', alpha=0.2)
