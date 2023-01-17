import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline
# covid cases in India
cases_df = pd.read_csv('../input/covid19-in-india/covid_19_india.csv').drop(['Sno', 'Time', 'ConfirmedIndianNational', 'ConfirmedForeignNational'], axis=1)
cases_df['Date'] = pd.to_datetime(cases_df['Date'], format='%d/%m/%y')
cases_df = cases_df.sort_values('Date')
# state wise count
state_wise_df = cases_df.groupby('State/UnionTerritory').tail(1)
state_wise_df = state_wise_df.sort_values('Confirmed', ascending=False).reset_index()
state_wise_df.loc['Total', :] = state_wise_df.sum(numeric_only=True, axis=0).reindex(state_wise_df.columns, fill_value='')
state_wise_df = state_wise_df.drop(['index'], axis=1)
state_wise_df[['Confirmed', 'Deaths', 'Cured']] = state_wise_df[['Confirmed', 'Deaths', 'Cured']].astype(int)
state_wise_df
# trend for Tamil Nadu
cases_df[cases_df['State/UnionTerritory'] == 'Tamil Nadu'].plot(x="Date", y=["Confirmed", "Deaths", "Cured"], kind="line", figsize=(15,10))