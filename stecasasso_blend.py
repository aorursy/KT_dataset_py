import pandas as pd
sub_expo = pd.read_csv('../input/submissions-covid19-public/submission_expo.csv')

sub_expo = sub_expo.rename({'ConfirmedCases': 'ConfirmedCases_expo', 'Fatalities': 'Fatalities_expo'}, axis=1)

sub_gam = pd.read_csv('../input/submissions-covid19-public/submission_gam.csv')

sub_gam = sub_gam.rename({'ConfirmedCases': 'ConfirmedCases_gam', 'Fatalities': 'Fatalities_gam'}, axis=1)

sub_power = pd.read_csv('../input/submissions-covid19-public/submission_power.csv')

sub_power = sub_power.rename({'ConfirmedCases': 'ConfirmedCases_power', 'Fatalities': 'Fatalities_power'}, axis=1)
sub = sub_expo.copy()

sub = sub.merge(sub_gam, on='ForecastId', how='left')

sub = sub.merge(sub_power, on='ForecastId', how='left')

sub.head()
sub['ConfirmedCases'] = sub[[c for c in sub.columns if c.startswith('ConfirmedCases_')]].mean(axis=1)

sub['Fatalities'] = sub[[c for c in sub.columns if c.startswith('Fatalities_')]].mean(axis=1)

sub.head()
sub[['ForecastId', 'ConfirmedCases', 'Fatalities']].to_csv('submission.csv', index=False)