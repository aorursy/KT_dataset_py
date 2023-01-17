# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import plotly.graph_objects as go
from tqdm import tqdm # progress bars for slow pandas ops
import matplotlib.pyplot as plt

sns.set()

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/lending-club-loan-data/loan.csv', low_memory=False)
pd.set_option('display.max_columns', None)

# Register tqdm with pandas
tqdm.pandas()
for column in df.columns:
    ## Remove columns that contain only NaN
    if df[column].isnull().values.any():
        df.drop(column, axis=1, inplace=True)
df['issue_d'] = df['issue_d'].progress_apply(lambda date: pd.to_datetime(date))
df.head(1)
df_description = pd.read_excel('/kaggle/input/lending-club-loan-data/LCDataDictionary.xlsx').dropna()

for column_name in df_description['LoanStatNew']:
    try:
        _ = df[column_name]
    except KeyError:
        df_description.drop(df_description[df_description['LoanStatNew'] == column_name].index, inplace=True)
df_description.style.set_properties(subset=['Description'], **{'width': '1000px'})
df['loan_status'].value_counts()
df['Fail'] = df['loan_status'].progress_apply(lambda status: np.NaN if status in ('Does not meet the credit policy. Status:Fully Paid', 
                                                                                  'Does not meet the credit policy. Status:Charged Off') 
                                              else status in ('Charged Off', 'Default'))
df['loan_status_num'] = df['loan_status'].progress_apply(lambda status: {'Fully Paid': 0, 
                                                                         'Current': 1, 
                                                                         'In Grace Period': 2, 'Late (16-30 days)': 3, 
                                                                         'Late (31-120 days)': 4, 
                                                                         'Charged Off': 5, 
                                                                         'Default': 6}.get(status, np.NaN))
df['purpose'].value_counts()
(round(pd.crosstab(df['purpose'], df[(df['loan_status'] != 'Does not meet the credit policy. Status:Fully Paid') & 
                                     (df['loan_status'] != 'Does not meet the credit policy. Status:Charged Off')]['loan_status'], 
                   normalize='columns') * 100,2)).style.background_gradient(cmap = sns.light_palette("green", as_cmap=True))
pd.crosstab(df[(df['loan_status'] != 'Does not meet the credit policy. Status:Fully Paid') &
               (df['loan_status'] != 'Does not meet the credit policy. Status:Charged Off')]['loan_status'], 
            df['grade']).style.background_gradient(cmap = sns.light_palette("green", as_cmap=True))
%matplotlib inline
plt.figure(figsize=(11.7,8.27))
sns.catplot(x='loan_amnt', y="purpose",hue='Fail',
            kind='violin', split=True, data=df);
g = sns.catplot(x="home_ownership",
                   y="loan_amnt",
                   data=df,
                   kind="violin",
                   split=True,
                   palette="coolwarm",
                   col="Fail",
                   hue="application_type")

g1 = sns.catplot(x="home_ownership",y="int_rate",data=df,
                 kind="violin",
                 split=True,
                 palette="coolwarm",
                 col="Fail",
                 hue="application_type")
g = sns.catplot(x="application_type",
                   y="loan_amnt",
                   data=df,
                   kind="violin",
                   palette="coolwarm",
                   col="Fail",
                   hue="application_type")

g1 = sns.catplot(x="application_type",
                 y="int_rate",
                 data=df,
                 kind="violin",
                 palette="coolwarm",
                 col="Fail",
                 hue="application_type")
# https://stackoverflow.com/questions/41623538/finding-median-time-stamp-in-python
median_loan_issue_date = pd.Timestamp.fromordinal(
    int(df['issue_d'].apply(lambda x: x.toordinal()).median()))
print(f'Median Loan Issue Date {median_loan_issue_date}')
# the size of A4 paper
sns.set_palette('husl')
sns.set(rc={'figure.figsize':(11.7,8.27)})

sns.lineplot(x='issue_d', y='int_rate', style='grade', hue='grade', estimator=np.mean, dashes=False, data=df)
df_pop = pd.read_csv('http://www2.census.gov/programs-surveys/popest/datasets/2010-2019/national/totals/nst-est2019-alldata.csv?#')
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

# thank you to @kinghelix and @trevormarburger for this idea
abbrev_us_state = dict(map(reversed, us_state_abbrev.items()))
df_pop['NAME'] = df_pop['NAME'].apply(lambda state_name: us_state_abbrev.get(state_name, state_name))
df_pop.head(1)
states = list(us_state_abbrev.values())
df_states = pd.DataFrame(columns = ['state', 'funding_per_cap', 'total_funding', 'total_issued_loans', 'total_issued_loans_per_cap', 'failure_severity_per_capita'])
median_loan_issue_year = str(median_loan_issue_date.year)

for state in states:
    try:
        # Number of loans issued for this state
        total_loans_issued = len(df[df['addr_state'] == state])

        ## Calculate total amount of money issued for this state
        total_funded = df[df['addr_state'] == state]['funded_amnt'].sum()

        pop_at_median_loan_issue_date = df_pop[df_pop['NAME'] == state]['POPESTIMATE' + median_loan_issue_year].iloc[0]
        total_funded_per_cap = total_funded / pop_at_median_loan_issue_date
        total_loans_issued_per_cap = total_loans_issued / pop_at_median_loan_issue_date
        
        ## Calculate failure severity by taking the average of loan status scores for this state
        avg_failure_severity = df[df['addr_state'] == state]['loan_status_num'].mean()

        df_states = df_states.append({'state': state, 
                                      'funding_per_cap': total_funded_per_cap, 
                                      'total_funding': total_funded,
                                      'total_issued_loans': total_loans_issued, 
                                      'total_issued_loans_per_cap': total_loans_issued_per_cap, 
                                      'failure_severity_per_capita': avg_failure_severity}, ignore_index=True)
    except IndexError:
        pass
import cufflinks as cf
import plotly
from plotly.offline import init_notebook_mode
init_notebook_mode()
cf.go_offline()
fig = go.Figure(data=go.Choropleth(
    locations=df_states['state'], # Spatial coordinates
    z = df_states['total_funding'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Greens',
    colorbar_title = 'Dollars Issued',
))

fig.update_layout(
    title_text = 'Total Loan Amount Issued by State',
    geo_scope='usa', # limite map scope to USA
)

fig.show()
fig = go.Figure(data=go.Choropleth(
    locations=df_states['state'], # Spatial coordinates
    z = df_states['total_issued_loans'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = 'Dollars Issued',
))

fig.update_layout(
    title_text = 'Total # Loans Issued Per Capita by State',
    geo_scope='usa', # limite map scope to USA
)

fig.show()
fig = go.Figure(data=go.Choropleth(
    locations=df_states['state'], # Spatial coordinates
    z = df_states['funding_per_cap'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Blues',
    colorbar_title = 'Dollars Issued',
))

fig.update_layout(
    title_text = 'Total Loan Amount Issued Per Capita by State',
    geo_scope='usa', # limite map scope to USA
)

fig.show()
fig = go.Figure(data=go.Choropleth(
    locations=df_states['state'], # Spatial coordinates
    z = df_states['failure_severity_per_capita'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = 'Failure Severity',
))

fig.update_layout(
    title_text = 'Average Failure Severity Per Capita',
    geo_scope='usa', 
)

fig.show()
from scipy.stats import zscore
state = 'CA'
total_loans_issued = len(df[df['addr_state'] == state])

## Calculate total amount of money issued for this state
total_funded = df[df['addr_state'] == state]['funded_amnt'].sum()

pop_at_median_loan_issue_date = df_pop[df_pop['NAME'] == state]['POPESTIMATE' + median_loan_issue_year].iloc[0]
total_funded_per_cap = total_funded / pop_at_median_loan_issue_date
total_loans_issued_per_cap = total_loans_issued / pop_at_median_loan_issue_date

## Calculate failure severity by taking the average of loan status scores for this state
avg_failure_severity = df[df['addr_state'] == state]['loan_status_num'].mean()
print(set(zscore(df[df['addr_state'] == state]['loan_status_num'], nan_policy='omit')))
print(avg_failure_severity)
pickle_filename = '/kaggle/working/lending_pickle.pkl'
df.to_pickle(pickle_filename)
pickle_filename = '/kaggle/working/lending_pickle.pkl'
df = pd.read_pickle(pickle_filename)
