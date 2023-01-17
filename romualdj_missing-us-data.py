import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

org = pd.read_csv('/kaggle/input/gthubdata-new/time_series_19-covid-Confirmed.csv')
train[train['Province/State']=='New York'].iloc[39:55]
org.columns
us = org[org['Country/Region']=='US']

us['Province/State'].unique()
days = us.columns[4:]

ny_state = us[us['Province/State']=='New York']

ny_counties = us[us['Province/State'].str.find('NY')>0]

(ny_counties[days].sum() + ny_state[days])[days[39:55]]
train[train['Province/State']=='New York'].iloc[39:55]
states = {

        'AK': 'Alaska',

        'AL': 'Alabama',

        'AR': 'Arkansas',

        'AS': 'American Samoa',

        'AZ': 'Arizona',

        'CA': 'California',

        'CO': 'Colorado',

        'CT': 'Connecticut',

        'DC': 'District of Columbia',

        'DE': 'Delaware',

        'FL': 'Florida',

        'GA': 'Georgia',

        'GU': 'Guam',

        'HI': 'Hawaii',

        'IA': 'Iowa',

        'ID': 'Idaho',

        'IL': 'Illinois',

        'IN': 'Indiana',

        'KS': 'Kansas',

        'KY': 'Kentucky',

        'LA': 'Louisiana',

        'MA': 'Massachusetts',

        'MD': 'Maryland',

        'ME': 'Maine',

        'MI': 'Michigan',

        'MN': 'Minnesota',

        'MO': 'Missouri',

        'MP': 'Northern Mariana Islands',

        'MS': 'Mississippi',

        'MT': 'Montana',

        'NA': 'National',

        'NC': 'North Carolina',

        'ND': 'North Dakota',

        'NE': 'Nebraska',

        'NH': 'New Hampshire',

        'NJ': 'New Jersey',

        'NM': 'New Mexico',

        'NV': 'Nevada',

        'NY': 'New York',

        'OH': 'Ohio',

        'OK': 'Oklahoma',

        'OR': 'Oregon',

        'PA': 'Pennsylvania',

        'PR': 'Puerto Rico',

        'RI': 'Rhode Island',

        'SC': 'South Carolina',

        'SD': 'South Dakota',

        'TN': 'Tennessee',

        'TX': 'Texas',

        'UT': 'Utah',

        'VA': 'Virginia',

        'VI': 'Virgin Islands',

        'VT': 'Vermont',

        'WA': 'Washington',

        'WI': 'Wisconsin',

        'WV': 'West Virginia',

        'WY': 'Wyoming'

}