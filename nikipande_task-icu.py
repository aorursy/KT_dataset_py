from IPython.display import Image
import os
!ls ../input/
Image("../input/images/corona.png")


import pandas as pd
import numpy as np
import matplotlib.pyplot as py
import seaborn as sb
hostipal_beds = pd.read_csv('../input/uploads/hospital-capacity-by-state-40-population-contracted.csv')
hostipal_beds['bed_to_people_ratio'] = (hostipal_beds['total_hospital_beds'] + hostipal_beds['total_icu_beds'])/(hostipal_beds['adult_population'] + hostipal_beds['population_65'])
cases_by_country = pd.read_csv('../input/uploads/johns-hopkins-covid-19-daily-dashboard-cases-by-country.csv')
cases_by_state = pd.read_csv('../input/uploads/johns-hopkins-covid-19-daily-dashboard-cases-by-states.csv')
cases_over_time = pd.read_csv('../input/uploads/johns-hopkins-covid-19-daily-dashboard-cases-over-time.csv')
cases_over_time = cases_over_time.loc[cases_over_time['country_region'] == 'US']
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


risk = hostipal_beds.sort_values(by=['bed_to_people_ratio'], ascending=False).head(52)
risk['state'] = pd.Series([states[x] for x in risk['state']], index=risk.index)
risk = risk[['state','total_hospital_beds', 'total_icu_beds','adult_population','population_65','bed_to_people_ratio']]
risk.index = risk.state
risk = risk.drop(['state'], axis=1)
s = cases_by_state.loc[cases_by_state['country_region'] == 'US']
s = s.rename(columns={"province_state": "state"})
s.index = s.state
risk['confirmed'] = pd.Series([x for x in s['confirmed']], index=s.index)
risk['deaths'] = pd.Series([x for x in s['deaths']], index=s.index)
risk['infected_ratio'] = risk['confirmed']/(risk['population_65'] + risk['adult_population'])
risk['hypothetical_beds_in_use'] = (risk['total_hospital_beds'] + risk['total_icu_beds']) - risk['confirmed']
risk = risk[['total_hospital_beds', 'total_icu_beds', 'adult_population','population_65', 'confirmed', 'deaths','hypothetical_beds_in_use', 'bed_to_people_ratio', 'infected_ratio']]
risk.head(5)
risk['M_1'] = risk['deaths'] / (risk['adult_population'] + risk['population_65'])
risk['M_2'] = risk['deaths'] / risk['confirmed']

fig, ax =plt.subplots(1,2, figsize=(20, 5))
_ = risk.sort_values(by=['M_1'], ascending=False).head(10)
ax[0].set_title('Ratio of deaths to total population')
sb.barplot(x=_['M_1'], y=_.index, palette='Blues_r',  orient='h', ax=ax[0])
_ = risk.sort_values(by=['M_2'], ascending=False).head(10)
ax[1].set_title('Ratio of deaths to confirmed cases')
sb.barplot(x=_['M_2'], y=_.index, palette='Greens_r',  orient='h', ax=ax[1]);
#Summary of Model
Image("../input/images/Task_6_1.png")
Image("../input/images/Task_6_2.png")
Image("../input/images/Task_6_3.png")
Image("../input/images/Task_6_4.png")
Image("../input/images/Task_6_5.png")
Image("../input/images/Task_6_6.png")
