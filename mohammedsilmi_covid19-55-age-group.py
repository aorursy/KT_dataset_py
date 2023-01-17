
import numpy as np 
import pandas as pd 
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        csvfile=os.path.join(dirname, filename)
cov = pd.read_csv(csvfile,index_col=False)

cov_df=cov.copy()

cov_df= cov_df[['State','Infected','Deaths']]
for i , row in cov_df.iterrows():
    cov_df.loc[i,"D/I ratio"] = row['Deaths'] / row['Infected']
cov_df=cov_df.drop(['Infected', 'Deaths'], axis=1)
cov_df=cov_df.sort_values(by='State', ascending=True)
cov_df.set_index('State', inplace=True)
Age_cov=cov.copy()
Age_cov=Age_cov[['State','Age 55+']]
final_df=pd.merge(Age_cov,cov_df, on="State")



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
states = {state: abbrev for abbrev, state in states.items()}

final_df['State'] = final_df['State'].map(states)

final_df=final_df.sort_values(by='Age 55+', ascending=True)


final_df=final_df.set_index('State')

final_df['D/I ratio'].plot(kind='bar',figsize=(20,8));
final_df['Age 55+'].plot(kind='line',figsize=(20,8));
fig, ax1 = plt.subplots(figsize=(20,11))

final_df=final_df.reset_index() 
    
ax1.set_title('Deaths per 100 Infections (Vs) % of Population over Age 55  ', fontsize=25, color="black")

ax1.set_xlabel('State', fontsize=16)
ax1.set_ylabel('Age 55+', fontsize=16)

ax1=sns.barplot(x='State',y='D/I ratio', data=final_df,palette='ch:3.7,-.2,dark=.5')

ax1.tick_params(axis='y')

ax2 = ax1.twinx()
color = 'tab:blue'

ax2.set_ylabel('Deaths per 100 Infections', fontsize=16)

ax2=sns.lineplot(x='State',y='Age 55+' , data=final_df, sort=False,color=color)
ax2.tick_params(axis='y', color=color)

ax2.legend('Age 55+')

plt.show()
