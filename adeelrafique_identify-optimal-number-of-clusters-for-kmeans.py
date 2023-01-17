import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans # for KMeans
import matplotlib.pyplot as plt
%matplotlib inline 
ds = pd.read_csv('../input/usarrests/USArrests.csv')
ds.head()
ds.rename(columns={'Unnamed: 0':'State'},inplace=True)
ds.head()
from sklearn.preprocessing import StandardScaler
X = ds[['Murder','Assault','UrbanPop','Rape']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform( X )
print (X_scaled[:5,:])
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X_scaled)
    distortions.append(kmeanModel.inertia_)
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('K - Clusters')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing distortion against K Clusters')
plt.show()
kmeanModel = KMeans(n_clusters=4)
kmeanModel.fit(X_scaled)
ds['Cluster_ID']=kmeanModel.labels_
ds['Cluster_ID']=ds['Cluster_ID'].astype('str')
ds.head()
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
ds['State_Short']=ds['State'].replace(us_state_abbrev)
import plotly.express as px  # Be sure to import express
fig = px.choropleth(ds,  # Input Pandas DataFrame
                    locations='State_Short',  # DataFrame column with locations
                    color="Cluster_ID",  # DataFrame column with color values
                    hover_name="State", # DataFrame column hover info
                    locationmode = 'USA-states') # Set to plot as US States
fig.update_layout(
    title_text = 'US States Crime Arrest Grouping', # Create a Title
    geo_scope='usa',  # Plot only the USA instead of globe
)
print(fig.show())