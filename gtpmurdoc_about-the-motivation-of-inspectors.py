# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print('List of datasets: ',os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore")
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/restaurant-scores-lives-standard.csv')
df.head()
df.info()
# clean data, delete columns
df.drop(columns=['business_city', 'business_state', 'business_postal_code', 'business_location', 'business_address', 'business_phone_number'],inplace=True)
df['inspection_type'].unique()

# note there are many different types of inspections.
# For some reason, the "Foodborne Illness Investigration" might be of high interest ...
# convert time to datetime of pandas
df['inspection_date'] = pd.to_datetime(df['inspection_date'])
df['inspection_month_day'] = df['inspection_date'].map(lambda x: x.strftime('%m-%d'))
df['inspection_year'] = df['inspection_date'].apply(lambda x: x.year)
df['inspection_month'] = df['inspection_date'].apply(lambda x: x.month)
# df['inspection_year_month'] = pd.to_datetime(df['inspection_date'], format='%Y%m')
df['inspection_dayofweek'] = df['inspection_date'].apply(lambda x: x.weekday())
# is there any day of the week where a surprising inspection is more likely to happen
sns.catplot(x="inspection_dayofweek", data=df[df['inspection_type']=='Routine - Unscheduled'], kind="count",  height=4, aspect=1.5);
plt.title('Number of Unscheduled Inspections for week days')
plt.xticks(range(7),['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',' Saturday',' Sunday']);
sns.catplot(x="inspection_year", 
            data=df, 
            kind="count",  height=4, aspect=1.5,sharey=False);
plt.title('Overall number of inspections per year')

sns.catplot(x="inspection_year", 
            col='inspection_type',
            data=df[(df['inspection_type']=='Routine - Scheduled') | (df['inspection_type']=='Routine - Unscheduled')| (df['inspection_type']=='Foodborne Illness Investigation')], 
            kind="count",  height=4, aspect=1,sharey=False);
# is there any time of the year where there are more inspections?
sns.catplot(x="inspection_month",
            row='inspection_type', 
            col='inspection_year', 
            data=df[(df['inspection_type']=='Routine - Scheduled') | (df['inspection_type']=='Routine - Unscheduled')| (df['inspection_type']=='Foodborne Illness Investigation')], 
            kind="count",  height=4, aspect=1.5,sharey=False);
df_tmp = df[(df['inspection_type']=='Routine - Unscheduled')].groupby('inspection_month_day').count()
fig, ax = plt.subplots(figsize=(30,4))
sns.barplot(x=df_tmp.index,
            y='inspection_year',
            data=df_tmp,
            ax=ax, 
            orient='v');
plt.setp(ax.get_xticklabels(), rotation=90);
gb = df.groupby('business_id')
sns.catplot(y="violation_description",
            row='risk_category', 
            data=df, kind="count",  
            height=12, aspect=1.5,
#             hue_order=['Low Risk', 'Moderate Risk', 'High Risk'],
            order = df['violation_description'].value_counts().index,
            orient='v');
sns.boxplot(x = 'risk_category',
            y = 'inspection_score',
            data=df,
            order = ['Low Risk','Moderate Risk', 'High Risk'],
            );
plt.title('Score per risk class');
sns.catplot(y="inspection_score",
            col='risk_category', 
            data=df, kind="count",  
            height=8, aspect=1,sharey=False, 
            orient='v',
           );
# nbr_businesses = df.groupby('business_name')['business_id'].count()
# nbr_businesses.values
# sns.barplot(y = nbr_businesses.index,x = 'business_id',data = nbr_businesses)
fig, ax = plt.subplots(figsize=(30,6))
sns.countplot(ax=ax, 
              y='business_name', 
              data=df, 
              order = df['business_name'].value_counts().iloc[:10].index,
              );
plt.title('Businesses with the most inspections');

# number of inspections to average score
df_unscheduled = df[(df['inspection_type']=='Routine - Unscheduled')]
gb_count = df_unscheduled.groupby('business_id')['business_name'].count()
gb_mean  = df_unscheduled.groupby('business_id')['inspection_score'].mean()

df_id = pd.DataFrame()
df_id['business_id'] = gb_count.index
df_id['inspection_score'] = gb_mean
df_id['inspection_count'] = gb_count

df_id.dropna(inplace=True)
sns.regplot(x='inspection_score', y='inspection_count', data=df_id, scatter_kws={'alpha':0.2});
plt.title('Inspection -  Score vs Count')
df_unscheduled = pd.concat([df_unscheduled, pd.get_dummies(df_unscheduled['risk_category'])], axis=1)
gb_risk = df_unscheduled.groupby('business_id')['Low Risk','Moderate Risk', 'High Risk'].sum()
gb_risk['most_severe_risk'] = gb_risk.apply(lambda x: np.argmax(x), axis=1)

df_concat = pd.concat([df_id,gb_risk['most_severe_risk']], axis=1)
df_concat.dropna(inplace=True)
fig, ax = plt.subplots(figsize=(5,7))
sns.scatterplot(x='inspection_score', 
                y='inspection_count', 
                data=df_concat, 
                alpha=0.7, 
                size='most_severe_risk',
                size_order = ['High Risk','Moderate Risk','Low Risk'],
                legend='full', 
                ax=ax);
plt.title('Inspection -  Score vs Count');
df_formap = pd.concat([df_concat, df_unscheduled[['business_name','business_latitude','business_longitude']]], axis=1)
df_formap.dropna(inplace=True)
import folium

sf_coords = (37.76, -122.45)

#create empty map zoomed in on SF
sf_map = folium.Map(location=sf_coords, zoom_start=13)

risks = {
    'Low Risk'      : 'green', 
    'Moderate Risk' : 'orange', 
    'High Risk'     : 'red', 
    }

def plotAddress(df):
    '''input: series that contains a numeric named latitude and a numeric named longitude
    this function creates a CircleMarker and adds it to your this_map'''
    #print("%s" %(risks[df.risk_category]))
    marker_color = risks[df['most_severe_risk']]
    folium.Marker(location=[df['business_latitude'], df['business_longitude']],
                        popup=df['business_name'],
                        icon=folium.Icon(color=marker_color, icon='circle',prefix='fa'),
                        
                       ).add_to(sf_map)
    
df_formap[df_formap['inspection_score']>98].apply(plotAddress, axis = 1)


display(sf_map)

