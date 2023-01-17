import numpy as np 
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns
color = sns.color_palette()
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm
import cufflinks as cf
cf.go_offline()
from sklearn import preprocessing
import missingno as msno # to view missing values
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from PIL import Image
import plotly.figure_factory as ff
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances
from scipy.stats import norm

import squarify
import warnings
warnings.filterwarnings('ignore')
import os
print('Parent Directory:     ', os.listdir("../input"))
print('io directory:         ', os.listdir("../input/io"))
print('usa-cities directory: ', os.listdir("../input/usa-cities"))
teachers = pd.read_csv('../input/io/Teachers.csv')
projects = pd.read_csv('../input/io/Projects.csv', parse_dates=["Project Posted Date","Project Fully Funded Date"])
donations = pd.read_csv('../input/io/Donations.csv', parse_dates=["Donation Received Date"])
donors = pd.read_csv('../input/io/Donors.csv', low_memory=False)
schools = pd.read_csv('../input/io/Schools.csv')
resources = pd.read_csv('../input/io/Resources.csv')


donors.head(3)
donations.head(3)
projects.head(3)
projects.info()
teachers.head(3)
schools.head(3)
resources.head()
print ("Donors row, columns    :", donors.shape)      # (2 122 640, 5)
print ("Donations row, columns :", donations.shape)   # (4 687 884, 7)
print ("Projects row, columns  :", projects.shape)    # (1 110 017, 18)
print ('Teachers row, columns  :', teachers.shape)    # (  402 900, 3)
print ('Schools row, columns   :', schools.shape)     # (  72993, 9)
print ("Resources row, columns :", resources.shape)   # ( 7 210 448, 5)
print(donors.isnull().sum())
msno.matrix(donors)
plt.show()
# how many total missing values do we have?
missing_values_count = donors.isnull().sum()
total_cells = np.product(donors.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
print((total_missing/total_cells) * 100, '% of Missing Values in Donors table')

print(donations.isnull().sum())
print(projects.isnull().sum())
# how many total missing values do we have?
missing_values_count = projects.isnull().sum()
total_cells = np.product(projects.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
print('% of Missing Values in Projects Data:')
print((total_missing/total_cells) * 100, "%")
print('Teacher Missing Data')
print(teachers.isnull().sum())
print(schools.isnull().sum())
# how many total missing values do we have?
missing_values_count = schools.isnull().sum()
total_cells = np.product(schools.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
print('% of Missing Values in Projects Data:')
print((total_missing/total_cells) * 100, "%")
# Merging Data
# Merge donation data with donor data 
donations_donors_merged = donations.merge(donors, left_on='Donor ID',right_on='Donor ID',how='left')
#donors_donations_merged.shape # (4 687 884, 11)
#donors_schools_df = donations_df.merge(donors_df, left_on='Donor ID', right_on='Donor ID')

donations_donors_projects_merged = donations_donors_merged.merge(projects[['Project ID', 'School ID']], 
                                            left_on='Project ID', right_on='Project ID')
donations_donors_projects__schools_merged = donations_donors_projects_merged.merge(schools, left_on='School ID', right_on='School ID')

donations_donors_projects__schools_merged.head(1)
#donations_donors_projects__schools_merged.shape # (4 614 082, 20)
msno.dendrogram(donations_donors_projects__schools_merged)
plt.savefig('merged_data_dendrogram.png')
plt.show()
donors['Donor ID'].count()
donations['Donation Amount'].sum()
print('Maximum Donation Amount: $', donations['Donation Amount'].max())
print('Minimum Donation Amount: $', donations['Donation Amount'].min())
print('Average (Mean) Donation Amount: $', donations['Donation Amount'].mean())
print('Median Donation Amount: $', donations['Donation Amount'].median())
sns.distplot(donations['Donation Amount'], fit=norm)
plt.show()
sns.distplot(donations[(donations['Donation Amount']>0.00)&(donations['Donation Amount']<65.00)]['Donation Amount'], bins = 25, fit=norm)
#sns.distplot(donations[(donations['Donation Amount']>1)]['Donation Amount'], fit=norm) # checking different values, experimental
plt.show()
from scipy.stats import norm
sns.distplot(donations[(donations['Donation Amount']>0.00)&(donations['Donation Amount']<=500)]['Donation Amount'], 
             fit=norm, 
             bins=25)
plt.show()
#skewness and kurtosis
print("Skewness: %f" % donations['Donation Amount'].skew())
print("Kurtosis: %f" % donations['Donation Amount'].kurt())

print('Occurance of Donors/Top 10 Repeating Donors')
donations['Donor ID'].value_counts().head(10)
print('Total Donation Received')
print(donors.shape[0])
repeating_donors=donations['Donor ID'].value_counts().to_frame()
print('Total Number of Repeating Donors')
print(repeating_donors[repeating_donors['Donor ID']>1].shape[0])
print('Total Number of Repeating Donors who donated more than 2 times')
print(repeating_donors[repeating_donors['Donor ID']>2].shape[0])
print('Total Number of Repeating Donors who donated more than 5 times')
print(repeating_donors[repeating_donors['Donor ID']>5].shape[0])
print('Total Number of Repeating Donors who donated more than 10 times')
print(repeating_donors[repeating_donors['Donor ID']>10].shape[0])

print('Second time returning donors %: ')
print((552941/donors['Donor ID'].count())*100, '%')
print('More than 5 times returning donors %: ')
print((98487/donors['Donor ID'].count())*100, '%')
print('More than 10 times returning donors %: ')
print((40299/donors['Donor ID'].count())*100, '%')
temp = donations['Donor Cart Sequence'].value_counts().head(10)
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',
         labels='labels',
         values='values', 
         title='Top Donor checked out carts')
temp = donations['Donation Included Optional Donation'].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })
df.iplot(kind='pie',
         labels='labels',
         values='values', 
         title='Percentage of Optional Donation Included')
donor_is_teacher = donors['Donor Is Teacher'].value_counts()
df = pd.DataFrame({'labels': donor_is_teacher.index,
                   'values': donor_is_teacher.values })
df.iplot(kind='pie',
         labels='labels',
         values='values', 
         title='Not Teacher vs Teacher')
teacher_led_projects = projects["Project Type"].dropna().value_counts()
df = pd.DataFrame({'labels': teacher_led_projects.index,
                   'values': teacher_led_projects.values })
df.iplot(kind='pie',
         labels='labels',
         values='values', 
         title='Teacher Led Projects Percentage')
# Merge donation data with donor data 
#donations_donors_merged = donations.merge(donors, on='Donor ID', how='inner')
state_wise_donation = donations_donors_merged.groupby('Donor State', as_index=False).agg({'Donation ID': 'count','Donation Amount':'sum'})   
state_wise_donation.columns = ["State","Donation_Count", "Donation_Sum"]
state_wise_donation["Donation_Average"] = state_wise_donation["Donation_Sum"]/state_wise_donation["Donation_Count"]
del state_wise_donation['Donation_Count']

for col in state_wise_donation.columns:
    state_wise_donation[col] = state_wise_donation[col].astype(str)
state_wise_donation['text'] = state_wise_donation['State'] + '<br>' +\
    'Average amount per donation: $' + state_wise_donation['Donation_Average']+ '<br>' +\
    'Total donation amount:  $' + state_wise_donation['Donation_Sum']
state_codes = {
    'District of Columbia' : 'DC','Mississippi': 'MS', 'Oklahoma': 'OK', 
    'Delaware': 'DE', 'Minnesota': 'MN', 'Illinois': 'IL', 'Arkansas': 'AR', 
    'New Mexico': 'NM', 'Indiana': 'IN', 'Maryland': 'MD', 'Louisiana': 'LA', 
    'Idaho': 'ID', 'Wyoming': 'WY', 'Tennessee': 'TN', 'Arizona': 'AZ', 
    'Iowa': 'IA', 'Michigan': 'MI', 'Kansas': 'KS', 'Utah': 'UT', 
    'Virginia': 'VA', 'Oregon': 'OR', 'Connecticut': 'CT', 'Montana': 'MT', 
    'California': 'CA', 'Massachusetts': 'MA', 'West Virginia': 'WV', 
    'South Carolina': 'SC', 'New Hampshire': 'NH', 'Wisconsin': 'WI',
    'Vermont': 'VT', 'Georgia': 'GA', 'North Dakota': 'ND', 
    'Pennsylvania': 'PA', 'Florida': 'FL', 'Alaska': 'AK', 'Kentucky': 'KY', 
    'Hawaii': 'HI', 'Nebraska': 'NE', 'Missouri': 'MO', 'Ohio': 'OH', 
    'Alabama': 'AL', 'Rhode Island': 'RI', 'South Dakota': 'SD', 
    'Colorado': 'CO', 'New Jersey': 'NJ', 'Washington': 'WA', 
    'North Carolina': 'NC', 'New York': 'NY', 'Texas': 'TX', 
    'Nevada': 'NV', 'Maine': 'ME', 'other': ''}

state_wise_donation['code'] = state_wise_donation['State'].map(state_codes)  
colorscale = [[0,"rgb(5, 10, 172)"],[0.35,"rgb(40, 60, 190)"],[0.5,"rgb(70, 100, 245)"],\
            [0.6,"rgb(90, 120, 245)"],[0.7,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]]
data = [ dict(
        type='choropleth',
        autocolorscale = False,
        colorscale = colorscale,
        reversescale = True,
        locations = state_wise_donation['code'], # state
        z = state_wise_donation['Donation_Sum'].astype(float), # map colors
        locationmode = 'USA-states', 
        text = state_wise_donation['text'], # mouse hovers text
        colorbar = dict(  
            title = "Donation in USD")  
        ) ]

layout = dict(
        title = 'Donations Distribution by States<br>(Hover over for details)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
fig = dict(data=data, layout=layout)
iplot(fig)
donor_state = donors['Donor State'].value_counts().head(10).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(19,9))
sns.barplot(donor_state.values, donor_state.index, ax=ax)
ax.set(xlabel= 'Number of Donors', 
       ylabel = 'State', 
       title = "Distribution of Donors by State")
plt.show()
donations_donors_merged['Donation Received Date'] = pd.to_datetime(donations_donors_merged['Donation Received Date'])
donations_donors_merged['year'] = donations_donors_merged['Donation Received Date'].dt.year
temp_df = donations_donors_merged[~donations_donors_merged.year.isin([2018])].sort_values('year') 
fig = {
    'data': [
        {
            'x': temp_df[temp_df['Donor State']==state].groupby('year').agg({'Donor ID' : 'count'}).reset_index()['year'],
            'y': temp_df[temp_df['Donor State']==state].groupby('year').agg({'Donor ID' : 'count'}).reset_index()['Donor ID'],
            'name': state, 'mode': 'line',
        } for state in ['California', 'New York', 'Texas', 'Florida', 'Illinois', 'North Carolina', 'other', 'Pennsylvania','Georgia', 'Massachusetts']
    ],
    'layout': {
        'title' : 'Donors Trends in Top 10 States (2012 - 2017)',
        'xaxis': {'title': 'Year'},
        'yaxis': {'title': "Number of Donors"}
    }
}
py.iplot(fig, filename='donor_trends_states')
fig = {
    'data': [
        {
            'x': temp_df[temp_df['Donor State']==state].groupby('year').agg({'Donation Amount' : 'sum'}).reset_index()['year'],
            'y': temp_df[temp_df['Donor State']==state].groupby('year').agg({'Donation Amount' : 'sum'}).reset_index()['Donation Amount'],
            'name': state, 'mode': 'line',
        } for state in ['California', 'New York', 'Texas', 'Florida', 'Illinois', 'North Carolina', 'other', 'Pennsylvania','Georgia', 'Massachusetts']
    ],
    'layout': {
        'title' : 'Donation Trend in Top 10 States (2012 - 2017)',
        'xaxis': {'title': 'Year'},
        'yaxis': {'title': "Donation Amount ($)"}
    }
}
py.iplot(fig, filename='states')
donor_cities = donors['Donor City'].value_counts().head(10).sort_values(ascending=True)
donor_cities.iplot(kind='barh', 
                   xTitle = 'Distribution of Donor Cities', 
                   yTitle = "Count", 
                   title = 'Distribution of Donor Cities', 
                   color='green')
fig = {
    'data': [
        {
            'x': temp_df[temp_df['Donor City']==city].groupby('year').agg({'Donor ID' : 'count'}).reset_index()['year'],
            'y': temp_df[temp_df['Donor City']==city].groupby('year').agg({'Donor ID' : 'count'}).reset_index()['Donor ID'],
            'name': city, 'mode': 'line',
        } for city in ['Chicago','New York','Brooklyn','Los Angeles','San Francisco','Houston','Indianapolis','Portland','Philadelphia','Seattle']
    ],
    'layout': {
        'title' : 'Donors Trends in Top 10 Cities (2012 - 2017)',
        'xaxis': {'title': 'Year'},
        'yaxis': {'title': "Number of Donors"}
    }
}
py.iplot(fig, filename='Cities')
import folium
import folium.plugins
from folium import IFrame
cities=pd.read_csv('../input/usa-cities/usa_cities.csv')
city_don=donations_donors_merged.groupby('Donor City')['Donation Amount'].sum().to_frame()
city_num=donors['Donor City'].value_counts().to_frame()
city_don=city_don.merge(city_num,left_index=True,right_index=True,how='left')
city_don.columns=[['Amount','Donors']]
map_cities=cities[['city','lat','lng']].merge(city_don,left_on='city',right_index=True)
map_cities.columns=[['City','lat','lon','Amount','Donors']]
map2 = folium.Map(location=[39.50, -98.35],tiles='Mapbox Control Room',zoom_start=3.5)
locate=map_cities[['lat','lon']]
count=map_cities['Donors']
city=map_cities['City']
amt=map_cities['Amount']
def color_producer(donors):
    if donors < 90:
        return 'orange'
    else:
        return 'green'
for point in map_cities.index:
    info='<b>City: </b>'+str(city.loc[point].values[0])+'<br><b>No of Donors: </b>'+str(count.loc[point].values[0])+'<br><b>Total Funds Donated: </b>'+str(amt.loc[point].values[0])+' <b>$<br>'
    iframe = folium.IFrame(html=info, width=250, height=250)
    folium.CircleMarker(list(locate.loc[point]),
                        popup=folium.Popup(iframe),
                        radius=amt.loc[point].values[0]*0.000005,
                        color=color_producer(count.loc[point].values[0]),
                        fill_color=color_producer(count.loc[point].values[0]),fill=True).add_to(map2)
map2
# Extract Year, Month, Date, Time
donations["Donation Received Year"]   = donations["Donation Received Date"].dt.year
donations["Donation Received Month"]  = donations["Donation Received Date"].dt.month
donations["Donation Received Day"]    = donations["Donation Received Date"].dt.day
donations["Donation Received Hour"]   = donations["Donation Received Date"].dt.hour
temp = donations["Donation Received Year"].value_counts().head()
temp.iplot(kind='bar', 
           xTitle = 'Year', 
           yTitle = "Number of Donations", 
           title = 'No. Donations Received by Year',
           color = '#47FE03')
projects['Project Current Status'].describe()
projects['Project Current Status'].unique()
project_status_percentage = projects['Project Current Status'].value_counts()
project_status_pie_chart = pd.DataFrame({'labels': project_status_percentage.index,'values': project_status_percentage.values})
labels = project_status_pie_chart['labels'].tolist()
values = project_status_pie_chart['values'].tolist()
colors = ['#1A9365', '#D62828','#F77F00','#ED6A5A']
trace = go.Pie(labels=labels, 
               values=values,
               hoverinfo='label+value', 
               textinfo='percent',
               textfont=dict(size=19, color='#EFEFEF'),
               marker=dict(colors=colors,
                           line=dict(color='#E2E2E2', width=1)))

py.iplot([trace], filename='project_status_pie_chart')
f,ax=plt.subplots(1,2,figsize=(25,8))
sns.barplot(ax=ax[0],
            x=projects['Project Posted Date'].dt.year.value_counts().index,
            y=projects['Project Posted Date'].dt.year.value_counts().values,
            palette=sns.color_palette('BuGn_r',20))
ax[0].set_title('Yearly Project Posted Trend')
sns.barplot(ax=ax[1],
            x=projects['Project Posted Date'].dt.month.value_counts().index,
            y=projects['Project Posted Date'].dt.month.value_counts().values,
            palette=sns.color_palette("cubehelix", 35))
ax[1].set_title('Monthly Project Posted Trend')
plt.show()
print('Occurance of Successfull Projects')
print(projects['Project Fully Funded Date'].dropna().value_counts().head())
print('Total Missing Values in Date Column')
print(projects['Project Fully Funded Date'].isnull().sum().sum())
fully_funded = projects[projects['Project Current Status'] == "Fully Funded"]
temp = fully_funded['Project Fully Funded Date'].dropna().value_counts()
temp.iplot(kind='bar', 
                               xTitle = 'Year & Month', 
                               yTitle = "Count", 
                               title = 'Successful Projects by Year & Month',
                               color = 'green')
projects['Funded Year'] = projects['Project Fully Funded Date'].dt.year
projects['Funded Month'] = projects['Project Fully Funded Date'].dt.month
projects['Funded Week'] = projects['Project Fully Funded Date'].dt.week
projects['Funded Day'] = projects['Project Fully Funded Date'].dt.day

temp = projects['Funded Year'].value_counts().head()
temp.iplot(kind='bar', 
           xTitle = 'Year & Month', 
           yTitle = "Count", 
           title = 'Successful Projects by Year',
           color = '#02005D')
temp = projects['Funded Month'].value_counts()
temp.iplot(kind='bar', 
           xTitle = 'Month', 
           yTitle = "Count", 
           title = 'Successful Projects by Month',
           color = '#3EFF02')
temp = projects['Funded Week'].value_counts()
temp.iplot(kind='bar', 
           xTitle = 'Week', 
           yTitle = "Count", 
           title = 'Successful Projects by Week',
           color = '#D8470E')
temp = projects['Funded Day'].value_counts()
temp.iplot(kind='bar', 
           xTitle = 'Week', 
           yTitle = "Count", 
           title = 'Successful Projects by Day',
           color = '#EABB25')
successful_projects = projects[projects['Project Current Status'] == "Fully Funded"]
not_successful_projects = projects[(projects['Project Current Status']!='Live')&(projects['Project Current Status']!='Fully Funded')]

print('Average Cost of Successful(Fully Funded) Projects:', successful_projects['Project Cost'].mean(), '$')
print('Average Cost of Not-Successful (Not Fully Funded) Projects:', not_successful_projects['Project Cost'].mean(),'$')

print('Median Cost of Successful(Fully Funded) Projects:',successful_projects['Project Cost'].median(),'$')
print('Median Cost of Not-Successful (Not Fully Funded) Projects', not_successful_projects['Project Cost'].median(),'$')
project_subject_category = projects['Project Title'].value_counts().head(25)
project_subject_category.iplot(kind='bar', 
                               xTitle = 'Project Title', 
                               yTitle = "Count", 
                               title = 'Top Project Titles',
                               color = 'green')
successful_projects = projects[projects['Project Current Status'] == "Fully Funded"]
successful_project_titles = successful_projects['Project Title'].value_counts().head(19)
successful_project_titles.iplot(kind='bar', 
                               xTitle = 'Project Title', 
                               yTitle = "Count", 
                               title = 'Top Successful Project Titles',
                               color = 'red')
project_subject_category = projects['Project Subject Category Tree'].value_counts().head(12)
project_subject_category.iplot(kind='bar', 
                               xTitle = 'Project Subject Category', 
                               yTitle = "Count", 
                               title = 'Top Project Subject Categories',
                               color = 'blue')
project_subject_subcategory = projects['Project Subject Subcategory Tree'].value_counts().head(19)
project_subject_subcategory.iplot(kind='bar', 
                               xTitle = 'Project Subject Subcategory', 
                               yTitle = "Count", 
                               title = 'Top Project Subject Subcategories',
                               color = 'yellow')
from wordcloud import ImageColorGenerator

mask_img = np.array(Image.open('../input/donor-choose-logo/donorchoose-logo2.png'))

stopwords = set(STOPWORDS)
filtered_stopwords = ["teacher", "title", "donor", "students", "school", "will", "donotremoveeassydivider", "DONOTREMOVEESSAYDIVIDER"]
stopwords.update(filtered_stopwords)

mask_img = np.array(Image.open('../input/donor-choose-logo/donorchoose-logo2.png'))
image_colors = ImageColorGenerator(mask_img)
wordcloud = WordCloud(
                          background_color='#fefefe',
                          mask = mask_img,
                          stopwords=stopwords,
                          max_words=250,
                          max_font_size=50, 
                          #width=1500, 
                          #height=500,
                          random_state=42,
                         ).generate(" ".join(projects["Project Essay"][~pd.isnull(projects["Project Essay"])].sample(10000)))

fig = plt.figure(figsize = (19,15))
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear")
plt.title("Project Essay Word Cloud ", fontsize=33)
plt.axis('off')
plt.savefig('wordcloud.png')         
plt.show()
successful_projects = projects[projects['Project Current Status'] == "Fully Funded"]
temp = successful_projects['Project Grade Level Category'].sort_values(ascending=True).value_counts()
temp.iplot(kind='bar', 
           xTitle = 'Project Grade Level Category', 
           yTitle = "Count", 
           title = 'Project Grade Level Category',
           color = '#0A9B81')
plt.figure(figsize=(19,8))
ax=schools['School State'].value_counts()[:10].plot.barh(width=0.8,color=sns.color_palette('cool', 12))
plt.gca().invert_yaxis()
for i, v in enumerate(schools['School State'].value_counts()[:10].values): 
    ax.text(.9, i, v,fontsize=12,color='black',weight='normal')
plt.title('Top 10 States with Highest Number of Schools')
plt.show()
# Color Pallets https://matplotlib.org/users/colormaps.html
# Limit data to include only those states that contain both donors and schools
# Learning it from James Shepherd https://www.kaggle.com/shep312/donorschoose-matching-donors-to-causes/
school_states = donations_donors_projects__schools_merged['School State'].unique()
donor_states = donations_donors_projects__schools_merged['Donor State'].unique()

states_to_keep_mask = [x in school_states for x in donor_states]
states = donor_states[states_to_keep_mask]
donations_donors_projects__schools_merged = donations_donors_projects__schools_merged[
    donations_donors_projects__schools_merged['School State'].isin(states)]
donations_donors_projects__schools_merged = donations_donors_projects__schools_merged[
    donations_donors_projects__schools_merged['Donor State'].isin(states)]
# donations_donors_projects__schools_merged.shape # (4 423 786, 20)

donor_to_school_total_donation_statewise = donations_donors_projects__schools_merged.pivot_table(columns='School State',
                                      index='Donor State', 
                                      values='Donation Amount', 
                                      aggfunc='sum',
                                      fill_value=0)
donor_to_school_total_donation_statewise.head()

# Take top donor states from this merged table
top_donor_states = donations_donors_projects__schools_merged.groupby('Donor State')['Donation Amount'].sum().sort_values(ascending=False)
#top_donor_states.drop('other', inplace=True)  
top_donor_states = top_donor_states[:10]

# Separate the top n donors
top_n_donors_destinations = donor_to_school_total_donation_statewise.loc[top_donor_states.index, :]

# Remove any states that none of them donate too
top_n_donors_destinations = top_n_donors_destinations.loc[:, top_n_donors_destinations.sum() > 0]

# Unpivot
donation_paths = top_n_donors_destinations.reset_index().melt(id_vars='Donor State')
donation_paths = donation_paths[donation_paths['value'] > 250000]  # Only significant amounts

# Encode state names to integers for the Sankey
donor_encoder, school_encoder = LabelEncoder(), LabelEncoder()
donation_paths['Encoded Donor State'] = donor_encoder.fit_transform(donation_paths['Donor State'])
donation_paths['Encoded School State'] = school_encoder.fit_transform(donation_paths['School State'])\
    + len(donation_paths['Encoded Donor State'].unique())
# Create a state to color dictionary
all_states = np.unique(np.array(donation_paths['School State'].unique().tolist() + donation_paths['Donor State'].unique().tolist()))
plotly_colors = ['#8424E0', '#FD28DC', '#B728FE', '#288DFF', '#00E2FB', '#ADE601', '#FCFF00', '#FEA128', '#FE0000', '#FF5203']

states_finished = False
state_colors = []
i = 0
while not states_finished:
    
    state_colors.append(plotly_colors[i]) 
    
    if len(state_colors) >= len(all_states):
        states_finished = True
        
    i += 1
    if i >= len(plotly_colors):
        i = 0
        
color_dict = dict(zip(all_states, state_colors))

sankey_labels = donor_encoder.classes_.tolist()  + school_encoder.classes_.tolist()
colors = []
for state in sankey_labels:
    colors.append(color_dict[state])

data = dict(
    type='sankey',
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(
        color = "black",
        width = 0.5
      ),
      label = sankey_labels,
      color = colors,
    ),
    link = dict(
      source = donation_paths['Encoded Donor State'],
      target = donation_paths['Encoded School State'],
      value = donation_paths['value'],
  ))

layout =  dict(
    title = "Donation Source vs Destination(Hover over to see values)",
    autosize=False,
    width=800,
    height=750,

    font = dict(
      size = 10
    )
)

fig = dict(data=[data], layout=layout)
py.iplot(fig, filename='source_destination_match.', validate=False)
school_metro_type = schools['School Metro Type'].value_counts().drop('unknown')
fig = plt.figure(figsize=(8,8))
plt.pie(school_metro_type.values, 
        labels = school_metro_type.index, 
        autopct='%1.1f%%')

empty_circle = plt.Circle((0,0), 0.7, color='white')
p = plt.gcf()
p.gca().add_artist(empty_circle)
plt.title('School Metro Type', fontsize=22)
plt.show()
projects_resources = projects.merge(resources)

successful_projects_resources = projects_resources[projects_resources["Project Current Status"] == "Fully Funded"]
successful_projects_resources['Resource Item Name'].fillna('Unknown')
successful_projects_resources.dropna()
temp = successful_projects_resources['Resource Item Name'].sort_values(ascending=True).value_counts().head(10)
temp.iplot(kind='barh', 
           xTitle = 'Project Resource Item Name', 
           yTitle = "Count", 
           title = 'Successful Projects\' Top Resource Items',
           color = '#0A9B81')