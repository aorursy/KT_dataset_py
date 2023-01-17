
# Library for data analysis and data structure with high performance and also easy to use.
import pandas as pd 

# Seaborn library for data visualization, for creating sophisticated graphics
import seaborn as sns 

# And we will use seaborn so that every time we print the graphs we can have the grid lines
# and a white background for a better understanding of the information that will be presented.
sns.set(style = 'whitegrid') 

# To ignore possible warnings
import warnings
warnings.filterwarnings('ignore')

# And in addition we will also call the matplotlib set of functions, pyplot allows you to create figures and graphs
import matplotlib.pyplot as plt 

# And lastly, if we want to visualize here on the jupyter notebook the graphics that we are going to create we will need this method
%matplotlib inline 
# Loading the dataset
df = pd.read_csv('../input/all-space-missions-from-1957/Space_Corrected.csv')
# Viewing the first 5 lines of the dataset
df.head()
# Selecting the dataset columns
df.columns
# Selecting the columns we will use
df = df[['Company Name', 'Location', 'Datum', 'Detail', 'Status Rocket', ' Rocket', 'Status Mission']]

# Renaming the 'Rocket' column to 'Cost_Mission'
df.rename(columns = {
    ' Rocket':'Cost_Mission',
    'Datum':'Date_Time'
}, inplace = True)
# Add the 'Year' and 'Country' columns

# Add the 'Year' column
df['Year'] = df.Date_Time.str.split(' ').str[3]

# Add the 'Country' column
df['Country'] = df.Location.str.split(',').str[-1]

# Correcting the names of some countries that are wrong
replace_countries = {
    'Russia' : 'Russian Federation',
    "Shahrud Missile Test Site": "Iran",
    'New Mexico' : 'USA',
    "Yellow Sea": 'China',
    "Barents Sea": 'Russian Federation',
    "Pacific Missile Range Facility": 'USA',
    "Gran Canaria": 'USA'
}
df['Country'] = df['Country'].replace(replace_countries)

# Add the 'Month' column
df['Month'] = df.Date_Time.str.split(' ').str[1]
# Viewing data types
df.dtypes
# Replacing the characters in the 'Cost_Mission' column
df.Cost_Mission = df.Cost_Mission.str.replace(',', '')

# Change the data type of the 'Cost_Mission' column
df.Cost_Mission = df.Cost_Mission.astype(float)

# Change the format of the 'Date_Time' column
df.Date_Time = pd.to_datetime(df['Date_Time']).apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
# Viewing the first 3 lines of the dataset
df.head(3)
# Replacing null values by 0
df.fillna(0, inplace = True)
# Viewing the first 3 lines of the dataset
df.head(3)
# Temos três tipos de status: 'Success', 'Failure', 'Prelaunch Failure' e 'Partial Failure'

# Visualizando a quantidade de missões por status
pd.DataFrame(df['Status Mission'].value_counts())
# Setting the size of the chart figure
fig = plt.subplots(figsize = (14, 7))

# Creating the chart
g = sns.barplot(
            x = df['Status Mission'].value_counts().index, 
            y = df['Status Mission'].value_counts().values, 
            data = df,
            palette = 'rocket')

# Chart title
plt.title('Total Missions by Status Mission', fontdict = {'size':18})

# Axis subtitles (x & y)
plt.xlabel('Status Mission', fontdict = {'size':16})
plt.ylabel('Amount of Missions', fontdict = {'size':16})

# Add labels to chart bars
for x in g.patches:
    g.annotate((int(x.get_height())), 
    (x.get_x()+0.35, 
    x.get_height()+20))
# Distribution of Success Missions based on Launch Year

# Filtering successful missions
mission_Success = df[df['Status Mission'] == 'Success']

# Setting the chart size
fig, x = plt.subplots(figsize=(20, 6.5))
g = sns.histplot(
    x = mission_Success.Year.sort_values(), 
    data = mission_Success, 
    color='#31a354', 
    kde = True
)

# Chart title
g.set_title('Distribution of Success Missions based on Launch Year', fontdict = {'size':18})

# Customizing the axes (x and y)
g.xaxis.set_label_text('Year', fontdict = {'size':15})
g.yaxis.set_label_text('Amount of Missions', fontdict = {'size':15})

# Customizing the chart labels
labels = mission_Success.Year.sort_values().unique()
g.set_xticklabels(labels, rotation = 45, ha = 'center', fontdict = {'size':12})
plt.show();
# Creating the chart
g = sns.catplot(
            x = mission_Success['Company Name'].value_counts().values,
            y = mission_Success['Company Name'].value_counts().index,
            data = mission_Success,
            kind = 'bar',
            aspect = 1.5,
            height = 8,
            palette = 'YlGnBu_r'
)

# Chart title
plt.title('Amount of Successful Launches per Company', fontdict = {'size':18})

# Define the axes (x & y)
plt.xlabel('Amount per Launch', fontdict = {'size':16})
plt.ylabel('Company Name', fontdict = {'size':16})

# Plotting the chart
plt.show()
# Create the chart
g = sns.catplot(x = mission_Success.Country.value_counts().index,
            y = mission_Success.Country.value_counts().values,
            data = mission_Success,
            kind = 'bar',
            aspect = 1.5,
            height = 8,
            palette = 'YlGn_r'
)

# Chart title
plt.title('Amount of Successful Launches per Country', fontdict = {'size':18})

# Definindo os eixos (x & y)
plt.xlabel('Amount per Launches', fontdict = {'size':16})
plt.ylabel('Country', fontdict = {'size':16})

# Customizing the axis labels
g.set_xticklabels(rotation = 45, ha = 'right')

for i in g.ax.patches:
    g.ax.annotate((int(i.get_height())), 
    (i.get_x()+0.16, 
    i.get_height()+20))

# Plotting the chart
plt.show()
# Average cost per mission

# Defining the variable to make the pivot
var = pd.DataFrame()

# Selecting only the cost of missions greater than zero
df_Cost_Mission = df[df['Cost_Mission'] > 0]

# Selecting the data and applying the average
for b in list(df_Cost_Mission.Month.unique()):
    for v in list(df_Cost_Mission['Status Mission'].unique()):
        z = df_Cost_Mission[(df_Cost_Mission.Month == b) & (df_Cost_Mission['Status Mission'] == v)]['Cost_Mission'].mean()
        var = var.append(pd.DataFrame({'Month':b , 'Status Mission':v , 'avgCost_Mission':z}, index=[0]))

# Resetting the index of our new dataset  
var = var.reset_index(drop = True)

# Filtering values and formats
var['avgCost_Mission'].fillna(0,inplace=True)
var['avgCost_Mission'].isnull().value_counts()
var['avgCost_Mission'] = var['avgCost_Mission'].astype(int)

# Viewing the first 5 lines of the new dataset
pd.DataFrame(var.head())
# Selecting the columns of our new dataset
tri = var.pivot('Month', 'Status Mission', 'avgCost_Mission')

# Creating the figure for the chart
fig, ax = plt.subplots(figsize=(15, 12.5))

# Creating the chart
sns.heatmap(tri, linewidths=1, cmap='YlGnBu', annot=True, ax=ax, fmt='d')

# Defining the chart title
ax.set_title('Heatmap - Average cost of a mission per month, as well as the mission status', fontdict={'size':20})

# Add title on the x and y axes
ax.xaxis.set_label_text('Status Missão',fontdict= {'size':20})
ax.yaxis.set_label_text('Month',fontdict= {'size':20})

# Plot the chart
plt.show()
# Chart size
fig = plt.subplots(figsize = (14, 5))

# Creating the chart
g = sns.barplot(
    order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    x = df.Month.value_counts().index,
    y = df.Month.value_counts().values,
    data = df,
    color = '#1c9099'
)

# Chart title
plt.title('Total Cost of Missions per Month', fontdict = {'size':18})

# Axis labels (x & y)
plt.xlabel('Month', fontdict = {'size':16})
plt.ylabel('Amount of Missions per Month', fontdict = {'size':16})

# Plot
plt.show()
# Group the columns 
cost_mission_company = df.groupby(['Company Name'])['Cost_Mission'].sum().reset_index()

# Mission cost > zero
cost_mission_company = cost_mission_company[cost_mission_company['Cost_Mission'] > 0]

# Create the chart
g = sns.catplot(
    x = 'Company Name',
    y = 'Cost_Mission',
    data = cost_mission_company,
    kind = 'bar',
    aspect = 2.2
)

# Graphic title
g.ax.set_title('Mission cost per Company', fontdict = {'size':18})

# Customizing the axes (x and y)
plt.xlabel('Company Name', fontdict = {'size':16})
plt.ylabel('Cost Mission', fontdict = {'size':16})

# Customizing the x-axis labels
g.set_xticklabels(rotation = 45, ha = 'right');
# Select the two countries, 'Russia' and 'USA'
russia_usa = df[df.Country.isin([' Russia', ' USA'])]

# Select all the successful missions
russia_usa_success = russia_usa[russia_usa['Status Mission'] == 'Success']

# Creating the graph figure
fig = plt.subplots(figsize = (18, 5))

# Creating the chart
g = sns.histplot(
    russia_usa_success,
    x = 'Year', hue = 'Country',
    palette = 'BuGn',
    multiple = 'stack',
    kde = True
);

# Chart title
plt.title('Russia vs USA - Successful Missions per Year', fontdict = {'size':18})

# Eixos x & y
plt.xlabel('Year', fontdict = {'size':16})
plt.ylabel('Amount of Successful Missions', fontdict = {'size':16})

# Customize the x-axis labels
labels = russia_usa_success.Year.sort_values().unique()
g.set_xticklabels(labels, rotation = 45)

# Print the chart
plt.show()
# Select all missions that were unsuccessful
russia_usa_failure = russia_usa[russia_usa['Status Mission'] != 'Success']

# Create the graph figure
fig = plt.subplots(figsize = (18, 5))

# Create the chart
g = sns.histplot(
    russia_usa_failure,
    x = 'Year', hue = 'Country',
    palette = 'YlOrRd',
    multiple = 'stack',
    kde = True
);

# Chart title
plt.title('Russia vs USA - Unsuccessful Missions per Year', fontdict = {'size':18})

# X & y axes
plt.xlabel('Year', fontdict = {'size':16})
plt.ylabel('Amount of Unsuccessful Missions', fontdict = {'size':16})

# Customize the x-axis labels
labels = russia_usa_success.Year.sort_values().unique()
g.set_xticklabels(labels, rotation = 45)

# Print the chart
plt.show()
