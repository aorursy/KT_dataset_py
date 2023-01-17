# math operations
from numpy import inf

# time operations
from datetime import timedelta

# for numerical analyiss
import numpy as np

# to store and process data in dataframe
import pandas as pd

# basic visualization package
import matplotlib.pyplot as plt

# advanced ploting
import seaborn as sns

# interactive visualization
import plotly.express as px
import plotly.graph_objs as go
# import plotly.figure_factory as ff
#from plotly.subplots import make_subplots

# for offline ploting
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)

# hide warnings
import warnings
warnings.filterwarnings('ignore')
# color pallette
cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 
# list files
# ==========

!ls ../input/corona-virus-report
# Full data
# =========

full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')

# Deep dive into the DataFrame
# Examine DataFrame (object type, shape, columns, dtypes)
full_table.info()

# type(full_table)
# full_table.shape
# full_table.columns
# full_table.dtypes
# full_table.head(20)
# Country wise
# ============

country_wise = pd.read_csv('../input/corona-virus-report/country_wise_latest.csv')

# Replace missing values '' with NAN and then 0
country_wise = country_wise.replace('', np.nan).fillna(0)

# Deep dive into the DataFrame
country_wise.info()
country_wise.head(10)

# Grouped by day, country
# =======================

full_grouped = pd.read_csv('../input/corona-virus-report/full_grouped.csv')
full_grouped.info()
full_grouped.head(10)

# Convert Date from Dtype "Object" (or String) to Dtype "Datetime"
full_grouped['Date'] = pd.to_datetime(full_grouped['Date'])
full_grouped.info()
# Day wise
# ========

day_wise = pd.read_csv('../input/corona-virus-report/day_wise.csv')
day_wise['Date'] = pd.to_datetime(day_wise['Date'])
day_wise.info()
day_wise.head(10)
# Worldometer data
# ================

worldometer_data = pd.read_csv('../input/corona-virus-report/worldometer_data.csv')

# Replace missing values '' with NAN and then 0
# What are the alternatives? Drop or impute. Do they make sense in this context?
worldometer_data = worldometer_data.replace('', np.nan).fillna(0)
worldometer_data['Case Positivity'] = round(worldometer_data['TotalCases']/worldometer_data['TotalTests'],2)
worldometer_data['Case Fatality'] = round(worldometer_data['TotalDeaths']/worldometer_data['TotalCases'],2)

# Case Positivity is infinity when there is zero TotalTests due to division by zero
worldometer_data[worldometer_data["Case Positivity"] == inf] = 0

# Qcut is quantile cut. Here we specify three equally sized bins and label them low, medium, and high, respectively.
worldometer_data ['Case Positivity Bin']= pd.qcut(worldometer_data['Case Positivity'], q=3, labels=["low", "medium", "high"])

# Population Structure
worldometer_pop_struc = pd.read_csv('../input/covid19-worldometer-snapshots-since-april-18/population_structure_by_age_per_contry.csv')

# Replace missing values with zeros
worldometer_pop_struc = worldometer_pop_struc.fillna(0)
#worldometer_pop_struc.info()

# Merge worldometer_data with worldometer_pop_struc
# Inner means keep only common key values in both datasets
worldometer_data = worldometer_data.merge(worldometer_pop_struc,how='inner',left_on='Country/Region', right_on='Country')

# Keep observations where column "Country/Region" is not 0
worldometer_data = worldometer_data[worldometer_data["Country/Region"] != 0]

# Inspect worldometer_data's metadata
worldometer_data.info()

# Inspect Data
# worldometer_data.info()
# worldometer_data.tail(20)
# worldometer_data["Case Positivity"].describe()

# The three arguments of the function are: 
# 1. input dataframe 
# 2. variable of interest 
# 3. color palette
def plot_map(df, col, pal):
    df = df[df[col]>0]
    
    # choropleth draws maps based on shape files of different countries
    fig = px.choropleth(df, locations="Country/Region", locationmode='country names', 
                  color=col, hover_name="Country/Region", 
                  title=col, hover_data=[col], color_continuous_scale=pal)
    # 1:50m resolution
    fig.update_geos (resolution=50)
    fig.show()
# Inspect dataframe worldometer_data
worldometer_data.info()
# Plot 'TotalCases' on world map using worldometer_data as the input dataframe with color palette 'matter'
plot_map(worldometer_data, 'TotalCases', 'matter')
# What do you read this?
plot_map(worldometer_data, 'Tot Cases/1M pop', 'matter')
# Plot log(Confirmed Cases) for each Country/Region Over time
fig = px.choropleth(full_grouped, locations="Country/Region", 
                    color=np.log(full_grouped["Confirmed"]),
                    locationmode='country names', hover_name="Country/Region", 
                    animation_frame=full_grouped["Date"].dt.strftime('%Y-%m-%d'),
                    title='Cases over time', color_continuous_scale=px.colors.sequential.matter)
fig.update(layout_coloraxis_showscale=False)
fig.update_geos (resolution=50)
fig.show()
full_grouped_Brazil=full_grouped.groupby('Country/Region').get_group('Brazil')
temp = full_grouped_Brazil.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()
temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],
                 var_name='Case', value_name='Count')
temp.head()

# Plot a stack area graph with the three types of cases (i.e., recovered, deaths, and active)
fig = px.area(temp, x="Date", y="Count", color='Case', height=600, width=700,
             title='Cases over time', color_discrete_sequence = [rec, dth, act])
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()
full_grouped_India=full_grouped.groupby('Country/Region').get_group('India')
temp = full_grouped_India.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()
temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],
                 var_name='Case', value_name='Count')
temp.head()

# Plot a stack area graph with the three types of cases (i.e., recovered, deaths, and active)
fig = px.area(temp, x="Date", y="Count", color='Case', height=600, width=700,
             title='Cases over time', color_discrete_sequence = [rec, dth, act])
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()
def gt_n(n):
    # Identify countries with confirmed cases greater than N
    # Then among these countries choose the unique set of countries
    countries = full_grouped[full_grouped['Confirmed']>n]['Country/Region'].unique()
    
    # Filter countries that are in the unique set of countries with confirmed cases greater than N
    temp = full_table[full_table['Country/Region'].isin(countries)]
    
    # Aggregate (i.e., sum up) confirmed cases by Country/Region and Date
    # Reset the index (it is no longer in running order)
    temp = temp.groupby(['Country/Region', 'Date'])['Confirmed'].sum().reset_index()
    
    # Filter observations with confirmed cases more than N
    temp = temp[temp['Confirmed']>n]
    # print(temp.head())

    # Identify the start date when confirmed cases exceed N for each country
    min_date = temp.groupby('Country/Region')['Date'].min().reset_index()
    
    # Name the columns in the dataframe min_date
    min_date.columns = ['Country/Region', 'Min Date']
    # print(min_date.head())

    # Merge dataframe temp with dataframe min_date by 'Country/Region'
    from_nth_case = pd.merge(temp, min_date, on='Country/Region')
    
    # Convert data type to datetime object
    from_nth_case['Date'] = pd.to_datetime(from_nth_case['Date'])
    from_nth_case['Min Date'] = pd.to_datetime(from_nth_case['Min Date'])
    
    # Create a variable that counts the number of days relative to the day when confirmed cases exceed N
    from_nth_case['N days'] = (from_nth_case['Date'] - from_nth_case['Min Date']).dt.days
    # print(from_nth_case.head())

    # Plot a line graph from dataframe from_nth_case with column 'N days' and 'Confirmed' mapped to x-axis and y-axis, respectively.
    # Distinguish each country by color (system-determined color)
    # str converts n integer into string and "'N days from '+ str(n) +' case'" is the title 
    fig = px.line(from_nth_case, x='N days', y='Confirmed', color='Country/Region', 
                  title='N days from '+ str(n) +' case', height=600)
    fig.show()
# Call function gt_n with argument 100000 (i.e., 100000 confirmed cases)
gt_n(100000)
# Draw horizontal bar plot with three arguments
# 1. variable of interest
# 2. top n countries
# 3. minimum population size (default value is 1000000)
def plot_hbar_wm(col, n, min_pop=1000000):
    df = worldometer_data[worldometer_data['Population']>min_pop]
    df = df.sort_values(col, ascending=True).tail(n)
    df.info()
    fig = px.bar(df,
                 x=col, y="Country/Region", color='WHO Region',  
                 text=col, orientation='h', width=700, 
                 color_discrete_sequence = px.colors.qualitative.Dark2)
    fig.update_layout(title=col+' (Only countries with Population > ' + str(min_pop), 
                      xaxis_title="", yaxis_title="", 
                      yaxis_categoryorder = 'total ascending',
                      uniformtext_minsize=8, uniformtext_mode='hide')
    fig.show()
    
# Draw histogram with two arguments
# 1. variable of interest
# 2. the number of bins
def plot_histogram_wm(col, bins):
    fig = px.histogram(worldometer_data[col], x=col, nbins=bins)
    fig.show()
# Draw bar chart for case fatality of the top 15 countries with the highest case fatality rate (with the minimum population of 1 million)
plot_hbar_wm('Case Fatality', 15, 1000000)
# Draw the histogram for case fatality rate (50 bins)
plot_histogram_wm("Case Fatality",50)
# Draw a joint plot to diagnose the relationship between Tests/1M population vs Total Cases/1M population
# Draw a regression line on the scatter plot
sns.jointplot(x = 'Tests/1M pop', y = 'Tot Cases/1M pop', data = worldometer_data, kind='reg')
# Show the descriptive statistics for case positivity bin (categorical variable)
worldometer_data.groupby('Case Positivity Bin')['Case Positivity Bin'].describe()
# Draw a Violin plot to diagnose the relationship betwen case positivity and case fatality rates
fig = go.Figure()

# Create a list of case positivity bin categories
bins = ['low', 'medium', 'high']

# Loop through each case positivity bin
for bin in bins:
    
    # worldometer_data['Case Positivity Bin'][worldometer_data['Case Positivity Bin'] == bin] means take the column 'Case Positivity Bin' and
    # filter the column, such that Case Positivity Bin equals 'low', 'medium', or 'high'
    fig.add_trace(go.Violin(x=worldometer_data['Case Positivity Bin'][worldometer_data['Case Positivity Bin'] == bin],
                            y=worldometer_data['Case Fatality'][worldometer_data['Case Positivity Bin'] == bin],
                            name=bin,
                            box_visible=True,
                            meanline_visible=True))
    
fig.update_layout(title='Case Fatality by Case Positivity Bins', 
                  yaxis_title="Case Fatality", xaxis_title="Case Positivity Bins", 
                  uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()
# Draw a joint plot to diagnose the relationship between fraction of population aged 65+ and case fatality rate
sns.jointplot(x = 'Fraction age 65+ years', y = "Case Fatality", data = worldometer_data, kind='reg')
# Show the summary statistics of column case positivity
worldometer_data["Case Positivity"].describe()

# Filter countries with Case Positivity less than 1% (i.e., 1 confirmed case out of 100 tests)
# These are countries that go for rigorous testing regime
benchmark_countries = worldometer_data[worldometer_data["Case Positivity"]<=0.01]
#benchmark_countries.info()
#benchmark_countries.head(20)
# Assume that the number of confirmed cases are close to the true infections rates for countries with gold standard testing regimes 
# Thus, their case fatality rates are closer to the true infection fatality rates
infection_fatality_rate = benchmark_countries['TotalDeaths'].sum() / benchmark_countries['TotalCases'].sum()

# Calculate the fraction of total Covid19 deaths for the population aged 65+ among the benchmark countries
benchmark_death_65y_pct = sum(benchmark_countries['TotalDeaths'] * benchmark_countries['Fraction age 65+ years']) / sum(benchmark_countries['TotalDeaths'])

print(infection_fatality_rate)
print(benchmark_death_65y_pct)

print('Estimated Infection Fatality Rate for a benchmark country with %.1f%s of population older than 65 years old \
is %.2f%s' %(100 * benchmark_death_65y_pct,'%',100 * infection_fatality_rate,'%'))
# Estimate Infection Fatality Ratio using the estimated fraction of total Covid19 deaths for the population aged 65+
worldometer_data['Estimated Infection Fatality Ratio'] \
    = ((worldometer_data['TotalDeaths'] * worldometer_data['Fraction age 65+ years']
        /worldometer_data['TotalDeaths']) / benchmark_death_65y_pct) * infection_fatality_rate

# Show descriptive statistics of the columns Estimated Infection Fatality Ratio and Case Fatality
worldometer_data['Estimated Infection Fatality Ratio'].describe()
worldometer_data['Case Fatality'].describe()

# Plot histogram of Estimated Infection Fatality Ratio and Case Fatality
px.histogram(worldometer_data, x='Estimated Infection Fatality Ratio', barmode="overlay")
px.histogram(worldometer_data, x='Case Fatality', barmode="overlay")

# Overlay both histograms for comparison
fig = go.Figure()

fig.add_trace(go.Histogram(x=worldometer_data['Estimated Infection Fatality Ratio'], 
    name = 'Estimated Infection Fatality Rate'
))

fig.add_trace(go.Histogram(x=worldometer_data['Case Fatality'], 
    name = 'Case Fatality Rate'
))

fig.update_layout(barmode='overlay', 
    title = 'Estimated Infection Fatality Rate vs. Case Fatality Rate',
    xaxis_title_text='Value', # xaxis label
    yaxis_title_text='Count', # yaxis label
)
                  
fig.update_traces(opacity=0.75)

fig.show()