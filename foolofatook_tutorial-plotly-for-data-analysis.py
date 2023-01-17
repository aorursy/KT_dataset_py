# Importing the necessary libraries



# Importing pandas to use dataframes

import pandas as pd



# Importing plotly express which will be used for creating the visualizations

import plotly.express as px

from plotly.offline import init_notebook_mode

# Doing this to make sure the graphs are visible in the kaggle kernels and not just a blank white screen

init_notebook_mode()
# Reading all the datasets and creating dataframes



heart = pd.read_csv('../input/heart-disease-uci/heart.csv')

covid = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')

co2 = pd.read_csv('../input/co2-ghg-emissionsdata/co2_emission.csv')

houses = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
heart.head()
covid.head()
co2.head()
houses.head()
fig = px.scatter(heart, # the dataframe that has the data points we want to plot

                 x = 'trestbps', # the name of the column in the dataframe whose values will be plotted on the x-axis

                 y = 'chol', # the name of the column in the dataframe whose values will be plotted on the x-axis

                 color = 'target',# the name of the column that will be used to assign colour to the marks on the Scatter Plot

                 title='Cholestrol vs Blood Pressure', # Title of the plot

                 template = 'ggplot2' # ggplot2 is one of the in-built templates in plotly, used for some theming of the graphs

                )

fig.show()
fig = px.scatter(heart, x = 'trestbps', y = 'chol', title='Cholestrol vs Blood Pressure', 

                 facet_col = 'sex', # the name of the column in the dataframe whose values are used for creating subplots

                 color = 'target', template = 'ggplot2')

fig.show()
fig = px.line(co2[co2['Entity'] == 'India'],# the dataframe

              x = 'Year', y = r'Annual CO₂ emissions (tonnes )', 

              title = 'Annual Co2 Emmission by India over the years' # title of the graph

             )

fig.show()
#calculating the sum of all Confirmed Cases in each Country/Region

covidCases = pd.DataFrame(covid.groupby('Country/Region')['Confirmed'].sum()).reset_index()

#sorting the dataset in descending order on the basis of number of Confirmed cases,i.e countries with the most Confirmed covid cases will be at the top

covidCases = covidCases.sort_values(by = 'Confirmed', ascending = False)



fig = px.bar(covidCases.iloc[:20], #plotting only the top 20 Countries

             x = 'Country/Region', y = 'Confirmed', title = 'Top 20 Countries based on number of Confirmed Covid Cases')

fig.show()
fig = px.bar(covidCases.iloc[:20],  

             x = 'Confirmed', y = 'Country/Region', #notice how we've swapped the x and y values here in comparison to the Vertical Bar Chart

             orientation = 'h', #orientation - 'h' signifies horizontal orientation, thus the bar chart converts into a row chart

             #however, even if we omit the orientation parameter, we would still get the same bar chart as swapping what goes on the x-axis and what on the y-axis was sufficient in our case

             title = 'Top 20 Countries based on number of Confirmed Covid Cases')

fig.show()
count = pd.DataFrame(heart.groupby('target')['slope'].value_counts().sort_index()) #calculating the number of samples for each Slope in both the targets

count = count.rename_axis(['target', 'Slope']).reset_index()

count['Counts'] = count['slope'] #adding a column with the name Counts



fig = px.bar(count, x = 'Slope', y = 'Counts',# Plotly Express Automatically labels the Axis based on the column names, thus, the fact, that I created a Column named Counts helps in creating meaning axis labels

             facet_col = 'target',# faceting using the target values, which are 0 and 1

             title = 'Slope Distribution Across Target')

fig.show()
fig = px.pie(heart, # the dataframe from which values would be taken for plotting

             names = 'slope',# the values from this column are used as labels for the sectors of the pie chart. Since we do not set the values parameters, the number of observations from this column are used

             color_discrete_sequence=px.colors.sequential.Inferno, #plotly has a lot of in-built colour scales, The Inferno color scale is used to assign the colors to each of the sectors here

             title = 'Demonstrating Pie Charts')

fig.show()
# We can verify that the Values of the sectos are indeed the number of observations for that Slope in our dataset 

heart['slope'].value_counts(normalize = True)
fig = px.pie(heart, names = 'slope',

             color_discrete_sequence=px.colors.sequential.RdBu,#using another color scale, because why not?

             hole=0.3, #determines the radius of the hole. Using 0.3 is a good standard according to me

             title = 'Demonstrating Donut Chart')

fig.show()
fig = px.pie(heart, names = 'slope',title = 'Advanced Customizations of Pie Chart')

# each chart in a plotly figure is called a trace. The plotly figure.update_traces allows us to have much finer control over the charts. 

fig.update_traces(textinfo = 'value',# we can now display actual values and not percentages in the pie chart

                  insidetextorientation = 'tangential',# the text would be tangetially oriented inside the chart

                  pull = [0.2,0,0] #pull the first sector, here the sector belonging to Slope 0

                 )

fig.show()
fig = px.histogram(houses,x = 'SalePrice',#the distribution of this column is plotted along the x-axis

                   title = 'Distribution of House Sale Price')

fig.show()
fig = px.histogram(houses,x = 'SalePrice', title = 'Transforming the x-axis into log scale', 

                   log_x=True # the x-axis values are transformed into log scale, this can be seen in the range of values in the x-axis

                  )

fig.show()
fig = px.histogram(houses,x = 'SalePrice', title = 'Distribution of House Price Across Years',

                   nbins=200, #this sets the number of bins to 200

                   color='YrSold' # each year would have a histogram plotted with different colours

                  )

fig.show()
fig = px.box(heart, x = 'slope',y='chol',title = 'Distribution of Cholestrol across various slope', color = 'target')

fig.show()
fig = px.box(heart, x = 'slope',y='chol',title = 'Distribution of Cholestrol across various slope with Underlying data', 

             color = 'target', points = 'all')

fig.show()
fig = px.violin(heart, x = 'slope',y='chol',title = 'Distribution of Cholestrol across various slope', color = 'target')

fig.show()
fig = px.violin(heart, x = 'slope',y='chol',title = 'Distribution of Cholestrol across various slope with Underlying data', 

             color = 'target', points = 'all')

fig.show()