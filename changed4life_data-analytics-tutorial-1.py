# Load the library and check its version, just to make sure we aren't using an older version.

import numpy as np

np.__version__
# Create a list comprising numbers from 0 to 9.

L = list(range(10))



[str(c) for c in L]
# Converting integers to string - this style of handling lists is known as list comprehension.

# List comprehension offers a versatile way to handle list manipulation tasks easily. We'll learn about them in future tutorials. 

# Here's an example.  



[type(item) for item in L]
# Create zero arrays

np.zeros(10, dtype='int')
# Create a 3 row x 5 column matrix of ones

np.ones((3, 5), dtype=float)
# Create a matrix with a predefined value

np.full((3, 5), 1.23)
# Create an array with a set sequence

np.arange(0, 20, 2)
# Create an array of even space between the given range of values

np.linspace(0, 1, 5)
# Create a 3 x 3 array of values picked from the normal distribution with mean 0 and standard deviation 1 in a given dimension

np.random.normal(0, 1, (3, 3))
# Create a 3 x 3 array of values picked from the Binomial (n, p) distribution in a given dimension

np.random.binomial(10, 0.5, (3, 5))
# Create an identity matrix

np.eye(3)
# Set a random seed

np.random.seed(0)
x1 = np.random.randint(10, size=6) # one dimension

x2 = np.random.randint(10, size=(3, 4)) # two dimensions

x3 = np.random.randint(10, size=(3, 4, 5)) # three dimensions



print("x3 ndim:", x3.ndim)

print("x3 shape:", x3.shape)

print("x3 size: ", x3.size)
# Create a 4 x 4 identity matrix



# Create a 3 x 10 matrix with all values equal '2'



# Create a 2 x 3 x 2 x 4 4D-matrix and print size and shape
x1 = np.array([4, 3, 4, 4, 8, 4])

x1
# Access the value at index zero

x1[0]
# Access the fifth value

x1[4]
# Access the last value

x1[-1]
# Access the second last value

x1[-2]
# In a multidimensional array, we need to specify row and column indices

x2
# 1st row and 2nd column value

x2[2, 3]
# 3rd row and last column value

x2[2, -1]
# Replace value at index [0, 0]

x2[0, 0] = 12

x2
# Using the x3 matrix (using index -1 when necessary)...



# Print the value at 3D-index [2, 0, 4]



# Modify the value at 3D-index [0, 3, 1] to be 100
x = np.arange(10)

x
# From the start to the 4th position

x[:5]
# From the 4th position to the end

x[3:]
# From 4th to 7th position

x[3:7]
# Return elements with even indices

x[:: 2]
# Return elements from the 1st position to the end by step two

x[1::2]
# Reverse the array

x[::-1]
# Create an array of consecutive numbers from 0 to 50 



# Print every number divisible by 6



# Then reverse that array



# Print the element at the 7th index
# You can concatenate two or more arrays at once.

x = np.array([1, 2, 3])

y = np.array([3, 2, 1])

z = [21, 21, 21]

np.concatenate([x, y, z])
# You can also use this function to create 2-dimensional arrays.

grid = np.array([[1, 2, 3], [4, 5, 6]])

np.concatenate([grid, grid])
# Using its axis parameter, you can define a row-wise or column-wise matrix

print(np.concatenate([grid, grid], axis=0))

print()

print(np.concatenate([grid, grid], axis=1))
x = np.array([3, 4, 5])

grid = np.array([[1, 2, 3], [17, 18, 19]])

np.vstack([x, grid])
# Similarly, you can merge arrays using np.hstack

z = np.array([[9], [9]])

np.hstack([grid, z])
x = np.arange(10)

x
x1, x2, x3 = np.split(x, [3, 6])

print(x1, x2, x3)
grid = np.arange(16).reshape((4, 4))

print(grid)

print()

upper, lower = np.vsplit(grid, [3])

print(upper)

print()

print(lower)
# Create a 1D-array of size 4 with numbers randomly distributed from 1 to 10



# Create a 2D-array of size 2 x 4 with numbers randomly distributed from 10 to 20



# Vertically stack the arrays to obtain a 3 x 4 matrix

# Load library - pd is just an alias. We use pd because it's short and literally abbreviates pandas.

# You can use any name as an alias. 

import pandas as pd
# Create a data frame - dictionary is used here where keys get converted to column names and values (represented as lists) to row values.

data = pd.DataFrame({'Country': ['Russia','Colombia','Chile','Equador','Nigeria'],

                    'Rank':[121,40,100,130,11]})

data
# We can do a quick analysis of any data set using:

data.describe()
# Among other things, it shows that the data set has 5 rows and 2 columns with their respective names.

data.info()
# Let's create another data frame.

data = pd.DataFrame({'group':['a', 'a', 'a', 'b','b', 'b', 'c', 'c','c'],'ounces':[4, 3, 12, 6, 7.5, 8, 3, 5, 6]})

data
# Let's sort the data frame by ounces, where inplace = True will make changes to the data

data.sort_values(by=['ounces'], ascending=True, inplace=False)
data.sort_values(by=['group','ounces'], ascending=[True,False], inplace=False)
# Create another data with duplicated rows

data = pd.DataFrame({'k1': ['one'] * 3 + ['two'] * 4, 'k2': [3, 2, 1, 3, 3, 4, 4]})

data
# Sort values

data.sort_values(by='k2')
# Remove duplicates - ta daaaa!

data.drop_duplicates()
data.drop_duplicates(subset='k1')
# Check data frame data

data
# Remove duplicates physically from data

data.drop_duplicates(inplace=True)

data
# Initialize data

data = pd.DataFrame({'k1': ['one'] * 3 + ['two'] * 4, 'k2': [3, 2, 1, 3, 3, 4, 4]})



# Drop duplicates from k2 column



# Sort by k2 column in ascending order

data = pd.DataFrame({'food': ['bacon', 'pulled pork', 'bacon', 'Pastrami','corned beef', 'Bacon', 'pastrami', 'honey ham','nova lox'],

                 'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})

data
meat_to_animal = {

'bacon': 'pig',

'pulled pork': 'pig',

'pastrami': 'cow',

'corned beef': 'cow',

'honey ham': 'pig',

'nova lox': 'salmon'

}



def meat_2_animal(series):

    if series['food'] == 'bacon':

        return 'pig'

    elif series['food'] == 'pulled pork':

        return 'pig'

    elif series['food'] == 'pastrami':

        return 'cow'

    elif series['food'] == 'corned beef':

        return 'cow'

    elif series['food'] == 'honey ham':

        return 'pig'

    else:

        return 'salmon'





# Create a new variable

data['animal'] = data['food'].map(str.lower).map(meat_to_animal)

data
# Another way of doing it is to convert the food values to lower case and apply the function meat_2_animal

lower = lambda x: x.lower()

data['food'] = data['food'].apply(lower)

data['animal2'] = data.apply(meat_2_animal, axis='columns')

data
data.assign(new_variable = data['ounces'] * 10)
data.drop('animal2',axis='columns',inplace=True)

data
data = pd.DataFrame({'k1': ['one'] * 3 + ['two'] * 4, 'k2': [3, 2, 1, 3, 3, 4, 4]})

data



# Build a function to convert k1 (string) into an integer number by using the above meat_2_animal function



# Create a new column by multiplying k2 and the new column you have just created



# Drop k1 column
# The function Series from pandas is used to create pandas arrays

data = pd.Series([1., -999., 2., -999., -1000., 3.])

data
# It is not a numpy array

type(data)
# Replace -999 with NaN values

data.replace(-999, np.nan, inplace=True)

data
# We can also replace multiple values at once

data = pd.Series([1., -999., 2., -999., -1000., 3.])

data.replace([-999, -1000], np.nan, inplace=True)

data
data = pd.DataFrame(np.arange(12).reshape((3, 4)),index=['Ohio', 'Colorado', 'New York'],columns=['one', 'two', 'three', 'four'])

data
# Using rename function

data.rename(index = {'Ohio':'SanF'}, columns={'one':'one_p','two':'two_p'}, inplace=True)

data
# You can also use string functions

data.rename(index = str.upper, columns=str.title, inplace=True)

data
ages = [20, 22, 25, 27, 21, 23, 37, 31, 61, 45, 41, 32]
# Understand the output - '(' means the value is included in the bin, '[' means the value is excluded

bins = [18, 25, 35, 60, 100]

cats = pd.cut(ages, bins)

cats
# To include the right bin value, we can do:

pd.cut(ages, bins, right=False)
# Let's check how many observations fall under each bin

pd.value_counts(cats)
bin_names = ['Youth', 'YoungAdult', 'MiddleAge', 'Senior']

new_cats = pd.cut(ages, bins, labels=bin_names)



pd.value_counts(new_cats)
# We can also calculate their cumulative sum

pd.value_counts(new_cats).cumsum()
salary = [120, 222, 25, 127, 121, 93, 337, 51, 31, 85, 41, 62]

bins = [0, 42, 126, 1000]

bin_names = ['Lower Class', 'Middle Class', 'Upper Class']

# Split the salaraies into the specified bins



# Apply the bin names to the categories

df = pd.DataFrame({'key1' : ['a', 'a', 'b', 'b', 'a'],

                   'key2' : ['one', 'two', 'one', 'two', 'one'],

                   'data1' : np.random.randn(5),

                   'data2' : np.random.randn(5)})

df
# Calculate the mean of data1 column by key1

grouped = df['data1'].groupby(df['key1'])

grouped.mean()
dates = pd.date_range('20130101', periods=6)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))

df
# Get first n rows from the data frame

df[:3]
# Slice based on date range

df['20130101':'20130104']
# Slicing based on column names

df.loc[:, ['A','B']]
# Slicing based on both row index labels and column names

df.loc['20130102': '20130103', ['A','B']]
# Slicing based on index of columns

df.iloc[3] # returns the 4th row (index is the 3rd)
# Returns a specific range of rows

df.iloc[2:4, 0:2]
# Returns specific rows and columns using lists containing columns or row indices

df.iloc[[1,5],[0,2]] 
df[df.A > 1]
# We can copy the data set

df2 = df.copy()

df2['E'] = ['one', 'one','two','three','four','three']

df2
# Select rows based on column values

df2[df2['E'].isin(['two', 'four'])]
# Select all rows except those with two and four

df2[~df2['E'].isin(['two','four'])]
# List all columns where A is greater than C

df.query('A > C')
# Using OR condition

df.query('A < B | C > A')
iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

iris



# What is the average measurements for each species



# Return these 2 indexs: [1,5] [0,2]



# Return the sepal_length and species column



# Return rows in list: ['setosa','virginica']



# Are there any siutations that either sepal or petal widths are greater that lengths?

# Create a data frame

data = pd.DataFrame({'group': ['a', 'a', 'a', 'b','b', 'b', 'c', 'c', 'c'],

                 'ounces': [4, 3, 12, 6, 7.5, 8, 3, 5, 6]})

data
# Calculate means of each group

data.pivot_table(values='ounces', index='group', aggfunc=np.mean)
# Calculate count by each group

data.pivot_table(values='ounces', index='group', aggfunc='count')
import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import urllib, json
np.random.seed(1)



N = 100000



df = pd.DataFrame(dict(x=np.random.randn(N), y=np.random.randn(N)))



fig = px.scatter(df, x="x", y="y", render_mode='webgl')



fig.update_traces(marker_line=dict(width=1, color='DarkSlateGray'))



fig.show()
# x and y given as DataFrame columns

iris = px.data.iris() # iris is a pandas DataFrame

fig = px.scatter(iris, x="sepal_width", y="sepal_length")

fig.show()
iris = px.data.iris()

fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species",

                 size='petal_length', hover_data=['petal_width'])

fig.show()
gapminder = px.data.gapminder().query("continent == 'Oceania'")

fig = px.line(gapminder, x='year', y='lifeExp', color='country')

fig.show()
N = 1000

t = np.linspace(0, 10, 100)

y = np.sin(t)



fig = go.Figure(data=go.Scatter(x=t, y=y, mode='markers'))



fig.show()
# Create random data with numpy

np.random.seed(1)



N = 100

random_x = np.linspace(0, 1, N)

random_y0 = np.random.randn(N) + 5

random_y1 = np.random.randn(N)

random_y2 = np.random.randn(N) - 5



fig = go.Figure()



# Add traces

fig.add_trace(go.Scatter(x=random_x, y=random_y0,

                    mode='markers',

                    name='markers'))

fig.add_trace(go.Scatter(x=random_x, y=random_y1,

                    mode='lines+markers',

                    name='lines+markers'))

fig.add_trace(go.Scatter(x=random_x, y=random_y2,

                    mode='lines',

                    name='lines'))



fig.show()
t = np.linspace(0, 10, 100)



fig = go.Figure()



fig.add_trace(go.Scatter(

    x=t, y=np.sin(t),

    name='sin',

    mode='markers',

    marker_color='rgba(152, 0, 0, .8)'

))



fig.add_trace(go.Scatter(

    x=t, y=np.cos(t),

    name='cos',

    marker_color='rgba(255, 182, 193, .9)'

))



# Set options common to all traces with fig.update_traces

fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)

fig.update_layout(title='Styled Scatter',

                  yaxis_zeroline=False, xaxis_zeroline=False)





fig.show()
data = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv")



fig = go.Figure(data=go.Scatter(x=data['Postal'],

                                y=data['Population'],

                                mode='markers',

                                marker_color=data['Population'],

                                text=data['State'])) # hover text goes here



fig.update_layout(title='Population of USA States')

fig.show()
fig = go.Figure(data=go.Scatter(

    y = np.random.randn(500),

    mode='markers',

    marker=dict(

        size=16,

        color=np.random.randn(500), #set color equal to a variable

        colorscale='Viridis', # one of plotly colorscales

        showscale=True

    )

))



fig.show()
from IPython.display import IFrame

IFrame(src= "https://dash-simple-apps.plotly.host/dash-linescatterplot/", width="100%",height="750px", frameBorder="0")
from IPython.display import IFrame

IFrame(src= "https://dash-simple-apps.plotly.host/dash-linescatterplot/code", width="100%",height=500, frameBorder="0")
gapminder = px.data.gapminder()



fig = px.scatter(gapminder.query("year==2007"), x="gdpPercap", y="lifeExp",size="pop", color="continent",hover_name="country", log_x=True, size_max=60)

fig.show()
# Add data

month = ['January', 'February', 'March', 'April', 'May', 'June', 'July',

         'August', 'September', 'October', 'November', 'December']

high_2000 = [32.5, 37.6, 49.9, 53.0, 69.1, 75.4, 76.5, 76.6, 70.7, 60.6, 45.1, 29.3]

low_2000 = [13.8, 22.3, 32.5, 37.2, 49.9, 56.1, 57.7, 58.3, 51.2, 42.8, 31.6, 15.9]

high_2007 = [36.5, 26.6, 43.6, 52.3, 71.5, 81.4, 80.5, 82.2, 76.0, 67.3, 46.1, 35.0]

low_2007 = [23.6, 14.0, 27.0, 36.8, 47.6, 57.7, 58.9, 61.2, 53.3, 48.5, 31.0, 23.6]

high_2014 = [28.8, 28.5, 37.0, 56.8, 69.7, 79.7, 78.5, 77.8, 74.1, 62.6, 45.3, 39.9]

low_2014 = [12.7, 14.3, 18.6, 35.5, 49.9, 58.0, 60.0, 58.6, 51.7, 45.2, 32.2, 29.1]



fig = go.Figure()

# Create and style traces

fig.add_trace(go.Scatter(x=month, y=high_2014, name='High 2014',

                         line=dict(color='firebrick', width=4)))

fig.add_trace(go.Scatter(x=month, y=low_2014, name = 'Low 2014',

                         line=dict(color='royalblue', width=4)))

fig.add_trace(go.Scatter(x=month, y=high_2007, name='High 2007',

                         line=dict(color='firebrick', width=4,

                              dash='dash') # dash options include 'dash', 'dot', and 'dashdot'

))

fig.add_trace(go.Scatter(x=month, y=low_2007, name='Low 2007',

                         line = dict(color='royalblue', width=4, dash='dash')))

fig.add_trace(go.Scatter(x=month, y=high_2000, name='High 2000',

                         line = dict(color='firebrick', width=4, dash='dot')))

fig.add_trace(go.Scatter(x=month, y=low_2000, name='Low 2000',

                         line=dict(color='royalblue', width=4, dash='dot')))



# Edit the layout

fig.update_layout(title='Average High and Low Temperatures in New York',

                   xaxis_title='Month',

                   yaxis_title='Temperature (degrees F)')





fig.show()
gapminder = px.data.gapminder()

fig = px.area(gapminder, x="year", y="pop", color="continent",line_group="country")

fig.show()
x=['a','b','c','d']

fig = go.Figure(go.Bar(x=x, y=[2, 5, 1, 9], name='Montreal'))

fig.add_trace(go.Bar(x=x, y=[1, 4, 9, 16], name='Ottawa'))

fig.add_trace(go.Bar(x=x, y=[6, 8, 4.5, 8], name='Toronto'))



fig.update_layout(barmode='stack', xaxis={'categoryorder':'array', 'categoryarray':['d','a','c','b']})

fig.show()
fig = go.Figure(data=[go.Bar(

    x=[1, 2, 3, 5.5, 10],

    y=[10, 8, 6, 4, 2],

    width=[0.8, 0.8, 0.8, 3.5, 4] # customize width here

)])



fig.show()
tips = px.data.tips()

fig = px.bar(tips, x="sex", y="total_bill", color="smoker", barmode="group",

             facet_row="time", facet_col="day",

             category_orders={"day": ["Thur", "Fri", "Sat", "Sun"],

                              "time": ["Lunch", "Dinner"]})

fig.show()
years = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003,

         2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012]



fig = go.Figure()

fig.add_trace(go.Bar(x=years,

                y=[219, 146, 112, 127, 124, 180, 236, 207, 236, 263,

                   350, 430, 474, 526, 488, 537, 500, 439],

                name='Rest of world',

                marker_color='rgb(55, 83, 109)'

                ))

fig.add_trace(go.Bar(x=years,

                y=[16, 13, 10, 11, 28, 37, 43, 55, 56, 88, 105, 156, 270,

                   299, 340, 403, 549, 499],

                name='China',

                marker_color='rgb(26, 118, 255)'

                ))



fig.update_layout(

    title='US Export of Plastic Scrap',

    xaxis_tickfont_size=14,

    yaxis=dict(

        title='USD (millions)',

        titlefont_size=16,

        tickfont_size=14,

    ),

    legend=dict(

        x=0,

        y=1.0,

        bgcolor='rgba(255, 255, 255, 0)',

        bordercolor='rgba(255, 255, 255, 0)'

    ),

    barmode='group',

    bargap=0.15, # gap between bars of adjacent location coordinates.

    bargroupgap=0.1 # gap between bars of the same location coordinate.

)

fig.show()
fig = go.Figure(go.Bar(

            x=[20, 14, 23],

            y=['giraffes', 'orangutans', 'monkeys'],

            orientation='h'))



fig.show()
tips = px.data.tips()

fig = px.bar(tips, x="total_bill", y="sex", color='day', orientation='h',

             hover_data=["tip", "size"],

             height=400,

             title='Restaurant bills')

fig.show()
top_labels = ['Strongly<br>agree', 'Agree', 'Neutral', 'Disagree',

              'Strongly<br>disagree']



colors = ['rgba(38, 24, 74, 0.8)', 'rgba(71, 58, 131, 0.8)',

          'rgba(122, 120, 168, 0.8)', 'rgba(164, 163, 204, 0.85)',

          'rgba(190, 192, 213, 1)']



x_data = [[21, 30, 21, 16, 12],

          [24, 31, 19, 15, 11],

          [27, 26, 23, 11, 13],

          [29, 24, 15, 18, 14]]



y_data = ['The course was effectively<br>organized',

          'The course developed my<br>abilities and skills ' +

          'for<br>the subject', 'The course developed ' +

          'my<br>ability to think critically about<br>the subject',

          'I would recommend this<br>course to a friend']



fig = go.Figure()



for i in range(0, len(x_data[0])):

    for xd, yd in zip(x_data, y_data):

        fig.add_trace(go.Bar(

            x=[xd[i]], y=[yd],

            orientation='h',

            marker=dict(

                color=colors[i],

                line=dict(color='rgb(248, 248, 249)', width=1)

            )

        ))



fig.update_layout(

    xaxis=dict(

        showgrid=False,

        showline=False,

        showticklabels=False,

        zeroline=False,

        domain=[0.15, 1]

    ),

    yaxis=dict(

        showgrid=False,

        showline=False,

        showticklabels=False,

        zeroline=False,

    ),

    barmode='stack',

    paper_bgcolor='rgb(248, 248, 255)',

    plot_bgcolor='rgb(248, 248, 255)',

    margin=dict(l=120, r=10, t=140, b=80),

    showlegend=False,

)



annotations = []



for yd, xd in zip(y_data, x_data):

    # labeling the y-axis

    annotations.append(dict(xref='paper', yref='y',

                            x=0.14, y=yd,

                            xanchor='right',

                            text=str(yd),

                            font=dict(family='Arial', size=14,

                                      color='rgb(67, 67, 67)'),

                            showarrow=False, align='right'))

    # labeling the first percentage of each bar (x_axis)

    annotations.append(dict(xref='x', yref='y',

                            x=xd[0] / 2, y=yd,

                            text=str(xd[0]) + '%',

                            font=dict(family='Arial', size=14,

                                      color='rgb(248, 248, 255)'),

                            showarrow=False))

    # labeling the first Likert scale (on the top)

    if yd == y_data[-1]:

        annotations.append(dict(xref='x', yref='paper',

                                x=xd[0] / 2, y=1.1,

                                text=top_labels[0],

                                font=dict(family='Arial', size=14,

                                          color='rgb(67, 67, 67)'),

                                showarrow=False))

    space = xd[0]

    for i in range(1, len(xd)):

            # labeling the rest of percentages for each bar (x_axis)

            annotations.append(dict(xref='x', yref='y',

                                    x=space + (xd[i]/2), y=yd,

                                    text=str(xd[i]) + '%',

                                    font=dict(family='Arial', size=14,

                                              color='rgb(248, 248, 255)'),

                                    showarrow=False))

            # labeling the Likert scale

            if yd == y_data[-1]:

                annotations.append(dict(xref='x', yref='paper',

                                        x=space + (xd[i]/2), y=1.1,

                                        text=top_labels[i],

                                        font=dict(family='Arial', size=14,

                                                  color='rgb(67, 67, 67)'),

                                        showarrow=False))

            space += xd[i]



fig.update_layout(annotations=annotations)



fig.show()
y_saving = [1.3586, 2.2623000000000002, 4.9821999999999997, 6.5096999999999996,

            7.4812000000000003, 7.5133000000000001, 15.2148, 17.520499999999998

            ]

y_net_worth = [93453.919999999998, 81666.570000000007, 69889.619999999995,

               78381.529999999999, 141395.29999999999, 92969.020000000004,

               66090.179999999993, 122379.3]

x = ['Japan', 'United Kingdom', 'Canada', 'Netherlands',

     'United States', 'Belgium', 'Sweden', 'Switzerland']





# Creating two subplots

fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,

                    shared_yaxes=False, vertical_spacing=0.001)



fig.append_trace(go.Bar(

    x=y_saving,

    y=x,

    marker=dict(

        color='rgba(50, 171, 96, 0.6)',

        line=dict(

            color='rgba(50, 171, 96, 1.0)',

            width=1),

    ),

    name='Household savings, percentage of household disposable income',

    orientation='h',

), 1, 1)



fig.append_trace(go.Scatter(

    x=y_net_worth, y=x,

    mode='lines+markers',

    line_color='rgb(128, 0, 128)',

    name='Household net worth, Million USD/capita',

), 1, 2)



fig.update_layout(

    title='Household savings & net worth for eight OECD countries',

    yaxis=dict(

        showgrid=False,

        showline=False,

        showticklabels=True,

        domain=[0, 0.85],

    ),

    yaxis2=dict(

        showgrid=False,

        showline=True,

        showticklabels=False,

        linecolor='rgba(102, 102, 102, 0.8)',

        linewidth=2,

        domain=[0, 0.85],

    ),

    xaxis=dict(

        zeroline=False,

        showline=False,

        showticklabels=True,

        showgrid=True,

        domain=[0, 0.42],

    ),

    xaxis2=dict(

        zeroline=False,

        showline=False,

        showticklabels=True,

        showgrid=True,

        domain=[0.47, 1],

        side='top',

        dtick=25000,

    ),

    legend=dict(x=0.029, y=1.038, font_size=10),

    margin=dict(l=100, r=20, t=70, b=70),

    paper_bgcolor='rgb(248, 248, 255)',

    plot_bgcolor='rgb(248, 248, 255)',

)



annotations = []



y_s = np.round(y_saving, decimals=2)

y_nw = np.rint(y_net_worth)



# Adding labels

for ydn, yd, xd in zip(y_nw, y_s, x):

    # labeling the scatter savings

    annotations.append(dict(xref='x2', yref='y2',

                            y=xd, x=ydn - 20000,

                            text='{:,}'.format(ydn) + 'M',

                            font=dict(family='Arial', size=12,

                                      color='rgb(128, 0, 128)'),

                            showarrow=False))

    # labeling the bar net worth

    annotations.append(dict(xref='x1', yref='y1',

                            y=xd, x=yd + 3,

                            text=str(yd) + '%',

                            font=dict(family='Arial', size=12,

                                      color='rgb(50, 171, 96)'),

                            showarrow=False))

# Source

annotations.append(dict(xref='paper', yref='paper',

                        x=-0.2, y=-0.109,

                        text='OECD "' +

                             '(2015), Household savings (indicator), ' +

                             'Household net worth (indicator). doi: ' +

                             '10.1787/cfc6f499-en (Accessed on 05 June 2015)',

                        font=dict(family='Arial', size=10, color='rgb(150,150,150)'),

                        showarrow=False))



fig.update_layout(annotations=annotations)



fig.show()
import plotly.figure_factory as ff



df = [dict(Task="Job A", Start='2009-01-01', Finish='2009-02-28'),

      dict(Task="Job B", Start='2009-03-05', Finish='2009-04-15'),

      dict(Task="Job C", Start='2009-02-20', Finish='2009-05-30')]



fig = ff.create_gantt(df)

fig.show()
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gantt_example.csv')



fig = ff.create_gantt(df, colors=['#333F44', '#93e4c1'], index_col='Complete',

                      show_colorbar=True, bar_width=0.2, showgrid_x=True, showgrid_y=True)

fig.show()
df = [

    dict(Task='Morning Sleep', Start='2016-01-01', Finish='2016-01-01 6:00:00', Resource='Sleep'),

    dict(Task='Breakfast', Start='2016-01-01 7:00:00', Finish='2016-01-01 7:30:00', Resource='Food'),

    dict(Task='Work', Start='2016-01-01 9:00:00', Finish='2016-01-01 11:25:00', Resource='Brain'),

    dict(Task='Break', Start='2016-01-01 11:30:00', Finish='2016-01-01 12:00:00', Resource='Rest'),

    dict(Task='Lunch', Start='2016-01-01 12:00:00', Finish='2016-01-01 13:00:00', Resource='Food'),

    dict(Task='Work', Start='2016-01-01 13:00:00', Finish='2016-01-01 17:00:00', Resource='Brain'),

    dict(Task='Exercise', Start='2016-01-01 17:30:00', Finish='2016-01-01 18:30:00', Resource='Cardio'),

    dict(Task='Post Workout Rest', Start='2016-01-01 18:30:00', Finish='2016-01-01 19:00:00', Resource='Rest'),

    dict(Task='Dinner', Start='2016-01-01 19:00:00', Finish='2016-01-01 20:00:00', Resource='Food'),

    dict(Task='Evening Sleep', Start='2016-01-01 21:00:00', Finish='2016-01-01 23:59:00', Resource='Sleep')

]



colors = dict(Cardio = 'rgb(46, 137, 205)',

              Food = 'rgb(114, 44, 121)',

              Sleep = 'rgb(198, 47, 105)',

              Brain = 'rgb(58, 149, 136)',

              Rest = 'rgb(107, 127, 135)')



fig = ff.create_gantt(df, colors=colors, index_col='Resource', title='Daily Schedule',

                      show_colorbar=True, bar_width=0.8, showgrid_x=True, showgrid_y=True)

fig.show()
labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']

values = [4500, 2500, 1053, 500]



fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.show()
labels = ["Asia", "Europe", "Africa", "Americas", "Oceania"]



fig = make_subplots(1, 2, specs=[[{'type': 'domain'}, {'type': 'domain'}]],

                    subplot_titles=['1980', '2007'])

fig.add_trace(go.Pie(labels=labels, values=[4, 7, 1, 7, 0.5], scalegroup='one',

                     name="World GDP 1980"), 1, 1)

fig.add_trace(go.Pie(labels=labels, values=[21, 15, 3, 19, 1], scalegroup='one',

                     name="World GDP 2007"), 1, 2)



fig.update_layout(title_text='World GDP')

fig.show()
df1 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/718417069ead87650b90472464c7565dc8c2cb1c/sunburst-coffee-flavors-complete.csv')

df2 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/718417069ead87650b90472464c7565dc8c2cb1c/coffee-flavors.csv')



fig = go.Figure()



fig.add_trace(go.Sunburst(

    ids=df1.ids,

    labels=df1.labels,

    parents=df1.parents,

    domain=dict(column=0)

))



fig.add_trace(go.Sunburst(

    ids=df2.ids,

    labels=df2.labels,

    parents=df2.parents,

    domain=dict(column=1),

    maxdepth=2

))



fig.update_layout(

    grid= dict(columns=2, rows=1),

    margin = dict(t=0, l=0, r=0, b=0)

)



fig.show()
fig = go.Figure(data=[go.Table(header=dict(values=['A Scores', 'B Scores']),

                 cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]]))

                     ])

fig.show()
df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv')



fig = go.Figure(data=[go.Table(

    header=dict(values=list(df.columns),

                fill_color='paleturquoise',

                align='left'),

    cells=dict(values=[df.Rank, df.State, df.Postal, df.Population],

               fill_color='lavender',

               align='left'))

])



fig.show()
fig = go.Figure(data=[go.Sankey(

    node = dict(

      pad = 15,

      thickness = 20,

      line = dict(color = "black", width = 0.5),

      label = ["A1", "A2", "B1", "B2", "C1", "C2"],

      color = "blue"

    ),

    link = dict(

      source = [0, 1, 0, 2, 3, 3], # indices correspond to labels, eg A1, A2, A2, B1, ...

      target = [2, 3, 3, 4, 4, 5],

      value = [8, 4, 2, 8, 4, 2]

  ))])



fig.update_layout(title_text="Basic Sankey Diagram", font_size=10)

fig.show()
url = 'https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json'

response = urllib.request.urlopen(url)

data = json.loads(response.read())



fig = go.Figure(data=[go.Sankey(

    valueformat = ".0f",

    valuesuffix = "TWh",

    node = dict(

      pad = 15,

      thickness = 15,

      line = dict(color = "black", width = 0.5),

      label =  data['data'][0]['node']['label'],

      color =  data['data'][0]['node']['color']

    ),

    link = dict(

      source =  data['data'][0]['link']['source'],

      target =  data['data'][0]['link']['target'],

      value =  data['data'][0]['link']['value'],

      label =  data['data'][0]['link']['label']

  ))])



fig.update_layout(

    title="Energy forecast for 2050<br>Source: Department of Energy & Climate Change, Tom Counsell via <a href='https://bost.ocks.org/mike/sankey/'>Mike Bostock</a>",

    font=dict(size = 10, color = 'white'),

    plot_bgcolor='black',

    paper_bgcolor='black'

)



fig.show()
# Load the data

train  = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
# Check data set

train.info()
print ("The train data has",train.shape)

print ("The test data has",test.shape)

('The train data has', (32561, 15))

('The test data has', (16281, 15))





# Let's have a glimpse of the data set

train.head()
nans = train.shape[0] - train.dropna().shape[0]

print ("%d rows have missing values in the train data" %nans)



nand = test.shape[0] - test.dropna().shape[0]

print ("%d rows have missing values in the test data" %nand)
# Only 3 columns have missing values

train.isnull().sum()
cat = train.select_dtypes(include=['O'])

cat.apply(pd.Series.nunique)
# Education

train.workclass.value_counts(sort=True)

train.workclass.fillna('Private',inplace=True)



# Occupation

train.occupation.value_counts(sort=True)

train.occupation.fillna('Prof-specialty',inplace=True)



# Native Country

train['native.country'].value_counts(sort=True)

train['native.country'].fillna('United-States',inplace=True)
train.isnull().sum()
# Check proportion of target variable

train.target.value_counts()/train.shape[0]
pd.crosstab(train.education, train.target,margins=True)/train.shape[0]
# Load sklearn and encode all object type variables

from sklearn import preprocessing



for x in train.columns:

    if train[x].dtype == 'object':

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train[x].values))

        train[x] = lbl.transform(list(train[x].values))
train.head()
# <50K = 0 and >50K = 1

train.target.value_counts()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score



y = train['target']

del train['target']



X = train

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)



#train the RF classifier

clf = RandomForestClassifier(n_estimators = 500, max_depth = 6)

clf.fit(X_train,y_train)



RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=6, max_features='auto', max_leaf_nodes=None,

            min_impurity_split=1e-07, min_samples_leaf=1,

            min_samples_split=2, min_weight_fraction_leaf=0.0,

            n_estimators=500, n_jobs=1, oob_score=False, random_state=None,

            verbose=0, warm_start=False)



clf.predict(X_test)
# Make prediction and check model's accuracy

prediction = clf.predict(X_test)

acc =  accuracy_score(np.array(y_test),prediction)

print ('The accuracy of Random Forest is {}'.format(acc))