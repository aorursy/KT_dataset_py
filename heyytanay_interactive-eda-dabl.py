import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

import datetime

from colorama import Fore, Style

from math import floor, ceil



import plotly.express as px

import plotly.graph_objs as go



try:

    import dabl

except:

    ! pip -q install dabl

    import dabl

plt.style.use('classic')
def cout(string: str, color: str) -> str:

    """

    Prints a string in the required color

    """

    print(color+string+Style.RESET_ALL)
data = pd.read_csv("../input/windows-store/msft.csv")

data.head()
data.describe()
cout(f"The Shape of the data is: {data.shape}", Fore.CYAN)
cout(f"There are: {int(data.isna().sum().any())} Nan Values in the Data", Fore.GREEN)
data = data.dropna()
def clean_prices(string):

    if string == "Free":

        return "F"

    else:

        string = string.replace(',', '')

        price = int(string[2:-3])

        if price <= 250:

            price = "C"

        elif price <= 600 and price > 250:

            price = "VC"

        else:

            price = "VVC"

        return price

data['Price'] = data['Price'].apply(clean_prices)
targets = data['Price'].value_counts().tolist()

values = list(dict(data['Price'].value_counts()).keys())



fig = px.pie(

    values=targets, 

    names=["Free", "Very Costly", "Costly", "Very Very Costly"],

    title='App Prices Distribution'

)

fig.show()
# Pie Chart

targets = data['Rating'].value_counts().tolist()

values = list(dict(data['Rating'].value_counts()).keys())



fig = px.pie(

    values=targets, 

    names=values,

    title='App Ratings Distribution'

)

fig.show()
# And it's Count Plot

targets = data['Rating'].value_counts().tolist()

values = list(dict(data['Rating'].value_counts()).keys())



fig = px.bar(

    x=values,

    y=targets,

    color=values,

    labels={'x':'Ratings', 'y':'Count'},

    title="Ratings Count Distribution"

)



fig.show()
data['Category'].unique()
# Let's look at it's pie chart

targets = data['Category'].value_counts().tolist()

values = list(dict(data['Category'].value_counts()).keys())



fig = px.pie(

    values=targets, 

    names=values,

    title='Categories Distribution'

)

fig.show()
# And also it's count plot

targets = data['Category'].value_counts().tolist()

values = list(dict(data['Category'].value_counts()).keys())



fig = px.bar(

    x=values,

    y=targets,

    color=values,

    labels={'x':'Categories', 'y':'Count'},

    title="Category Count Distribution"

)



fig.show()
mean_reviews = floor(data['No of people Rated'].mean())

max_rev = data['No of people Rated'].max()

min_rev = data['No of people Rated'].min()

max_reviews_apps = data[data['No of people Rated'] == data['No of people Rated'].max()]['Name'].tolist()

min_reviews_apps = data[data['No of people Rated'] == data['No of people Rated'].min()]['Name'].tolist()



cout(f"Average App reviews are: {mean_reviews}", Fore.CYAN)

cout(f"Apps with most reviews are: {max_reviews_apps} having: {max_rev} reviews.", Fore.BLUE)

cout(f"Apps with most reviews are: {min_reviews_apps} having: {min_rev} reviews.", Fore.MAGENTA)
plt.rcParams['figure.figsize'] = (18, 6)

plt.style.use('fivethirtyeight')

dabl.plot(data, target_col = 'Price')
plt.rcParams['figure.figsize'] = (18, 6)

plt.style.use('fivethirtyeight')

dabl.plot(data, target_col = 'Category')
plt.rcParams['figure.figsize'] = (18, 6)

plt.style.use('fivethirtyeight')

dabl.plot(data, target_col = 'Rating')