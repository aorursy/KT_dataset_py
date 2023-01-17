# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Read data

data = pd.read_csv("../input/acquisitions.csv")

# Check the shape of data

data.shape
# List all columns

data.columns
# Also, let us view first 5 row entries to see how the data looks like.

data.head()
# Also, a quick glance at last five row entries

data.tail()
# Let's check column types and some statistics info

data.info()
data.isnull().sum()
# ------- First update missing vals for MonthDate Column -----------

# Let us find all distinct possible values for MonthDate column

print(np.unique(data['AcquisitionMonthDate']))



# Since, it is float64 type (we saw earlier), we will convert it to numeric type first.

data['AcquisitionMonthDate'] = pd.to_numeric(data['AcquisitionMonthDate'])



# Replace missing values for this column with mode

data['AcquisitionMonthDate'] = data['AcquisitionMonthDate'].fillna(data['AcquisitionMonthDate'].mode()[0])
# ------- Next update missing vals for Country Column -----------

# Find distinct country names for Country column

print(set(data['Country']))



# Let us replace null values with empty string

data['Country'] = data['Country'].fillna('Other')



# Check again for missing data

data['Country'].isnull().sum()
# ------- First update missing vals for Derived products Column -----------

print(set(data['Derived products'])) # We see nan in entries



# Replace null with empty string.

data['Derived products'] = data['Derived products'].fillna(' ')



# Check for missing values now

print(data['Derived products'].isna().sum())
# Total number of acquisitions in each year

for col in ('AcquisitionMonth','AcquisitionYear','Country'):

    plt.figure(figsize = (16,6))

    data[col].value_counts(sort = False).plot.bar(grid = True) 

    plt.xlabel(str(col) + " -->")

    plt.title("Acquisitions as per each {}".format(col))

    plt.show()
# Total number of acquisitions by each company

plt.figure(figsize = (12,4))

data['ParentCompany'].value_counts(sort = False).plot.bar(grid = True) 

plt.xlabel("Parent Company ->")

plt.title("Parent companies with their respective no. of acquisitions")

plt.show()
# Most and least dominant business sectors

plt.figure(figsize = (8,8))

buss = data['Business'].value_counts(sort = True)

buss[:10].plot.bar(grid = True) 

plt.xlabel("Businesses ->")

plt.title("Most dominant businesses")

plt.show()

plt.figure(figsize = (8,8))

buss[::-140].plot.bar(grid = True) 

plt.xlabel("Businesses ->")

plt.title("Least dominant businesses")

plt.show()
import plotly.offline as py                #visualization

py.init_notebook_mode(connected = True)    #visualization

import plotly.graph_objs as go             #visualization

import plotly.tools as tls                 #visualization

import plotly.figure_factory as ff         #visualization



data[['Country', 'Value (USD)','ParentCompany', 'Business', 'Company']]

def Scatterplot(par_com,color) :

    tracer = go.Bar(x = data[data["ParentCompany"] == par_com]["Company"],

                    y = data[data["ParentCompany"] == par_com]["Value (USD)"]

                        )

    return tracer



def layout_title(title) :

    layout = go.Layout(dict(title = title,

                            plot_bgcolor  = "rgb(243,243,243)",

                            paper_bgcolor = "rgb(243,243,243)",

                            xaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                         title = "Company",

                                         zerolinewidth = 1,ticklen = 5,gridwidth = 2),

                            yaxis = dict(gridcolor = 'rgb(255, 255, 255)',

                                         title = "Value (in USD)",

                                         zerolinewidth = 1,ticklen = 5,gridwidth = 2),

                            height = 700

                           )

                      )

    return layout



for comp in (set(data.ParentCompany)):

    trace = Scatterplot(comp,'red')

    data1 = [trace]

    layout1  = layout_title(comp)

    fig1 = go.Figure(data = data1,layout = layout1)

    py.iplot(fig1)