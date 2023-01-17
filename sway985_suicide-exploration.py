import os

print(os.listdir("../input"))

from IPython.display import display

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# Read in data

data = pd.read_csv("../input/master.csv")



# Number of rows and columns

print(data.shape)



# Display first 5 rows

data.head()
# Dataframe info 

data.info()
# Convert gdp_for_year to dtype int

data['gdp_for_year ($)'] = data[' gdp_for_year ($) '].str.replace(',','').astype(int)

data.drop(' gdp_for_year ($) ', axis=1, inplace=True)



# Descriptive stats of numerical columns

data.describe().round(2)
# Number of unique values in categorical columns

categoricals = data.select_dtypes(['object'])

display(categoricals.nunique())
# Average Country Population Plot

plt.figure(figsize=(15,25))

plt.title('Average Suicides / 100k Population')

ax = sns.barplot(x='suicides/100k pop',y='country',data=data,ci=None)



# Annotate counties           

for p in ax.patches:

    width = p.get_width()

    plt.text(p.get_width(), p.get_y()+.55*p.get_height(),

             '{:1.2f}'.format(width),

             ha='left', va='center')

plt.show()
data_by_country_mean = data.groupby('country').mean()

mean_suicides = data_by_country_mean[['suicides/100k pop']]

mean_suicides.sort_values('suicides/100k pop',ascending=False)[:10]
mean_suicides.sort_values('suicides/100k pop')[:10]
# Group data by year

data_by_year = data.groupby('year').sum()



# Display first and last 5 rows

display(data_by_year.head())

display(data_by_year.tail())
# Remove 2016

data_by_year = data_by_year[:-1]



# Rescale columns betwwen 0 - 1 to visualize on one plot

def rescale(values):

    max_val = max(values)

    min_val = min(values)

    scaled_values = []

    for val in values:

        new_val = (val - min_val) / (max_val - min_val)

        scaled_values.append(new_val)

    return scaled_values

# Apply rescaling function to all columns

rescaled = data_by_year.drop('HDI for year',axis=1).apply(rescale)



# Display first and last 5 rows

display(rescaled.round(2).head())

display(rescaled.tail())
# Plot Global Time Series Data

rescaled.plot(figsize=(10,8))

plt.title('Global Time Series')

plt.show()



# Show correlation heatmap

sns.heatmap(rescaled.corr(),annot=True)

plt.show()
# Create pie charts of suicide numbers and population by category

def pie_chart(dataframe, group_col):

    columns = [group_col, 'suicides_no','population']

    grouped_sum = dataframe[columns].groupby(group_col).sum()

    display(grouped_sum)

    

    fig = plt.figure()



    ax1 = fig.add_axes([0, 0, .65, .65])

    ax1.pie(grouped_sum.population,

            labels=grouped_sum.index,

            autopct='%1.1f%%')

    ax1.set_title('Global Population 1985-2016')



    ax2 = fig.add_axes([.65, 0, .65, .65])

    ax2.pie(grouped_sum.suicides_no,

            labels=grouped_sum.index,

            autopct='%1.1f%%')

    ax2.set_title('Global Suicides 1985-2016')



    plt.show()

    

# Create plots of suicide numbers and population by category

def plot_time_series(dataframe, group_col):

    categories = dataframe[group_col].unique()

    for category in categories:

        df = dataframe[dataframe[group_col] == category][

            [group_col,'year','suicides_no','population']]

        # Exclude 2016

        group_data = df.groupby('year').mean()[:-1]

        group_data.apply(rescale).plot(figsize=(10,2))

        plt.title(category)

        plt.show()
pie_chart(data, 'age')

plot_time_series(data, 'age')
pie_chart(data, 'sex')

plot_time_series(data, 'sex')
pie_chart(data, 'generation')

plot_time_series(data, 'generation')