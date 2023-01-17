from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
""" 

A Python Module to do automated Exploratory Data Analysis and some light weight data prep.

https://github.com/TareqAlKhatib/Lazy-EDA

"""

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display, HTML



numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    

def full_report(df, target_column=None):

	"""Tries to run every possible report on the provided dataframe"""

	display(HTML("<h1>Lazy EDA Report</h1>"))

	

	breakdown_date(df)

	show_dtypes(df)

	plot_nulls(df)

	plot_long_lat(df, target_column)

	if target_column is not None:

		plot_scatter_target(df, target_column)

		plot_hist_target(df, target_column)

		plot_correlations(df, target_column)            



def plot_correlations(df, target_column):

	display(HTML("<h2>Column Data Types</h2>"))

	display(HTML("<p>Below is a plot of the correlation coefficients of the dataframe's numeric columns and the target column</p>"))

	

	num_df = df.select_dtypes(include=numerics)

	del(num_df[target_column])

	num_df.corrwith(df[target_column]).sort_values(ascending=False).plot(

		kind='barh', figsize=(12,12), title="Correlation Coefficient with Target")

	plt.show()

	

def breakdown_date(df):

	"""

	Creates new columns in a dataframe representing the components of a date (year, month, day of year, & week day name)

	"""

	date_cols = df.dtypes[df.dtypes == 'datetime64[ns]'].index

	display(HTML("<h2>Breaking down date columns</h2>"))

	if len(date_cols) > 0:

		display(HTML("<p>The following columns will be broken down into year, month, day of year, and weekday columns</p> <ul>"))

		

		for date_column in date_cols:

			display(HTML("<li>{}</li>".format(date_column)))

			df['{}_year'.format(date_column)] = df[date_column].dt.year

			df['{}_month'.format(date_column)] = df[date_column].dt.month

			df['{}_dayofyear'.format(date_column)] = df[date_column].dt.dayofyear

			df['{}_weekday'.format(date_column)] = df[date_column].dt.weekday_name

		

		display(HTML("</ul>"))

	else:

		display(HTML("<p>No Date columns found to breakdown.</p>"))

		

	return df



def plot_nulls(df):

	"""

	Displays a horizontal bar chart representing the percentage of nulls in each column

	"""

	display(HTML("<h2>Plot Nulls</h2>"))

	

	null_percentage = df.isnull().sum()/df.shape[0]*100

	null_percentage_filtered = null_percentage[null_percentage > 0].sort_values()

	

	if len(null_percentage_filtered) > 0:

		display(HTML("<p>The plot below shows the percentage of NaNs in each column in the dataframe</p>"))

		null_percentage_filtered.plot(kind='barh', figsize=(12,12), title="Plot Null Percentages")

		plt.show()

		

	else:

		display(HTML("<p>The dataframe does not contain any missing data</p>"))

	return null_percentage_filtered



def show_dtypes(df):

	"""Shows the data types of all columns"""

	

	display(HTML("<h2>Column Data Types</h2>"))

	

	# Saving the old display max

	old_max = pd.options.display.max_rows

	pd.options.display.max_rows = len(df.columns)

	

	# Display DTypes

	dtype_df = pd.DataFrame({"Column Name": df.dtypes.index,"DType": df.dtypes.values})

	display(dtype_df)

	

	# Restoring the old display max

	pd.options.display.max_rows = old_max

	

def plot_scatter_target(df, target_column):

	"""Plots a sorted scatter plot of the values in a numerical target column"""

	display(HTML("<h2>Plot Scatter Target</h2>"))

	display(HTML("<p>Below is a sorted scatter plot of the values in the target column</p>"))

	

	plt.scatter(range(df[target_column].shape[0]), np.sort(df[target_column].values))

	plt.xlabel('index', fontsize=12)

	plt.ylabel(target_column, fontsize=12)

	plt.show()



def plot_hist_target(df, target_column):

	display(HTML("<h2>Plot Histogram Target</h2>"))

	display(HTML("<p>Below is a histogram of the values in the target column</p>"))

	

	# Filter 1st and 99th percentiles

	ulimit = np.percentile(df.logerror.values, 99)

	llimit = np.percentile(df.logerror.values, 1)

	df['tempTarget'] = df[target_column]

	df['tempTarget'].ix[df['tempTarget']>ulimit] = ulimit

	df['tempTarget'].ix[df['tempTarget']<llimit] = llimit

	

	# Plot

	plt.figure(figsize=(12,8))

	sns.distplot(df['tempTarget'])

	plt.xlabel(target_column, fontsize=12)

	plt.show()

	

	del[df['tempTarget']]

	

def plot_long_lat(df, target_column):

	if 'latitude' in df.columns.str.lower() and 'longitude' in df.columns.str.lower():

		display(HTML("<h2>Plot longitude/latitude</h2>"))

		display(HTML("<p>Below is a scatter plot of long/lat coordinate in the dataframe</p>"))

		

		plt.figure(figsize=(12,12))

		

		if target_column is None:

			sns.jointplot(x=df.latitude.values, y=df.longitude.values, size=10)

		else:

			df['tempTarget'] = (df['logerror'] - df['logerror'].min())/(df['logerror'].max() - df['logerror'].min())

			plt.scatter(x=df.latitude.values, y=df.longitude.values, c=df['tempTarget'].values)

			del(df['tempTarget'])

		plt.ylabel('Longitude', fontsize=12)

		plt.xlabel('Latitude', fontsize=12)

		plt.show()

train_df = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])

prop_df = pd.read_csv("../input/properties_2016.csv", dtype={

    'hashottuborspa': 'object', 

    'propertycountylandusecode': 'object',

    'propertyzoningdesc': 'object',

    'fireplaceflag': 'object',

    'taxdelinquencyflag': 'object'

})



train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')
full_report(train_df, target_column='logerror')