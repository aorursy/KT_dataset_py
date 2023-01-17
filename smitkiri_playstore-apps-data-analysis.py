import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
sns.set_style('whitegrid')
df = pd.read_csv('../input/googleplaystore.csv')
df.sample(5)
#Keeping apps with type as either free or paid
df = df[(df['Type'] == 'Free') | (df['Type'] == 'Paid')]

#Removing null values
df = df[(df['Android Ver'] != np.nan) & (df['Android Ver'] != 'NaN')]

#Remove anomalies where rating is less than 0 and greater than 5
df = df[df['Rating'] < 5]
df = df[df['Rating'] > 0]

#Convert all sizes to MB and removing 'M' and 'k'
df['Size'] = df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(str(x).replace('k', ''))/1000 if 'k' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(x))

#Remove ' and up' to get the minimum android version
df['Android Ver'] = df['Android Ver'].apply(lambda x: str(x).replace(' and up', '') if ' and up' in str(x) else x)

#Remove '$' from the price to convert it to float
df['Price'] = df['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else x)
df['Price'] = df['Price'].apply(lambda x: float(x))

#Convert number of reviews to int
df['Reviews'] = df['Reviews'].apply(lambda x: int(x))
print('Number of apps in the dataset: ', len(df))
trace = [go.Pie(
    values = df['Category'].value_counts(),
    labels = df['Category'].value_counts().index)]

layout = go.Layout(title = 'Distribution of app categories in the market')

fig = go.Figure(data = trace, layout = layout)
iplot(fig)
print('The average rating of the apps in the dataset is: ',np.mean(df['Rating']))

trace = [go.Histogram(
    x = df['Rating'],
    xbins=dict(start = 0.0, end = 5.0, size = 0.1))]

layout = go.Layout(title = 'Distribution of the ratings of apps in the dataset')

fig = go.Figure(data = trace, layout = layout)
iplot(fig)
most_reviewed = df.sort_values('Reviews', ascending=False).head(100)

print('Average rating of top 100 most reviewed apps: ',np.mean(most_reviewed['Rating']))

trace = [go.Scatter(
    x = most_reviewed['Reviews'],
    y = most_reviewed['Rating'],
    mode = 'markers')]

layout = go.Layout(title = 'Ratings of top 100 most reviewed apps')

fig = go.Figure(data = trace, layout = layout)
iplot(fig)
print('The average rating of the apps in the dataset is: ',np.mean(df['Size']))

trace = [go.Histogram(x = df['Size'])]

layout = go.Layout(title = 'Distribution of the size of apps in the dataset')

fig = go.Figure(data = trace, layout = layout)
iplot(fig)
sns.jointplot('Size', 'Rating', data=df)
plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (16,6))

ax1.hist(df[df['Type'] == 'Free']['Rating'], bins = 50)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_title('Free apps rating')

ax2.hist(df[df['Type'] == 'Paid']['Rating'], bins = 50)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.set_title('Paid apps rating')

plt.show()
trace = [go.Scatter(
    x = df[df['Type'] == 'Paid']['Price'],
    y = df[df['Type'] == 'Paid']['Rating'],
    mode = 'markers')]

layout = go.Layout(title = 'Rating based on price')

fig = go.Figure(data = trace, layout = layout)
iplot(fig)
trace = [go.Scatter(
    x = df[(df['Type'] == 'Paid') & (df['Price'] < 50)]['Price'],
    y = df[df['Type'] == 'Paid']['Rating'],
    mode = 'markers')]

layout = go.Layout(title = 'Rating based on price')

fig = go.Figure(data = trace, layout = layout)
iplot(fig)
df[df['Price'] > 250][['App', 'Category', 'Price']]
top_categories = df['Category'].value_counts()[:6].index
top_apps = df[df['Category'].isin(top_categories)]
top_apps = top_apps[top_apps['Price'] < 50]

trace = [go.Scatter(
    x = top_apps['Price'],
    y = top_apps['Category'],
    mode = 'markers')]

layout = go.Layout(title = 'Pricing in top 6 categories')

fig = go.Figure(data = trace, layout = layout)
iplot(fig)
trace = [go.Pie(
    values = df[df['Type'] == 'Free']['Installs'].value_counts(),
    labels = df[df['Type'] == 'Free']['Installs'].value_counts().index)]

layout = go.Layout(title = 'Installs of free apps')

fig = go.Figure(data = trace, layout = layout)
iplot(fig)
trace = [go.Pie(
    values = df[df['Type'] == 'Paid']['Installs'].value_counts(),
    labels = df[df['Type'] == 'Paid']['Installs'].value_counts().index)]

layout = go.Layout(title = 'Installs of paid apps')

fig = go.Figure(data = trace, layout = layout)
iplot(fig)
x = df[['Content Rating', 'Installs', 'Type']].copy()
x.dropna(inplace = True)
x = pd.DataFrame(x.groupby(['Content Rating', 'Installs'])['Type'].count())
x.reset_index(inplace=True)
x = x.pivot('Installs', 'Content Rating', 'Type')

for y in x:
    x[y] = x[y].apply(lambda z: int(str(z).replace('nan', '0')) if 'nan' in str(z) else z)

fig, ax = plt.subplots(figsize=(12,12))
ax = sns.heatmap(x, annot=True, cmap=sns.light_palette("green"), fmt='.0f')
ax.set_title('Number of installs by content rating')

plt.show()