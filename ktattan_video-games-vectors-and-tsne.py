%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn.manifold

sns.set_style("darkgrid")
df = pd.read_csv('../input/vgsales.csv',usecols=['Name','Platform','Year','Genre','Publisher','Global_Sales'])

df.head(10)
def ohe_features_normalize_sales(data,cols):

    new_data = pd.get_dummies(data,columns=cols)

    new_data.dropna(inplace=True)

    new_data.reset_index(drop=True,inplace=True)

    new_data['Global_Sales'] = new_data['Global_Sales'] / new_data.groupby('Year')['Global_Sales'].transform('sum')

    new_data['Year'] = new_data['Year'].astype(int) # convert year to int rather than float

    return new_data
# choose what columns we want to OHE

use_cols = ['Platform','Genre','Publisher']

df_dummies = ohe_features_normalize_sales(df,use_cols)

df_dummies.head(10)
df_dummies.shape
class Plot:

    """

    A class that takes in a dataframe and groups by a columns and sums by another column.

    It then takes that dict to make a seaborn plot.

    We can specify the type of plot, 'pointplot' or 'barplot' through *plot_style*

    """

    def __init__(self, data, group_col, sum_col, plot_style, n_largest = None):

        self.data = data

        self.group_col = group_col

        self.sum_col = sum_col

        self.plot_style = plot_style

        self.n_largest = n_largest

        

    # Transform dataframe into grouped + summed dataframe (e.g. Group by Year and Sum all the sales in that year)

    def get_new_dataframe(self):

        d = dict(self.data.groupby([self.group_col])[self.sum_col].sum())

        d = pd.DataFrame.from_dict(d,orient='index')

        d = d.reset_index()

        d.columns = [self.group_col, self.sum_col]

        if self.n_largest:

            d = d.nlargest(n=self.n_largest,columns=self.sum_col)

        return d

    

    # Plot all the data in the new data frame

    def get_plot(self):

        d = self.get_new_dataframe()

        if self.plot_style == 'pointplot':

            g = sns.pointplot(x=self.group_col, y=self.sum_col, data=d);

            g.xlabel(self.group_col)

            g.ylabel(self.sum_col)

        elif self.plot_style == 'barplot':

            g = sns.barplot(x=self.group_col, y=self.sum_col, data=d);

        for item in g.get_xticklabels():

            item.set_rotation(80)

        return g
g = Plot(df, 'Year', 'Global_Sales', 'pointplot')

g.get_plot()
g = Plot(df, 'Platform', 'Global_Sales', 'barplot')

g.get_plot()
g = Plot(df, 'Genre', 'Global_Sales', 'barplot')

g.get_plot()
g = Plot(df, 'Publisher', 'Global_Sales', 'barplot', n_largest=10)

g.get_plot()
top_3_consoles = df[(df['Platform'] == 'Wii') | (df['Platform'] == 'PS3') | (df['Platform'] == 'X360')]

# group by Genre and Platform and sum by Global_Sales

genre_platform_sales = top_3_consoles.groupby(['Genre','Platform'])['Global_Sales'].sum()

genre_platform_sales.unstack().plot(kind='bar',stacked=True,  colormap='Blues', grid=False, figsize=(13,5));

plt.title('Stacked Bar Plot of Sales per Genre for 3 Platforms', fontsize=15)

plt.xlabel('Genre', fontsize=15)

plt.ylabel('Global_Sales', fontsize=15)

plt.xticks(fontsize=12,rotation=70);
# columns we want to use from dataframe

cols_to_use = list(df_dummies.columns)

cols_to_use.remove('Name') # this is the label
# transform dataframe to matrix. Each row is a game (observation), each column is a feature

matrix = df_dummies.as_matrix(columns=cols_to_use)

matrix
matrix.shape # everything but the 'name' label is in the matrix
# this make take 5 minutes

tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)

matrix_2d = tsne.fit_transform(matrix)
df_tsne = pd.DataFrame(matrix_2d)

df_tsne['Name'] = df_dummies['Name']

df_tsne.columns = ['x','y', 'Name']

# rearrange columns

cols = ['Name','x','y']

df_tsne = df_tsne[cols]

# show the 2D coordinates of the TSNE output

df_tsne.head(10)
g = df_tsne.plot.scatter("x", "y", s=10, figsize=(20, 12), fontsize=20)

g.set_ylabel('Y',size=20)

g.set_xlabel('X',size=20)
class PlotTsneRegion:

    def __init__(self, data, x_bounds, y_bounds, rand_points=None):

        self.data = data

        self.x_bounds = x_bounds

        self.y_bounds = y_bounds

        self.rand_points = rand_points

        

    def get_slice(self):

        slice = self.data[

            (self.x_bounds[0] <= self.data.x) &

            (self.data.x <= self.x_bounds[1]) & 

            (self.y_bounds[0] <= self.data.y) &

            (self.data.y <= self.y_bounds[1])

        ]

        return slice

    

    def plot_region(self):

        slice = self.get_slice()

        # sample a fraction of rand_points of *slice* incase region is too dense with points

        if self.rand_points:

            slice = slice.sample(frac=self.rand_points)

        ax = slice.plot.scatter("x", "y", s=35, figsize=(10, 8))

        for i, point in slice.iterrows():

            ax.text(point.x + 0.02, point.y + 0.02, point.Name, fontsize=11)
x_bounds, y_bounds = (80,90), (-15,0)

region = PlotTsneRegion(df_tsne,x_bounds=x_bounds, y_bounds=y_bounds, rand_points=0.6)

region.plot_region()
df[df.Name.isin(list(region.get_slice()['Name']))].head(10)
x_bounds,y_bounds = (65,75),(-30,-15)

region = PlotTsneRegion(df_tsne,x_bounds=x_bounds, y_bounds=y_bounds, rand_points=0.3)

region.plot_region()
df[df.Name.isin(list(region.get_slice()['Name']))].head(10)
x_bounds,y_bounds = (-80,-75),(-10,5)

region = PlotTsneRegion(df_tsne,x_bounds=x_bounds, y_bounds=y_bounds, rand_points=0.4)

region.plot_region()
df[df.Name.isin(list(region.get_slice()['Name']))].head(10)
x_bounds,y_bounds = (14,24),(-2,15)

region = PlotTsneRegion(df_tsne,x_bounds=x_bounds, y_bounds=y_bounds, rand_points=0.3)

region.plot_region()
df[df.Name.isin(list(region.get_slice()['Name']))].head(10)