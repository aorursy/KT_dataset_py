

import pandas as pd

import numpy as np

#from matplotlib import pyplot as plt

#plt.style.use('ggplot')



#import seaborn as sns # for making plots with seaborn

#color = sns.color_palette()

#sns.set(rc={'figure.figsize':(25,15)})



import plotly

# connected=True means it will download the latest version of plotly javascript library.

plotly.offline.init_notebook_mode(connected=True)

import plotly.graph_objs as go



import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/googleplaystore.csv')

df.info()

print("data frame has {} rows and {} columns".format(df.shape[0],df.shape[1]))
df.head()
# Convert the 'Last Updated' to datetime format and save it to a new column

# Keep the original datetime column in case the format is not consistent

df['last_updated_date'] = pd.to_datetime(df['Last Updated'], format = '%B %d, %Y', errors='coerce')



# Check if there's any NA value caused by the datetime convert

df[df['last_updated_date'].isnull()]
life_array = df[df.App == 'Life Made WI-Fi Touchscreen Photo Frame'].values

new_life_list = np.insert(life_array, 1, 'unknown').tolist()

df.loc[10472] = new_life_list[:-1]
# Convert to datetime again and check if there's any NA values cased by the transform.

df['last_updated_date'] = pd.to_datetime(df['Last Updated'], format = '%B %d, %Y', errors='coerce')

df[df['last_updated_date'].isnull()]
# Convert 'Reviews' column to integer type

df['Reviews'] = df['Reviews'].astype('int')

df[df['Reviews'].isnull()]
# Convert 'Price' column to float type

df['Price'] = df['Price'].str.replace("$",'').astype('float')

df[df['Price'].isnull()]
# Convert 'Installs' column to float type

df['Installs'] = df['Installs'].str.replace("+",'').str.replace(",",'').astype('int')

df[df['Installs'].isnull()]
# Drop invalid ratings and convert to float type

df = df[~df['Rating'].isnull()]

df['Rating'] = df['Rating'].astype('float')

df[df['Rating'].isnull()]
#drop duplicated rows in the dataset and the 'Last Updated' column is no longer needed

df_unique = df.drop_duplicates().drop('Last Updated', axis = 1)

print("After dropping duplicated rows, there are {} rows and {} columns". format(df_unique.shape[0],df_unique.shape[1]))

print(f"There are {len(df_unique['App'].unique())} unique app names")
print(f"There are still {df_unique.shape[0]-len(df_unique['App'].unique())} rows with duplicated app names")
# Make a subset of the dataframe with duplicated app names

dup_app_list = [a for a in df_unique[df_unique.duplicated('App')]['App']]

dup_app_df = df_unique[df_unique['App'].isin(dup_app_list)].sort_values('App')

dup_app_df.head(20)
#Take the most recent updated reocord

dup_app_df = dup_app_df[dup_app_df.last_updated_date == dup_app_df.groupby('App')['last_updated_date'].transform('max')]

dup_app_df.shape
dup_app_df.groupby('App').apply(lambda x: x.nunique()).max()
# We can groupby the duplicated categorical columns and calculate average values of

# Rating, Reviews, Installs to replace the original values

method = 'mean'

#method = 'max'

#method = 'median'

dup_app_df[['Rating','Reviews','Installs']] = (dup_app_df

                                               .groupby(['App','Category','Current Ver'])['Rating','Reviews','Installs']

                                               .transform(method))
# Calculate how many times of each category occurs in the same App name

dup_app_df['category_count'] = (dup_app_df

                                        .groupby(['App','Category'])['Category']

                                        .transform('count'))
# Calculate the count of unique versions of the same App name

dup_app_df['unique_version_count'] = (dup_app_df

                                        .groupby(['App'])['Current Ver']

                                        .transform(lambda x: x.nunique()))
dup_app_df[dup_app_df['unique_version_count']>1]
#drop the older version

dup_app_df = dup_app_df.drop(index = 489)
dup_app_df = dup_app_df.sort_values(['App','category_count'], ascending = [True, False])

dup_app_df = dup_app_df.drop_duplicates(subset=['App'],keep = 'first')

dup_app_df.head(10)
dup_app_df = dup_app_df.drop(['category_count','unique_version_count'], axis = 1)

print(f"Duplicated App dataframe now have {dup_app_df.shape[0]} rows and {dup_app_df.shape[1]} columns after cleaning")

print(f"It has {dup_app_df.App.nunique()} unique App names")
no_dup_df = df_unique[~df_unique['App'].isin(dup_app_list)].sort_values('App')

final_df = pd.concat([no_dup_df,dup_app_df],axis = 0)



print(f"Duplicated App dataframe now have {final_df.shape[0]} rows and {final_df.shape[1]} columns after cleaning")

print(f"It has {final_df.App.nunique()} unique App names")
final_df['year'] = pd.DatetimeIndex(final_df['last_updated_date']).year

final_df['month'] = pd.DatetimeIndex(final_df['last_updated_date']).month

final_df['day'] = pd.DatetimeIndex(final_df['last_updated_date']).day

final_df['gross_revenue'] = final_df['Installs']*final_df['Price']
final_df[['Rating']].describe()
top_apps_index = final_df.groupby('Category')['gross_revenue'].nlargest(3).index.get_level_values(1)

top_three_app = final_df.loc[top_apps_index,['App','Category','gross_revenue']]

top_three_app.sort_values(['Category','gross_revenue'], ascending=[True, False])
number_of_apps_in_category = top_three_app.groupby('Category')['gross_revenue'].mean().sort_values(ascending=False)

number_of_apps_in_category = number_of_apps_in_category[number_of_apps_in_category>0]



data = [go.Bar(

        x = number_of_apps_in_category.index,

        y = number_of_apps_in_category.values,

        hovertext=top_three_app.groupby('Category')['App'].unique().tolist()

    

)]



plotly.offline.iplot(data, filename='top_three_apps_category')