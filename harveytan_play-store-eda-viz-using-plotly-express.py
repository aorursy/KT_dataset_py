# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")

data_reviews = pd.read_csv(r"/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv")
data.sample(5)
data.info()
data.shape[0] - data.count()

# the below display the number of missing values in each column
list(data['Installs'].unique())

# we can approximate the number of installs to be the middle value of two categories

# for example, if an app belong to the '50+' category under the column 'Installs', we approximate the number of installs to be 75 since the next highest category is 100+

# while not super accurate, it is 'sufficiently' accurate
# creating a dictionary for us to replace the category to a numerical value

# if an app belong to the '50+' category, we approximate the number of installs to be 75 since the next highest category is 100+

install_replacer = {

                    'Free' : 0,

                    '0' : 0,

                    '0+' : 1,

                    '5+' : 8,

                    '1+' : 3,

                    '10+' : 30,

                    '500+' : 750,

                    '100+' : 300,

                    '50+' : 75,

                    '500,000,000+' : 750000000,

                    '1,000+' : 3000,

                    '1,000,000,000+' : 1000000000, # this is the only category where there is no mid-point value for us to approximate

                    '100,000,000+' : 300000000,

                    '5,000+' : 7500,

                    '10,000,000+' : 30000000,

                    '1,000,000+' : 3000000,

                    '50,000+' : 75000,

                    '100,000+' : 300000,

                    '50,000,000+' : 75000000,

                    '5,000,000+' : 7500000,

                    '500,000+' : 750000,

                    '10,000+' : 30000

                    }
data['approx_number_of_installs'] = data['Installs'].replace(install_replacer)

data['approx_number_of_installs'] = data['approx_number_of_installs'].astype('int64')
data['Price'].unique() 

# two problematic data points ---> 'Everyone' and '0'
data['price_revised'] = data['Price'].replace({'Everyone' : '$0'}) # replace 'Everyone' with '$0'

data['price_revised'] = data['price_revised'].str.strip('$')

data['price_revised'] = data['price_revised'].astype('float64') # change the datatype to something more appropriate for this column 
data['Reviews'].replace({'3.0M' : 3000000}, inplace=True)

data['Reviews'] = data['Reviews'].astype('int64') # change the datatype to something more appropriate for this column 
data_final = data.drop(['Size','Installs','Last Updated','Current Ver','Android Ver'], axis=1) 

data_final
data_final.info()
data_reviews.head() 

# This file contains the first 'most relevant' 100 reviews for each app
data_reviews.info()
data_reviews['Sentiment'].unique()
data_reviews['Sentiment_Value'] = data_reviews['Sentiment'].replace({'Positive' : 1,

                                                                     np.NaN : 0,

                                                                     'Neutral' : 0,

                                                                     'Negative' : -1

                                                                    })

data_reviews['Sentiment_Value'] = data_reviews['Sentiment_Value'].astype('float64')
data_reviews_group = data_reviews.groupby('App').mean().reset_index().round(2) 

data_reviews_group.rename(columns={'Sentiment_Polarity' : 'Average_Sentiment_Polarity',

                                   'Sentiment_Subjectivity' : 'Average_Sentiment_Subjectivity',

                                   'Sentiment_Value' : 'Average_Sentiment_Value'

                                  }, inplace=True)

data_reviews_group
# The final dataset ('data_f') is a combination of the main dataset and the review dataset

data_f = data_final.merge(data_reviews_group, left_on='App', right_on='App', how='left')

data_f
data_f['Content Rating'].unique()
data_f['Category'].unique() # there is an incorrect data point , i.e. '1.9' - look near the bottom of the results
data_f['Genres'].unique() # there is an incorrect data point , i.e. 'February 11, 2018' - look near the bottom of the results
data_f[data_f['Category'] == '1.9']
data_f[data_f['Genres'] == 'February 11, 2018'] # this particular row item ('App'=='Life Made WI-Fi Touchscreen Photo Frame') is problematic where there are several incorrect data points ('Genres','Category', 'Price') 
index_to_drop = data_f[data_f['App'] == 'Life Made WI-Fi Touchscreen Photo Frame'].index[0]

data_f.drop(labels=index_to_drop, axis='index', inplace=True) 
data_f.info()
data_f['Category'] = data_f['Category'].str.title().str.replace("_",' ')

data_f['approx_number_of_installs_in_M'] = data_f['approx_number_of_installs'] / 1000000
data_f.head() # this is the finalized dataset to be used for EDA and Visualization
category_data = data_f.groupby('Category').agg({'Rating':'mean', 'approx_number_of_installs_in_M' : 'sum'}).rename(\

columns={'Rating':'Average Rating', 'approx_number_of_installs_in_M':'Total Number of Installs in Millions (est.)'}).reset_index().round(1)



top_category = category_data.sort_values(by='Total Number of Installs in Millions (est.)', ascending=False).head()

bot_category = category_data.sort_values(by='Total Number of Installs in Millions (est.)', ascending=False).tail()



all_category = list(data_f['Category'].unique())

top_categories = list(top_category['Category'])

not_top_categories = [i for i in all_category if i not in top_categories]



bot_categories = list(bot_category['Category'])

top_bot_categories = top_categories + bot_categories



print('Top 5 categories based on the number of installs (estimated)')

print('\n')

print(top_category.to_string(index=False))



print('\n')

print('===================================')

print('\n')



print('Observations:')

print('#1 Gaming, communication and productivity apps are top 3 most popular type of apps in terms of the number of installs (estimated)')

print('#2 In terms of average rating by category, all the categories scored fairly similarly')
a = pd.crosstab(data_f.Category, data_f.Type, margins=True, normalize='index') * 100 

a.rename(columns=lambda x: x + ' %', inplace=True)



b = pd.crosstab(data_f.Category, data_f.Type)

b.rename(columns=lambda x: f'Number of {x} apps', inplace=True )



c = pd.concat([a.round(1), b], axis=1)



print("Categories with the highest proportion of free apps")

print(c.sort_values('Free %', ascending=False).head())



print('\n')

print('=============')

print('\n')



print("Categories with the lowest proportion of free apps")

print(c.sort_values('Free %', ascending=False).tail())
a = data_f.sort_values(by='Average_Sentiment_Value', ascending=False).head().to_string(index=False)

print('Top 5 Apps with highest average sentiment value')

print('\n')

print(a)
print('High Level Statistics on App Ratings')

print('===================================')

print(data_f.Rating.describe())

# The average rating across all the apps is 4.2
data_subset = data_f[data_f['Category'].isin(top_categories)]

fig = px.sunburst(data_subset, path=['Category','Genres'], values='approx_number_of_installs_in_M',width=750, height=750)

fig.update_layout(title_text='Top 5 Categories - Breakdown of Installs', title_font_size=20)

fig.show()
fig = px.sunburst(data_f, path=['Content Rating','Category'], values='approx_number_of_installs_in_M', width=750, height=750)

fig.update_layout(title_text='Number of Installs by Content Rating', title_font_size=20)

fig.show()
data_f_non_nan = data_f[data_f['Type'].isin(['Free','Paid'])]

fig = px.histogram(data_f_non_nan, x='Rating', marginal='box', color='Type', opacity=0.5)

fig.update_layout(title_text='Distribution of Ratings Across All Apps', title_font_size=20)

fig.update_xaxes(showgrid=False)

fig.update_yaxes(showgrid=False)

fig.show()
data_f_ex_top_cat = data_f[data_f['Category'].isin(not_top_categories)]

fig = px.histogram(data_f_ex_top_cat, x='Rating', marginal='box')

fig.update_layout(title_text='Distribution of Ratings Across All Apps That Are Not In The Top 5 Categories', title_font_size=20)

fig.update_xaxes(showgrid=False)

fig.update_yaxes(showgrid=False)

fig.show()
data_f_top_cat = data_f[data_f['Category'].isin(top_categories)]

fig = px.histogram(data_f_top_cat, x='Rating', marginal='box')

fig.update_layout(title_text='Distribution of Ratings Across All Apps That Are In The Top 5 Categories', title_font_size=20)

fig.update_xaxes(showgrid=False)

fig.update_yaxes(showgrid=False)

fig.show()
data_f_paid_top_cat = data_f[(data_f['Category'].isin(top_categories)) & (data_f['Type'] == 'Paid')]

fig = px.scatter(data_f_paid_top_cat, x='Rating', y ='price_revised', facet_col='Category', color='Category', marginal_x='histogram', labels={'price_revised':'Price'})

fig.update_layout(title_text='Paid Apps in Top Categories - Distribution of Ratings against Price', title_font_size=20)

fig.update_xaxes(showgrid=False)

fig.update_yaxes(showgrid=False)

fig.show()
data_f_free_top_cat = data_f[(data_f['Category'].isin(top_categories)) & (data_f['Type'] == 'Free')]

fig = px.histogram(data_f_free_top_cat, x='Rating', facet_col='Category', color='Category')

fig.update_layout(title_text='Free Apps in Top Categories - Distribution of Rating', title_font_size=20)

fig.update_xaxes(showgrid=False)

fig.update_yaxes(showgrid=False)

fig.show()