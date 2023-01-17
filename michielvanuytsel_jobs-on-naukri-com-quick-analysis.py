import numpy as np

import pandas as pd



data_filepath = '../input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv'

df = pd.read_csv(data_filepath)

df.head()
import pandas_profiling

pandas_profile = pandas_profiling.ProfileReport(df, progress_bar=False)

pandas_profile.to_widgets()
# Let's look at the freshness of the data using the crawl timestamp



import plotly.express as px

df['Crawl Timestamp_dt'] = pd.to_datetime(df['Crawl Timestamp']) #Convert to Pandas DateTime

px.box(df, y="Crawl Timestamp_dt", points="all", hover_data=["Uniq Id"])
# Let's now look at categorical columns such as Functional Area



func_area_df = (df['Functional Area'].str.split(' , ', expand=True) #Multiple categories are combined in the column as comma-separated so we first need to split this

     .stack() # We then stack them again as they'd otherwise be represented as separate columns

     .value_counts() # As last transformation we count all the values

    )



# We only want to show the top values, and combine the smaller values together

top_func_area_df = pd.concat([func_area_df[:20], pd.Series(func_area_df[20:].sum(), index=["Others"])])





fig = px.pie(top_func_area_df, values=top_func_area_df.values, names=top_func_area_df.index)

fig.update_traces(textposition='inside', textinfo='percent+label')
# Let's do the same with Industry and Key Skills



industry_df = (df['Industry'].str.split(', ', expand=True) #Multiple categories are combined in the column as comma-separated so we first need to split this

     .stack() # We then stack them again as they'd otherwise be represented as separate columns

     .value_counts() # As last transformation we count all the values

    )



# We only want to show the top values, and combine the smaller values together

top_industry_df = pd.concat([industry_df[:30], pd.Series(industry_df[30:].sum(), index=["Others"])])

px.bar(top_industry_df)
# Let's do the same with Industry and Key Skills



key_skills_df = (df['Key Skills'].str.split('\| ', expand=True) #Multiple categories are combined in the column as comma-separated so we first need to split this

     .stack() # We then stack them again as they'd otherwise be represented as separate columns

     .value_counts() # As last transformation we count all the values

    )



# We only want to show the top values, and combine the smaller values together

top_key_skills_df = pd.concat([key_skills_df[:30], pd.Series(key_skills_df[30:].sum(), index=["Others"])])

fig = px.bar(top_key_skills_df)

fig.update_layout(yaxis_type="log")
%matplotlib inline

from wordcloud import WordCloud

import matplotlib.pyplot as plt



wc = WordCloud(max_words=30, background_color='white', width = 2400, height = 800, min_font_size = 10)



plt.imshow(wc.generate_from_frequencies(df['Role'].astype(str).value_counts()))