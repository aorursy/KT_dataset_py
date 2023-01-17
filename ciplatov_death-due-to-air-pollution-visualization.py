import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set()
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/death-due-to-air-pollution-19902017/death-rates-from-air-pollution.csv')
df.head()
df.isnull().sum()
df = df.fillna('NaN')
#We can see that only country reductions are missing. We will not use the abbreviations, so drop it.
df[df['Code']=='NaN']['Entity'].unique()
df.drop('Code', axis=1, inplace=True)
# make sure that all years meet the same number of times equal to the number of countries
len(df['Entity'].unique())
df_g = df.groupby('Entity').aggregate('count')
df_g.describe()
for uniqe_year in df['Year'].unique():
    print((df[df['Year']==uniqe_year].count()), uniqe_year)
# check emissions
df[df.select_dtypes(include = ['float64']).columns].hist(figsize=(15,10), bins=50)
# check if column "Air pollution (total) (deaths per 100,000)" is equal to the sum of the others
df_dif_of_total_and_sum = df["Air pollution (total) (deaths per 100,000)"] -\
df['Indoor air pollution (deaths per 100,000)'] -\
df['Outdoor particulate matter (deaths per 100,000)'] -\
df['Outdoor ozone pollution (deaths per 100,000)']
# see slight discrepancies
df_dif_of_total_and_sum.hist()
df['Total'] = df['Indoor air pollution (deaths per 100,000)'] + \
df['Outdoor particulate matter (deaths per 100,000)'] + \
df['Outdoor ozone pollution (deaths per 100,000)']
# replace it with the real amount
df.drop('Air pollution (total) (deaths per 100,000)', axis=1, inplace=True)
# visualize the situation in the first and last years
df_1990 = df[df['Year']==1990]
df_1990 = df_1990.reset_index()
df_2017 = df[df['Year']==2017]
df_2017 = df_2017.reset_index()
temp_df = pd.DataFrame(df_1990['Total'])
fig = px.choropleth(temp_df, locations=df_1990['Entity'],
                    color=np.log10(temp_df['Total']),
                    hover_data=['Total'],
                    color_continuous_scale=px.colors.sequential.Plasma,locationmode="country names")
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title_text="Total Air pollution on 1990 Heat Map")
fig.update_coloraxes(colorbar_title="Air pollution",colorscale="reds")
fig.show()
temp_df = pd.DataFrame(df_2017['Total'])
fig = px.choropleth(temp_df, locations=df_2017['Entity'],
                    color=np.log10(temp_df['Total']),
                    hover_data=['Total'],
                    color_continuous_scale=px.colors.sequential.Plasma,locationmode="country names")
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title_text="Total Air pollution on 1990 Heat Map")
fig.update_coloraxes(colorbar_title="Air pollution",colorscale="reds")
fig.show()
df_dif_1990_and_2017 = pd.DataFrame(df_1990['Total'] - df_2017['Total'])
df_dif_1990_and_2017.rename(columns={'Total' : 'Total_dif'}, inplace=True)
# visualize the difference in the situation in the first and last years
df_dif_1990_and_2017['Entity'] = df_2017['Entity']
temp_df = pd.DataFrame(df_dif_1990_and_2017['Total_dif'])
fig = px.choropleth(temp_df, locations=df_dif_1990_and_2017['Entity'],
                    color=np.log10(temp_df['Total_dif'] + abs(min(temp_df['Total_dif']))+0.1),
                    hover_data=['Total_dif'],
                    color_continuous_scale=px.colors.sequential.Plasma,locationmode="country names")
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(title_text="Difference of Total Air pollution on 1990 and 2017 Heat Map")
fig.update_coloraxes(colorbar_title="Difference of Total Air pollution",colorscale="greens")
fig.show()
df.index = df['Year']
df.drop('Year', axis=1, inplace=True)
fig = plt.subplots(figsize=(20,7))
fig = sns.lineplot(x=df.index, y="Total", data=df)
fig = sns.lineplot(x=df.index, y='Indoor air pollution (deaths per 100,000)', data=df)
fig = sns.lineplot(x=df.index, y="Outdoor particulate matter (deaths per 100,000)", data=df)
fig = sns.lineplot(x=df.index, y="Outdoor ozone pollution (deaths per 100,000)", data=df)
plt.show()
df_g = df.groupby('Entity', as_index=False).aggregate('sum')
df_g = df_g.sort_values('Total')
fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(20,7))
sns.barplot(x='Entity' ,y="Total", ax=ax1, data=df_g.head())
sns.barplot(x='Entity', y="Total", ax=ax2, data=df_g.tail())
ax1.set_ylabel("all-time deathss")
ax2.set_ylabel("all-time deaths")
ax1.set_title("countries with the least deaths")
ax2.set_title("countries with the highest number of deaths")
