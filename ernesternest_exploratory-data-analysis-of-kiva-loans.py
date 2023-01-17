import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import warnings 

warnings.filterwarnings('ignore')

import plotly.graph_objects as go

import plotly.express as px
dfo = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
df = dfo[dfo['country']=='Cambodia'].reset_index(drop = True)
df.head(5)
df.columns
df.shape
df.info()
df.describe()
df.describe(include = 'O')
null  = df.isnull().sum().sort_values(ascending = False).reset_index()

null.columns = ['Column', 'Frequency']

null
fig = go.Figure()

colors=[' #61725f ']*len(null.Column)

fig.add_trace(go.Bar(y=null.Frequency,x=null.Column,marker_color=colors))

fig.update_layout(

title = 'Distribution of Null Values in Columns',

    title_x=0.5,

    xaxis_title = 'Columns',

    yaxis_title = 'No of missing Values'

)

fig.show()
df.drop('tags', axis=1,inplace=True)
df['funded_time'].mode()
df['funded_time'].fillna(df['funded_time'].mode(), inplace = True)
df['use'].fillna(df['use'].mode(), inplace = True)
df['borrower_genders'].fillna(df['borrower_genders'].mode(), inplace = True)
df.dropna(inplace = True)
df.isna().sum()
fig = go.Figure()

fig.add_trace(go.Box(name='funded amount',y=df.funded_amount))



fig.update_layout(

title = 'Boxplot Distribution of Funded amount in Cambodia',

title_x = 0.5,

yaxis_title='Amount in dollars')

fig.show()
fig = go.Figure()

fig.add_trace(go.Histogram(x=df.funded_amount, xbins=dict(start=0,end=5000)))

fig.update_layout(

    xaxis_title = 'Funded Amount',

    yaxis_title = 'Frequency',

    title = 'Histogram of Funded Amount',

    title_x = 0.3

)

fig.show()
fig = go.Figure()

fig.add_trace(go.Box(name='loan amount',y=df.loan_amount))

fig.update_layout(

title = 'Boxplot Distribution of Loan amount in Cambodia',

title_x = 0.5,

yaxis_title='Amount in dollars')

fig.show()
fig = go.Figure()

fig.add_trace(go.Histogram(x=df.loan_amount, xbins=dict(start=0,end=3000)))

fig.update_layout(

    xaxis_title = 'Loan Amount',

    yaxis_title = 'Frequency',

    title = 'Histogram of Loan Amount',

    title_x = 0.5

)

fig.show()
fig = go.Figure()

fig.add_trace(go.Box(name='term in months',y=df.term_in_months))

fig.update_layout(

title = 'Boxplot Distribution of term in months',

title_x = 0.5,

yaxis_title='months')

fig.show()
fig = go.Figure()

fig.add_trace(go.Histogram(x=df.term_in_months, xbins=dict(start=0,end=100)))

fig.update_layout(

    xaxis_title = 'Months',

    yaxis_title = 'Frequency',

    title = 'Histogram of Tern In Months',

    title_x = 0.3

)

fig.show()
fig = go.Figure()

fig.add_trace(go.Box(name='Lender Count',y=df.lender_count))

fig.update_layout(

title = 'Boxplot Distribution of Lender Count',

title_x = 0.5,

yaxis_title='No of Lenders')

fig.show()
fig = go.Figure()

fig.add_trace(go.Histogram(x=df.lender_count,xbins=dict(start=0,end=100)))

fig.update_layout(

    xaxis_title = 'Lender Count',

    yaxis_title = 'Frequency',

    title = 'Histogram of Lender Count',

    title_x = 0.3

)

fig.show()
activity = df.activity.value_counts().to_frame().head(20).reset_index()

activity.columns=['Activity','Frequency']
activity
fig = go.Figure()

colors=[' #61725f ']*len(activity.Activity)

fig.add_trace(go.Bar(y=activity.Activity,x=activity.Frequency,orientation='h',marker_color=colors))

fig.update_yaxes(autorange='reversed')

fig.update_layout(

title = 'Top 20 Activities Funded By Kiva',

    title_x=0.5,

    xaxis_title = 'Frequency',

    yaxis_title = 'Activity'

)

fig.show()
sector = df.sector.value_counts().to_frame().head(20).reset_index()

sector.columns=['Sector','Frequency']
sector
fig = go.Figure()

colors=[' #61725f ']*len(sector.Sector)

fig.add_trace(go.Bar(y=sector.Sector,x=sector.Frequency,orientation='h',marker_color=colors))

fig.update_yaxes(autorange='reversed')

fig.update_layout(

title = 'Top 20 Sectors Funded By Kiva',

    title_x=0.5,

    xaxis_title = 'Frequency',

    yaxis_title = 'sector'

)

fig.show()
labels = sector.Sector

values = sector.Frequency

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_layout(

title='Represention of Sectors Funded by Kiva Loans In Cambodia ',

title_x = 0.2)

fig.show()
region_list = []

for region in df.region.values:

    if str(region) != "nan":

        region_list.extend( [lst.strip() for lst in region.split(",")] )

temp_data = pd.Series(region_list).value_counts()
Region=temp_data.to_frame().reset_index()

Region.columns=['Region','Frequency']

region = Region.head(20)

region
fig = go.Figure()

colors=[' #61725f ']*len(region.Region)

fig.add_trace(go.Bar(y=region.Region,x=region.Frequency,orientation='h',marker_color=colors))

fig.update_yaxes(autorange='reversed')

fig.update_layout(

title = 'Top 20 Regions Funded By Kiva',

    title_x=0.5,

    xaxis_title = 'Frequency',

    yaxis_title = 'Region'

)

fig.show()
dfo1 = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')

dfl = dfo1[dfo1['country']=='Cambodia'].reset_index()

dfl = dfl[['region','lat','lon']]

dfl.columns = ['Region','lat','lon']

dfl.at[1,['Region','lat','lon']]=['Battambang',13.0957,103.2022]

dfl.at[4,['lat','lon']] = [11.4650,104.52085]

dfl.dropna(inplace=True)
dfl.set_index('Region', inplace=True)
dfs = [Region,dfl]
from functools import reduce

dfc = reduce(lambda left,right:pd.merge(left,right,on='Region'),dfs)

dfc
px.set_mapbox_access_token('pk.eyJ1IjoiZXJuZXN0NDA0IiwiYSI6ImNrOWlmOG1idjAwdTEzbHBjdnB5MzFndXEifQ.i_TnCFGI64JcmoA0caIhgQ')
px.scatter_mapbox(dfc, lat = 'lat', lon = 'lon', color = 'Region', size = 'Frequency', size_max = 15, title = 'Mapbox Showing Different Regions vs Number of Loan They Recieved')

gender_list=[]

for gender in df.borrower_genders.values:#Goes through every row in the column

    if str(gender) != 'nan':# skips null cells

        gender_list.extend([lst.strip() for lst in gender.split(',')])

        #In the cell,we strip() remove white spaces eg " kenya " and split comma separated values into individual elements

        #Using extend

temp_data = pd.Series(gender_list).value_counts()

gender = temp_data.to_frame().head(20).reset_index()

gender.columns = ['Gender', 'Frequency']

gender
labels = gender.Gender

values = gender.Frequency

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_layout(

title='Represention of Genders Funded by Kiva Loans In Cambodia ',

title_x = 0.2)

fig.show()
repayment_interval = df['repayment_interval'].value_counts().to_frame().reset_index()

repayment_interval.columns = ['Repayment_interval','Frequency']
repayment_interval
labels = repayment_interval.Repayment_interval

values = repayment_interval.Frequency

fig = go.Figure(data=[go.Pie(labels=labels, values=values)])

fig.update_layout(

title='Represention of Repayment Intervals In Cambodia ',

title_x = 0.2)

fig.show()
sns.scatterplot(x='funded_amount',y='loan_amount',data=df);

np.corrcoef(df.funded_amount,df.loan_amount)
sns.scatterplot(x='funded_amount',y='term_in_months',data=df);
np.corrcoef(df.funded_amount,df.term_in_months)
# sns.scatterplot(x='disbursed_to_funded_time',y='funded_amount',data=df)
sns.scatterplot(x='funded_amount',y='lender_count',data=df);
np.corrcoef(df.funded_amount,df.lender_count)
sns.scatterplot(y='loan_amount',x='term_in_months',data=df);
np.corrcoef(df.loan_amount,df.term_in_months)
sns.scatterplot(x='loan_amount',y='lender_count',data=df);
np.corrcoef(df.loan_amount,df.lender_count)
sns.scatterplot(x='term_in_months',y='lender_count',data=df);
np.corrcoef(df.term_in_months,df.lender_count)
count = round(df.groupby(['sector'])['loan_amount'].sum().sort_values(ascending=False))

fig = go.Figure()

fig.add_trace(go.Bar(y=count.index,x=count.values,orientation='h'))

fig.update_yaxes(autorange='reversed')

fig.update_layout(

title = 'Top Sectors By Total Loan Amount Recieved',

    title_x=0.5,

    xaxis_title='loan amount in Dollar',

    yaxis_title='Sector'

)

fig.show()
count = round(df.groupby(['sector'])['loan_amount'].mean().sort_values(ascending=False))

fig = go.Figure()

fig.add_trace(go.Bar(y=count.index,x=count.values,orientation='h'))

fig.update_yaxes(autorange='reversed')

fig.update_layout(

title = 'Top Sectors By Average Loan Amount Recieved',

    title_x=0.5,

    xaxis_title='loan amount in Dollars',

    yaxis_title='Sector'

)

fig.show()
count = round(df.groupby(['region'])['loan_amount'].sum().sort_values(ascending=False)).head(20)

fig = go.Figure()

fig.add_trace(go.Bar(y=count.index,x=count.values,orientation='h'))

fig.update_yaxes(autorange='reversed')

fig.update_layout(

title = 'Top Region By Total Loan Amount Recieved',

    title_x=0.5,

    xaxis_title='loan amount in Dollar',

    yaxis_title='Region'

)

fig.show()
count = round(df.groupby(['region'])['loan_amount'].mean().sort_values(ascending=False)).head(20)

fig = go.Figure()

fig.add_trace(go.Bar(y=count.index,x=count.values,orientation='h'))

fig.update_yaxes(autorange='reversed')

fig.update_layout(

title = 'Top Region By Average Loan Amount Recieved',

    title_x=0.5,

    xaxis_title='loan amount in Dollar',

    yaxis_title='Region'

)

fig.show()
count = round(df.groupby(['repayment_interval'])['loan_amount'].sum().sort_values(ascending=False)).head(20)

fig = go.Figure()

fig.add_trace(go.Bar(y=count.index,x=count.values,orientation='h'))

fig.update_yaxes(autorange='reversed')

fig.update_layout(

title = 'Repayment Interval By Total Loan Amount',

    title_x=0.5,

    xaxis_title='loan amount in Dollar',

    yaxis_title='repayment interval'

)

fig.show()
count = round(df.groupby(['repayment_interval'])['loan_amount'].mean().sort_values(ascending=False)).head(20)

fig = go.Figure()

fig.add_trace(go.Bar(y=count.index,x=count.values,orientation='h'))

fig.update_yaxes(autorange='reversed')

fig.update_layout(

title = 'Repayment Interval By Average Loan Amount',

    title_x=0.5,

    xaxis_title='loan amount in Dollar',

    yaxis_title='repayment interval'

)

fig.show()
df.index = pd.to_datetime(df['funded_time'])

fund_time = df['funded_time'].resample('w').count().to_frame()

fund_time.columns  = ['Frequency']

fig = go.Figure()

fig.add_trace(go.Scatter(x=fund_time.index, y=fund_time.Frequency,

                    mode='lines',

                    name='lines'))

fig.update_layout(

    title='Loan Trends of Over Time(weekly)',

    title_x=0.5,

    yaxis_title='No. of loans',

    xaxis_title='Timeline'



)

fig.show()

country_rank = dfo['country'].value_counts().to_frame().head(20).reset_index()

country_rank.columns=['country','Number']
country_rank.head(5)
rank = country_rank.index[country_rank.country == 'Cambodia'].tolist()

rank = rank[0]
fig = go.Figure()

colors=[' #61725f ']*len(country_rank.country)

colors[rank]= 'crimson'

fig.add_trace(go.Bar(y=country_rank.country,x=country_rank.Number,orientation='h',marker_color=colors))

fig.update_yaxes(autorange='reversed')

fig.update_layout(

title = 'Number of Loans in Cambodia compared to other countries',

    title_x=0.5,

)

fig.show()
country_fund = dfo.groupby('country').sum()['loan_amount'].sort_values(ascending = False).to_frame().reset_index()

country_fund.columns = ['Country', 'Total_amount']

country_fund.head(10)
fig = px.choropleth(country_fund, 

                    locations="Country", 

                    locationmode = "country names",

                    color="Total_amount",

                    

                    hover_name="Country"

                   )

fig.update_layout(

    title_text = 'Top Countries By Total Amount Loaned',

    title_x = 0.5,

    geo=dict(

        showframe = False,

        showcoastlines = False,

    ))

    

fig.show()