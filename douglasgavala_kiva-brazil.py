import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
sns.set(color_codes = True)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
kiva = pd.read_csv('/kaggle/input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
brazil = kiva[kiva['country'] == 'Brazil'].reset_index(drop = True)
brazil.head(2)
brazil.head(5)
brazil.columns
brazil.shape
brazil.info()
brazil['region'].nunique()
brazil['region'].unique()
brazil.describe()
brazil.duplicated().sum()
brazil.isna().sum()
null  = brazil.isnull().sum().to_frame().reset_index()
null.columns = ['Column', 'Frequency']
null.sort_values('Frequency',inplace=True)
fig = go.Figure()
colors=[' black ']*len(null.Column)
fig.add_trace(go.Bar(y=null.Frequency,x=null.Column,marker_color=colors))
fig.update_layout(
title = 'Distribution of Null Values in Columns',
    title_x=0.5,
    xaxis_title = 'Columns',
    yaxis_title = 'No of missing Values'
)
fig.show()
brazil['region'].value_counts()
loans = brazil.groupby('region')['loan_amount'].sum().sort_values(ascending =False).reset_index().head(9)
loans
loans = brazil.groupby('sector')['loan_amount'].sum().sort_values(ascending =False).reset_index().head(9)
loans
loans = brazil.groupby('activity')['loan_amount'].sum().sort_values(ascending =False).reset_index().head(9)
loans
def gender_lead(gender):
    gender = str(gender)
    if gender.startswith('f'):
        gender = 'female'
    else:
        gender = 'male'
    return gender
brazil['gender_lead'] = brazil['borrower_genders'].apply(gender_lead)
brazil['gender_lead'].nunique()
f = brazil['gender_lead'].value_counts()[0]
m = brazil['gender_lead'].value_counts()[1]
print('{} females ({}%) vs {} males ({}%) got loans'.format(f,round(f*100/(f+m),2),m,round(m*100/(f+m)),2))
gender_lead = brazil.gender_lead.value_counts().to_frame().head(20).reset_index()
gender_lead.columns=['gender_lead','Frequency']
gender_lead
labels = gender_lead.gender_lead
values = gender_lead.Frequency
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.update_layout(
title='Represention of Gender Funded by Kiva Loans In Brazil ',
title_x = 0.2)
fig.show()
fig = go.Figure()
fig.add_trace(go.Box(name='funded amount',y=brazil.funded_amount))

fig.update_layout(
title = 'Boxplot Distribution of Funded amount in brazil',
title_x = 0.5,
yaxis_title='Amount in dollars')
fig.show()
plt.figure(figsize = (10,5))
plt.title('Funded Amount', fontsize = 15)
plt.hist(brazil['funded_amount'], edgecolor = 'k', bins = 15)
xaxis_title = 'Funded Amount',
yaxis_title = 'Frequency',
plt.show()
plt.figure(figsize = (10,5))
plt.title('Distribution of Term in Months', fontsize = 15)
plt.hist(brazil['term_in_months'], edgecolor = 'k', bins = 15)
plt.show()
activity = brazil.activity.value_counts().to_frame().head(20).reset_index()
activity.columns=['Activity','Frequency']
activity
fig = go.Figure()
colors=[' black ']*len(activity.Activity)
fig.add_trace(go.Bar(y=activity.Activity,x=activity.Frequency,orientation='h',marker_color=colors))
fig.update_yaxes(autorange='reversed')
fig.update_layout(
title = 'Top 20 Activities Funded By Kiva',
    title_x=0.5,
    xaxis_title = 'Frequency',
    yaxis_title = 'Activity'
)
fig.show()
repayment_interval = brazil['repayment_interval'].value_counts().to_frame().reset_index()
repayment_interval.columns = ['Repayment_interval','Frequency']
repayment_interval
labels = repayment_interval.Repayment_interval
values = repayment_interval.Frequency
fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
fig.update_layout(
title='Represention of Repayment Intervals In Brazil ',
title_x = 0.2)
fig.show()
sns.scatterplot(x='funded_amount',y='loan_amount',data=brazil);
np.corrcoef(brazil.funded_amount,brazil.term_in_months)
sns.scatterplot(x='funded_amount',y='lender_count',data=brazil);
np.corrcoef(brazil.funded_amount,brazil.lender_count)
count = round(brazil.groupby(['sector'])['loan_amount'].sum().sort_values(ascending=False))
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
count = round(brazil.groupby(['region'])['loan_amount'].sum().sort_values(ascending=False)).head(20)
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
brazil.index = pd.to_datetime(brazil['funded_time'])
fund_time = brazil['funded_time'].resample('w').count().to_frame()
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
themes = pd.read_csv('/kaggle/input/data-science-for-good-kiva-crowdfunding/loan_themes_by_region.csv')
themes_brazil = themes[themes['country'] == 'Brazil']
themes_brazil.head(20)

px.set_mapbox_access_token('pk.eyJ1IjoiZGdhdmFsYSIsImEiOiJja2QxN2h0ZjkxMHF4MnNtdm1zNXBqenZ0In0.T6EaM2miEr6XrTflmfkhFQ')
px.scatter_mapbox(themes_brazil, lat = 'lat', lon = 'lon', color = 'region',size = 'amount', size_max = 15)