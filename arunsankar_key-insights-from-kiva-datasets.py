# Import required packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import re
from datetime import datetime

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

# Read data
folder = "../input/"
loans = pd.read_csv(folder + "kiva_loans.csv", parse_dates=['posted_time', 'disbursed_time', 'funded_time', 'date'], infer_datetime_format=True)
locations = pd.read_csv(folder + "kiva_mpi_region_locations.csv")
themes = pd.read_csv(folder + "loan_theme_ids.csv")
themes_by_region = pd.read_csv(folder + "loan_themes_by_region.csv")

# Function to create USA currency flag
def USD_flag(currency):
    if currency == "USD":
        return 1
    else:
        return 0

# Function to classify gender of the group of borrowers
def classify_genders(x):
    if x==0:
        return "Only Females"
    elif x==1:
        return "Only Males"
    elif x==0.5:
        return "Equal Males and Females"
    elif x<0.5:
        return "More Females"
    elif x>0.5:
        return "More Males"
    
# Initial data processing - features for analysis
loans['percentage_funding'] = loans['funded_amount'] * 100 / loans['loan_amount']
loans['USD'] = loans['currency'].apply(lambda x: USD_flag(x))
#loans.dropna(subset=['borrower_genders'])
loans['borrower_genders'] = loans['borrower_genders'].astype(str)
loans['male_borrowers'] = loans['borrower_genders'].apply(lambda x: len(re.findall(r'\bmale', x)))
loans['female_borrowers'] = loans['borrower_genders'].apply(lambda x: len(re.findall(r'\bfemale', x)))
loans['borrowers_count'] = loans['male_borrowers'] + loans['female_borrowers']
loans['male_borrower_ratio'] = loans['male_borrowers'] / loans['borrowers_count']
loans['gender_class'] = loans['male_borrower_ratio'].apply(lambda x: classify_genders(x))


sectors = loans['sector'].unique()
activities = loans['activity'].unique()
df_temp = loans.groupby('currency')['country'].nunique().sort_values(ascending=False).reset_index().head()

plt.figure(figsize=(6,3))
sns.barplot(x="country", y="currency", data=df_temp, palette=sns.color_palette("Spectral", 5), alpha=0.6)
plt.title("Number of Countries using a currency for loans", fontsize=16)
plt.xlabel("Number of countries", fontsize=16)
plt.ylabel("Currency", fontsize=16)
plt.show()
print("Number of loans using USD as currency: " + str(loans[loans['USD']==1]['USD'].count()))
print("Percentage of overall loans using USD as currency: " + str(round(loans[loans['USD']==1]['USD'].count() * 100 / loans['USD'].count(),2)) + "%")
female_borrowers_only_loans = round(
    100 * loans[loans['male_borrower_ratio']==0]['male_borrower_ratio'].count() / loans['male_borrower_ratio'].count(), 2)

male_borrowers_only_loans = round(
    100 * loans[loans['male_borrower_ratio']==1]['male_borrower_ratio'].count() / loans['male_borrower_ratio'].count(), 2)

male_female_borrowers_loans = round(
    100 - female_borrowers_only_loans - male_borrowers_only_loans, 2)

one_borrowers = round(
    100 * loans[loans['borrowers_count']==1]['borrowers_count'].count() / loans['borrowers_count'].count(), 2)

one_lenders = round(
    100 * loans[loans['lender_count']==1]['lender_count'].count() / loans['lender_count'].count(), 2)

one_female_borrowers = round(
    100 * loans[(loans['female_borrowers']==1) & (loans['borrowers_count']==1)]['female_borrowers'].count() 
    / loans[loans['borrowers_count']==1]['female_borrowers'].count(), 2)

print("% of loans with only female borrowers: " + str(female_borrowers_only_loans) + "%")
print("% of loans with only male borrowers: " + str(male_borrowers_only_loans) + "%")
print("% of loans with both male and female borrowers: " + str(male_female_borrowers_loans) + "%")
print("% of loans with only one borrower: " + str(one_borrowers) + "%")
print("% of loans with only one female borrower: " + str(one_female_borrowers) + "%")
print("% of loans with only one lender: " + str(one_lenders))
plt.figure(figsize=(10,6))
sns.lvplot(x="gender_class", 
           y="term_in_months", 
           data=loans[loans['term_in_months']<=36], 
           palette=sns.color_palette("PiYG", 5))
plt.title("Distribution of term_in_months vs borrower gender", fontsize=16)
plt.xlabel("Borrower gender classes", fontsize=16)
plt.ylabel("Term in months", fontsize=16)
plt.show()
df_country = loans.groupby(['country', 'country_code'])['male_borrower_ratio'].mean().reset_index()

data = [dict(
    type = 'choropleth',
    locations = df_country['country'],
    locationmode = 'country names',
    z = df_country['male_borrower_ratio'],
    text = df_country['country'],
    colorscale = [[0,"rgb(159,51,51)"],
                  [0.2,"rgb(221,66,66)"],
                  [0.5,"rgb	(249,217,217)"],
                  [0.8,"rgb(188,240,255)"],
                  [1,"rgb(26,94,118)"]],
    autocolorscale = False,
    reversescale = True,
    marker = dict(
        line = dict (
            color = 'rgb(180,180,180)',
            width = 0.5
        ) ),
    colorbar = dict(
        autotick = False,
        tickprefix = '',
        title = 'Mean Male Borrower Ratio'),
) ]

layout = dict(
    title = 'Male Borrower Ratio across countries',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'Mercator'
        )
    )
)

fig = dict(data=data, layout=layout )
py.iplot(fig, validate=False, filename='d3-world-map' )
gender_vs_sector = loans.groupby(['sector'])['male_borrower_ratio'].describe().sort_values('mean').reset_index()

plt.figure(figsize=(4, 6))
ax = sns.barplot(x="mean", y="sector", data=gender_vs_sector, palette=sns.color_palette("PiYG", 15), alpha=0.6)

for p in ax.patches:
    width = p.get_width()
    ax.text(p.get_width() + 0.05,
            p.get_y() + p.get_height()/2,
            '{:1.2f}%'.format(width), # * loan_funding_by_gender.shape[0]),
            fontsize=10,
            ha="center") 

plt.title("Sector wise % of male borrowers", fontsize=16)
plt.xlabel("% of male borrowers", fontsize=16)
plt.ylabel("Sector", fontsize=16)
plt.show()
gender_vs_sector = loans.groupby(['activity'])['male_borrower_ratio'].describe().sort_values('mean', ascending=False).reset_index()

plt.figure(figsize=(4, 4))
ax = sns.barplot(x="mean", y="activity", data=gender_vs_sector.head(10), palette=sns.color_palette("PiYG_r", 10), alpha=0.6)

for p in ax.patches:
    width = p.get_width()
    ax.text(p.get_width() + 0.08,
            p.get_y() + p.get_height()/2,
            '{:1.2f}%'.format(width), 
            fontsize=10,
            ha="center") 

plt.title("Top 10 activities with high % of male borrowers", fontsize=16)
plt.xlabel("% of male borrowers", fontsize=16)
plt.ylabel("Activity", fontsize=16)
plt.show()
gender_vs_sector = loans.groupby(['activity'])['male_borrower_ratio'].describe().sort_values('mean').reset_index()

plt.figure(figsize=(4, 4))
ax = sns.barplot(x="mean", y="activity", data=gender_vs_sector.head(10), palette=sns.color_palette("PiYG", 10), alpha=0.6)

for p in ax.patches:
    width = p.get_width()
    ax.text(p.get_width() + 0.005,
            p.get_y() + p.get_height()/2,
            '{:1.2f}%'.format(width), 
            fontsize=10,
            ha="center") 

plt.title("Top 10 activities with low % of male borrowers", fontsize=16)
plt.xlabel("% of male borrowers", fontsize=16)
plt.ylabel("Activity", fontsize=16)
plt.show()
loans['posted_to_funded'] = (loans['funded_time'] - loans['posted_time']).apply(lambda x: round(x.total_seconds() / 86400, 1))
loans['funded_to_disbursement'] = (loans['disbursed_time'] - loans['funded_time']).apply(lambda x: round(x.total_seconds() / 86400, 1))

trace1 = go.Histogram(
    x=loans[loans['posted_to_funded']<50]['posted_to_funded'],
    opacity=0.75,
    name = 'Time from posting to funding'
)
trace2 = go.Histogram(
    x=loans[(loans['funded_to_disbursement']<50) & (loans['funded_to_disbursement'] > -75)]['funded_to_disbursement'],
    opacity=0.75,
    name = 'Time from funding to disbursement'
)

data = [trace1, trace2]
layout = go.Layout(barmode='overlay')
fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='overlaid histogram')
df_country_time = loans.groupby(['country'])['posted_to_funded', 'funded_to_disbursement'].mean().reset_index()

# Create a trace
trace = go.Scatter(
    x = df_country_time['funded_to_disbursement'],
    y = df_country_time['posted_to_funded'],
    mode = 'markers',
    text = df_country_time['country']
)

layout = go.Layout(
    autosize=False,
    width=800,
    height=800, 
    title='Country wise Time2Fund vs Time2Disburse loans',
    xaxis=dict(
        title='Time to Disburse a Loan after funding',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Time to Fund a Loan after posting',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig = go.Figure(data=[trace], layout=layout)

# Plot and embed in ipython notebook!
py.iplot(fig, filename='bubblechart-size')