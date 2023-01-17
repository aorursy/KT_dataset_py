import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.set(color_codes = True)
import plotly.express as px
import plotly.graph_objects as go

kiva = pd.read_csv('kiva_loans.csv')
ug = kiva[kiva['country'] == 'Uganda'].reset_index()
ug.head(2)
ug.info()
ug.shape
ug.isna().sum() # Missing variables as proportinn of total: 11.4% funded_time; 25.8% tags >> too high.
ug.duplicated().sum()
ug.drop('tags', axis = 1,inplace = True) # Drop the tags column
ug.shape
ug.describe()
ug.describe(include = 'O') # need to deal with multiple currencies
ug['currency'].unique()
ug['currency'].nunique()
ug['currency'].value_counts()
ug.groupby('currency').sum().reset_index() # significant amount disbersed in USD
sector = ug_sector['sector'] 
loan = ug_sector['loan_amount'] 
fund = ug_sector['funded_amount'] 
lender = ug_sector['lender_count']
ug_sector = ug.groupby('sector')['loan_amount', 'lender_count', 'funded_amount'].sum().sort_values(by = 'loan_amount', ascending = False).reset_index()
ug_sector
kiva['country'].unique()
kiva['country'].nunique()
kiva['avg_loan_by_lender'] = kiva['loan_amount'] / kiva['lender_count']
kiva['avg_loan_by_lender']
kiva.groupby('country')['loan_amount', 'lender_count'].sum().sort_values(by = 'loan_amount', ascending = False).reset_index().head(20)
# Rwanda has fewer lenders but higher loan amount despite 
# being a much smaller country than Uganda while Kenya has over 2x more loans that uganda
ug.index = pd.to_datetime(ug['funded_time'])

fund_time = ug['funded_time'].resample('w').count().to_frame()
fund_time.columns  = ['Frequency']
fig = go.Figure()
fig.add_trace(go.Scatter(x=fund_time.index, y=fund_time.Frequency,
                    mode='lines',
                    name='lines'))
fig.update_layout(
    title='Loans Issued of Over Time in Uganda (weekly)',
    title_x=0.5,
    yaxis_title = 'No. of loans',
    xaxis_title = 'Timeline')
fig.show()
rw = kiva[kiva['country'] == 'Rwanda'].reset_index()

rw.index = pd.to_datetime(rw['funded_time'])

fund_time = rw['funded_time'].resample('w').count().to_frame()
fund_time.columns  = ['Frequency']
fig = go.Figure()
fig.add_trace(go.Scatter(x=fund_time.index, y=fund_time.Frequency,
                    mode='lines',
                    name='lines'))
fig.update_layout(
    title='Loans Issued of Over Time in Rwanda (weekly)',
    title_x=0.5,
    yaxis_title = 'No. of loans',
    xaxis_title = 'Timeline')
fig.show()
ke = kiva[kiva['country'] == 'Kenya'].reset_index()

ke.index = pd.to_datetime(ke['funded_time'])

fund_time = ke['funded_time'].resample('w').count().to_frame()
fund_time.columns  = ['Frequency']
fig = go.Figure()
fig.add_trace(go.Scatter(x=fund_time.index, y=fund_time.Frequency,
                    mode='lines',
                    name='lines'))
fig.update_layout(
    title='Loans Issued of Over Time in Kenya (weekly)',
    title_x=0.5,
    yaxis_title = 'No. of loans',
    xaxis_title = 'Timeline')
fig.show()
ug.info()
px.histogram(ug, x = 'loan_amount', range_x = [0,6000])
px.histogram(ug, x = 'funded_amount', range_x = [0,6000])
px.histogram(ug, x = 'lender_count', range_x = [0,100])
px.histogram(ug, x = 'term_in_months', range_x = [0,50])
ug_activity = ug.groupby('activity')['loan_amount', 'lender_count', 'funded_amount'].sum().sort_values(by = 'loan_amount', ascending = False).reset_index()
ug_activity.head(10)
ug_region = ug.groupby('region')['loan_amount', 'lender_count', 'funded_amount'].sum().sort_values(by = 'loan_amount', ascending = False).reset_index()
ug_region.head(20)
labels = sector
sizes = loan

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  

plt.show()
plt.figure(figsize = (15,15)) 

plt.subplot(2,2,1)
plt.title('Loan Amount by Sector') 
plt.xticks(rotation = 90) 
sns.barplot(x = 'sector', y = 'loan_amount', data = ug_sector, ci = None, color = 'lightblue', estimator= sum)


plt.subplot(2,2,2)
plt.title('Funded Amount by Sector') 
plt.xticks(rotation = 90) 
sns.barplot(x = 'sector', y = 'funded_amount', data = ug_sector, ci = None, color = 'lightblue')


plt.subplot(2,2,3)
plt.title('Lender Count by Sector') 
plt.xticks(rotation = 90) 
sns.barplot(x = 'sector', y = 'lender_count', data = ug_sector, ci = None, color = 'lightblue')

plt.show()
plt.figure(figsize = (15,10))

plt.title('Loan Amount and Funded Amount by Sector', fontsize = 15) 
plt.xlabel('Sector', fontsize = 15) 
plt.ylabel('Loan and Funded Amount', fontsize = 15)

x_indices = np.arange(len(sector)) 
width = 0.3

plt.xticks(ticks = x_indices, labels = sector, rotation = 90)

plt.bar((x_indices + width), fund, width = width, label = 'Funded Amount')

plt.bar(x_indices, loan, width = width, label = 'Loan Amount') 
plt.legend() 
plt.show()
ug_sector['Diff_Loan_Lender_Amounts'] = ug_sector['loan_amount'] - ug_sector['funded_amount']
ug_sector.sort_values(by = 'Diff_Loan_Lender_Amounts', ascending = False) # sectors where loan amount is much greater than funded 
# amount implies unmet demand for credit, namely Retail, Food, Agriculture, Housing
ug_sector_med = ug.groupby('sector')['loan_amount', 'lender_count', 'funded_amount'].median().sort_values(by = 'loan_amount', ascending = False).reset_index()
ug_sector_med # Use median because of skewed distribution?
ug_sector_mean = ug.groupby('sector')['loan_amount', 'lender_count', 'funded_amount'].mean().sort_values(by = 'loan_amount', ascending = False).reset_index()
ug_sector_mean
ug_sector_mean['loan_per_lender'] = ug_sector['loan_amount'] / ug_sector['lender_count']
ug_sector_mean.sort_values(by = 'loan_per_lender', ascending = False).reset_index()
loan_mean = ug_sector_mean['loan_amount']
fund_mean = ug_sector_mean['funded_amount'] 
lender_mean = ug_sector_mean['lender_count'].sort_values(ascending = False)
amount_lender_mean = ug_sector_mean['loan_per_lender']
plt.figure(figsize = (15,10))

plt.title('Average - Lender Count & Lender Loan Amount by Sector', fontsize = 15) 
plt.xlabel('Sector', fontsize = 15) 

x_indices = np.arange(len(sector)) 
width = 0.3

plt.xticks(ticks = x_indices, labels = sector, rotation = 90)

plt.bar((x_indices + width), lender_mean, width = width, label = 'Avg. Lender Count') 
plt.bar(x_indices, amount_lender_mean, width = width, label = 'Avg. Lender Amount') 

plt.legend() 
plt.show()