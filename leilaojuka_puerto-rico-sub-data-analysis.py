import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
import plotly.graph_objects as go
import plotly.express as px
data = pd.read_csv('/kaggle/input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv')
df = data[data['country']=='Puerto Rico'].reset_index()
df.head(5)
df.tail(5)
df.info()
df.columns
df.describe()
df.shape
df.info()
df.duplicated().sum()
df.drop('index',axis=1,inplace=True)
df
df.isna().sum()
missing  = df.isna().sum().to_frame().reset_index()
missing.columns = ['Column', 'Frequency']
missing.sort_values('Frequency',inplace=True)
fig = go.Figure()
colors=[' #34567f ']*len(missing.Column)
fig.add_trace(go.Bar(y=missing.Frequency,x=missing.Column,marker_color=colors))
fig.update_layout(
title = 'Distribution of Missing Values in Columns',
    title_x=0.5,
    xaxis_title = 'Columns',
    yaxis_title = 'No of missing Values'
)
fig.show()
df.drop('region', axis=1,inplace=True)
puerto = data[data['country'] == 'Puerto Rico'].reset_index(drop = True)
puerto.head(2)
activity_df = puerto.groupby('sector')['loan_amount', 'lender_count', 'funded_amount'].sum()\
         .sort_values(by = 'loan_amount', ascending = False).reset_index().head(10)

activity_df
sector = activity_df['sector']
loan = activity_df['loan_amount']
fund = activity_df['funded_amount']
lender = activity_df['lender_count']
df['sector'].value_counts()
loans = df.groupby('sector')['loan_amount'].sum().sort_values(ascending = False).reset_index().head(10)
loans
plt.figure(figsize = (10,5))

plt.title('Loan Amount by Sector', fontsize = 15)
plt.xlabel('Sector', fontsize = 15)
plt.ylabel('Loan Amount', fontsize = 15)

plt.xticks(rotation = 60)

plt.bar(sector, loan, edgecolor = 'k')

plt.show()
fund = df.groupby('sector')['funded_amount'].sum().sort_values(ascending = False).reset_index()
fund
plt.Figure(figsize = (10,5))

x = np.array(['funded_amount'])
plt.Figure(figsize = (10,5))
plt.hist(x)
plt.show()
df.head(2)
time = df.groupby('sector')['posted_time','disbursed_time'].sum().sort_values(by ='posted_time', ascending = False).reset_index().head(10)
time
pay = df.groupby('sector')['loan_amount','term_in_months'].sum().sort_values(by ='loan_amount', ascending = True).reset_index().head(10)
pay
plt = go.Figure()
plt.add_trace(go.Box(name='term in months',y=df.term_in_months))
plt.update_layout(
title = 'Boxplot Distribution of term in months',
title_x = 0.5,
yaxis_title='months')
plt.show()
repayment_interval = df.groupby('repayment_interval')['loan_amount'].sum().sort_values().reset_index()
repayment_interval
repayment_interval = df.groupby(['repayment_interval','sector'])['loan_amount'].sum().sort_values().reset_index()
repayment_interval
repayment_interval = df.groupby(['repayment_interval','activity'])['loan_amount'].sum().sort_values().reset_index()
repayment_interval
def gender_lead(gender):
    gender = str(gender)
    if gender.startswith('f'):
        gender = 'female'
    else:
        gender = 'male'
    return gender
df['gender_lead'] = df['borrower_genders'].apply(gender_lead)
df['gender_lead'].nunique()
f = df['gender_lead'].value_counts()[0]
m = df['gender_lead'].value_counts()[1]

print('{} females ({}%) vs {} males ({}%) got loans'.format(f,round(f*100/(f+m),2),m,round(m*100/(f+m)),2))
df_gender = pd.DataFrame(dict(gender = ['female','male'], counts = [f,m]))
df_gender
import matplotlib.pyplot as plt

plt.bar(df_gender.gender,df_gender.counts) 

plt.show()
amount = df.groupby('sector')['loan_amount','funded_amount'].sum().sort_values(by ='loan_amount', ascending = True).reset_index().head(10)
amount
sns.scatterplot(x='funded_amount',y='loan_amount',data=amount);
loan_term= df.groupby('sector')['loan_amount','term_in_months'].sum().sort_values(by ='loan_amount', ascending = True).reset_index().head(10)
loan_term
plt.xticks (rotation = 60)
sns.barplot(x='loan_amount',y='term_in_months',data=loan_term);

fund_term= df.groupby('sector')['funded_amount','term_in_months'].sum().sort_values(by ='funded_amount', ascending = True).reset_index().head(10)
fund_term
sns.barplot(x='term_in_months',y='funded_amount',data=fund_term);
count = df.groupby('sector')['loan_amount','lender_count'].sum().sort_values(by ='loan_amount', ascending = True).reset_index().head(10)
count
sns.scatterplot(x='loan_amount',y='lender_count',data=count);
fund_count = df.groupby('sector')['funded_amount','lender_count'].sum().sort_values(by ='funded_amount', ascending = True).reset_index().head(10)
fund_count
sns.scatterplot(x='funded_amount',y='lender_count',data=fund_count);
term_count = df.groupby('sector')['term_in_months','lender_count'].sum().sort_values(by ='lender_count', ascending = True).reset_index().head(10)
term_count
sns.barplot(x='lender_count',y='term_in_months',data=term_count);
loans_amnt = df.groupby('sector')['loan_amount'].sum().sort_values(ascending = False).reset_index().head(10)
loans_amnt
loans_amnt = df.groupby('activity')['loan_amount'].sum().sort_values(ascending = False).reset_index().head(10)
loans_amnt
activity_df = puerto.groupby('activity')['loan_amount', 'lender_count', 'funded_amount'].sum()\
         .sort_values(by = 'loan_amount', ascending = False).reset_index().head(10)

activity_df
activity = activity_df['activity']
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.title('Loan Amount by Sector')

plt.xticks(rotation = 75)
plt.xlabel('Sector')
plt.ylabel('Loan Amount')

plt.plot(sector,loan)

plt.subplot(1,2,2)
plt.title('Loan Amount by Activity')

plt.xticks(rotation = 75)
plt.xlabel('Activity')
plt.ylabel('Loan Amount')

plt.plot(activity,loan)

plt.show()
country_rank = data['country'].value_counts().to_frame().reset_index()
country_rank.columns=['country','Number']
country_rank.head(10)
country_rank = df['country'].value_counts().to_frame().reset_index()
country_rank.columns=['country','Number']
country_rank
country_loan = data.groupby('country').sum()['loan_amount'].sort_values(ascending = False).to_frame().reset_index()
country_loan.columns = ['Country', 'Total_amount']
country_loan.head(10)
country_loan = df.groupby('country').sum()['loan_amount'].sort_values(ascending = False).to_frame().reset_index()
country_loan.columns = ['Country', 'Total_amount']
country_loan





