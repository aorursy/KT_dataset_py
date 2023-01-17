import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
% matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from matplotlib_venn import venn3
df_train = pd.read_csv('../input/application_train.csv')
df_train.shape
df_train.head()
df_test = pd.read_csv('../input/application_test.csv')
df_test.shape
df_bu_balance = pd.read_csv('../input/bureau_balance.csv')
df_bu_balance.shape
df_bu_balance.head()
df_bureau = pd.read_csv('../input/bureau.csv')
df_bureau.shape
df_bureau.head()
df_cre_balance = pd.read_csv('../input/credit_card_balance.csv')
df_cre_balance.shape
df_cre_balance.head()
df_installment = pd.read_csv('../input/installments_payments.csv')
df_installment.shape
df_installment.head()
df_pos_cash = pd.read_csv('../input/POS_CASH_balance.csv')
df_pos_cash.shape
df_pos_cash.head()
df_prev_app = pd.read_csv('../input/previous_application.csv')
df_prev_app.shape
df_prev_app.head()
def missing_values(df):
    missing_value = df.isnull().sum()
    missing_percent = (df.isnull().sum()/len(df)*100).round(2)
    table = pd.concat([missing_value, missing_percent], axis=1).rename(
            columns={0:'Missing Values (Count)', 1: 'Missing Values (%)'})
    table = table[table['Missing Values (Count)']>0].sort_values(by='Missing Values (Count)', ascending=False)
    return table
missing_values(df_train).head(30)
missing_values(df_test).head(30)
missing_values(df_bu_balance)
missing_values(df_bureau)
missing_values(df_cre_balance)
missing_values(df_installment)
missing_values(df_pos_cash)
missing_values(df_prev_app)
df_viz = df_train["TARGET"].value_counts().rename(index={0: 'Not default', 1: 'Default'})
trace = go.Pie(labels=df_viz.index, values=df_viz.values)
data = [trace]
layout = go.Layout(title='Distribution of Default', titlefont=dict(size=22))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
df_viz = df_train['NAME_CONTRACT_TYPE'].value_counts()
trace = go.Pie(labels=df_viz.index, values=df_viz.values)
data = [trace]
layout = go.Layout(title='Distribution of Loan Type', titlefont=dict(size=22))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
both = round(len(df_train[(df_train['FLAG_OWN_REALTY']=='Y') & 
                          (df_train['FLAG_OWN_CAR']=='Y')]) / len(df_train) * 100, 2)
realty_only = round(len(df_train[(df_train['FLAG_OWN_REALTY']=='Y') & 
                                 (df_train['FLAG_OWN_CAR']=='N')]) / len(df_train) * 100, 2)
car_only = round(len(df_train[(df_train['FLAG_OWN_REALTY']=='N') & 
                              (df_train['FLAG_OWN_CAR']=='Y')]) / len(df_train) * 100, 2)
plt.figure(figsize=(10, 10))
venn = venn2(subsets = (realty_only, car_only, both), set_labels = ('Real Estate', 'Car'))
for t in venn.set_labels: t.set_fontsize(16)
for t in venn.subset_labels: t.set_fontsize(16)
plt.title('Real Estate/ Car Ownership (%)', fontsize=22);
plt.figure(figsize=(12, 5))
sns.distplot(df_train['AMT_INCOME_TOTAL'])
plt.title('Distribution of Applicant\'s Total Income', fontsize=22)
plt.xlabel('Income');
plt.figure(figsize=(12, 5))
sns.distplot(df_train['AMT_CREDIT'])
plt.title('Distribution of Applicant\'s Credit Amount', fontsize=22)
plt.xlabel('Credit Amount');
plt.figure(figsize=(12, 5))
sns.distplot(df_train['AMT_GOODS_PRICE'].dropna())
plt.title('Distribution of the Price of Goods for Consumer Loans', fontsize=22)
plt.xlabel('Price of Goods');
temp = df_train['NAME_FAMILY_STATUS'].value_counts()
trace = go.Bar(x=temp.index, y=temp, 
               marker=dict(color='coral'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Distribution of Applicant\'s Marital Status', titlefont=dict(size=22),
                   xaxis=dict(title='Marital Status',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)),
                   yaxis=dict(title='Count',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = df_train['NAME_HOUSING_TYPE'].value_counts()
trace = go.Bar(x=temp.index, y=temp, 
               marker=dict(color='plum'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Distribution of Applicant\'s Housing Type', titlefont=dict(size=22),
                   xaxis=dict(title='Housing Type',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14),
                              tickangle=20),
                   yaxis=dict(title='Count',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = df_train['NAME_EDUCATION_TYPE'].value_counts()
trace = go.Bar(x=temp.index, y=temp, 
               marker=dict(color='yellow'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Distribution of Applicant\'s Education', titlefont=dict(size=22),
                   xaxis=dict(title='Education',
                              titlefont=dict(size=16),
                              tickfont=dict(size=12),
                              tickangle=0),
                   yaxis=dict(title='Count',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = df_train['NAME_INCOME_TYPE'].value_counts()
trace = go.Bar(x=temp.index, y=temp, 
               marker=dict(color='powderblue'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Distribution of Applicant\'s Income Source', titlefont=dict(size=22),
                   xaxis=dict(title='Income Source',
                              titlefont=dict(size=16),
                              tickfont=dict(size=12),
                              tickangle=0),
                   yaxis=dict(title='Count',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = df_train['OCCUPATION_TYPE'].value_counts()
trace = go.Bar(x=temp.index, y=temp, 
               marker=dict(color='powderblue'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Distribution of Applicant\'s Occupation', titlefont=dict(size=22),
                   xaxis=dict(title='Occupation',
                              titlefont=dict(size=16),
                              tickfont=dict(size=12)),
                   yaxis=dict(title='Count',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = df_train['ORGANIZATION_TYPE'].value_counts().sort_values().tail(20)
trace = go.Bar(x=temp, y=temp.index, orientation='h',
               marker=dict(color='powderblue'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Distribution of Applicant\'s Work Industry', width = 900, titlefont=dict(size=22),
                   xaxis=dict(title='Count',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)),
                   yaxis=dict(title='Industry',
                              titlefont=dict(size=16),
                              tickfont=dict(size=12)),
                   margin=dict(l=200))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
df_train['AGE'] = -(df_train['DAYS_BIRTH'])/365
plt.figure(figsize=(12, 5))
sns.distplot(df_train['AGE'].dropna())
plt.title('Distribution of Applicant\'s Age', fontsize=20)
plt.xlabel('Age');
plt.figure(figsize=(12, 5))
sns.distplot(-(df_train['DAYS_EMPLOYED'].dropna())/365)
plt.title('Distribution of Applicant\'s Years of Employment', fontsize=20)
plt.xlabel('Year');
df_train['AGE_BIN'] = pd.cut(df_train['AGE'], np.linspace(20, 70, num = 11))
temp = df_train.groupby('AGE_BIN')['TARGET'].mean()
trace = go.Bar(x=temp.index.astype(str), y=temp, 
               marker=dict(color='pink'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Default Rate by Applicant\'s Age', titlefont=dict(size=22),
                   xaxis=dict(title='Age Bin',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)),
                   yaxis=dict(title='Default Rate (%)',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = df_train.groupby('NAME_CONTRACT_TYPE')['TARGET'].mean()
trace = go.Bar(x=temp.index, y=temp, 
               marker=dict(color='pink'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Default Rate by Loan Type', titlefont=dict(size=22),
                   xaxis=dict(title='Loan Type',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)),
                   yaxis=dict(title='Default Rate (%)',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = df_train.groupby('REGION_RATING_CLIENT_W_CITY')['TARGET'].mean()
trace = go.Bar(x=['Region1', 'Region2', 'Region3'], y=temp, 
               marker=dict(color='pink'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Default Rate by Region', titlefont=dict(size=22),
                   xaxis=dict(title='Region',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)),
                   yaxis=dict(title='Default Rate (%)',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = df_train.groupby('TARGET')['DEF_30_CNT_SOCIAL_CIRCLE'].mean()
trace = go.Bar(x=['Non-Default', 'Default'], y=temp, 
               marker=dict(color='pink'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Mean No. of People in Applicant\'s Social Surroundings Defaulted on 30 DPD', 
                   titlefont=dict(size=20),
                   xaxis=dict(titlefont=dict(size=16),
                              tickfont=dict(size=14)),
                   yaxis=dict(title='No. of Default',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
temp = df_train.groupby('TARGET')['DEF_60_CNT_SOCIAL_CIRCLE'].mean()
trace = go.Bar(x=['Non-Default', 'Default'], y=temp, 
               marker=dict(color='pink'), opacity=0.6)
data = [trace]
layout = go.Layout(title='Mean No. of People in Applicant\'s Social Surroundings Defaulted on 60 DPD',
                   titlefont=dict(size=20),
                   xaxis=dict(titlefont=dict(size=16),
                              tickfont=dict(size=14)),
                   yaxis=dict(title='No. of Default',
                              titlefont=dict(size=16),
                              tickfont=dict(size=14)))
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)