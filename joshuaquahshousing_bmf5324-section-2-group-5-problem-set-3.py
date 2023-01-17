import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt # visualization

import os

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objs as go
# 1. S&P 

snp = pd.read_csv('../input/snp500-max/GSPC latest snp.csv')

snp['Date'] = pd.to_datetime(snp['Date'])

snp.rename(columns={'Adj Close':'snp'}, inplace=True)

snp['snp_return'] = snp['snp'].pct_change()

snp['snp_volatility_1m'] = (snp['snp_return'].rolling(20).std())*(20)**(1/2) # Annualize daily standard deviation

snp['snp_volatility_1y'] = (snp['snp_return'].rolling(252).std())*(252)**(1/2) # 252 trading days per year

snp = snp[['Date','snp','snp_return','snp_volatility_1m','snp_volatility_1y']]

# Calculate 1-month forward cumulative returns

snp['one_month_forward_snp_return'] = snp['snp_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]

snpdate=snp[(snp['Date']>'1985-1-1')]



# 2. DJI

dji = pd.read_csv('../input/dji-index/DJI data.csv')

dji["Date"] = pd.to_datetime(dji["Date"])

dji.rename(columns={'Adj Close':'dji'}, inplace=True)

dji['dji_return'] = dji['dji'].pct_change()

dji['dji_volatility_1m'] = (dji['dji_return'].rolling(20).std())*(20)**(1/2) 

dji['dji_volatility_1y'] = (dji['dji_return'].rolling(252).std())*(252)**(1/2) 

dji = dji[['Date','dji','dji_return','dji_volatility_1m','dji_volatility_1y']]

dji['one_month_forward_dji_return'] = dji['dji_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]



snp2010=snp[(snp['Date']>'2010-1-1')]

dji2010=dji[(dji['Date']>'2010-1-1')]





# Create figure with secondary y-axis

fig = make_subplots(specs=[[{"secondary_y": True}]])



# Add traces

fig.add_trace(

    go.Scatter(x=snp2010['Date'], y=snp2010['snp'], name = 'S&P'),  

    secondary_y=False,

)



fig.add_trace(

    go.Scatter(x=dji2010['Date'], y=dji2010['dji'],opacity=0.5, name = 'DJI'), 

    secondary_y=True,

)





# Add figure title

fig.update_layout(

    title_text="S&P and DJI Past 10 Years"

)



# Set x-axis title

fig.update_xaxes(title_text="Date")



# Set y-axes titles

fig.update_yaxes(title_text="<b>S&P</b>", secondary_y=False)

fig.update_yaxes(title_text="<b>DJI", secondary_y=True)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
snp2020=snp[(snp['Date']>'2020-1-1')]

dji2020=dji[(dji['Date']>'2020-1-1')]





# Create figure with secondary y-axis

fig = make_subplots(specs=[[{"secondary_y": True}]])



# Add traces

fig.add_trace(

    go.Scatter(x=snp2020['Date'], y=snp2020['snp'], name = 'S&P'),  

    secondary_y=False,

)



fig.add_trace(

    go.Scatter(x=dji2020['Date'], y=dji2020['dji'],opacity=0.5, name = 'DJI'), 

    secondary_y=True,

)





# Add figure title

fig.update_layout(

    title_text="S&P and DJI Past in 2020"

)



# Set x-axis title

fig.update_xaxes(title_text="Date")



# Set y-axes titles

fig.update_yaxes(title_text="<b>S&P</b>", secondary_y=False)

fig.update_yaxes(title_text="<b>DJI", secondary_y=True)

fig.show()
# Create figure with secondary y-axis

fig = make_subplots(specs=[[{"secondary_y": True}]])



# Add traces

fig.add_trace(

    go.Scatter(x=snp2020['Date'], y=snp2020['snp_return'], name = 'S&P return'),  

    secondary_y=False,

)



fig.add_trace(

    go.Scatter(x=dji2020['Date'], y=dji2020['dji_return'],opacity=0.7, name = 'DJI Return'), 

    secondary_y=True,

)





# Add figure title

fig.update_layout(

    title_text="S&P Return and DJI Return in 2020"

)



# Set x-axis title

fig.update_xaxes(title_text="Date")



# Set y-axes titles

fig.update_yaxes(title_text="<b>S&P Return</b>", secondary_y=False)

fig.update_yaxes(title_text="<b>DJI\' Return</b>", secondary_y=True)

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
from scipy.stats import t



snpdji2020 = pd.merge(snp2020, dji2020, how='left', right_on='Date', left_on='Date')

snpdji2020['diff']=snpdji2020['snp_return']-snpdji2020['dji_return']





sample_mean = snpdji2020['diff'].mean()

sample_standard_error = snpdji2020['diff'].sem()

tt = sample_mean/sample_standard_error

df = snpdji2020['diff'].count()-1

pval = t.sf(np.abs(tt), df)*2  # two-sided pvalue = Prob(abs(t)>tt)



print("Point estimate : " + str(sample_mean))

print("Standard error : " + str(sample_standard_error))

print("Degree of freedom : " + str(df))

print("T-statistic : " + str(tt))



# interpret p-value 

alpha = 0.05

print("P-value : " + str(pval))

if pval <= alpha: 

    print('There is a difference in SNP return and DJI return (reject H0)') 

else: 

    print('There is no difference in SNP return and DJI return (fail to reject H0)') 
confidence_level = 0.95



confidence_interval = t.interval(confidence_level, df, sample_mean, sample_standard_error)



print("Point estimate : " + str(sample_mean))

print("Confidence interval (0.025, 0.975) : " + str(confidence_interval))
bin_labels_3 = ['Bin 1', 'Bin 2', 'Bin 3']



snpdji2020['snp_bin'] = pd.qcut(snpdji2020['snp'], q=3, labels=bin_labels_3)

snpdji2020['dji_bin'] = pd.qcut(snpdji2020['dji'], q=3, labels=bin_labels_3)



contingency_table = [[snpdji2020.groupby('dji_bin')['dji'].sum(),snpdji2020.groupby('snp_bin')['snp'].sum()]]

contingency_table
from scipy.stats import chi2_contingency 



stat, p, dof, expected = chi2_contingency(contingency_table) 

  

# interpret p-value 

alpha = 0.05

print("p value is " + str(p)) 

if p <= alpha: 

    print('Dependent (reject H0)') 

else: 

    print('Independent (fail to reject H0)') 

    

snpdji = pd.merge(snp, dji, how='left', right_on='Date', left_on='Date')

snpdji['diff']=snpdji['snp_return']-snpdji['dji_return']





sample_mean = snpdji['diff'].mean()

sample_standard_error = snpdji['diff'].sem()

tt = sample_mean/sample_standard_error

df = snpdji['diff'].count()-1

pval = t.sf(np.abs(tt), df)*2  # two-sided pvalue = Prob(abs(t)>tt)



print("Point estimate : " + str(sample_mean))

print("Standard error : " + str(sample_standard_error))

print("Degree of freedom : " + str(df))

print("T-statistic : " + str(tt))



# interpret p-value 

alpha = 0.05

print("P-value : " + str(pval))

if pval <= alpha: 

    print('There is a difference in SNP return and DJI return (reject H0)') 

else: 

    print('There is no difference in SNP return and DJI return (fail to reject H0)') 