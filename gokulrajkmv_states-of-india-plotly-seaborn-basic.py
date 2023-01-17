#let us start with some basic imports


# for data Analysis

import numpy as np
import pandas as pd


# for data visualization

import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

# for interactive data visualization

from plotly import __version__

import cufflinks as cf

from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot

init_notebook_mode(connected=True)

cf.go_offline()
# for importing data

df = pd.read_csv('../input/indian-statewise-data-from-rbi/RBI DATA states_wise_population_Income.csv')
# head of Dataframe

df.head(4)
# Dataframe info.

df.info()
# basic descriptive statistics for 2001

df[['2000-01-INC','2001 - LIT','2001 - POP','2001 -SEX_Ratio','2001 -UNEMP','2001 -Poverty']].describe()

# descriptive statistics for 2011

df[['2011-12-INC','2011- LIT','2011- POP','2011 -SEX_Ratio','2011 -UNEMP','2011 -Poverty']].describe()
# missing data in percentage

df.isnull().sum()/len(df)
#plotting pariplot for 2001 state's income, Literacy rate, Poverty rate and umeployment rate

# data for plot
df_plot_data_2001 = df[['2000-01-INC','2001 - LIT','2001 - POP','2001 -UNEMP','2001 -Poverty']]

#plotting in seaborn
sns.set_style(style='darkgrid')
plt.figure(figsize = (10,6))
sns.set_context(context ='notebook',font_scale=1)
sns.pairplot(df_plot_data_2001,aspect=1,palette='summer')
plt.tight_layout()
#plotting pariplot for 2001 state's income, Literacy rate, Poverty rate and umeployment rate

# data for plot
df_plot_data_2011 = df[['2011-12-INC','2011- LIT','2011- POP','2011 -UNEMP','2011 -Poverty']]

# plotting in seaborn
sns.set_style(style='darkgrid')
plt.figure(figsize = (10,6))
sns.set_context(context ='notebook',font_scale=1)
sns.pairplot(df_plot_data_2011,aspect=1,palette='summer')
plt.tight_layout()
# correlation of the df
df_corr = df.corr()

#plotting in seaborn
plt.figure(figsize=(10,7))
sns.heatmap(df_corr,cmap='summer_r',annot=True,linewidths=0.2,linecolor='white')
plt.tight_layout()
# plotting Indian State's Income vs Literacy rate in year 2001

# data for plot
df_lit_inc_2001 = df[['States_Union Territories','2000-01-INC','2001 - LIT']]

# line plot in plotly
df_lit_inc_2001.iplot(kind='line',x='States_Union Territories',secondary_y='2001 - LIT',
                      colors=['#53D1BA','#D11E5D'],title='India State Income vs Literacy rate year_2001',
                      xTitle='States in India',yTitle='State Income_SDP',
                      secondary_y_title='literacy rate',theme='pearl')

# to see how closely the each state's income and Literacy rate is
# plotting Indian State's Income vs Literacy rate in year 2011

# data for plot
df_lit_inc_2011 = df[['States_Union Territories','2011-12-INC','2011- LIT']]

# plot
df_lit_inc_2011.iplot(x ='States_Union Territories',y='2011-12-INC',secondary_y ='2011- LIT',
                      colors = ['#70a3f9','#e7c269'],title = 'India State Income vs Literacy rate year_2011',
                      xTitle='States in India',yTitle ='State Income_SDP',secondary_y_title = 'literacy rate',
                     theme = 'pearl')

# calculating percentage change in the Income of the states and plotting

#increase = Increase ÷ Original Number × 100.

# took a copy of df and named it
df1 = df.copy()

# percentage change in income of the state
df1['INC_percentage_change'] = (df1['2011-12-INC'] - df1['2000-01-INC']) / df1['2000-01-INC']



# sorting before plot

df1_sorted = df1.sort_values(by='INC_percentage_change',ascending=True)

# plotting percentage change in states income from 2000 to 2011

df1_sorted.iplot(kind='bar',x='States_Union Territories',y='INC_percentage_change',theme='white',colors='#2997B7',
                title='percentage change in states income from 2000 to 2011',yTitle = 'percentage')
#ploting percentage change in Literacy rate in each state

#data for plot
df1['literacy_pct_change'] = df1['2011- LIT'] - df1['2001 - LIT']

#plot
df1.sort_values('literacy_pct_change').iplot(kind = 'bar',x = 'States_Union Territories',yTitle = 'percentage',y = 'literacy_pct_change',theme='white',colors='#1ABC9C',title = 'percentage change in states Literacy rate from 2000 to 2011')
# sorting and ploting percentage change in Literacy rate

df1.sort_values(by='literacy_pct_change').iplot(kind='line',x='States_Union Territories',y='INC_percentage_change',
                                               secondary_y='literacy_pct_change',secondary_y_title='Literacy rate change',
                                               title='Percentage change in Income vs percentage change in literacy rate from 2001 to 2011',
                                                yTitle='percentage',xTitle = 'States in India')
#plotting Poverty rate vs Literacy rate in 2001

pov_lit_rate_2001 = df1[['States_Union Territories','2001 -Poverty','2001 - LIT']]

pov_lit_rate_2001.iplot(kind='line',x='States_Union Territories',y='2001 - LIT',secondary_y='2001 -Poverty',secondary_y_title='poverty rate',
                        colors=['#ff6f69','#ffcc5c'],title='Poverty rate vs Literacy rate in each states in the 2001',
                        xTitle='States in India',yTitle='Literacy rate')
pov_lit_rate_2011 = df1[['States_Union Territories','2011 -Poverty','2011- LIT']]
        

pov_lit_rate_2011.sort_values(by='2011 -Poverty').iplot(kind='line',x='States_Union Territories',y='2011- LIT',secondary_y='2011 -Poverty',secondary_y_title='poverty rate',
                        colors=['#ff6f69','#ffcc5c'],title='Poverty rate vs Literacy rate in each states in the year 2011',
                        yTitle='Literacy rate',xTitle='States in India')

#data for plot
pov_income_rate_2001 = df1[['States_Union Territories','2001 -Poverty','2000-01-INC']]
        
# plotting state income vs poverty rate in 2001
pov_income_rate_2001.sort_values(by='2000-01-INC').iplot(kind='line',x='States_Union Territories',y='2000-01-INC',secondary_y='2001 -Poverty',secondary_y_title='poverty rate',
                        colors=['#1ebbd7','#ffcc5c'],title='state Income vs Poverty rate in each states in the 2001',
                        yTitle='State income',xTitle ='States in india')
state_income_Pov_rate_2011 = df1[['States_Union Territories','2011 -Poverty','2011-12-INC']]
        

state_income_Pov_rate_2011.sort_values(by='2011-12-INC').iplot(kind='line',
                                                               x ='States_Union Territories',y ='2011-12-INC',
                                                               secondary_y ='2011 -Poverty',secondary_y_title='poverty rate',
                                                               colors =['#1ebbd7','#ffcc5c'],
                                                               title ='state Income vs Poverty rate in each states in the 2011',
                                                               yTitle ='State income',xTitle ='States in india')