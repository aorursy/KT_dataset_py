from IPython.display import Image
Image("../input/inputmarketcapitaladditionduringpandemic/bigshort1.JPG")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import cufflinks as cf
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objs as go
mark_lead=pd.read_csv('../input/market-capital-addition-during-pandemic/companies_market_cap.csv')
mark_lead.head(n=15)
            
mark_lead.info()
mark_lead.describe().T
mark_lead.columns
mark_lead.shape
mark_lead.corr()
def missing_values_table(data):
        # Total missing values
        miss_values = mark_lead.isnull().sum()
        
        # Percentage of missing values
        miss_value_percent = 100 * mark_lead.isnull().sum() / len(mark_lead)
        
        # Make a table with the results
        miss_value_table = pd.concat([miss_values, miss_value_percent], axis=1)
        
        # Rename the columns
        miss_value_table_ren_columns = miss_value_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing==>descending
        mis_val_table_ren_columns = miss_value_table_ren_columns[
            miss_value_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(data.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        
        # Return the dataframe with missing information
        return miss_value_table_ren_columns


missing_values= missing_values_table(mark_lead)
missing_values.style.background_gradient(cmap='Reds')  
mark_lead.dtypes
mark_lead['Country'].unique()
mark_lead['Sector'].value_counts()
mark_lead['Sector'].unique()
mark_lead['Company'].unique()
sns.set_context("paper")
plt.style.use('seaborn-poster')
plt.figure(figsize = (12,15))
ax = sns.barplot(y = 'Country' , x = 'Market cap added', data = mark_lead, palette = 'mako', edgecolor = 'black')
ax.set_title('Market Cap added (Country-Wise)' , size = 20, pad = 20)
for p in ax.patches:
        ax.annotate("%.f" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),
            xytext=(5, 0), textcoords='offset points', ha="left", va="center")
labels = mark_lead['Sector'].value_counts().index
values = mark_lead['Market cap added'].value_counts().values
fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label',
                             insidetextorientation='radial')])
fig.show()
sns.set_context("paper")
plt.style.use('seaborn-poster')
plt.figure(figsize = (12,15))
ax = sns.barplot(y = 'Sector' , x = 'Market cap added', data = mark_lead, palette = 'mako', edgecolor = 'black')
ax.set_title('Market Cap added (Country-Wise)' , size = 20, pad = 20)
for p in ax.patches:
        ax.annotate("%.f" % p.get_width(), xy=(p.get_width(), p.get_y()+p.get_height()/2),
            xytext=(5, 0), textcoords='offset points', ha="left", va="center")
mark_lead_tech = mark_lead[mark_lead['Sector']=='Technology']
fig = px.bar(mark_lead_tech, x='Company', y='Market cap added',color='Company',height=500)
fig.show()
figure = px.scatter(mark_lead_tech, x="Company", y="Change",
        size="Change", color="Company",
                 hover_name="Company",size_max=30)
figure.show()

mark_lead_health = mark_lead[mark_lead['Sector']=='Healthcare']
fig = px.bar(mark_lead_health, x='Company', y='Market cap added',color='Company',height=500)
fig.show()
figure = px.scatter(mark_lead_health, x="Company", y="Change",
        size="Change", color="Company",
                 hover_name="Company",size_max=30)
figure.show()
mark_lead_commservices = mark_lead[mark_lead['Sector']=='Communication services']
fig = px.bar(mark_lead_commservices, x='Company', y='Market cap added',color='Company',height=500)
fig.show()
figure = px.scatter(mark_lead_commservices, x="Company", y="Change",
        size="Change", color="Company",
                 hover_name="Company",size_max=30)
figure.show()
fig = px.bar(mark_lead, x='Company', y='Market cap added',color='Change',)
fig.show()
fig = px.bar(mark_lead, x='Company', y='Change',color='Market cap added')
fig.show()
mark_lead.groupby('Market cap added').Change.agg(['count','max','min','mean'])
mark_lead[mark_lead.Sector=='Technology'].Change.agg(['count','max','min','mean'])
mark_lead[mark_lead.Sector=='Communication services'].Change.agg(['count','max','min','mean'])

mark_lead[mark_lead.Country=='India'].Change.agg(['count','max','min','mean'])
mark_lead[mark_lead.Country=='US'].Change.agg(['count','max','min','mean'])