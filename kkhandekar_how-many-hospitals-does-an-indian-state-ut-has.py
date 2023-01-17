import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, gc, warnings,re 

warnings.filterwarnings("ignore")



# Plot

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import plotly.express as px
# Load Data

url = '../input/hospitals-count-in-india-statewise/Hospitals count in India - Statewise.csv'

df = pd.read_csv(url, header='infer')



# Total Record

print("Total Records: ", df.shape[0])



# Renaming Columns

df.rename(columns={"States/UTs":"States",

                   "Number of hospitals in public sector": "Public",

                   "Number of hospitals in private sector": "Private",

                   "Total number of hospitals (public+private)":"Total"

                  }, inplace=True)





# Drop Columns with Null/Missing Values

df.dropna(inplace=True)



# Helper Function to Clean the Numeric Values

def clean (num):

    return re.sub("[^\d\.]", "", num)





# Cleaning & Converting the Data Type

cols = ['Public','Private','Total']



for col in cols:

    df[col] = df[col].apply(lambda x: clean(x))

    df[col] = pd.to_numeric(df[col])





#Inspect

df.head()
'''Stat Summary'''

df.describe().T
fig = px.bar(df, x="States", y=["Public","Private"], title="Statewise Hospital Count",text='Total',

             color_discrete_sequence = ['lightslategray','lightsteelblue'])

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.update_layout(uniformtext_minsize=8.5, uniformtext_mode='hide')

fig.show()
def eda():



    # Plot Config

    fig, axes = plt.subplots(3, 2, figsize=(25,15))

    fig.suptitle('Hospital Count - EDA', fontsize=20)

    axes = axes.flatten()

    

    ### Public Hospital EDA

    temp = df[['States','Public']]

    

    # Highest

    tempL = temp.nlargest(5, columns=['Public'])

    tempL.reset_index(drop=True, inplace=True)

    

    sns.barplot(ax=axes[0], x=tempL.States.values, y=tempL.Public.values, palette="mako_r")

    axes[0].set_title("Top 5 States with Highest Public Hospitals")

    

    # Lowest

    tempS = temp.nsmallest(5, columns=['Public'])

    tempS.reset_index(drop=True, inplace=True)

    

    sns.barplot(ax=axes[1], x=tempS.States.values, y=tempS.Public.values, palette="mako_r")

    axes[1].set_title("Top 5 States with Lowest Public Hospitals")

    

    

    ### Private Hospital EDA

    temp = df[['States','Private']]

    

    # Highest

    tempL = temp.nlargest(5, columns=['Private'])

    tempL.reset_index(drop=True, inplace=True)

    

    sns.barplot(ax=axes[2], x=tempL.States.values, y=tempL.Private.values, palette="mako_r")

    axes[2].set_title("Top 5 States with Highest Private Hospitals")

    

    # Lowest

    tempS = temp.nsmallest(5, columns=['Private'])

    tempS.reset_index(drop=True, inplace=True)

    

    sns.barplot(ax=axes[3], x=tempS.States.values, y=tempS.Private.values, palette="mako_r")

    axes[3].set_title("Top 5 States with Lowest Private Hospitals")    





    ### Total Hospital EDA

    temp = df[['States','Total']]

    

    # Highest

    tempL = temp.nlargest(5, columns=['Total'])

    tempL.reset_index(drop=True, inplace=True)

    

    sns.barplot(ax=axes[4], x=tempL.States.values, y=tempL.Total.values, palette="mako_r")

    axes[4].set_title("Top 5 States with Highest Number of Hospitals")

    

    # Lowest

    tempS = temp.nsmallest(5, columns=['Total'])

    tempS.reset_index(drop=True, inplace=True)

    

    sns.barplot(ax=axes[5], x=tempS.States.values, y=tempS.Total.values, palette="mako_r")

    axes[5].set_title("Top 5 States with Lowest Number of Hospitals")    

    



# Execute Function

eda ()
gc.collect()