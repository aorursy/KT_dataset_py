# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib import cm
import plotly.offline as py
py.init_notebook_mode(connected=True)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_kiva_loans=pd.read_csv('../input/kiva_loans.csv')
df_kiva_loans.head()
df_kiva_mpi_region_locations=pd.read_csv('../input/kiva_mpi_region_locations.csv')
df_kiva_mpi_region_locations.head()
df_loan_theme_ids=pd.read_csv('../input/loan_theme_ids.csv')
df_loan_theme_ids.head()
df_loan_themes_by_region=pd.read_csv('../input/loan_themes_by_region.csv')
df_loan_themes_by_region.head()
df_kiva_loans.shape
counts_by_country = df_kiva_loans['country'].value_counts().head(25)
counts_by_country.plot(kind='bar',figsize=(10,10),title='Loan Distribution By Country',position=0.8 )
plt.show()

counts_by_country
data = [dict(
        type='choropleth',
        locations= counts_by_country.index,
        locationmode='country names',
        z=counts_by_country.values,
        text=counts_by_country.index,
        colorscale='Green',
        marker=dict(line=dict(width=0.7)),
        colorbar=dict(autotick=False, tickprefix='', title='Countries with Most Number of loans'),
)]
layout = dict(title = 'Top Countries with Most Number of loans',)
fig = dict(data=data, layout=layout)
py.iplot(fig, validate=False)
counts_by_sector = df_kiva_loans['sector'].value_counts()
counts_by_sector.plot(kind='bar',figsize=(10,10),title='Loan Distribution By Sectory',position=0.8 )
plt.show()
counts_by_sector = df_kiva_loans['activity'].value_counts().head(25)
counts_by_sector.plot(kind='bar',figsize=(10,10),title='Loan Distribution By Activity',position=0.8 )
plt.show()
counts_by_term_in_months = df_kiva_loans['term_in_months'].value_counts().head(25)
counts_by_term_in_months.plot(kind='bar',figsize=(10,10),title='Loan Distribution By term_in_months',position=0.8 )
plt.show()
bins =[i for i in range(1,3000,50)]
#temp=df_kiva_loans[df_kiva_loans['country_code']=='PH']

df_kiva_loans['loan_amount'].hist(bins=bins,figsize =(15,15) )
plt.show()
lenders_count = df_kiva_loans['lender_count'].value_counts().sort_index().head(30)#.sort_values(ascending =False).head(50)
lenders_count.plot(kind='bar',figsize=(10,10),title='Lender Count Distribution',position=0.8 )
plt.show()
df_kiva_loans['Log Loan Amount']=np.log(df_kiva_loans['loan_amount'])
#color = dict(boxes='DarkGreen', whiskers='DarkOrange',medians='DarkBlue', caps='Gray')
df_kiva_loans.boxplot(column='Log Loan Amount',by='sector',figsize=(20,10))
plt.suptitle('')
plt.title("Log Loan Amount by Sector",fontsize=20, color='red')
plt.xlabel('Sectors', fontsize=15, color='green')
plt.ylabel('Log Loan Amount', fontsize=15, color='green')
plt.show()
#lenders_count.plot(kind='bar',figsize=(10,10),title='Lender Count Distribution',position=0.8 )
df_kiva_loans['Log Funded Amount']=np.log(df_kiva_loans['funded_amount'])
#color = dict(boxes='DarkGreen', whiskers='DarkOrange',medians='DarkBlue', caps='Gray')
df_kiva_loans.boxplot(column='Log Funded Amount',by='sector',figsize=(20,10))
plt.suptitle('')
plt.title("Log Funded Amount by Sector",fontsize=20, color='red')
plt.xlabel('Sectors', fontsize=15, color='green')
plt.ylabel('Log Funded Amount', fontsize=15, color='green')
plt.show()
df_kiva_loans.boxplot(column='term_in_months',by='sector',figsize=(20,10))
plt.suptitle('')
plt.title("Terms In Months By Sector",fontsize=20, color='red')
plt.xlabel('Sectors', fontsize=15, color='green')
plt.ylabel('Terms In Months', fontsize=15, color='green')
plt.show()
from wordcloud import WordCloud, STOPWORDS

plt.figure(figsize=(12, 10))

stopwords = set(STOPWORDS)
wordcloud = WordCloud(

    background_color='pink',
    stopwords=stopwords,
    max_words=150,
    max_font_size=40,
    width=600, height=300,
    random_state=42,
).generate(str(df_kiva_loans['use']))

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
#plt.title("WORD CLOUD - DESCRIPTION")
plt.axis('off')
plt.show()
df_loan_themes_by_region.head(5)
print("Top Kiva Field Partner Names with funding count : ", len(df_loan_themes_by_region["Field Partner Name"].unique()))
field_partner_count = df_loan_themes_by_region['Field Partner Name'].value_counts().head(30)#.sort_values(ascending =False).head(50)
print(field_partner_count.head(5))
field_partner_count.plot(kind='bar',figsize=(12,12),title='Field Partner Count',position=0.8 )
plt.show()
