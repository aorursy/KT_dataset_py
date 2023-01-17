# OVERVIEW



# I am conducting an exploratory data analysis on CFPB consumer complaints, with the aim to understand the broad composition of said complaints.



# The research questions I will address in this notebook include:

#     1. Which companies have the most complaints? (Sneak peek: the 3 major credit bureaus)

#     2. What are the most common complaint categories for the credit bureaus?

#     3. Have credit bureau complaints been consistent across time, or have they changed?

#     4. Identify which of the 3 credit bureaus are most likely to respond in a timely manner to complaints

#     5. Identify the most common outcomes of complaints to the 3 credit bureaus

#     6. Identify which of the 3 credit bureaus are most likely to have disputed claims

    
# DATA PROFILE & PRIOR WORK



# DATASET 1: CFPB complaints, from data.gov

# https://catalog.data.gov/dataset/consumer-complaint-database



# The Consumer Financial Protection Bureau (CFPB) is a U.S. government agency that protects American citizens in the financial

# sector. Consumers can find detailed, easy-to-understand documentation (such as mortgage lending), as well as lodge

# complaints against institutions. Their mission is "We protect, promote, and preserve the financial wellbeing of the 

# American consumer."



# Consumer complaints are logged and documented. 1.5 milion of complaints - including metadata, the verbatims, and 

# complaint resolution status. This data is well-suited to understand which companies



# As a new homeowner who had to go through mortgage process, and major Warren fan, I am curious to explore what qualities

# of complaints (as seen in the verbatims) are most likely to be resolved (by reviewing the company public response.)





# DATASET 2: state population 2010-2019, from census.gov

# https://www.census.gov/data/tables/time-series/demo/popest/2010s-state-total.html



# As the CFPB contains state information, I will tie state population data to complaint data to see which state residents

# are most likely to make complaints with the CFPB.
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import datetime as dt



# Plot

plt.style.available

plt.style.use('fivethirtyeight')



font = {'size'   : 22}



plt.rc('font', **font)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

complaints = pd.read_csv("../input/cfpbcomplaints/complaints.csv")

state_pop = pd.read_csv("../input/censuspop/state-pop.csv")
# ANALYSIS
complaints.head()
# Review what data are available by column name

complaints.columns
# Data cleanning

# Remove zip code, tags, consumer consent = 'Consent provided', date sent to company, complaint id

cfpb = complaints[['Date received', 'Product', 'Sub-product', 'Issue', 'Sub-issue',

       'Consumer complaint narrative', 'Company public response', 'Company','State', 'Consumer consent provided?',

       'Submitted via', 'Company response to consumer','Timely response?', 'Consumer disputed?']]



# cfpb.head()
# QUESTION 1: Which companies have the most complaints?

# Most common companies which have complaints lodged against them

complaints.groupby('Company').size().sort_values(ascending=False).head(10).plot(kind='bar', figsize=(20,8))



# Results

# The most common companies to have complaints lodged against them are Equifax, Experian, and Transunion. 

# To scope the rest of this analsis, only data related to the credit bureaus will be analyzed further.
# Create new df to focus on credit bureaus only

# These were the 3 top companies against whom claims were lodged (see graph above)

cfpb_credit = cfpb[(cfpb.Company.isin(["EQUIFAX, INC."

                                       ,"Experian Information Solutions Inc."

                                       ,"TRANSUNION INTERMEDIATE HOLDINGS, INC."]))]



# Change date from YYYY-MM-DD to YYYY

cfpb_credit['Date received'] = cfpb_credit['Date received'].str[0:4]

cfpb_credit = cfpb_credit.rename(columns={"Date received": "Date"})

cfpb_credit.head()
# RESEARCH QUESTION 2: What are the most common complaint categories for the credit bureaus?

# Calculate which companies had which types of product isses



# group by

company_product_issue = cfpb_credit.groupby(["Company","Product"]).size() 



# Reset index

company_product_issue = company_product_issue.reset_index() 



# Rename column to "Value"

company_product_issue = company_product_issue.rename(columns={0: "Value"})



# Create pivot table

company_product_issue = company_product_issue.pivot(index='Product',columns='Company')['Value']



# Rename company names to be readable

company_product_issue = company_product_issue.rename(columns={'EQUIFAX, INC.': "Equifax"

                                            , 'Experian Information Solutions Inc.': 'Experian'

                                            , 'TRANSUNION INTERMEDIATE HOLDINGS, INC.': 'Transunion'})

# Fill NaN data with 0

company_product_issue.fillna(0)
# Convert whole numbers to percentages

company_product_issue_pcts = company_product_issue/company_product_issue.sum()*100
# Plot pivot table of consumer disputes by company

company_product_issue_pcts.plot(kind='bar', figsize=(20,8))
# RESEARCH QUESTION 3: Have credit bureau complaints been consistent across time, or have they changed?

# Calculate number of complaints by year



# group by

company_year = cfpb_credit.groupby(["Company","Date"]).size() 



# Reset index

company_year = company_year.reset_index() 



# Rename column to "Value"

company_year = company_year.rename(columns={0: "Value"})



# Create pivot table

company_year = company_year.pivot(index='Date',columns='Company')['Value']



# Rename company names to be readable

company_year = company_year.rename(columns={'EQUIFAX, INC.': "Equifax"

                                            , 'Experian Information Solutions Inc.': 'Experian'

                                            , 'TRANSUNION INTERMEDIATE HOLDINGS, INC.': 'Transunion'})

# Drop years with incomplete data

company_year = company_year.drop(['2012','2020'])

company_year
# Plot pivot table of consumer disputes by company

company_year.plot(kind='bar', figsize=(20,8), stacked=True)
# RESEARCH QUESTION 4: Identify which of the 3 credit bureaus are most likely to respond in a timely manner to complaints

# Calculate which companies had timely responses



# group by

company_timely_response = cfpb_credit.groupby(["Company","Timely response?"]).size() 



# Reset index

company_timely_response = company_timely_response.reset_index() # 



# Rename column to "Value"

company_timely_response = company_timely_response.rename(columns={0: "Value"})



# Create pivot table

company_timely_response = company_timely_response.pivot(index='Timely response?',columns='Company')['Value']



# Rename company names to be readable

company_timely_response = company_timely_response.rename(columns={'EQUIFAX, INC.': "Equifax"

                                                                      , 'Experian Information Solutions Inc.': 'Experian'

                                                                      , 'TRANSUNION INTERMEDIATE HOLDINGS, INC.': 'Transunion'})



# Print out table

company_timely_response
# Convert whole numbers to percentages

company_timely_response_pcts = company_timely_response/company_timely_response.sum()*100



# Replace NaN with zeros

company_timely_response_pcts.fillna(0)
# Plot pivot table of timeliness of responses by company

company_timely_response_pcts.plot(kind='bar', figsize=(20,8))
# RESEARCH QUESTION 5: Identify the most common outcomes of complaints to the 3 credit bureaus

# Calculate which companies had which most common outcomes



# group by

company_consumer_response = cfpb_credit.groupby(["Company","Company response to consumer"]).size() 



# Reset index

company_consumer_response = company_consumer_response.reset_index() 



# Rename column to "Value"

company_consumer_response = company_consumer_response.rename(columns={0: "Value"})



# Create pivot table

company_consumer_response = company_consumer_response.pivot(index='Company response to consumer',columns='Company')['Value']



# Rename company names to be readable

company_consumer_response = company_consumer_response.rename(columns={'EQUIFAX, INC.': "Equifax"

                                                                      , 'Experian Information Solutions Inc.': 'Experian'

                                                                      , 'TRANSUNION INTERMEDIATE HOLDINGS, INC.': 'Transunion'})

# Replace NaN with zeros

company_consumer_response.fillna(0)
# Convert whole numbers to percentages

company_consumer_response_pcts = company_consumer_response/company_consumer_response.sum()*100



# Replace NaN with zeros

company_consumer_response_pcts.fillna(0)
# Plot pivot table of responses to consumer by company

company_consumer_response_pcts.plot(kind='bar', figsize=(20,8))
# RESEARCH QUESTION 6: Identify which of the 3 credit bureaus are most likely to have disputed claims

# Calculate which companies had disputed complaints



# group by

company_disputed_response = cfpb_credit.groupby(["Company","Consumer disputed?"]).size() 



# Reset index

company_disputed_response = company_disputed_response.reset_index() # 



# Rename column to "Value"

company_disputed_response = company_disputed_response.rename(columns={0: "Value"})



# Create pivot table

company_disputed_response = company_disputed_response.pivot(index='Consumer disputed?',columns='Company')['Value']



# Rename company names to be readable

company_disputed_response = company_disputed_response.rename(columns={'EQUIFAX, INC.': "Equifax"

                                                                      , 'Experian Information Solutions Inc.': 'Experian'

                                                                      , 'TRANSUNION INTERMEDIATE HOLDINGS, INC.': 'Transunion'})



# Print out table

company_disputed_response
# Convert whole numbers to percentages

company_disputed_response_pcts = company_disputed_response/company_disputed_response.sum()*100

company_disputed_response_pcts
# Plot pivot table of consumer disputes by company

company_disputed_response_pcts.plot(kind='bar', figsize=(20,8))
# CONCLUSIONS



# 1. Which companies have the most complaints?

#     The 3 major creit bureaus (Equifax, Experian and Transunion) have the most complaints lodged by consumers.

    

# 2. What are the most common complaint categories for the credit bureaus?

#     Unsurprisingly, credit reports and credit repair services were the most common complaints for the credit bureaus.

#     Debt collection was a distant third.

#     All other types of complaints had near 0% totals.

    

# 3. Have credit bureau complaints been consistent across time, or have they changed?

#     Complaints from consumers have risen since the beginning of data collection.

#     This likely reflects overall increase of complaints as more consumers become aware of the CFPB.



# 4. Identify which of the 3 credit bureaus are most likely to respond in a timely manner to complaints

#     Most complaints, regardless of company, were responded to in a timely manner.

#         Experian was most likely to have a timely response, with 99.99% of complaints responded to in a timely manner.

#         Transunion was the next most likely to have a timely response, with 99.93% of complaints responded to in a timely manner.

#         Equifax was the least likely to have a timely response, with 99.88% of complaints responded to in a timely manner.



# 5. Identify the most common outcomes of complaints to the 3 credit bureaus

#     The most common outcomes were closed with explanation, and then closed with non-monetary relief.

#     Equifax was the most likely to continue having complaints in progress.



# 6. Identify which of the 3 credit bureaus are most likely to have disputed claims

#     Equifax was most likely to have disputed claims, with 20.83% of complaints disputed.

#     Transunion was the next most likely to have disputed claims, with 14.09% of complaints disputed.

#     Experian was the least likely to have disputed claims, with 11.68% of complaints disputed.





# DIRECTIONS FOR FUTURE WORK



# The logical extension of this work would be to analyze the verbatims provided by consumers with Machine Learning techniques

# to identify which words, phrasing, or other features contribute to the best outcomes from consumer complaints.



# With approval, I would like to use this same data set to approach this particular issue.