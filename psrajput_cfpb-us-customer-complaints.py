# Importing
import spacy
%matplotlib inline
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
from spacy import displacy
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#Loading
nlp = spacy.load("en_core_web_sm")

# nlp = spacy.load('de')
pd.options.mode.chained_assignment = None

sns.set_style("dark", {"axes.facecolor": "white"})
raw_data = pd.read_csv('../input/complaints.csv')
raw_data.shape
# NULL Percentages
round((raw_data.isnull().sum() / len(data) * 100), 2)
# Dropping ZIP code, Tags, Consumer consent provided? & Consumer disputed?

data = raw_data[[
    'Complaint ID', 
    'Date received', 
    'Product',
    'Issue',
    'Sub-issue',
    'Consumer complaint narrative', 
    'Company public response', 
    'Company', 
    'State', 
    'Consumer consent provided?', 
    'Submitted via', 
    'Date sent to company',
    'Company response to consumer',
    'Timely response?'
]]
# Renaming Columns

data.columns = [
    'Complaint_id', 
    'Data_received', 
    'Product',
    'Issue',
    'Sub_issue',
    'Complaint_narrative', 
    'Company_response', 
    'Company', 
    'State', 
    'Consumer_consent',
    'Submitted_via', 
    'Response_date',
    'Response',
    'Timely_response'
]

data.head(2)
# Creating Columns of Year and Month from the Date received
data['Data_received'] = pd.to_datetime(data['Data_received'])
data['Year'] = data['Data_received'].dt.year
data['Month'] = data['Data_received'].dt.month

data.head(2)
#Adding Another Column for Complaint Count
data['Count'] = pd.Series(np.ones(len(data)), dtype=int)
data.head(2)
# Filtered Data for 2018
data_2018 = data[data['Year'] == 2018]
data_2018.head(2)
# Product-wise Count of Complaints
fig,ax = plt.subplots(figsize=(20,6))
PvsC = sns.countplot(x='Product',data=data)

for label in ax.get_xticklabels():
    label.set_rotation(90)
    
ax.set_xlabel("Products", fontsize=12, color="red");
ax.set_ylabel("Count", fontsize=12, color="red");

plt.show()
# Product-wise Count of Complaints
fig,ax = plt.subplots(figsize=(20,6))
PvsC = sns.countplot(x='Product',data=data)

for label in ax.get_xticklabels():
    label.set_rotation(90)
    
ax.set_xlabel("Products", fontsize=12, color="red");
ax.set_ylabel("Count", fontsize=12, color="red");

plt.show()
# Month-wise Count of Complaints

fig,ax = plt.subplots(figsize=(20,6))
MvsC = sns.countplot(x='Month',data=data)

for label in ax.get_xticklabels():
    label.set_rotation(90)
    
ax.set_xlabel("Month", fontsize=12, color="red");
ax.set_ylabel("Count", fontsize=12, color="red");

plt.show()
# Year-wise Count of Complaints
fig,ax = plt.subplots(figsize=(20,6))
YvsC = sns.countplot(x='Year',data=data)

for label in ax.get_xticklabels():
    label.set_rotation(90)
    
ax.set_xlabel("Year", fontsize=12, color="red");
ax.set_ylabel("Count", fontsize=12, color="red");

plt.show()
# Srate-wise Count of Complaints
fig,ax = plt.subplots(figsize=(20,6))

SvsC = sns.countplot(x='State',data=data)

for label in ax.get_xticklabels():
    label.set_rotation(90)
    
ax.set_xlabel("State", fontsize=12, color="red");
ax.set_ylabel("Count", fontsize=12, color="red");

plt.show()
# Medium of Submission
fig,ax = plt.subplots(figsize=(20,6))
SMvsC = sns.countplot(x='Submitted_via',data=data)

for label in ax.get_xticklabels():
    label.set_rotation(90)
    
ax.set_xlabel("Submition Medium", fontsize=12, color="red");
ax.set_ylabel("Count", fontsize=12, color="red");

plt.show()
# Responses Provided
fig,ax = plt.subplots(figsize=(20,6))
RvsC = sns.countplot(x='Response',data=data)

for label in ax.get_xticklabels():
    label.set_rotation(90)
    
ax.set_xlabel("Response", fontsize=12, color="red");
ax.set_ylabel("Count", fontsize=12, color="red");

plt.show()

# Timely Responses are given or not
fig,ax = plt.subplots(figsize=(20,6))
TRvsC = sns.countplot(x='Timely_response',data=data)

for label in ax.get_xticklabels():
    label.set_rotation(90)
    
ax.set_xlabel("Timely Response", fontsize=12);
ax.set_ylabel("Count", fontsize=12);

plt.show()
FPVSCount = data.groupby(['Product', 'Month'], as_index=False).count()
FPVSCount = FPVSCount[['Product', 'Month', 'Count']]
FPVSCount.head(3)
g = sns.catplot(x="Product", y="Count", hue="Month", data=FPVSCount, height=8, kind="bar", aspect=2.2, legend=True)
for ax in g.axes.flat:
    for label in ax.get_xticklabels():
        label.set_rotation(90)
plt.show()
CC = data.Complaint_narrative
CC.head(3)