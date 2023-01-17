# necessary libraries 
import numpy as np 
import pandas as pd 
import datetime as dt

# for ploting and exploratory data analysis 
import seaborn as sns 
import matplotlib.pyplot as plt 
data = pd.read_csv("//kaggle//input//us-consumer-finance-complaints//consumer_complaints.csv", low_memory=False)  
data.head() 
print(data.shape) 
for col in data.columns: 
    print(col,":",data[col].nunique(dropna=True)) 
data = data.set_index("complaint_id") 
data.head() 
# this function returns percentage of data missing from each column
def missing_summary(data): 
    total_rows = data.shape[0] 
    missing_rows = data.isnull().sum() 
    missing_summary = dict() 
    for i in range(0, missing_rows.shape[0]):
        missing_summary[missing_rows.index[i]] = round(missing_rows[i] / total_rows, 4) * 100
    return missing_summary 

missing_data = missing_summary(data) 
missing_data = pd.DataFrame.from_dict(missing_data, orient="index", columns=["Percentage Missing"]).reset_index()
missing_data.sort_values(by="Percentage Missing", ascending=False)  
data.tags.value_counts()
plt.figure(figsize=(11, 6))
sns.countplot(data.tags) 
plt.show() 
data["tags"] = data.tags.replace(to_replace=np.nan, value="Others") 
data.tags[:10]
data.tags.value_counts()
plt.figure(figsize=(11, 6))
sns.countplot(data.tags) 
plt.show() 
data["consumer_consent_provided"].unique() 
data.drop("consumer_consent_provided", axis=1, inplace=True) 
data["consumer_complaint_narrative"].unique() 
data.drop("consumer_complaint_narrative", axis=1, inplace=True) 
data["company_public_response"].value_counts()
# total number of complaints where a company chose to respond
data["company_public_response"].value_counts().sum() 
sizes = data["company_public_response"].value_counts()
labels = ['Responded to consumer and CFPB', 'No public response', 'Company acted within law', 'Misunderstanding',
          'Disputes the facts', 'Actions of third party', 'Isolated error', "Can't verify the facts", 
          'Room for improvement in service', 'Discontinued policy']

cmap = plt.get_cmap("tab20c") 
colors = cmap(np.arange(10) * 2)
fig1, ax1 = plt.subplots()
fig1.set_size_inches(21,21)
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, colors=colors, labeldistance=1.05,  
        textprops={'fontsize': 14})
ax1.axis('equal')
plt.show()
# replacing null values in Company public response with "No response" 
data["company_public_response"].fillna("No response", inplace=True) 
# checking if any null entries left
print("Missing entries in column -", data["company_public_response"].isnull().sum())  
print("Company public response") 
data["company_public_response"].value_counts() 
data[data["consumer_disputed?"].isnull()]["company_public_response"].value_counts() 
data[data.state.isnull()] 
# check for number of missing rows
data.state.isnull().sum() 
Statewise_Product_complaints = data.groupby("state")[["product"]].agg('count') 
Statewise_Product_complaints = Statewise_Product_complaints.sort_values("product", ascending=False)

# ploting statewise product usage
plt.figure(figsize=(19,19))
sns.barplot(x="product", y=Statewise_Product_complaints.index, data=Statewise_Product_complaints, palette="Blues_d") 
plt.title("Statewise Complaints")
plt.show()
Products_Across_State = pd.crosstab(data.state, data.product, normalize="index") 
Products_Across_State = Products_Across_State.T
Products_Across_State.head() 
plt.figure(figsize=(20,20))
yticks = Products_Across_State.index
keptticks = yticks[::int(len(yticks)/10)]
yticks = ['' for y in yticks]
yticks[::int(len(yticks)/10)] = keptticks

xticks = Products_Across_State.columns
keptticks = xticks[::int(len(xticks)/10)]
xticks = ['' for y in xticks]
xticks[::int(len(xticks)/10)] = keptticks

sns.heatmap(Products_Across_State, yticklabels=yticks, xticklabels=xticks, square=True, 
            cbar_kws={'fraction' : 0.01}, cmap='OrRd', linewidth=1.5)

# This sets the yticks "upright" with 0, as opposed to sideways with 90.
plt.yticks(rotation=0) 

plt.show()
from scipy.stats import chi2_contingency 

chi_stat, p_value, dof, e_table = chi2_contingency(Products_Across_State)  
print("Chi Statistic = ", chi_stat) 
print("P-value =", p_value) 
e_table
data.Issue.value_counts()
Statewise_Issues = pd.crosstab(data.State, data.Issue) 
Statewise_Issues
chi_stat, p_value, dof, e_table = chi2_contingency(Statewise_Issues)  
print("Chi Statistic = ", chi_stat) 
print("P-value =", p_value) 
data.drop("ZIP code", axis=1, inplace=True) 
data.columns
# converting string to datetime 
data["Date received"] = pd.to_datetime(data["Date received"]) 
data["Date sent to company"] = pd.to_datetime(data["Date sent to company"])  
data["Forwarding time"] = data["Date sent to company"] - data["Date received"]  
data.head() 
data.drop(["Date received", "Date sent to company", "Forwarding time"], axis=1, inplace=True)
data.columns
data.drop(["Sub-product", "Sub-issue"], axis=1, inplace=True)
data.columns
common_response = data["Consumer disputed?"].mode()
common_response = "No"
# replacing with most common consumer response
data["Consumer disputed?"].fillna(common_response, inplace=True)
plt.figure(figsize=(21, 6)) 
chart = sns.countplot(data[data["Timely response?"] == "Yes"]["Product"], palette='Set1')    
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right', fontweight='light', fontsize='x-large')
plt.title("Timely Response? Yes")
plt.show() 
plt.figure(figsize=(21, 6)) 
chart = sns.countplot(data[data["Timely response?"] == "No"]["Product"], palette='Set1')    
chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right', fontweight='light', fontsize='x-large')
plt.title("Timely Response? No")
plt.show() 
submitted = data["Submitted via"].value_counts() 
itr = 0
for i in submitted:
    print(submitted.index[itr], round(i / data.shape[0], 4))  
    itr += 1 
not_timely = data[data["Timely response?"] == "No"] 
submitted = not_timely["Submitted via"].value_counts() 
itr = 0
for i in submitted:
    print(submitted.index[itr], round(i/data.shape[0], 4))  
    itr += 1                    
timely = data[data["Timely response?"] == "Yes"] 
submitted = timely["Submitted via"].value_counts() 
itr = 0
for i in submitted:
    print(submitted.index[itr], round(i/data.shape[0], 4))  
    itr += 1  
data.drop("Submitted via", axis=1, inplace=True)
# revaluating missing data
missing_data = missing_summary(data)  
missing_data = pd.DataFrame.from_dict(missing_data, orient="index", columns=["Percentage Missing"]).reset_index()
missing_data.sort_values(by="Percentage Missing", ascending=False)  
data.dropna(inplace=True)  
# revaluating missing data
missing_data = missing_summary(data)  
missing_data = pd.DataFrame.from_dict(missing_data, orient="index", columns=["Percentage Missing"]).reset_index()
missing_data.sort_values(by="Percentage Missing", ascending=False)  
# encoding the data for clustering
cols = data.columns
for col in cols:
    data[col]=data[col].astype('category')

encoded_data = pd.get_dummies(data[cols], columns=cols)
encoded_data.shape
Consumer_Complaint_Encoded_Sample = encoded_data.sample(frac=0.1) 
Consumer_Complaint_Encoded_Sample.head() 
Consumer_Complaint_Encoded_Sample.shape
Consumer_Complaint_Encoded_Sample.to_csv("Consumer Compaints Dataset Encoded Sample.csv", index=False) 
data.loc[Consumer_Complaint_Encoded_Sample.index].to_csv("Consumer Compaints Dataset Sample.csv", index=False)