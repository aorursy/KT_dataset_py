# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
import os
from textwrap import wrap

# Set default fontsize and colors for graphs
SMALL_SIZE, MEDIUM_SIZE, BIG_SIZE = 10, 12, 20
plt.rc('font', size=MEDIUM_SIZE)       
plt.rc('axes', titlesize=BIG_SIZE)     
plt.rc('axes', labelsize=MEDIUM_SIZE)  
plt.rc('xtick', labelsize=MEDIUM_SIZE) 
plt.rc('ytick', labelsize=MEDIUM_SIZE) 
plt.rc('legend', fontsize=SMALL_SIZE)  
plt.rc('figure', titlesize=BIG_SIZE)
my_colors = 'rgbkymc'

# Disable scrolling for long output
from IPython.display import display, Javascript
disable_js = """
IPython.OutputArea.prototype._should_scroll = function(lines) {
    return false;
}
"""
display(Javascript(disable_js))
# Read the input training and test data
input_data_path = os.path.join("../input", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
train_data = pd.read_csv(input_data_path)
train_data.sample(5)
# Total number of records
print("Total number of records in training dataset:", train_data.shape)
# What are the features available and what are their data type?
train_data.info()
# Descriptive statistics of training data
train_data.describe().transpose()
# Is there any empty data in training dataset?
train_data.isnull().sum()/train_data.shape[0]
print("Total number of unique customerID values =", len(train_data.customerID.unique()))
fig, axes = plt.subplots(figsize=(8,5))
data = train_data["Churn"].value_counts(normalize=True)
axes.bar(data.index, data*100, color=['green', 'red'])
axes.set_title('Distribution of Churn %')
axes.set_ylabel('Percentage')
axes.set_xlabel('Did the customer leave?')
plt.show()
# Create column ChurnVal with - Yes = 1 and No = 0
churn_mapping = {"No": 0, "Yes": 1}
train_data['ChurnVal'] = train_data['Churn'].map(churn_mapping)
fig, axes1 = plt.subplots(figsize=(8,5))

# Plot distribution of gender data
data = train_data["gender"].value_counts(normalize=True)
axes1.bar(data.index, data*100, color=my_colors)
axes1.set_title('Distribution of Gender %')
axes1.set_ylabel('Percentage')
axes1.set_ylim(0,100)

plt.show()
fig, axes2 = plt.subplots(figsize=(12,8))

# Pie chart of churn percentage
width = 0.5

# Percentage of Churned vs Retained
data = train_data.Churn.value_counts().sort_index()
axes2.pie(
    data,
    labels=['Retained', 'Churned'],
    autopct='%1.1f%%',
    pctdistance=0.8,
    startangle=90,
    textprops={'color':'black', 'fontweight':'bold'},
    wedgeprops = {'width':width, 'edgecolor':'w'},
    radius=1,
)

# Percentage of Gender based on Churn
data = train_data.groupby(["Churn", "gender"]).size().reset_index()
axes2.pie(
    data.iloc[:,2], 
    labels=list(data.gender),
    autopct='%1.1f%%',
    startangle=90,
    textprops={'color':'white', 'fontweight':'bold'},
    wedgeprops = {'width':width, 'edgecolor':'w'},
    radius=1-width,
    rotatelabels=True,
)

axes2.set_title('What is the ratio of the customer\'s gender and churn?')
#axes2.legend(loc='best', bbox_to_anchor=(1,1))
axes2.axis('equal')

plt.show()
fig, axes3 = plt.subplots(figsize=(8,5))

# Chances of churn based on gender
sns.barplot(x="gender", y=train_data["ChurnVal"]*100, data=train_data, ci=None, ax=axes3)
axes3.set_xlabel('Gender')
axes3.set_ylabel('Churn %')
axes3.set_title('\n'.join(wrap('Among customers in each gender, what is the percentage of churn?', 35)))
axes3.set_ylim(0,100)
plt.show()
fig, (axes1, axes2) = plt.subplots(2, 1, figsize=(10,12))

# Plot distribution of SeniorCitizen data
data = train_data["SeniorCitizen"].value_counts(normalize=True).sort_index()
axes1.bar(data.index, data*100, color=my_colors)
axes1.set_title('Distribution of Senior Citizen %')
axes1.set_ylabel('Percentage')
axes1.set_xticks([0, 1])
axes1.set_xticklabels(['Non-Senior Citizen', 'Senior Citizen'])
axes1.set_ylim(0,100)

# Chances of churn based on gender
sns.barplot(x="SeniorCitizen", y="ChurnVal", data=train_data, ci=None, ax=axes2)
axes2.set_xlabel('Senior Citizen')
axes2.set_ylabel('Churn %')
axes2.set_title('Among senior citizens and other category, what is the percentage of churn?')
axes2.set_xticklabels(['Non-Senior Citizen', 'Senior Citizen'])
axes2.set_ylim(0,1)

plt.show()
fig, [axes1, axes2] = plt.subplots(1, 2, figsize=(15,6))

# Plot distribution of partner data
data = train_data["Partner"].value_counts(normalize=True)
axes1.bar(data.index, data*100, color=my_colors)
axes1.set_title('Distribution of Partner data %')
axes1.set_xlabel('Living with partner?')
axes1.set_ylabel('Percentage')
axes1.set_ylim(0,100)

# Chances of churn based on partner availibility
sns.barplot(x="Partner", y=train_data.ChurnVal*100, data=train_data, ci=None, ax=axes2)
axes2.set_xlabel('Living with partner?')
axes2.set_ylabel('Churn %')
axes2.set_title("\n".join(wrap('What is the chance of churn based on presence of a partner?', 30)))
axes2.set_ylim(0,100)

plt.show()
from textwrap import wrap
fig, [axes1, axes2] = plt.subplots(1, 2, figsize=(15,6))

# Plot distribution of partner data
data = train_data["Dependents"].value_counts(normalize=True)
axes1.bar(data.index, data*100, color=my_colors)
axes1.set_title('Distribution of Dependents data %')
axes1.set_ylabel('Percentage')
axes1.set_ylim(0,100)

# Chances of churn based on partner availibility
sns.barplot(x="Dependents", y=train_data.ChurnVal*100, data=train_data, ci=None, ax=axes2)
axes2.set_ylabel('Churn %')
axes2.set_title("\n".join(wrap('What is the chance of churn based on presence of dependents?', 30)))
axes2.set_ylim(0,100)

plt.show()
fig, [axes1, axes2] = plt.subplots(1,2,figsize=(15,5))

# Plot tenure occurance
sns.distplot(train_data.tenure, color='orange', ax=axes1)
axes1.set_title('% of Customers by Tenure')
axes1.set_xticks(np.arange(0, 100, 10))
axes1.set_xlim(0,95)

# Plot relation between tenure and churn
axes2.scatter(train_data.Churn, train_data.tenure, color='red', s=10, alpha=0.2)
axes2.set_xlabel('Churned?')
axes2.set_ylabel('Tenure')
axes2.set_title('Does tenure determine churn?')

plt.show()
# Divide the tenure into bins
bins = [0, 12, 24, 36, 48, 60, 72]
labels = ['1 yr', '2 yr', '3 yr', '4 yr', '5 yr', '6 yr']
train_data['tenureGroup'] = pd.cut(train_data["tenure"], bins, labels=labels)

# Draw a bar plot of tenure vs churn
fig, [axes1, axes2] = plt.subplots(1,2,figsize=(15,6))
sns.barplot(x="tenureGroup", y=train_data.ChurnVal*100, data=train_data, ci=None, ax=axes1)
axes1.set_xlabel('Tenure')
axes1.set_ylabel('Churn %')
axes1.set_ylim(0,100)
axes1.set_title('Does tenure determine churn?')

# Draw a bar plot of tenure vs churn vs contract
sns.barplot(x="tenureGroup", y=train_data.ChurnVal*100, hue="Contract", data=train_data, ci=None, ax=axes2)
axes2.set_xlabel('Tenure')
axes2.set_ylabel('Churn %')
axes2.set_ylim(0,100)
axes2.set_title('What role is contract playing in churn?')


plt.show()
from textwrap import wrap
fig, [axes1, axes2] = plt.subplots(1, 2, figsize=(15,6))

# Plot distribution of Contract data
data = train_data["Contract"].value_counts(normalize=True)
axes1.bar(data.index, data*100, color=my_colors)
axes1.set_title('Distribution of Contract data %')
axes1.set_ylabel('Percentage')
axes1.set_ylim(0,100)

# Chances of churn based on Contract
sns.barplot(x="Contract", y=train_data.ChurnVal*100, data=train_data, ci=None, ax=axes2)
axes2.set_ylabel('Churn %')
axes2.set_title("\n".join(wrap('What is the chance of churn based on Contract duration?', 30)))
axes2.set_ylim(0,100)

plt.show()
Services = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
n_cols = 2
n_rows = len(Services)
fig = plt.figure(figsize=(15,40))
#fig.suptitle('Distribution of Service Types and relation with Churn')
idx = 0

for serviceType in enumerate(Services):
    # Fetch data of Service Type
    data = train_data[serviceType[1]].value_counts(normalize=True).sort_index()

    # Now, plot the data
    i = 0
    for i in range(n_cols):
        idx+=1
        axes = fig.add_subplot(n_rows, n_cols, idx)

        # On column 1 - Plot the data distribution on bar plot
        if idx%2 != 0:
            axes.bar(data.index, data*100, color=my_colors)
        # On column 2 - Plot the percentage of churns on each service type
        else:
            sns.barplot(x=serviceType[1], y=train_data.ChurnVal*100, data=train_data, ci=None, ax=axes)

        if idx == 1 : axes.set_title('Distribution of service category data')
        if idx == 2 : axes.set_title('% of churn in each service category')
            
        axes.set_xlabel(serviceType[1])
        axes.set_ylabel('')
        axes.set_ylim(0,100)

fig.tight_layout()
plt.show()
from textwrap import wrap
fig, [axes1, axes2] = plt.subplots(1, 2, figsize=(15,6))

# Plot distribution of Contract data
data = train_data["PaperlessBilling"].value_counts(normalize=True)
axes1.bar(data.index, data*100, color=my_colors)
axes1.set_title('Distribution of Paperless Billing data %')
axes1.set_ylabel('Percentage')
axes1.set_ylim(0,100)

# Chances of churn based on Contract
sns.barplot(x="PaperlessBilling", y=train_data.ChurnVal*100, data=train_data, ci=None, ax=axes2)
axes2.set_ylabel('Churn %')
axes2.set_title("\n".join(wrap('What is the chance of churn based on bill delivery method?', 30)))
axes2.set_ylim(0,100)

plt.show()
from textwrap import wrap
fig, [axes1, axes2] = plt.subplots(1, 2, figsize=(15,6))

# Plot distribution of Contract data
data = train_data["PaymentMethod"].value_counts(normalize=True)
axes1.bar(data.index, data*100, color=my_colors)
axes1.set_title('Distribution of Payment Method data %')
axes1.set_ylabel('Percentage')
axes1.set_xticklabels(data.index, rotation=90)
axes1.set_ylim(0,100)

# Chances of churn based on Contract
sns.barplot(x="PaymentMethod", y=train_data.ChurnVal*100, data=train_data, ci=None, ax=axes2)
axes2.set_ylabel('Churn %')
[items.set_rotation(90) for items in axes2.get_xticklabels()]
axes2.set_title("\n".join(wrap('What is the chance of churn based on Payment Method?', 30)))
axes2.set_ylim(0,100)

plt.show()
fig, [axes1, axes2] = plt.subplots(1,2,figsize=(15,6))

# Plot MonthlyCharges occurance
sns.distplot(train_data.MonthlyCharges, color='orange', ax=axes1)
axes1.set_title('Distribution of Monthly Charges')
axes1.set_xlim(0,140)

# Categorize MonthlyCharges into bins and plot
train_data['MonthlyChargesCategory'] = pd.cut(train_data["MonthlyCharges"], bins=10)
sns.barplot(x='MonthlyChargesCategory', y='ChurnVal', data=train_data, ci=None, ax=axes2)
[items.set_rotation(90) for items in axes2.get_xticklabels()]
axes2.set_title('Churn rates based on monthly charges')

plt.show()
# Before plotting, we will temporarily drop missing values 
# and convert the column into float 
tmp_df = train_data[~train_data.TotalCharges.str.contains(" ")]
tmp_df['TotalCharges'] = pd.to_numeric(tmp_df['TotalCharges'])

# Now perform the calculations as described above, and plot
tenure_calc = tmp_df.TotalCharges/tmp_df.MonthlyCharges
fig, axes = plt.subplots(figsize=(10,5))
sns.kdeplot(tmp_df.tenure, marker='o', c='b', label="Actual", ax=axes)
sns.kdeplot(tenure_calc, marker='+', c='r', label="Calculated", ax=axes)
axes.set_xlabel('Tenure')
axes.legend()
plt.show()
train_data['TotalCharges'] = train_data['TotalCharges'].replace(" ", (train_data.MonthlyCharges * train_data.tenure))
train_data['TotalCharges'] = pd.to_numeric(train_data['TotalCharges'])
fig, [axes1, axes2] = plt.subplots(1,2,figsize=(15,6))

# Plot TotalCharges occurance
sns.distplot(train_data.TotalCharges, color='green', ax=axes1)
axes1.set_title('Distribution of Total Charges')
axes1.set_xlim(0,10000)

# Categorize TotalCharges into bins and plot
train_data['TotalChargesCategory'] = pd.cut(train_data["TotalCharges"], bins=20)
sns.barplot(x='TotalChargesCategory', y='ChurnVal', data=train_data, ci=None, ax=axes2)
[items.set_rotation(90) for items in axes2.get_xticklabels()]
axes2.set_title('Churn rates based on total charges')

plt.show()
features = ['tenure', 'MonthlyCharges', 'TotalCharges']
corr_matrix = train_data[features].corr()
mask = np.zeros_like(corr_matrix, dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True 

fig, axes = plt.subplots(figsize=(5,5))
sns.heatmap(corr_matrix, linewidths=0.25, vmax=1.0, square=True, cmap=plt.get_cmap('jet'),
            linecolor='w', annot=True, mask=mask, cbar_kws={"shrink": 0.8}, ax=axes)
axes.set_title('Correlation between numeric features', fontsize=15)
plt.show()