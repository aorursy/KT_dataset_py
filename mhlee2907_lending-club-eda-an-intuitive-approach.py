import pandas as pd
import matplotlib.pyplot as plt  
import math
from math import pi
import numpy as np
import seaborn as sns
import warnings  
warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore')
%matplotlib inline 
# Load data
loan_df = pd.read_csv('../input/lending-club-loan-data/loan.csv',header=0,skip_blank_lines=True)

# Get the first 5 rows of the data
loan_df.head()
# Print information of the dataframe
loan_df.info()
# Drop columns with more than 80% of missing values
na_thresh = len(loan_df)*80/100
loan_df = loan_df.dropna(thresh=na_thresh, axis=1)

# Check the new dimension
loan_df.shape
# Convert 'issue_d' column to datetime format
loan_df['issue_d'] = pd.to_datetime(loan_df['issue_d'])
year_dist = loan_df.groupby(['issue_d']).size()

plt.figure(figsize=(15,6))
sns.set()

ax1 = plt.subplot(1, 2, 1)
ax1 = year_dist.plot()
ax1 = plt.title('Loan Issued Amount By Year')
ax1 = plt.xlabel('Year')
ax1 = plt.ylabel('Frequency')

ax2 = plt.subplot(1, 2, 2)
ax2 = sns.distplot(loan_df['loan_amnt'])
ax2 = plt.title('Loan Amount Distribution')
ax2 = plt.xlabel('Loan Amount')
# Create a region dictionary from US regions dateset
regions = pd.read_csv('../input/region/regions.csv',header=0,skip_blank_lines=True)
regions_dict = pd.Series(regions['region'].values,index=regions['state']).to_dict()

# Create a new column 'region' by mapping the state value to region from the region dict
loan_df['region'] = loan_df['addr_state'].map(regions_dict)

region_dist = loan_df.groupby(['region']).size()
explode = (0, 0, 0.1, 0) 
colors = ["#efc186","#a6ddb5","#dda19b","#9ebfe2"]
plt.figure(figsize=(15,6))
ax1 = plt.subplot(1, 2, 1)
ax1 = plt.pie(region_dist,autopct='%1.1f%%',explode=explode, labels=region_dist.index,colors = colors)
ax1 = plt.title('Loan Distribution By Region')
ax1 = plt.xlabel('Region')
ax1 = plt.axis('equal')  

ax2 = plt.subplot(1, 2, 2)
ax2 = sns.countplot(y=loan_df['purpose'],order = loan_df['purpose'].value_counts().index)
ax2 = plt.title('Loan Purpose Distribution')
ax2 = plt.ylabel('Loan Purposes')
ax2 = plt.xlabel('Frequency')
ax2 = plt.tight_layout()
plt.figure(figsize=(15,6))
ax = sns.countplot(y=loan_df['loan_status'],order = loan_df['loan_status'].value_counts().index)
ax = plt.title('Loan Status Distribution')
# Create a new status dictionary
new_status_dict = {
    'Fully Paid': 'Fully Paid',
    'Charged Off': 'Charged Off',
    'Current': 'Current',
    'Default': 'Default',
    'Late (31-120 days)': 'Late',
    'In Grace Period': 'Late',
    'Late (16-30 days)': 'Late',
    'Does not meet the credit policy. Status:Fully Paid': 'Fully Paid',
    'Does not meet the credit policy. Status:Charged Off': 'Charged Off',
    'Issued': 'Issued'
}

loan_df['new_status'] = loan_df['loan_status'].map(new_status_dict)
pd.Series(pd.unique(loan_df['new_status'])).to_frame()
plt.figure(figsize=(15,6))

ax = sns.lineplot(x="issue_d", y="loan_amnt",hue="new_status", data=loan_df)
ax = plt.title('Time Series Plot of Loan Amount By Loan Status')
ax = plt.xlabel('Year')
ax = plt.ylabel('Loan Amount')
plt.figure(figsize=(15,6))

region_freq = loan_df.groupby(['region','issue_d']).size().unstack().fillna(0)
dates = np.sort(pd.unique(loan_df['issue_d']))

ax = plt.fill_between(dates,region_freq.values[2],label='South')
ax = plt.fill_between(dates,region_freq.values[3],label='West')
ax = plt.fill_between(dates,region_freq.values[1],label='Northeast')
ax = plt.fill_between(dates,region_freq.values[0],label='Midwest')

ax = plt.legend(loc='upper left')
ax = plt.xlabel('Year')
ax = plt.ylabel('Frequency')
ax = plt.title('Time Series Plot of Loan Issued By Loan Status')
plt.figure(figsize=(15,7))

ax = sns.violinplot(x="region", y="loan_amnt", data=loan_df)
ax = plt.xlabel('Regions')
ax = plt.ylabel('Loan Amount')
ax = plt.title('Violin Plot of Loan Amount Against Region')
plt.figure(figsize=(15,9))
ax = sns.boxplot(y="purpose", x="loan_amnt", data=loan_df)
ax = plt.xlabel('Loan Amount')
ax = plt.ylabel('Loan Purpose')
ax = plt.title('Loan Amount Versus Purpose')
plt.figure(figsize=(13,7))
plot_data = loan_df.groupby(['purpose','region']).size().reset_index()
plot_data.columns = ['purpose','region','value']
plot_data = plot_data.pivot('purpose','region','value')

ax = sns.heatmap(plot_data,annot=True, fmt="d",cmap="GnBu",linewidths=.2)
ax = plt.xlabel('Region')
ax = plt.ylabel('Loan Purpose')
ax = plt.title('Purpose Versus Region By Frequency')
# Here we exclude newly issued loan as they don't give any significant insight 
loan_df = loan_df[loan_df['loan_status'] != 'Issued']

# Categorize loan into 'Good' or 'Bad'
# Create dictionary that map loan status to loan type
loan_type_dict = {
    'Fully Paid': 'Good',
    'Charged Off': 'Bad',
    'Current': 'Good',
    'Default': 'Bad',
    'Late (31-120 days)': 'Bad',
    'In Grace Period': 'Bad',
    'Late (16-30 days)': 'Bad',
    'Does not meet the credit policy. Status:Fully Paid': 'Good',
    'Does not meet the credit policy. Status:Charged Off': 'Bad'
}

loan_df['loan_type'] = loan_df['loan_status'].map(loan_type_dict)
plt.figure(figsize=(12,6))
ax = sns.countplot(x="loan_type", data=loan_df,palette="Set2")
ax = plt.xlabel('Loan Type')
ax = plt.ylabel('Frequency')
ax = plt.title('Loan Type Distribution')
plt.figure(figsize=(10,6))

pal = {"Good": "#6bad97", "Bad": "#d8617f"}

ax = sns.violinplot(x="loan_type", y="loan_amnt", data=loan_df,palette=pal)
ax = plt.xlabel('Loan Type')
ax = plt.ylabel('Loan Amount')
ax = plt.title('Loan Amount Distribution')
plt.figure(figsize=(10,6))
plot_data = loan_df.groupby(['region','loan_type']).size().unstack().T
r = range(4)

ax = plt.bar(r, plot_data.values[0], color='#f9966b', edgecolor='white',label='Bad')

ax = plt.bar(r, plot_data.values[1], bottom=plot_data.values[0], color='#7fd1b8', edgecolor='white',label='Good')
names = plot_data.columns
ax = plt.xticks(r, names)
ax = plt.legend(loc='upper left')
ax = plt.xlabel('Region')
ax = plt.ylabel('Frequency')
ax = plt.title('Loan Type Distribution By Region')
plt.figure(figsize=(11,6))

ax = sns.boxplot(x="loan_type", y="annual_inc",  data=loan_df,showfliers=False,palette=pal)
ax = plt.xlabel('Loan Type')
ax = plt.ylabel('Annual Income')
ax = plt.title('Loan Type Distribution By Annual Income')
plt.figure(figsize=(15,7))
plot_data = loan_df.groupby(['emp_length','loan_type']).size().unstack().T
cols = ['< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years']
plot_data = plot_data[cols]
r = range(11)

ax = plt.bar(r, plot_data.values[0], color='#f9966b', edgecolor='white',label='Bad')

ax = plt.bar(r, plot_data.values[1], bottom=plot_data.values[0], color='#7fd1b8', edgecolor='white',label='Good')
names = plot_data.columns
ax = plt.xticks(r, names)
ax = plt.legend(loc='upper right')
ax = plt.xlabel('Employment Length')
ax = plt.ylabel('Frequency')
ax = plt.title('Loan Type Distribution By Employment Length')
plt.figure(figsize=(20,10))
plot_data = loan_df.groupby(['loan_type','purpose']).size().reset_index()
plot_data.columns = ['loan_type','purpose','value']

# Apply logarithm to the frequency value to reduce skewness
plot_data['value'] = plot_data['value'].apply(math.log)

# Retrieve good loans
good_loan = plot_data[plot_data['loan_type'] == 'Good']

# Retrieve bad loans
bad_loan = plot_data[plot_data['loan_type'] == 'Bad']

N = len(pd.unique(good_loan['purpose']))
good_values = good_loan.value.values
good_values = np.append(good_values, [good_values[:1]])

bad_values = bad_loan.value.values
bad_values = np.append(bad_values, [bad_values[:1]])

angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

ax = plt.subplot(1,2,1, polar=True)
ax = plt.xticks(angles[:-1], pd.unique(good_loan['purpose']), color='grey', size=8)
ax = plt.plot(angles, good_values, linewidth=1, linestyle='solid',label='Good',color='g')
ax = plt.fill(angles, good_values, 'g', alpha=0.3)
ax = plt.xlabel('Loan Purpose')

ax2 = plt.subplot(1, 2, 1,polar=True)
ax2 = plt.xticks(angles[:-1], pd.unique(bad_loan['purpose']), color='grey', size=8)
ax2 = plt.plot(angles, bad_values, linewidth=1, linestyle='solid',label='Bad',color='r')
ax2 = plt.fill(angles, bad_values, 'r', alpha=0.3)
ax2 = plt.xlabel('Loan Purpose')

ax = plt.legend(loc='upper left')
plot_data1 = loan_df.groupby(['home_ownership','loan_type']).size().unstack().fillna(0)
plot_data1['total'] = plot_data1['Bad'] + plot_data1['Good']
cols = ['ANY','MORTGAGE','NONE','OWN','OTHER','RENT']
plot_data1 = plot_data1.reindex(index=cols)

plot_data2 = loan_df.groupby(['home_ownership','loan_type']).size().reset_index().fillna(0)
plot_data2.columns = ['home_ownership', 'loan_type','value']
plot_data2 = plot_data2.reindex([0,1,2,3,4,7,8,5,6,9,10])
plt.figure(figsize=(8,8))
ax = plt.axis('equal')

# Plot the outer circle
pie, _ = plt.pie(np.array(plot_data1['total']), radius=1.3,labels=plot_data1.index,colors=['#13194c','#f2da32','#2fb74d','#b71d60','#91683a','#545ca8'])
ax = plt.setp(pie, width=0.3, edgecolor='white')
 
# Plot the inner circle
pie2, _ = plt.pie(plot_data2.value.values, radius=1.3-0.3, labels=plot_data2['loan_type'].values, labeldistance=0.7,
                   colors = ['#91683a','#f7efc0','#efe294','#b7ba3f','#b7ba3f','#d192c5','#e256a3','#b7ba3f','#b7ba3f','#99e2ff','#75aec6'])
ax = plt.setp(pie2, width=0.4, edgecolor='white')
ax = plt.margins(0,0)

plt.figure(figsize=(10,6))
plot_data = loan_df.groupby(['grade','loan_type']).size().unstack().T
r = range(7)

ax = plt.bar(r, plot_data.values[0], color='#f9966b', edgecolor='white',label='Bad')
ax = plt.bar(r, plot_data.values[1], bottom=plot_data.values[0], color='#7fd1b8', edgecolor='white',label='Good')
names = plot_data.columns
ax = plt.xticks(r, names)
ax = plt.legend(loc='upper left')
ax = plt.xlabel('Loan Grade')
ax = plt.ylabel('Frequency')
ax = plt.title('Loan Type Distribution By Loan Grade')
plt.figure(figsize=(10,8))
ax = sns.violinplot(y="int_rate", x="loan_type", data=loan_df)
ax = plt.xlabel('Loan Type')
ax = plt.title('Loan Type Distribution By Interest Rate')
plt.figure(figsize=(15,6))
ax1 = plt.subplot(1, 2, 1)
plot_data = loan_df
plot_data['cut'] = pd.cut(plot_data['loan_amnt'],5)
ax = sns.violinplot(x="cut", y="int_rate", hue="loan_type",data=plot_data,split=True)
ax = plt.ylabel('Interest Rate')
ax = plt.xlabel('Loan Amount')
ax = plt.title('Loan Amount Distribution By Loan Type')

ax2 = plt.subplot(1, 2, 2)
plot_data = loan_df.sample(frac=0.005)
ax2 = sns.regplot(x="loan_amnt", y="int_rate", data=plot_data,color='#c96895')
ax2 = plt.tight_layout()
ax = plt.ylabel('Interest Rate')
ax = plt.xlabel('Loan Amount')
ax = plt.title('Interest Rate Versus Loan Amount')
plt.figure(figsize=(10,6))
plot_data = loan_df.groupby('grade')['int_rate'].mean()
ax = sns.barplot(x=plot_data.index,y=plot_data.values,palette='OrRd')
ax = plt.xlabel('Loan Grade')
ax = plt.ylabel('Interest Rate')
ax = plt.title('Interest Rate Versus Loan Grade')
plt.figure(figsize=(13,7))
plot_data = loan_df.groupby('purpose')['int_rate'].mean()
ax = sns.barplot(y=plot_data.index,x=plot_data.values,palette='BuPu')
ax = plt.ylabel('Loan Purpose')
ax = plt.xlabel('Interest Rate')
ax = plt.title('Interest Rate Versus Loan Grade')
plt.figure(figsize=(13,7))
plot_data = loan_df.sample(frac=0.005,random_state=1)
sns.set_style("white")
ax = sns.regplot(x="annual_inc", y="int_rate", data=plot_data,color='#dbdb64')
ax = plt.xlabel('Annual Income')
ax = plt.ylabel('Interest Rate')
ax = plt.title('Interest Rate Versus Annual Income')
plt.figure(figsize=(13,7))

cols = ['< 1 year','1 year','2 years','3 years','4 years','5 years','6 years','7 years','8 years','9 years','10+ years']

ax = sns.boxplot(x="emp_length", y="annual_inc",  data=loan_df,order=cols,showfliers=False,palette='Spectral')

ax = plt.ylabel('Annual Income')
ax = plt.xlabel('Employment Length')
ax = plt.title('Annual Income Versus Employment Length')
# Create a 'target' column based on 'loan type' column
target_dict = {
    'Good': 0,
    'Bad' : 1
}

loan_df['target'] = loan_df['loan_type'].map(target_dict)

# Create dummy variables for catagorical features (encoding)
dummies_df =pd.get_dummies(loan_df,columns=['term','grade','sub_grade','emp_length','home_ownership','pymnt_plan','purpose','addr_state','initial_list_status','application_type'])
dummies_df.dropna(inplace=True)

# Get input variables
X_df = dummies_df.loc[:, ~(dummies_df.columns.isin(pd.Series(loan_df.select_dtypes(['object']).columns)))]

# Get target variable
y = dummies_df['target'].values

X_df = X_df.loc[:, ~(X_df.columns.isin(['target','id','member_id','cut','issue_d']))]

# Create an input matrix 
X = X_df.values
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

# Initialize a random forest estimator
forest = ExtraTreesClassifier(n_estimators=50,random_state=0)

# Fit the input matrix and target values to the classifier
forest.fit(X, y)

# Retrieve the computed feature importance
importances = forest.feature_importances_

# Plot the feature importance
plot_data = pd.DataFrame({'features' : pd.Series(X_df.columns),'importance' : pd.Series(importances)})
plt.figure(figsize=(15,10))
plot_data = plot_data.sort_values('importance',ascending=False)
plot_data = plot_data[plot_data['importance'] > 0.01]
ax = sns.barplot(x=plot_data['importance'],y=plot_data['features'],)
plt.figure(figsize=(15,9))

# Calculate correlation between each pair of variable
corr_matrix=loan_df.corr()

mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask)] = True

# Plot a correlation heatmap
with sns.axes_style("white"):
    ax = sns.heatmap(corr_matrix, mask=mask, square=True)