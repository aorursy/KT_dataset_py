# Import libraries.
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
# Function to crate a figure and get instance of Axis.
def axex(size):
    fig = plt.figure(figsize=(size[0],size[1])) # define plot area
    ax = fig.gca() # define axis  
    return ax
loan_data = pd.read_csv('../input/loan.csv', low_memory = False)
loan_data.describe()
loan_data.info()
# Count of different data-types
loan_data.dtypes.value_counts()
# Find number of unique occurences within column that has dtype = object
loan_data.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

#  Count of unique member id
loan_data['member_id'].value_counts().head()
# loan_data['loan_amnt'].isnull().sum()

# Function to find sum and percent of missing values for each column.
# Remove the one's that have no missing values.

def missing_values(df):
#     Sum null values. Then divide by the total occurrence
    sum = df.isnull().sum()
    percent = df.isnull().sum()/len(df)*100
    missing_stats = pd.concat([sum, percent], axis=1).rename(
        columns = {
            0: 'Number',
            1: 'Percent'
        }
    )
#   drop all rows that are equal to 0.
    missing_stats = missing_stats[missing_stats.iloc[:,1] != 0]
    missing_stats.reset_index(inplace=True)
    
    return missing_stats

missing_values(loan_data)
#  Removing columns with more than 70% of missing data.

temp = [i for i in loan_data.count()<887379 *0.30]
loan_data.drop(loan_data.columns[temp],axis=1,inplace=True)
loan_data.info()
# Find unique state entries

loan_data['addr_state'].unique()
# Statewise loan amount plot
ax = axex([15,15])
loan_data['addr_state'].value_counts().plot.bar(ax = ax)
loan_data['loan_amnt'].describe()
ax = axex([10,10])
loan_data['loan_amnt'].plot.hist(ax=ax)
# Add year to the data-frame

date = pd.to_datetime(loan_data['issue_d'])
date.head()
loan_data['year'] = date.dt.year
# Get info for loan amount distribution per year
ax = axex([10,10])
loan_data[['loan_amnt', 'year']].boxplot(by = 'year', ax = ax)
# Plot graph for average loan and income rate over the years.

fig, ax = plt.subplots(1, 2, figsize=(16,5))
# Average loan plot
loan_data[['loan_amnt', 'year']].groupby(['year']).mean().plot(ax = ax[0])
ax[0].set_title('Yearly Average Loan')
ax[1].set_ylabel('Average Loan')

# Average annual income plot
loan_data[['annual_inc', 'year']].groupby(['year']).mean().plot(ax = ax[1])
ax[1].set_title('Yearly Average Income')
ax[1].set_ylabel('Average Income')
# Plot loan occurances for every loan_status 
fig = plt.figure(figsize=(10,10))
axe = axex([10,10])
loan_data['loan_status'].value_counts().plot.bar(ax = axe)

axe.set_title('Loan Status Count')
axe.set_xlabel('Loan Status') # Set text for the x axis
axe.set_ylabel('Number of Loan')# Set text for y axis
grade_count = loan_data['grade'].value_counts()

# plot bar graph of term.
fig = plt.figure(figsize=(10,10))
axe = fig.gca()
grade_count.plot.bar()

axe.set_title('Grade Count')
axe.set_xlabel('Grade') # Set text for the x axis
axe.set_ylabel('Count')# Set text for y axis

# Grouping loan data by default, charged off, fully paid and current loan status

ax = axex([10,10])
new_df = loan_data[(loan_data['loan_status'] == 'Default') | (loan_data['loan_status'] == 'Charged Off') | (loan_data['loan_status'] == 'Fully Paid') | (loan_data['loan_status'] == 'Current')]
new_df['loan_status'].value_counts().plot.bar(ax=ax)

ax.set_title('Status vs Loan Amount')
ax.set_xlabel('Loan Status')
ax.set_ylabel('Loan Amount')
# Plot default loan count for every grade

ax = axex([10,10])
loan_data[loan_data['loan_status']=='Default'].groupby('grade')['loan_status'].count().plot.bar()

ax.set_title('Grade vs Default Loan Count')
ax.set_xlabel('Grade')
ax.set_ylabel('Default Loan Count')
# Box plot of loan amount vs loan status to see distribution of loan for each status.
ax = axex([10,10])
new_df[['loan_amnt', 'loan_status']].boxplot(by = 'loan_status', ax = ax)

ax.set_title('Loan Amount vs Loan Status')
ax.set_xlabel('Loan Status')
ax.set_ylabel('Loan Amount')
# Get interest rate distribution for every grade

ax = axex([10,10])
loan_data[['int_rate', 'grade']].boxplot(by = 'grade', ax = ax)

ax.set_title('Interest Rate vs Grade')
ax.set_ylabel('Grade')
ax.set_ylabel('Interest Rate')
# Plot interest rate for 4 chosen loan status. 

ax = axex([10,10])

new_df[['int_rate', 'loan_status']].boxplot(by='loan_status', ax = ax)
ax.set_title('Interest Rate vs Loan Status')
ax.set_xlabel('Status')
ax.set_ylabel('Rate')
# new_df['installment'].head()
# new_df['pymnt_plan'].loc[new_df['loan_status'] == 'Default'].value_counts().plot.bar()
# df[['loan_status', 'pymnt_plan']].groupby(['pymnt_plan']).count()
# new_df[['pymnt_plan', 'loan_status']].barplot()
# plot occurances of 36/60 month term loans.

term_count = loan_data['term'].value_counts()
print (term_count)

# plot bar graph of term.
fig = plt.figure(figsize=(10,10))
axe = fig.gca()
term_count.plot.bar()

axe.set_xlabel('Term') # Set text for the x axis
axe.set_ylabel('Number')# Set text for y axis
new_df.groupby("term").agg({"int_rate":np.mean, "loan_amnt": np.mean}, axis = 1).reset_index()
# Dropping Current from new_df
new_df = new_df.drop(new_df[new_df['loan_status'] == 'Current'].index)
new_df = new_df.drop('addr_state', axis=1)

# Make default and charged off as 1
new_df['Target'] = new_df['loan_status'].apply(lambda x: 1 if  x == 'Default' or x == 'Charged Off' else 0)
new_df.info()
new_df.head()

#  Check correlation

corr = new_df.corr()['Target'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', corr.tail(10))
print('\nMost Negative Correlations:\n', corr.head(10))

#  Values to Drop
new_df.drop([    
                'mths_since_last_delinq',
                'total_pymnt',
                'total_rec_prncp',           
                'total_rec_int',            
                'initial_list_status',
                'last_pymnt_amnt',
                'collections_12_mths_ex_med',
                'policy_code',
                'acc_now_delinq',
                'open_acc',
                'revol_util',
                'revol_bal',
                'total_acc',
                'total_pymnt_inv',
                'tot_coll_amt',
                'tot_cur_bal',
                'total_rev_hi_lim',           
                 'id',
                 'member_id',
                 'emp_title',
                 'title',
                 'url',
                 'zip_code',
                 'policy_code',
                 'verification_status',
                 'home_ownership',
                 'issue_d',
                 'earliest_cr_line',
                 'last_pymnt_d',
                 'next_pymnt_d',
                'initial_list_status',
                 'last_credit_pull_d',
                 'inq_last_6mths',
                 'sub_grade',
                 'loan_status',
                 'year'

                                    ], axis=1, inplace=True)

# Heatmap of final features

import seaborn as sns
heat_df = new_df.drop(list(new_df.select_dtypes('object')), axis=1)
corr = heat_df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr,
            annot=True, fmt=".3f",
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
heat_df.info()
# sns.heatmap(heat_df)
# Drop features that are highly correlated with each other.

new_df.drop(['funded_amnt', 'funded_amnt_inv', 'installment', 'out_prncp_inv', 'collection_recovery_fee'], axis=1, inplace=True)

# Convert emp_length to int
new_df['emp_length'].fillna(value=0,inplace=True)
new_df['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
new_df.info()
missing_values(new_df)
new_df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)
# Label encode categorical var with 2 unique entries. 
proc_df = new_df.copy(deep=True)

# gives the categories an arbitrary ordering

count = 0
for col in proc_df:
    if proc_df[col].dtype == 'object':
        if len(list(proc_df[col].unique())) <= 2:     
            le = preprocessing.LabelEncoder()
            proc_df[col] = le.fit_transform(proc_df[col])
            count += 1
            print (col)
            
print('%d columns were label encoded.' % count)
proc_df.head()
# One Hot encoding other categorical variables.
proc_df = pd.get_dummies(proc_df)
# Convert features to uint8 to reduce memory
# describe to check min/max since max int8 takes is 128
print(proc_df[['term', 'application_type', 'pymnt_plan']].describe())

proc_df[['Target', 'term', 'application_type', 'pymnt_plan']] = \
proc_df[['Target', 'term', 'application_type', 'pymnt_plan']].astype('uint8')
proc_df.info()
# Percent of Default to Good loans
proc_df['Target'].value_counts()/len(proc_df)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
# from sklearn import metrics, cross_validation
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(proc_df.drop('Target',axis=1),proc_df['Target'],test_size=0.20,random_state=154)
print(len(X_train))
print(len(y_train))
np_y_train = y_train.values
np_X_train = X_train.values
total_num_of_ones = int(np.sum(y_train))
# print(total_num_of_ones)
zero_counter = 0
indices_to_remove = []

for i in range(np_y_train.shape[0]):
    if np_y_train[i] == 0:
        if zero_counter < total_num_of_ones:
            zero_counter += 1
        else:
            indices_to_remove.append(i)

X_train = np.delete(np_X_train, indices_to_remove, axis=0)
y_train = np.delete(np_y_train, indices_to_remove, axis=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.transform(X_test)
# # sm = SMOTE(sampling_strategy= float)
# # x_train_r, y_train_r = sm.fit_sample(X_train, y_train)

# sm = SMOTE(random_state=12, ratio = 1.0)
# x_train_r, y_train_r = sm.fit_sample(X_train, y_train)
# Init logisticRegressor
logreg = LogisticRegression()

log_reg = LogisticRegression(random_state=21)
# Train the model on balanced data.
log_reg.fit(X_train, y_train)

res = cross_val_score(log_reg, X_train, y_train, cv=10, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
# Predicted based on test dataset.
y_pred = log_reg.predict(X_test)
# confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# Calculaute precision to find % of correctly classified bad loans.
precision = cm[0,0] / (cm[0,0] + cm[0,1])
print('Precision', (precision)*100)
print('Confusion Matrix:\n', cm)
# prediction accruacy score
log_reg.score(X_test, y_test)

