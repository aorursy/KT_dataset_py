#imports

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt



% matplotlib inline



import os

data_path = os.path.join('../input','loan.csv')



loan = pd.read_csv(data_path, low_memory=False)



#for predictive analysis

from sklearn.feature_selection import chi2

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
print(loan.info())
loan.head(5)
missing_val_count_by_column = (loan.isnull().sum())

total_entries = len(loan)*len(loan.columns)

percent_missing_entries = sum(missing_val_count_by_column)/total_entries

percent_missing_columns = len(missing_val_count_by_column[missing_val_count_by_column > 0.8*len(loan)])/len(loan.columns)



print("Total missing entries:")

print(sum(missing_val_count_by_column),'\n')

print("Percentage of missing entries:")

print(percent_missing_entries*100, '\n')

print("Percentage of columns with more than 80 percent missing values:")

print(percent_missing_columns*100, '\n')

print("Columns with more than 80 percent missing values:")

print(missing_val_count_by_column[missing_val_count_by_column > 0.8*len(loan)])
loan['date_time'] = pd.to_datetime(loan['issue_d'])
plt.figure(figsize=(14,5))



for x in ['loan_amnt','funded_amnt','funded_amnt_inv']:

    loan.groupby('date_time').mean()[x].plot(label=x)



plt.title('Mean of Loan Amounts')

plt.xlabel('Year')

plt.ylabel('Dollars')

plt.legend(loc='center left', bbox_to_anchor = (1,0.5))
plt.figure(figsize=(14,5))



loan.groupby('date_time').mean()['int_rate'].plot()



plt.title('Mean of Interest Rate (%)')

plt.xlabel('Year')

plt.ylabel('%')
plt.figure(figsize=(14,5))



loan.groupby('date_time').std()['int_rate'].plot()



plt.title('Standard Deviation of Interest Rate (%)')

plt.xlabel('Year')

plt.ylabel('%')
plt.figure(figsize=(14,5))



loan.groupby('date_time').median()['int_rate'].plot()



plt.title('Median of Interest Rate (%)')

plt.xlabel('Year')

plt.ylabel('%')
loan['year'] = loan['date_time'].dt.year
plt.figure(figsize=(12,5))

loan.groupby('year').count()['loan_status'].plot()

plt.xlabel('Year')

plt.ylabel('Count')

plt.title('Number of Loans')
plt.figure(figsize=(15,8))

sns.countplot(x='year',hue='loan_status',data=loan)

plt.title('Loan Status')

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
print('Count of Loan Status:')

print(loan['loan_status'].value_counts(), '\n')

print('Proportion of Fully Paid Loans (amongst completed loans):')

print(loan['loan_status'].value_counts()['Fully Paid']/np.sum([loan['loan_status'] != 'Current'])*100)
def fully_paid(x):

    if x == 'Fully Paid':

        return int(1)

    else:

        return int(0)
completed_loan = loan.copy()

completed_loan = completed_loan[completed_loan['loan_status'] != 'Current']

completed_loan['fully_paid_dummy'] = completed_loan['loan_status'].apply(fully_paid)
plt.figure(figsize=(15,8))

sns.countplot(x='year',hue='grade',data=completed_loan,hue_order=['A','B','C','D','E','F','G'])

plt.title('Grade')

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
print(completed_loan.groupby('grade')['fully_paid_dummy', 'int_rate'].mean())
print('Summary Stats for Annual Income (amongst completed loans):')

print(completed_loan.describe()['annual_inc'])
# Creating bins for income



p1 = np.nanpercentile(np.array(completed_loan['annual_inc']),25)

p2 = np.nanpercentile(np.array(completed_loan['annual_inc']),50)

p3 = np.nanpercentile(np.array(completed_loan['annual_inc']),75)





def income_bin(x):

    if x <= p1:

        return 'Low'

    elif x <= p2:

        return 'Middle_Low'

    elif x <= p3:

        return 'Middle_High'

    else:

        return 'High'
completed_loan['income_group'] = completed_loan['annual_inc'].apply(income_bin)
print(completed_loan.groupby(['income_group','home_ownership'])['fully_paid_dummy','int_rate'].mean().sort_values('fully_paid_dummy', ascending = False))
print(completed_loan.groupby('home_ownership')['fully_paid_dummy','int_rate'].mean().sort_values('int_rate', ascending = False))
west = ['CA', 'OR', 'UT','WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID']

south_west = ['AZ', 'TX', 'NM', 'OK']

south_east = ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN' ]

mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND']

north_east = ['CT', 'NY', 'PA', 'NJ', 'RI','MA', 'MD', 'VT', 'NH', 'ME']



def main_region(x):

    if x in west:

        return 'West'

    elif x in south_west:

        return 'South West'

    elif x in south_east:

        return 'South East'

    elif x in mid_west:

        return 'Mid West'

    else:

        return 'North East'
completed_loan['main_region'] = completed_loan['addr_state'].apply(main_region)
print(completed_loan.groupby('main_region')['fully_paid_dummy', 'int_rate'].mean().sort_values('fully_paid_dummy',ascending = False))
print(completed_loan.groupby('purpose')['fully_paid_dummy', 'int_rate'].mean().sort_values('fully_paid_dummy',ascending = False))
def get_int(x):

    if x == '10+ years':

        return int(10)

    elif x == np.nan:

        return np.nan

    else: 

        tokens = x.split()

        for s in tokens:

            if s.isdigit():

                return int(s)

        
completed_loan['emp_int'] = completed_loan['emp_length'].astype(str).apply(get_int)



print(completed_loan.groupby('purpose')['fully_paid_dummy', 'int_rate'].mean().sort_values('fully_paid_dummy',ascending = False))
# cleaning up data 



completed_loan['term int'] = completed_loan['term'].apply(get_int)





# remove columns with more than 10% missing values

cols_with_missing = [col for col in completed_loan.columns 

                                 if completed_loan[col].isnull().sum() > 0.1*len(completed_loan)]



reduced_df = completed_loan.drop(cols_with_missing, axis = 1)



# drop irrelevant columns

reduced_df.drop(['term','emp_length','emp_title','zip_code', 'addr_state','earliest_cr_line',

        'disbursement_method','year','date_time','total_rec_late_fee', 'policy_code','num_tl_120dpd_2m',

                'last_pymnt_d','last_credit_pull_d','loan_status','issue_d','title'],axis=1,inplace=True)



# encoded categorical variables

dummies = pd.get_dummies(reduced_df.select_dtypes(include='object'),drop_first=True)





# training data

train = pd.concat([reduced_df.select_dtypes(exclude='object'), dummies], axis = 1).dropna(axis=0)
x_train = train.drop('fully_paid_dummy', axis = 1)



y = train['fully_paid_dummy']
# Setup

n_folds = 5

skf = StratifiedKFold(n_splits=n_folds, random_state=1, shuffle=True)

y_oof = y*0 

feat_impt = 0

model = RandomForestClassifier(n_estimators=50,random_state=0)
print("\n Begin Setting up cv. Executing {} folds cross validation \n".format(n_folds))

for i, (train_index,test_index) in enumerate(skf.split(x_train,y)):

        

    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]

    X_train, X_valid = x_train.iloc[train_index], x_train.iloc[test_index]

    print("\n Starting: fold {}".format(i+1))

        

    # fit model

    model.fit(X_train,y_train)

    

    # predict

    oof_pred = model.predict(X_valid)

    print('Accuracy Score:',accuracy_score(y_valid,oof_pred), '\n')

    print('Classification Report:','\n', classification_report(y_valid,oof_pred))

    

    # save

    y_oof.iloc[test_index] = oof_pred

    feat_impt += feat_impt + model.feature_importances_
feat_impt = pd.DataFrame(feat_impt,index=x_train.columns)

feat_impt.columns = ['Feature Importance']

plt.figure(figsize=(15,8))

sns.barplot(x=feat_impt.sort_values(by='Feature Importance',ascending=False).head(10).index,

            y=feat_impt.sort_values(by='Feature Importance',ascending=False)['Feature Importance'].head(10),

            palette='viridis')

plt.tight_layout()