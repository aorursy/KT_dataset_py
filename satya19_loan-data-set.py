import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# data visualization

import seaborn as sns

%matplotlib inline

import matplotlib.pyplot as plt

from matplotlib import style

plt.rcParams['figure.figsize']=(20,10) # set the figure size

plt.style.use('fivethirtyeight') # using the fivethirtyeight matplotlib theme



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



loans_data = pd.read_csv("../input/lending_club_loans.csv",skiprows=1)



loans_data.head()
def draw_missing_data_table(loans_data):

    total = loans_data.isnull().sum().sort_values(ascending=False)

    percent = (loans_data.isnull().sum()/loans_data.isnull().count()).sort_values(ascending=False)

    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent(%)'])

    return missing_data
draw_missing_data_table(loans_data).head(10)
half_count = len(loans_data) / 2

loans_data = loans_data.dropna(thresh=half_count,axis=1) # Drop any column with more than 50% missing values

loans_data = loans_data.drop(['desc'],axis=1)      # These columns are not useful for our purposes
loans_data.shape
# Calculate Loan Status Ratio

good_loan =  len(loans_data[(loans_data.loan_status == 'Fully Paid') |

                    (loans_data.loan_status == 'Current') | 

                    (loans_data.loan_status == 'Does not meet the credit policy. Status:Fully Paid')])

print ('Good/Bad Loan Ratio: %.2f%%'  % (good_loan/len(loans_data)*100))
data_dictionary = pd.read_csv('../input/LCDataDictionary.csv') # Loading in the data dictionary

print(data_dictionary.shape[0])

print(data_dictionary.columns.tolist())
data_dictionary = data_dictionary.rename(columns={'LoanStatNew': 'name', 'Description': 'description'})

data_dictionary.head()
loans_data_dtypes = pd.DataFrame(loans_data.dtypes,columns=['dtypes'])

loans_data_dtypes = loans_data_dtypes.reset_index()

loans_data_dtypes['name'] = loans_data_dtypes['index']

loans_data_dtypes = loans_data_dtypes[['name','dtypes']]



loans_data_dtypes['first value'] = loans_data.loc[0].values

preview = loans_data_dtypes.merge(data_dictionary, on='name',how='left')

preview.head()
preview[:19]
drop_list = ['id','member_id','funded_amnt','funded_amnt_inv','int_rate','sub_grade','emp_title','issue_d','url']

loans_data = loans_data.drop(drop_list,axis=1)
preview[19:38]
drop_cols = [ 'zip_code','out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv']

loans_data = loans_data.drop(drop_cols, axis=1)
preview[38:]
drop_cols = ['total_rec_prncp','total_rec_int','total_rec_late_fee','recoveries',

             'collection_recovery_fee','last_pymnt_d','last_pymnt_amnt']



loans_data = loans_data.drop(drop_cols, axis=1)
print(loans_data['fico_range_low'].unique())

print(loans_data['fico_range_high'].unique())
fico_columns = ['fico_range_high','fico_range_low']



print(loans_data.shape[0])

loans_data.dropna(subset=fico_columns,inplace=True)

print(loans_data.shape[0])



loans_data[fico_columns].plot.hist(alpha=0.5,bins=20);
loans_data['fico_average'] = (loans_data['fico_range_high'] + loans_data['fico_range_low']) / 2
cols = ['fico_range_low','fico_range_high','fico_average']

loans_data[cols].head()
drop_cols = ['fico_range_low','fico_range_high','last_fico_range_low',             'last_fico_range_high']

loans_data = loans_data.drop(drop_cols, axis=1)

loans_data.shape
preview[preview.name == 'loan_status']
loans_data["loan_status"].value_counts()
# Plot Loan Status

plt.figure(figsize= (12,6))

plt.ylabel('Loan Status')

plt.xlabel('Count')

loans_data['loan_status'].value_counts().plot(kind = 'barh', grid = True)

plt.show()
meaning = [

    "Loan has been fully paid off.",

    "Loan for which there is no longer a reasonable expectation of further payments.",

    "While the loan was paid off, the loan application today would no longer meet the credit policy and wouldn't be approved on to the marketplace.",

    "While the loan was charged off, the loan application today would no longer meet the credit policy and wouldn't be approved on to the marketplace.",

    "Loan is up to date on current payments.",

    "The loan is past due but still in the grace period of 15 days.",

    "Loan hasn't been paid in 31 to 120 days (late on the current payment).",

    "Loan hasn't been paid in 16 to 30 days (late on the current payment).",

    "Loan is defaulted on and no payment has been made for more than 121 days."]



status, count = loans_data["loan_status"].value_counts().index, loans_data["loan_status"].value_counts().values



loan_statuses_explanation = pd.DataFrame({'Loan Status': status,'Count': count,'Meaning': meaning})[['Loan Status','Count','Meaning']]

loan_statuses_explanation
loans_data = loans_data[(loans_data["loan_status"] == "Fully Paid") |

                            (loans_data["loan_status"] == "Charged Off")]



mapping_dictionary = {"loan_status":{ "Fully Paid": 1, "Charged Off": 0}}

loans_data = loans_data.replace(mapping_dictionary)
fig, axs = plt.subplots(1,2,figsize=(14,7))

sns.countplot(x='loan_status',data=loans_data,ax=axs[0])

axs[0].set_title("Frequency of each Loan Status")

loans_data.loan_status.value_counts().plot(x=None,y=None, kind='pie', ax=axs[1],autopct='%1.2f%%')

axs[1].set_title("Percentage of each Loan status")

plt.show()
loans_data = loans_data.loc[:,loans_data.apply(pd.Series.nunique) != 1]
for col in loans_data.columns:

    if (len(loans_data[col].unique()) < 4):

        print(loans_data[col].value_counts())

        print()
print(loans_data.shape[1])

loans_data = loans_data.drop('pymnt_plan', axis=1)

print("We've been able to reduced the features to => {}".format(loans_data.shape[1]))
print(loans_data.shape)

loans_data.head()
null_counts = loans_data.isnull().sum()

print("Number of null values in each column:\n{}".format(null_counts))
filtered_loans = loans_data.drop("pub_rec_bankruptcies",axis=1)

filtered_loans = loans_data.dropna()
print("Data types and their frequency\n{}".format(filtered_loans.dtypes.value_counts()))
object_columns_df = filtered_loans.select_dtypes(include=['object'])

print(object_columns_df.iloc[0])
filtered_loans['revol_util'] = filtered_loans['revol_util'].str.rstrip('%').astype('float')
cols = ['home_ownership', 'grade','verification_status', 'emp_length', 'term', 'addr_state']

for name in cols:

    print(name,':')

    print(object_columns_df[name].value_counts(),'\n')
for name in ['purpose','title']:

    print("Unique Values in column: {}\n".format(name))

    print(filtered_loans[name].value_counts(),'\n')
drop_cols = ['last_credit_pull_d','addr_state','title','earliest_cr_line']

filtered_loans = filtered_loans.drop(drop_cols,axis=1)
mapping_dict = {

    "emp_length": {

        "10+ years": 10,

        "9 years": 9,

        "8 years": 8,

        "7 years": 7,

        "6 years": 6,

        "5 years": 5,

        "4 years": 4,

        "3 years": 3,

        "2 years": 2,

        "1 year": 1,

        "< 1 year": 0,

        "n/a": 0



    },

    "grade":{

        "A": 1,

        "B": 2,

        "C": 3,

        "D": 4,

        "E": 5,

        "F": 6,

        "G": 7

    }

}



filtered_loans = filtered_loans.replace(mapping_dict)

filtered_loans[['emp_length','grade']].head()
nominal_columns = ["home_ownership", "verification_status", "purpose", "term"]

dummy_df = pd.get_dummies(filtered_loans[nominal_columns])

filtered_loans = pd.concat([filtered_loans, dummy_df], axis=1)

filtered_loans = filtered_loans.drop(nominal_columns, axis=1)

filtered_loans.head()
filtered_loans.info()
filtered_loans.head()
from sklearn.model_selection import KFold

from matplotlib import pyplot

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
Y = filtered_loans['loan_status'].copy()

X = filtered_loans.drop(columns='loan_status')
models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

for name, model in models:

 kfold = KFold(n_splits=10, random_state=7)

 cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

 results.append(cv_results)

 names.append(name)

 msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

 print(msg)

# boxplot algorithm comparison

fig = pyplot.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

pyplot.boxplot(results)

ax.set_xticklabels(names)

pyplot.show()