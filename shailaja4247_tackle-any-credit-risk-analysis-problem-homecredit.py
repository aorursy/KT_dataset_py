import pandas as pd # package for high-performance, easy-to-use data structures and data analysis

import numpy as np # fundamental package for scientific computing with Python

import matplotlib

import matplotlib.pyplot as plt # for plotting

import seaborn as sns # for making plots with seaborn

color = sns.color_palette()

import plotly.offline as py

py.init_notebook_mode(connected=True)

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.offline as offline

offline.init_notebook_mode()

import cufflinks as cf

cf.go_offline()

application_train = pd.read_csv('../input/home-credit-default-risk/application_train.csv')

pos_cash= pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv')

bureau_balance = pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv')

previous_application = pd.read_csv('../input/home-credit-default-risk/previous_application.csv')

insta_payments = pd.read_csv('../input/home-credit-default-risk/installments_payments.csv')

credit_card_balance = pd.read_csv('../input/home-credit-default-risk/credit_card_balance.csv')

bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv')

application_test = pd.read_csv('../input/home-credit-default-risk/application_test.csv')



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)



application_train.head()

#List of non-numerical variables

application_train.select_dtypes(include=['O']).columns
#We cannot have non-numerical columns for modelling. We can have only numerical columns. Non-numerical columns can also be ordinal or categorical variables.  

col_for_dummies=application_train.select_dtypes(include=['O']).columns.drop(['FLAG_OWN_CAR','FLAG_OWN_REALTY','EMERGENCYSTATE_MODE'])

application_train_dummies = pd.get_dummies(application_train, columns = col_for_dummies, drop_first = True)

application_test_dummies = pd.get_dummies(application_test, columns = col_for_dummies, drop_first = True)
application_train_dummies.select_dtypes(include=['O']).columns
application_train_dummies['EMERGENCYSTATE_MODE'].value_counts()
#We cannot convert flag_own_car and flag_own_realty to column with yes or no etc. Lets rather map yes to 1 and no to 0

application_train_dummies['FLAG_OWN_CAR'] = application_train_dummies['FLAG_OWN_CAR'].map( {'Y':1, 'N':0})

application_train_dummies['FLAG_OWN_REALTY'] = application_train_dummies['FLAG_OWN_REALTY'].map( {'Y':1, 'N':0})

application_train_dummies['EMERGENCYSTATE_MODE'] = application_train_dummies['EMERGENCYSTATE_MODE'].map( {'Yes':1, 'No':0})



application_test_dummies['FLAG_OWN_CAR'] = application_train_dummies['FLAG_OWN_CAR'].map( {'Y':1, 'N':0})

application_test_dummies['FLAG_OWN_REALTY'] = application_train_dummies['FLAG_OWN_REALTY'].map( {'Y':1, 'N':0})

application_test_dummies['EMERGENCYSTATE_MODE'] = application_train_dummies['EMERGENCYSTATE_MODE'].map( {'Yes':1, 'No':0})

print(application_train_dummies.shape)

print(application_test_dummies.shape)
#We have 4 columns less in application_test_dummies. Lets see which are those 4 columns

#Sometimes test data does not have certain columns.

application_train_dummies.columns.difference(application_test_dummies.columns)
train_labels = application_train_dummies['TARGET']



# Align the training and testing data, keep only columns present in both dataframes

application_train_dummies, application_test_dummies = application_train_dummies.align(application_test_dummies, join = 'inner', axis = 1)



# Add the target back in

application_train_dummies['TARGET'] = train_labels



print('Training Features shape: ', application_train_dummies.shape)

print('Testing Features shape: ', application_test_dummies.shape)
from sklearn.experimental import enable_iterative_imputer

# now you can import normally from sklearn.impute

from sklearn.impute import IterativeImputer

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.linear_model import BayesianRidge

import random
y=application_train_dummies[['SK_ID_CURR','TARGET']]

X=application_train_dummies.drop(columns=['TARGET'], axis=1)

X.head()

X_imputation = X.loc[:, (X.nunique() > 1000)]
X_imputation.columns
imputer = IterativeImputer(BayesianRidge())

imputed_total = pd.DataFrame(imputer.fit_transform(X_imputation))

imputed_total.columns = X_imputation.columns
from sklearn.ensemble import IsolationForest

rs=np.random.RandomState(0)

clf = IsolationForest(max_samples=100,random_state=rs, contamination=.1) 

clf.fit(imputed_total)

if_scores = clf.decision_function(imputed_total)



pred = clf.predict(imputed_total)

imputed_total['anomaly']=pred

outliers=imputed_total.loc[imputed_total['anomaly']==-1]

outlier_index=list(outliers.index)

#print(outlier_index)

#Find the number of anomalies and normal points here points classified -1 are anomalous

print(imputed_total['anomaly'].value_counts())

outlier_ID=list(outliers['SK_ID_CURR'])

X_new = X[~X.SK_ID_CURR.isin(outlier_ID)]

y_new = y[~y.SK_ID_CURR.isin(outlier_ID)]
print(X_new.shape)

print(X.shape)
X_new.describe()
#Checking the anamalous variables values in years

print('DAYS_BIRTH stats in years:','\n',(X_new['DAYS_BIRTH'] / -365).describe(),'\n')

print('Check the stats in years to see if there is any anomalous behavior')

print('DAYS_EMPLOYED stats in years:','\n',(X_new['DAYS_EMPLOYED'] / -365).describe(),'\n')

print('DAYS_REGISTRATION stats in years:','\n',(X_new['DAYS_REGISTRATION'] / -365).describe(),'\n')

print('DAYS_ID_PUBLISH stats in years:','\n',(X_new['DAYS_ID_PUBLISH'] / -365).describe(),'\n')

print('DAYS_LAST_PHONE_CHANGE stats in years:','\n',(X_new['DAYS_LAST_PHONE_CHANGE'] / -365).describe(),'\n')
X_new['DAYS_EMPLOYED'].max()
# Replace the error values in Days_employed with nan

X_new['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

application_test_dummies['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
X_new.describe()
# checking missing data

total = X_new.isnull().sum().sort_values(ascending = False)

percent = (X_new.isnull().sum()/X_new.isnull().count()*100).sort_values(ascending = False)

missing_application_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_application_train_data.head(20)
columns_without_id = [col for col in X_new.columns if col!='SK_ID_CURR']

#Checking for duplicates in the data.

X_new[X_new.duplicated(subset = columns_without_id, keep=False)]

print('The no of duplicates in the data:',X_new[X_new.duplicated(subset = columns_without_id, keep=False)]

      .shape[0])
y_new['TARGET'].value_counts()
X_new.head()
import seaborn as sns # for making plots with seaborn

color = sns.color_palette()



plt.figure(figsize=(12,5))

plt.title("Distribution of AMT_INCOME_TOTAL")

ax = sns.distplot(X_new["AMT_INCOME_TOTAL"])
X_new["AMT_INCOME_TOTAL"].describe()
application_train=pd.merge(X_new,y_new,on='SK_ID_CURR')
(application_train[application_train['AMT_INCOME_TOTAL'] > 1000000]['TARGET'].value_counts())/len(application_train[application_train['AMT_INCOME_TOTAL'] > 1000000])*100
#boxcox=0 means we are taking log transformation of data to show it as normal form



from scipy.stats import boxcox

from matplotlib import pyplot





np.log(application_train['AMT_INCOME_TOTAL']).iplot(kind='histogram', bins=100,

                               xTitle = 'log(INCOME_TOTAL)',yTitle ='Count corresponding to Incomes',

                               title='Distribution of log(AMT_INCOME_TOTAL)')
import seaborn as sns # for making plots with seaborn

color = sns.color_palette()



plt.figure(figsize=(12,5))

plt.title("Distribution of AMT_CREDIT")

ax = sns.distplot(application_train["AMT_CREDIT"])
application_train["AMT_CREDIT"].describe()
(application_train[application_train['AMT_CREDIT']>2000000]['TARGET'].value_counts())/len(application_train[application_train['AMT_CREDIT']>2000000])*100
original_train_data = pd.read_csv('../input/home-credit-default-risk/application_train.csv')





contract_val = original_train_data['NAME_CONTRACT_TYPE'].value_counts()

contract_df = pd.DataFrame({'labels': contract_val.index,

                   'values': contract_val.values

                  })

contract_df.iplot(kind='pie',labels='labels',values='values', title='Types of Loan')



original_train_data["NAME_INCOME_TYPE"].iplot(kind="histogram", bins=20, theme="white", title="Passenger's Income Types",

                                            xTitle='Name of Income Types', yTitle='Count')

education_val = original_train_data['NAME_INCOME_TYPE'].value_counts()



education_val_y0 = []

education_val_y1 = []

for val in education_val.index:

    education_val_y1.append(np.sum(original_train_data['TARGET'][original_train_data['NAME_INCOME_TYPE']==val] == 1))

    education_val_y0.append(np.sum(original_train_data['TARGET'][original_train_data['NAME_INCOME_TYPE']==val] == 0))



data = [go.Bar(x = education_val.index, y = ((education_val_y1 / education_val.sum()) * 100), name='Default' ),

        go.Bar(x = education_val.index, y = ((education_val_y0 / education_val.sum()) * 100), name='No default' )]



layout = go.Layout(

    title = "Income of people affecting default on loans",

    xaxis=dict(

        title='Income of people',

       ),

    yaxis=dict(

        title='Count of people accompanying in %',

        )

)



fig = go.Figure(data = data, layout=layout) 

fig.layout.template = 'plotly_dark'

py.iplot(fig)
original_train_data["NAME_TYPE_SUITE"].iplot(kind="histogram", bins=20, theme="white", title="Accompanying Person",

                                            xTitle='People accompanying', yTitle='Count')

education_val = original_train_data['NAME_EDUCATION_TYPE'].value_counts()



education_val_y0 = []

education_val_y1 = []

for val in education_val.index:

    education_val_y1.append(np.sum(original_train_data['TARGET'][original_train_data['NAME_EDUCATION_TYPE']==val] == 1))

    education_val_y0.append(np.sum(original_train_data['TARGET'][original_train_data['NAME_EDUCATION_TYPE']==val] == 0))



data = [go.Bar(x = education_val.index, y = ((education_val_y1 / education_val.sum()) * 100), name='Default' ),

        go.Bar(x = education_val.index, y = ((education_val_y0 / education_val.sum()) * 100), name='No default' )]



layout = go.Layout(

    title = "Education sources of Applicants in terms of loan is repayed or not  in %",

    xaxis=dict(

        title='Education of Applicants',

       ),

    yaxis=dict(

        title='Count of applicants in %',

        )

)



fig = go.Figure(data = data, layout=layout) 

fig.layout.template = 'plotly_dark'

py.iplot(fig)
education_val = original_train_data['NAME_FAMILY_STATUS'].value_counts()



education_val_y0 = []

education_val_y1 = []

for val in education_val.index:

    education_val_y1.append(np.sum(original_train_data['TARGET'][original_train_data['NAME_FAMILY_STATUS']==val] == 1))

    education_val_y0.append(np.sum(original_train_data['TARGET'][original_train_data['NAME_FAMILY_STATUS']==val] == 0))



data = [go.Bar(x = education_val.index, y = ((education_val_y1 / education_val.sum()) * 100), name='Default' ),

        go.Bar(x = education_val.index, y = ((education_val_y0 / education_val.sum()) * 100), name='No default' )]



layout = go.Layout(

    title = "Family status of Applicant in terms of loan is repayed or not in %",

    xaxis=dict(

        title='Family status of Applicants',

       ),

    yaxis=dict(

        title='Count of applicants in %',

        )

)



fig = go.Figure(data = data, layout=layout) 

fig.layout.template = 'plotly_dark'

py.iplot(fig)
education_val = original_train_data['NAME_HOUSING_TYPE'].value_counts()



education_val_y0 = []

education_val_y1 = []

for val in education_val.index:

    education_val_y1.append(np.sum(original_train_data['TARGET'][original_train_data['NAME_HOUSING_TYPE']==val] == 1))

    education_val_y0.append(np.sum(original_train_data['TARGET'][original_train_data['NAME_HOUSING_TYPE']==val] == 0))



data = [go.Bar(x = education_val.index, y = ((education_val_y1 / education_val.sum()) * 100), name='Default' ),

        go.Bar(x = education_val.index, y = ((education_val_y0 / education_val.sum()) * 100), name='No default' )]



layout = go.Layout(

    title = "Housing Type of Applicant in terms of loan is repayed or not in %",

    xaxis=dict(

        title='Housing Type of Applicants',

       ),

    yaxis=dict(

        title='Count of applicants in %',

        )

)



fig = go.Figure(data = data, layout=layout) 

fig.layout.template = 'plotly_dark'

py.iplot(fig)
(original_train_data["DAYS_BIRTH"]/-365).iplot(kind="histogram", bins=20, theme="white", title="Customer's Ages",

                                            xTitle='Age of customer', yTitle='Count')

parameter_val = original_train_data['OCCUPATION_TYPE'].value_counts()



parameter_val_y0 = []

parameter_val_y1 = []

for val in parameter_val.index:

    parameter_val_y1.append(np.sum(original_train_data['TARGET'][original_train_data['OCCUPATION_TYPE']==val] == 1))

    parameter_val_y0.append(np.sum(original_train_data['TARGET'][original_train_data['OCCUPATION_TYPE']==val] == 0))



data = [go.Bar(x = parameter_val.index, y = ((parameter_val_y1 / parameter_val.sum()) * 100), name='Default' ),

        go.Bar(x = parameter_val.index, y = ((parameter_val_y0 / parameter_val.sum()) * 100), name='No default' )]



layout = go.Layout(

    title = "Occupation type of people affecting default on loans",

    xaxis=dict(

        title='Occupation type of people',

       ),

    yaxis=dict(

        title='Count of people Occupation that type of housing in %',

        )

)



fig = go.Figure(data = data, layout=layout) 

fig.layout.template = 'plotly_dark'

py.iplot(fig)
#Flag to represent when credit > income

application_train_dummies['Credit_flag'] = application_train_dummies['AMT_INCOME_TOTAL'] > application_train_dummies['AMT_CREDIT']

application_train_dummies['Percent_Days_employed'] = application_train_dummies['DAYS_EMPLOYED']/application_train_dummies['DAYS_BIRTH']*100

application_train_dummies['Annuity_as_percent_income'] = application_train_dummies['AMT_ANNUITY']/ application_train_dummies['AMT_INCOME_TOTAL']*100

application_train_dummies['Credit_as_percent_income'] = application_train_dummies['AMT_CREDIT']/application_train_dummies['AMT_INCOME_TOTAL']*100



application_test_dummies['Credit_flag'] = application_test_dummies['AMT_INCOME_TOTAL'] > application_test_dummies['AMT_CREDIT']

application_test_dummies['Percent_Days_employed'] = application_test_dummies['DAYS_EMPLOYED']/application_test_dummies['DAYS_BIRTH']*100

application_test_dummies['Annuity_as_percent_income'] = application_test_dummies['AMT_ANNUITY']/ application_test_dummies['AMT_INCOME_TOTAL']*100

application_test_dummies['Credit_as_percent_income'] = application_test_dummies['AMT_CREDIT']/application_test_dummies['AMT_INCOME_TOTAL']*100

# Combining numerical features

grp = bureau.drop(['SK_ID_BUREAU'], axis = 1).groupby(by=['SK_ID_CURR']).mean().reset_index()

grp.columns = ['BUREAU_'+column if column !='SK_ID_CURR' else column for column in grp.columns]

application_bureau = application_train_dummies.merge(grp, on='SK_ID_CURR', how='left')

application_bureau.update(application_bureau[grp.columns].fillna(0))



application_bureau_test = application_test_dummies.merge(grp, on='SK_ID_CURR', how='left')

application_bureau_test.update(application_bureau_test[grp.columns].fillna(0))



# Combining categorical features

bureau_categorical = pd.get_dummies(bureau.select_dtypes('object'))

bureau_categorical['SK_ID_CURR'] = bureau['SK_ID_CURR']

grp = bureau_categorical.groupby(by = ['SK_ID_CURR']).mean().reset_index()

grp.columns = ['BUREAU_'+column if column !='SK_ID_CURR' else column for column in grp.columns]

application_bureau = application_bureau.merge(grp, on='SK_ID_CURR', how='left')

application_bureau.update(application_bureau[grp.columns].fillna(0))



application_bureau_test = application_bureau_test.merge(grp, on='SK_ID_CURR', how='left')

application_bureau_test.update(application_bureau_test[grp.columns].fillna(0))
# Number of past loans per customer

grp = bureau.groupby(by = ['SK_ID_CURR'])['SK_ID_BUREAU'].count().reset_index().rename(columns = {'SK_ID_BUREAU': 'BUREAU_LOAN_COUNT'})



application_bureau = application_bureau.merge(grp, on='SK_ID_CURR', how='left')

application_bureau['BUREAU_LOAN_COUNT'] = application_bureau['BUREAU_LOAN_COUNT'].fillna(0)



application_bureau_test = application_bureau_test.merge(grp, on='SK_ID_CURR', how='left')

application_bureau_test['BUREAU_LOAN_COUNT'] = application_bureau_test['BUREAU_LOAN_COUNT'].fillna(0)
# Number of types of past loans per customer 

grp = bureau[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by = ['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(columns={'CREDIT_TYPE': 'BUREAU_LOAN_TYPES'})



application_bureau = application_bureau.merge(grp, on='SK_ID_CURR', how='left')

application_bureau['BUREAU_LOAN_TYPES'] = application_bureau['BUREAU_LOAN_TYPES'].fillna(0)



application_bureau_test = application_bureau_test.merge(grp, on='SK_ID_CURR', how='left')

application_bureau_test['BUREAU_LOAN_TYPES'] = application_bureau_test['BUREAU_LOAN_TYPES'].fillna(0)
# Debt over credit ratio 

bureau['AMT_CREDIT_SUM'] = bureau['AMT_CREDIT_SUM'].fillna(0)

bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)



grp1 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM': 'TOTAL_CREDIT_SUM'})



grp2 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_DEBT':'TOTAL_CREDIT_SUM_DEBT'})



grp1['DEBT_CREDIT_RATIO'] = grp2['TOTAL_CREDIT_SUM_DEBT']/grp1['TOTAL_CREDIT_SUM']



del grp1['TOTAL_CREDIT_SUM']



application_bureau = application_bureau.merge(grp1, on='SK_ID_CURR', how='left')

application_bureau['DEBT_CREDIT_RATIO'] = application_bureau['DEBT_CREDIT_RATIO'].fillna(0)

application_bureau['DEBT_CREDIT_RATIO'] = application_bureau.replace([np.inf, -np.inf], 0)

application_bureau['DEBT_CREDIT_RATIO'] = pd.to_numeric(application_bureau['DEBT_CREDIT_RATIO'], downcast='float')



application_bureau_test = application_bureau_test.merge(grp1, on='SK_ID_CURR', how='left')

application_bureau_test['DEBT_CREDIT_RATIO'] = application_bureau_test['DEBT_CREDIT_RATIO'].fillna(0)

application_bureau_test['DEBT_CREDIT_RATIO'] = application_bureau_test.replace([np.inf, -np.inf], 0)

application_bureau_test['DEBT_CREDIT_RATIO'] = pd.to_numeric(application_bureau_test['DEBT_CREDIT_RATIO'], downcast='float')
(application_bureau[application_bureau['DEBT_CREDIT_RATIO'] > 0.5]['TARGET'].value_counts()/len(application_bureau[application_bureau['DEBT_CREDIT_RATIO'] > 0.5]))*100
# Overdue over debt ratio

bureau['AMT_CREDIT_SUM_OVERDUE'] = bureau['AMT_CREDIT_SUM_OVERDUE'].fillna(0)

bureau['AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM_DEBT'].fillna(0)



grp1 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM_OVERDUE']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_OVERDUE': 'TOTAL_CUSTOMER_OVERDUE'})



grp2 = bureau[['SK_ID_CURR','AMT_CREDIT_SUM_DEBT']].groupby(by=['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(columns={'AMT_CREDIT_SUM_DEBT':'TOTAL_CUSTOMER_DEBT'})



grp1['OVERDUE_DEBT_RATIO'] = grp1['TOTAL_CUSTOMER_OVERDUE']/grp2['TOTAL_CUSTOMER_DEBT']



del grp1['TOTAL_CUSTOMER_OVERDUE']



application_bureau = application_bureau.merge(grp1, on='SK_ID_CURR', how='left')

application_bureau['OVERDUE_DEBT_RATIO'] = application_bureau['OVERDUE_DEBT_RATIO'].fillna(0)

application_bureau['OVERDUE_DEBT_RATIO'] = application_bureau.replace([np.inf, -np.inf], 0)

application_bureau['OVERDUE_DEBT_RATIO'] = pd.to_numeric(application_bureau['OVERDUE_DEBT_RATIO'], downcast='float')



application_bureau_test = application_bureau_test.merge(grp1, on='SK_ID_CURR', how='left')

application_bureau_test['OVERDUE_DEBT_RATIO'] = application_bureau_test['OVERDUE_DEBT_RATIO'].fillna(0)

application_bureau_test['OVERDUE_DEBT_RATIO'] = application_bureau_test.replace([np.inf, -np.inf], 0)

application_bureau_test['OVERDUE_DEBT_RATIO'] = pd.to_numeric(application_bureau_test['OVERDUE_DEBT_RATIO'], downcast='float')
import gc



gc.collect()
def isOneToOne(df, col1, col2):

    first = df.drop_duplicates([col1, col2]).groupby(col1)[col2].count().max()

    second = df.drop_duplicates([col1, col2]).groupby(col2)[col1].count().max()

    return first + second == 2



isOneToOne(previous_application,'SK_ID_CURR','SK_ID_PREV')
# Number of previous applications per customer

grp = previous_application[['SK_ID_CURR','SK_ID_PREV']].groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].count().reset_index().rename(columns={'SK_ID_PREV':'PREV_APP_COUNT'})



# Take only the IDs which are present in application_bureau

application_bureau_prev = application_bureau.merge(grp, on =['SK_ID_CURR'], how = 'left')

application_bureau_prev_test = application_bureau_test.merge(grp, on =['SK_ID_CURR'], how = 'left')



#Fill NA for previous application counts (lets say there was an application ID present in application_bureau but not present

# in grp, then that means that person never took loan previously, so count of previous loan for that person = 0)

application_bureau_prev['PREV_APP_COUNT'] = application_bureau_prev['PREV_APP_COUNT'].fillna(0)

application_bureau_prev_test['PREV_APP_COUNT'] = application_bureau_prev_test['PREV_APP_COUNT'].fillna(0)
# Combining numerical features



#Take the mean of all the parameters (grouping by SK_ID_CURR)

grp = previous_application.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()



#Add prefix prev in front of all columns so that we know that these columns are from previous_application

prev_columns = ['PREV_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]



#Change the columns

grp.columns = prev_columns



application_bureau_prev = application_bureau_prev.merge(grp, on =['SK_ID_CURR'], how = 'left')

application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

application_bureau_prev_test = application_bureau_prev_test.merge(grp, on =['SK_ID_CURR'], how = 'left')

application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))



# Combining categorical features

prev_categorical = pd.get_dummies(previous_application.select_dtypes('object'))

prev_categorical['SK_ID_CURR'] = previous_application['SK_ID_CURR']



grp = prev_categorical.groupby('SK_ID_CURR').mean().reset_index()

grp.columns = ['PREV_'+column if column != 'SK_ID_CURR' else column for column in grp.columns]



application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')

application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))



application_bureau_prev_test = application_bureau_prev_test.merge(grp, on=['SK_ID_CURR'], how='left')

application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))

gc.collect()
# Combining numerical features

grp = pos_cash.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()

prev_columns = ['POS_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]

grp.columns = prev_columns
application_bureau_prev = application_bureau_prev.merge(grp, on =['SK_ID_CURR'], how = 'left')

application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))



application_bureau_prev_test = application_bureau_prev_test.merge(grp, on =['SK_ID_CURR'], how = 'left')

application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))
# Combining categorical features

pos_cash_categorical = pd.get_dummies(pos_cash.select_dtypes('object'))

pos_cash_categorical['SK_ID_CURR'] = pos_cash['SK_ID_CURR']



grp = pos_cash_categorical.groupby('SK_ID_CURR').mean().reset_index()

grp.columns = ['POS_'+column if column != 'SK_ID_CURR' else column for column in grp.columns]
application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')

application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))



application_bureau_prev_test = application_bureau_prev_test.merge(grp, on=['SK_ID_CURR'], how='left')

application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))
gc.collect()
# Combining numerical features and there are no categorical features in this dataset

grp = insta_payments.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()

prev_columns = ['INSTA_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]

grp.columns = prev_columns

application_bureau_prev = application_bureau_prev.merge(grp, on =['SK_ID_CURR'], how = 'left')

application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))

application_bureau_prev_test = application_bureau_prev_test.merge(grp, on =['SK_ID_CURR'], how = 'left')

application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))
gc.collect()
credit_card=credit_card_balance

# Combining numerical features

grp = credit_card.drop('SK_ID_PREV', axis =1).groupby(by=['SK_ID_CURR']).mean().reset_index()

prev_columns = ['CREDIT_'+column if column != 'SK_ID_CURR' else column for column in grp.columns ]

grp.columns = prev_columns

application_bureau_prev = application_bureau_prev.merge(grp, on =['SK_ID_CURR'], how = 'left')

application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))



application_bureau_prev_test = application_bureau_prev_test.merge(grp, on =['SK_ID_CURR'], how = 'left')

application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))
# Combining categorical features

credit_categorical = pd.get_dummies(credit_card.select_dtypes('object'))

credit_categorical['SK_ID_CURR'] = credit_card['SK_ID_CURR']



grp = credit_categorical.groupby('SK_ID_CURR').mean().reset_index()

grp.columns = ['CREDIT_'+column if column != 'SK_ID_CURR' else column for column in grp.columns]



application_bureau_prev = application_bureau_prev.merge(grp, on=['SK_ID_CURR'], how='left')

application_bureau_prev.update(application_bureau_prev[grp.columns].fillna(0))



application_bureau_prev_test = application_bureau_prev_test.merge(grp, on=['SK_ID_CURR'], how='left')

application_bureau_prev_test.update(application_bureau_prev_test[grp.columns].fillna(0))