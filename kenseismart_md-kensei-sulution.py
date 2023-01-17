import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import random

from scipy.stats import norm

from scipy import stats



from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier 

from sklearn.svm import SVC 

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB  



from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import (roc_auc_score, roc_curve, auc, confusion_matrix,

                            accuracy_score, classification_report, plot_confusion_matrix,

                            plot_precision_recall_curve, precision_recall_curve,plot_roc_curve)



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV



pd.set_option("display.max_columns",200)
def check_division(df, collumn_name):

    sns.distplot(df[collumn_name], fit=norm)



    mu, sigma = norm.fit(df[collumn_name])



    print(f'mu = {mu:.2f} and sigma = {sigma:.2f}')



    plt.legend(

        [f'Нормальное распределение. ($\mu=$ {mu:.2f} и $\sigma=$ {sigma:.2f} )'])

    plt.ylabel('Частота')

    plt.title(f'Распределение {collumn_name}')



    # QQ-plot

    fig = plt.figure()

    res = stats.probplot(df[collumn_name], plot=plt)



    

def fix_home_ownership(df):

    df.loc[df['Home Ownership'] == 'Have Mortgage', 'Home Ownership'] = 'Home Mortgage'

    return df





def fix_annual_income(df):

    mean_income = df['Annual Income'].mean()

    df['Annual Income'] = df['Annual Income'].fillna(mean_income)

    

    return df





def fix_years_in_current_job(df):

    year_mode = df['Years in current job'].mode().astype(str)[0]

    df['Years in current job'] = df['Years in current job'].fillna(year_mode)



    

    return df





def year_in_current_job_str_to_num(df):

    df.loc[(df['Years in current job'] == '< 1 year') | (df['Years in current job'] == '1 year') |  (df['Years in current job'] == '2 years') | (df['Years in current job'] == '3 years') | (df['Years in current job'] == '4 years'),  'Years in current job'] = '4 or less years'

    df.loc[(df['Years in current job'] == '5 years') | (df['Years in current job'] == '6 years') | (df['Years in current job'] == '7 years'),  'Years in current job'] = '5-7 years'

    df.loc[(df['Years in current job'] == '8 years') | (df['Years in current job'] == '9 years') | (df['Years in current job'] == '10+ years'),  'Years in current job'] = '8 or more years'

    

    return df

    





def fix_last_delinquent(df):

    my_mean = df['Months since last delinquent'].mode()

    df['Months since last delinquent'] =  df['Months since last delinquent'].fillna(my_mean[0])

    return df





def fix_bankruptcies(df):

    df['Bankruptcies'] = df['Bankruptcies'].fillna(0)

    return df





def fix_purpose(df):

    df.loc[df['Purpose'] == 'take a trip', 'Purpose'] = 'vacation'

    df.loc[df['Purpose'] == 'renewable energy', 'Purpose'] = 'business loan'

    df.loc[df['Purpose'] == 'small business', 'Purpose'] = 'business loan'

    df.loc[df['Purpose'] == 'educational expenses', 'Purpose'] = 'major purchase'

    df.loc[df['Purpose'] == 'medical bills', 'Purpose'] = 'major purchase'

    df.loc[df['Purpose'] == 'wedding', 'Purpose'] = 'major purchase'

    df.loc[df['Purpose'] == 'moving', 'Purpose'] = 'other'

    

    return df





def fix_cur_loan_am(df):

    df.loc[df['Current Loan Amount'] > (df['Annual Income'] * 100), 'Current Loan Amount'] = df['Annual Income'] * 50 * 0.7

    df.loc[df['Current Loan Amount'] > (df['Annual Income'] * 50), 'Current Loan Amount'] = df['Annual Income'] * 50 * 0.7

    mean_loan = df['Current Loan Amount'].mean()

    df.loc[df['Current Loan Amount'].isnull() == True,'Current Loan Amount'] = mean_loan

    

    return df



def fix_cur_credit_balance(df):

    df.loc[df['Current Credit Balance'] > (df['Annual Income'] * 30), 'Current Credit Balance'] = df['Annual Income'] * 30

    

    return df



def fix_last_delinquent_months(df):

    mean_df = df['Months since last delinquent'].mean()

    df['Months since last delinquent'] = df['Months since last delinquent'].fillna(mean_df)

    

    return df



def fix_credit_score(df):

    df.loc[df['Credit Score'] > 1000, 'Credit Score'] = df['Credit Score'] / 10

    df_mean = df['Credit Score'].mean()

    df['Credit Score'] = df['Credit Score'].fillna(df_mean)

    

    return df



def drop_col(df):

    df = df.drop('Number of Open Accounts', axis=1)

    

    return df



def new_feature(df):

    df['month_income'] = df['Annual Income'] / 12

    df['left_after_pay'] =  df['month_income'] - df['Monthly Debt']

    

    df.loc[(df['Current Credit Balance'] - df['Current Loan Amount'] ) < 0, 'Current Credit Balance'] = df['Current Loan Amount']

    df['how_much_used_credit_balance'] =   df['Current Credit Balance'] / df['Current Credit Balance'] * 100

    

    return df



def fix_all(df):

    df = fix_home_ownership(df)

    df = fix_annual_income(df)

    df = fix_years_in_current_job(df)

    df = year_in_current_job_str_to_num(df)

    df = fix_last_delinquent(df)

    df = fix_bankruptcies(df)

    df = fix_purpose(df)

    df = fix_cur_loan_am(df)

    df = fix_cur_credit_balance(df)

    df = fix_last_delinquent(df)

    df = fix_credit_score(df)

    df = drop_col(df)

    df = new_feature(df)

    df = pd.get_dummies(df)

    return df
train_df = pd.read_csv('../input/credit-default/train.csv')

test_df = pd.read_csv('../input/credit-default/test.csv')

train_df_1 = train_df.loc[train_df['Credit Default'] == 1]

train_df_2 = train_df.append(train_df_1)

train_df_2 = train_df_2.reset_index(drop=True)

train_df_2 = fix_all(train_df_2)

test_df = fix_all(test_df)
features = list(train_df_2.loc[:, (train_df_2.columns != 'Id')].corrwith(

    train_df_2['Credit Default']).abs().sort_values(ascending=False)[1:].index)



target = 'Credit Default'



X = train_df_2[features]

y = train_df_2[target]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=2)

X_train.shape, X_test.shape
MOD = RandomForestClassifier(max_depth=25, max_features=20, min_samples_split=5,

                       n_estimators=440, oob_score=True)



# m_params = [

#             {

#                     "n_estimators" : range(1,500),  

#                     "max_depth": range(1,30), 

#                     "min_samples_split": range(1,50),  

#                     "max_features": ["sqrt", "log2",10, 20, None],

#                     "oob_score": [True],

#                     "bootstrap": [True]

#                     }

# ]



# scoreFunction = {"recall": "recall", "precision": "precision"}



# random_search = RandomizedSearchCV(MOD,

#                                        m_params,

#                                        n_iter = 20,

#                                        scoring = scoreFunction,               

#                                        refit = "recall",

#                                        return_train_score = True,

#                                        random_state = 42,

#                                        cv = 5) 





MOD.fit(X_train, y_train)





# MOD = random_search.best_estimator_
MOD_pred_test = MOD.predict(X_test)
# random_search.best_estimator_

# RandomForestClassifier(max_depth=25, max_features=20, min_samples_split=5,

#                        n_estimators=440, oob_score=True)
plot_confusion_matrix(MOD, X_test, y_test, cmap=plt.cm.Blues);
print(classification_report(y_test, MOD_pred_test))
proba = MOD.predict_proba(X_test)

proba[:5]
pd.Series(proba[:,1]).hist()
MOD_pred_test = np.where(proba[:,1] >= 0.6, 1, 0)
print(classification_report(y_test, MOD_pred_test))
test_df = pd.read_csv('../input/credit-default/test.csv')

test_df = fix_all(test_df)
submit = pd.read_csv('../input/credit-default/sample_submission.csv')
proba = MOD.predict_proba(test_df)
proba.max()
proba.min()
m = stats.mode(proba)

m1 = m[0][0]

m1[1]
pd.Series(proba[:,1]).hist()
MOD_pred_test = np.where(proba[:,1] >= (m1[1]), 0, 1)
submit['Credit Default'] = MOD_pred_test

submit['Credit Default'].value_counts()
submit.to_csv('./predictions.csv', index=False, encoding='utf-8', sep=',')