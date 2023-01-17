import collections



import numpy as np

import pandas as pd



# Visualisation

from pprint import pprint

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# display options

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)



# sklearn

# from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import auc, roc_auc_score, roc_curve

from sklearn import tree



# Save model

import pickle

def diff(li1, li2):

    '''

    This function returns different elements between 2 lists

    '''

    return (list(set(li1) - set(li2)))



def plot_stats(df, feature, target_ftr, label_rotation=False, horizontal_layout=True):

    '''

    This function plot the categorical feature distribution according to target variable

    '''

    temp = df[feature].value_counts()

    df1 = pd.DataFrame({feature: temp.index,'Number of repaid-loans': temp.values})



    # Calculate the percentage of target=1 per category value

    cat_perc = df[[feature, target_ftr]].groupby([feature],as_index=False).mean()

    cat_perc.sort_values(by=target_ftr, ascending=False, inplace=True)

    

    sns.set_color_codes("pastel")

    s = sns.barplot(x = feature, y="Number of repaid-loans",data=df1)

    if(label_rotation):

        s.set_xticklabels(s.get_xticklabels(),rotation=60)



    plt.tick_params(axis='both', which='major', labelsize=10)



    plt.show();





def get_rocauc(model, xTest, yTest): 

    '''

    This function produces the Area under the curve for the model. 

    The 'auto' method calculates this metric by using the roc_auc_score function from sklearn.

    Range: 0 to 1 (0 being the worst predictive model, 0.5 being the random and 1 being the best)

    '''

    predictions = model.predict_proba(xTest)[:, 1]

    roc_auc = roc_auc_score(yTest, predictions)

    print('Model Performance:')

    print('--'*5)

    print('--'*5)

    print('ROC = {:0.2f}%'.format(roc_auc))

    

    return roc_auc



def plot_roc(yTest, yPred):

    '''

    This function plots the ROC and gives the AUC.

    Range for Area under the curve : 0 to 1 (0 being the worst and 1 being best for predictive model)

    '''

    fpr, tpr, thresholds = roc_curve(yTest, yPred)

    roc_auc = roc_auc_score(yTest, yPred)

    plt.figure(figsize=(10,10))

    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)

    plt.plot([0, 1], [0, 1], 'k--', label='random')

    plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')

    plt.xlim([-0.05, 1.05])

    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('ROC Curve')

    plt.legend(loc="lower right")

    plt.show()



def perf_measure(y_actual, y_pred):

    '''

    Function for calculating TP, FP, TN and FN

    '''

    TP = 0

    FP = 0

    TN = 0

    FN = 0



    for i in range(len(y_pred)): 

        if y_actual[i]==y_pred[i]==1:

            TP += 1

        if y_pred[i]==1 and y_actual[i]!=y_pred[i]:

            FP += 1

        if y_actual[i]==y_pred[i]==0:

            TN += 1

        if y_pred[i]==0 and y_actual[i]!=y_pred[i]:

            FN += 1



    return(TP, FP, TN, FN)



def capture_curve(test_df, y_test, preds, roc, title):

    '''

    Function for to plot capture curve for risky or non-repaid loans

    This is similar to gain and lift chart in statistics.

    x-axis: Population % of granted loans

    y-axis: Risk / Non-repaid loan %

    '''

    fpr, tpr, threshold = roc_curve(y_test, preds)

    roc_auc = auc(fpr, tpr)

    rate = []

    for i in threshold:

        T = perf_measure(list(y_test),[1 if j >= i else 0 for j in preds])

        rate.append(T[0]+T[1])

    rate2 = [i/len(test_df) for i in rate]

    plt.figure(figsize=[12,12])

    plt.plot(rate2, tpr, label='ROC_AUC {}'.format(roc) % roc_auc, linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=4)

    plt.xlim([-0.05, 1.0])

    plt.ylim([-0.05, 1.05])

    plt.xlabel('Granted Loans', fontsize=18)

    plt.ylabel('Captured out of total non-repaid proportion', fontsize=18)

    plt.title('Capture plot for {}'.format(title), fontsize=18)

    plt.legend(loc="lower right",prop={'size':30})

    plt.show()



def decile_cutoff_risk_detected(df_pred):

    '''

    Function to find the decile-wise risky loan application %

    '''

    pop_perc = list(np.arange(0.1,1.1,0.1))

    perc_list = []

    risk_perc_list = []

    risk_num_list = []

    avg_pred_prob_list = []

    min_pred_prob_list = []

    max_pred_prob_list = []

    df_pop_risk = pd.DataFrame(columns=['loan_percentage','non-repaid_percentage', '#non-repaid_loans', 'avg_pred_prob'])

    total_risk_count = df_pred[df_pred['loan_quality']==1]['loan_id'].count()

    start = 0

    for perc in pop_perc:

        split_len = int(perc*len(df_pred))

        sorted_results_final = df_pred.iloc[start:split_len]

        risk_count = sorted_results_final[sorted_results_final['loan_quality']==1]['loan_id'].count()

        min_pred_prob = sorted_results_final['preds'].min()

        max_pred_prob = sorted_results_final['preds'].max()

        avg_pred_prob = sorted_results_final['preds'].mean()

        perc_list.append(int(perc*100))

        risk_perc_list.append(round(((risk_count/total_risk_count)*100),2))

        risk_num_list.append(round(risk_count,2))

        min_pred_prob_list.append(round(min_pred_prob, 2))

        max_pred_prob_list.append(round(max_pred_prob, 2))

        avg_pred_prob_list.append(round(avg_pred_prob, 2))

        start = split_len

    df_pop_risk['loan_percentage'] = perc_list

    df_pop_risk['non-repaid_percentage'] = risk_perc_list  

    df_pop_risk['#non-repaid_loans'] = risk_num_list

    df_pop_risk['min_pred_prob'] = min_pred_prob_list

    df_pop_risk['max_pred_prob'] = max_pred_prob_list

    df_pop_risk['avg_pred_prob'] = avg_pred_prob_list

    return df_pop_risk



def cum_decile_cutoff_risk_detected(df_pred):

    '''

    Function to find the decile-wise risky loan application %

    '''

    pop_perc = list(np.arange(0.1,1.1,0.1))

    perc_list = []

    risk_perc_list = []

    risk_num_list = []

    avg_pred_prob_list = []

    min_pred_prob_list = []

    max_pred_prob_list = []

    df_pop_risk = pd.DataFrame(columns=['loan_percentage','non-repaid_percentage', '#non-repaid_loans', 'avg_pred_prob'])

    total_risk_count = df_pred[df_pred['loan_quality']==1]['loan_id'].count()

    for perc in pop_perc:

        split_len = int(perc*len(df_pred))

        sorted_results_final = df_pred.iloc[:split_len]

        risk_count = sorted_results_final[sorted_results_final['loan_quality']==1]['loan_id'].count()

        min_pred_prob = sorted_results_final['preds'].min()

        max_pred_prob = sorted_results_final['preds'].max()

        avg_pred_prob = sorted_results_final['preds'].mean()

        perc_list.append(int(perc*100))

        risk_perc_list.append(round(((risk_count/total_risk_count)*100),2))

        risk_num_list.append(round(risk_count,2))

        min_pred_prob_list.append(round(min_pred_prob, 2))

        max_pred_prob_list.append(round(max_pred_prob, 2))

        avg_pred_prob_list.append(round(avg_pred_prob, 2))

    df_pop_risk['loan_percentage'] = perc_list

    df_pop_risk['non-repaid_percentage'] = risk_perc_list  

    df_pop_risk['#non-repaid_loans'] = risk_num_list

    df_pop_risk['min_pred_prob'] = min_pred_prob_list

    df_pop_risk['max_pred_prob'] = max_pred_prob_list

    df_pop_risk['avg_pred_prob'] = avg_pred_prob_list

    return df_pop_risk



def profitability_diff(score1_name, score2_name, score1, score2):

    '''

    This function takes 2 scores as input and gives Absolute Difference between scores as output

    '''

    if score1 > score2:

        print('%s is %d point better than %s' % (score1_name, abs(score1 - score2), score2_name))

        print('--'*30)

        print('%s is %f times better than %s' % (score1_name, (score1/score2), score2_name))

    else:

        print('%s is %d point better than %s' % (score2_name, abs(score1 - score2), score1_name))

        print('--'*30)

        print('%s is %f times better than %s' % (score2_name, (score2/score1), score1_name))

# Directory Path

dir_path = '../input/bank-loan-data/'
borrower = pd.read_csv(dir_path + 'borrower_table.csv')

print(borrower.shape)

borrower.head(3)
loan = pd.read_csv(dir_path + 'loan_table.csv')

print(loan.shape)

loan.head(3)
print(f'There are {loan.shape[0] - loan.loan_id.nunique()} duplicates for loan id in the loan table')

print(f'There are {borrower.shape[0] - borrower.loan_id.nunique()} duplicates for loan id in the borrower table')
borrower_loan_id = list(borrower['loan_id'])

loan_loan_id = list(loan['loan_id'])



if collections.Counter(borrower_loan_id) == collections.Counter(loan_loan_id): 

    print ("All the loan ids are same") 

else : 

    print ("Different loan ids are present in both the datasets") 
print(diff(borrower_loan_id, loan_loan_id))
# Merge Dataframes



df_loan = loan.merge(borrower, how='left', on='loan_id')

print(df_loan.shape)

df_loan.head(3)
print(df_loan.info())
# Numerical features



df_loan.describe().transpose()
# Calculate missing value count and percentage



missing_value_df_loan = pd.DataFrame(index = df_loan.keys(), data =df_loan.isnull().sum(), columns = ['Missing_Value_Count'])

missing_value_df_loan['Missing_Value_Percentage'] = np.round(((df_loan.isnull().mean())*100),2)

missing_value_df_loan.sort_values('Missing_Value_Count',ascending= False)
# Make correlation table according to spearman's correlation



corr_spearman = df_loan.corr()

corr_spearman
# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr_spearman, dtype=np.bool))



# Visualisation of heatmap matrix

fig, ax = plt.subplots(figsize=(10,10))

ax = sns.heatmap(corr_spearman, mask=mask, annot = True)
# granted & ungranted



granted   = df_loan.loc[(df_loan['loan_granted'] == 1)]

ungranted = df_loan.loc[(df_loan['loan_granted'] == 0)]
# repaid & non-repaid



repaid     = granted.loc[(granted['loan_repaid'] == 1)]

non_repaid = granted.loc[(granted['loan_repaid'] == 0)]
# Class Imbalance check



temp_ln_rpd = granted['loan_repaid'].value_counts()

df_ln_rpd = pd.DataFrame({'labels': temp_ln_rpd.index,

                   'values': temp_ln_rpd.values})

plt.figure(figsize = (6,6))

plt.title('Application loans repaid')

sns.set_color_codes("pastel")

sns.barplot(x = 'labels', y="values", data=df_ln_rpd)

locs, labels = plt.xticks()

plt.show()
plot_stats(granted, 'dependent_number', 'loan_repaid', label_rotation=False, horizontal_layout=True)
plot_stats(granted, 'loan_purpose', 'loan_repaid', label_rotation=True, horizontal_layout=True)
plot_stats(granted, 'is_employed', 'loan_repaid', label_rotation=False, horizontal_layout=True)
# How many of employed repay their loans

print('Employed who repay their loans (%) :')

print(round(((granted[(granted['is_employed']==1) & (granted['loan_repaid']==1)].shape[0])/granted[(granted['is_employed']==1)].shape[0])*100,1))



print('--'*20)

print('--'*20)



# How many of un-employed repay their loans

print('Unemployed who repay their loans (%) :')

print(round(((granted[(granted['is_employed']==0) & (granted['loan_repaid']==1)].shape[0])/granted[(granted['is_employed']==0)].shape[0])*100,1))

plot_stats(granted, 'currently_repaying_other_loans', 'loan_repaid', label_rotation=False, horizontal_layout=True)
# How many repay the loan while paying for another loan

print('People who repay their loans having existing loan payment (%) :')

print(round(((granted[(granted['currently_repaying_other_loans']==1) & (granted['loan_repaid']==1)].shape[0])/granted[(granted['currently_repaying_other_loans']==1)].shape[0])*100,2))



print('--'*40)

print('--'*40)



# How many repay the loan when not paying for other loans

print('People who repay their loans not having any other loan payment (%) :')

print(round(((granted[(granted['currently_repaying_other_loans']==0) & (granted['loan_repaid']==1)].shape[0])/granted[(granted['currently_repaying_other_loans']==0)].shape[0])*100,2))

plot_stats(granted, 'fully_repaid_previous_loans', 'loan_repaid', label_rotation=False, horizontal_layout=True)
# How many of people repay their loans who have fully repaid previous loans

print('People who repay their loans who have fully repaid previous loans (%) :')

print(round(((granted[(granted['fully_repaid_previous_loans']==1) & (granted['loan_repaid']==1)].shape[0])/granted[(granted['fully_repaid_previous_loans']==1)].shape[0])*100,2))



print('--'*40)

print('--'*40)



# How many of people repay their loans who haven't fully repaid previous loans

print('People who repay their loans who have not fully repaid previous loans (%) :')

print(round(((granted[(granted['fully_repaid_previous_loans']==0) & (granted['loan_repaid']==1)].shape[0])/granted[(granted['fully_repaid_previous_loans']==0)].shape[0])*100,2))

fig, axes = plt.subplots(1, 3)



fig.set_size_inches(12, 4)



repaid.hist('saving_amount', bins=100, ax=axes[0])

axes[0].set_xlabel('repaid')

non_repaid.hist('saving_amount', bins=100, ax=axes[1])

axes[1].set_xlabel('non_repaid')

ungranted.hist('saving_amount', bins=100, ax=axes[2])

axes[2].set_xlabel('ungranted')



plt.show()
print(f'Average saving_amount by repaid group: {repaid["saving_amount"].median()}')

print(f'Average saving_amount by non-repaid group: {non_repaid["saving_amount"].median()}')

print(f'Average saving_amount by ungranted group: {ungranted["saving_amount"].median()}')
fig, axes = plt.subplots(1, 3)



fig.set_size_inches(12, 4)



repaid.hist('checking_amount', bins=100, ax=axes[0])

axes[0].set_xlabel('repaid')

non_repaid.hist('checking_amount', bins=100, ax=axes[1])

axes[1].set_xlabel('non_repaid')

ungranted.hist('checking_amount', bins=100, ax=axes[2])

axes[2].set_xlabel('ungranted')



plt.show()
print(f'Average checking_amount by repaid group: {repaid["checking_amount"].median()}')

print(f'Average checking_amount by non-repaid group: {non_repaid["checking_amount"].median()}')

print(f'Average checking_amount by ungranted group: {ungranted["checking_amount"].median()}')
fig, axes = plt.subplots(1, 3)



fig.set_size_inches(12, 4)



repaid.hist('yearly_salary', bins=100, ax=axes[0])

axes[0].set_xlabel('repaid')

non_repaid.hist('yearly_salary', bins=100, ax=axes[1])

axes[1].set_xlabel('non_repaid')

ungranted.hist('yearly_salary', bins=100, ax=axes[2])

axes[2].set_xlabel('ungranted')



plt.show()
print(f'Average yearly_salary by repaid group: {repaid["yearly_salary"].median()}')

print(f'Average yearly_salary by non-repaid group: {non_repaid["yearly_salary"].median()}')

print(f'Average yearly_salary by ungranted group: {ungranted["yearly_salary"].median()}')
fig, axes = plt.subplots(1, 3)



fig.set_size_inches(12, 4)



repaid.hist('total_credit_card_limit', bins=100, ax=axes[0])

axes[0].set_xlabel('repaid')

non_repaid.hist('total_credit_card_limit', bins=100, ax=axes[1])

axes[1].set_xlabel('non_repaid')

ungranted.hist('total_credit_card_limit', bins=100, ax=axes[2])

axes[2].set_xlabel('ungranted')



plt.show()
print(f'Average total_credit_card_limit by repaid group: {repaid["total_credit_card_limit"].median()}')

print(f'Average total_credit_card_limit by non-repaid group: {non_repaid["total_credit_card_limit"].median()}')

print(f'Average total_credit_card_limit by ungranted group: {ungranted["total_credit_card_limit"].median()}')
fig, axes = plt.subplots(1, 3)



fig.set_size_inches(12, 4)



repaid.hist('avg_percentage_credit_card_limit_used_last_year', bins=100, ax=axes[0])

axes[0].set_xlabel('repaid')

non_repaid.hist('avg_percentage_credit_card_limit_used_last_year', bins=100, ax=axes[1])

axes[1].set_xlabel('non_repaid')

ungranted.hist('avg_percentage_credit_card_limit_used_last_year', bins=100, ax=axes[2])

axes[2].set_xlabel('ungranted')



plt.show()
print(repaid['avg_percentage_credit_card_limit_used_last_year'].median())

print(non_repaid['avg_percentage_credit_card_limit_used_last_year'].median())

print(ungranted['avg_percentage_credit_card_limit_used_last_year'].median())
print(df_loan['loan_granted'].unique())

print(df_loan['loan_repaid'].unique())
print(df_loan.loc[df_loan.loan_granted ==0 ]['loan_repaid'].unique())

print(df_loan.loc[df_loan.loan_granted ==1 ]['loan_repaid'].unique())
df_loan.loc[((df_loan['loan_granted'] == 1) & (df_loan['loan_repaid'] == 1)), 'loan_quality'] = 0

df_loan.loc[((df_loan['loan_granted'] == 1) & (df_loan['loan_repaid'] == 0)), 'loan_quality'] = 1

df_loan.loc[(df_loan['loan_granted'] == 0), 'loan_quality'] = -1
print(df_loan['loan_quality'].unique())

print('--'*10)

print(df_loan.loc[(df_loan['loan_granted']==1)&(df_loan['loan_repaid']==1)]['loan_quality'].unique())

print(df_loan.loc[(df_loan['loan_granted']==1)&(df_loan['loan_repaid']==0)]['loan_quality'].unique())

print(df_loan.loc[(df_loan['loan_granted']==0)]['loan_quality'].unique())
# Create a copy of the loan dataset for feature transformation



df_loan_tr = df_loan.copy()
# Cast the datatype of "date" field

df_loan_tr['date']= pd.to_datetime(df_loan_tr['date'])

df_loan_tr['date'].dtypes
# Year from date

df_loan_tr['year'] = df_loan_tr['date'].dt.year



# Month from date

df_loan_tr['month'] = df_loan_tr['date'].dt.month



# Day from date

df_loan_tr['day'] = df_loan_tr['date'].dt.day



# Quarter from date

df_loan_tr['quarter'] = df_loan_tr['date'].dt.quarter



# Semester from date

df_loan_tr['semester'] = np.where(df_loan_tr.quarter.isin([1,2]),1,2)



# Day of the week from date

df_loan_tr['dayofweek'] = df_loan_tr['date'].dt.dayofweek



df_loan_tr[['date', 'year', 'month', 'day', 'quarter', 'semester', 'dayofweek']].head()
# Check the unique day of week

df_loan_tr['dayofweek'].unique()
# Unique values in loan_purpose



df_loan_tr["loan_purpose"].unique()
# Need to convert the datatypes of the feature to 'category' before Label encoding.

df_loan_tr["loan_purpose"] = df_loan_tr["loan_purpose"].astype('category')



# Label Encoding

df_loan_tr["loan_purpose_cat"] = df_loan_tr["loan_purpose"].cat.codes



df_loan_tr[["loan_purpose", "loan_purpose_cat"]].head(3)
# Calculate missing value count and percentage



missing_value_df_loan_tr = pd.DataFrame(index = df_loan_tr.keys(), data =df_loan_tr.isnull().sum(), columns = ['Missing_Value_Count'])

missing_value_df_loan_tr['Missing_Value_Percentage'] = np.round(((df_loan_tr.isnull().mean())*100),2)

missing_value_df_loan_tr.sort_values('Missing_Value_Count',ascending= False)
print(df_loan_tr['fully_repaid_previous_loans'].unique())

print(df_loan_tr.loc[df_loan_tr['fully_repaid_previous_loans'].isnull()]['is_first_loan'].unique())



# Impute fully_repaid_previous_loans with some numerical values for all the first loans



print(df_loan_tr['currently_repaying_other_loans'].unique())

print(df_loan_tr.loc[df_loan_tr['currently_repaying_other_loans'].isnull()]['is_first_loan'].unique())



# Impute currently_repaying_other_loans with some numerical values for all the first loans
# print(df_loan_tr['avg_percentage_credit_card_limit_used_last_year'].unique())

print(df_loan_tr.loc[df_loan_tr['avg_percentage_credit_card_limit_used_last_year'].isnull()]['total_credit_card_limit'].unique())

print(df_loan_tr.loc[df_loan_tr['total_credit_card_limit']==0]['avg_percentage_credit_card_limit_used_last_year'].unique())
# replace null with -1



df_loan_tr['fully_repaid_previous_loans'].fillna(-1, inplace=True)

df_loan_tr['currently_repaying_other_loans'].fillna(-1, inplace=True)

df_loan_tr['avg_percentage_credit_card_limit_used_last_year'].fillna(-1, inplace=True)

feature_set = ['is_first_loan', 

'fully_repaid_previous_loans', 

'currently_repaying_other_loans', 

'total_credit_card_limit', 

'avg_percentage_credit_card_limit_used_last_year', 

'saving_amount', 

'checking_amount', 

'is_employed', 

'yearly_salary',

'age', 

'dependent_number',

# 'month', 

# 'day', 

# 'quarter', 

# 'semester', 

# 'dayofweek', 

'loan_purpose_cat']
## Train and test on granted loans



granted_loan = df_loan_tr.loc[(df_loan_tr['loan_quality'] == 1) | (df_loan_tr['loan_quality'] == 0)]

print(f'#loan_id: {granted_loan.shape[0]} ')
print('cross check the target values')

print('--'*15)

print(granted_loan['loan_quality'].unique())

print(granted_loan['loan_granted'].unique())

print(granted_loan['loan_repaid'].unique())
## Predict / Score Population on ungranted loans



ungranted_loan = df_loan_tr.loc[(df_loan_tr['loan_quality'] == -1)]

print(f'#loan_id: {ungranted_loan.shape[0]} ')
print('cross check the target values')

print('--'*15)

print(ungranted_loan['loan_quality'].unique())

print(ungranted_loan['loan_granted'].unique())

print(ungranted_loan['loan_repaid'].unique())
train_granted = granted_loan.loc[~(granted_loan['month'].isin([1,2,3]))]

exog_train_granted = train_granted[feature_set]

endog_train_granted = train_granted['loan_quality']



print(train_granted.shape)
# Insample train and validation split



x_train, x_val, y_train, y_val = train_test_split(exog_train_granted, endog_train_granted, test_size=0.1, random_state=42)

test_granted = granted_loan.loc[(granted_loan['month'].isin([1,2,3]))]

exog_test = test_granted[feature_set]



print(test_granted.shape)
test_granted['loan_quality'].unique()
# Random Forest Model



rf = RandomForestClassifier(random_state = 42)
# Parameters used by the current forest



print('Parameters currently in use:\n')

pprint(rf.get_params())
base_model = RandomForestClassifier(n_estimators = 100, max_depth=5, random_state = 42)

base_model.fit(x_train, y_train)

base_accuracy = get_rocauc(base_model, x_val, y_val)
# Compute cross-validated AUC scores for the training set: cv_auc



cv_auc = cross_val_score(base_model, x_train, y_train, cv=5, scoring = 'roc_auc')



# Print list of AUC scores

print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))
features = exog_train_granted.columns

importances = base_model.feature_importances_

indices = np.argsort(importances)



plt.title('Feature Importances')

plt.barh(range(len(indices)), importances[indices], color='blue', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.show()
# save the model to disk

filename = 'randomforest_basemodel'

pickle.dump(base_model, open(filename, 'wb'))
# # Number of trees in random forest

# n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]



# # Number of features to consider at every split

# max_features = ['auto', 'sqrt']



# # Maximum number of levels in tree

# max_depth = [int(x) for x in np.linspace(1, 50, num = 15)]

# max_depth.append(None)



# # Minimum number of samples required to split a node

# min_samples_split = [2, 5, 10]



# # Minimum number of samples required at each leaf node

# min_samples_leaf = [1, 2, 4]



# # Method of selecting samples for training each tree

# bootstrap = [True, False]

# # Create the random grid

# random_grid = {'n_estimators': n_estimators,

#                'max_features': max_features,

#                'max_depth': max_depth,

#                'min_samples_split': min_samples_split,

#                'min_samples_leaf': min_samples_leaf,

#                'bootstrap': bootstrap}

# pprint(random_grid)
# rf1 = RandomForestClassifier()



# # Random search of parameters, using 3 fold cross validation

# # Search across 100 different combinations, and by using all available cores

# rf_random = RandomizedSearchCV(estimator = rf1, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)



# # Fit the random search model

# rf_random.fit(x_train, y_train)

granted_model = pickle.load(open('randomforest_basemodel', 'rb'))

pred_test_granted = granted_model.predict_proba(exog_test)[:, 1]



print(pred_test_granted)
roc_test_granted = roc_auc_score(list(test_granted['loan_quality']), pred_test_granted)

print(round(roc_test_granted,2))
plot_roc(test_granted['loan_quality'], pred_test_granted)
capture_curve(test_granted, test_granted['loan_quality'], pred_test_granted, round(roc_test_granted,2), 'Granted Loans in 2012 Q1')
# Create a copy of test data

test_granted_preds=test_granted.copy()
test_granted_preds['preds']=list(pred_test_granted)
sorted_test_granted_preds = test_granted_preds.sort_values(by ='preds' , ascending=False)

sorted_test_granted_preds.shape
test_granted_preds.shape
df_decile_cutoff_test_granted = decile_cutoff_risk_detected(sorted_test_granted_preds)

df_decile_cutoff_test_granted
df_cum_decile_cutoff_test_granted = cum_decile_cutoff_risk_detected(sorted_test_granted_preds)

df_cum_decile_cutoff_test_granted
# independent variables of ungranted loans

exog_ungranted = ungranted_loan[feature_set]
# predictions for ungranted loans

pred_ungranted = granted_model.predict_proba(exog_ungranted)[:, 1]

print(pred_ungranted)
# create copy of ungranted loans

ungranted_preds=ungranted_loan.copy()



# Attaching predictions with ungranted dataframes

ungranted_preds['preds']=list(pred_ungranted)
# cross checking shape of both the dataframes



print(ungranted_loan.shape)

print(ungranted_preds.shape)
ungranted_lt_1_risk = ungranted_preds.loc[ungranted_preds['preds'] < 0.15]

print(ungranted_lt_1_risk.shape)

ungranted_lt_1_risk.head()
ungranted_lt_10_risk = ungranted_preds.loc[ungranted_preds['preds'] < 0.4]

print(ungranted_lt_10_risk.shape)

ungranted_lt_10_risk.head()
# Export loan_id with < 1% risk

ungranted_lt_1_risk['loan_id'].to_csv('loan_preds_lt_1_risk.csv')



# Export loan_id with < 10% risk

ungranted_lt_10_risk['loan_id'].to_csv('loan_preds_lt_10_risk.csv')
granted_2012Q1 = granted_loan.loc[granted_loan['month'].isin([1,2,3])]

print('Maximum Possible Score:')

print('--'*12)

print((granted_2012Q1.shape[0]) + (ungranted_loan.shape[0]))
# Bank Profitability Calculation



bank_score = (granted_2012Q1.loc[(granted_2012Q1['loan_repaid']==1)].shape[0]) - (granted_2012Q1.loc[(granted_2012Q1['loan_repaid']==0)].shape[0])



print('Bank score:')

print('--'*12)

print(bank_score)



# print((granted_2012Q1.loc[(granted_2012Q1['loan_repaid']==1)].shape[0]))

# print((granted_2012Q1.loc[(granted_2012Q1['loan_repaid']==0)].shape[0]))

# ML Model Profitability Calculation with < 1% risk



granted_2012Q1_lt_1_risk      = test_granted_preds[test_granted_preds['preds'] < 0.15]

loss_granted_2012Q1_lt_1_risk = granted_2012Q1_lt_1_risk[granted_2012Q1_lt_1_risk['loan_repaid']==0]

ml_model_score = ((granted_2012Q1_lt_1_risk.shape[0]) - (loss_granted_2012Q1_lt_1_risk.shape[0])) + (ungranted_lt_1_risk.shape[0] - int((ungranted_lt_1_risk.shape[0])/100))



print('ML Model score:')

print('--'*10)

print(ml_model_score)
profitability_diff('bank_score', 'ml_model_score', bank_score, ml_model_score)