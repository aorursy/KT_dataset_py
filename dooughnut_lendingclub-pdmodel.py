import numpy as np

import pandas as pd
loan_data_inputs_train = pd.read_csv('../input/lendingclub-cleaneddata-withdummies/loan_data_inputs_train.csv', index_col = 0)

loan_data_targets_train = pd.read_csv('../input/lendingclub-cleaneddata-withdummies/loan_data_targets_train.csv', index_col = 0, header = None)

loan_data_inputs_test = pd.read_csv('../input/lendingclub-cleaneddata-withdummies/loan_data_inputs_test.csv', index_col = 0)

loan_data_targets_test = pd.read_csv('../input/lendingclub-cleaneddata-withdummies/loan_data_targets_test.csv', index_col = 0, header = None)
loan_data_inputs_train.head()
loan_data_targets_train.head()
loan_data_inputs_train.shape
loan_data_targets_train.shape
loan_data_inputs_test.shape
loan_data_targets_test.shape
# Here we select a limited set of input variables in a new dataframe.

inputs_train_with_ref_cat = loan_data_inputs_train.loc[: , ['grade:A',

'grade:B',

'grade:C',

'grade:D',

'grade:E',

'grade:F',

'grade:G',

'home_ownership:RENT_OTHER_NONE_ANY',

'home_ownership:OWN',

'home_ownership:MORTGAGE',

'addr_state:ND_NE_IA_NV_FL_HI_AL',

'addr_state:NM_VA',

'addr_state:NY',

'addr_state:OK_TN_MO_LA_MD_NC',

'addr_state:CA',

'addr_state:UT_KY_AZ_NJ',

'addr_state:AR_MI_PA_OH_MN',

'addr_state:RI_MA_DE_SD_IN',

'addr_state:GA_WA_OR',

'addr_state:WI_MT',

'addr_state:TX',

'addr_state:IL_CT',

'addr_state:KS_SC_CO_VT_AK_MS',

'addr_state:WV_NH_WY_DC_ME_ID',

'verification_status:Not Verified',

'verification_status:Source Verified',

'verification_status:Verified',

'purpose:educ__sm_b__wedd__ren_en__mov__house',

'purpose:credit_card',

'purpose:debt_consolidation',

'purpose:oth__med__vacation',

'purpose:major_purch__car__home_impr',

'initial_list_status:f',

'initial_list_status:w',

'term:36',

'term:60',

'emp_length:0',

'emp_length:1',

'emp_length:2-4',

'emp_length:5-6',

'emp_length:7-9',

'emp_length:10',

'mths_since_issue_d:<38',

'mths_since_issue_d:38-39',

'mths_since_issue_d:40-41',

'mths_since_issue_d:42-48',

'mths_since_issue_d:49-52',

'mths_since_issue_d:53-64',

'mths_since_issue_d:65-84',

'mths_since_issue_d:>84',

'int_rate:<9.548',

'int_rate:9.548-12.025',

'int_rate:12.025-15.74',

'int_rate:15.74-20.281',

'int_rate:>20.281',

'mths_since_earliest_cr_line:<140',

'mths_since_earliest_cr_line:141-164',

'mths_since_earliest_cr_line:165-247',

'mths_since_earliest_cr_line:248-270',

'mths_since_earliest_cr_line:271-352',

'mths_since_earliest_cr_line:>352',

'delinq_2yrs:0',

'delinq_2yrs:1-3',

'delinq_2yrs:>=4',

'inq_last_6mths:0',

'inq_last_6mths:1-2',

'inq_last_6mths:3-6',

'inq_last_6mths:>6',

'open_acc:0',

'open_acc:1-3',

'open_acc:4-12',

'open_acc:13-17',

'open_acc:18-22',

'open_acc:23-25',

'open_acc:26-30',

'open_acc:>=31',

'pub_rec:0-2',

'pub_rec:3-4',

'pub_rec:>=5',

'total_acc:<=27',

'total_acc:28-51',

'total_acc:>=52',

'acc_now_delinq:0',

'acc_now_delinq:>=1',

'total_rev_hi_lim:<=5K',

'total_rev_hi_lim:5K-10K',

'total_rev_hi_lim:10K-20K',

'total_rev_hi_lim:20K-30K',

'total_rev_hi_lim:30K-40K',

'total_rev_hi_lim:40K-55K',

'total_rev_hi_lim:55K-95K',

'total_rev_hi_lim:>95K',

'annual_inc:<20K',

'annual_inc:20K-30K',

'annual_inc:30K-40K',

'annual_inc:40K-50K',

'annual_inc:50K-60K',

'annual_inc:60K-70K',

'annual_inc:70K-80K',

'annual_inc:80K-90K',

'annual_inc:90K-100K',

'annual_inc:100K-120K',

'annual_inc:120K-140K',

'annual_inc:>140K',

'dti:<=1.4',

'dti:1.4-3.5',

'dti:3.5-7.7',

'dti:7.7-10.5',

'dti:10.5-16.1',

'dti:16.1-20.3',

'dti:20.3-21.7',

'dti:21.7-22.4',

'dti:22.4-35',

'dti:>35',

'mths_since_last_delinq:Missing',

'mths_since_last_delinq:0-3',

'mths_since_last_delinq:4-30',

'mths_since_last_delinq:31-56',

'mths_since_last_delinq:>=57',

'mths_since_last_record:Missing',

'mths_since_last_record:0-2',

'mths_since_last_record:3-20',

'mths_since_last_record:21-31',

'mths_since_last_record:32-80',

'mths_since_last_record:81-86',

]]
# Here we store the names of the reference category dummy variables in a list.

ref_categories = ['grade:G',

'home_ownership:RENT_OTHER_NONE_ANY',

'addr_state:ND_NE_IA_NV_FL_HI_AL',

'verification_status:Verified',

'purpose:educ__sm_b__wedd__ren_en__mov__house',

'initial_list_status:f',

'term:60',

'emp_length:0',

'mths_since_issue_d:>84',

'int_rate:>20.281',

'mths_since_earliest_cr_line:<140',

'delinq_2yrs:>=4',

'inq_last_6mths:>6',

'open_acc:0',

'pub_rec:0-2',

'total_acc:<=27',

'acc_now_delinq:0',

'total_rev_hi_lim:<=5K',

'annual_inc:<20K',

'dti:>35',

'mths_since_last_delinq:0-3',

'mths_since_last_record:0-2']
inputs_train = inputs_train_with_ref_cat.drop(ref_categories, axis = 1)

# From the dataframe with input variables, we drop the variables with variable names in the list with reference categories. 

inputs_train.head()
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
reg = LogisticRegression()

# We create an instance of an object from the 'LogisticRegression' class.
pd.options.display.max_rows = None

# Sets the pandas dataframe options to display all columns/ rows.
reg.fit(inputs_train, loan_data_targets_train)

# Estimates the coefficients of the object from the 'LogisticRegression' class

# with inputs (independent variables) contained in the first dataframe

# and targets (dependent variables) contained in the second dataframe.
reg.intercept_

# Displays the intercept contain in the estimated ("fitted") object from the 'LogisticRegression' class.
reg.coef_

# Displays the coefficients contained in the estimated ("fitted") object from the 'LogisticRegression' class.
inputs_train.columns.values
feature_name = inputs_train.columns.values

# Stores the names of the columns of a dataframe in a variable.
summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)

# Creates a dataframe with a column titled 'Feature name' and row values contained in the 'feature_name' variable.

summary_table['Coefficients'] = np.transpose(reg.coef_)

# Creates a new column in the dataframe, called 'Coefficients',

# with row values the transposed coefficients from the 'LogisticRegression' object.

summary_table.index = summary_table.index + 1

# Increases the index of every row of the dataframe with 1.

summary_table.loc[0] = ['Intercept', reg.intercept_[0]]

# Assigns values of the row with index 0 of the dataframe.

summary_table = summary_table.sort_index()

# Sorts the dataframe by index.

summary_table
# P values for sklearn logistic regression.



# Class to display p-values for logistic regression in sklearn.



from sklearn import linear_model

import scipy.stats as stat



class LogisticRegression_with_p_values:

    

    def __init__(self,*args,**kwargs):#,**kwargs):

        self.model = linear_model.LogisticRegression(*args,**kwargs)#,**args)



    def fit(self,X,y):

        self.model.fit(X,y)

        

        #### Get p-values for the fitted model ####

        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))

        denom = np.tile(denom,(X.shape[1],1)).T

        F_ij = np.dot((X / denom).T,X) ## Fisher Information Matrix

        Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix

        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))

        z_scores = self.model.coef_[0] / sigma_estimates # z-score for eaach model coefficient

        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores] ### two tailed test for p-values

        

        self.coef_ = self.model.coef_

        self.intercept_ = self.model.intercept_

        #self.z_scores = z_scores

        self.p_values = p_values

        #self.sigma_estimates = sigma_estimates

        #self.F_ij = F_ij
from sklearn import linear_model

import scipy.stats as stat



class LogisticRegression_with_p_values:

    

    def __init__(self,*args,**kwargs):

        self.model = linear_model.LogisticRegression(*args,**kwargs)



    def fit(self,X,y):

        self.model.fit(X,y)

        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))

        denom = np.tile(denom,(X.shape[1],1)).T

        F_ij = np.dot((X / denom).T,X)

        Cramer_Rao = np.linalg.inv(F_ij)

        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))

        z_scores = self.model.coef_[0] / sigma_estimates

        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]

        self.coef_ = self.model.coef_

        self.intercept_ = self.model.intercept_

        self.p_values = p_values
reg = LogisticRegression_with_p_values()

# We create an instance of an object from the newly created 'LogisticRegression_with_p_values()' class.
reg.fit(inputs_train, loan_data_targets_train)

# Estimates the coefficients of the object from the 'LogisticRegression' class

# with inputs (independent variables) contained in the first dataframe

# and targets (dependent variables) contained in the second dataframe.
# Same as above.

summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)

summary_table['Coefficients'] = np.transpose(reg.coef_)

summary_table.index = summary_table.index + 1

summary_table.loc[0] = ['Intercept', reg.intercept_[0]]

summary_table = summary_table.sort_index()

summary_table
# This is a list.

p_values = reg.p_values

# We take the result of the newly added method 'p_values' and store it in a variable 'p_values'.
# Add the intercept for completeness.

p_values = np.append(np.nan, np.array(p_values))

# We add the value 'NaN' in the beginning of the variable with p-values.
summary_table['p_values'] = p_values

# In the 'summary_table' dataframe, we add a new column, called 'p_values', containing the values from the 'p_values' variable.
summary_table
# We are going to remove some features, the coefficients for all or almost all of the dummy variables for which,

# are not tatistically significant.



# We do that by specifying another list of dummy variables as reference categories, and a list of variables to remove.

# Then, we are going to drop the two datasets from the original list of dummy variables.



# Variables

inputs_train_with_ref_cat = loan_data_inputs_train.loc[: , ['grade:A',

'grade:B',

'grade:C',

'grade:D',

'grade:E',

'grade:F',

'grade:G',

'home_ownership:RENT_OTHER_NONE_ANY',

'home_ownership:OWN',

'home_ownership:MORTGAGE',

'addr_state:ND_NE_IA_NV_FL_HI_AL',

'addr_state:NM_VA',

'addr_state:NY',

'addr_state:OK_TN_MO_LA_MD_NC',

'addr_state:CA',

'addr_state:UT_KY_AZ_NJ',

'addr_state:AR_MI_PA_OH_MN',

'addr_state:RI_MA_DE_SD_IN',

'addr_state:GA_WA_OR',

'addr_state:WI_MT',

'addr_state:TX',

'addr_state:IL_CT',

'addr_state:KS_SC_CO_VT_AK_MS',

'addr_state:WV_NH_WY_DC_ME_ID',

'verification_status:Not Verified',

'verification_status:Source Verified',

'verification_status:Verified',

'purpose:educ__sm_b__wedd__ren_en__mov__house',

'purpose:credit_card',

'purpose:debt_consolidation',

'purpose:oth__med__vacation',

'purpose:major_purch__car__home_impr',

'initial_list_status:f',

'initial_list_status:w',

'term:36',

'term:60',

'emp_length:0',

'emp_length:1',

'emp_length:2-4',

'emp_length:5-6',

'emp_length:7-9',

'emp_length:10',

'mths_since_issue_d:<38',

'mths_since_issue_d:38-39',

'mths_since_issue_d:40-41',

'mths_since_issue_d:42-48',

'mths_since_issue_d:49-52',

'mths_since_issue_d:53-64',

'mths_since_issue_d:65-84',

'mths_since_issue_d:>84',

'int_rate:<9.548',

'int_rate:9.548-12.025',

'int_rate:12.025-15.74',

'int_rate:15.74-20.281',

'int_rate:>20.281',

'mths_since_earliest_cr_line:<140',

'mths_since_earliest_cr_line:141-164',

'mths_since_earliest_cr_line:165-247',

'mths_since_earliest_cr_line:248-270',

'mths_since_earliest_cr_line:271-352',

'mths_since_earliest_cr_line:>352',

'inq_last_6mths:0',

'inq_last_6mths:1-2',

'inq_last_6mths:3-6',

'inq_last_6mths:>6',

'acc_now_delinq:0',

'acc_now_delinq:>=1',

'annual_inc:<20K',

'annual_inc:20K-30K',

'annual_inc:30K-40K',

'annual_inc:40K-50K',

'annual_inc:50K-60K',

'annual_inc:60K-70K',

'annual_inc:70K-80K',

'annual_inc:80K-90K',

'annual_inc:90K-100K',

'annual_inc:100K-120K',

'annual_inc:120K-140K',

'annual_inc:>140K',

'dti:<=1.4',

'dti:1.4-3.5',

'dti:3.5-7.7',

'dti:7.7-10.5',

'dti:10.5-16.1',

'dti:16.1-20.3',

'dti:20.3-21.7',

'dti:21.7-22.4',

'dti:22.4-35',

'dti:>35',

'mths_since_last_delinq:Missing',

'mths_since_last_delinq:0-3',

'mths_since_last_delinq:4-30',

'mths_since_last_delinq:31-56',

'mths_since_last_delinq:>=57',

'mths_since_last_record:Missing',

'mths_since_last_record:0-2',

'mths_since_last_record:3-20',

'mths_since_last_record:21-31',

'mths_since_last_record:32-80',

'mths_since_last_record:81-86',

]]



ref_categories = ['grade:G',

'home_ownership:RENT_OTHER_NONE_ANY',

'addr_state:ND_NE_IA_NV_FL_HI_AL',

'verification_status:Verified',

'purpose:educ__sm_b__wedd__ren_en__mov__house',

'initial_list_status:f',

'term:60',

'emp_length:0',

'mths_since_issue_d:>84',

'int_rate:>20.281',

'mths_since_earliest_cr_line:<140',

'inq_last_6mths:>6',

'acc_now_delinq:0',

'annual_inc:<20K',

'dti:>35',

'mths_since_last_delinq:0-3',

'mths_since_last_record:0-2']
inputs_train = inputs_train_with_ref_cat.drop(ref_categories, axis = 1)

inputs_train.head()
# Here we run a new model.

reg2 = LogisticRegression_with_p_values()

reg2.fit(inputs_train, loan_data_targets_train)
feature_name = inputs_train.columns.values
# Same as above.

summary_table = pd.DataFrame(columns = ['Feature name'], data = feature_name)

summary_table['Coefficients'] = np.transpose(reg2.coef_)

summary_table.index = summary_table.index + 1

summary_table.loc[0] = ['Intercept', reg2.intercept_[0]]

summary_table = summary_table.sort_index()

summary_table
# We add the 'p_values' here, just as we did before.

p_values = reg2.p_values

p_values = np.append(np.nan,np.array(p_values))

summary_table['p_values'] = p_values

summary_table

# Here we get the results for our final PD model.
import pickle
pickle.dump(reg2, open('pd_model.sav', 'wb'))

# Here we export our model to a 'SAV' file with file name 'pd_model.sav'.
# Here, from the dataframe with inputs for testing, we keep the same variables that we used in our final PD model.

inputs_test_with_ref_cat = loan_data_inputs_test.loc[: , ['grade:A',

'grade:B',

'grade:C',

'grade:D',

'grade:E',

'grade:F',

'grade:G',

'home_ownership:RENT_OTHER_NONE_ANY',

'home_ownership:OWN',

'home_ownership:MORTGAGE',

'addr_state:ND_NE_IA_NV_FL_HI_AL',

'addr_state:NM_VA',

'addr_state:NY',

'addr_state:OK_TN_MO_LA_MD_NC',

'addr_state:CA',

'addr_state:UT_KY_AZ_NJ',

'addr_state:AR_MI_PA_OH_MN',

'addr_state:RI_MA_DE_SD_IN',

'addr_state:GA_WA_OR',

'addr_state:WI_MT',

'addr_state:TX',

'addr_state:IL_CT',

'addr_state:KS_SC_CO_VT_AK_MS',

'addr_state:WV_NH_WY_DC_ME_ID',

'verification_status:Not Verified',

'verification_status:Source Verified',

'verification_status:Verified',

'purpose:educ__sm_b__wedd__ren_en__mov__house',

'purpose:credit_card',

'purpose:debt_consolidation',

'purpose:oth__med__vacation',

'purpose:major_purch__car__home_impr',

'initial_list_status:f',

'initial_list_status:w',

'term:36',

'term:60',

'emp_length:0',

'emp_length:1',

'emp_length:2-4',

'emp_length:5-6',

'emp_length:7-9',

'emp_length:10',

'mths_since_issue_d:<38',

'mths_since_issue_d:38-39',

'mths_since_issue_d:40-41',

'mths_since_issue_d:42-48',

'mths_since_issue_d:49-52',

'mths_since_issue_d:53-64',

'mths_since_issue_d:65-84',

'mths_since_issue_d:>84',

'int_rate:<9.548',

'int_rate:9.548-12.025',

'int_rate:12.025-15.74',

'int_rate:15.74-20.281',

'int_rate:>20.281',

'mths_since_earliest_cr_line:<140',

'mths_since_earliest_cr_line:141-164',

'mths_since_earliest_cr_line:165-247',

'mths_since_earliest_cr_line:248-270',

'mths_since_earliest_cr_line:271-352',

'mths_since_earliest_cr_line:>352',

'inq_last_6mths:0',

'inq_last_6mths:1-2',

'inq_last_6mths:3-6',

'inq_last_6mths:>6',

'acc_now_delinq:0',

'acc_now_delinq:>=1',

'annual_inc:<20K',

'annual_inc:20K-30K',

'annual_inc:30K-40K',

'annual_inc:40K-50K',

'annual_inc:50K-60K',

'annual_inc:60K-70K',

'annual_inc:70K-80K',

'annual_inc:80K-90K',

'annual_inc:90K-100K',

'annual_inc:100K-120K',

'annual_inc:120K-140K',

'annual_inc:>140K',

'dti:<=1.4',

'dti:1.4-3.5',

'dti:3.5-7.7',

'dti:7.7-10.5',

'dti:10.5-16.1',

'dti:16.1-20.3',

'dti:20.3-21.7',

'dti:21.7-22.4',

'dti:22.4-35',

'dti:>35',

'mths_since_last_delinq:Missing',

'mths_since_last_delinq:0-3',

'mths_since_last_delinq:4-30',

'mths_since_last_delinq:31-56',

'mths_since_last_delinq:>=57',

'mths_since_last_record:Missing',

'mths_since_last_record:0-2',

'mths_since_last_record:3-20',

'mths_since_last_record:21-31',

'mths_since_last_record:32-80',

'mths_since_last_record:81-86'

]]
# And here, in the list below, we keep the variable names for the reference categories,

# only for the variables we used in our final PD model.

ref_categories = ['grade:G',

'home_ownership:RENT_OTHER_NONE_ANY',

'addr_state:ND_NE_IA_NV_FL_HI_AL',

'verification_status:Verified',

'purpose:educ__sm_b__wedd__ren_en__mov__house',

'initial_list_status:f',

'term:60',

'emp_length:0',

'mths_since_issue_d:>84',

'int_rate:>20.281',

'mths_since_earliest_cr_line:<140',

'inq_last_6mths:>6',

'acc_now_delinq:0',

'annual_inc:<20K',

'dti:>35',

'mths_since_last_delinq:0-3',

'mths_since_last_record:0-2']
inputs_test = inputs_test_with_ref_cat.drop(ref_categories, axis = 1)

inputs_test.head()
y_hat_test = reg2.model.predict(inputs_test)

# Calculates the predicted values for the dependent variable (targets)

# based on the values of the independent variables (inputs) supplied as an argument.
y_hat_test

# This is an array of predicted discrete classess (in this case, 0s and 1s).
y_hat_test_proba = reg2.model.predict_proba(inputs_test)

# Calculates the predicted probability values for the dependent variable (targets)

# based on the values of the independent variables (inputs) supplied as an argument.
y_hat_test_proba

# This is an array of arrays of predicted class probabilities for all classes.

# In this case, the first value of every sub-array is the probability for the observation to belong to the first class, i.e. 0,

# and the second value is the probability for the observation to belong to the first class, i.e. 1.
y_hat_test_proba[:][:,1]

# Here we take all the arrays in the array, and from each array, we take all rows, and only the element with index 1,

# that is, the second element.

# In other words, we take only the probabilities for being 1.
y_hat_test_proba = y_hat_test_proba[: ][: , 1]

# We store these probabilities in a variable.
y_hat_test_proba

# This variable contains an array of probabilities of being 1.
loan_data_targets_test_temp = loan_data_targets_test
loan_data_targets_test_temp.reset_index(drop = True, inplace = True)

# We reset the index of a dataframe.
df_actual_predicted_probs = pd.concat([loan_data_targets_test_temp, pd.DataFrame(y_hat_test_proba)], axis = 1)

# Concatenates two dataframes.
df_actual_predicted_probs.shape
df_actual_predicted_probs.columns = ['loan_data_targets_test', 'y_hat_test_proba']
df_actual_predicted_probs.index = loan_data_inputs_test.index

# Makes the index of one dataframe equal to the index of another dataframe.
df_actual_predicted_probs.head()
tr = 0.9

# We create a new column with an indicator,

# where every observation that has predicted probability greater than the threshold has a value of 1,

# and every observation that has predicted probability lower than the threshold has a value of 0.

df_actual_predicted_probs['y_hat_test'] = np.where(df_actual_predicted_probs['y_hat_test_proba'] > tr, 1, 0)
pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted'])

# Creates a cross-table where the actual values are displayed by rows and the predicted values by columns.

# This table is known as a Confusion Matrix.
pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]

# Here we divide each value of the table by the total number of observations,

# thus getting percentages, or, rates.
(pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[0, 0] + (pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[1, 1]

# Here we calculate Accuracy of the model, which is the sum of the diagonal rates.
from sklearn.metrics import roc_curve, roc_auc_score
roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])

# Returns the Receiver Operating Characteristic (ROC) Curve from a set of actual values and their predicted probabilities.

# As a result, we get three arrays: the false positive rates, the true positive rates, and the thresholds.
fpr, tpr, thresholds = roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])

# Here we store each of the three arrays in a separate variable. 
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
plt.plot(fpr, tpr)

# We plot the false positive rate along the x-axis and the true positive rate along the y-axis,

# thus plotting the ROC curve.

plt.plot(fpr, fpr, linestyle = '--', color = 'k')

# We plot a seconary diagonal line, with dashed line style and black color.

plt.xlabel('False positive rate')

# We name the x-axis "False positive rate".

plt.ylabel('True positive rate')

# We name the x-axis "True positive rate".

plt.title('ROC curve')

# We name the graph "ROC curve".
AUROC = roc_auc_score(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])

# Calculates the Area Under the Receiver Operating Characteristic Curve (AUROC)

# from a set of actual values and their predicted probabilities.

AUROC
from sklearn.metrics import classification_report
a = classification_report(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'])
print(a)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(inputs_train, loan_data_targets_train)
preds = clf.predict(inputs_test)
a = classification_report(df_actual_predicted_probs['loan_data_targets_test'], preds)

print(a)