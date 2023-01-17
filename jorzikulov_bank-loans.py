# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Activation

from tensorflow.keras.models import Sequential

from keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping

from keras.regularizers import l1_l2

from tensorflow.keras.layers import BatchNormalization



from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression



from sklearn.tree import export_graphviz

from sklearn.tree import plot_tree



import xgboost as xgb



import shap

pd.options.display.max_columns=25

df = pd.read_csv("../input/loan.csv", nrows=550000)
#Data Preparation

df.reset_index(inplace=True)

df = df[['loan_status', 'total_pymnt', 'loan_amnt', 'term', 'int_rate', 

          'installment', 'purpose', 'annual_inc', 'verification_status',

        'emp_length', 'home_ownership',  'open_acc','last_pymnt_amnt',

        'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'revol_bal', 'revol_util',

         'total_acc', 'addr_state']]

indices = df[(df['loan_status'] == 'Current')|(df['loan_status'] == 'In Grace Period')].index

df.drop(indices, inplace=True)

df.dropna(inplace=True)



df['loan_status'].replace(['Fully Paid','Does not meet the credit policy. Status:Fully Paid'], 0, inplace=True)

df['loan_status'].replace(['Late (16-30 days)','Late (31-120 days)', 'Charged Off', 'Default',

                           'Does not meet the credit policy. Status:Charged Off'], 1, inplace=True)

df['term'] = df['term'].str.split(' ').str.get(1)

df['term'] = pd.to_numeric(df['term'])

df['int_rate'] = pd.to_numeric(df['int_rate'])

df.reset_index(inplace=True)

df['loan_status']=df['loan_status'].astype('category')

df.drop(['index'], axis=1, inplace=True)
#Descriptive Statistics

print('Descriptive Statistics by Default Category')

print(df[['loan_status','total_pymnt', 'loan_amnt', 'term', 'int_rate',

       'installment', 'annual_inc', 'open_acc', 'last_pymnt_amnt',

       'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'revol_bal',

       'revol_util', 'total_acc']].groupby('loan_status').agg(['mean', 'std']))



plt.figure()

sns.violinplot(x='loan_status', y='loan_amnt', data=df)

plt.ylabel('Loan Amount in USD')

plt.xlabel('Loan Status')

plt.show()

plt.subplot(2,1,1)

plt.hist(df[df['loan_status'] == 1]['loan_amnt'], label='Default', density=True)  

plt.legend()

plt.subplot(2,1,2)

plt.hist(df[df['loan_status'] == 0]['loan_amnt'], color='green', label='Non-default', density=True)

plt.xlabel('Loan Amount in USD')

plt.legend()



plt.figure()

sns.violinplot(x='loan_status', y='total_pymnt', data=df)

plt.ylabel('Total Payment in USD')

plt.xlabel('Loan Status')

plt.show()

plt.subplot(2,1,1)

plt.hist(df[df['loan_status'] == 1]['total_pymnt'], label='Default', density=True)  

plt.legend()

plt.subplot(2,1,2)

plt.hist(df[df['loan_status'] == 0]['total_pymnt'], color='green', label='Non-default', density=True)

plt.xlabel('Total Payment in USD')

plt.legend()



plt.figure()

sns.violinplot(x='loan_status', y='term', data=df)

plt.ylabel('Loan Term in Months')

plt.xlabel('Loan Status')

plt.show()

plt.subplot(2,1,1)

plt.hist(df[df['loan_status'] == 1]['term'], label='Default', density=True)  

plt.legend()

plt.subplot(2,1,2)

plt.hist(df[df['loan_status'] == 0]['term'], color='green', label='Non-default', density=True)

plt.xlabel('Term in Months')

plt.legend()



plt.figure()

sns.violinplot(x='loan_status', y='int_rate', data=df)

plt.ylabel('Interest Rate')

plt.xlabel('Loan Status')

plt.show()

plt.subplot(2,1,1)

plt.hist(df[df['loan_status'] == 1]['int_rate'], label='Default', density=True)  

plt.legend()

plt.subplot(2,1,2)

plt.hist(df[df['loan_status'] == 0]['int_rate'], color='green', label='Non-default', density=True)

plt.xlabel('Interest Rate')

plt.legend()



plt.figure()

sns.violinplot(x='loan_status', y='installment', data=df)

plt.ylabel('Installment Per Month in USD')

plt.xlabel('Loan Status')

plt.show()

plt.subplot(2,1,1)

plt.hist(df[df['loan_status'] == 1]['installment'], label='Default', density=True)  

plt.legend()

plt.subplot(2,1,2)

plt.hist(df[df['loan_status'] == 0]['installment'], color='green', label='Non-default', density=True)

plt.xlabel('Installment Per Month')

plt.legend()



plt.figure()

sns.violinplot(x='loan_status', y='last_pymnt_amnt', data=df)

plt.ylabel('Last Payment in USD')

plt.xlabel('Loan Status')

plt.show()

plt.subplot(2,1,1)

plt.hist(df[df['loan_status'] == 1]['last_pymnt_amnt'],

         label='Default', density=True, bins=200)  

plt.legend()

plt.xlim([0, 2000])

plt.subplot(2,1,2)

plt.hist(df[df['loan_status'] == 0]['last_pymnt_amnt'], color='green', label='Non-default', density=True)

plt.xlabel('Last Payment in USD')

plt.legend()



plt.figure()

sns.violinplot(x='loan_status', y='annual_inc', data=df)

plt.ylabel('Annual Income in USD')

plt.xlabel('Loan Status')

plt.show()

plt.subplot(2,1,1)

plt.hist(df[df['loan_status'] == 1]['annual_inc'],

         label='Default', density=True, bins=700)  

plt.xlim([10000, 160000])

plt.legend()

plt.subplot(2,1,2)

plt.hist(df[df['loan_status'] == 0]['annual_inc'], color='green',

         label='Non-default', density=True, bins=700)

plt.xlim([10000, 160000])

plt.xlabel('Annual Income in USD')

plt.legend()



plt.figure()

sns.violinplot(x='loan_status', y='revol_util', data=df)

plt.ylabel('% Usage of Credit Balance')

plt.xlabel('Loan Status')

plt.show()

plt.subplot(2,1,1)

plt.hist(df[df['loan_status'] == 1]['revol_util'],

         label='Default', density=True, bins=100) 

plt.xlim([0, 100]) 

plt.ylim([0, 0.03])

plt.legend()

plt.subplot(2,1,2)

plt.hist(df[df['loan_status'] == 0]['revol_util'], color='green',

         label='Non-default', density=True, bins=100)

plt.ylim([0, 0.03])

plt.xlim([0, 100])

plt.xlabel('% Usage of Credit Balance')

plt.legend()



plt.figure()

sns.violinplot(x='loan_status', y='revol_bal', data=df)

plt.ylabel('Revolving Balance in USD')

plt.xlabel('Loan Status')

plt.show()

plt.subplot(2,1,1)

plt.hist(df[df['loan_status'] == 1]['revol_bal'],

         label='Default', density=True, bins=200) 

plt.xlim([0, 100000]) 

# plt.ylim([0, 0.03])

plt.legend()

plt.subplot(2,1,2)

plt.hist(df[df['loan_status'] == 0]['revol_bal'], color='green',

         label='Non-default', density=True, bins=200)

# plt.ylim([0, 0.03])

plt.xlim([0, 100000])

plt.xlabel('Revolving Balance in USD')

plt.legend()



#Descriptive for Categorical Data

print('No of Defaults By Loan Purpose')

print(df[df['loan_status']==1]['purpose'].value_counts())

print('No of Non-Defaults By Loan Purpose')

print(df[df['loan_status']==0]['purpose'].value_counts())

print('No of Defaults By Employment Length')

print(df[df['loan_status']==1]['emp_length'].value_counts())

print('No of Non-Defaults By Employment Length')

print(df[df['loan_status']==0]['emp_length'].value_counts())

print('No of Defaults By Home Ownership')

print(df[df['loan_status']==1]['home_ownership'].value_counts())

print('No of Non-Defaults By Home Ownership')

print(df[df['loan_status']==0]['home_ownership'].value_counts())



#Correlation Table

plt.clf()

sns.heatmap(df[['total_pymnt', 'loan_amnt', 'term', 'int_rate',

       'installment', 'annual_inc', 'open_acc', 'last_pymnt_amnt',

       'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'revol_bal',

       'revol_util', 'total_acc']].corr(), cmap='RdYlGn')

plt.title('Correlation Matrix')

plt.show()
#Getting Dummies

purpose = pd.get_dummies(df['purpose'], prefix='pur', drop_first=True)



df = df.merge(purpose, how='outer', left_index=True, right_index=True)



df['verification_status'].replace('Source Verified', 'Verified', inplace=True)

verst = pd.get_dummies(df['verification_status'], prefix='verst', drop_first=True)

df = df.merge(verst, how='outer', left_index=True, right_index=True)



emp_ln = pd.get_dummies(df['emp_length'], prefix='emp_l', drop_first=True)



df = df.merge(emp_ln, how='outer', left_index=True, right_index=True)



df['home_ownership'].replace('NONE', 'OTHER', inplace=True)

home_ow = pd.get_dummies(df['home_ownership'], prefix='home', drop_first=True)

df = df.merge(home_ow, how='outer', left_index=True, right_index=True)



addr = pd.get_dummies(df['addr_state'], prefix='st', drop_first=True)

df = df.merge(addr, how='outer', left_index=True, right_index=True)



df.drop(['purpose', 'verification_status', 'emp_length', 'home_ownership', 'addr_state'], axis=1, inplace=True)
#Equalizing 1's and  0's

df_ones = df[df['loan_status']==1]

df_zeros = df[df['loan_status']==0]

df_zeros_eq = df_zeros.sample(n=len(df_ones), random_state=40)

df_zeros_test = df_zeros.drop(df_zeros_eq.index)

df_zeros_test = df_zeros_test.sample(4000, random_state=40)

df_eq = pd.concat([df_ones, df_zeros_eq], axis=0)



#Train and Test Split. Ratio: 80/20

df_test = df_eq.sample(frac=0.2, random_state=40)

df_train = df_eq.drop(df_test.index)

#Joining Test data With Zeros that left from equalizing

df_test = pd.concat([df_test, df_zeros_test], axis=0, ignore_index=True)





##############################################################

#To use unbalanced data, comment the three lines above, and uncomment the following

# df_test = df.sample(frac=0.2, random_state=40)

# df_train = df.drop(df_test.index)

###############################################################



#Scaling parameters

train_params_to_st=df_train.iloc[:, 1:15]

test_params_to_st=df_test.iloc[:, 1:15]



train_mean = train_params_to_st.mean()

train_se = train_params_to_st.std()

train_std = (train_params_to_st - train_mean)/train_se

train_std.reset_index(inplace=True)

train_std.drop(['index'], axis=1, inplace=True)



test_std = (test_params_to_st - train_mean)/train_se

test_std.reset_index(inplace=True)

test_std.drop(['index'], axis=1, inplace=True)





#Separating Dummy Params Since They don't need scaling

train_params_dum = df_train.drop(df_train.columns[1:15], axis=1)

train_params_dum.reset_index(inplace=True)

train_params_dum.drop(['index'], axis=1, inplace=True)



test_params_dum = df_test.drop(df_test.columns[1:15], axis=1)

test_params_dum.reset_index(inplace=True)

test_params_dum.drop(['index'], axis=1, inplace=True)





#Obtaining Scaled Train and Test data

train_scaled = pd.concat([train_params_dum, train_std], axis=1)

test_scaled = pd.concat([test_params_dum, test_std], axis=1)



print('Total Data by Category')

print(df['loan_status'].value_counts())

print(df[['loan_status', 'total_pymnt', 'loan_amnt', 'term',

       'int_rate', 'installment', 'annual_inc', 'open_acc', 'last_pymnt_amnt',

       'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'revol_bal',

       'revol_util', 'total_acc']].describe())

print('Test Data by Category')

print(df_test['loan_status'].value_counts())

print(df_test[['loan_status', 'total_pymnt', 'loan_amnt', 'term',

       'int_rate', 'installment', 'annual_inc', 'open_acc', 'last_pymnt_amnt',

       'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'revol_bal',

       'revol_util', 'total_acc']].describe())
#For CLASSIFICATION Training and Testing

X_train_cl = train_scaled.drop(['loan_status', 'total_pymnt'], axis=1).values

y_train_cl = train_scaled['loan_status'].values



X_test_cl = test_scaled.drop(['loan_status', 'total_pymnt'], axis=1).values

y_test_cl = test_scaled['loan_status'].values



'''For REGRESSION Training. To train our regression, we need only defaults as we want to predict amount

paid only by possible defaults (non-defaulters are assumed to pay the full sum anyway).

Our target variable is total_pymnt'''

train_scaled_ones = train_scaled[train_scaled['loan_status']==1]

X_train_reg = train_scaled_ones.drop(['loan_status', 'total_pymnt'], axis=1).values

y_train_reg = train_scaled_ones['total_pymnt'].values





#######################################################################

#To run unscaled data, commment all codes above starting from line 'For CLASSIFICATION Training and Testing'

#and run the following code

##For CLASSIFICATION Training and Testing

# X_train_cl = df_train.drop(['loan_status', 'total_pymnt'], axis=1).values

# y_train_cl = df_train['loan_status'].values



# X_test_cl = df_test.drop(['loan_status', 'total_pymnt'], axis=1).values

# y_test_cl = df_test['loan_status'].values

#########################################################################





thresh=0.5       #probabilities above this threshhold are considered as default
#MLP Classification Training

target = to_categorical(y_train_cl)

n_cols = X_train_cl.shape[1]

model = Sequential()

model.add(Dense(50, activation='relu', input_shape=(n_cols, )))

model.add(BatchNormalization())

model.add(Dense(50, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))

model.add(BatchNormalization())

model.add(Dense(40, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))

model.add(BatchNormalization())

model.add(Dense(40, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))

model.add(BatchNormalization())

model.add(Dense(30, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))

model.add(BatchNormalization())

model.add(Dense(30, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))

model.add(BatchNormalization())

model.add(Dense(20, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))

model.add(BatchNormalization())

model.add(Dense(20, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))

model.add(BatchNormalization())

model.add(Dense(10, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))

model.add(BatchNormalization())

model.add(Dense(10, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))

model.add(BatchNormalization())

model.add(Dense(2, activation='softmax'))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping_monitor = EarlyStopping(patience=5)

model.fit(X_train_cl, target, epochs=50, batch_size=10, callbacks=[early_stopping_monitor])



probs_mlp = model.predict(X_test_cl)[:, 1]

preds_mlp = (probs_mlp > thresh).astype(int)  



print('MLP')

print(classification_report(y_test_cl, preds_mlp))



confusion_df_mlp = pd.DataFrame(confusion_matrix(y_test_cl, preds_mlp),

              columns=["Predicted Class MLP " + str(class_name) for class_name in [0,1]],

              index = ["Class " + str(class_name) for class_name in [0,1]])

print(confusion_df_mlp)
#MLP REG Training

n_cols = X_train_reg.shape[1]

model_reg = Sequential()

model_reg.add(Dense(100, activation='relu', input_shape=(n_cols, )))

model.add(BatchNormalization())

model_reg.add(Dense(100, activation='relu'))

model.add(BatchNormalization())

model_reg.add(Dense(70, activation='relu'))

model.add(BatchNormalization())

model_reg.add(Dense(70, activation='relu'))

model.add(BatchNormalization())

model_reg.add(Dense(50, activation='relu'))

model.add(BatchNormalization())

model_reg.add(Dense(50, activation='relu'))

model.add(BatchNormalization())

model_reg.add(Dense(30, activation='relu'))

model.add(BatchNormalization())

model_reg.add(Dense(30, activation='relu'))

model.add(BatchNormalization())

model_reg.add(Dense(10, activation='relu'))

model.add(BatchNormalization())

model_reg.add(Dense(10, activation='relu'))

model.add(BatchNormalization())

model_reg.add(Dense(1))

model_reg.compile(optimizer='adam', loss='mean_squared_error')

model_reg.fit(X_train_reg, y_train_reg, epochs=50, batch_size=10, callbacks=[early_stopping_monitor])
# #XGB Classifier Tuning

# xgb_cl = xgb.XGBClassifier(random_state=41, n_jobs=-1)

# params_xgb_cl={'n_estimators':[100, 200, 300, 400, 500, 1000, 1500, 2000],

#                'max_depth':[1, 3, 5, 7, 9],

#                'learning_rate':[0.01, 0.05, 0.1, 0.15],

#                'colsample_bytree':[0.1, 0.5, 0.7, 1],

#               'subsample':[0.1, 0.3, 0.5, 0.7, 0.9, 1],

#               'alpha':[0, 0.01, 0.05, 0.1],

#               'lambda':[0.05, 1, 1.5]}

# grid_xgb_cl = RandomizedSearchCV(estimator = xgb_cl, param_distributions=params_xgb_cl, cv=3,

#                        scoring='accuracy', verbose=1, n_jobs=-1, n_iter=3000)

# grid_xgb_cl.fit(X_train_cl, y_train_cl)

# best_hyperparams_cl = grid_xgb_cl.best_params_

# print('Best Classification HyperParams:\n', best_hyperparams_cl)
#XGB Classifier Training

xgb_cl = xgb.XGBClassifier(subsample = 0.9, n_estimators = 500, max_depth = 7,

                           learning_rate = 0.05, reg_lambda = 1.5, colsample_bytree = 1, alpha = 0.1, random_state = 40)

xgb_cl.fit(X_train_cl, np.ravel(y_train_cl))

probs_xgb = xgb_cl.predict_proba(X_test_cl)[:, 1]

preds_xgb = (probs_xgb > thresh).astype(int)  

print('XGBoost')

print(classification_report(y_test_cl, preds_xgb))



confusion_df_xgb = pd.DataFrame(confusion_matrix(y_test_cl, preds_xgb),

              columns=["Predicted Class XGB " + str(class_name) for class_name in [0,1]],

              index = ["Class " + str(class_name) for class_name in [0,1]])

print(confusion_df_xgb)



# #Getting Feature Importances and Their Impact

# model_xgb_cl = xgb_cl.fit(X_train_cl, np.ravel(y_train_cl))

# data_xgb_cl = train_scaled.drop(['loan_status', 'total_pymnt'], axis=1)

# data_xgb_cl.columns = ['pur_credit_card', 'pur_debt_consolidation', 'pur_home_improvement',

#        'pur_house', 'pur_major_purchase', 'pur_medical', 'pur_moving',

#        'pur_other', 'pur_renewable_energy', 'pur_small_business',

#        'pur_vacation', 'verst_Verified', 'emp_l_10+ years', 'emp_l_2 years',

#        'emp_l_3 years', 'emp_l_4 years', 'emp_l_5 years', 'emp_l_6 years',

#        'emp_l_7 years', 'emp_l_8 years', 'emp_l_9 years', 'emp_l_1 year or less',

#        'home_MORTGAGE', 'home_OWN', 'home_RENT', 'st_AL', 'st_AR', 'st_AZ',

#        'st_CA', 'st_CO', 'st_CT', 'st_DC', 'st_DE', 'st_FL', 'st_GA', 'st_HI',

#        'st_ID', 'st_IL', 'st_IN', 'st_KS', 'st_KY', 'st_LA', 'st_MA', 'st_MD',

#        'st_ME', 'st_MI', 'st_MN', 'st_MO', 'st_MS', 'st_MT', 'st_NC', 'st_ND',

#        'st_NE', 'st_NH', 'st_NJ', 'st_NM', 'st_NV', 'st_NY', 'st_OH', 'st_OK',

#        'st_OR', 'st_PA', 'st_RI', 'st_SC', 'st_SD', 'st_TN', 'st_TX', 'st_UT',

#        'st_VA', 'st_VT', 'st_WA', 'st_WI', 'st_WV', 'st_WY', 'loan_amnt',

#        'term', 'int_rate', 'installment', 'annual_inc', 'open_acc',

#        'last_pymnt_amnt', 'delinq_2yrs', 'inq_last_6mths',

#        'mths_since_last_delinq', 'revol_bal', 'revol_util', 'total_acc']

# explainer_xgb_cl = shap.TreeExplainer(model_xgb_cl)

# shap_values = explainer_xgb_cl.shap_values(data_xgb_cl)



# shap.summary_plot(shap_values, data_xgb_cl)

# plt.show()
# #XGB Regression Tuning

# xgb_reg = xgb.XGBRegressor(random_state=41, n_jobs=-1)

# params_xgb_reg={'n_estimators':[100, 200, 300, 400, 500, 1000, 1500, 2000],

#                'max_depth':[1, 3, 5, 7, 9],

#                'learning_rate':[0.01, 0.05, 0.1, 0.15],

#                'colsample_bytree':[0.1, 0.5, 0.7, 1],

#               'subsample':[0.1, 0.3, 0.5, 0.7, 1],

#               'alpha':[0, 0.01, 0.05, 0.1],

#               'lambda':[0.05, 1, 1.5]}

# grid_xgb_reg = RandomizedSearchCV(estimator = xgb_reg, param_distributions = params_xgb_reg, cv=3,

#                        scoring='neg_mean_squared_error', verbose=1, n_jobs=-1, n_iter=3000)

# grid_xgb_reg.fit(X_train_reg, y_train_reg)

# best_hyperparams_reg = grid_xgb_reg.best_params_

# print('Best Classification HyperParams:\n', best_hyperparams_reg)
# XGB Regression Training

xgb_reg = xgb.XGBRegressor(subsample = 0.7, n_estimators = 1500, max_depth = 7, learning_rate = 0.05,

                           reg_lambda = 1.5, colsample_bytree = 0.5, alpha = 0.05)

xgb_reg.fit(X_train_reg, y_train_reg)



# #Getting Feature Importances and Their Impact

# model_xgb_reg = xgb_reg.fit(X_train_reg, np.ravel(y_train_reg))

# data_xgb_reg = train_scaled.drop(['loan_status', 'total_pymnt'], axis=1)

# data_xgb_reg.columns = ['pur_credit_card', 'pur_debt_consolidation', 'pur_home_improvement',

#        'pur_house', 'pur_major_purchase', 'pur_medical', 'pur_moving',

#        'pur_other', 'pur_renewable_energy', 'pur_small_business',

#        'pur_vacation', 'verst_Verified', 'emp_l_10+ years', 'emp_l_2 years',

#        'emp_l_3 years', 'emp_l_4 years', 'emp_l_5 years', 'emp_l_6 years',

#        'emp_l_7 years', 'emp_l_8 years', 'emp_l_9 years', 'emp_l_1 year or less',

#        'home_MORTGAGE', 'home_OWN', 'home_RENT', 'st_AL', 'st_AR', 'st_AZ',

#        'st_CA', 'st_CO', 'st_CT', 'st_DC', 'st_DE', 'st_FL', 'st_GA', 'st_HI',

#        'st_ID', 'st_IL', 'st_IN', 'st_KS', 'st_KY', 'st_LA', 'st_MA', 'st_MD',

#        'st_ME', 'st_MI', 'st_MN', 'st_MO', 'st_MS', 'st_MT', 'st_NC', 'st_ND',

#        'st_NE', 'st_NH', 'st_NJ', 'st_NM', 'st_NV', 'st_NY', 'st_OH', 'st_OK',

#        'st_OR', 'st_PA', 'st_RI', 'st_SC', 'st_SD', 'st_TN', 'st_TX', 'st_UT',

#        'st_VA', 'st_VT', 'st_WA', 'st_WI', 'st_WV', 'st_WY', 'loan_amnt',

#        'term', 'int_rate', 'installment', 'annual_inc', 'open_acc',

#        'last_pymnt_amnt', 'delinq_2yrs', 'inq_last_6mths',

#        'mths_since_last_delinq', 'revol_bal', 'revol_util', 'total_acc']

# explainer_xgb_reg = shap.TreeExplainer(model_xgb_reg)

# shap_values = explainer_xgb_reg.shap_values(data_xgb_reg)



# shap.summary_plot(shap_values, data_xgb_reg)

# plt.show()

# # Random Forests Classification Hyperparameter Tuning

# random_forests_cl = RandomForestClassifier(random_state=41, n_jobs=-1)

# params_rf_cl = {'n_estimators':[100, 200, 300, 400, 500, 1000, 1500, 2000],

#               'max_depth':[1, 3, 5, 7, 9, 11],

#               'max_features':['log2', 'sqrt'],

#               'min_samples_leaf':[0, 0.01, 0.03, 0.05],

#               'min_samples_split':[0, 0.01, 0.03, 0.05]}

# grid_rf_cl = GridSearchCV(estimator=random_forests_cl, param_grid=params_rf_cl, cv=3,

#                        scoring='accuracy', verbose=1, n_jobs=-1)

# grid_rf_cl.fit(X_train_cl, y_train_cl)

# best_hyperparams_cl = grid_rf_cl.best_params_

# print('Best Classification HyperParams:\n', best_hyperparams_cl)
## Random Forests Classification with Tuned Parameters

random_forests_cl = RandomForestClassifier(random_state=41,

                                        n_jobs=-1,

                                        n_estimators=200,

                                        max_depth=9,

                                        max_features='sqrt',

                                        min_samples_leaf=0.01,

                                          min_samples_split = 0.01)

random_forests_cl.fit(X_train_cl, y_train_cl)



probs_rf = random_forests_cl.predict_proba(X_test_cl)[:,1]

preds_rf = (probs_rf > thresh).astype(int)  



print('Random Forests Classifier')

print(classification_report(y_test_cl, preds_rf))



confusion_df_rf = pd.DataFrame(confusion_matrix(y_test_cl, preds_rf),

              columns=["Predicted Class Random Forests " + str(class_name) for class_name in [0,1]],

              index = ["Class " + str(class_name) for class_name in [0,1]])

print(confusion_df_rf)



# #Getting Feature Importances and Their Impact

# model_rf_cl = random_forests_cl.fit(X_train_cl, y_train_cl)

# data_rf_cl = train_scaled.drop(['loan_status', 'total_pymnt'], axis=1)

# explainer_rf_cl = shap.TreeExplainer(model_rf_cl)

# shap_values = explainer_rf_cl.shap_values(data_rf_cl)

# shap.summary_plot(shap_values[1], data_rf_cl)

# plt.show()





# #Choose Arbitrary Tree

# estimator_rf_cl = random_forests_cl.estimators_[50]

# # Export as dot file

# export_graphviz(estimator_rf_cl, out_file='tree.dot', 

#                 feature_names = train_scaled.drop(['loan_status', 'total_pymnt'], axis=1).columns.to_numpy(),

#                 class_names = ['Non-default', 'Default'],

#                 rounded = True, proportion = False, 

#                 precision = 2, filled = True)



# # Convert to png using system command (requires Graphviz)

# from subprocess import call

# call(['dot', '-Tpng', 'tree.dot', '-o', 'tree_rf_cl.png', '-Gdpi=600'])



# # Display in jupyter notebook

# from IPython.display import Image

# Image(filename = 'tree_rf_cl.png')
# #Random Forests Regression Hyperparemeter Tuning

# random_forests_reg = RandomForestRegressor(random_state=41, n_jobs=-1)

# params_rf_reg = {'n_estimators':[100, 200, 300, 400, 500, 1000, 1500, 2000],

#               'max_depth':[1, 3, 5, 7, 9, 11],

#               'max_features':['log2', 'sqrt'],

#               'min_samples_leaf':[0, 0.01, 0.03, 0.05],

#               'min_samples_split':[0, 0.01, 0.03, 0.05]}

# grid_rf_reg = GridSearchCV(estimator=random_forests_reg, param_grid = params_rf_reg, cv=3,

#                        scoring='neg_mean_squared_error', verbose = 1, n_jobs = -1)

# grid_rf_reg.fit(X_train_reg, y_train_reg)

# best_hyperparams_reg = grid_rf_reg.best_params_

# print('Best Regression HyperParams:\n', best_hyperparams_reg)
#Random Forests Regression Training

random_forests_reg = RandomForestRegressor(max_depth=11, max_features='sqrt', min_samples_leaf=0.01,

                                           n_estimators=300, n_jobs=-1, random_state=40, min_samples_split=0.01)

random_forests_reg.fit(X_train_reg, y_train_reg)



# #Getting Feature Importances and Their Impact

# model_rf_reg = random_forests_reg.fit(X_train_reg, y_train_reg)

# data_rf_reg = train_scaled.drop(['loan_status', 'total_pymnt'], axis=1)

# explainer_rf_reg = shap.TreeExplainer(model_rf_reg)

# shap_values = explainer_rf_reg.shap_values(data_rf_reg)

# plt.figure()

# shap.summary_plot(shap_values, data_rf_reg)

# plt.show()



# #Choose Arbitrary Tree

# estimator_rf_reg = random_forests_reg.estimators_[31]

# # Export as dot file

# export_graphviz(estimator_rf_reg, out_file='tree.dot', 

#                 feature_names = train_scaled.drop(['loan_status', 'total_pymnt'], axis=1).columns.to_numpy(),

#                 rounded = True, proportion = False, 

#                 precision = 2, filled = True)

# # Convert to png using system command (requires Graphviz)

# from subprocess import call

# call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



# # Display in jupyter notebook

# from IPython.display import Image

# Image(filename = 'tree.png')

# plt.show()

# #Logit Hyperparameter Tuning

# logit = LogisticRegression(random_state=40, n_jobs=-1)

# params_logit = {'penalty':['l2', 'none'], 'C':[0, 0.5, 1,1.5], 'max_iter':[50, 100, 150, 200]}

# grid_logit = GridSearchCV(estimator=logit, param_grid=params_logit, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)

# grid_logit.fit(X_train_cl, y_train_cl)

# best_params_logit = grid_logit.best_params_

# print('Best Regression HyperParams:\n', best_params_logit)
# Logistic Regression

logit = LogisticRegression(random_state=41, penalty='none', max_iter=50, C=0, n_jobs=-1)

logit.fit(X_train_cl, y_train_cl)



logit_intercept = logit.intercept_

logit_slopes = pd.DataFrame(logit.coef_.T[:, 0], train_scaled.drop(['loan_status', 'total_pymnt'], axis=1).columns)

logit_slopes = logit_slopes.sort_values(by=0, ascending=False)



print('Logistic Regression Slopes')

print(logit_slopes.head(15))

print(logit_slopes.tail(15))

print('Intercept')

print(logit_intercept)



print('Slopes of Continuous Parameters')

print(logit_slopes.loc[['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'open_acc','last_pymnt_amnt',

                   'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'revol_bal', 'revol_util','total_acc']])

plt.figure()

logit_slopes.loc[['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'open_acc','last_pymnt_amnt',

                   'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'revol_bal',

              'revol_util','total_acc']].plot(kind='barh', legend=None)

plt.title('Slopes of Logistic Regression Continuous Parameters')

plt.xlabel('Log Odds Slope')

plt.figure()

logit_slopes.reset_index(inplace=True)

logit_top_params = logit_slopes.drop(list(range(13,77)))

logit_top_params.set_index('index', inplace=True)

logit_top_params.index.name=None

logit_top_params.plot(kind='barh', color='green', legend=None)

plt.xlabel('Log Odds Slope')

plt.title('Slopes of Logistic Regression All Parameters')

plt.show()



probs_logit = logit.predict_proba(X_test_cl)[:,1]

preds_logit = (probs_logit > thresh).astype(int)  



print('Logistic Regression')

print(classification_report(y_test_cl, preds_logit))

confusion_df_logit = pd.DataFrame(confusion_matrix(y_test_cl, preds_logit),

              columns=["Predicted Class Logit " + str(class_name) for class_name in [0,1]],

              index = ["Class " + str(class_name) for class_name in [0,1]])

print(confusion_df_logit)
#OLS Regression Training

ols = LinearRegression(n_jobs=-1)

ols.fit(X_train_reg, y_train_reg)

print('OLS Regression Results')

print('Intercept')

ols_intercept = ols.intercept_

ols_slopes = pd.DataFrame(ols.coef_.T, train_scaled.drop(['loan_status', 'total_pymnt'], axis=1).columns)

ols_slopes = ols_slopes.sort_values(by=0, ascending=False)



print('OLS Slopes')

print(ols_slopes.head(15))

print(ols_slopes.tail(15))

print('Intercept')

print(ols_intercept)



print('Slopes of Continuous Parameters')

print(ols_slopes.loc[['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'open_acc','last_pymnt_amnt',

                   'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'revol_bal', 'revol_util','total_acc']])

plt.figure()

ols_slopes.loc[['loan_amnt', 'term', 'int_rate', 'installment', 'annual_inc', 'open_acc','last_pymnt_amnt',

                   'delinq_2yrs', 'inq_last_6mths', 'mths_since_last_delinq', 'revol_bal',

              'revol_util','total_acc']].plot(kind='barh', legend=None)

plt.title('Slopes of OLS Regression Continuous Parameters')

plt.xlabel('Slopes')

plt.figure()

ols_slopes.reset_index(inplace=True)

ols_top_params = ols_slopes.drop(list(range(13,77)))

ols_top_params.set_index('index', inplace=True)

ols_top_params.index.name=None

ols_top_params.plot(kind='barh', color='green', legend=None)

plt.xlabel('Slopes')

plt.title('Slopes of OLS Regression All Parameters')

plt.show()
#Ensemble

probs_ensemble = (probs_mlp + probs_xgb+probs_rf+probs_logit)/4

preds_ensemble = (probs_ensemble > thresh).astype(int) 

print('Ensemble')

print(classification_report(y_test_cl, preds_ensemble))



#Extra Classification Tests

print('AUC Score')

print('MLP')

print(roc_auc_score(y_test_cl, probs_mlp))

print('XGB')

print(roc_auc_score(y_test_cl, probs_xgb))

print('Random Forests')

print(roc_auc_score(y_test_cl, probs_rf))

print('Logistic Regression')

print(roc_auc_score(y_test_cl, probs_logit))

print('Ensemble')

print(roc_auc_score(y_test_cl, probs_ensemble))

plt.clf()



fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_test_cl, probs_mlp)

fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_test_cl, probs_xgb)

fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test_cl, probs_rf)

fpr_logit, tpr_logit, thresholds_logit = roc_curve(y_test_cl, probs_logit)



plt.clf()

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_mlp, tpr_mlp, label='MLP')

plt.plot(fpr_xgb, tpr_xgb, label='XGB')

plt.plot(fpr_rf, tpr_rf, label='Random Forests')

plt.plot(fpr_logit, tpr_logit, label='Logit')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend()

plt.show()
'''The results show that XGB is the best performer in classification. 

Therefore, based on XGB classification results, 

all four regression models will estimate possible payments by defaulters.

For XGB REGRESSION TESTING we use the same data as classification. Classification predicts defaults.

We get defaulters and for them predict their total payment using regression

For those samples predicted to be default, we do regression to estimate total pymnt'''



#Step 1. Obtaining indices of observations which were predicted as defaults by XGB classificaiton

preds_xgb_reg_index = []

for i in range(len(preds_xgb)):

    if preds_xgb[i] == 1:

        preds_xgb_reg_index.append(i)



#Step 2. Separating loans, which were predicted as 'default' by XGB classifier 

X_test_reg_xgb = test_scaled.drop(['loan_status', 'total_pymnt'], axis=1).values[preds_xgb_reg_index, :]

y_test_reg = df_test[['total_pymnt','term', 'installment', 'loan_status', 'loan_amnt']].reset_index()

y_test_reg = y_test_reg.drop(['index'], axis=1)



#Prepare y_test for regression

y_test_reg_xgb = y_test_reg.iloc[preds_xgb_reg_index, :]

y_test_reg_xgb.reset_index(inplace = True)

y_test_reg_xgb.drop(['index'],axis = 1, inplace = True)

y_test_reg_xgb['max_total_pymnt'] = y_test_reg_xgb.loc[:,'term'] * y_test_reg_xgb.loc[:, 'installment']

y_test_reg_xgb.columns = ['actual_pymnt', 'term', 'installment', 'actual_loan_status', 'loan_amnt', 'max_total_pymnt']



probs_for_reg_xgb = probs_xgb[preds_xgb_reg_index]   #Probability of Default for identified 'defaulters'



#Step 3. Since the data is standardized, we need to destandardize it

mean_pymnt = df_train['total_pymnt'].mean()

se_pymnt = df_train['total_pymnt'].std()



#Regression may predict negive cash flows. In this case cash flow is set to 0

def positive(row):

    if row <=0:

        return 0

    else:

        return row



#Regression may predict values exceeding maximum payment (interest+principal), thus cap is imposed

def cap(raw):

    if raw['predicted_pymnt'] > raw['max_total_pymnt']:

        return raw['max_total_pymnt']

    else:

        return raw['predicted_pymnt']



#MLP REG TESTING

pred_reg_scaled_mlp = model.predict(X_test_reg_xgb)[:,0]

pred_reg_mlp = pred_reg_scaled_mlp *se_pymnt + mean_pymnt

df_mlp = pd.DataFrame({'predicted_pymnt':pred_reg_mlp})

df_mlp['predicted_pymnt'] = df_mlp['predicted_pymnt'].map(positive)



#Collecting predictions to one dataframe called df_reg

df_reg = pd.concat([y_test_reg_xgb, df_mlp], axis=1)

df_reg['prob_default'] = probs_for_reg_xgb

df_reg['predicted_pymnt'] = df_reg.apply(cap, axis=1)

df_reg.rename(columns={'predicted_pymnt':'MLP_predicted_pymnt'}, inplace=True)



#XGB REG TESTING

pred_reg_scaled_xgb = xgb_reg.predict(X_test_reg_xgb)

pred_reg_xgb = pred_reg_scaled_xgb *se_pymnt + mean_pymnt  

df_xgb = pd.DataFrame({'predicted_pymnt':pred_reg_xgb})

df_xgb['predicted_pymnt'] = df_xgb['predicted_pymnt'].map(positive)



df_reg = pd.concat([df_reg, df_xgb], axis=1)

df_reg['predicted_pymnt'] = df_reg.apply(cap, axis=1)

df_reg.rename(columns={'predicted_pymnt':'XGB_predicted_pymnt'}, inplace=True)





#Random Forests REG TESTING

pred_reg_scaled_rf = random_forests_reg.predict(X_test_reg_xgb)

pred_reg_rf = pred_reg_scaled_rf *se_pymnt + mean_pymnt

df_rf = pd.DataFrame({'predicted_pymnt':pred_reg_rf})

df_rf['predicted_pymnt'] = df_rf['predicted_pymnt'].map(positive)



df_reg = pd.concat([df_reg, df_rf], axis=1)

df_reg['predicted_pymnt'] = df_reg.apply(cap, axis=1)

df_reg.rename(columns={'predicted_pymnt':'RF_predicted_pymnt'}, inplace=True)



#Linear Regression TESTING

pred_reg_scaled_ols = ols.predict(X_test_reg_xgb)

pred_reg_ols = pred_reg_scaled_ols *se_pymnt + mean_pymnt  

df_ols = pd.DataFrame({'predicted_pymnt':pred_reg_ols})

df_ols['predicted_pymnt'] = df_ols['predicted_pymnt'].map(positive)



df_reg = pd.concat([df_reg, df_ols], axis=1)

df_reg['predicted_pymnt'] = df_reg.apply(cap, axis=1)

df_reg.rename(columns={'predicted_pymnt':'OLS_predicted_pymnt'}, inplace=True)



df_reg = df_reg[['loan_amnt', 'actual_pymnt', 'MLP_predicted_pymnt', 'XGB_predicted_pymnt', 'RF_predicted_pymnt',

                         'OLS_predicted_pymnt', 'prob_default', 'term', 'installment', 'actual_loan_status',

                 'max_total_pymnt']]

print('Predicted Payment')

print(df_reg[['MLP_predicted_pymnt', 'XGB_predicted_pymnt', 'RF_predicted_pymnt',

                         'OLS_predicted_pymnt']].sum())





df_reg['loss_OLS'] = df_reg['max_total_pymnt'] - df_reg['OLS_predicted_pymnt']

df_reg['loss_RF'] = df_reg['max_total_pymnt'] - df_reg['RF_predicted_pymnt']

df_reg['loss_XGB'] = df_reg['max_total_pymnt'] - df_reg['XGB_predicted_pymnt']

df_reg['loss_MLP'] = df_reg['max_total_pymnt'] - df_reg['MLP_predicted_pymnt']

print('Expected Loss Due to Prepayment and Default')

print(df_reg[['loss_OLS', 'loss_RF', 'loss_XGB','loss_MLP']].sum())



#Actual Payment, Loss Calculations. They will be used as benchmark for regression

reg_test_data = df_test[df_test['loan_status'] == 1]

reg_test_data = reg_test_data[['total_pymnt', 'loan_amnt', 'term', 'installment']]

reg_test_data['max_payment'] = reg_test_data['term'] * reg_test_data['installment']

reg_test_data['loss'] = reg_test_data['max_payment'] - reg_test_data['total_pymnt']

print('Actual Figures on Loan Repayment and Loss for Defaul Loans')

print(reg_test_data[['total_pymnt', 'max_payment', 'loss']].sum())
#Comparing Regression Performance Based on Mean Squared Error

print('MSE Estimates')

print('MLP')

print(mean_squared_error(df_reg['actual_pymnt'].values, df_reg['MLP_predicted_pymnt'].values))

print('XGB')

print(mean_squared_error(df_reg['actual_pymnt'].values, df_reg['XGB_predicted_pymnt'].values))

print('Random Forests')

print(mean_squared_error(df_reg['actual_pymnt'].values, df_reg['RF_predicted_pymnt'].values))

print('OLS')

print(mean_squared_error(df_reg['actual_pymnt'].values, df_reg['OLS_predicted_pymnt'].values))
# Plotting

# Use to plot by color based on actual default or non-default

fig, ax = plt.subplots()

groups = df_reg.groupby('actual_loan_status')



#MLP

for name, group in groups:

    ax.plot(group.actual_pymnt, group.MLP_predicted_pymnt, marker='o', linestyle='', ms=12, label=name, alpha=0.5)

ax.legend(numpoints=1, loc='upper left')

plt.xlim(0,30000)

plt.ylim(0,30000)

plt.ylabel('MLP_Predicted Payment')

plt.xlabel('Actual Payment')

plt.gca().set_aspect('equal', adjustable='box')



#XGB

fig, ax = plt.subplots()

for name, group in groups:

    ax.plot(group.actual_pymnt, group.XGB_predicted_pymnt, marker='o', linestyle='', ms=12, label=name, alpha=0.5)

ax.legend(numpoints=1, loc='upper left')

plt.xlim(0,30000)

plt.ylim(0,30000)

plt.ylabel('XGB_Predicted Payment')

plt.xlabel('Actual Payment')

plt.gca().set_aspect('equal', adjustable='box')



#Random Forests

fig, ax = plt.subplots()

for name, group in groups:

    ax.plot(group.actual_pymnt, group.RF_predicted_pymnt, marker='o', linestyle='', ms=12, label=name, alpha=0.5)

ax.legend(numpoints=1, loc='upper left')

plt.xlim(0,30000)

plt.ylim(0,30000)

plt.ylabel('RF_Predicted Payment')

plt.xlabel('Actual Payment')

plt.gca().set_aspect('equal', adjustable='box')



#OLS

fig, ax = plt.subplots()

for name, group in groups:

    ax.plot(group.actual_pymnt, group.OLS_predicted_pymnt, marker='o', linestyle='', ms=12, label=name, alpha=0.5)

ax.legend(numpoints=1, loc='upper left')

plt.xlim(0,30000)

plt.ylim(0,30000)

plt.ylabel('OLS_Predicted Payment')

plt.xlabel('Actual Payment')

plt.gca().set_aspect('equal', adjustable='box')

plt.show()
#Estimating Profit and Loss for Differernt Values of Threshhold

loss = []

revenue = []

for threshold in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]: 

    preds_xgb = (probs_xgb > threshold).astype(int)

    #Step 1. Obtaining indices of observations which were predicted as defaults by XGB classificaiton

    preds_xgb_reg_index = []

    for i in range(len(preds_xgb)):

        if preds_xgb[i] == 1:

            preds_xgb_reg_index.append(i)

    #Step 2. Separating loans, which were predicted as 'default' by XGB classifier 

    X_test_reg_xgb = test_scaled.drop(['loan_status', 'total_pymnt'], axis=1).values[preds_xgb_reg_index, :]

    y_test_reg = df_test[['total_pymnt','term', 'installment', 'loan_status']].reset_index()

    y_test_reg = y_test_reg.drop(['index'], axis=1)

    

    #Prepare y_test for regression

    y_test_reg_xgb = y_test_reg.iloc[preds_xgb_reg_index, :]

    y_test_reg_xgb.reset_index(inplace = True)

    y_test_reg_xgb.drop(['index'],axis = 1, inplace = True)

    y_test_reg_xgb['max_total_pymnt'] = y_test_reg_xgb.loc[:,'term'] * y_test_reg_xgb.loc[:, 'installment']

    y_test_reg_xgb.columns = ['actual_pymnt', 'term', 'installment', 'actual_loan_status', 'max_total_pymnt']

   

    

    #REG TESTING

    pred_reg_scaled_xgb = xgb_reg.predict(X_test_reg_xgb)

    pred_reg_xgb = pred_reg_scaled_xgb *se_pymnt + mean_pymnt

    df_xgb = pd.DataFrame({'predicted_pymnt':pred_reg_xgb})

    df_xgb['predicted_pymnt'] = df_xgb['predicted_pymnt'].map(positive)

    #Collecting predictions to one dataframe called df_reg

    df_reg = pd.concat([y_test_reg_xgb, df_xgb], axis=1)

    df_reg['loss'] = df_reg['max_total_pymnt'] - df_reg['predicted_pymnt']

    loss.append(df_reg['loss'].sum())

    revenue.append(df_reg['predicted_pymnt'].sum())



plt.plot([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], loss, label='Expected Loss')

plt.plot([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], revenue, label = 'Expected Total Repayment')

plt.ylabel('USD')

plt.xlabel('Threshold for Categorizing as Default')

plt.legend()

plt.show()
