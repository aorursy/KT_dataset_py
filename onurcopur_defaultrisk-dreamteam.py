import csv

import os



import pandas as pd

import numpy as np



pd.set_option('display.max_columns', None)



# Plotting



from matplotlib import pyplot

import seaborn as sns 

import plotly

import plotly.offline as py

from plotly.offline import iplot

import plotly.graph_objs as go

import cufflinks as cf



py.init_notebook_mode(connected=True)

cf.go_offline()



# Data Preprocessing, Models, Feature and Model Selection



from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score

from sklearn.model_selection import train_test_split, KFold



import lightgbm as lgb
# Load Necessary Datasets



application_train = pd.read_csv('../input/home-credit-default-risk/application_train.csv')

application_test = pd.read_csv('../input/home-credit-default-risk/application_test.csv')

previous_applications = pd.read_csv("../input/home-credit-default-risk/previous_application.csv")

bureau = pd.read_csv("../input/home-credit-default-risk/bureau.csv")



# Data Cleaning



# Convert DAYS_BIRTH to age in years of each client (it's expressed in negative days)



application_train["DAYS_BIRTH"] = application_train["DAYS_BIRTH"]/(-365)
def missing_data(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    missing = pd.concat([total, percent], axis=1)

    missing.rename(columns= {0:'Total', 1:'Percent'}, inplace = True)

    return missing



def plot_iploty_stats(application_train, feature):

    temp = application_train[feature].value_counts()

    df1 = pd.DataFrame({feature: temp.index,'Number of contracts': temp.values})

    

    # Calculate the percentage of target=1 per category value

    

    cat_perc = application_train[[feature, 'TARGET']].groupby([feature],as_index=False).mean()

    cat_perc.sort_values(by='TARGET', ascending=False, inplace=True)

    

    trace = go.Bar(

        x = temp.index,

        y = temp / temp.sum()*100)

    data = [trace]

    layout = go.Layout(

        title = 'Percentage of contracts according to '+feature,

        xaxis=dict(

            title='Values',

            tickfont=dict(

                size=14,

                color='rgb(107, 107, 107)'

            )

        ),

        yaxis=dict(

            title='Percentage of contracts',

            titlefont=dict(

                size=16,

                color='rgb(107, 107, 107)'

            ),

            tickfont=dict(

                size=14,

                color='rgb(107, 107, 107)'

            )

            

        )

    )

    fig = go.Figure(data=data, layout=layout)

    fig.update_yaxes(range=[0, 100])

    

    py.iplot(fig, filename='statistics')

    

    trace = go.Bar(

        x = cat_perc[feature],

        y = cat_perc.TARGET

    )

    data = [trace]

    layout = go.Layout(

        title = 'Percent of contracts with TARGET==1 according to '+feature,

        xaxis=dict(

            title='Values',

            tickfont=dict(

                size=14,

                color='rgb(107, 107, 107)'

            )

        ),

        yaxis=dict(

            title='Percent of target with value 1',

            titlefont=dict(

                size=16,

                color='rgb(107, 107, 107)'

            ),

            tickfont=dict(

                size=14,

                color='rgb(107, 107, 107)'

            )

        )

    )

    fig = go.Figure(data=data, layout=layout)

    

    py.iplot(fig, filename='schoolStateNames')

    

def plot_repayed_perc(application_train, feature, round_feat=-1):

    #percentage of the loans repayed or not according to the feature chosen in input 

    if round_feat > -1:

        application_train[feature] = np.round(application_train[feature], round_feat)

    temp = application_train[feature].value_counts()

    

    temp_y0 = []

    temp_y1 = []

    for val in temp.index:

        temp_y1.append(np.sum(application_train["TARGET"][application_train[feature]==val] == 1))

        temp_y0.append(np.sum(application_train["TARGET"][application_train[feature]==val] == 0))    

    trace1 = go.Bar(

        x = temp.index,

        y = (temp_y1 / temp.sum()) * 100,

        name='YES'

    )

    trace2 = go.Bar(

        x = temp.index,

        y = (temp_y0 / temp.sum()) * 100, 

        name='NO'

    )



    data = [trace1, trace2]

    fig = go.Figure(data=data)

    fig.update_layout(showlegend=True, title = 'Loan Defaults Decomposed by ' + feature + ' (Percentage)')



    iplot(fig)

    

def heatmap_coor_matrix(application_train, corr_pearson):

    data = [go.Heatmap(

        z= corr_pearson,

        x=application_train.columns.values,

        y=application_train.columns.values,

        colorscale='Viridis',

        reversescale = False,

        opacity = 1.0 )

       ]

    fig = go.Figure(data=data)

    fig.update_layout(title = 'Pearson Correlation between features', 

                      xaxis = dict(ticks='', nticks=36),

                      yaxis = dict(ticks='' ),

                      width = 900, height = 900,

                      margin=dict(l=240)) 

    py.iplot(fig, filename='coorelation_heatmap')
application_train["DAYS_BIRTH"].head()
application_train['NAME_TYPE_SUITE'].unique()
application_train['NAME_TYPE_SUITE'].isna().sum()
application_train['CODE_GENDER'].value_counts()
temp = application_train["TARGET"].value_counts()

temp = (temp / temp.sum())*100

temp.iplot(kind='bar', labels='labels', values='values', colors ='green', title='Loan Repayed or Not')
temp = application_train["TARGET"].value_counts()

df = pd.DataFrame({'labels': temp.index,

                   'values': temp.values})



df.iplot(kind='pie', labels='labels', values='values', title='Loan Repayed or not')
plot_iploty_stats(application_train, 'OCCUPATION_TYPE')
plot_iploty_stats(application_train, 'CODE_GENDER')
# Tabulated Default Rates Decomposed by Gender



np.round(pd.crosstab(application_train.CODE_GENDER, application_train.TARGET, margins=True, normalize=0), 3)
plot_iploty_stats(application_train, 'NAME_FAMILY_STATUS')
plot_iploty_stats(application_train, "ORGANIZATION_TYPE")
plot_repayed_perc(application_train, "NAME_INCOME_TYPE")
plot_iploty_stats(application_train, "NAME_CONTRACT_TYPE")
plot_repayed_perc(application_train, "DAYS_BIRTH", round_feat=0)
previous_applications['NAME_CASH_LOAN_PURPOSE'].value_counts()
temp = previous_applications['NAME_CASH_LOAN_PURPOSE'].value_counts()



temp.iplot(kind='bar', color="blue", 

           xTitle = 'Organization Name', yTitle = "Count", 

           title = 'Types of NAME_CASH_LOAN_PURPOSE in previous applications ')
missing_data(application_train)
missing_data(bureau)
missing_data(previous_applications)
# Spin out categorical variables into dummied 0/1 indicator variables



train_one_hot = pd.get_dummies(application_train)
# Check what our data looks like now



train_one_hot.head()
# Check the dimensions after dummying out factors



train_one_hot.shape
corr_pearson = train_one_hot.corr().values
heatmap_coor_matrix(application_train, corr_pearson)    
# Absolute value correlation matrix



corr_matrix = train_one_hot.corr().abs()
corr_matrix
# where: Replace values where the condition is False. 

# triu: Upper triangle of an array. Return a copy of a matrix with the elements below the k-th diagonal zeroed.



corr_upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(bool))
corr_upper
# Select columns with correlations above threshold



threshold = 0.9

drop = []

for column in corr_upper.columns:

    if any(corr_upper[column]>threshold):

        drop.append(column)





print("columns to drop " + str(len(drop)))
# Some of the features we re going to drop



drop[:10]
train = train_one_hot.drop(columns = drop)
train
threshold = 0.55

train_missing = (train.isnull().sum() / len(train)).sort_values(ascending = False)
train_missing = train_missing.index[train_missing > threshold]



print("we are going to drop " + str(len(train_missing))+" columns")
train.drop(columns = train_missing, inplace = True)
train.head()
#Save and Drop ['SK_ID_CURR'] column because the id is just number and it shouldn't have a predictive power



ID = train["SK_ID_CURR"] # the ids



train_clean = train.drop(columns = ['SK_ID_CURR'] )





# Save the "TARGET" column and drop it



train_target = train['TARGET'] # the target list

train_clean.drop(columns = ['TARGET'], inplace = True)
train_clean.head()
print('we started from ', application_train.shape)

print('With just one-hot encoding we had ', train_one_hot.shape)

print('after feature selection we now have ', train_clean.shape)
from sklearn.preprocessing import StandardScaler
#One-hot encoding on the test set 

test_one_hot = pd.get_dummies(application_test)



#and remove irrelevant features from the test set



relevant_features = list(train_clean.columns)



to_drop = [col for col in test_one_hot.columns if col not in relevant_features]

test = test_one_hot.drop(columns = to_drop)



print('we started from ', application_test.shape)

print('With just one-hot encoding we had ', test_one_hot.shape)

print('after feature selection we now have ', test.shape)

#there are 3 columns in train that are not present in test, remove them from train



for col in train_clean.columns:

    if col not in test.columns.tolist():

        train_clean.drop(columns = [col], inplace = True)



print('now we have ', train_clean.shape)
sc = StandardScaler()

train = sc.fit_transform(train_clean)



test  = sc.fit_transform(test)
# the set of parameters for Light GBM

params = {}

params['learning_rate'] = 0.003

params['boosting_type'] = 'gbdt'

params['objective'] = 'binary'

params['metric'] = 'auc'

params['sub_feature'] = 0.5

params['num_leaves'] = 10

params['min_data'] = 50

params['max_depth'] = 10
# Split the train dataset in test and train



x_train, x_test, y_train, y_test = train_test_split(train ,train_target , test_size=0.4, random_state=18)
# Create the LightGBM data containers



train_data = lgb.Dataset(x_train, label=y_train)



test_data = lgb.Dataset(x_test, label=y_test)
# Train the model



model = lgb.train(params,

                  train_data,

                  valid_sets=test_data,

                  num_boost_round=5000,

                  early_stopping_rounds=100, 

                  verbose_eval=False)
# Predict on train 



pred_train = model.predict(train)
# Accuracy 



y_target = np.array(train_target)



y_predictions = np.array(pred_train)

auc = roc_auc_score(y_target, y_predictions)

auc
# Calculate ROC curves



lgbm_fpr, lgbm_tpr, _ = roc_curve(y_target, pred_train)



# Plot the roc curve for the model



pyplot.plot(lgbm_fpr, lgbm_tpr, marker='.', label='LightGBM')



# Axis labels



pyplot.xlabel('False Positive Rate')

pyplot.ylabel('True Positive Rate')



# Show the legend



pyplot.legend()



# Show the plot



pyplot.show()
# Precict on the test set to make a submission



preds = model.predict(test)



#save the prediction into a csv file

submissions = pd.DataFrame()

submissions['SK_ID_CURR'] = application_test['SK_ID_CURR']

submissions['TARGET'] = preds

submissions.to_csv("predictions.csv", index=False)
def numeric_aggregation(table, key, name ):

    for col in table:

        if col != key and "SK_ID" in col:

            table = table.drop(columns=col)

    numeric_table = table.select_dtypes("number")

    numeric_table[key] = table[key]

    agg = numeric_table.groupby(key).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()



    columns = [key]



    for var in agg.columns.levels[0]:

        if var != key:

            for stat in agg.columns.levels[1][:-1]:

                columns.append('%s_%s_%s' % (name, var, stat))

    agg.columns = columns

    return agg



def correlation_func(table):

    corrs = []



    for col in table.columns:

        if col != "TARGET":

            corr = table['TARGET'].corr(table[col])

            corrs.append((col, corr))



    corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)

    return corrs



def categorical_aggregation(table, key, name ):

    try:

        categoricals = pd.get_dummies(table.select_dtypes("object"))

        categoricals[key] = table[key]

    except ValueError:

        return None



    agg = categoricals.groupby(key).agg(["sum", "mean"])



    columns = []



    for var in agg.columns.levels[0]:

        for stat in ["count", "count_norm"]:

            columns.append('%s_%s_%s' % (name, var, stat))

    agg.columns = columns

    return agg
bureau = pd.read_csv("../input/home-credit-default-risk/bureau.csv")

bureau_balance = pd.read_csv("../input/home-credit-default-risk/bureau_balance.csv")

train_data = pd.read_csv("../input/home-credit-default-risk/application_train.csv")

test_data = pd.read_csv("../input/home-credit-default-risk/application_test.csv")

previous_application = pd.read_csv("../input/home-credit-default-risk/previous_application.csv")

POS_CASH_balance = pd.read_csv("../input/home-credit-default-risk/POS_CASH_balance.csv")

installments_payments = pd.read_csv("../input/home-credit-default-risk/installments_payments.csv")

credit_card_balance = pd.read_csv("../input/home-credit-default-risk/credit_card_balance.csv")
bureau_num_agg = numeric_aggregation(bureau, key="SK_ID_CURR", name="bureau")

bureau_categorical_agg = categorical_aggregation(bureau, key="SK_ID_CURR", name="bureau")
previous_application_num_agg = numeric_aggregation(previous_application, key="SK_ID_CURR", name="previous_application")

previous_application_categorical_agg = categorical_aggregation(previous_application, key="SK_ID_CURR", name="previous_application")

del previous_application
POS_CASH_balance_num_agg = numeric_aggregation(POS_CASH_balance, key="SK_ID_CURR", name="POS_CASH_balance")

POS_CASH_balance_categorical_agg = categorical_aggregation(POS_CASH_balance, key="SK_ID_CURR", name="POS_CASH_balance")

del POS_CASH_balance
installments_payments_num_agg = numeric_aggregation(installments_payments, key="SK_ID_CURR", name="installments_payments")

del installments_payments
credit_card_balance_num_agg = numeric_aggregation(credit_card_balance, key="SK_ID_CURR", name="credit_card_balance")

credit_card_balance_categorical_agg = categorical_aggregation(credit_card_balance, key="SK_ID_CURR", name="credit_card_balance")

del credit_card_balance
bureau_balance_num_agg = numeric_aggregation(bureau_balance, key="SK_ID_BUREAU", name="bureau_balance")

bureau_balance_categorical_agg = categorical_aggregation(bureau_balance, key="SK_ID_BUREAU", name="bureau_balance")

del bureau_balance
bureau_by_loan = bureau_balance_num_agg.merge(bureau_balance_categorical_agg, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')

bureau_by_loan = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_by_loan, on = 'SK_ID_BUREAU', how = 'left')

bureau_balance_by_client = numeric_aggregation(bureau_by_loan.drop(columns=['SK_ID_BUREAU']), key='SK_ID_CURR', name='client')

del bureau

del bureau_by_loan
train_data = train_data.merge(bureau_num_agg, on = 'SK_ID_CURR', how = 'left')

train_data = train_data.merge(bureau_categorical_agg, on = 'SK_ID_CURR', how = 'left')

train_data = train_data.merge(previous_application_num_agg, on = 'SK_ID_CURR', how = 'left')

train_data = train_data.merge(previous_application_categorical_agg, on = 'SK_ID_CURR', how = 'left')

train_data = train_data.merge(POS_CASH_balance_num_agg, on = 'SK_ID_CURR', how = 'left')

train_data = train_data.merge(POS_CASH_balance_categorical_agg, on = 'SK_ID_CURR', how = 'left')

train_data = train_data.merge(installments_payments_num_agg, on = 'SK_ID_CURR', how = 'left')

train_data = train_data.merge(credit_card_balance_num_agg, on = 'SK_ID_CURR', how = 'left')

train_data = train_data.merge(credit_card_balance_categorical_agg, on = 'SK_ID_CURR', how = 'left')

train_data = train_data.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')
test_data = test_data.merge(bureau_num_agg, on = 'SK_ID_CURR', how = 'left')

del bureau_balance_num_agg

test_data = test_data.merge(bureau_categorical_agg, on = 'SK_ID_CURR', how = 'left')

del bureau_categorical_agg

test_data = test_data.merge(previous_application_num_agg, on = 'SK_ID_CURR', how = 'left')

del previous_application_num_agg

test_data = test_data.merge(previous_application_categorical_agg, on = 'SK_ID_CURR', how = 'left')

del previous_application_categorical_agg

test_data = test_data.merge(POS_CASH_balance_num_agg, on = 'SK_ID_CURR', how = 'left')

del POS_CASH_balance_num_agg

test_data = test_data.merge(POS_CASH_balance_categorical_agg, on = 'SK_ID_CURR', how = 'left')

del POS_CASH_balance_categorical_agg

test_data = test_data.merge(installments_payments_num_agg, on = 'SK_ID_CURR', how = 'left')

del installments_payments_num_agg

test_data = test_data.merge(credit_card_balance_num_agg, on = 'SK_ID_CURR', how = 'left')

del credit_card_balance_num_agg

test_data = test_data.merge(credit_card_balance_categorical_agg, on = 'SK_ID_CURR', how = 'left')

del credit_card_balance_categorical_agg

test_data = test_data.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')

del bureau_balance_by_client
train_labels = train_data["TARGET"]

train_data, test_data = train_data.align(test_data, join="inner", axis=1)

train_data["TARGET"] = train_labels

print(train_data.shape)

print(test_data.shape)



# handling the missing values

mis_val_count = train_data.isnull().sum()

percentage = mis_val_count/len(train_data)



# drop the columns with missing values higher than a threshold

columns = percentage[percentage < 0.4].index

train_data = train_data[columns]

test_data = test_data[columns[:-1]] #drop the target column
corrs = train_data.corr()



# Set the threshold

threshold = 0.8



# Empty dictionary to hold correlated variables

above_threshold_vars = {}





# For each column, record the variables that are above the threshold

for col in corrs:

    above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])



cols_to_remove = []

cols_seen = []

cols_to_remove_pair = []





# Iterate through columns and correlated columns

for key, value in above_threshold_vars.items():

    # Keep track of columns already examined

    cols_seen.append(key)

    for x in value:

        if x == key:

            next

        else:

            # Remove one of the columns

            if x not in cols_seen:

                cols_to_remove.append(x)

                cols_to_remove_pair.append(key)



cols_to_remove = list(set(cols_to_remove))

print('Number of columns to remove: ', len(cols_to_remove))



train_corrs_removed = train_data.drop(columns = cols_to_remove)

test_corrs_removed = test_data.drop(columns = cols_to_remove)
train_corrs_removed.to_csv('extended_train.csv', index = False)

test_corrs_removed.to_csv('extended_test.csv', index = False)
del train_corrs_removed, test_corrs_removed
extended_train = pd.read_csv("extended_train.csv")

extended_test = pd.read_csv("extended_test.csv")
print(extended_test.shape)

print(extended_train.shape)
x_train = pd.get_dummies(extended_train.iloc[:,:-1])

y_train = extended_train.iloc[:,-1]

x_test = pd.get_dummies(extended_test)

del extended_train

del extended_test
drop_list = []

for column in x_train.columns:

    if column in x_test.columns:

        continue

    else:

        drop_list.append(column)
x_train = x_train.drop(columns = drop_list)

print(x_train.shape)

print(x_test.shape)
imp_median = SimpleImputer(missing_values=np.nan, strategy='median')

imp_median.fit(x_train)

x_train = imp_median.transform(x_train)

imp_median.fit(x_test)

x_test = imp_median.transform(x_test)
sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.fit_transform(x_test)
kfold = KFold(n_splits = 5, shuffle = True, random_state = 50)



inputs=x_train[:,1:]

outputs = y_train

x_test = x_test[:,1:]



# Empty array for test predictions

y_pred = np.zeros(x_test.shape[0])

    

for train, test in kfold.split(inputs, outputs):

    

    model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 

                           class_weight = 'balanced', learning_rate = 0.05, 

                           reg_alpha = 0.1, reg_lambda = 0.1, 

                           subsample = 0.8, n_jobs = -1, random_state = 50)

    # Train the model

    model.fit(inputs[train, :], outputs[train], eval_metric = 'auc',

              eval_set = [(inputs[test, :], outputs[test]), (inputs[train, :], outputs[train])],

                  eval_names = ['valid', 'train'],

              early_stopping_rounds = 100, verbose = 200)

    # Record the best iteration

    best_iteration = model.best_iteration_

    print(1)

    # Make predictions

    y_pred += model.predict_proba(x_test, num_iteration = best_iteration)[:, 1] / kfold.n_splits
submission = pd.read_csv("../input/home-credit-default-risk/sample_submission.csv")
submission["TARGET"] = y_pred
submission.to_csv('submission.csv', index = False)