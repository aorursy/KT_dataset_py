import numpy as np # Linear algebra



# Plotting libraries.

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import unicodedata

import re



# Machine learning algorithms.

from collections import Counter



import lightgbm as lgb



import xgboost as xgb

from sklearn import metrics

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 10
# Read train and test dataset



train_df = pd.read_excel("../input/Final_Train.xlsx")

test_df = pd.read_excel("../input/Final_Test.xlsx")

df_test = test_df.copy()
train_df.head()
test_df.head()
# Check shape of dataset



train_df.shape, test_df.shape
# check train column types



ctype = train_df.dtypes.reset_index()

ctype.columns = ["Count", "Column Type"]

ctype.groupby("Column Type").aggregate('count').reset_index()
# check test column types



ctype = test_df.dtypes.reset_index()

ctype.columns = ["Count", "Column Type"]

ctype.groupby("Column Type").aggregate('count').reset_index()
# Check the Maximum and Minimum number of qualifications



# Train set

dat_train = train_df.Qualification.apply(lambda x: len(x.split(',')))

print("Maximum qualifications of a doctor in the Train dataset is {}\n".format(dat_train.max()))

print("And the qualifications is --> {}\n\n".format(train_df.Qualification[dat_train.idxmax()]))

print("Minimum qualification of a doctor in the Train dataset is {}\n".format(dat_train.min()))

print("And the qualifications is --> {}\n\n".format(train_df.Qualification[dat_train.idxmin()]))



# Test set

dat_test = test_df.Qualification.apply(lambda x: len(x.split(',')))

print("Maximum qualifications of a doctor in the Test dataset is {}\n".format(dat_test.max()))

print("And the qualifications is --> {}\n\n".format(test_df.Qualification[dat_test.idxmax()]))

print("Minimum qualification of a doctor in the Test dataset is {}\n".format(dat_test.min()))

print("And the qualifications is --> {}".format(test_df.Qualification[dat_test.idxmin()]))
sorted(test_df.Qualification[test_df.Qualification.apply(lambda x: len(x.split(','))).idxmax()].split(","))
# Define function to remove inconsistencies in the data

def sortQual(text):

    arr = re.sub(r'\([^()]+\)', lambda x: x.group().replace(",","-"), text) # to replace ',' with '-' inside brackets only

    return ','.join(sorted(arr.lower().replace(" ","").split(",")))
# Apply the function on the Qualification set



# Train Set

train_df.Qualification = train_df.Qualification.apply(lambda x: sortQual(x))



# Test Set

test_df.Qualification = test_df.Qualification.apply(lambda x: sortQual(x))
# Define a function to create a doc of all Qualifications seprataed by ','



def doc(series):

    Quals = ''

    for i in series:

        Quals += i + ','

    return Quals
# List of top 10 unique Qualifications along with there occurence in Train Set



text = doc(train_df.Qualification)

df = pd.DataFrame.from_dict(dict(Counter(text.split(',')).most_common()), orient='index').reset_index()

df.columns=['Qualification','Count']

df.head(10)
# List of top 10 unique Qualifications along with there occurence in Test Set



text = doc(test_df.Qualification)

df = pd.DataFrame.from_dict(dict(Counter(text.split(',')).most_common()), orient='index').reset_index()

df.columns=['Qualification','Count']

df.head(10)
text = doc(test_df.Qualification)

df = pd.DataFrame.from_dict(dict(Counter(text.split(',')).most_common()), orient='index').reset_index()

df.columns=['Qualification','Count']

df['code'] = df.Qualification.astype('category').cat.codes

df.head(10)
qual_dict = dict(zip(df.Qualification, df.code))
def qual_col(dataframe, col, col_num):

    return dataframe[col].str.split(',').str[col_num]
# for training set

for i in range(0,dat_train.max()):

    qual = "Qual_"+ str(i+1)

    train_df[qual] = qual_col(train_df,'Qualification', i)



    

# for test set

for i in range(0,dat_test.max()):

    qual = "Qual_"+ str(i+1)

    test_df[qual] = qual_col(test_df,'Qualification', i)

train_df.head()
# Train set

train_df['years_exp'] = train_df['Experience'].str.slice(stop=2).astype(int)



# Test set

test_df['years_exp'] = test_df['Experience'].str.slice(stop=2).astype(int)
train_df.head()
# Train set

train_df['Rating'].fillna('0%',inplace = True)

train_df['Rating'] = train_df['Rating'].str.slice(stop=-1).astype(int)



# Test set

test_df['Rating'].fillna('0%',inplace = True)

test_df['Rating'] = test_df['Rating'].str.slice(stop=-1).astype(int)
train_df.head()
# Train Set

train_df['City'] = train_df['Place'].str.split(',').str[1]

train_df['Locality'] = train_df['Place'].str.split(',').str[0]





# Test Set

test_df['City'] = test_df['Place'].str.split(',').str[1]

test_df['Locality'] = test_df['Place'].str.split(',').str[0]
train_df.head()
list(train_df.Miscellaneous_Info[0:10])
# Train set

train_df.Miscellaneous_Info = train_df.Miscellaneous_Info.str.replace(unicodedata.lookup('Indian Rupee Sign'), 'INR ')



# Test set

test_df.Miscellaneous_Info = test_df.Miscellaneous_Info.str.replace(unicodedata.lookup('Indian Rupee Sign'), 'INR ')
list(train_df.Miscellaneous_Info[0:10])
# Define function to return the Feedback numbers



def find_feedback(data):

    result = re.search(r' (.*?) Feedback',data)

    if result:

        return int(result.group(1))

    else:

        return 0
# Fetch out the feedback numbers in different records. 



# Train set

train_df['feedack_num'] = train_df.Miscellaneous_Info.apply(lambda x: find_feedback(x) if '%' in str(x) else 0)



# Test set

test_df['feedack_num'] = test_df.Miscellaneous_Info.apply(lambda x: find_feedback(x) if '%' in str(x) else 0)
train_df.head()
# Let us have a look at the different Fee value in the records.



list(train_df.Miscellaneous_Info[train_df.Miscellaneous_Info.str.contains('INR', na = False)].sample(10))
# Define function to return the Fees Value



def find_fees(data):

    result = re.search(r'INR (\d*)',data)

    if result:

        return int(result.group(1))

    else:

        return 0

# Fetch out the Fees value in different records. 



# Train set

train_df['fees_val'] = train_df.Miscellaneous_Info.apply(lambda x: find_fees(x) if 'INR' in str(x) else 0)



# Test set

test_df['fees_val'] = test_df.Miscellaneous_Info.apply(lambda x: find_fees(x) if 'INR' in str(x) else 0)
train_df.head()
# Select Qualification categorical columns to be encoded



column_test = ['Qual_1', 'Qual_2', 'Qual_3', 'Qual_4',

           'Qual_5', 'Qual_6', 'Qual_7', 'Qual_8', 'Qual_9', 'Qual_10', 'Qual_11',

           'Qual_12', 'Qual_13', 'Qual_14', 'Qual_15', 'Qual_16', 'Qual_17']



column_train = ['Qual_1', 'Qual_2', 'Qual_3', 'Qual_4',

           'Qual_5', 'Qual_6', 'Qual_7', 'Qual_8', 'Qual_9', 'Qual_10']
# Encode categorical columns for Test and Train set



for i in column_train:

    train_df.replace({i: qual_dict}, inplace=True)

    

    

for i in column_test:

    test_df.replace({i: qual_dict}, inplace=True)
train_df.head()
test_df.head()
# Define function to label encode the selected categorical variable for modeling



def encode(data):

    return data.astype('category').cat.codes
# Encode categorical column of test data



columns = ['Profile','City','Locality']



for i in columns:

    col = i+"_code"

    test_df[col] = encode(test_df[i])
test_df.head()
# Create unique lists of [variable, variable code] combination and drop duplicate pairs.



df_test_merge_1 = test_df[['Profile','Profile_code']].drop_duplicates()

df_test_merge_2 = test_df[['City','City_code']].drop_duplicates()

df_test_merge_3 = test_df[['Locality','Locality_code']].drop_duplicates()
# Pull the respective encoded variables list in the train data (Using a left join) to avoid any merging issue.



train_df = pd.merge(train_df,df_test_merge_1[['Profile','Profile_code']],on='Profile', how='left')

train_df = pd.merge(train_df,df_test_merge_2[['City','City_code']],on='City', how='left')

train_df = pd.merge(train_df,df_test_merge_3[['Locality','Locality_code']],on='Locality', how='left')

# Train set after merging encoded categories



train_df.head()




def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns



missing_values_table(train_df)
cols_to_use = ['Qual_1', 'Qual_2', 'Qual_3', 'Qual_4',

               'Qual_5', 'Qual_6', 'Qual_7', 'Qual_8',

               'Qual_9', 'Qual_10', 'Profile_code', 'City_code', 'Locality_code',

               'feedack_num', 'fees_val','years_exp','Rating']



target_col = 'Fees'
for i in cols_to_use:

    train_df[i] = pd.to_numeric(train_df[i].astype(str).str.replace(',',''), errors='coerce').fillna(-1).astype(int)
train_df.fillna(-1, inplace=True)
# Define LGBM function



def run_lgb(train_X, train_y, val_X, val_y, test_X):

    params = {

        'metric': 'rmse',

        "objective" : "regression",

        "boosting": "gbdt",

        "random_state": 2019,

        "learning_rate" : 0.01,

        "verbosity" : -1

    }

    

    lgtrain = lgb.Dataset(train_X, label=train_y)

    lgval = lgb.Dataset(val_X, label=val_y)

    model = lgb.train(params, lgtrain, num_boost_round = 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)

    

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)

    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)

    return pred_test_y, model, pred_val_y
## K-FOLD train



from sklearn import model_selection



train_X = train_df[cols_to_use]

test_X = test_df[cols_to_use]

train_y = train_df[target_col].values



pred_lgb = 0



kf = model_selection.KFold(n_splits = 5, random_state=2019, shuffle=True)

for trn_index, val_index in kf.split(train_X):

  trn_X, val_X = train_X.loc[trn_index,:], train_X.loc[val_index,:]

  trn_y, val_y = train_y[trn_index], train_y[val_index]

  pred_test_tmp, model, evals_result = run_lgb(trn_X, trn_y, val_X, val_y, test_X)

  pred_lgb += pred_test_tmp
# Take average of 5 predictions and create submission file.



pred_lgb /= 5



test_lgb = df_test

test_lgb['Fees'] = pred_lgb

test_lgb.to_csv('submission_lgb.csv')
# plot feature importance of LGB



fig, ax = plt.subplots(figsize=(12,18))

lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

ax.grid(False)

plt.title("LightGBM - Feature Importance", fontsize=15)

plt.show()


# XGBoost CV

def modelfit(algo, train, test, features, label, cv_folds=5, early_stopping_rounds=50, metric="auc"):

    

    xgb_param = algo.get_xgb_params()

    xgtrain = xgb.DMatrix(train[features], label=train[label], feature_names=features)

    xgtest = xgb.DMatrix(test[features])

    cv_result = xgb.cv(params = xgb_param,

                      dtrain = xgtrain,

                      num_boost_round = algo.get_params()['n_estimators'],

                      nfold = cv_folds,

                      metrics = metric,

                      early_stopping_rounds = early_stopping_rounds

                     )

        

    #Fit the algorithm on the data

    model = xgb.train(xgb_param,

                      xgtrain,

                      num_boost_round = cv_result.shape[0],

                      verbose_eval=15

                      )

        

    #Predict on testing data:

    y_pred = model.predict(xgtest)

    

    #Display feature importance graph

    xgb.plot_importance(model);

    

    return y_pred
# Define Param and call function to execute XGBoost model



xgb1 = xgb.XGBRegressor(

    learning_rate =0.1,

    n_estimators=1000,

    gamma=0,

    objective= 'reg:linear',

    nthread=-1,

    seed=2019)



features = cols_to_use

label = target_col

train = train_df

test = test_df

pred_xgb = 0



pred_xgb = modelfit(xgb1, train, test, features, label, metric = 'rmse')
# Submit XGB prediction



test_xgb = df_test

test_xgb['Fees'] = pred_xgb

test_xgb.to_csv('submission_xgb.csv')
# Submit average of LGB and XGB



df_test["Fees"] = 0.5*test_xgb["Fees"] + 0.5*test_lgb["Fees"]

df_test.to_csv("submission_average.csv", index=False)