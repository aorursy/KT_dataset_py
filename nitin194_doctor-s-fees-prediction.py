import numpy as np 

import pandas as pd 

import unicodedata

import re

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 10



from collections import Counter



import lightgbm as lgb

import xgboost as xgb

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from catboost import Pool, CatBoostRegressor



from sklearn import metrics

from sklearn import model_selection



import gc

import time



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
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

    text = ''

    for i in series:

        text += i + ','

    return text
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
train_df['Qual_count'] = train_df.Qualification.apply(lambda x: len(x.split(',')))

test_df['Qual_count'] = test_df.Qualification.apply(lambda x: len(x.split(',')))
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
train_df.Place = train_df.Place.apply(lambda x: ','.join(str(x).lower().replace(" ","").split(",")))

test_df.Place = train_df.Place.apply(lambda x: ','.join(str(x).lower().replace(" ","").split(",")))
# Train Set

train_df['City'] = train_df['Place'].apply(lambda x: str(x).replace(' ','').split(',')[-1])

train_df['Locality'] = train_df['Place'].apply(lambda x: str(x).rsplit(',', 1)[0])





# Test Set

test_df['City'] = test_df['Place'].apply(lambda x: str(x).replace(' ','').split(',')[-1])

test_df['Locality'] = test_df['Place'].apply(lambda x: str(x).rsplit(',', 1)[0])
# Lets Check Unique Cities in test set



test_df.City.value_counts()
# Lets Check Unique Cities in train set



train_df.City.value_counts()
train_df[train_df.City == 'e']
train_df.loc[3980, 'Place'] = np.nan

train_df.loc[3980, 'City'] = np.nan

train_df.loc[3980, 'Locality'] = np.nan
# Define function to dummify feature



def get_dummies(dataframe,feature_name):

  dummy = pd.get_dummies(dataframe[feature_name], prefix=feature_name)

  dummy.drop(dummy.columns[0], axis=1, inplace=True) #avoid dummy trap

  return pd.concat([dataframe,dummy], axis = 1)
train_df = get_dummies(train_df, 'City')

test_df = get_dummies(test_df, 'City')
# Checkout dataframe after dummification of City



train_df.head()
train_df.Profile.value_counts()
train_df.Profile = train_df.Profile.apply(lambda x: str(x).lower().replace(" ",""))

test_df.Profile = train_df.Profile.apply(lambda x: str(x).lower().replace(" ",""))
train_df = get_dummies(train_df, 'Profile')

test_df = get_dummies(test_df, 'Profile')
train_df.head()
# train_df.Locality.value_counts()
# List of top 10 Localities along with there occurence in Train Set

train_df['Locality'] = train_df['Locality'].apply(str) # Convert int64 dtype to str type first

text = doc(train_df.Locality)

df = pd.DataFrame.from_dict(dict(Counter(text.split(',')).most_common()), orient='index').reset_index()

df.columns=['Locality','Count']

df.head(10)
# List of top 10 unique Localities along with there occurence in Test Set

test_df['Locality'] = test_df['Locality'].apply(str) # Convert int64 dtype to str type first

text = doc(test_df.Locality)

df = pd.DataFrame.from_dict(dict(Counter(text.split(',')).most_common()), orient='index').reset_index()

df.columns=['Locality','Count']

df.head(10)
# Define function to label encode the selected categorical variable for modeling



def encode(data):

    return data.astype('category').cat.codes
# Encode Locality column of test data



columns = ['Locality']



for i in columns:

    col = i+"_code"

    test_df[col] = encode(test_df[i])
# Check test dataset after encoding locality

test_df.head()
# Create unique lists of [variable, variable code] combination and drop duplicate pairs.



df_test_merge = test_df[['Locality','Locality_code']].drop_duplicates()
# Pull the respective encoded variables list in the train data (Using a left join) to avoid any merging issue.



train_df = pd.merge(train_df,df_test_merge[['Locality','Locality_code']],on='Locality', how='left')
# Train set after merging encoded Locality



train_df.head()
list(train_df.Miscellaneous_Info[0:10])
train_df.Miscellaneous_Info = train_df.Miscellaneous_Info.str.replace(",","")

test_df.Miscellaneous_Info = test_df.Miscellaneous_Info.str.replace(",","")
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
train_df.Fees.value_counts().reset_index().sort_values(by='index')
train_df[train_df.Fees < 50]
train_df.years_exp.describe()
test_df.years_exp.describe()
train_df[train_df.years_exp == 0]
test_df[test_df.Qualification.str.contains('getinspiredbyremarkablestoriesofpeoplelikeyou')]
train_df.fees_val.describe()
train_df.fees_val.value_counts()
train_df.loc[train_df.fees_val > 999, ['Miscellaneous_Info',"Fees"]]
test_df.loc[test_df.fees_val > 999, ['Miscellaneous_Info']]
# Define function as per above requirement.



def mark_100(data):

    data.Fees = np.where(data.Qualification.str.contains('getinspiredbyremarkablestoriesofpeoplelikeyou', na = False),

                      100,

                      data.Fees)

    return data
cols_to_use = ['Rating', 'Qual_1', 'Qual_2', 'Qual_3', 'Qual_4',

       'Qual_5', 'Qual_6', 'Qual_7', 'Qual_8', 'Qual_9', 'Qual_10',

       'Qual_count', 'years_exp', 'City_chennai', 'City_coimbatore', 'City_delhi', 'City_ernakulam', 'City_hyderabad',

       'City_mumbai', 'City_nan', 'City_thiruvananthapuram', 'Profile_dentist', 'Profile_dermatologists', 'Profile_entspecialist', 

       'Profile_generalmedicine', 'Profile_homeopath', 'Locality_code', 'feedack_num', 'fees_val']



target_col = 'Fees'
train = train_df[cols_to_use].copy()

# train['Fees'] = train_df.Fees.copy()

test = test_df[cols_to_use].copy()
train.head()
test.head()
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



missing_values_table(train)
for i in cols_to_use:

    train[i] = pd.to_numeric(train[i].astype(str), errors='coerce').fillna(-1).astype(int)

    test[i] = pd.to_numeric(test[i].astype(str), errors='coerce').fillna(-1).astype(int)
# Define Train and test set

train_X = train.copy()

test_X = test.copy()

train_y = train_df.Fees.copy()



train_X.shape, train_y.shape, test_X.shape
n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=4)
def train_model(X=train_X,

                X_test=test_X,

                y=train_y,

                params=None,

                folds=folds,

                model_type='lgb',

                plot_feature_importance=False,

                averaging='usual',

                make_oof=False,

                num_class=0

               ):

    result_dict = {}

    

    if make_oof:

        if num_class:

            oof = np.zeros((len(X), num_class))

        else:

            oof = np.zeros((len(X)))

    

    if num_class:

        prediction = np.zeros((len(X_test), num_class))

    else:

        prediction = np.zeros((len(X_test)))

    

    scores = []

    

    feature_importance = pd.DataFrame()

    

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):

        gc.collect()

        print("")

        print('Fold', fold_n + 1, 'started at', time.ctime())

        

        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        

        

        if model_type == 'lgb':

            train_data = lgb.Dataset(X_train, label=y_train,

#                                      categorical_feature = cat_cols

                                    )

            valid_data = lgb.Dataset(X_valid, label=y_valid,

#                                      categorical_feature = cat_cols

                                    )

            

            model = lgb.train(

                params,

                train_data,

                num_boost_round = 1000,

                valid_sets = [valid_data],

                verbose_eval = 100,

                early_stopping_rounds = 100

            )



            del train_data, valid_data

            

            y_pred_valid = model.predict(X_valid, num_iteration=model.best_iteration)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration)

            

            del X_valid

            gc.collect()            

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid)



            model = xgb.train(

                params = params,

                dtrain = train_data,

                num_boost_round = 1500,

                evals=[(valid_data, "Validation")],

                early_stopping_rounds = 100,

                verbose_eval = 100,

                )

            

            del train_data, valid_data

            

            y_pred_valid = model.predict(xgb.DMatrix(X_valid), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test), ntree_limit=model.best_ntree_limit)

            

            del X_valid

            gc.collect()

            

        if model_type == 'cat':

            train_pool = Pool(X_train, y_train) 

            valid_pool = Pool(X_valid, y_valid)

            

            model =  CatBoostRegressor(iterations=5000,

                                        learning_rate=0.01,

                                        eval_metric='RMSE',

                                        random_seed = 4,

                                        metric_period = 200,

                                      )

            

            model.fit(train_pool,

                      eval_set=(valid_pool),

#                       cat_features=cat_cols,

                      use_best_model=True

                     )

            

            del train_pool, valid_pool

                    

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test).reshape(-1,)

            

            del X_valid

            gc.collect()            



        if make_oof:

            oof[valid_index] = y_pred_valid

                   

#         scores.append(kappa(y_valid, y_pred_valid.argmax(1)))        

#         print('Fold kappa:', kappa(y_valid, y_pred_valid.argmax(1)))

#         print('')

        

        if averaging == 'usual':

            prediction += y_pred

        elif averaging == 'rank':

            prediction += pd.Series(y_pred).rank().values

        

        # feature importance

        if model_type == 'lgb' or model_type == 'xgb' or model_type == 'cat':

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = X.columns

            if model_type == 'lgb':

                fold_importance["importance"] = model.feature_importance()

            elif model_type == 'xgb':

                fold_importance["importance"] = fold_importance.feature.map(model.get_score()).fillna(0)

            else:

                fold_importance["importance"] = model.feature_importances_            

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= n_fold

    result_dict['prediction'] = prediction

    

#     print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    if plot_feature_importance:

        plt.figure(figsize=(15, 15));

        feature_importance = pd.DataFrame(feature_importance.groupby("feature")["importance"].mean().sort_values(ascending=False).reset_index())[:50]

        sns.barplot(x="importance", y="feature", data=feature_importance);

        plt.title('Feature Importance (avg over folds)');



        result_dict['feature_importance'] = feature_importance

            

    if make_oof:

        result_dict['oof'] = oof

    

    return result_dict
lgb_params = {

    'metric': 'rmse',

    "objective" : "regression",

    'min_data_in_leaf': 149, 

    'max_depth': 9,

    "boosting": "gbdt",

    "lambda_l1": 0.2634,

    "random_state": 133,

    "num_leaves" : 30,

    "min_child_samples" : 100,

    "learning_rate" : 0.1,

    "bagging_fraction" : 0.7,

    "feature_fraction" : 0.5,

    "bagging_frequency" : 5,

    "bagging_seed" : 4,

    "verbosity" : -1

    }



result_dict_lgb = train_model(X=train_X,

                              X_test=test_X,

                              y=train_y,

                              params=lgb_params,

                              model_type='lgb',

                              plot_feature_importance=True,

                              make_oof=True,

                              num_class=0                           

                             )
test_lgb = test_df[['Qualification', 'Experience', 'Rating', 'Place', 'Profile','Miscellaneous_Info']].copy()

test_lgb['Fees'] = result_dict_lgb['prediction']
xgb_params = {

    "learning_rate" : 0.01,

    "n_estimators" : 3400,

    "max_depth" : 7,

    "min_child_weight" : 0,

    "gamma": 0,

    "subsample" : 0.7,

    "colsample_bytree" : 0.7,

    "objective" : 'reg:linear',

    "nthread" : -1,

    "scale_pos_weight" : 1,

    "seed" : 4,

    "reg_alpha" : 0.00006

}



result_dict_xgb = train_model(X=train_X,

                              X_test=test_X,

                              y=train_y,

                              params=xgb_params,

                              model_type='xgb',

                              plot_feature_importance=True,

                              make_oof=True,

                              num_class=0

                             )
# Create XGB prediction



test_xgb = test_df[['Qualification', 'Experience', 'Rating', 'Place', 'Profile','Miscellaneous_Info']].copy()

test_xgb['Fees'] = result_dict_xgb['prediction']
result_dict_cat = train_model(X=train_X,

                              X_test=test_X,

                              y=train_y,

                              model_type='cat',

                              plot_feature_importance=True,

                              make_oof=True,

                              num_class=0                           

                             )
# Create CAT prediction



test_cat = test_df[['Qualification', 'Experience', 'Rating', 'Place', 'Profile','Miscellaneous_Info']].copy()

test_cat['Fees'] = result_dict_cat['prediction']
# Create X and Y dataset



Y = train_y.copy()

X = train[cols_to_use]
from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization

from keras.optimizers import Adam



# Function to create model

def baseline_model(learn_rate=0.01, init_mode='normal', activation = 'relu', dropout_rate=0.1, weight_constraint=1, neurons = 100):

    model = Sequential()

    model.add(Dense(units = neurons, kernel_initializer = init_mode, activation = activation, input_dim = 29))

    model.add(BatchNormalization())

#     model.add(Dropout(rate = dropout_rate))

    model.add(Dense(units = int(neurons / 2), kernel_initializer = init_mode, activation = activation))

#     model.add(Dropout(rate = dropout_rate))

    model.add(Dense(units = int(neurons / 4), kernel_initializer = init_mode, activation = activation))

#     model.add(Dropout(rate = dropout_rate))

    model.add(Dense(units = 1, kernel_initializer = init_mode, activation = 'linear'))

    optimizer = Adam(lr=learn_rate)

    model.compile(optimizer = optimizer, loss = 'mean_squared_logarithmic_error', metrics = ['mse'])

    return model



# fix random seed for reproducibility

seed = 4

np.random.seed(seed)
from keras.callbacks import ModelCheckpoint



checkpoint_name = 'weights.best.hdf5' 

checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose = 1, save_best_only = True, mode ='auto')

callbacks_list = [checkpoint]



# Train the model

model = baseline_model()

model.summary()

model.fit(X, Y, epochs=100, batch_size=58, validation_split = 0.2, callbacks=callbacks_list)
weights_file = checkpoint_name # choose the best checkpoint 

model.load_weights(weights_file) # load it

model.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.01), metrics=['mean_absolute_error'])
# Predictions

pred_dnn = model.predict(test[cols_to_use])
# Create submission

test_dnn = test_df[['Qualification', 'Experience', 'Rating', 'Place', 'Profile','Miscellaneous_Info']].copy()

test_dnn['Fees'] = pred_dnn
# Create average of LGB, XGB and DNN



df_test = test_df[['Qualification', 'Experience', 'Rating', 'Place', 'Profile','Miscellaneous_Info']].copy()

df_test["Fees"] = (test_xgb["Fees"] + test_lgb["Fees"] + test_cat["Fees"] + test_dnn["Fees"])/4
test_lgb = mark_100(test_lgb.copy())

test_xgb = mark_100(test_xgb.copy())

test_cat = mark_100(test_cat.copy())

test_dnn = mark_100(test_dnn.copy())

df_test = mark_100(df_test.copy())
test_lgb.to_csv('submission_lgb.csv', index=False)

test_xgb.to_csv('submission_xgb.csv', index=False)

test_cat.to_csv('submission_cat.csv', index=False)

test_dnn.to_csv('submission_dnn.csv', index=False)

df_test.to_csv('submission_average.csv', index=False)