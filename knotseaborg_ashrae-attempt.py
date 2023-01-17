import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pandas.read_csv)

import seaborn as sns

import random

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import DecisionTreeRegressor

import lightgbm as lgb

from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import train_test_split

import warnings

from tqdm import tqdm_notebook



warnings.filterwarnings('ignore')



import gc

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#This reduces the memory usage!

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
#Reading all the data

def read_data(data_map, location):

    for key in data_map.keys():

        if key == location[39:-4]:

            data = pd.read_csv(location)

            print(key)

            print("Size:",data.shape)

            print("-----------------------------------------------------")

            data_map[key] = reduce_mem_usage(data)



list_of_loc = ["/kaggle/input/ashrae-energy-prediction/train.csv",

               "/kaggle/input/ashrae-energy-prediction/building_metadata.csv",

               "/kaggle/input/ashrae-energy-prediction/sample_submission.csv",

               "/kaggle/input/ashrae-energy-prediction/weather_test.csv",

               "/kaggle/input/ashrae-energy-prediction/weather_train.csv",

               "/kaggle/input/ashrae-energy-prediction/test.csv"]



data_map = {'train':None, 'building_metadata': None, 'sample_submission': None, 'weather_test':None, 'weather_train': None, 'test':None}

for loc in list_of_loc:

    read_data(data_map, loc)
#This function will show a distplot of all numerical features of a dataframe.



def show_dist(data_map, key):

    df = data_map[key].select_dtypes('number')

    list_of_col = df.columns

    print(list_of_col)



    fig, list_of_axis = plt.subplots(ncols = 2, nrows=int(np.ceil(len(df.columns)/2)), figsize = (15,12))



    for i,col in enumerate(list_of_col):

        sns.distplot(df.loc[~df[col].isna(), col], ax = list_of_axis[int(i/2),i%2])
#Let's analyze the distribution of numerical values of train and test sets in order to ensure that they're having a 

#compatible distrbution!

#Let's try to define a function for this



def compare_dist(data_map, train, test):

    df_train = data_map[train].select_dtypes('number')

    df_test = data_map[test].select_dtypes('number')

    list_of_col = set(df_train.columns).intersection(set(df_test.columns))

    #print(list_of_col)



    fig, list_of_axis = plt.subplots(ncols = 2, nrows=len(list_of_col), figsize = (15,4*len(list_of_col)))

    

    for i,col in enumerate(list_of_col):

        train_plot = sns.distplot(df_train.loc[~df_train[col].isna(), col], ax = list_of_axis[i,0])

        test_plot = sns.distplot(df_test.loc[~df_test[col].isna(), col], ax = list_of_axis[i,1])

        

        if i == 0:

            train_plot.set_title("Train data")

            test_plot.set_title("Test data")

            

#compare_dist(data_map, 'train', 'test')
data_map['weather_train'].head()
#Alright! We performed a little preprocessing, ensured that both test and train data belong to the same distribution.

#As of now, I believe that the problem has provided the new weather conditions and all we must do is predict, the power used.

#So let's tackle the missing values now.



def show_missing(data_map, keys):

    for key in keys:

        list_of_missing_col = []

        df = data_map[key]

        for col in df.columns:

            if df[col].isna().sum().any() >0:

                list_of_missing_col.append((col, data_map[key][col].isna().sum()/data_map[key].shape[0]))

        

        if list_of_missing_col:

            print(key)

            print(list_of_missing_col)

        

show_missing(data_map, data_map.keys())
#Let us drop year_built and floor_count. Pretty useless!

#data_map['building_metadata'].drop(['year_built', 'floor_count'], axis=1, inplace=True)
building_id = random.randint(0,1448)

fig, ax_l = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

meter_map = {0:'Electricity', 1: 'Chilledwater', 2:'Steam', 3:'Hotwater'}

for i in range(4):

    data = data_map['train'][(data_map['train']['building_id'] == building_id) & (data_map['train']['meter'] == i)]

    sns.scatterplot(range(data.shape[0]), data['meter_reading'], ax=ax_l[int(i/2), int(i%2)]).set_title("Meter for %s"%meter_map[i])

    

#Looks like each meter has their own trend. We should make 4 different meters to boost the working.
#We intend to create models to fill some of the numerical values in the column. Though not very effective, it is definitely bettwr than simply filling with mode or mean.

#Initially I thought of dropping the columns with large number of missing values, however we'll run a column selection test later and use XGB to choose best combination of columns.



def extract_datetime(df):

    if 'timestamp' in df.columns:

        time_df = pd.to_datetime(df['timestamp'])

        df['year'] = time_df.dt.year

        df['month'] = time_df.dt.month

        df['day'] = time_df.dt.day

        df['hour'] = time_df.dt.hour

        

        #Finally, dropping timestamp as we don't need it anymore

        #df.drop('timestamp', axis=1, inplace=True)

    return df



def build_weather_regressor(df, num_features, cat_features, target):

    print("Building regressor for",target)

    regressor = DecisionTreeRegressor(random_state=0)

    

    df = pd.get_dummies(df[num_features+cat_features+[target]], columns=cat_features, prefix=cat_features)[~df[target].isna()]

    #We gotta use timestamp too!

    extract_datetime(df)

    scores = cross_val_score(regressor, df.drop(labels=target, axis=1), df[target], cv=5)

    print("Mean Cross validation score:", np.mean(scores))



    regressor.fit(df.drop(labels=target, axis=1), df[target])

    score = regressor.score(df.drop(labels=target, axis=1), df[target])

    print("Regressor Accuracy(On train data itself)", score)

    

    return regressor



def fill_missing_weather(train,test):



    #Let's fill the mode of air_temperature, wind_direction and dew_temperature for their respective values

    missing_features = ['air_temperature', 'wind_direction', 'dew_temperature']

    for feature in missing_features:

        data_map[train].loc[data_map[train][feature].isna(), feature] = data_map[train][feature].mode()[0]

        data_map[test].loc[data_map[test][feature].isna(), feature] = data_map[test][feature].mode()[0]

    

    #Now, we use these to predict other missing values, which are 'cloud_coverage', 'sea_level_pressure', 'wind_direction', 'wind_speed'

    missing_features = ['cloud_coverage', 'sea_level_pressure', 'wind_direction', 'wind_speed', 'precip_depth_1_hr' ]

    num_features = ['air_temperature', 'wind_direction', 'dew_temperature']

    cat_features = ['site_id'] 

    

    regressors = {}

    

    for key in (train, test):

        for target in missing_features:

            if target not in regressors:

                #Building and training the regressor. Might as well use both training and test data for this.

                regressors[target] = build_weather_regressor(pd.concat([data_map[train], data_map[test]]), num_features, cat_features, target)



            missing_data = pd.get_dummies(data_map[key][num_features+cat_features+[target]], columns=cat_features, prefix=cat_features)[data_map[key][target].isna()]

            missing_data.drop(labels=target, axis=1, inplace = True)

            #We gotta use timestamp too!

            extract_datetime(missing_data)

            

            if missing_data.shape[0] > 0:

                data_map[key].loc[data_map[key][target].isna(), target] = regressors[target].predict(missing_data)
#Fill missing values for weather data in here

fill_missing_weather('weather_train', 'weather_test')
data_map['weather_train'].drop(['precip_depth_1_hr', 'cloud_coverage'], axis=1, inplace=True)

data_map['weather_test'].drop(['precip_depth_1_hr', 'cloud_coverage'], axis=1, inplace=True)
#We gotta get rid of zeros, because they're a lot! Skewing and screwing our model!

#filter_ = data_map['train']['meter_reading']>0

#data_map['train'] = data_map['train'][filter_]
#Since we're applying log, we better not have it as zero

filter_ = data_map['train']['meter_reading'] == 0

data_map['train'].loc[filter_,'meter_reading'] = 0.01
#Some values are negative for precipitation, which is not possible.

#data_map['weather_train']['precip_depth_1_hr'][data_map['weather_train']['precip_depth_1_hr'] < 0] = 0

#data_map['weather_test']['precip_depth_1_hr'][data_map['weather_test']['precip_depth_1_hr'] < 0] = 0
#Now that's done, we need to work on some features. They are timestano, primary_use and site_id. 

#We need to convert the time of meter reading into timeseries and perform analysis. Let's go! Also, we just

#Split the timestamp into year, month, day and hour. Let's go!



def extract_datetime(df):

    if 'timestamp' in df.columns:

        time_df = pd.to_datetime(df['timestamp'])

        df['year'] = time_df.dt.year

        df['month'] = time_df.dt.month

        df['day'] = time_df.dt.day

        df['hour'] = time_df.dt.hour

        

        #Finally, dropping timestamp as we don't need it anymore

        #df.drop('timestamp', axis=1, inplace=True)

    return df



#Something like a pipeline.. here. Only extracts date and provides one_hot_encoding.

def preprocess(df, cat_features=None):

    extract_datetime(df)

    if cat_features:

        df = pd.concat([pd.get_dummies(df, columns=cat_features, prefix=cat_features), df[cat_features]], axis=1)

    gc.collect()



    return df



data_map['weather_train'] = preprocess(data_map['weather_train'])

data_map['weather_test'] = preprocess(data_map['weather_test'])

data_map['building_metadata'] = preprocess(data_map['building_metadata'], ['primary_use']) #We exclude site id here, because weather data already has dummies of it!
#Let's join them to create one big chunk of data

def combine_dataframes(main_df, weather_df):

    metadata_df = data_map['building_metadata'] #Static info

    #First merge

    df = pd.merge(left=main_df, right=metadata_df, how="left", on=["building_id"])

    #Second merge

    df = pd.merge(left=df, right=weather_df, how="left", on=["site_id", "timestamp"])



    gc.collect()

    return df
features = ['building_id', 'site_id', 'square_feet','primary_use_Education',

       'primary_use_Entertainment/public assembly',

       'primary_use_Food sales and service', 'primary_use_Healthcare',

       'primary_use_Lodging/residential',

       'primary_use_Manufacturing/industrial', 'primary_use_Office',

       'primary_use_Other', 'primary_use_Parking',

       'primary_use_Public services', 'primary_use_Religious worship',

       'primary_use_Retail', 'primary_use_Services',

       'primary_use_Technology/science', 'primary_use_Utility',

       'primary_use_Warehouse/storage', 'air_temperature', 'cloud_coverage',

       'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',

       'wind_direction', 'wind_speed', 'year', 'month', 'day', 'hour']



target = 'meter_reading'
models = {}

for i in range(4):

    models[i] = lgb.LGBMRegressor(reg_alpha=0.5, reg_lambda=0.5, random_state=0, n_jobs=4, subsample=0.9)
#Okay, we've now successfully combined the data into test and train set after some analysis. Let's move on to train the model



def train_model(model, train_df, testing = False):

    #train_df = data_map['train']

    train = combine_dataframes(train_df, data_map['weather_train']).drop(labels=['timestamp', 'primary_use', 'meter'], axis=1)

    if testing:

        X_train, X_test, y_train, y_test = train_test_split(train.drop('meter_reading', axis=1), np.log(train['meter_reading']))

    else:

        X_train, y_train = train.drop('meter_reading', axis=1), np.log(train['meter_reading'])



    model.fit(X_train,y_train,verbose=False)



    if testing:

        y_pred = model.predict(X_test)

        error = np.sqrt(mean_squared_log_error( np.exp(y_test), np.exp(y_pred) ))

        print("Test error", error)

        return error

    gc.collect()



#train_model(model, data_map['train'], testing = False)
#All features

for i in range(4):

    train_data = data_map['train'][data_map['train']['meter'] == i]

    train_model(models[i], train_data, testing = True)
def make_prediction(model, test_df, output, batch_size=100000, features=None):

    #test_df = data_map['test']

    row_ids = test_df['row_id'].reset_index(drop=True)

    test = combine_dataframes(test_df, data_map['weather_test']).drop(labels=['timestamp', 'primary_use', 'row_id', 'meter'], axis=1)

    if features != None:

        test = test[features]



    for i in tqdm_notebook(range(0,test.shape[0], batch_size)):

        test_batch = test.iloc[i:i+batch_size,:]

        selected_row_ids = row_ids[i:i+batch_size]

        #print(model.predict(test_batch).shape, output.iloc[selected_row_ids, 'meter_Reading'].shape)

        output.loc[selected_row_ids, 'meter_reading'] = np.exp(model.predict(test_batch))



#result = make_prediction(model, data_map['test'], 100000)
reading = pd.DataFrame(data_map['test']['row_id'])

reading['meter_reading'] = 0

for i in range(4):

    print("For meter %d"%i)

    test_data = data_map['test'][data_map['test']['meter'] == i]

    make_prediction(models[i], test_data, reading)

    gc.collect()
reading.to_csv('solution.csv', index=False)
def find_best_parameters():

    alpha_grid = np.linspace(0,1,1)

    lambda_grid = np.linspace(0,1,1)

    num_leaves = list(range(20,21,1))



    result = []

    

    for alpha in alpha_grid:

        for lambda_ in lambda_grid:

            for nleaves in num_leaves:

                model = lgb.LGBMRegressor(reg_alpha=alpha, reg_lambda=lambda_, random_state=0, n_jobs=4, subsample=0.9, num_leaves=nleaves,  learning_rate=0.01, )

                error = train_model(model, data_map['train'], testing = True)

                

                result.append((alpha, lambda_,num_leaves, error))



    gc.collect()

    

    return result

    

#pd.DataFrame(find_best_parameters()).to_csv('parameter_grid.csv')