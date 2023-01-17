import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import pickle

import gc

import math

import scipy.stats

import numpy as np

import pandas as pd

import datetime as dt

import xgboost as xgb

import lightgbm as lgb

import matplotlib.pyplot as plt

from tqdm import tqdm

from time import time

from sklearn import preprocessing

from scipy.sparse import csc_matrix

from sklearn.model_selection import TimeSeriesSplit

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold, StratifiedKFold

from imblearn.over_sampling import RandomOverSampler



from kaggle.competitions import nflrush



pd.set_option('display.max_rows', 200)

pd.set_option('display.max_columns', 200)

pd.set_option('mode.chained_assignment', None)



train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

train_df = train.copy()
def extract_rusher(data): 

    rusher_df = pd.DataFrame()

    for A in tqdm(range(0, len(data), 22)):

        section_df = data[A:A+22]

        rusher_row = section_df.loc[section_df['NflId'] == section_df['NflIdRusher']]

        

        # add distance column, select top 5 rows based on distance

        section_df = select_distance(rusher_row, section_df) 

        

        rusher_row['Horziontal_Velocity_Rusher'] = rusher_row['S'] * math.cos(rusher_row['Dir'])

        #rusher_row['Force_Rusher'] = rusher_row['PlayerWeight']*rusher_row['A']*math.cos(rusher_row['Dir'])  #calculate the horizontal force F = ma

        #rusher_row['Momentum_Rusher'] = rusher_row['PlayerWeight']*rusher_row['S']*math.cos(rusher_row['Dir'])  # momentum = mv

        #rusher_row = add_force(rusher_row, section_df)

        #rusher_row = add_momentum(rusher_row, section_df)

        rusher_row = add_horizontal_velocity(rusher_row, section_df)

        rusher_row = add_speed(rusher_row, section_df)

        rusher_row = add_acceleration(rusher_row, section_df)

        rusher_row = add_motion_direction(rusher_row, section_df)

        rusher_row = add_distance_traveled(rusher_row, section_df)

        rusher_row = add_distance(rusher_row, section_df)

        rusher_df = pd.concat([rusher_df, rusher_row],sort=False)

        del section_df, rusher_row

    del data  

    return rusher_df



def test_pipeline(rusher_row, test_df):  #sfm

    



    rusher_row['Horziontal_Velocity_Rusher'] = rusher_row['S'] * math.cos(rusher_row['Dir'])

    #rusher_row['Force_Rusher'] = rusher_row['PlayerWeight']*rusher_row['A']*math.cos(rusher_row['Dir'])  #calculate the horizontal force F = ma

    #rusher_row['Momentum_Rusher'] = rusher_row['PlayerWeight']*rusher_row['S']*math.cos(rusher_row['Dir'])  # momentum = mv

    #rusher_row = add_force(rusher_row, test_df)

    #rusher_row = add_momentum(rusher_row, test_df)

    rusher_row = add_horizontal_velocity(rusher_row, test_df)

    rusher_row = add_speed(rusher_row, test_df)

    rusher_row = add_acceleration(rusher_row, test_df)

    rusher_row = add_motion_direction(rusher_row, test_df)

    rusher_row = add_distance_traveled(rusher_row, test_df)

    rusher_row = add_distance(rusher_row, test_df)

    #rusher_row = rusher_row.drop(columns = ['NflId','NflIdRusher','Humidity','Orientation','PlayerHeight_cm','PlayerAge'])

    rusher_row = rusher_row.drop(columns = ['NflId','NflIdRusher'])

    #rusher_row = sfm.transform(rusher_row)

    

    return rusher_row





# Create a new column: US location will be 1, non-US location will be 0

def us_location(data):

    us_location_list = list(set(list(data['Location'])))

    non_us_location = ['London, England','London','Mexico City']

    for A in non_us_location:

        if A in us_location_list:

            us_location_list.remove(A)

    US_location_list = []

    for A in list(data['Location']):

        if A in us_location_list:

            US_location_list.append(1)

        else:

            US_location_list.append(0)

    data['US_location'] =  US_location_list

    del US_location_list,us_location_list

    return data



def generate_string(column, upper_limit):

    string_no = range(0,upper_limit)

    string_list = []

    for A in string_no:

        string_list.append(column+str(A))   

    return string_list



def add_horizontal_velocity(rusher_row, section_df):

    non_rusher_df = section_df.loc[section_df['NflId'] != section_df['NflIdRusher']]



    horizontal_velocity_non_rusher = []

    for i in zip(non_rusher_df['S'],non_rusher_df['Dir']):

        horizontal_velocity_non_rusher.append((i[0]*math.cos(i[1])))

    

    string_list = generate_string('Horizontal_Velocity_',len(horizontal_velocity_non_rusher))

    

    rusher_row_joined = np.concatenate((rusher_row.values, [horizontal_velocity_non_rusher]),axis = 1)

    rusher_row_joined_df = pd.DataFrame(data=rusher_row_joined, columns = list(rusher_row.columns) + string_list) 

    del non_rusher_df, horizontal_velocity_non_rusher, string_list

    return rusher_row_joined_df



def add_speed(rusher_row, section_df):

    non_rusher_df = section_df.loc[section_df['NflId'] != section_df['NflIdRusher']]

    speed_non_rusher = list(non_rusher_df['S'])

    string_list = generate_string('Speed_',len(speed_non_rusher))

    rusher_row_joined = np.concatenate((rusher_row.values, [speed_non_rusher]),axis = 1)

    rusher_row_joined_df = pd.DataFrame(data=rusher_row_joined, columns = list(rusher_row.columns) + string_list) 

    del non_rusher_df, speed_non_rusher, string_list

    

    return rusher_row_joined_df



def add_acceleration(rusher_row, section_df):

    non_rusher_df = section_df.loc[section_df['NflId'] != section_df['NflIdRusher']]

    acceleration_non_rusher =  list(non_rusher_df['A'])

    string_list = generate_string('Acceleration_',len(acceleration_non_rusher))

    rusher_row_joined = np.concatenate((rusher_row.values, [acceleration_non_rusher]),axis = 1)

    rusher_row_joined_df = pd.DataFrame(data=rusher_row_joined, columns = list(rusher_row.columns) + string_list) 

    

    del non_rusher_df, acceleration_non_rusher, string_list

    return rusher_row_joined_df





def add_force(rusher_row, section_df):

    non_rusher_df = section_df.loc[section_df['NflId'] != section_df['NflIdRusher']]

    force_non_rusher = []

    

    for i in zip(non_rusher_df['PlayerWeight'],non_rusher_df['A'],non_rusher_df['Dir']):

        force_non_rusher.append((i[0]*i[1]*math.cos(i[2])))

    string_list = generate_string('Force_',len(force_non_rusher))

    

    for j in range(len(string_list)):

        rusher_row[string_list[j]] =  force_non_rusher[j] 

    del non_rusher_df, force_non_rusher, string_list

    return rusher_row



def add_momentum(rusher_row, section_df):

    non_rusher_df = section_df.loc[section_df['NflId'] != section_df['NflIdRusher']]

    momentum_non_rusher = []

    

    for i in zip(non_rusher_df['PlayerWeight'],non_rusher_df['S'],non_rusher_df['Dir']):

        momentum_non_rusher.append((i[0]*i[1]*math.cos(i[2])))

    string_list = generate_string('Momentum_',len(momentum_non_rusher))

    

    for j in range(len(string_list)):

        rusher_row[string_list[j]] =  momentum_non_rusher[j] 

    del non_rusher_df, momentum_non_rusher, string_list

    return rusher_row



def add_motion_direction(rusher_row, section_df):

    non_rusher_df = section_df.loc[section_df['NflId'] != section_df['NflIdRusher']]

    direction_non_rusher = list(non_rusher_df['Dir'])

    string_list = generate_string('Direction_',len(direction_non_rusher))

    rusher_row_joined = np.concatenate((rusher_row.values, [direction_non_rusher]),axis = 1)

    rusher_row_joined_df = pd.DataFrame(data=rusher_row_joined, columns = list(rusher_row.columns) + string_list)

    

    del non_rusher_df, direction_non_rusher, string_list

    return rusher_row_joined_df





def add_distance_traveled(rusher_row, section_df):

    non_rusher_df = section_df.loc[section_df['NflId'] != section_df['NflIdRusher']]

    dis_non_rusher = list(non_rusher_df['Dis'])

    string_list = generate_string('Dis_',len(dis_non_rusher))

    rusher_row_joined = np.concatenate((rusher_row.values, [dis_non_rusher]),axis = 1)

    rusher_row_joined_df = pd.DataFrame(data=rusher_row_joined, columns = list(rusher_row.columns) + string_list)

    

    del non_rusher_df, dis_non_rusher, string_list

    return rusher_row_joined_df





def OffensePersonnel(data):  

    data['RB'] = data['OffensePersonnel'].apply(lambda x: int(x[0]))

    data['TE'] = data['OffensePersonnel'].apply(lambda x: int(x[6]))

    data['WR'] = data['OffensePersonnel'].apply(lambda x: int(x[12]))

    

    return data





def select_distance(rusher_row, section_df):

    

    X_rusher = rusher_row['X'].values[0]

    Y_rusher = rusher_row['Y'].values[0]

    distance_to_non_rusher = []

    for A in zip(section_df['X'], section_df['Y']):

        current_distance = ((A[0]-X_rusher)**2 + (A[1]-Y_rusher)**2)**0.5

        distance_to_non_rusher.append(current_distance)

    

    section_df['straightline_dist'] = distance_to_non_rusher

    section_df = section_df.sort_values('straightline_dist', ascending= True)

    #section_df = section_df.iloc[:6] # select the closest 6 rows 

    return section_df

     



def add_distance(rusher_row, section_df):

    

    non_rusher_df = section_df.loc[section_df['NflId'] != section_df['NflIdRusher']]

    distance_to_non_rusher = list(non_rusher_df['straightline_dist'])

    string_list = generate_string('straightline_dist_',len(distance_to_non_rusher))



    rusher_row_joined = np.concatenate((rusher_row.values, [distance_to_non_rusher]),axis = 1)

    rusher_row_joined_df = pd.DataFrame(data=rusher_row_joined, columns = list(rusher_row.columns) + string_list)

        

    del non_rusher_df, distance_to_non_rusher, string_list

    return rusher_row_joined_df







def time_processing(data):

    

    data["timestamp"] = pd.to_datetime(data["TimeHandoff"]) 

    #data["month"] = data["timestamp"].dt.month

    #data["dayOfMonth"] = data["timestamp"].dt.day

    #data["hour"] = data["timestamp"].dt.hour

    #data["minute"] = data["timestamp"].dt.minute

    data = data.drop(columns = ['timestamp'])

    

    return data



def height_processing(data):

    height_list = []

    for A in data['PlayerHeight']:

        if len(A) == 3:

            height = round((int(A[0])*30.48+int(A[2])*2.54),3)

        else:

            height = round((int(A[0])*30.48+int(A[-2:])*2.54),3)

        height_list.append(height)  

    data['PlayerHeight_cm'] = height_list 

    return data





drop_columns_train = [

                'GameId', 'PlayId','DisplayName', 'JerseyNumber',

                'GameClock', 'FieldPosition', 

                'OffensePersonnel','DefensePersonnel','TimeHandoff','TimeSnap',

                'PlayerHeight','PlayerBirthDate','PlayerCollegeName',

                 'Stadium','Location','Turf','OffenseFormation',

                'GameWeather','StadiumType','WindDirection', 'WindSpeed', #'NflId','NflIdRusher',

               

                'Season','Week'#'Position','HomeTeamAbbr','VisitorTeamAbbr','Humidity','PossessionTeam',

                #'Yards'   # TARGET

          

                ]



drop_columns_test = [

                'GameId','PlayId','DisplayName','JerseyNumber',

                'GameClock', 'FieldPosition',

                'OffensePersonnel','DefensePersonnel','TimeHandoff','TimeSnap',

                'PlayerHeight','PlayerBirthDate','PlayerCollegeName',

                'Stadium','Location','Turf','OffenseFormation',

                'GameWeather','StadiumType','WindDirection','WindSpeed',#'NflId','NflIdRusher',

               

                'Season','Week'#'Position','HomeTeamAbbr','VisitorTeamAbbr','Humidity','PossessionTeam',



                ]



from sklearn import preprocessing

lbl = preprocessing.LabelEncoder()



def preprocessing(data, drop_columns):

    

    data = data.fillna(-999)

    data = us_location(data)

    data = OffensePersonnel(data)

    data = time_processing(data)

    data = height_processing(data)

    

    GameClock_list = [int(A[:2])*60 + int(A[3:5]) for A in data['GameClock']]

    data['GameTimeRemain'] = GameClock_list

    

    age_list = [2020-int(str(A)[-4:]) for A in data['PlayerBirthDate']]

    data['PlayerAge'] = age_list

    

    # convert to KG

    data['PlayerWeight'] = data['PlayerWeight']*0.453592

    

    '''

   

    onehot_columns = ['HomeTeamAbbr','VisitorTeamAbbr']

    data = pd.get_dummies(data, prefix_sep="__",columns=onehot_columns)

    '''

    

    data = data.drop(columns= drop_columns)

    

    for f in data.columns:

        if data[f].dtype=='object':

            lbl.fit(data[f].values)

            data[f] = lbl.transform(data[f].values)



    return data





def convert_data_type(data):

    

    convert_columns = ['Yards','Team','Quarter','Position',

                      'DefendersInTheBox','PlayerAge','PlayDirection']

    

    for A in convert_columns:

        data[A] = data[A].astype('int64')

    

    return data

    
train_df_processed = preprocessing(train_df,drop_columns_train)

rusher_df =  extract_rusher(train_df_processed)

rusher_df = convert_data_type(rusher_df)

target = rusher_df.Yards.values

#rusher_df = rusher_df.drop(columns = ['NflId','NflIdRusher','Yards','Humidity','Orientation','PlayerHeight_cm','PlayerAge'])

rusher_df = rusher_df.drop(columns = ['NflId','NflIdRusher','Yards']) #,'Yards'

target_list = sorted(list(set(target)))

gc.collect()
'''

import matplotlib.pyplot as plt



corr = rusher_df.corr()

corr.style.background_gradient(cmap='coolwarm')

corr.style.background_gradient(cmap='coolwarm').set_precision(4)

'''
#rusher_df = rusher_df.drop(columns = ['Yards']) 
rusher_df
'''



def split_train(rusher_df, target, train_fraction):

    

    X_train = rusher_df.iloc[:int(len(rusher_df)//(1/train_fraction))]

    X_valid = rusher_df.iloc[int(len(rusher_df)//(1/train_fraction)):]

    y_train = target[:int(len(rusher_df)//(1/train_fraction))]

    y_valid = target[int(len(rusher_df)//(1/train_fraction)):]

    

    return X_train, X_valid, y_train, y_valid





X_train, X_valid, y_train, y_valid = split_train(rusher_df, target, train_fraction = 0.9)





clf = xgb.XGBClassifier(

                            

                        n_estimators=50,

                        min_child_weight = 2,

                        max_depth=6,

                        verbosity = 1,

                        n_jobs=-1,                                              

                        scale_pos_weight=1.025,

                        tree_method='exact',

                        objective = 'multi:softmax',

                        num_class = len(target_list),

                        predictor='cpu_predictor',

                        colsample_bytree = 0.66,

                        subsample = 1,

                        gamma = 0,

                        learning_rate=0.15,

                        num_parallel_tree = 1    

    

                       )



clf.fit(X_train, y_train, eval_metric="merror",early_stopping_rounds=20,

        eval_set=[(X_train, y_train),(X_valid, y_valid)], verbose=True)

        

'''
'''

# No validation Set

clf = xgb.XGBClassifier(

                            

                        n_estimators=50,

                        min_child_weight = 2,

                        max_depth=6,

                        verbosity = 1,

                        n_jobs=-1,                                              

                        scale_pos_weight=1.025,

                        tree_method='exact',

                        objective = 'multi:softmax',

                        num_class = len(target_list),

                        predictor='cpu_predictor',

                        colsample_bytree = 0.66,

                        subsample = 1,

                        gamma = 0,

                        learning_rate=0.15,

                        num_parallel_tree = 1    

                       )



clf.fit(rusher_df, target, eval_metric="merror",early_stopping_rounds=20,

        eval_set=[(rusher_df, target)], verbose=True)

'''
'''



feature_importance_list = clf.feature_importances_



feature_dict = dict()

for A in range(len(list(rusher_df.columns))):

    feature_dict[list(rusher_df.columns)[A]] = feature_importance_list[A]

    

import operator

sorted_feature_dict = sorted(feature_dict.items(), key=operator.itemgetter(1))



sorted_feature_dict

'''
'''



from sklearn.feature_selection import SelectFromModel

np.random.seed(1)

sfm = SelectFromModel(clf, threshold = 0.005)

sfm.fit(rusher_df, target)





rusher_df_reduced = sfm.transform(rusher_df)

rusher_df_reduced.shape



clf.fit(rusher_df, target, eval_metric="merror",early_stopping_rounds=20,

        eval_set=[(rusher_df_reduced, target)], verbose=True)

'''
ros = RandomOverSampler(random_state=999)

X_ros, y_ros = ros.fit_sample(rusher_df, target)



X_train, X_valid, y_train, y_valid = train_test_split(X_ros, y_ros, shuffle = True,

                                test_size = 0.2,random_state = 999)





clf = xgb.XGBClassifier(

                            

                        n_estimators=500,

                        min_child_weight = 2,

                        max_depth=6,

                        verbosity = 1,

                        n_jobs=-1,                                              

                        scale_pos_weight=1.025,

                        tree_method='hist',

                        objective = 'multi:softmax',

                        num_class = len(target_list),

                        predictor='cpu_predictor',

                        colsample_bytree = 0.66,

                        subsample = 1,

                        gamma = 0,

                        learning_rate=0.15,

                        num_parallel_tree = 1    

    

                       )



clf.fit(X_train, y_train, eval_metric="merror",early_stopping_rounds=30,

        eval_set=[(X_train, y_train),(X_valid, y_valid)], verbose=True)
env = nflrush.make_env()

iter_test = env.iter_test()

batch_no = 0



for (test_df, sample_prediction_df) in iter_test:

    print(f'Predicting Play Number {batch_no}')

    rusher_row = test_df.loc[test_df['NflId'] == test_df['NflIdRusher']]

    rusher_row = preprocessing(rusher_row, drop_columns_test)

    test_df = select_distance(rusher_row, test_df)

    rusher_row = test_pipeline(rusher_row, test_df)    #sfm

    #preds_proba_mean = sum([model.predict_proba(rusher_row) for model in models])/len(models)

    preds_proba = clf.predict_proba(rusher_row.values)

    

    pred_df = np.zeros((1, 199))   

    for A in range(len(target_list)):      

        pred_df[0][target_list[A]+99] = preds_proba[0][A]        # preds_proba_mean

    for A in range(1,len(pred_df[0]),1):

        pred_df[0][A] = pred_df[0][A]+pred_df[0][A-1]   

    pred_df =  np.clip(pred_df, a_min= 0, a_max=1)

    pred_df[0][-1] = 1.0

    

    final_pred_df = pd.DataFrame(data=pred_df, columns=sample_prediction_df.columns)

    env.predict(final_pred_df)

    batch_no += 1

env.write_submission_file()
'''

env = nflrush.make_env()

iter_test = env.iter_test()

(test_df, sample_prediction_df) = next(iter_test)

'''