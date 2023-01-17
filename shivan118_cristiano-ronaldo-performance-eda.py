# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = '/kaggle/input/yds_data.csv'
data = pd.read_csv(path) # reading data from the CSV file

data.head()
data.shape # checking how many rows and columns are in the data
data.columns
# Using descriptive Statistics to find some insights

data.describe()
# Finding the dtypes of Columns to get some Insights

data.info()
# Percentage and Sum of Missing values in each Columns

missing_data = pd.DataFrame({'total_missing': data.isnull().sum(), 'perc_missing': (data.isnull().sum()/data.shape[0])*100})

missing_data
# Exploring The Target Variable 'is_goal'

data.is_goal.value_counts()
#1. Droping Unnecessary Columns

data.drop(["Unnamed: 0",  'remaining_min.1', 'power_of_shot.1','knockout_match.1', 'remaining_sec.1', 'distance_of_shot.1'], axis=1, inplace=True)
data.head() # looking at the dataset after transformation
data.columns # to see if the columns are dropped succesfully
#2. Changing dtypes to datetime

data.date_of_game = pd.to_datetime(data.date_of_game, errors='coerce')

data['game_season'] = data['game_season'].astype('object')

data['game_season']
# Labelencoding the 'game_season' 
l_unique = data['game_season'].unique() # fteching out the unique values from game_season/

l_unique
v_unique = np.arange(len(l_unique)) # obtaining values in the range of the length of I_unique

v_unique
data['game_season'].replace(to_replace=l_unique, value=v_unique, inplace=True) # replacing categorical data with numerical values

data['game_season'].head()
data['game_season'] = data['game_season'].astype('int') # converting the datatype of the column from int64 to int32

data['game_season'].head()
# Filling NaN values in Column "remaining_sec" with MEAN

data['power_of_shot'].fillna(value=data['power_of_shot'].mean(), inplace=True)

data.isnull().sum() # number of missing values for power_of_shot column should be zero

# Filling NaN values in Column "type_of_combined_shot" with MODE

mode_com  = data.type_of_combined_shot.value_counts().keys()[0]

print('moded is: ',mode_com)

data.type_of_combined_shot.fillna(value=mode_com, inplace=True)

data.isnull().sum() # number of missing values for type_of_combined_shot column should be zero
# Filling NaN values in Column "remaining_sec" with MEDIAN

data.remaining_sec.fillna(value=data.remaining_sec.median(), inplace=True)

data.isnull().sum() # number of missing values for remaining_sec column should be zero
# Shot_id_no.

data.shot_id_number = pd.Series(np.arange(1,data.shot_id_number.shape[0]+1))

data.isnull().sum() # number of missing values for shot_id_number column should be zero
# Filling NaN values in Columns "location_x" and "location_y" with 0

data['location_x'].fillna(value=0, inplace=True)

data['location_y'].fillna(value=0, inplace=True)

data.isnull().sum() # number of missing values for location_x and location_y columns should be zero
# Using Forward Filling method in appropriate Columns

print('Null values in column home/away before forward fill =',data['home/away'].isnull().sum())

col = ['home/away','lat/lng', 'team_name','match_id','match_event_id', 'team_id', 'remaining_min', 'knockout_match',  'game_season' ]

data.loc[:,col] = data.loc[:,col].ffill()

print('Null values in column home/away after the forward fill =',data['home/away'].isnull().sum())
# Filling Missing Values In "shot_basics" based on "range_of_short" column!

# if the range of the shot is 16-24 ft it's a mid range shot



data.loc[(data.range_of_shot == '16-24 ft.'), 'shot_basics'] = data[data.range_of_shot == '16-24 ft.'].shot_basics.fillna(value='Mid Range')



# if the range of the shot is less than 8 ft then randomly assign goal line or goal area value to the shot 



data.loc[(data.range_of_shot == 'Less Than 8 ft.')&(data.shot_basics.isnull()), 'shot_basics']   =  pd.Series(data[(data.range_of_shot == 'Less Than 8 ft.')&(data.shot_basics.isnull())].shot_basics.apply(lambda x: x if type(x)==str else np.random.choice(['Goal Area', 'Goal Line'],1,p=[0.7590347263095939, 0.24096527369040613])[0]))



# if the range of the shot is  8-16 ft then randomly assign goal line or mid range value to the shot



data.loc[(data.range_of_shot == '8-16 ft.')&(data.shot_basics.isnull()), 'shot_basics']          =  pd.Series(data[(data.range_of_shot == '8-16 ft.')&(data.shot_basics.isnull())].shot_basics.apply(lambda x: x if type(x)==str else np.random.choice(['Mid Range', 'Goal Line'],1,p=[0.6488754615642833, 0.35112453843571667])[0]))



# if the range of the shot is more than 24 ft then randomly assign one of the values from'Penalty Spot', 'Right Corner', 'Left Corner' to shot_basic field



data.loc[(data.range_of_shot == '24+ ft.')&(data.shot_basics.isnull()), 'shot_basics']            =  pd.Series(data[(data.range_of_shot == '24+ ft.')&(data.shot_basics.isnull())].shot_basics.apply(lambda x: x if type(x)==str else np.random.choice(['Penalty Spot', 'Right Corner', 'Left Corner'],1,p=[0.8932384341637011, 0.06192170818505338, 0.044839857651245554])[0]))



# if the shot is a back court shot then randomly assign one of the values from''Mid Ground Line', 'Penalty Spot' to shot_basic field



data.loc[(data.range_of_shot == 'Back Court Shot')&(data.shot_basics.isnull()), 'shot_basics']    =  pd.Series(data[(data.range_of_shot == 'Back Court Shot')&(data.shot_basics.isnull())].shot_basics.apply(lambda x: x if type(x)==str else np.random.choice(['Mid Ground Line', 'Penalty Spot'],1,p=[0.8441558441558441, 0.15584415584415584])[0]))

data.isna().sum()
data['shot_basics'].unique() # now we have populated the shot types and reduced the number of missing values. Earlier we had 1575 missing values for this column, now we have only 66.
# Filling Missing Values In "range_of_short" based on "short_basics" column!



# if shot_basics is Goal Area, then range of shot is Less Than 8 ft



data.loc[(data.shot_basics == 'Goal Area'), 'range_of_shot']       = data[data.shot_basics == 'Goal Area'].range_of_shot.fillna(value='Less Than 8 ft.')

# if shot_basics is Penalty Spot, then range of shot is  24+ ft.



data.loc[(data.shot_basics == 'Penalty Spot'), 'range_of_shot']    = data[data.shot_basics == 'Penalty Spot'].range_of_shot.fillna(value= '24+ ft.')

# if shot_basics is Right Corner, then range of shot is  24+ ft.



data.loc[(data.shot_basics == 'Right Corner'), 'range_of_shot']    = data[data.shot_basics == 'Right Corner'].range_of_shot.fillna(value='24+ ft.')

# if shot_basics is Left Corner, then range of shot is  24+ ft.



data.loc[(data.shot_basics == 'Left Corner'), 'range_of_shot']     = data[data.shot_basics == 'Left Corner'].range_of_shot.fillna(value='24+ ft.')

# if shot_basics is Mid Ground Line , then range of shot is  Back Court Shot



data.loc[(data.shot_basics == 'Mid Ground Line'), 'range_of_shot'] = data[data.shot_basics == 'Mid Ground Line'].range_of_shot.fillna(value='Back Court Shot')

# if shot_basics is Mid Range then randomly assign '16-24 ft.' or  '8-16 ft.' to range of shot



data.loc[(data.shot_basics == 'Mid Range')&(data.range_of_shot.isnull()), 'range_of_shot']       = pd.Series(data[(data.shot_basics == 'Mid Range')&(data.range_of_shot.isnull())].range_of_shot.apply(lambda x: x if type(x)==str else np.random.choice(['16-24 ft.', '8-16 ft.'],1,p=[0.6527708850289495, 0.34722911497105047])[0]))

# if shot_basics is Goal Line then randomly assign ''8-16 ft.' or  'Less Than 8 ft.' to range of shot



data.loc[(data.shot_basics == 'Goal Line')&(data.range_of_shot.isnull()), 'range_of_shot']       = pd.Series(data[(data.shot_basics == 'Goal Line')&(data.range_of_shot.isnull())].range_of_shot.apply(lambda x: x if type(x)==str else np.random.choice(['8-16 ft.', 'Less Than 8 ft.'],1,p=[0.5054360956752839, 0.49456390432471614])[0]))



data.isnull().sum() # number of missing values for range_of_shot column should have been reduced
data['range_of_shot'].unique() # the number of missing values has fallen from 1564 to 66
# Filling the remaining missing values incase they both have NaN values using the forward fill method

data.shot_basics.fillna(method='ffill', inplace=True)

data.range_of_shot.fillna(method='ffill', inplace=True)

data.isnull().sum() # number of missing values for shot_basics and range_of_shot columns should be zero
# Filling the missing value in "ärea_of_short" Column

data.area_of_shot.fillna(value='Center(C)', inplace=True) # all the missing values get filled by  'Centre(C)'

data.isnull().sum() # number of missing values for area_of_shot column should be zero
data['distance_of_shot'].unique()
#Filling the Missing values in "distance_of_shot"

# if distance_of_shot isnull randomly assign a value from 20,45,44,37

data.loc[data['distance_of_shot'].isnull(), 'distance_of_shot'] = pd.Series(data.loc[data['distance_of_shot'].isnull(), 'distance_of_shot'].apply(lambda x: x if type(x)==str else np.random.choice([20,45,44,37],1,p=[0.5278056615137523,0.18630797028709095,0.14384661714515157,0.1420397510540052])[0])) 

data.isnull().sum() # number of missing values for distance_of_shot column should be zero


# Making the train Dataset

train = data[data.is_goal.notnull()]

print('the Shape of Train Dataset',train.shape)

train.set_index(np.arange(train.shape[0]),inplace=True)

train.head()

# Making the Test Dataset

test = data[data.is_goal.isnull()]

print('The Shape of Test Dataset',test.shape)

test.set_index(np.arange(test.shape[0]), inplace=True)

test.head()
l_goal   = train[train.is_goal == 1].type_of_shot.value_counts().head(6).keys()     # Top six shots when it was goal

l_goal
p_g_sum  = train[train.is_goal == 1].type_of_shot.value_counts().head(6).sum() # Top six shots when it was goal

p_goal   = (train[train.is_goal == 1].type_of_shot.value_counts().head(6) / p_g_sum ).tolist()  # There respective probablities

p_goal
# if is_goal is 1, if type of shot is a string value, fill with the same or else fill with randomly choosing value from l_goal

g = pd.Series(train[train.is_goal == 1].type_of_shot.apply(lambda x: x if type(x)==str else np.random.choice(l_goal,1,p=p_goal)[0]))

g
# # if is_goal is 1, if type of shot is null then type of shot becomes equal to the value of g based on the index

train.loc[(train.is_goal == 1)&(train.type_of_shot.isnull()), 'type_of_shot'] = g
train['type_of_shot'].isna().sum() # number of missing values got reduced from more than 15k to 6723
l_no_goal   = train[train.is_goal == 0].type_of_shot.value_counts().head(5).keys()     # Top five shots when it was not a goal

p_no_sum  = train[train.is_goal == 0].type_of_shot.value_counts().head(5).sum()

p_no_goal   = (train[train.is_goal == 0].type_of_shot.value_counts().head(5) / p_no_sum ).tolist() # There respective probablities 

ng = pd.Series(train[train.is_goal == 0].type_of_shot.apply(lambda x: x if type(x)==str else np.random.choice(l_no_goal,1,p=p_no_goal)[0]))

train.loc[(train.is_goal == 0)&(train.type_of_shot.isnull()), 'type_of_shot'] = ng 

train['type_of_shot'].isna().sum() # number of missing values got reduced to zero
#Handeling the remaing values in test dataset with a smilira approach

test.loc[test['type_of_shot'].isnull(), 

         'type_of_shot'] = pd.Series(test.loc[test['type_of_shot'].isnull(), 

                                              'type_of_shot'].apply(lambda x: x if type(x)==str else np.random.choice(['shot - 39', 'shot - 36', 'shot - 4'],1,p=[0.37377133988618727, 0.33419555095706155, 0.2920331091567512])[0])) 

test['type_of_shot'].isna().sum() # we have removed the missing values from test set as well
%%time

# Labeling the catagories with integers

for col in train.columns:

    if train[col].dtypes == object: # if the column has categorical values

        l_unique = train[col].unique() # find the unique values

        v_unique = np.arange(len(l_unique)) # create a list of number from zero to the length of the I_unique values

        train[col].replace(to_replace=l_unique, value=v_unique, inplace=True) # replace the categorical values with numerical values

        train[col] = train[col].astype('int') # change the type from int64 to int32

        

        # same has been done for test data as well

        test[col].replace(to_replace=l_unique, value=v_unique, inplace=True)

        test[col] = test[col].astype('int')

        
# Dropping the unnecessary Columns

train.drop(['date_of_game'], axis=1, inplace=True)

train.head()
test.drop(['date_of_game'], axis=1, inplace=True)

test.head()
# Splliting the Target Column from the Dataset

y = train.is_goal

y.head()
train.drop(['is_goal'], axis=1, inplace=True)

train.head()

test.drop(['is_goal'], axis=1, inplace=True)

test.head()
train.info() # we have converted all the categorical columns to numeric ones
train.isna().sum() # we have don't have any missing values as well. Our data is ready to be fed to a machine learning model.