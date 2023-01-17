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
import pandas as pd

import numpy as np

import seaborn as sns

from tqdm import tqdm

from collections import Counter

import numpy as np

import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')
# reference : https://www.kaggle.com/erikbruin/data-science-bowl-2019-eda-and-baseline

# In the reference blog they have used two separate function for featurization. I slightly modified and compressed it in only one function.

def get_features(installation_id , dataframe_for_an_id , test_flag=False):

    '''

    

    This function will calculate features for train and test data. 

    It will create 4 columns for four unique world(including None) and

    will create 44 columns for 44 unique title and

    will create 4 columns for four unique type and

    will create 42 columns for 42 unique event_code and

    will create 6 more columns for 'total_duration','total_action','correct_count','incorrect_count','accuracy','accuracy_group'

               ---

        total  100 columns 

    

    except total_duration, accuracy and accuracy_group all other features is number of counts of those feature in a game_session

    if test_flag is True then return last entry of list 

    '''

    # temp_dict initialized with keys (100 columns) and value = 0

    features = []

    features.extend(list(set(train['world'].unique()).union(set(test['world'].unique())))) # all unique worlds in train and test data

    features.extend(list(set(train['title'].unique()).union(set(test['title'].unique())))) # all unique title in train and test data

    features.extend(list(set(train['type'].unique()).union(set(test['type'].unique())))) # all unique type in train and test data

    features.extend(list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))) # all unique event_code in train and test data 

    features.extend(['total_duration','total_action','correct_count','incorrect_count','accuracy','accuracy_group'])

    temp_dict = dict.fromkeys(features,0)

    list_of_features = []

    

    

    def get_all_attempt(sample_df):

        '''

        This fuction will return the dataframe which is used to calculate accuracy_group

        '''

        if sample_df['title'].iloc[0] != 'Bird Measurer (Assessment)':

            all_attempt = sample_df.query('event_code == 4100')

        elif sample_df['title'].iloc[0] == 'Bird Measurer (Assessment)':

            all_attempt = sample_df.query('event_code == 4110')

        return all_attempt

    

    for i, sample_df in dataframe_for_an_id.groupby(by = ['game_session']):

        # sample_df is groupby object 

        # In sample_df 'type','title' and 'world' will not change so first entry of those column is piced

        temp_dict['installation_id'] = installation_id

        temp_type = sample_df['type'].iloc[0]

        temp_title = sample_df['title'].iloc[0]

        temp_world = sample_df['world'].iloc[0]

        temp_event_code = Counter(sample_df['event_code'].values)



        session_size = len(sample_df)



        temp_dict[temp_type]+=session_size

        temp_dict[temp_title]+=session_size    # corresponding type , title and world is incremented by session size

        temp_dict[temp_world]+=session_size                 



        for code, code_count in temp_event_code.items():    # corresponding event_code is incremented

            temp_dict[code]+= code_count                  

            

        duration_in_sec = float(sample_df['game_time'].iloc[-1])/1000   # total_duration is duration of game_session in seconds

        temp_dict['total_duration'] += duration_in_sec

        

        action_count_in_game_session = session_size     # total number of action performed in game_session

        temp_dict['total_action'] += action_count_in_game_session  

        

        isAssessment = temp_type == 'Assessment'

        isBirdMeasureAssessment = isAssessment and temp_title == 'Bird Measurer (Assessment)'

        isAssessment_with_code4110 = isBirdMeasureAssessment and 4110 in list(sample_df['event_code'])

        isNonBirdMeasureAssessment = isAssessment and temp_title != 'Bird Measurer (Assessment)'

        isAssessment_with_code4100 = isNonBirdMeasureAssessment and 4100 in list(sample_df['event_code'])

        

        criterion_to_accuracy_group = isAssessment_with_code4110 or isAssessment_with_code4100 

        

        

        if test_flag and isAssessment and (criterion_to_accuracy_group == False):

            temp_dict['accuracy'] = 0           # there are lots of installation_id in test data that attempt for

            temp_dict['accuracy_group'] = 0     # Assessment but not with event_code 4100 or 4110

            list_of_features.append(temp_dict)  # So I assumed those id belongs to class 0

            

            

        if criterion_to_accuracy_group == False:

            continue

        

        # below section is only performed when criterion_to_accuracy_group is True

        

        all_attempt = get_all_attempt(sample_df)

        correct_count = all_attempt['event_data'].str.contains('true').sum()     

        incorrect_count = all_attempt['event_data'].str.contains('false').sum()

        temp_dict['correct_count'] = correct_count  

        temp_dict['incorrect_count'] = incorrect_count



        if correct_count == 0 and incorrect_count == 0:

            temp_dict['accuracy'] = 0

        else:

            temp_dict['accuracy'] = correct_count/(correct_count + incorrect_count)



        if temp_dict['accuracy']==1:

            temp_dict['accuracy_group']=3

        elif temp_dict['accuracy']==0.5:

            temp_dict['accuracy_group']=2

        elif temp_dict['accuracy']==0:

            temp_dict['accuracy_group']=0

        else :

            temp_dict['accuracy_group']=1



        list_of_features.append(temp_dict)

        temp_dict = dict.fromkeys(features,0)

        

        

    if test_flag:                    # If given data is from test data then return only the last entry of the list

        return list_of_features[-1]

    

    return list_of_features
sample_df = train[train.installation_id == "0006a69f"]

list_of_feature = get_features('0006a69f',sample_df,True)

list_of_feature

temp_df = pd.DataFrame(list_of_feature,index=[0])

temp_df
from tqdm.notebook import tqdm

final_test_data_list = []

test_groupby_id = test.groupby(by=['installation_id'])

for installation_id , df_with_unique_id in tqdm(test_groupby_id):

    final_test_data_list.append(get_features(installation_id,df_with_unique_id,True))

final_test_data = pd.DataFrame(final_test_data_list)
from random import seed

from random import randint



seed(1)

random_int = []

for _ in range(1000):

    value = randint(0,3)

    random_int.append(value)
my_submission = pd.DataFrame({'installation_id':final_test_data['installation_id'],'accuracy_group':random_int})

my_submission.to_csv('submission.csv',index =False)