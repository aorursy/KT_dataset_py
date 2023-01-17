import pandas as pd

import numpy as np

import os

from tqdm import tqdm_notebook



import zipfile

from subprocess import check_output
print(check_output(["ls", "../input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2"]).decode("utf8"))
# This will not working because of limitation

# zip_path = '../input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/train.zip'

# with zipfile.ZipFile(zip_path,"r") as z:

#     z.extractall('')
# Getting list of files in zip file

zip_path = '../input/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2/train.zip'

with zipfile.ZipFile(zip_path,"r") as z:

    file_list = z.namelist()



# #Filtering csv files

# file_list = [file_name for file_name in file_list if '.csv' in file_name ]

# print(file_list[:10]) 
#Extract Alice log

with zipfile.ZipFile(zip_path,"r") as z:

    z.extract('train/Alice_log.csv')

    

#Extract first 100 files of other users

LIMIT = 100

step = 0

with zipfile.ZipFile(zip_path,"r") as z:

    for file in z.namelist():        

        if file.startswith('train/other_user_logs/'):

            z.extract(file)

            step += 1

        if step == LIMIT:

            break

            

            
print(check_output(["ls", "train"]).decode("utf8"))
# If you run this notebok locally and you have already unzipped folder manually, specify your paths here

alice_path = 'train/Alice_log.csv'

other_user_path = 'train/other_user_logs'
def prepare_data(path, target):

    raw_df = pd.read_csv(path)

    

    site_names = ['site{}'.format(i) for i in range(1, 11)]

    time_names = ['time{}'.format(i) for i in range(1, 11)]

    feature_names = [None] * (2 * len(site_names))

    feature_names[1::2] = time_names

    feature_names[0::2] = site_names

    

    # prepare 30-min steps

    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])

    time_diff = raw_df['timestamp'].diff().astype(int)

    time_diff = np.where(time_diff < 0, np.nan, time_diff)

    raw_df['min_diff'] = time_diff/(1e9*60) # nanoseconds to minutes 

    raw_df['min_diff'].fillna(0, inplace = True)

    raw_df['min_cumsum'] = raw_df['min_diff'].cumsum()

    raw_df['step'] = (raw_df['min_cumsum']//30).astype(int)

    

    step_list = raw_df['step'].unique()

    

    stacking_list = []

    for step in step_list:

        temp_part = raw_df[raw_df['step'] == step][['timestamp', 'site']].to_numpy()

        

        # infill matrix by NaN`s

        if temp_part.shape[0]%10 != 0:

            temp_padding = np.full((10 - temp_part.shape[0]%10, 2), np.nan)

            temp_part = np.vstack([temp_part, temp_padding])



        # https://stackoverflow.com/questions/3678869/pythonic-way-to-combine-two-lists-in-an-alternating-fashion

        temp_combine = [None] * (2 * len(temp_part))

        temp_combine[::2] = temp_part[:, 1]

        temp_combine[1::2] = temp_part[:, 0]



        temp_result = np.array(temp_combine).reshape((-1, 20))

        stacking_list.append(temp_result)



    data = np.vstack(stacking_list)

    

    df = pd.DataFrame(data, columns = feature_names)

    df['target'] = np.full(df.shape[0], target)

    

    return df  
%%time

alice_df = prepare_data(alice_path, target = 1)
files_list = sorted([file for file in os.listdir(other_user_path) if 'csv' in file])

other_users_df = pd.DataFrame(columns = alice_df.columns)





for file_name in tqdm_notebook(files_list):

    temp_df = prepare_data(os.path.join(other_user_path, file_name), target = 0)

    other_users_df = pd.concat([other_users_df, temp_df])

    

other_users_df.reset_index(drop = True, inplace = True)

    
result_df = pd.concat([alice_df, other_users_df]).sort_values('time1')



result_df.reset_index(drop = True, inplace = True)

result_df['session_id'] = result_df.index.to_numpy() + 1
#reorder columns

columns = result_df.columns.to_list()

result_df = result_df[['session_id'] + columns[:-1]]

result_df.head()
# result_df.to_csv('Data/additional_train_data.csv')