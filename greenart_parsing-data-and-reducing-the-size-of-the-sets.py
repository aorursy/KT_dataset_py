import numpy as np

import pandas as pd



from pandas.io.json import json_normalize
specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

specs.head()
specs['args'].value_counts()
specs = specs.drop(columns='args')
specs['info'].value_counts()
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



specs['info'] = le.fit_transform(specs['info'])

specs
specs.index = specs.event_id

specs = specs.drop(columns='event_id')

specs
test_head = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv', nrows=10000)

train_head = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv', nrows=10000)
print(f'Columns in test: {test_head.shape[1]}')

print(f'Columns in train: {train_head.shape[1]}')
test_head.sample(5)
train_head.sample(5)
train_head.info()
print(f'Unique values:')



for column in train_head.columns:

    unique = train_head[column].value_counts().count()

    notNA = train_head[column].value_counts().sum()

    

    print(f'{column} : {(unique/notNA):.0%} ({unique} of {notNA}), type: {train_head[column].dtype}')

    

del column    

del unique

del notNA

del train_head

del test_head
categorical_columns = [

    'game_session', 

    'installation_id', 

    'event_code',

    'title',

    'type', 

    'world'

]



merging_cols = [

    'event_id',

]



cols_for_time_parsing = [

    'timestamp',

]



json_cols = [

    'event_data',

]



numerical_cols = [

    'event_count',

    'game_time',

]
# event_data column

import json

def cols_in_json():

    path = '/kaggle/input/data-science-bowl-2019/test.csv'

    column = 'event_data'

    size = 500000

    

    # read and transform data from train file

    file_part_json = pd.read_csv(path, usecols=[column], nrows=size)

    file_part_json = file_part_json['event_data'].apply(json.loads)

    file_part_json = json_normalize(file_part_json)



    # make a list of columns that have values in more than 30% of rows

    cols_to_save = []

    for col in file_part_json.columns:

        has_value = file_part_json[col].value_counts().sum()



        if (has_value/size > 0.3):

            print(f'{(has_value):6.0f} values - {(has_value/size):4.0%} - in column {col}')

            cols_to_save.append(col)





    #print results

    print('')

    print(f'cols_to_save ({len(cols_to_save)} columns): {cols_to_save}')

    

    return cols_to_save



# cols_to_save = cols_in_json() @making error on kaggle, but works at home, I'm trying to find out why 

cols_to_save = ['event_code', 'event_count', 'round', 'game_time', 'coordinates.x', 'coordinates.y', 'coordinates.stage_width', 'coordinates.stage_height', 'description', 'identifier', 'media_type', 'duration']
def mem_reduce(df):

    for col in df.columns:

        if df[col].dtype=='float64': 

            df[col] = df[col].astype('float32')

        if df[col].dtype=='int64': 

            if df[col].max()<1: df[col] = df[col].astype(bool)

            elif df[col].max()<128: df[col] = df[col].astype('int8')

            elif df[col].max()<32768: df[col] = df[col].astype('int16')

            else: df[col] = df[col].astype('int32')

    return df
size = 100001
def file_processing(path):

    result_set = pd.DataFrame()

    for column in range(2):



        column_name = pd.read_csv(path, usecols=[column], nrows=1).columns[0]

        if column_name in json_cols:

            result_col = pd.DataFrame(columns=cols_to_save)

        else:

            result_col = pd.DataFrame(columns=[column_name])





        #starting a loop

        skiprows = 0

        file_is_over=False

        while file_is_over == False:



            # read next part, rename the columns and concat with the result df

            file_part = pd.read_csv(path, usecols=[column], nrows=size, skiprows=skiprows)





            # json processing for event data col

            if column_name in json_cols: 

                file_part = file_part[file_part.columns[0]].apply(json.loads)

                file_part = json_normalize(file_part)



                for col in file_part.columns:

                    if col not in cols_to_save:

                        file_part = file_part.drop(columns=col)



            else:

                file_part.columns=[column_name]



            #time parsing

            if column_name in cols_for_time_parsing: 

                file_part['timestamp'] = pd.to_datetime(file_part['timestamp'], format='%Y-%m-%dT%H:%M:%S.%fZ')    





            result_col = pd.concat([result_col, file_part], sort=False)



            #iterate until the 'tail' of the file

            file_is_over = True if len(file_part) < size else False 

            skiprows += size



            result_col = mem_reduce(result_col)



            print(f'Read {(skiprows):.0f} rows of the column #{column+1} ({column_name})')





        if column_name in categorical_columns:

            result_col[column_name] = le.fit_transform(result_col.values)

            result_col[column_name] = result_col[column_name].astype('category') 



        if column_name in merging_cols:

            result_col = pd.merge(result_col, specs, on='event_id', how='left')





        result_set = pd.merge(result_set, result_col, how='right', left_index=True, right_index=True)



    return result_set
test_set = file_processing('/kaggle/input/data-science-bowl-2019/test.csv')
test_set.head()
test_set.info()
train_set = file_processing('/kaggle/input/data-science-bowl-2019/train.csv')
train_set.head()
train_set.info()