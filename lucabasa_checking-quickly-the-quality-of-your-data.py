from os.path import join, isfile
from os import path, scandir, listdir

# standard
import pandas as pd
import numpy as np

import gc
def list_all_files(location='../input', pattern=None, recursive=True):
    subdirectories= [f.path for f in scandir(location) if f.is_dir()]
    files = [join(location, f) for f in listdir(location) if isfile(join(location, f))]
    if recursive:
        for directory in subdirectories:
            files.extend(list_all_files(directory))
    if pattern:
        files = [f for f in files if pattern in f]
    return files
list_all_files(pattern='_ACS_income')
root = '../input/cpe-data/'
dept_list = listdir(root)
dept_list
list_all_files(root + dept_list[0])
def consistency(location):
    dept_list = [d.path for d in scandir(location) if d.is_dir()]
    for dept in dept_list:
        dept_num = dept.split('_')[1]
        print('Checking department {}'.format(dept_num))
        # Check if we have some kind of data about crime or police
        crime_files = list_all_files(dept, pattern='.csv', recursive=False)
        if len(crime_files) < 1:
            print("Department {} does not have data about police interventions".format(dept_num))
        # Check the ACS data consistency
        data_path = dept + '/' + dept_num + '_ACS_data/'
        # Check if we have all the topics (poverty, education, etc)
        topics = [d.path for d in scandir(data_path) if d.is_dir()]
        if len(topics) < 7:
            print('Department {} does not have all the 7 ACS categories'.format(dept_num))
        # Check if the data have consistent id's and columns
        files = list_all_files(data_path, pattern='_with_ann.csv')
        ids = []
        for file in files:
            meta = file.replace('_with_ann.csv', '_metadata.csv')
            data = pd.read_csv(file, skiprows=[1], low_memory=False)
            metadata = pd.read_csv(meta, header=None, names=['key', 'description'])
            if not data.columns.all() in list(metadata['key']):
                print("In {} inconsistent metadata".format(file))
            tmp_ids = data['GEO.id2'].unique()
            if len(tmp_ids) != data.shape[0]:
                print("In {} inconsistent id's".format(file))
            if len(ids) < 1: # the first time it creates the "base" of id's
                ids = tmp_ids
            if set(tmp_ids) != set(ids):
                print("In {} inconsistent id's with the other files".format(file))
    print("Done")
consistency(root)
def import_topic(path, tolerance=0.7):
    # find the file with the ACS data and load it
    datafile = list_all_files(path, pattern='_with_ann.csv')[0]
    data = pd.read_csv(datafile, skiprows=[1], low_memory=False)
    # take out the ids momentarily
    ids = data[[col for col in data.columns if 'GEO' in col]]
    rest = data[[col for col in data.columns if 'GEO' not in col]]
    # convert to numeric and force na's if necessary
    rest = rest.apply(pd.to_numeric, errors='coerce')
    # put data together again
    data = ids.join(rest)
    print('Shape: {}'.format(data.shape))
    cols = data.columns
    nrows = data.shape[0]
    removed = 0
    for col in cols:
        mis = data[col].isnull().sum() / nrows
        if mis > tolerance:
            removed += 1
            del data[col]
    if removed > 0:
        print("Removed {} columns because more than {}% of the values are missing".format(removed, 
                                                                                      tolerance*100))
        print("New shape: {}".format(data.shape))
    meta = datafile.replace('_with_ann.csv', '_metadata.csv')
    metadata = pd.read_csv(meta, header=None, names=['key', 'description'])
    return data, metadata


def import_dept(location):
    dept_num = location.split('_')[1]
    print('Importing department {}'.format(dept_num))
    print('\n')
    data_path = location + '/' + dept_num + '_ACS_data/'
    data_list = {}
    topics = listdir(data_path)
    for topic in topics:
        topic_name = topic.split('_')[-1]
        print('Importing {}'.format(topic_name))
        data, meta = import_topic(data_path + topic)
        data_list[topic_name] = data
        data_list[topic_name + '_meta'] = meta
    gc.collect() # in case some of the files were really big
    return data_list
test = root + dept_list[4]
print(test)
print("_"*40)
print('\n')
result = import_dept(test)
result['poverty'].sample(10)
result['poverty_meta'].sample(10)
result.keys()