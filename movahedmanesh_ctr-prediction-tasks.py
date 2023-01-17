! pip install xlearn
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xlearn as xl
import gc
from sklearn.datasets import dump_svmlight_file

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
base_path = '/kaggle/input/outbrain-click-prediction/'
suffix = '.csv.zip'
get_path = lambda name: base_path + name + suffix
file_names = [
    'clicks_train',
    'clicks_test',
    'events',
    'page_views_sample',
    'promoted_content',
    'sample_submission',
    'documents_entities',
    'documents_topics',
    'documents_categories',
    'documents_meta',
    
]
file_names
sample_sub = pd.read_csv(get_path(file_names[5]))


# files = []

# for fn in file_names:
#     files.append(pd.read_csv(get_path(fn)))
    

# for idx, name in enumerate(file_names, start=0):
#     print('\n')
#     print(name)
#     print(files[idx].head())
# click_train = pd.read_csv(get_path(file_names[0]))
# click_train
# event = pd.read_csv(get_path(file_names[2]))

# test_f = pd.merge(left=click_train, right=event, how='inner', on='display_id')

# test_f.head()
# test_f
# test_f.memory_usage()
# ct = files[0]
# ct.loc[ct['ad_id'] == 1]
# ev = files[2]
# ev.loc[ev['display_id'].isin([805481, 3040931])]

event = pd.read_csv(get_path(file_names[2]))
prom_cont = pd.read_csv(get_path(file_names[4]))
doc_data = []
for fn in file_names[6:]:
    doc_data.append(pd.read_csv(get_path(fn)).dropna())
NUM_OF_CHUNK = 1
CHUNK_SIZE = 50_000

for i in range(0,NUM_OF_CHUNK):
    click_train = pd.read_csv(get_path(file_names[0]), nrows=CHUNK_SIZE, skiprows=range(1, CHUNK_SIZE*(i)+1));
    print(click_train)
    clk_tr_ev = click_train.merge(right=event, on='display_id')
    clk_tr_ev_doc_data = clk_tr_ev
    for dd in doc_data:
        clk_tr_ev_doc_data = clk_tr_ev.merge(right=dd, on='document_id')
    clk_tr_ev_doc_data_ad_data = clk_tr_ev_doc_data.merge(right=prom_cont, on='ad_id')
    data_train = clk_tr_ev_doc_data_ad_data
    
    features = ['display_id',
                'ad_id','uuid',
                'document_id_x',
                'timestamp',
                'platform',
                'geo_location',
                'source_id',
                'publisher_id',
                'publish_time',
                'document_id_y',
                'campaign_id',
                'advertiser_id'
               ]
    label = 'clicked'
    
    print('joined tables')
    
    
    
    Xdf = pd.get_dummies(data_train[features], sparse=True)
    ydf = data_train[label]
    
    print('before values')
    
    X = Xdf.values
    y = ydf.values
    
    
    print('writing to file')
    dump_svmlight_file(X, y, '/kaggle/working/train'+str(i+1)+'.libsvm')
#     data_train.to_csv('/kaggle/working/click_data_train_' +str(i+1)+ '.csv')
    try:
        del data_train
        del clk_tr_ev_doc_data_ad_data
        del clk_tr_ev_doc_data
        del clk_tr_ev
        del dd
        del click_train
        del X
        del y
        del Xdf
        del ydf
        print('collected all')
    except:
        print('not collected all')
    gc.collect()

get_train_file_path = lambda i: '/kaggle/working/train'+str(i)+'.libsvm'

ffm_model = xl.create_ffm()
ffm_model.setTrain(get_train_file_path(1))
ffm_model.setValidate(get_train_file_path(2))
param = {'task':'binary', 'lr':0.2,
         'lambda':0.002, 'metric':'acc'}

ffm_model.fit(param, './kaggle/working/model.out')

# todo: data test ham ijad beshe
# Prediction task
ffm_model.setTest('UNFINISHED=> test_file_path')  # Test data
ffm_model.setSigmoid()  # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
ffm_model.predict("./kaggle/working/model.out", "./kaggle/working/output.txt")