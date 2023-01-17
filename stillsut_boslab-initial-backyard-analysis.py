import os, copy

import numpy as np 

import pandas as pd

from matplotlib import pyplot as plt



#vanity import - kernels already come with this package(?)

from Bio import Seq, SeqIO



data_repo = '../input'

print(os.listdir(data_repo))



data_path = os.path.join(data_repo, 'data', 'data')

print(os.listdir(data_path))
data_analysis_path = os.path.join(data_path, 'analysisfiles')



dir_analysis_path = os.listdir(data_analysis_path)



#eliminate xlsx files

files_analysis_path = [_fn for _fn in dir_analysis_path if '.xlsx' not in _fn]



#eliminate sub-directories

files_analysis_path = [_fn for _fn in files_analysis_path 

                       if not(os.path.isdir(os.path.join(data_analysis_path, _fn)))

                      ]



files_analysis_path[:4]
#shorten the filenames

fn0 = files_analysis_path[0]



split_chars = 'fa.'

base_fn_chars, *tail_ = fn0.split(split_chars)

base_fn_chars += split_chars

print('base_fn_chars: ', base_fn_chars)



curt_analysis_files = [_fn.split(base_fn_chars)[1] for _fn in files_analysis_path]

curt_analysis_files[:4]
# load all datasets into a dict

dict_analysis_tables = {}



for curt_fn in curt_analysis_files:

    

    data_fn = os.path.join(data_analysis_path, (base_fn_chars + curt_fn) )

    

    _df = pd.read_csv(data_fn, delimiter='\t')

    _df = _df.dropna(axis=1)

    

    dict_analysis_tables[curt_fn] = _df.copy()
# show a preview for each table

for curt_fn, data_tbl in dict_analysis_tables.items():

    print(curt_fn)

    display(data_tbl[:3])

    
#TODO 

# [x] clean Unnamed:11 from tables

# [x] remove percentages tables

# [x] filter for sample_record-type tables

# [ ] build a composite table
# build base: concise container with most standardized data

base_tbls = {}



for curt_fn, data_tbl in dict_analysis_tables.items():

    

    tbl_name = curt_fn.split('.')[0]

    

    # eliminate percentages-tbl's and non-standard tables.

    # identified as 1st column name not equal to file name:

    # so, OTU + mapping for this set of tables

    if (('percent' in curt_fn) 

        or ( list(data_tbl.columns)[0] !=  tbl_name) ):

        continue

        

    base_tbls[tbl_name] = data_tbl.copy()

    

print(list(base_tbls.keys()))
# summarize features

_tmp = {}

for tbl_name, tbl_data in base_tbls.items():

    

    _n = len(tbl_data)

    _s_records = ' - '.join([str(r)for r in list(tbl_data[tbl_name])])

    

    _tmp[tbl_name] = [_n, _s_records]

    

features_summary_base = pd.DataFrame(_tmp)
# format the output: 



# flip and rename

features_summary = features_summary_base.rename(index={0:'n', 1:'records'},inplace=False)



features_summary = features_summary.T



# sort descending

features_summary.sort_values(by=['n'], inplace=True)



# control the table size for large string display

# 500 is a reasonable preview size; -1 : expand full width, is clunky

# pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_colwidth', 500)



features_summary
# https://medium.com/sfu-big-data/advanced-visualization-for-data-scientists-with-matplotlib-15c28863c41c
tbl_name = 'phylum' 

# sample_name = 'L1'

sample_name = 'SH3'



def pie_from_base(tbl_name, sample_name):



    df_tbl = base_tbls[tbl_name]



    tmp_df = df_tbl.set_index(keys=[tbl_name],inplace=False)



    vec_dict = tmp_df[sample_name].to_dict()



    pie_data = [(k,v) for k,v in vec_dict.items()]

    pie_names = [kv[0] for kv in pie_data]

    pie_vals = [kv[1] for kv in pie_data]



    _ = plt.pie(x=pie_vals,labels=pie_names,)

    plt.title(tbl_name + ' - ' + sample_name)

    plt.show()

    

pie_from_base(tbl_name, sample_name)
# 'L1', 'L2', 'L3', 'L4', 'M1', 'M2', 'SH3', 'SH4', 'SH5','SH6'

tbl_name = 'phylum'

samples = [ 'L3', 'L4', 'M1', 'SH3']

for _sample in samples:

    pie_from_base(tbl_name, _sample)