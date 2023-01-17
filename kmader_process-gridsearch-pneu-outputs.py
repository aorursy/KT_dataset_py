USER_ID = 'kevinbot'
USER_SECRET = '92eb541794f49790b15ca07e894311de'
KERNEL_LIST_PATH = '../input/kernels.csv'
import os, json, nbformat, pandas as pd
from itertools import product
import copy
from nbformat import v4 as nbf
import json
import numpy as np
from glob import glob
import shutil
kl_df = pd.read_csv(KERNEL_LIST_PATH)
kl_df.sample(3)
kaggle_conf_dir = os.path.join(os.path.expandvars('$HOME'), '.kaggle')
os.makedirs(kaggle_conf_dir, exist_ok = True)
with open(os.path.join(kaggle_conf_dir, 'kaggle.json'), 'w') as f:
    json.dump({'username': USER_ID, 'key': USER_SECRET}, f)
!chmod 600 {kaggle_conf_dir}/kaggle.json
df_to_dict = lambda in_df: {f'{k}_{s}': v for k, v_dict in 
              in_df.set_index('split').to_dict().items() 
                            for s, v in v_dict.items()}
def process_kernel(in_id):
    """
    delete existing files
    download updates
    read the first csv file
    """
    !rm *.csv
    # download kernel
    !kaggle kernels output -w -q -o -k {in_id}
    csv_files = glob('*.csv')
    
    if len(csv_files)>0:
        # process csv files
        out_dict = df_to_dict(pd.read_csv(csv_files[0]))
        # ge the execution time
        with open('kernel-log.log', 'r') as f:
            out_dict['exec_time'] = json.load(f)[-1]['time']

        for c_img_path in glob('*.pdf')+glob('*.png'):
            my_id = in_id[-4:]
            if '_gs_' not in c_img_path:
                file_name, file_ext = os.path.splitext(c_img_path)
                shutil.move(c_img_path, f'{file_name}_gs_{my_id}{file_ext}')
        return out_dict
    else:
        return dict()
from IPython.display import clear_output
results_df = kl_df.apply(lambda in_row: 
                         pd.Series(process_kernel(in_row['id'])), 1)
clear_output() # kaggle is one noisey api
all_results_df = pd.concat([kl_df, results_df], 1)
all_results_df.to_csv('results.csv',index=False)
all_results_df
all_results_df['exec_time'].hist()
all_results_df['binary_accuracy_test'].hist()
import seaborn as sns
sns.factorplot(y='binary_accuracy_test', 
               x = 'BASE_MODEL', 
               hue = 'USE_ATTN',
              data = all_results_df,
               kind='swarm',
              size = 8)
sns.factorplot(y='binary_accuracy_test', 
               x = 'BATCH_SIZE', 
               hue='BASE_MODEL',
               col = 'USE_ATTN',
              data = all_results_df,
               kind='swarm',
              size = 8)
!ls
