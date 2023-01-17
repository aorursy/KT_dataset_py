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
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
from typing import Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.offline


# read data
in_kaggle = True

# base report output path
reports_folder = 'reports/'

def get_data_file_path(is_in_kaggle: bool) -> Tuple[str, str, str, str, str]:
    train_path = ''
    test_path = ''

    if is_in_kaggle:
        # running in Kaggle, inside the competition
        train_path = '../input/lish-moa/train_features.csv'
        train_targets_path = '../input/lish-moa/train_targets_scored.csv'
        train_targets_nonscored_path = '../input/lish-moa/train_targets_nonscored.csv'
        test_path = '../input/lish-moa/test_features.csv'
        sample_submission_path = '../input/lish-moa/sample_submission.csv'
    else:
        # running locally
        train_path = 'data/train_features.csv'
        train_targets_path = 'data/train_targets_scored.csv'
        train_targets_nonscored_path = 'data/train_targets_nonscored.csv'
        test_path = 'data/test_features.csv'
        sample_submission_path = 'data/sample_submission.csv'

    return train_path, train_targets_path, train_targets_nonscored_path, test_path, sample_submission_path
    

# Import data
train_set_path, train_set_targets_path, train_set_targets_nonscored_path, test_set_path, sample_subm_path = get_data_file_path(in_kaggle)


a = pd.read_csv(train_set_path)
b = pd.read_csv(test_set_path)
c = pd.read_csv(train_set_targets_nonscored_path)
d = pd.read_csv(train_set_targets_path)

print(a.shape,b.shape,c.shape,d.shape)
merged = pd.concat([a,b])

# Datasets for treated and control experiments
treated = a[a['cp_type'] == 'trt_cp']
control = a[a['cp_type'] == 'ctl_vehicle']

# Treatment time datasets
cp24 = a[a['cp_time']== 24]
cp48 = a[a['cp_time']== 48]
cp72 = a[a['cp_time']== 72]

# Merge scored and nonscored labels
all_drugs = pd.merge(d, c, on='sig_id', how='inner')

# Treated drugs without control
treated_list = treated['sig_id'].to_list()
drugs_tr = d[d['sig_id'].isin(treated_list)]

# Non-treated control observations
control_list = control['sig_id'].to_list()
drugs_cntr = d[d['sig_id'].isin(control_list)]

# Treated drugs:
nonscored = c[c['sig_id'].isin(treated_list)]
scored = d[d['sig_id'].isin(treated_list)]

# adt = All Drugs Treated
adt = all_drugs[all_drugs['sig_id'].isin(treated_list)]

# Select the columns c-
c_cols = [col for col in a.columns if 'c-' in col]

# Filter the columns c-
cells_tr = treated[c_cols]
cells_cntr = control[c_cols]


# Select the columns g-
g_cols = [col for col in a.columns if 'g-' in col]

# Filter the columns g-
genes_tr = treated[g_cols]
genes_cntr = control[g_cols]

labels_in_training = pd.merge(a, d, on='sig_id', how='inner')
treated_labels_in_training = labels_in_training[labels_in_training['sig_id'].isin(treated_list)]

target_label = 'nfkb_inhibitor'
cols_to_research = c_cols.copy()
cols_to_research.append(target_label)

final_training_df = treated_labels_in_training[cols_to_research]
!pip install autoviz 
final_training_df.dtypes
from autoviz.AutoViz_Class import AutoViz_Class
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
dft = AV.AutoViz(filename='', sep='' , depVar='nfkb_inhibitor', dfte=final_training_df, header=0, verbose=2, lowess=False, 
                 chart_format='svg', max_rows_analyzed=25000, max_cols_analyzed=40)

