# Current CWD

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Installing and Updating  Dependencies

!/opt/conda/bin/python3.7 -m pip install --upgrade pip

!pip install sweetviz

!pip install pandas_profiling
# Importing Libraries

import numpy as np

import pandas as pd

import sweetviz

from pandas_profiling import ProfileReport



# Downloading Files from Kernel

from IPython.display import FileLink
# Importing Datasets

train = pd.read_csv('/kaggle/input/bad-customer-or-good-customer/x_train.csv')

test = pd.read_csv('/kaggle/input/bad-customer-or-good-customer/x_test.csv')
# Pandas Profiling Report

prof = ProfileReport(train, title = 'Train PP Report')

prof.to_widgets()
# Download Profiling Report as HTML

prof.to_file(output_file = 'Pandas Profiling Output.html')

FileLink(r'Pandas Profiling Output.html')
# Sweetviz Report - Train

my_report = sweetviz.analyze([train, "Train"], target_feat = 'target')

my_report.show_html('Train Sweetviz.html')
# Download Sweetviz Report - Train as HTML

FileLink(r'Train Sweetviz.html')
# Sweetviz Report - Train & Test Comparative

my_report1 = sweetviz.compare([train, "Train"], [test, "Test"], 'target')

my_report1.show_html('Train Test Sweetviz.html')
# Sweetviz Report - Train & Test Comparative

FileLink(r'Train Test Sweetviz.html')