# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib

%matplotlib inline  



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# with open('../input/stack-overflow-developer-survey-results-2019/README_2019.txt') as f:

#     print(f.read())



pd.options.display.max_columns = None

pd.options.display.max_rows = None



    

data = pd.read_csv('../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv')

schema = pd.read_csv('../input/stack-overflow-developer-survey-results-2019/survey_results_schema.csv')
list(data.columns)
data["OpenSourcer"]