# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas_profiling
import seaborn as sns
live_raw = pd.read_csv("/kaggle/input/Restaurant_Scores_-_LIVES_Standard.csv")
# Use the necessary columns:
necessary_list = ['business_id','business_name','inspection_id','inspection_date','inspection_score','inspection_type','violation_id','violation_description','risk_category']
live = live_raw.loc[:,necessary_list]
#Drop the missing value for inspection_score:
live = live.dropna()
live.head(10)
profile = pandas_profiling.ProfileReport(live, title='Pandas Profiling Report', html={'style':{'full_width':True}})
profile.to_widgets()
# low_risk = live[live['risk_category']=="Low Risk"]
# moderate_risk = live[live['risk_category']=="Moderate Risk"]
# high_risk = live[live['risk_category']=="High Risk"]

ax = sns.boxenplot(x="risk_category", y="inspection_score", data=live)
