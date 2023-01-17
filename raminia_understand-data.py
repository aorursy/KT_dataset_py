# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib 
import matplotlib.pyplot as plt
%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

df_train =  pd.read_csv("../input/gender_age_train.csv") # the gender dataset is now a Pandas DataFrame
df_app_event =  pd.read_csv("../input/app_events.csv") 
df_app_labels =  pd.read_csv("../input/app_labels.csv")
df_test =  pd.read_csv("../input/gender_age_test.csv") 
df_label_cat =  pd.read_csv("../input/label_categories.csv")
df_phone =  pd.read_csv("../input/phone_brand_device_model.csv")
df_submission =  pd.read_csv("../input/sample_submission.csv")
df_train =  pd.read_csv("../input/gender_age_train.csv")


df_train.head()