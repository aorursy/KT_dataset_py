# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#app_events.csv
#app_labels.csv
#events.csv
#gender_age_test.csv
#gender_age_train.csv
#label_categories.csv
#phone_brand_device_model.csv
#sample_submission.csv
train_gender_age = pd.read_csv('../input/gender_age_train.csv')
train_gender_age.head()
events = pd.read_csv('../input/events.csv')
events.head()
app_events = pd.read_csv('../input/app_events.csv')
app_events.head()
app_labels = pd.read_csv('../input/app_labels.csv')
app_labels.head()
label_cat = pd.read_csv('../input/label.csv')
app_labels.head()