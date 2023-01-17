# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print("Loading...." + check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
app_events = pd.read_csv("../input/app_events.csv", index_col=['event_id', 'app_id'])
gender_age_train = pd.read_csv("../input/gender_age_train.csv", index_col=['device_id'])
app_labels = pd.read_csv("../input/app_labels.csv", index_col=['app_id'])
label_categories = pd.read_csv("../input/label_categories.csv", index_col=['label_id'])
events = pd.read_csv("../input/events.csv", index_col=['event_id', 'device_id', 'timestamp'])
phone_brand_device_model = pd.read_csv("../input/phone_brand_device_model.csv", index_col=['device_id'])
gender_age_test = pd.read_csv("../input/gender_age_test.csv", index_col=['device_id'])
sample_submission = pd.read_csv("../input/sample_submission.csv", index_col=['device_id'])
print("Data loaded")
app_events.head(10)
events.index
