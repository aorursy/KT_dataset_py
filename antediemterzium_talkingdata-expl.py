# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
app_events = pd.read_csv("../input/app_events.csv", dtype={'device_id': np.str})
app_labels = pd.read_csv("../input/app_labels.csv", dtype={'device_id': np.str})
events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})
train = pd.read_csv("../input/gender_age_train.csv", dtype={'device_id': np.str})
test = pd.read_csv("../input/gender_age_test.csv", dtype={'device_id': np.str})
label_categories = pd.read_csv("../input/label_categories.csv", dtype={'device_id': np.str})
sample = pd.read_csv("../input/sample_submission.csv", dtype={'device_id': np.str})
phone_brand = pd.read_csv("../input/phone_brand_device_model.csv", dtype={'device_id': np.unicode})

app_events.head()
app_labels.head()
events.head()
train.head()
test.head()
label_categories.head()
phone_brand[:20]
#phone_brand.drop_duplicates('device_id', keep='first', inplace=True)
train_df = pd.merge(train, events,how="right",on="device_id")
train_df = pd.merge(train_df, app_events,how="right",on="event_id")
train_df[:20]
train_df.shape
test.shape
train.shape
