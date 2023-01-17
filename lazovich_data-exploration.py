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
app_events = pd.read_csv("../input/app_events.csv")
app_labels = pd.read_csv("../input/app_labels.csv")
events = pd.read_csv("../input/events.csv")
gender_age_train = pd.read_csv("../input/gender_age_train.csv")
gender_age_test = pd.read_csv("../input/gender_age_test.csv")
label_categories = pd.read_csv("../input/label_categories.csv")
phone_brand_device_model = pd.read_csv("../input/phone_brand_device_model.csv")
submit = pd.read_csv("../input/sample_submission.csv")


app_labels = app_labels.set_index('app_id')
label_categories = label_categories.set_index('label_id')
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

label_enc = LabelEncoder()
oh_enc = OneHotEncoder()
train_labels = oh_enc.fit_transform(label_enc.fit_transform(gender_age_train['group']).reshape(-1,1)).todense()
train_labels
def construct_features(device_id):
    device_events = events[events['device_id'] == device_id]
    
    app_ids = []
    for id in np.unique(device_events['event_id']):
        apps = app_events[app_events['event_id'] == id]
        
        for app_id in apps['app_id']:
            app_ids.append(app_id)
    
    unique_apps = np.unique(app_ids)
    
    categories = ''
    
    for id in unique_apps:
        curr_lab = np.array(app_labels.loc[id])
        for lab in curr_lab:
            cat = label_categories.loc[lab]['category']
            categories += cat + " "
    
    
    return categories
test = construct_features(3315513013457872370)
test
