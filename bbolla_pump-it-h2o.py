# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/Pump_it_Up_Data_Mining_the_Water_Table_-_Training_set_values.csv")
y_train = pd.read_csv("../input/Pump_it_Up_Data_Mining_the_Water_Table_-_Training_set_labels.csv")
train.shape
train.head()
y_train.head()
y_train.shape
test = pd.read_csv("../input/Pump_it_Up_Data_Mining_the_Water_Table_-_Test_set_values.csv")
test.head()
train = pd.merge(train, y_train, how= 'inner', on='id')
train.shape
train.head()
import h2o
h2o.init()
data_train = h2o.H2OFrame(train)
data_test = h2o.H2OFrame(test)
y = 'status_group'
x = data_train.names
x.remove(y)
train, test = data_train.split_frame([0.8])
train.shape
test.shape
from h2o.automl import H2OAutoML
mA = H2OAutoML(max_runtime_secs=300)
mA.train(x,"status_group",data_train)
predict_automl = mA.leader.predict(test)
predict_automl
mA.leader.model_performance(test)
mA.leaderboard
predict_automl.types
predict_automl = mA.leader.predict(data_test)
predict_automl
autoML_df = predict_automl.as_data_frame(use_pandas=True)
autoML_df['predict'].value_counts()
submission = pd.read_csv("../input/Pump_it_Up_Data_Mining_the_Water_Table_-_Submission_format.csv")
submission.head()
submission = submission.drop('status_group', axis=1)

submission.head()
autoML_df.head()
submission['status_group'] = autoML_df['predict']
submission.head()
submission.to_csv('sub.csv', index = False)


# import the modules we'll need
from IPython.display import HTML
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "sub.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)


# create a link to download the dataframe
create_download_link(submission)

# ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 


