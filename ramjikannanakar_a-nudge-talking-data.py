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
print("Read Events")
events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})
print("Read App Events")
app_events = pd.read_csv("../input/app_events.csv")
events.device_id.value_counts(sort=True, ascending=False).head(5)
sample_events = events[events.device_id == '-8340098378141155823']
print("Read App labels")
app_labels = pd.read_csv("../input/app_labels.csv")
label_categ = pd.read_csv("../input/label_categories.csv")
app_label_categ = pd.merge(app_labels,label_categ,on='label_id')
app_event_def = pd.merge(app_events,app_label_categ,on='app_id')
app_event_def.head(5)
