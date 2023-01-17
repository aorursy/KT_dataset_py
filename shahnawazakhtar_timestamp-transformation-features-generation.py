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

events = pd.read_csv("../input/events.csv", parse_dates =["timestamp"])   # 200MB event_id, device_id, timestamp, longitude, latitude

test = pd.read_csv("../input/gender_age_test.csv")   # device_id

train = pd.read_csv("../input/gender_age_train.csv")  # device_id, gender, age, group



events["timestamp"].loc[(events.longitude != 0.0) & (events.latitude != 0.0)] += events["longitude"].apply(lambda x: pd.Timedelta(seconds = (240* (x - 116.407))))

events['hourly'] = events.timestamp.dt.hour

events["hourly"].loc[(events.longitude != 0.0) & (events.latitude != 0.0)] = np.nan



hourly = events.groupby("device_id")["hourly"].apply(lambda x: " ".join(str(s) for s in x))



train["hourly"] = "Hourly:"+train["device_id"].map(hourly).astype(str)

test["hourly"] = "Hourly:"+test["device_id"].map(hourly).astype(str)



print(train.loc[["device_id","hourly"]])