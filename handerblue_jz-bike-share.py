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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
data = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

data['time_hr'] = data['datetime'].transform(lambda x:int(x[11:13]))

test['time_hr'] = test['datetime'].transform(lambda x:int(x[11:13]))



cols = data.columns.tolist()

cols = cols[-1:] + cols[:-1]

data = data[cols]



cols = test.columns.tolist()

cols = cols[-1:] + cols[:-1]

test = test[cols]





data.describe()
data.head(30)
test.head()
hr24_avg_reg_list = []

for i in range(24):  

    hr_reg = data[data['time_hr'] == i]['registered']

    hr24_avg_reg_list.append(hr_reg.sum(axis = 0) / hr_reg.shape[0])

    

hr24_avg_reg_df = pd.DataFrame({"hr24_avg_reg": hr24_avg_reg_list})

hr24_avg_reg_df.plot(kind='bar')

plt.show()
working_reg_list = []

non_working_reg_list = []

for i in range(24):  

    hr_reg = data[(data['time_hr'] == i) & (data['workingday'] == 1)]['registered']

    working_reg_list.append(hr_reg.sum(axis = 0) / hr_reg.shape[0])

    

    hr_reg = data[(data['time_hr'] == i) & (data['workingday'] == 0)]['registered']

    non_working_reg_list.append(hr_reg.sum(axis = 0) / hr_reg.shape[0])



    

working_reg_df = pd.DataFrame({"working_reg": working_reg_list, "non_working": non_working_reg_list})

working_reg_df.plot(kind='bar')

plt.title("Registered Count by 24hrs\n working vs non_working")

plt.show()
holiday_reg_list = []

non_holiday_reg_list = []

for i in range(24):    

    hr_reg = data[(data['time_hr'] == i) & (data['holiday'] == 1)]['registered']

    holiday_reg_list.append(hr_reg.sum(axis = 0) / hr_reg.shape[0])

    

    hr_reg = data[(data['time_hr'] == i) & (data['holiday'] == 0)]['registered']

    non_holiday_reg_list.append(hr_reg.sum(axis = 0) / hr_reg.shape[0])



    

holiday_df = pd.DataFrame({"non_holidy": non_holiday_reg_list, "holiday": holiday_reg_list})

holiday_df.plot(kind='bar')

plt.title("Registered Count by 24hrs\n holiday vs non_holiday")

plt.show()
season_list = []

for i in range(1,5):

    season_reg = data[data['season'] == i]['registered']

    season_list.append(season_reg.sum(axis = 0) / season_reg.shape[0])

    

    hr_reg = data[(data['time_hr'] == i) & (data['holiday'] == 0)]['registered']

    non_holiday_reg_list.append(hr_reg.sum(axis = 0) / hr_reg.shape[0])



    

season_df = pd.DataFrame({"Season": season_list})

season_df.plot(kind='bar')

plt.title("Registered count by Seasons")

plt.show()
data.head()
plt.scatter(data['temp'], data['registered'])

plt.title("temp VS registered")

plt.show()

plt.title("atemp VS registered")

plt.scatter(data['atemp'], data['registered'])

plt.show()
plt.scatter(data['temp'], data['atemp'])

plt.show()
plt.scatter(data['humidity'], data['registered'])

plt.title("Humidity VS registered")

plt.show()
plt.scatter(data['windspeed'], data['registered'])

plt.show()
data.columns
# skip atemp and humidity according to plot charts

# skip casual and cout, beacuse data[casual] + data[registered] = data[count]

cols = ['time_hr', 'season', 'holiday', 'workingday', 'weather','temp', 'windspeed']

x_data = data[cols]

y_data = data['registered']



x_test = test[cols]
x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, random_state = 0, test_size = 1/3)
model = LinearRegression(normalize = True)

model.fit(x_train, y_train)

predictions = model.predict(x_valid)

# Review distribution of predictions 

from scipy import stats

stats.describe(predictions)
plt.plot(predictions[:100])

plt.show()
### Invalid value Handling: Clip to Minimun count

min_bike_count = 1

for i in range(predictions.shape[0]):

    if predictions[i] < min_bike_count:

        predictions[i] = min_bike_count



# Evaluation by Root Mean Squared Logarithmic Error

RMSLE = np.sqrt( ((np.log(1 + y_valid) - np.log(1 + predictions)) ** 2).sum() / predictions.shape[0] )

print("RMSLE = ", RMSLE)
predict_test = model.predict(x_test)

predict_test = predict_test.astype(int)

min_bike_count = 1

for i in range(predict_test.shape[0]):

    if predict_test[i] < min_bike_count:

        predict_test[i] = min_bike_count

result = pd.DataFrame({"datetime":test['datetime'], "count": predict_test})

result.head(300)



result.to_csv("submission.csv", header=True, index=False)
view_result = pd.read_csv("./submission.csv")

view_result.head(30)