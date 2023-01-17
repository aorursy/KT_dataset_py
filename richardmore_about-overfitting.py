# Import all the necessary packages 

import kagglegym

import numpy as np

import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.linear_model import LinearRegression, Ridge

import math

import matplotlib.pyplot as plt



%matplotlib inline
# Read the full data set stored as HDF5 file

full_df = pd.read_hdf('../input/train.h5')

# A custom function to compute the R score

def get_reward(y_true, y_fit):

    R2 = 1 - np.sum((y_true - y_fit)**2) / np.sum((y_true - np.mean(y_true))**2)

    R = np.sign(R2) * math.sqrt(abs(R2))

    return(R)
# Some pre-processing as seen from most of the public scripts.

# The "environment" is our interface for code competitions

env = kagglegym.make()



# We get our initial observation by calling "reset"

observation = env.reset()

target_var = 'y'



# Get the train dataframe

train = observation.train

mean_values = train.median(axis=0)

train.fillna(mean_values, inplace=True)



# Observed with histograns:

low_y_cut = -0.086093

high_y_cut = 0.093497



y_is_above_cut = (train.y > high_y_cut)

y_is_below_cut = (train.y < low_y_cut)

y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
### https://www.kaggle.com/ymcdull/two-sigma-financial-modeling/ridge-lb-0-0100659/run/545100

# The "environment" is our interface for code competitions

env = kagglegym.make()



# We get our initial observation by calling "reset"

observation = env.reset()



# cols_to_use for ridge model

#cols_to_use = ['technical_30', 'technical_20', 'fundamental_11']

cols_to_use = ['technical_20']



# model build

model = Ridge()

model.fit(np.array(train.loc[y_is_within_cut, cols_to_use].values), train.loc[y_is_within_cut, target_var])



# getting the y mean dict for averaging

ymean_dict = dict(train.groupby(["id"])["y"].mean())



# weighted average of model & mean

def get_weighted_y(series):

    id, y = series["id"], series["y"]

    return 0.95 * y + 0.05 * ymean_dict[id] if id in ymean_dict else y



y_actual_list = []

y_pred_list = []

r1_overall_reward_list = []

ts_list = []

r1_lms = [] # mean squares between pred_y and acctual_y on each timestamp 

r1_lms_pre = [] # mean predicted return on each timestamp

r1_lms_acc = [] # mean acctual return on each timestamp

while True:

    timestamp = observation.features["timestamp"][0]

    actual_y = list(full_df[full_df["timestamp"] == timestamp]["y"].values)

    observation.features.fillna(mean_values, inplace=True)

    test_x = np.array(observation.features[cols_to_use].values)

    observation.target.y = model.predict(test_x).clip(low_y_cut, high_y_cut)

    

    ## weighted y using average value

    observation.target.y = observation.target.apply(get_weighted_y, axis = 1)

    target = observation.target

    observation, reward, done, info = env.step(target)

    

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

    

    pred_y = list(target.y.values)

    y_actual_list.extend(actual_y)

    y_pred_list.extend(pred_y)

    r1_lms.append(np.average( (np.array(actual_y)-np.array(pred_y) )**2))

    tmp = np.average( np.abs(np.array(actual_y))  ) 

    r1_lms_acc.append(  tmp  )

    tmp = np.average( np.abs( np.array(pred_y)  )   ) 

    r1_lms_pre.append(  tmp )

    

    

    overall_reward = get_reward(np.array(y_actual_list), np.array(y_pred_list))

    r1_overall_reward_list.append(overall_reward)

    ts_list.append(timestamp)

    if done:

        break

    

print(info)
plt.plot(ts_list,r1_lms)

plt.title("mean square between pred_y and acctual_y on each timestamp")

plt.show()

plt.plot(ts_list,r1_lms_acc)

plt.title("mean absolute value of acctual_y on each timestamp")

plt.show()

plt.plot(ts_list,r1_lms_pre)

plt.title("mean absolute value of pred_y on each timestamp")

plt.show()

fig = plt.figure(figsize=(12, 6))

plt.plot(ts_list, r1_overall_reward_list, c='blue')

plt.plot(ts_list, [0]*len(ts_list), c='red')

plt.title("Cumulative R value change for Univariate Ridge (technical_20)")

plt.ylim([-0.04,0.04])

plt.xlim([850, 1850])

plt.show()
from sklearn import linear_model as lm

cols_to_use = ['technical_30', 'technical_20', 'technical_40', 'technical_19']

# Get first observation

env = kagglegym.make()

observation = env.reset()

train = observation.train

mean_values = train.mean(axis=0)

train.fillna(mean_values, inplace=True)



model = lm.LinearRegression()

model.fit(np.array(train[cols_to_use]), train.y.values)



y_pred_list = []

mycode_overall_reward_list = []

y_actual_list = []

mycode_lms = []

mycode_lms_acc = []

mycode_lms_pre = []



while True:

    # code for the statistics

    timestamp = observation.features["timestamp"][0]

    actual_y = list(full_df[full_df["timestamp"] == timestamp]["y"].values)

    

    observation.features.fillna(mean_values, inplace=True)

    test_x = np.array(observation.features[cols_to_use])

    observation.target.y = model.predict(test_x)

    target = observation.target

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

    # code for the statistics    

    pred_y = list(target.y.values)    

    y_pred_list.extend(pred_y)

    y_actual_list.extend(actual_y)

    mycode_lms.append(np.average( (np.array(actual_y)-np.array(pred_y) )**2))

    

    tmp = np.average( np.abs(np.array(actual_y))  ) 

    mycode_lms_acc.append(  tmp  )

    tmp = np.average( np.abs( np.array(pred_y)  )   ) 

    mycode_lms_pre.append(  tmp )

    

    overall_reward = get_reward(np.array(y_actual_list), np.array(y_pred_list))

    mycode_overall_reward_list.append(overall_reward)

    

    observation, reward, done, info = env.step(target)

    if done:

        break

print(info)


plt.plot(ts_list,np.array(mycode_lms_acc)/50+0.001,color="y",label="transformed acutal")

plt.plot(ts_list,mycode_lms_pre,label="mycode",color='r')

plt.plot(ts_list,r1_lms_pre,label="r1",color='g')

plt.legend(loc='upper right')

plt.title("the magnitude of pred_y from r1 model, pred_y from mycode,transformed actual_y on each timestamp")

plt.show()
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(ts_list, mycode_overall_reward_list, c='green', label='mycode')

ax.plot(ts_list, r1_overall_reward_list, c='blue', label='ridge-1')

ax.plot(ts_list, [0]*len(ts_list), c='red', label='zero line')

ax.legend(loc='lower right')

ax.set_ylim([-0.04,0.04])

ax.set_xlim([850, 1850])

plt.title("Cumulative R value change for ridge1 and mycode")

plt.show()
# just for get the statistics....

observation = env.reset()

y_actual_list = []

cumu_average_return = []

cumu_std_return = []

average_return = []

std_return = [] 



while True:

    # code for the statistics

    timestamp = observation.features["timestamp"][0]

    actual_y = list(full_df[full_df["timestamp"] == timestamp]["y"].values)

    target = observation.target

    if timestamp % 100 == 0:

        print("Timestamp #{}".format(timestamp))

    # code for the statistics    

    y_actual_list.extend(actual_y)

    average_return.append(np.average(actual_y))

    std_return .append(np.std(actual_y))

    cumu_average_return.append(np.average(y_actual_list))

    cumu_std_return.append(np.std(y_actual_list))

    observation, reward, done, info = env.step(target)

    if done:

        break
plt.plot(ts_list, average_return, c='red', label='average_rtn')

plt.show()

plt.plot(ts_list, std_return, c='yellow', label='average_rtn')

plt.show()

plt.plot(ts_list, cumu_average_return, c='red', label='average_rtn')

plt.show()

plt.plot(ts_list, cumu_std_return, c='yellow', label='average_rtn')

plt.show()


r1_lms_sqrt =[np.sqrt(i) for i in r1_lms] 

mycode_lms_sqrt =[np.sqrt(i) for i in mycode_lms] 



fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(ts_list, mycode_lms_sqrt, c='green', label='mycode')

ax.plot(ts_list, r1_lms_sqrt, c='blue', label='ridge-1')

plt.title("the new  for ridge1 and mycode")

plt.show()
plt.plot(ts_list,mycode_lms_pre)

plt.show()

plt.plot(ts_list,mycode_lms_acc)

plt.show()

plt.plot(ts_list,r1_lms_pre)

plt.show()

plt.plot(ts_list,r1_lms_acc)

plt.show()