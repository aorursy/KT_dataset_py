import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import wilcoxon
root = "../input/rafedata/SmartLight_Data-master/Single Intersection"

delay = pd.read_csv(root+ "/SmartLight_['traffic_flow_high_equal.rou.xml']_01_16_16_59_delay/log_rewards.csv")

wait_time = pd.read_csv(root + "/SmartLight_['traffic_flow_high_equal.rou.xml']_01_16_15_04_waiting_time/log_rewards.csv")

halt = pd.read_csv(root + "/SmartLight_['traffic_flow_high_equal.rou.xml']_01_16_12_38_halting_number/log_rewards.csv")



delay.shape, wait_time.shape, halt.shape
# ai 2 dataframe e row ekta beshi ken?

wait_time.drop(index=71902, inplace=True)

halt.drop(index=71902, inplace=True)
#Significance test

wilcoxon(wait_time['wait_time'], delay['wait_time'])
wilcoxon(delay['wait_time'], halt['wait_time'])
plt.figure(figsize=(14,6))

x = delay['count']

plt.plot(x, delay['wait_time'], label='delay')

plt.plot(x, halt['wait_time'], label='halt')

plt.plot(x, wait_time['wait_time'], label='wait_time')

plt.title("wait_time")

plt.legend();
plt.figure(figsize=(14,6))

x = delay['count']

plt.plot(x, delay['waiting_time_per_veh'], label='delay')

plt.plot(x, halt['waiting_time_per_veh'], label='halt')

plt.plot(x, wait_time['waiting_time_per_veh'], label='wait_time')

plt.title("waiting_time_per_veh")

plt.legend();





























