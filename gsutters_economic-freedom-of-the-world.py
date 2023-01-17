import numpy as np # linear algebra
import pandas as pd # data processing, CSV file
import matplotlib.pyplot as plt # plotting
import seaborn as sns # plotting
from mpl_toolkits.mplot3d import Axes3D

# Input data files are available in the "../input/" directory.

import os
efw = pd.read_csv("../input/efw_cc.csv")
#Slice data frame for the years we want to study. We will use these later.
efw_2016 = efw.loc[efw['year'] == 2016]
efw_1980 = efw.loc[efw['year'] == 1980]

#Select main categories from sliced data frames for summary statistics
efw2016_mc = efw_2016[["ECONOMIC FREEDOM", "rank", "1_size_government", "2_property_rights", "3_sound_money", "4_trade", "5_regulation"]]
efw1980_mc = efw_1980[["ECONOMIC FREEDOM", "rank", "1_size_government", "2_property_rights", "3_sound_money", "4_trade", "5_regulation"]]
print("Summary statistics for 2016")
display(efw2016_mc.describe())
print("Summary statistics for 1980")
display(efw1980_mc.describe())

sns.kdeplot(efw_2016["ECONOMIC FREEDOM"], label="2016", shade=True)
sns.kdeplot(efw_1980["ECONOMIC FREEDOM"], label="1980", shade=True)
plt.legend()
plt.title("Economic Freedom, 1980 and 2016")
_ = plt.xlabel("Economic Freedom score")
efw_gb = efw.groupby("year").mean()
_ = efw_gb.plot(y=["ECONOMIC FREEDOM", "1_size_government", "2_property_rights", "3_sound_money", "4_trade", "5_regulation"], figsize = (10,10), subplots=True)
_ = plt.xticks(rotation=360)
efw2016_corr = efw_2016[["1_size_government", "2_property_rights", "3_sound_money", "4_trade", "5_regulation"]]
sns.heatmap(efw2016_corr.corr(), square=True, cmap='RdYlGn')
plt.show()
efw_mc = efw[["year","ECONOMIC FREEDOM", "1_size_government", "2_property_rights", "3_sound_money", "4_trade", "5_regulation"]]
efw_gb = efw_mc.groupby("year").mean()

efw_gb.loc[2016] - efw_gb.loc[1970]
