# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import datetime

import random



# Plots

import seaborn as sns

import matplotlib.pyplot as plt



# Models

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import Ridge, RidgeCV

from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor

import lightgbm as lgb

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor



# Stats

from scipy.stats import skew, norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



# Misc

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA



pd.set_option('display.max_columns', None)



# Ignore useless warnings

import warnings

warnings.filterwarnings(action="ignore")
sample = pd.read_csv("../input/daily-temperature-of-major-cities/city_temperature.csv")
sample.head()
print(sample.shape)

print("-"*60)

print(sample.info())

print("-"*60)

print(sample.describe())

print("-"*60)

print(sample.isnull().sum())
print(sample["Year"].unique())
sample = sample[sample["Year"] != 200]

sample = sample[sample["Year"] != 201]
sns.set()

plt.figure(figsize=(15,5))

sns.countplot(y="Region", data=sample)
# drop 2020 data







# groupby countries

africa = sample.groupby("Region").get_group("Africa")

asia = sample.groupby("Region").get_group("Asia")

south_pacific = sample.groupby("Region").get_group("Australia/South Pacific")

europe = sample.groupby("Region").get_group("Europe")

middle_east = sample.groupby("Region").get_group("Middle East")

north_america = sample.groupby("Region").get_group("North America")

carribean = sample.groupby("Region").get_group("South/Central America & Carribean")
combine = [

    africa,

    asia,

    south_pacific,

    europe,

    middle_east,

    north_america,

    carribean

]



for dataset in combine:

    dataset = dataset.query("AvgTemperature > -99.0", inplace=True)
sns.set()



plt.figure(figsize=(15,5))

sns.countplot(y="Country", data=africa)

plt.title("Countries of Africa")
sns.set()



plt.figure(figsize=(15,5))

sns.countplot(y="Country", data=asia)

plt.title("Countries of Asia")
sns.set()



plt.figure(figsize=(15,5))

sns.countplot(y="Country", data=south_pacific)

plt.title("Countries of South_pacific")
sns.set()



plt.figure(figsize=(15,5))

sns.countplot(y="Country", data=europe)

plt.title("Countries of Europe")
sns.set()



plt.figure(figsize=(15,5))

sns.countplot(y="Country", data=middle_east)

plt.title("Countries of Middle_east")
sns.set()



plt.figure(figsize=(15,5))

sns.countplot(y="Country", data=north_america)

plt.title("Countries of North_america")
sns.set()



plt.figure(figsize=(15,5))

sns.countplot(y="Country", data=carribean)

plt.title("Countries of Carribean")


def yearly_temp(dataset):

    region_yearly_temp =dataset.groupby("Year")["AvgTemperature"].mean()

    return region_yearly_temp



def plot_yearly_temp(dataset,title):

    sns.set()

    

    x = dataset.groupby("Year").mean()

    x = x.index

    y = yearly_temp(dataset)

    

    plt.figure(figsize=(15,4))

    plt.plot(x, y, marker="o", label="Temperature")

    plt.xlabel("Year")

    plt.ylabel("AvgTemperature")

    plt.legend()

    

    plt.title(title)
plot_yearly_temp(africa.query("Year<2020"), "Africa")

plot_yearly_temp(asia.query("Year<2020"), "Asia")

plot_yearly_temp(south_pacific.query("Year<2020"), "South_pacific")

plot_yearly_temp(europe.query("Year<2020"), "Europe")

plot_yearly_temp(middle_east.query("Year<2020"), "Middle_east")

plot_yearly_temp(north_america.query("Year<2020"), "North_america")

plot_yearly_temp(carribean.query("Year<2020"), "Carribean")
region = {

    "Africa":africa,

    "Asia":asia,

    "South_pacific":south_pacific,

    "Europa":europe,

    "Middle_east":middle_east,

    "North_america":north_america,

    "Carribean":carribean

}



plt.figure(figsize=(15,5))



for i, j in zip(region.values(), region.keys()):

    i = i.query("Year<2020")

    temp = i.groupby("Year")["AvgTemperature"].mean()

    

    x = temp.axes[0]

    

    plt.plot(x, temp, marker="o", label=j)

    plt.xlabel("Year")

    plt.ylabel("AvgTemperature")

    plt.title("Regions of AvgTemperature")



    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    
# total avgtemperature



total_avg = sample.groupby("Year").mean()

total_avg = total_avg.query("Year<2020")



sns.set()



plt.figure(figsize=(15,6))

plt.scatter(total_avg.index, total_avg["AvgTemperature"])



plt.ylabel("AvgTemperature")

plt.xlabel("Year")

plt.title("Yearly avgtemperature in the world")
#seasons

spring = [3,4,5]

summer = [6,7,8]

autumn = [9,10,11]

winter = [12,1,2]



def seasons_temp(dataset, month):

    region_temp = dataset[dataset["Month"] == month]

    region_avg_temp = region_temp.groupby("Year")["AvgTemperature"].mean()

    return region_avg_temp



def seasons_yearly_temp(month1, month2, month3, seasons):

    fig,axs = plt.subplots(3,figsize=(15,10))

    

    axs[0].plot(month1.axes[0], month1, marker="o", color="blue")

    axs[0].set_xlabel("Year")

    axs[0].set_ylabel("AvgTemperature")

    axs[0].set_title("Month:" + str(seasons[0]))

    

    axs[1].plot(month2.axes[0], month2, marker="o", color="red")

    axs[1].set_xlabel("Year")

    axs[1].set_ylabel("AvgTemperature")

    axs[1].set_title("Month:" + str(seasons[1]))

    

    axs[2].plot(month3.axes[0],month3, marker="o", color="green" )

    axs[2].set_xlabel("Year")

    axs[2].set_ylabel("AvgTemperature")

    axs[2].set_title("Month:" + str(seasons[2]))

    

    for ax in axs.flat:

        ax.label_outer()
africa_Mar = seasons_temp(africa, spring[0])

africa_Apr = seasons_temp(africa, spring[1])

africa_May = seasons_temp(africa, spring[2])



seasons_yearly_temp(africa_Mar, africa_Apr, africa_May,spring)
africa_Jun = seasons_temp(africa, summer[0])

africa_Jul = seasons_temp(africa, summer[1])

africa_Aug = seasons_temp(africa, summer[2])



seasons_yearly_temp(africa_Jun, africa_Jul, africa_Aug,summer)
africa_Sep = seasons_temp(africa, autumn[0])

africa_Oct = seasons_temp(africa, autumn[1])

africa_Nov = seasons_temp(africa, autumn[2])



seasons_yearly_temp(africa_Sep, africa_Oct, africa_Nov,autumn)
africa_Dec = seasons_temp(africa, winter[0])

africa_Jan = seasons_temp(africa, winter[1])

africa_Feb = seasons_temp(africa, winter[2])



seasons_yearly_temp(africa_Dec, africa_Jan, africa_Feb,winter)