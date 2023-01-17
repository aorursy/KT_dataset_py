# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns 

import matplotlib.pyplot as plt 

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))







# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv",parse_dates=["date"],index_col="id",dtype={"store_nbr":np.int32,"unit_sales":np.float64})
df_stores = pd.read_csv("../input/stores.csv")
df_items = pd.read_csv("../input/items.csv")
df_transactions = pd.read_csv("../input/transactions.csv",parse_dates=["date"])
df_holidays = pd.read_csv("../input/holidays_events.csv",parse_dates=["date"])
holidays = pd.get_dummies(df_holidays["type"])

locale = pd.get_dummies(df_holidays['locale'],drop_first=True)
df_holidays = pd.concat([df_holidays,holidays,locale],axis=1)
df_holidays.drop("type",axis=1,inplace=True)

df_holidays.drop("description",axis=1,inplace=True)
df_holidays.drop("locale",axis=1,inplace=True)
df_holidays = pd.concat([df_holidays,locale],axis=1)
df_holidays['transferred'] = df_holidays['transferred'].astype("bool")
df_holidays['transferred'] = df_holidays['transferred'].map({False:0,True:1})
df_holidays.head()
# locale_names = pd.get_dummies(df_holidays['locale_name'],drop_first=True)

# df_holidays = pd.concat([df_holidays,locale_names],axis=1)
df_train_2017 = df_train[df_train['date'].dt.year == 2017]
df_train_2016 = df_train[df_train['date'].dt.year == 2016]
df_train_2015 = df_train[df_train['date'].dt.year == 2015]
df_train_2017 = df_train_2017.merge(df_holidays,how="left",on="date")
df_train_2017.head()
df_items.head()
df_items.groupby(["perishable","family"]).size()
home_dict = {"HOME AND KITCHEN I":"home","HOME AND KITCHEN II":"home","CLEANING":"home"}

food_non_perishable_dict = {"GROCERY I":"non_perishable_food","GROCERY II":"non_perishable_food","FROZEN FOODS":"non_perishable_food"}

clothing_dict = {"LADIESWEAR":"women_wear","LINGERIE":"women_wear"}

personal_dict = {"PERSONAL CARE":"personal","BEAUTY":"personal","BABY CARE":"personal","HOME CARE":"personal","CELEBRATION":"personal"}

others_dict = {"BOOKS":"others","MAGAZINES":"others","SCHOOL AND OFFICE SUPPLIES":"others","PET SUPPLIES":"others","LAWN AND GARDEN":"others"}

appliance_dict = {"HOME APPLIANCES":"appliance","PLAYERS AND ELECTRONICS":"appliance","HARDWARE":"appliance"}

drinks_dict = {"BEVERAGES":"drink","LIQUOR,WINE,BEER":"drink"}
df_items['family'].value_counts()