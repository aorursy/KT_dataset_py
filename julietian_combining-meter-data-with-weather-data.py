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
weather = pd.read_csv("../input/bdgp2-further-cleaned-datasets/weather_cleaned.csv")

lamb = pd.read_csv("/kaggle/input/meter-sums/lamb_meter_sums.csv")

crow = pd.read_csv("/kaggle/input/meter-sums/crow_meter_sums.csv")

bear = pd.read_csv("/kaggle/input/meter-sums/bear_meter_sums.csv")

robin = pd.read_csv("/kaggle/input/meter-sums/robin_meter_sums.csv")

panther = pd.read_csv("/kaggle/input/meter-sums/panther_meter_sums.csv")

bobcat = pd.read_csv("/kaggle/input/meter-sums/bobcat_meter_sums.csv")

rat = pd.read_csv("/kaggle/input/meter-sums/rat_meter_sums.csv")

fox = pd.read_csv("/kaggle/input/meter-sums/fox_meter_sums.csv")

shrew = pd.read_csv("/kaggle/input/meter-sums/shrew_meter_sums.csv")

mouse = pd.read_csv("/kaggle/input/meter-sums/mouse_meter_sums.csv")

peacock = pd.read_csv("/kaggle/input/meter-sums/peacock_meter_sums.csv")

hog = pd.read_csv("/kaggle/input/meter-sums/hog_meter_sums.csv")

cockatoo = pd.read_csv("/kaggle/input/meter-sums/cockatoo_meter_sums.csv")

moose = pd.read_csv("/kaggle/input/meter-sums/moose_meter_sums.csv")

gator = pd.read_csv("/kaggle/input/meter-sums/gator_meter_sums.csv")

eagle = pd.read_csv("/kaggle/input/meter-sums/eagle_meter_sums.csv")

wolf = pd.read_csv("/kaggle/input/meter-sums/wolf_meter_sums.csv")

bull = pd.read_csv("/kaggle/input/meter-sums/bull_meter_sums.csv")
panther.head()
weather.head()
weather.shape
weather.dtypes
weather["timestamp"] = pd.to_datetime(weather["timestamp"], format = "%Y-%m-%d %H:%M:%S")
weather = weather.drop("Unnamed: 0", axis = 1)
weather = weather.set_index("timestamp")
weather = weather.resample("W").mean()
weather.head()
#pull all "Panther" from weather dataframe 

cols = list(weather.columns)

w = [x for x in cols if x.startswith('Panther')]

site_weather = weather[w]



#edit panther dataframe to join 

panther["timestamp"] = pd.to_datetime(panther["timestamp"], format = "%Y-%m-%d %H:%M:%S")

combine = panther.set_index("timestamp")



#join dataframes

panther_all = pd.concat([site_weather, combine], axis = 1)
#pull all "Robin" from weather dataframe 

cols = list(weather.columns)

w = [x for x in cols if x.startswith('Robin')]

site_weather = weather[w]



#edit panther dataframe to join 

robin["timestamp"] = pd.to_datetime(robin["timestamp"], format = "%Y-%m-%d %H:%M:%S")

combine = robin.set_index("timestamp")



#join dataframes

robin_all = pd.concat([site_weather, combine], axis = 1)
#pull all "Fox" from weather dataframe 

cols = list(weather.columns)

w = [x for x in cols if x.startswith('Fox')]

site_weather = weather[w]



#edit panther dataframe to join 

fox["timestamp"] = pd.to_datetime(fox["timestamp"], format = "%Y-%m-%d %H:%M:%S")

combine = fox.set_index("timestamp")



#join dataframes

fox_all = pd.concat([site_weather, combine], axis = 1)
#pull all "Rat" from weather dataframe 

cols = list(weather.columns)

w = [x for x in cols if x.startswith('Rat')]

site_weather = weather[w]



#edit panther dataframe to join 

rat["timestamp"] = pd.to_datetime(rat["timestamp"], format = "%Y-%m-%d %H:%M:%S")

combine = rat.set_index("timestamp")



#join dataframes

rat_all = pd.concat([site_weather, combine], axis = 1)
#pull all "Bear" from weather dataframe 

cols = list(weather.columns)

w = [x for x in cols if x.startswith('Bear')]

site_weather = weather[w]



#edit panther dataframe to join 

bear["timestamp"] = pd.to_datetime(bear["timestamp"], format = "%Y-%m-%d %H:%M:%S")

combine = bear.set_index("timestamp")



#join dataframes

bear_all = pd.concat([site_weather, combine], axis = 1)
#pull all "Lamb" from weather dataframe 

cols = list(weather.columns)

w = [x for x in cols if x.startswith('Lamb')]

site_weather = weather[w]



#edit panther dataframe to join 

lamb["timestamp"] = pd.to_datetime(lamb["timestamp"], format = "%Y-%m-%d %H:%M:%S")

combine = lamb.set_index("timestamp")



#join dataframes

lamb_all = pd.concat([site_weather, combine], axis = 1)
#pull all "Peacock" from weather dataframe 

cols = list(weather.columns)

w = [x for x in cols if x.startswith('Peacock')]

site_weather = weather[w]



#edit panther dataframe to join 

peacock["timestamp"] = pd.to_datetime(peacock["timestamp"], format = "%Y-%m-%d %H:%M:%S")

combine = peacock.set_index("timestamp")



#join dataframes

peacock_all = pd.concat([site_weather, combine], axis = 1)
#pull all "Moose" from weather dataframe 

cols = list(weather.columns)

w = [x for x in cols if x.startswith('Moose')]

site_weather = weather[w]



#edit panther dataframe to join 

moose["timestamp"] = pd.to_datetime(moose["timestamp"], format = "%Y-%m-%d %H:%M:%S")

combine = moose.set_index("timestamp")



#join dataframes

moose_all = pd.concat([site_weather, combine], axis = 1)
#pull all "Gator" from weather dataframe 

cols = list(weather.columns)

w = [x for x in cols if x.startswith('Gator')]

site_weather = weather[w]



#edit panther dataframe to join 

gator["timestamp"] = pd.to_datetime(gator["timestamp"], format = "%Y-%m-%d %H:%M:%S")

combine = gator.set_index("timestamp")



#join dataframes

gator_all = pd.concat([site_weather, combine], axis = 1)
#pull all "Bull" from weather dataframe 

cols = list(weather.columns)

w = [x for x in cols if x.startswith('Bull')]

site_weather = weather[w]



#edit panther dataframe to join 

bull["timestamp"] = pd.to_datetime(bull["timestamp"], format = "%Y-%m-%d %H:%M:%S")

combine = bull.set_index("timestamp")



#join dataframes

bull_all = pd.concat([site_weather, combine], axis = 1)
#pull all "Bobcat" from weather dataframe 

cols = list(weather.columns)

w = [x for x in cols if x.startswith('Bobcat')]

site_weather = weather[w]



#edit panther dataframe to join 

bobcat["timestamp"] = pd.to_datetime(bobcat["timestamp"], format = "%Y-%m-%d %H:%M:%S")

combine = bobcat.set_index("timestamp")



#join dataframes

bobcat_all = pd.concat([site_weather, combine], axis = 1)
#pull all "Crow" from weather dataframe 

cols = list(weather.columns)

w = [x for x in cols if x.startswith('Crow')]

site_weather = weather[w]



#edit panther dataframe to join 

crow["timestamp"] = pd.to_datetime(crow["timestamp"], format = "%Y-%m-%d %H:%M:%S")

combine = crow.set_index("timestamp")



#join dataframes

crow_all = pd.concat([site_weather, combine], axis = 1)
#pull all "Shrew" from weather dataframe 

cols = list(weather.columns)

w = [x for x in cols if x.startswith('Shrew')]

site_weather = weather[w]



#edit panther dataframe to join 

shrew["timestamp"] = pd.to_datetime(shrew["timestamp"], format = "%Y-%m-%d %H:%M:%S")

combine = shrew.set_index("timestamp")



#join dataframes

shrew_all = pd.concat([site_weather, combine], axis = 1)
#pull all "Wolf" from weather dataframe 

cols = list(weather.columns)

w = [x for x in cols if x.startswith('Wolf')]

site_weather = weather[w]



#edit panther dataframe to join 

wolf["timestamp"] = pd.to_datetime(wolf["timestamp"], format = "%Y-%m-%d %H:%M:%S")

combine = wolf.set_index("timestamp")



#join dataframes

wolf_all = pd.concat([site_weather, combine], axis = 1)
#pull all "Hog" from weather dataframe 

cols = list(weather.columns)

w = [x for x in cols if x.startswith('Hog')]

site_weather = weather[w]



#edit panther dataframe to join 

hog["timestamp"] = pd.to_datetime(hog["timestamp"], format = "%Y-%m-%d %H:%M:%S")

combine = hog.set_index("timestamp")



#join dataframes

hog_all = pd.concat([site_weather, combine], axis = 1)
#pull all "Eagle" from weather dataframe 

cols = list(weather.columns)

w = [x for x in cols if x.startswith('Eagle')]

site_weather = weather[w]



#edit panther dataframe to join 

eagle["timestamp"] = pd.to_datetime(eagle["timestamp"], format = "%Y-%m-%d %H:%M:%S")

combine = eagle.set_index("timestamp")



#join dataframes

eagle_all = pd.concat([site_weather, combine], axis = 1)
#pull all "Cockatoo" from weather dataframe 

cols = list(weather.columns)

w = [x for x in cols if x.startswith('Cockatoo')]

site_weather = weather[w]



#edit panther dataframe to join 

cockatoo["timestamp"] = pd.to_datetime(cockatoo["timestamp"], format = "%Y-%m-%d %H:%M:%S")

combine = cockatoo.set_index("timestamp")



#join dataframes

cockatoo_all = pd.concat([site_weather, combine], axis = 1)
#pull all "Mouse" from weather dataframe 

cols = list(weather.columns)

w = [x for x in cols if x.startswith('Mouse')]

site_weather = weather[w]



#edit panther dataframe to join 

mouse["timestamp"] = pd.to_datetime(mouse["timestamp"], format = "%Y-%m-%d %H:%M:%S")

combine = mouse.set_index("timestamp")



#join dataframes

mouse_all = pd.concat([site_weather, combine], axis = 1)
panther_all.to_csv("panther_meters_and_weather.csv")

moose_all.to_csv("moose_meters_and_weather.csv")

eagle_all.to_csv("eagle_meters_and_weather.csv")

cockatoo_all.to_csv("cockatoo_meters_and_weather.csv")

fox_all.to_csv("fox_meters_and_weather.csv")

peacock_all.to_csv("peacock_meters_and_weather.csv")

bull_all.to_csv("bull_meters_and_weather.csv")

hog_all.to_csv("hog_meters_and_weather.csv")

crow_all.to_csv("crow_meters_and_weather.csv")

bobcat_all.to_csv("bobcat_meters_and_weather.csv")

robin_all.to_csv("robin_meters_and_weather.csv")

bear_all.to_csv("bear_meters_and_weather.csv")

lamb_all.to_csv("lamb_meters_and_weather.csv")

rat_all.to_csv("rat_meters_and_weather.csv")

gator_all.to_csv("gator_meters_and_weather.csv")

wolf_all.to_csv("wolf_meters_and_weather.csv")

shrew_all.to_csv("shrew_meters_and_weather.csv")

mouse_all.to_csv("mouse_meters_and_weather.csv")



#nothing in swan