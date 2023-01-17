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
df = pd.read_csv("/kaggle/input/us-used-cars-dataset/used_cars_data.csv", low_memory=False, nrows=500000)
df.shape
df.head()
df.set_index("vin", inplace=True)
a = df.power.str.replace(",", "").str.split().str[0:4:3]
a = pd.DataFrame([[np.nan, np.nan] if type(i).__name__ == "float" else np.asarray(i).astype("float") for i in a],
                 columns=["horsepower", "horsepower_rpm"])
df[["horsepower", "horsepower_rpm"]] = a


a = df.torque.str.replace(",", "").str.split().str[0:4:3]
a = pd.DataFrame([[np.nan, np.nan] if type(i).__name__ == "float" else np.asarray(i).astype("float") for i in a], 
                 columns=["torque", "torque_rpm"])
df[["torque", "torque_rpm"]] = a

# del a
df.bed_length = df.bed_length.replace("--", np.nan)
df.bed_length = df.bed_length.str.split().str[0].astype(np.float)
extract_num_from_series_single_unit = lambda series: series.str.split().str[0].astype(np.float)

columns = ["back_legroom", "wheelbase", "width", "length", "height", 
           "fuel_tank_volume", "front_legroom", "maximum_seating"]

df[columns] = df[columns].replace({",": "", "--": np.nan}).apply(extract_num_from_series_single_unit)

df.major_options = df.major_options.apply(lambda x: eval(x) if type(x).__name__ == "str" else x)
drop_columns = ["bed_height", "power", "vehicle_damage_category"]
df.drop(columns=drop_columns, inplace=True)
df.shape
df.columns
df.head()