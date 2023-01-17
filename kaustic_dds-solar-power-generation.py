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
# Task 1. Load the data from the CSV files



df_p1_gen = pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv")

df_p1_sen = pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")

df_p2_gen = pd.read_csv("../input/solar-power-generation-data/Plant_2_Generation_Data.csv")

df_p2_sen = pd.read_csv("../input/solar-power-generation-data/Plant_2_Weather_Sensor_Data.csv")
# Getting basic overview of 1 dataset with .info()



print(df_p1_gen.info())
# Trying to get same information by hand



def df_info(df):

    cols_data = "\n".join([f"{i}. '{col}' - {df[col].dtype}, Non-null: {df[col].count()}" for i, col in enumerate(df.columns, 1)])

    return f"Number of Rows: {len(df.index)}\nColumns (n: {df.columns.size}): \n{cols_data}"





print("Generator Data (Plant 1)\n", df_info(df_p1_gen), sep="\n")
print("Sensor Data (Plant 1)\n", df_info(df_p1_sen), sep="\n")
print("Generator Data (Plant 2)\n", df_info(df_p2_gen), sep="\n")
len(df_p1_gen.index) - len(df_p2_gen.index)
print("Sensor Data (Plant 2)\n", df_info(df_p2_sen), sep="\n")
# More detailed view of Data from Plant 1's generator



df_p1_gen.describe()
# Number of unqiue SOURCE_KEYs (generators)



print("Number of unique generators:", df_p1_gen["SOURCE_KEY"].unique().size)

print(df_p1_gen["SOURCE_KEY"].value_counts())
# Looks like lots of zeroes in DC_POWER, AC_POWER, and DAILY_YIELD columns



p1_gen_zeroes_count_dc = df_p1_gen["DC_POWER"].value_counts()[0]

p1_gen_zeroes_count_ac = df_p1_gen["AC_POWER"].value_counts()[0]

p1_gen_zeroes_count_dy = df_p1_gen["DAILY_YIELD"].value_counts()[0]  # Suffix 'dy' = DAILY_YIELD



print("Number of zeroes in columns: ")

for colname in ["DC_POWER", "AC_POWER", "DAILY_YIELD"]:

    print(f"{colname}: {df_p1_gen[colname].value_counts()[0]}")
# Check if the values of DC_POWER and AC_POWER are 0 in the same rows



p1_gen_dc_zero = df_p1_gen[df_p1_gen.DC_POWER == 0]



print(f"Number of rows: {len(p1_gen_dc_zero)}")

print(f"Number of zeroes in AC_POWER column: {p1_gen_dc_zero['AC_POWER'].value_counts()[0]}")
# Let's check how DAILY_YIELD varies when DC_POWER and AC_POWER are 0



p1_gen_dc_zero["DAILY_YIELD"].value_counts()
p1_gen_dc_zero.describe()
df_p2_sen.describe()
# Total irradiation per day



p1_total_irrad, p2_total_irrad = df_p1_sen["IRRADIATION"].sum(), df_p2_sen["IRRADIATION"].sum()



print(f"Plant 1: {p1_total_irrad}", f"Plant 2: {p2_total_irrad}", sep="\n")
# Number of inverters in each plant



p1_invs, p2_invs = df_p1_gen["SOURCE_KEY"].unique().size, df_p2_gen["SOURCE_KEY"].unique().size



print(f"Plant 1: {p1_invs}", f"Plant 2: {p2_invs}", sep="\n")
p1_gen_total_dc = df_p1_gen.groupby("SOURCE_KEY")["DC_POWER"].sum().sort_values(ascending=False)

print(p1_gen_total_dc, "\n")



# Since sorted in descending order, first element will be max

p1_gen_max_dc_inv = p1_gen_total_dc.keys()[0]

p1_gen_max_dc = p1_gen_total_dc[0]

print(f"Inverter with max DC power in Plant 1: {p1_gen_max_dc_inv} ({p1_gen_max_dc})")
p2_gen_total_dc = df_p2_gen.groupby("SOURCE_KEY")["DC_POWER"].sum().sort_values(ascending=False)

print(p2_gen_total_dc, "\n")



p2_gen_max_dc_inv = p2_gen_total_dc.keys()[0]

p2_gen_max_dc = p2_gen_total_dc[0]

print(f"Inverter with max DC power in Plant 2: {p2_gen_max_dc_inv} ({p2_gen_max_dc})")
print(p1_gen_dc_zero["SOURCE_KEY"].value_counts())
# Would be better if we had proper datetime field instead of datetime as string. 

# Then we can see at what times was the power generation zero



p1_gen_dc_by_dt = df_p1_gen.groupby("DATE_TIME")["DC_POWER"].sum().sort_values(ascending=False)



print(p1_gen_dc_by_dt.describe())

print(p1_gen_dc_by_dt.head(10))
# Inverter with maximum DC_POWER in any 15 minutes



df_p1_gen["DATE_TIME"][df_p1_gen["DC_POWER"].idxmax()]
# For doing "time" operations with the DATE_TIME column, I tried to convert it to a

# more standard format with pd.to_datetime, but it was giving some errors while trying to convert that to time

# This approach is straightforward and easy



df_p1_gen_dt_to_time = df_p1_gen["DATE_TIME"].apply(lambda x: x[-5:])
df_p1_gen_with_time = df_p1_gen.copy()

df_p1_gen_with_time["TIME"] = df_p1_gen_dt_to_time



df_p1_gen_with_time.head()
# Looping over it is slow!!



df_p1_gen_6_to_20 = df_p1_gen_with_time[[True if int(row["TIME"][:2]) > 6 and int(row["TIME"][:2]) < 20 else False for index, row in df_p1_gen_with_time.iterrows()]]

df_p1_gen_6_to_20
df_p1_gen_with_time[df_p1_gen_with_time.DC_POWER == 0.0]["TIME"].value_counts()
# For most times in day, the number of failures is in single digit or 0

# On some days, sunset happened before 20:00 and therefore their count is more



df_p1_gen_6_to_20_dc_zero = df_p1_gen_6_to_20[df_p1_gen_6_to_20.DC_POWER == 0.0]



print(df_p1_gen_6_to_20_dc_zero.head())

print(df_p1_gen_6_to_20_dc_zero["TIME"].value_counts())
df_p1_gen["DATE_TIME"] = pd.to_datetime(df_p1_gen["DATE_TIME"], format="%d-%m-%Y %H:%M")

df_p1_gen["DATE"] = df_p1_gen["DATE_TIME"].apply(lambda x: x.date())

df_p1_gen["TIME"] = df_p1_gen["DATE_TIME"].apply(lambda x: x.time())



print(df_p1_gen["DATE"])

print(df_p1_gen["TIME"])
# Visualizing data with matplotlib



from matplotlib import pyplot as plt



import seaborn as sns
# Plot shows that DC_POWER and AC_POWER are linearly related!

# y = mx + c



plt.scatter(x=df_p1_gen["DC_POWER"], y=df_p1_gen["AC_POWER"])
plt.figure(figsize=(12, 10))

plt.plot(df_p1_gen["DATE_TIME"], df_p1_gen["DC_POWER"])

plt.plot(df_p1_gen["DATE_TIME"], df_p1_gen["AC_POWER"])

plt.show()
df_p1_gen_dc_by_date = df_p1_gen.groupby("DATE")["DC_POWER"].sum()



plt.figure(figsize=(12, 10))

plt.bar(df_p1_gen_dc_by_date.keys(), df_p1_gen_dc_by_date)



plt.show()
# Fix the 1st generator data (DC_POWER was 10x the correct one)



df_p1_gen.DC_POWER = df_p1_gen.DC_POWER.apply(lambda x: x*0.1)
plt.figure(figsize=(12, 10))

plt.plot(df_p1_gen["DATE_TIME"], df_p1_gen["DC_POWER"])

plt.plot(df_p1_gen["DATE_TIME"], df_p1_gen["AC_POWER"])

plt.show()
plt.figure(figsize=(12, 10))

plt.plot(df_p2_gen["DATE_TIME"], df_p2_gen["DC_POWER"])

plt.plot(df_p2_gen["DATE_TIME"], df_p2_gen["AC_POWER"])

plt.show()
df_p2_gen.DATE_TIME = pd.to_datetime(df_p2_gen.DATE_TIME)

df_p2_gen["DATE"] = df_p2_gen.DATE_TIME.apply(lambda x: x.date())

df_p2_gen["TIME"] = df_p2_gen.DATE_TIME.apply(lambda x: x.time())



df_p2_gen.info()
df_gen_both = pd.concat([df_p1_gen, df_p2_gen])



df_gen_both.info()

df_gen_both.describe()
# What is the maximum AC/DC power generated by an inverter in a time interval / day?



from collections import deque



def get_max_per_arbitrary_time(df: pd.DataFrame, param: str, td: np.timedelta64) -> tuple:

    queues_map = {}  # dict[ str -> deque[np.datetime64] ]

    sum_map = {}

    max_q_map = {}

    max_sum_map = {}



    for ind, row in df[ ["SOURCE_KEY", "DATE_TIME", param] ].iterrows():

        key = row["SOURCE_KEY"]



        if queues_map.get(key):

            if row["DATE_TIME"] - df["DATE_TIME"][queues_map[key][0]] < td:

                queues_map[key].append(ind)

                sum_map[key] += row[param]

            else:

                while len(queues_map[key]) > 0 and not row["DATE_TIME"] - df["DATE_TIME"][queues_map[key][0]] < td:

                    queues_map[key].popleft()

                    sum_map[key] -= df[param][queues_map[key][0]]

    

                queues_map[key].append(ind)

                sum_map[key] += row[param]

        else:

            queues_map[key] = deque()

            queues_map[key].append(ind)

            sum_map[key] = row[param]

        

        if sum_map[key] > max_sum_map.get(key, 0):

            max_sum_map[key] = sum_map[key]

            max_q_map[key] = (queues_map[key][0], queues_map[key][-1])



    print(sum_map)

    print(max_sum_map)

    print(max_q_map)

    print(queues_map)

    

    return (max_sum_map, max_q_map)



                        

#     for key, df_for_key in df.groupby("SOURCE_KEY"):

#         queues_map[key] = deque()

#         queues_map[key].append(df_for_key["DATE_TIME"])

#         sum_map[key] = df_for_key[param]

#         temp_deq = deque()



#         for ind, row in df_for_key[1:].iterrows():

#             if sum_map[key] > max_sum_map.get(key, 0):

#                 max_sum_map[key] = sum_map[key]

#                 max_q_map[key] = queues_map[key][0]  # Only store the start



#             if row["DATE_TIME"] - queues_map[key][0] <= td:

#                 queues_map[key].append(row["DATE_TIME"])

#                 sum_map[key] += row[param]

#                 temp_deq.append(ind)

#             else:

#                 while True:

#                     if len(queues_map[key]) > 0:

#                         if row["DATE_TIME"] - queues_map[key][0] <= td:

#                             queues_map[key].append(row["DATE_TIME"])

#                             sum_map[key] += row[param]

#                             temp_deq.append(ind)

#                             break

#                         else:

#                             queues_map[key].popleft()

#                             sum_map[key] -= df[param][temp_deq.popleft()]

#                     else:

#                         queues_map[key].append(row["DATE_TIME"])

#                         sum_map[key] = row[param]

#                         break

    

#     print(queues_map)

#     print(max_q_map)

#     print(max_sum_map)



#     return max_sum_map
# Takes around 23 seconds for p1_gen dataset. Perhaps could be made faster!? (O(n^2) if pandas has constant lookup time)

from datetime import datetime, timedelta



before = datetime.now()

max_sum_df_p1_gen_new = get_max_per_arbitrary_time(df_p1_gen, "DC_POWER", timedelta(days=1))

after = datetime.now()



print(f"Time taken: {(after - before).seconds} seconds")
max_dc_pow_1_day, max_dc_pow_start_and_end_ind = max_sum_df_p1_gen_new
df_p1_gen.groupby(["SOURCE_KEY", "DATE"])["DC_POWER"].sum().groupby("SOURCE_KEY").max().sort_values(ascending=False)
# The values printed by this are slightly more than those given by the above command which gives 

# the max DC_POWER for any given SOURCE_KEY in one day

# This proves the validity of the algorithm (which gives the max power for a inverter within any duration of 24 hours)



print(max_dc_pow_1_day["adLQvlD726eNBSB"])

print(max_dc_pow_1_day["1IF53ai7Xc0U56Y"])

print(max_dc_pow_1_day["bvBOhCH3iADSZry"])
max_dc_pow_dts = dict([(key, (df_p1_gen["DATE_TIME"][indices[0]], df_p1_gen["DATE_TIME"][indices[1]])) for key, indices in max_dc_pow_start_and_end_ind.items()])



max_dc_pow_dts
df_p1_sen["DATE_TIME"] = pd.to_datetime(df_p1_sen["DATE_TIME"])

df_p2_sen["DATE_TIME"] = pd.to_datetime(df_p2_sen["DATE_TIME"])
df_p1_sen.info()



df_p2_sen.info()
df_p1_sen.describe()
df_p2_sen.describe()
# Let's visualize the data to see what we got



plt.figure(figsize=(12, 8))

plt.grid((1, 1))



plt.scatter(x=df_p1_sen["AMBIENT_TEMPERATURE"], y=df_p1_sen["MODULE_TEMPERATURE"])
plt.figure(figsize=(12, 8))

plt.grid((1, 1))



plt.plot(df_p1_sen["DATE_TIME"], df_p1_sen["IRRADIATION"])
plt.figure(figsize=(12, 8))

plt.grid((1, 1))



plt.scatter(x=df_p1_sen["MODULE_TEMPERATURE"], y=df_p1_sen["IRRADIATION"])
df_p1_gen.DAILY_YIELD.mean(), df_p2_gen.DAILY_YIELD.mean()
df_p1_sen["DATE"] = df_p1_sen["DATE_TIME"].apply(lambda x: x.date())

df_p2_sen["DATE"] = df_p2_sen["DATE_TIME"].apply(lambda x: x.date())



p1_sen_total_irrad = df_p1_sen.groupby("DATE")["IRRADIATION"].sum()

p2_sen_total_irrad = df_p2_sen.groupby("DATE")["IRRADIATION"].sum()
p1_sen_total_irrad.describe()
p2_sen_total_irrad.describe()
# Let's calculate number of missing readings



p1_max_minus_min_dt = (df_p1_gen["DATE_TIME"].max() - df_p1_gen["DATE_TIME"].min())

p1_max_minus_min_minutes = p1_max_minus_min_dt.days * 24 * 60 + p1_max_minus_min_dt.seconds / (60.)

p1_ideal_readings_per_inverter = p1_max_minus_min_minutes / 15



p2_max_minus_min_dt = (df_p2_gen["DATE_TIME"].max() - df_p2_gen["DATE_TIME"].min())

p2_max_minus_min_minutes = p2_max_minus_min_dt.days * 24 * 60 + p2_max_minus_min_dt.seconds / (60.)

p2_ideal_readings_per_inverter = p2_max_minus_min_minutes / 15



p1_ideal_readings_per_inverter, p2_ideal_readings_per_inverter
p1_gen_missing_readings_per_inverter = df_p1_gen["SOURCE_KEY"].value_counts().apply(lambda x: p1_ideal_readings_per_inverter - x)

p2_gen_missing_readings_per_inverter = df_p2_gen["SOURCE_KEY"].value_counts().apply(lambda x: p2_ideal_readings_per_inverter - x)



print(p1_gen_missing_readings_per_inverter)

print(p2_gen_missing_readings_per_inverter)