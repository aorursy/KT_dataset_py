# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

PATH_TO_CSV_FILE = os.path.join('..', 'input', 'daily-power-generation-in-india-20172020')
for dirname, _, filenames in os.walk(PATH_TO_CSV_FILE):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
CSV_FILE_LIST = ['State_Region_corrected.csv', 'file.csv']

state_wise_area_share_df = pd.read_csv(os.path.join(PATH_TO_CSV_FILE, CSV_FILE_LIST[0]))
power_generation_df = pd.read_csv(os.path.join(PATH_TO_CSV_FILE, CSV_FILE_LIST[1]))

state_wise_area_share_df.head(10)
power_generation_df.head(10)
POWER_GENERATION_SHAPE = power_generation_df.shape
STATE_WISE_AREA_SHARE_SHAPE = state_wise_area_share_df.shape

print("State Wise Power Share Dataset:")
print(f"Number of Rows: {STATE_WISE_AREA_SHARE_SHAPE[0]}")
print(f"Number of Columns: {STATE_WISE_AREA_SHARE_SHAPE[1]}")
print()

print("Power Generation Dataset:")
print(f"Number of Rows: {POWER_GENERATION_SHAPE[0]}")
print(f"Number of Columns: {POWER_GENERATION_SHAPE[1]}")
state_wise_area_share_df['Region'].unique()
power_generation_df['Region'].unique()
state_wise_area_share_df['State / Union territory (UT)'].unique()
# States in the North Region
NORTH_REGION_STATES = [
    'Ladakh',
    'Himachal Pradesh',
    'Uttarakhand',
    'Punjab',
    'Haryana',
    'Jammu and Kashmir',
    'Rajasthan',
    'Delhi',
    'Chandigarh',
    'Uttar Pradesh',
]

# States in the North Eastern Region
NORTH_EASTERN_REGION_STATES = [  
    'Arunachal Pradesh',
    'Assam',
    'Meghalaya',
    'Manipur',
    'Mizoram',
    'Nagaland',
    'Tripura',
]


# States in the South Region
SOUTH_REGION_STATES = [
    'Tamil Nadu',
    'Telangana',
    'Kerala',
    'Karnataka',
    'Andhra Pradesh',
    'Puducherry',
]

# States in the West Region
WEST_REGION_STATES = [
    'Maharashtra',
    'Gujarat',
    'Madhya Pradesh',
    'Dadra and Nagar Haveli and Daman and Diu',
    'Chhattisgarh',
    'Goa',
]

# States in the East Region
EAST_REGION_STATES = [
    'Odisha',
    'Bihar',
    'West Bengal',
    'Jharkhand',
    'Sikkim',
]
REGION_NAMES = {'N': 'Northern', 'W': 'Western', 'S': 'Southern', 'E': 'Eastern', 'NE': 'NorthEastern'}

state_wise_area_share_df.loc[state_wise_area_share_df['State / Union territory (UT)'].isin(NORTH_REGION_STATES), 'Region'] = REGION_NAMES['N']
state_wise_area_share_df.loc[state_wise_area_share_df['State / Union territory (UT)'].isin(SOUTH_REGION_STATES), 'Region'] = REGION_NAMES['S']
state_wise_area_share_df.loc[state_wise_area_share_df['State / Union territory (UT)'].isin(EAST_REGION_STATES), 'Region'] = REGION_NAMES['E']
state_wise_area_share_df.loc[state_wise_area_share_df['State / Union territory (UT)'].isin(WEST_REGION_STATES), 'Region'] = REGION_NAMES['W']
state_wise_area_share_df.loc[state_wise_area_share_df['State / Union territory (UT)'].isin(NORTH_EASTERN_REGION_STATES), 'Region'] = REGION_NAMES['NE']

state_wise_area_share_df['Region'].unique()
state_wise_area_share_df.head(10)
COLUMN_HEADERS = power_generation_df.columns

COLUMN_HEADERS
power_generation_df.dtypes
for column_name in COLUMN_HEADERS:
    if column_name in ['Thermal Generation Actual (in MU)', 'Thermal Generation Estimated (in MU)']:
        power_generation_df[column_name] = power_generation_df[column_name].str.replace(',', '')
        power_generation_df[column_name] = pd.to_numeric(power_generation_df[column_name])


power_generation_df.head(10)
power_generation_df[['Thermal Generation Actual (in MU)', 'Nuclear Generation Actual (in MU)', 'Hydro Generation Actual (in MU)']].sum(axis=1)
# This also gives the same result as the previous cell
# power_generation_df.loc[:, ['Thermal Generation Actual (in MU)', 'Nuclear Generation Actual (in MU)', 'Hydro Generation Actual (in MU)']].sum(axis=1)
power_generation_df['Total Power Generation Actual (in MU)'] = power_generation_df[
                                                                                    [
                                                                                        'Thermal Generation Actual (in MU)',
                                                                                        'Nuclear Generation Actual (in MU)',
                                                                                        'Hydro Generation Actual (in MU)',
                                                                                    ]
                                                                                   ].sum(axis=1)
power_generation_df['Total Power Generation Estimated (in MU)'] = power_generation_df[
                                                                                        [
                                                                                            'Thermal Generation Estimated (in MU)',
                                                                                            'Nuclear Generation Estimated (in MU)',
                                                                                            'Hydro Generation Estimated (in MU)',
                                                                                        ]
                                                                                      ].sum(axis=1)
power_generation_df.head(10)
CHECKS = [
    math.isclose(624.23 + 30.36 + 273.27, 927.86),
    math.isclose(1106.89 + 25.17 + 72.00, 1204.06),
    math.isclose(576.66 + 62.73 + 111.57, 750.96),
]


if all(CHECKS):
    print('Looking Good!')
else:
    print('Something went wrong!!')
CHECKS = [
    math.isclose(484.21 + 35.57 + 320.81, 840.59),
    math.isclose(1024.33 + 3.81 + 21.53, 1049.67),
    math.isclose(578.55 + 49.80 + 64.78, 693.13),
]


if all(CHECKS):
    print('Looking Good!')
else:
    print('Something went wrong!!')
power_generation_df['Date'] = pd.to_datetime(power_generation_df['Date'], format='%Y-%m-%d')

power_generation_df.head(10)
power_generation_df.dtypes
power_generation_df.isnull()
# power_generation_df.isnull().values
# type(power_generation_df.isnull().values)  # Converts the DataFrame into a numpy array
# power_generation_df.isnull().values.ravel()  # Flattens the 2D array into a 1D array
NO_OF_NAN_VALUES = np.count_nonzero(power_generation_df.isnull().values.ravel())
print(f"The number of NaN values in the dataset is: {NO_OF_NAN_VALUES}")
NO_OF_ROWS_WITH_NAN_VALUES = power_generation_df.shape[0] - power_generation_df.dropna().shape[0]
print(f"The number of rows with NaN values are: {NO_OF_ROWS_WITH_NAN_VALUES}")
AVG_NO_OF_NANs_PER_ROW = np.count_nonzero(power_generation_df.isnull().values.ravel()) / (power_generation_df.shape[0] - power_generation_df.dropna().shape[0])
print(f"So there are about {AVG_NO_OF_NANs_PER_ROW} NaN values per row")
power_generation_df.isnull().any()
power_generation_df[['Region', 'Nuclear Generation Actual (in MU)', 'Nuclear Generation Estimated (in MU)']].groupby('Region').sum()
power_generation_df.columns
pd.DatetimeIndex(power_generation_df["Date"]).year.unique()
power_generation_df[['Region', 'Nuclear Generation Actual (in MU)', 'Nuclear Generation Estimated (in MU)']].groupby('Region').sum()
power_generation_df.fillna(0.0, inplace=True)
power_generation_df
power_generation_df.describe()
power_generation_df.groupby('Region').describe().stack()
power_generation_df.groupby('Region').sum().reset_index()
power_generation_df.groupby('Region').sum()["Total Power Generation Actual (in MU)"]
# power_generation_df.groupby('Region').sum()["Total Power Generation Estimated (in MU)"]
power_generation_df.groupby('Region').sum().index.values
power_generation_df.groupby('Region').sum()[["Total Power Generation Actual (in MU)", "Total Power Generation Estimated (in MU)"]]
TEMP = power_generation_df.groupby('Region').sum()[["Total Power Generation Actual (in MU)", "Total Power Generation Estimated (in MU)"]]

ax = plt.figure(figsize=(4,3), dpi=150)
plt.title("Actual Vs Estimated\nPower Generation per Region", y=1.08, fontsize=12)
TEMP.plot.bar(rot=75, ax=plt.gca(), width=.75)
plt.legend(fancybox=True, shadow=True, bbox_to_anchor=(1, 0.5), loc='center left')
plt.ylabel('Power Generated in Mega Units (MU)')

plt.show()


