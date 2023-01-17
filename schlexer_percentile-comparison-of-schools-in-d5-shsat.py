import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pandas as pd
from scipy import stats
# function file to data frame
def file_to_df(file):
    filename, file_extension = os.path.splitext(file)
    if file_extension=='.csv':
        df = pd.read_csv(file, sep=',', header=0)
    elif file_extension=='.tsv':
        df = pd.read_csv(file, sep='\t', header=0)
    else:
        print('Please provide csv or tsv file format.')
    return df

# read in the data and give the columns useful names
df = file_to_df('../input/data-science-for-good/D5 SHSAT Registrations and Testers.csv')
colnames = ['DBN','school_name','year','grade','october_school_enrollment','registered_for_test','took_test']
df.columns = colnames
df = df.sort_values(by=['registered_for_test'])
# print(df.head())

# calculate the ratios
df['enroll_to_take_ratio'] = np.nan_to_num(df['took_test']/df['october_school_enrollment']*100.00)
df['register_to_take_ratio'] = df['took_test']/df['registered_for_test']*100.00
df['register_to_take_ratio'] = df['register_to_take_ratio'].dropna()
df = df.sort_values(by=['year','enroll_to_take_ratio'],ascending=True)

# create a funciton to calculate the percentiles
def get_percentiles(column, bins_percentile = [0,20,40,60,80,100]):
    data_percentile = 100*column.rank(pct=True, method='min')
    steps = 100/len(bins_percentile)-1
    bins = steps*np.digitize(data_percentile, bins_percentile, right=True)
    return data_percentile, bins
# calculate the percentile ranks of the relevant columns
df['enroll_to_take_ratio_percentiles'] = get_percentiles(df['enroll_to_take_ratio'])[0]
df['enroll_to_take_ratio_bins'] = get_percentiles(df['enroll_to_take_ratio'])[1]
df['register_to_take_ratio_percentiles'] = get_percentiles(df['register_to_take_ratio'])[0]
df['register_to_take_ratio_bins'] = get_percentiles(df['register_to_take_ratio'])[1]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = df['year']
y = df['enroll_to_take_ratio_bins']
z = df['enroll_to_take_ratio']
ax.scatter(x, y, z)
plt.ylim((0,100))
ax.set_xticks(np.arange(min(x), max(x)+1, 1.0))
ax.set_yticks(np.arange(10, 100, 20.0))
plt.xlabel(r'Year', fontsize=16)
plt.ylabel(r'Percentile', fontsize=16)
plt.title(r'Percent pupils taking test', fontsize=16)
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = df['year']
y = df['register_to_take_ratio_bins']
z = df['register_to_take_ratio']
ax.scatter(x, y, z)
plt.ylim((0,100))
ax.set_xticks(np.arange(min(x), max(x)+1, 1.0))
ax.set_yticks(np.arange(0, 100, 20.0))
plt.title(r'Percent registered pupils taking test', fontsize=16)
plt.show()

df['register_percentile'] = get_percentiles(df['registered_for_test'])[0]
df['took_test_percentile'] = get_percentiles(df['took_test'])[0]
file_name = 'augmented_D5_SHSAT_Registrations_and_Testers.csv'
df.to_csv(file_name)




