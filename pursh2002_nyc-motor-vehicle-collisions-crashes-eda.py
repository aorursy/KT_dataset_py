# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
dataset=pd.read_csv("../input/nyc-motor-vehicle-collisions-crashes-2018/MVC.csv")
dataset.head()
dataset.shape
import seaborn as sns
sns.heatmap(dataset.isnull(), cbar=False)
null_counts = dataset.isnull().sum()
null_counts
null_counts_pct = null_counts / dataset.shape[0] * 100
null_counts_pct
null_df = pd.DataFrame({'null_counts': null_counts,'null_pct': null_counts_pct})
# Rotate the dataframe so that rows become columns and vice-versa
null_df = null_df.T.astype(int) 
print(null_df)
killed_cols = [col for col in dataset.columns if 'KILLED' in col]
print(null_df[killed_cols])
injured_cols = [col for col in dataset.columns if 'INJURED' in col]
print(null_df[injured_cols])
killed = dataset[killed_cols].copy()

killed_manual_sum = killed.iloc[:,:3].sum(axis=1)
killed_mask = killed_manual_sum != killed['NUMBER OF PERSONS KILLED']
killed_non_eq = killed_mask[killed_mask]
killed_non_eq
killed_cols = [col for col in dataset.columns if 'KILLED' in col]
killed = dataset[killed_cols].copy()
#killed['NUMBER OF PERSONS KILLED'].fillna(0, inplace=True)
#killed['NUMBER OF PERSONS KILLED']= killed['NUMBER OF PERSONS KILLED'].astype(int)
# Select the first three columns from killed and sum each row https://s3.amazonaws.com/dq-content/370/verify_totals_2.svg
killed_manual_sum = killed.iloc[:,:3].sum(axis=1)
# Create a boolean mask that checks whether each value in killed_manual_sum is not equal to the values in the total_killed column 
# https://s3.amazonaws.com/dq-content/370/verify_totals_3.svg
killed_mask = killed_manual_sum != killed['NUMBER OF PERSONS KILLED']
# Use killed_mask to filter the rows in killed, https://s3.amazonaws.com/dq-content/370/verify_totals_4.svg
killed_non_eq = killed[killed_mask]
killed_non_eq
import pandas as pd
fruit = pd.Series(['apple', 'Bannana','bannana'])
bol = fruit == 'Bannana'
bol
result = fruit.mask(bol,'Pear')
result
killed_null = killed['NUMBER OF PERSONS KILLED'].isnull()
killed['NUMBER OF PERSONS KILLED'] = killed['NUMBER OF PERSONS KILLED'].mask(killed_null, killed_manual_sum)
killed
killed['NUMBER OF PERSONS KILLED'] = killed['NUMBER OF PERSONS KILLED'].mask(killed['NUMBER OF PERSONS KILLED'] != killed_manual_sum, np.nan)
print(killed[killed_mask])
# Create an injured dataframe and manually sum values
injured = dataset[[col for col in dataset.columns if 'INJURED' in col]].copy()
injured_manual_sum = injured.iloc[:,:3].sum(axis=1)

injured_manual_sum = injured.iloc[:,:3].sum(axis=1)
injured['NUMBER OF PERSONS INJURED'] = injured['NUMBER OF PERSONS INJURED'].mask(injured['NUMBER OF PERSONS INJURED'].isnull(), injured_manual_sum)
injured['NUMBER OF PERSONS INJURED'] = injured['NUMBER OF PERSONS INJURED'].mask(injured['NUMBER OF PERSONS INJURED'] != injured_manual_sum,np.nan)


injured

summary = {'injured': [dataset['NUMBER OF PERSONS INJURED'].isnull().sum(),
                      injured['NUMBER OF PERSONS INJURED'].isnull().sum()],
          'killed': [ dataset['NUMBER OF PERSONS KILLED'].isnull().sum(),
                    killed['NUMBER OF PERSONS KILLED'].isnull().sum()]}

print(pd.DataFrame(summary, index=['before','after']))


summary['injured']
dataset['NUMBER OF PERSONS INJURED'] = injured['NUMBER OF PERSONS INJURED']
dataset['NUMBER OF PERSONS KILLED'] = killed['NUMBER OF PERSONS KILLED']
import matplotlib.pyplot as plt
import seaborn as sns

def plot_null_matrix(df,figsize=(18,15)):
    # initiate the figure 
    plt.figure(figsize = figsize)
    # create a boolean dataframe based on whether values are null
    df_null = df.isnull()
    # create a heatmap of the boolean dataframe
    sns.heatmap(~df_null, cbar = False, yticklabels = False)
    plt.xticks(rotation = 90, size='x-large')
    plt.show()

plot_null_matrix(dataset.head(1), figsize=(18,1))
print(dataset.head(1))
plot_null_matrix(dataset)
cols_with_missing_vals = dataset.columns[dataset.isnull().sum() > 0]
missing_corr = dataset[cols_with_missing_vals].isnull().corr()
print(missing_corr)
import matplotlib.pyplot as plt
import seaborn as sns

def plot_null_correlations(df):
    # create a correlation matrix only for columns with at least
    # one missing value
    cols_with_missing_vals = df.columns[df.isnull().sum() > 0]
    missing_corr = df[cols_with_missing_vals].isnull().corr()
    
    # create a triangular mask to avoid repeated values and make
    # the plot easier to read
    missing_corr = missing_corr.iloc[1:, :-1]
    mask = np.triu(np.ones_like(missing_corr), k=1)
    
    # plot a heatmap of the values
    plt.figure(figsize=(20,14))
    ax = sns.heatmap(missing_corr, vmin=-1, vmax=1, cbar=False,
                     cmap='RdBu', mask=mask, annot=True)
    
    # format the text in the plot to make it easier to read
    for text in ax.texts:
        t = float(text.get_text())
        if -0.05 < t < 0.01:
            text.set_text('')
        else:
            text.set_text(round(t, 2))
        text.set_fontsize('x-large')
    plt.xticks(rotation=90, size='x-large')
    plt.yticks(rotation=0, size='x-large')

    plt.show()
    
veh_cols = [c for c in dataset.columns if 'VEHICLE' in c]
plot_null_correlations(dataset[veh_cols])
col_labels = ['v_number', 'vehicle_missing', 'cause_missing']

vc_null_data = []

for v in range(1,6):
    v_col = "VEHICLE TYPE CODE {}".format(v)
    c_col = 'CONTRIBUTING FACTOR VEHICLE {}'.format(v)
    
    v_null = (dataset[v_col].isnull() & dataset[c_col].notnull()).sum()
    c_null = (dataset[c_col].isnull() & dataset[v_col].notnull()).sum()
    
    vc_null_data.append([v, v_null, c_null])

vc_null_df = pd.DataFrame(vc_null_data, columns=col_labels)
vc_null_df
contr_cols = [c for c in dataset.columns if "CONTRIBUTING FACTOR VEHICLE" in c]
cause = dataset[contr_cols]
print(cause.head())
cause_1d = cause.stack()
print(cause_1d.head())
cause_counts = cause_1d.value_counts()
top10_causes = cause_counts.head(10)
print(top10_causes)
v_cols = [c for c in dataset.columns if c.startswith("VEHICLE")]
vehicles = dataset[v_cols]
vehicles_1d = vehicles.stack()
vehicles_counts = vehicles_1d.value_counts()
top10_vehicles = vehicles_counts.head(10)
top10_vehicles
# create a mask for each column
v_missing_mask = dataset['VEHICLE TYPE CODE 1'].isnull() & dataset['CONTRIBUTING FACTOR VEHICLE 1'].notnull()
c_missing_mask = dataset['CONTRIBUTING FACTOR VEHICLE 1'].isnull() & dataset['VEHICLE TYPE CODE 1'].notnull()

# replace the values matching the mask for each column
dataset['VEHICLE TYPE CODE 1'] =  dataset['CONTRIBUTING FACTOR VEHICLE 1'].mask(v_missing_mask, "Unspecified")
dataset['CONTRIBUTING FACTOR VEHICLE 1'] =  dataset['VEHICLE TYPE CODE 1'].mask(c_missing_mask, "Unspecified")
def summarize_missing():
    v_missing_data = []

    for v in range(1,6):
        v_col = 'VEHICLE TYPE CODE {}'.format(v)
        c_col = 'CONTRIBUTING FACTOR VEHICLE {}'.format(v)

        v_missing = (dataset[v_col].isnull() & dataset[c_col].notnull()).sum()
        c_missing = (dataset[c_col].isnull() & dataset[v_col].notnull()).sum()

        v_missing_data.append([v, v_missing, c_missing])

    col_labels = columns=["vehicle_number", "vehicle_missing", "cause_missing"]
    return pd.DataFrame(v_missing_data, columns=col_labels)

summary_before = summarize_missing()

for v in range(1,6):
    v_col = 'VEHICLE TYPE CODE {}'.format(v)
    c_col = 'CONTRIBUTING FACTOR VEHICLE {}'.format(v)
    
    v_missing_mask = dataset[v_col].isnull() & dataset[c_col].notnull()
    c_missing_mask = dataset[c_col].isnull() & dataset[v_col].notnull()

    dataset[v_col] = dataset[v_col].mask(v_missing_mask, "Unspecified")
    dataset[c_col] = dataset[c_col].mask(c_missing_mask, "Unspecified")

summary_after = summarize_missing()
loc_cols = ['BOROUGH', 'LOCATION', 'ON STREET NAME', 'OFF STREET NAME', 'CROSS STREET NAME']
location_data = dataset[loc_cols]
print(location_data)
print(location_data.isnull().sum())
plot_null_correlations(location_data)
sorted_location_data = location_data.sort_values(loc_cols)
plot_null_matrix(sorted_location_data)
