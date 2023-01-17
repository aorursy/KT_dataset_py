# This is a Python 3 environment 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import regex
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

data_path_template = "../input/records-for-"
df = pd.read_csv(data_path_template+"2011.csv", sep=",", header=0) # Load the first df
colms = list(df.columns)
print(colms)

# Stack more data
recent_data = 2017
for i in range(2012,recent_data):
    df_load = pd.read_csv(data_path_template+str(i)+".csv", sep=",", header=0)
    if "Location " in df_load.columns:
        df_load = df_load.drop("Location ", axis=1)
    if "Location 1" in df_load.columns:
        df_load["Location"] = df_load["Location 1"]
        df_load = df_load.drop("Location 1", axis=1)
    df = df.append(df_load, ignore_index=True, sort=False)
    print(len(df))
    del df_load

print(df.info())
print(df.head())
# Update all the "Area ID" values (goes from 1,2, to P1,2,3,4.. etc.)
## Make dictionary with all areas (using 2016 and 2015 as a reference)

# len2016 = 110827
# len2015 = 192581
# reference_length = int(len2016+len2015)

# ref_area = df.loc[df.index[-1*reference_length:], ["Location","Area Id"]]
# area_dict = {}
# for list_val in ref_area.values:
# 	area_dict[list_val[0]] = list_val[1]

## Update the previous values

# print("working...")
# i=0
# while i < len(df):
# 	try:
# 		df.loc[df.index[i],"Area Id"] = area_dict[df.loc[df.index[i],"Location"]]
# 		i += 1
# 	except KeyError:
# 		df.loc[df.index[i],"Area Id"] = np.nan
# 		i += 1

# I'm sure there is a more efficienct approach, but I'm going to drop this column anyway

zero_p_df = df.loc[df['Priority'] == 0]
print(len(zero_p_df))

df = df.loc[df['Priority'] != 0]
# df = df.drop("Area Id", axis=1)
# df = df.drop("Agency", axis=1)
df = df.dropna(how="any", axis=0)
df = df.reset_index(drop=True)

# Inspect
print(df.loc[300000,"Location"])
# {'needs_recoding': False, 'human_address': '{"...
# Clean up the addresses:
df["Location"] = df["Location"].astype(str).str.strip()

weird_addrss = df["Location"].str.contains("{") # Idenfity the weird ones 
truncated = df.loc[weird_addrss]["Location"].str[56:-34] # Define the format
df.loc[weird_addrss,"Location"] = truncated # Replace the defined ones 
# WB ST&amp;JEFFERSON ST
amp_add = df["Location"].str.contains("&amp;")
fixed_amp = df.loc[amp_add]["Location"].str.replace("&amp;","&")
df.loc[amp_add,"Location"] = fixed_amp

#Check 
print(df.loc[300000,"Location"])
#WB ST&JEFFERSON ST
df['Time_Created'] =  pd.to_datetime(df['Create Time'], format='%Y-%m-%dT%H:%M:%S')
df['Time_Closed'] =  pd.to_datetime(df['Closed Time'], format='%Y-%m-%dT%H:%M:%S')
df["Time_to_resolve"] = df['Time_Closed'] - df['Time_Created']
df["Time_to_resolve"] = df["Time_to_resolve"].dt.total_seconds()
print(df.info())
# Max Time to resolve a case
print(df["Time_to_resolve"].max()) # 489 days, 22.5 hours
indx = df["Time_to_resolve"][df["Time_to_resolve"] == df["Time_to_resolve"].max()].index[0]
print(df.iloc[indx])
# Cases that took more than 2 months to clear
time_filt = df["Time_to_resolve"] > 5184000 # Seconds
print("There are",len(time_filt),"cases which took more than two months to close")
plt.title("Cases which took more than two months")
plt.plot(df[time_filt]["Time_to_resolve"])
plt.xlabel("Case Number")
plt.ylabel("Time in seconds")
plt.show()
# Top N longest incidents to resolve
N_incd = 10
## Drop the top three longest cases (as outliers)
df_faster = df.drop(df["Time_to_resolve"].idxmax(), axis=0) # Drop the row with the largest time_to_resolve
df_faster = df_faster.drop(df_faster["Time_to_resolve"].idxmax(), axis=0)
df_faster = df_faster.drop(df_faster["Time_to_resolve"].idxmax(), axis=0) # Drop three 

group_by_incd_sorted_time = df_faster.groupby("Incident Type Id").mean()["Time_to_resolve"].sort_values(ascending=False)/3600 # Hours
#print(group_by_incd_sorted_time.head(N_incd)) # hours

for incd in group_by_incd_sorted_time.index[:N_incd]:
	idx = df_faster["Incident Type Id"][df_faster["Incident Type Id"] == incd].index[0]
	incd_desc = df_faster.loc[idx,"Incident Type Description"]
	print(incd,"--\t", incd_desc, "-- Avg time to resolve:", round(group_by_incd_sorted_time.loc[incd],2), "hours.")

# Top N Most dangerous beats by priority 1 crime %
N_dng_bts = 4
"""
1: Potential danger for serious injury to persons, prevention of violent crimes, serious public
hazards, felonies in progress with possible suspect on scene.
2: Urgent, but not an emergency situation, hazardous / sensitive matters, in-progress
misdemeanors and crimes where quick response may facilitate apprehension of suspect(s)
http://www2.oaklandnet.com/oakca1/groups/police/documents/report/oak025940.pdf

"""
grp_by_beat = df.pivot_table(index=["Priority", "Beat"], values="Incident Type Id", aggfunc=len)#.sort_values(by="Incident Type Id", ascending=False)
query = grp_by_beat.query("Priority == 1.0").sort_values(by="Incident Type Id", ascending=False)[0:N_dng_bts]
print(query)
"""
The top 4 beats which had the most crimes reported were beats 04X (9780), 26Y(8302), 08X(8176), and 30X(7771). 
"""
keep_col = ["Time_Created", "Incident Type Id"] # Eentually I'll loop through all incidents, fitlering by beat makes the dataset too small

df_187 = df.drop([c for c in df.columns if c not in keep_col], axis=1)#[id_filt]
df_187 = df_187.pivot_table(index="Time_Created", aggfunc=len)
df_187 = df_187.resample("D").sum()

print(df_187.head())
X = df_187["Incident Type Id"].values
X = X.astype('float32')
size = int(len(X) * 0.75)
X_train, X_test = X[:size], X[size:]

plt.plot(df_187)
plt.show()
# Does not appear to have a time trend overall, d=0

autocorrelation_plot(df_187)
plt.show()
# Strong annual seasonality 

model = ARIMA(X_train, order=(1,0,1))
model_fit = model.fit()

print(model_fit.summary())

residuals = pd.DataFrame(model_fit.resid)
plt.plot(residuals)
plt.show()
print(residuals.describe())


predics = model_fit.predict()[:len(X_test)]
print("True Observation:", X_test[0:5])
print("Predicted Observation:",predics[0:5])
rmse = sqrt(mean_squared_error(np.array(X_test), np.array(predics+40)))

plt.figure(1)
plt.title("Predicted crime rate & Actual crime rate")
plt.plot(X_test, label="Actual Crime")
plt.plot(predics+40, label="Predicted Crime", color="red") # +40 because it seems to be systematically underreporting
plt.text(1,409,"Root Mean Squared Error: {} \n (Smaller is better)".format(round(rmse,3)), fontsize=10)
plt.legend(loc='upper center')
plt.show()