# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
## Load in dataset, and see information ##
df = pd.read_csv("../input/bus-breakdown-and-delays.csv")
print(df.info())

print(df.head(10))
for c in df.columns:
    print("---",c,"---")
    unq = df[c].value_counts() 
    print(unq)
 
dtype_structure = {
    "category":["School_Year", "Busbreakdown_ID", "Run_Type", "Bus_No", "Route_Number", "Reason", 
                "Schools_Serviced", "Boro", "Bus_Company_Name", "Incident_Number", 
                "Breakdown_or_Running_Late", "School_Age_or_PreK"],
    #"float":   ["How_Long_Delayed"], # This will need to be cleaned later
    "int":     ["Number_Of_Students_On_The_Bus"],
    "datetime64":["Occurred_On", "Created_On", "Informed_On", "Last_Updated_On"],
    "bool":    ["Has_Contractor_Notified_Schools", "Has_Contractor_Notified_Parents", "Have_You_Alerted_OPT"],
    "object":  []    
}
print(df["How_Long_Delayed"][0:10])
delayed_str = df["How_Long_Delayed"].values
trial = ["-", "hr", "min", "/"]
for t in trial:
    matching = [s for s in delayed_str if t in str(s)]
    print(matching[0:6])
    
"""
Let's come back and clean this up later.
"""

for dtp, col in dtype_structure.items():
    df[col] = df[col].astype(dtp)
    
print(df.info())
rsn = df.groupby("Reason").size()
lth = range(len(rsn))
plt.bar(lth, rsn)
plt.xticks(lth, rsn.index, rotation=90)
plt.show()
delay_time = df.groupby(df["Occurred_On"].map(lambda t: t.hour)).size()
lth = range(len(delay_time))
plt.bar(lth, delay_time)
plt.xticks(lth, delay_time.index, rotation=90)
plt.title("Number of Delays by Hour of the Day")
plt.ylabel("Number of delays reported")
plt.xlabel("Hour of the day")
plt.show()
traff_day = df[df["Reason"] == "Heavy Traffic"].groupby(
                                                df["Occurred_On"].map(lambda t: t.weekday())).size()
wkd = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
lth = range(len(traff_day))
plt.bar(lth, traff_day)
plt.xticks(lth, wkd, rotation=90)
plt.title("Occurances of Heavy Traffic by Day of the Week")
plt.ylabel("Heavy Traffic Reports")
plt.show()
wkd = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]#, "Saturday", "Sunday"]
nrows = 4
ncols = 3

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 6), dpi=80)
plt.suptitle("Occurance of Delay Reason by Weekday")
i = 1
for rsn in df["Reason"].unique():
    reason_day = df[df["Reason"] == rsn].groupby(
                                        df["Occurred_On"].map(lambda t: t.weekday())).size()
    lth = range(len(reason_day))
    plt.subplot(nrows,ncols,i)
    plt.bar(lth, reason_day)
    plt.xticks(lth, wkd, rotation=90)
    #plt.title("Occurances of {r} by Day of the Week".format(r=rsn))
    plt.ylabel(str(rsn))
    i+=1 

# delete empty axes
# axes[5,0].set_axis_off()
plt.show()
bus_comp_val = df["Bus_Company_Name"].value_counts()

print(bus_comp_val[0:4])
print("...And so on")
merge_char = 5
exclude_cnt = 200

df["Short_Bus_Comp_Name"] = df["Bus_Company_Name"].str[0:merge_char]
# print(df["Short_Bus_Comp_Name"].value_counts())

bus_comp_cnt = df["Short_Bus_Comp_Name"].value_counts().tolist()
bus_comp_bns = df["Short_Bus_Comp_Name"].value_counts().index.tolist()

keep_cnt_lst = [k for k in bus_comp_cnt if k>exclude_cnt]
keep_bus_lst = bus_comp_bns[0:len(keep_cnt_lst)]
# print(keep_bus_lst) #61 elements, down from 117


grp_mech = df[df["Reason"] == "Mechanical Problem"].groupby("Short_Bus_Comp_Name").size() # Number of delays due to mechanical problems by company
grp_dely = df.groupby("Short_Bus_Comp_Name").size() # Number of any delays for any reason by company
# print(len(grp_mech), len(grp_dely)) The lengths don't match, some short companies don't have mech failures reported


mech_rates = {}
for comp in grp_mech[keep_bus_lst].index:
   mech_rates[comp] = grp_mech[comp]/grp_dely[comp] * 100
    
sorted_lst = sorted(mech_rates.items(), reverse=True, key=lambda x: x[1])

n = 10
top_n = sorted_lst[0:n]
lth = range(n)

# Let's narrow to the top 10:

plt.bar(lth, [v for k,v in top_n])
plt.xticks(lth, [k for k,v in top_n], rotation=90)
plt.show()






