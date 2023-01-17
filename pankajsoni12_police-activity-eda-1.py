import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
pa = pd.read_csv("../input/Police_activity.csv")

pa.head(2)
print(pa.columns)

print(pa.columns.size) # or print(pa.shape[1])
print(pa.index)

print(pa.shape)
pa.isnull().sum()
for col in pa.columns:

    if pa[col].isnull().sum() > 50:

        pa.drop(col,axis="columns",inplace=True)
pa.isnull().sum() # small missing values, we are going to replace with mean,median or mode based on need
pa.head(2)
np.array([pa.location_raw != pa.district]).sum() # checking if location_raw is exactly same as district. if sum = 0 then same
pa.drop(["state","location_raw"],axis=1,inplace=True)
pa.head(2)
pa.isnull().sum()
for col in pa.columns:

    if pa[col].isnull().sum():

        print("Data type of %s column is %s"%(col,pa[col].dtype))
pa.dropna(subset=["stop_date","stop_time"],inplace=True)
pa.isnull().sum()
pa.head(2)
pa["start_stop_time"] = pa.stop_date+" "+pa.stop_time
pa.head(2)
pa.start_stop_time.dtype
pa.start_stop_time = pd.to_datetime(pa.start_stop_time,infer_datetime_format=True)
pa.start_stop_time.dtype
pa.start_stop_time.head(5) # We can see that dtype is converted as pandas `datetime` dtype
pa.head(2)
plt.figure(figsize=(16,4))

pa.start_stop_time.dt.hour.value_counts().plot(kind="bar",color="r")

plt.xlabel("Hour")

plt.ylabel("Hourly_Count")

plt.title("Hourly_Count_Details")

plt.show()
plt.figure(figsize=(16,4))

sns.distplot(pa.start_stop_time.dt.hour)

plt.xlabel("Hour")

plt.ylabel("Hourly_freq")

plt.title("Hourly_Count_Details_Histogram")

plt.show()
plt.figure(figsize=(16,4))

pa.start_stop_time.dt.year.value_counts().plot(kind="bar",color="g")

plt.xlabel("Year")

plt.ylabel("Yearly_Count")

plt.title("Yearly_Count_Details")

plt.show()
plt.figure(figsize=(16,4))

sns.distplot(pa.start_stop_time.dt.year)

plt.xlabel("Year")

plt.ylabel("Yearly_Count")

plt.title("Yearly_Count_Details_Histogram")

plt.show()
plt.figure(figsize=(16,4))

pa.start_stop_time.dt.month.value_counts().plot(kind="bar",color="c")

plt.xlabel("Month_Number")

plt.ylabel("Monthly_Count")

plt.title("Monthly_Count_Details")

plt.show()
plt.figure(figsize=(16,4))

sns.distplot(pa.start_stop_time.dt.month)

plt.xlabel("Month")

plt.ylabel("Monthly_Count")

plt.title("Monthly_Count_Details_Histogram")

plt.show()
plt.figure(figsize=(16,4))

sns.distplot(pa.start_stop_time.dt.day)

plt.xlabel("Daily")

plt.ylabel("Daily_Count")

plt.title("Dailly_Count_Details_Histogram")

plt.show()
pa.head(2)
np.log(pa.drugs_related_stop.value_counts().values)
pa.drugs_related_stop.value_counts().plot(kind="bar")

plt.xlabel("True/False")

plt.ylabel("True/False_Count")

plt.title("drugs_related_True/False_Count_Details")

plt.show()
np.log(pa.drugs_related_stop.value_counts()).plot(kind="bar")

plt.xlabel("True/False")

plt.ylabel("True/False_Count")

plt.title("Log: drugs_related_True/False_Count_Details")

plt.show()
pa.head(2)
pa["year"] = pa.start_stop_time.dt.year

pa["month"] = pa.start_stop_time.dt.month

pa.head(2)
grp1 = pa.groupby(["year"])
plt.figure(figsize=(16,4))

for key in grp1.groups:

    true_val = grp1.get_group(key).search_conducted.value_counts().loc[True]

    false_val = grp1.get_group(key).search_conducted.value_counts().loc[False]

    plt.bar(str(key),false_val,color="#934666",width=.5)

    plt.text(str(key),false_val-11000,false_val,rotation=90)

    plt.bar(str(key),true_val,bottom=false_val,color="#930006",width=.5)

    plt.text(str(key),true_val+false_val+500,true_val)

plt.xticks(rotation=90)

plt.legend(["False","True"])

plt.xlabel("Year")

plt.ylabel("Yearly_Count")

plt.title("Grouping With Yearly search_conducted Count_Details")

plt.show()

pa.head(2)
plt.figure(figsize=(16,4))

for key in grp1.groups:

    true_val = grp1.get_group(key).contraband_found.value_counts().loc[True]

    false_val = grp1.get_group(key).contraband_found.value_counts().loc[False]

    plt.bar(str(key),false_val,color="#934666",width=.5)

    plt.text(str(key),false_val-11000,false_val,rotation=90)

    plt.bar(str(key),true_val,bottom=false_val,color="k",width=.5)

    plt.text(str(key),true_val+false_val+500,true_val)

plt.xticks(rotation=90)

plt.legend(["False","True"])

plt.xlabel("Year")

plt.ylabel("Yearly_Count")

plt.title("Grouping With Yearly Contraband_found Count_Details")



plt.show()
plt.figure(figsize=(16,4))

for key in grp1.groups:

    true_val = grp1.get_group(key).drugs_related_stop.value_counts().loc[True]

    false_val = grp1.get_group(key).drugs_related_stop.value_counts().loc[False]

    plt.bar(str(key),false_val,color="#934666",width=.5)

    plt.text(str(key),false_val-11000,false_val,rotation=90)

    plt.bar(str(key),true_val,bottom=false_val,color="b",width=.5)

    plt.text(str(key),true_val+false_val+500,true_val)

plt.xticks(rotation=90)

plt.legend(["False","True"])

plt.xlabel("Year")

plt.ylabel("Yearly_Count")

plt.title("Grouping With Yearly frugs_related Count_Details")

plt.show()
pa.head(2)
grp2 = pa.groupby(["month"])
plt.figure(figsize=(16,4))

for key in grp2.groups:

    true_val = grp2.get_group(key).search_conducted.value_counts().loc[True]

    false_val = grp2.get_group(key).search_conducted.value_counts().loc[False]

    plt.bar(str(key),false_val,color="#934666",width=.5)

    plt.text(str(key),false_val-11000,false_val,rotation=90)

    plt.bar(str(key),true_val,bottom=false_val,color="#930006",width=.5)

    plt.text(str(key),true_val+false_val+500,true_val)

plt.xticks(rotation=90)

plt.legend(["False","True"])

plt.xlabel("Month")

plt.ylabel("Monthly_Count")

plt.title("Grouping With Monthly search_conducted Count_Details")

plt.show()
plt.figure(figsize=(16,4))

for key in grp2.groups:

    true_val = grp2.get_group(key).contraband_found.value_counts().loc[True]

    false_val = grp2.get_group(key).contraband_found.value_counts().loc[False]

    plt.bar(str(key),false_val,color="m",width=.5)

    plt.text(str(key),false_val-11000,false_val,rotation=90)

    plt.bar(str(key),true_val,bottom=false_val,color="k",width=.5)

    plt.text(str(key),true_val+false_val+500,true_val)

plt.xticks(rotation=90)

plt.legend(["False","True"])

plt.xlabel("Month")

plt.ylabel("Monthly_Count")

plt.title("Grouping With Monthly contraband_found Count_Details")

plt.show()
plt.figure(figsize=(16,4))

for key in grp2.groups:

    true_val = grp2.get_group(key).drugs_related_stop.value_counts().loc[True]

    false_val = grp2.get_group(key).drugs_related_stop.value_counts().loc[False]

    plt.bar(str(key),false_val,color="c",width=.5)

    plt.text(str(key),false_val-11000,false_val,rotation=90)

    plt.bar(str(key),true_val,bottom=false_val,color="b",width=.5)

    plt.text(str(key),true_val+false_val+500,true_val)

plt.xticks(rotation=90)

plt.legend(["False","True"])

plt.xlabel("Month")

plt.ylabel("Monthly_Count")

plt.title("Grouping With Monthly drugs_related_stop Count_Details")

plt.show()
pa.head(2)
plt.figure(figsize=(16,4))

data_to_display = 5000

sns.scatterplot(pa.stop_date[:data_to_display],pa.search_conducted[:data_to_display],

                hue=pa.district[:data_to_display],

                alpha=.1,

                s=20*pa.start_stop_time.dt.hour[:data_to_display]

               )

plt.xticks(pa.stop_date[:data_to_display:2],rotation=90)

plt.xlabel("Date")

plt.ylabel("Daily_Search_Conducted")

plt.title("Scatter plot Date Vs search Conducted Vs District Vs Hours")

plt.show()